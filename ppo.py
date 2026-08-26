import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

"""
Example of how gym works:

env = gym.make("CartPole-v1")

observation, info = env.reset()  # start an episode, get initial state
# observation: shape (4,) -- cart position, cart velocity, pole angle, pole angular velocity

action = env.action_space.sample()  # sample a random action (0 -- push left, 1 -- push right)

observation, reward, terminated, truncated, info = env.step(action)  # take action, get next state, reward, termination, truncation, info

# terminated: episode naturally ended -- pole fell
# truncated: episode hit  a time limit
"""

def random_agent_loop(env):
    """
    Just keep taking actions until terminated or truncated and see reward
    """
    obs, info = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Total reward: {total_reward}")
    return total_reward

# since this environment has a discrete action space, we use a categorical 
# distribution to sample actions
class ActorPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions) # logits, not probabilities
        )
    
    def forward(self, obs):
        return self.net(obs)

# Value function.  
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # output is size 1 for value function. 
        )

    def forward(self, obs):
        return self.net(obs)

def ppo_agent_loop(
    env, policy, critic, n_episodes=100, n_steps=1000, 
    gamma=0.99, lambda_=0.95, eps=0.2, alpha=0.001, c_1=0.5, c_2=0.01, k=5
):
    s_t, info = env.reset()
    s_t = torch.from_numpy(s_t)
    
    done = False

    pi_optimizer = optim.Adam(policy.parameters(), lr=alpha, betas=(0.9, 0.999))
    critic_optimizer = optim.Adam(
        critic.parameters(), lr=alpha, betas=(0.9, 0.999)
    )
    for episode in range(n_episodes):
        # just do one actor for now
        # run policy for n_steps, collect data

        delta_t = []
        actions = []
        states = []
        rewards = []
        log_probs = []
        critic_outputs = []
        term_steps = []
        # we can't just do total_reward += reward over all the steps because
        # this environment gives +1 reward every step no matter what. 
        # so what we really want is like reward per episode or episode length. 
        curr_episode_rewards = []
        curr_reward = 0

        for step in range(n_steps):
            logits = policy(s_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            # don't want to backprog through this, so detach()
            log_probs.append(dist.log_prob(action).detach()) 
            actions.append(action)
            
            action = action.item()
            s_t_plus_1, reward, terminated, truncated, info = env.step(action)

            states.append(s_t)
            rewards.append(reward)
            curr_reward += reward

            with torch.no_grad():
                critic_output = critic(s_t)
            critic_outputs.append(critic_output)

            if not terminated and not truncated:
                s_t_plus_1 = torch.from_numpy(s_t_plus_1).float()
                with torch.no_grad():
                    delta_t.append(
                        reward + gamma * critic(s_t_plus_1) - critic_output
                    )
                # move the state forward
                s_t = s_t_plus_1
            else:
                # when we are in the terminal state V(s_t+1) = 0
                term_steps.append(step)
                delta_t.append(
                    reward + gamma * 0 - critic_output
                )
                s_t, info = env.reset()
                s_t = torch.from_numpy(s_t)
                curr_episode_rewards.append(curr_reward)
                curr_reward = 0

        print(f"Total reward from episode {episode} was {sum(curr_episode_rewards) / len(curr_episode_rewards)}")

        # compute advantage estimates in reverse time order
        advantage_t = [delta_t[-1]]
        for t in range(n_steps - 2, -1, -1):
            if t in term_steps:
                # if this was a terminal state, don't accumulate. 
                advantage_t.append(delta_t[t])
            else:
                advantage_t.append(
                    delta_t[t] + (gamma * lambda_) * advantage_t[-1]
                )
        advantage_t = torch.stack(advantage_t[::-1])

        # optimize policy for k epochs
        def clip(x):
            if x < 1 - eps: return 1 - eps
            if x > 1 + eps: return 1 + eps
            return x

        for epoch in range(k):
            # idk what to do here. 
            L_clip = 0
            L_VF = 0
            entropy_bonus = 0
            for t in range(n_steps):
                d = Categorical(logits=policy(states[t]))
                entropy_bonus += d.entropy()
                new_log_prob = d.log_prob(actions[t])
                # r_t = pi(a_t|s_t) / pi_old(a_t | s_t)
                r_t = torch.exp(new_log_prob - log_probs[t]) 
                L_clip += min(r_t * advantage_t[t], clip(r_t) * advantage_t[t])

                # In GAE, since A = Return - V(s) => Return = A + V(s)
                V_target = advantage_t[t] + critic_outputs[t]
                L_VF += (critic(states[t]) - V_target) ** 2

            L_clip /= n_steps
            entropy_bonus /= n_steps # don't really need this. 
            L = L_clip - (c_1 * L_VF) + (c_2 * entropy_bonus)
            L = -1 * L # doing gradient ascent. 

            pi_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            L.backward()
            pi_optimizer.step()
            critic_optimizer.step()
 

    
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # random_agent_loop(env)

    actor = ActorPolicy(env.observation_space.shape[0], env.action_space.n)
    critic = Critic(env.observation_space.shape[0])
    ppo_agent_loop(env, actor, critic)