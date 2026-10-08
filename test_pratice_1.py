import threading
import time
import unittest

from pratice_1 import Scheduler


def recorder(order, name, value=None):
    """A job fn that appends its name to `order` and returns `value`."""
    def fn():
        order.append(name)
        return value
    return fn


def wait_until(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.005)


class OrderingTests(unittest.TestCase):
    # gpus=1 so jobs run one at a time and the run order is deterministic.

    def test_higher_priority_runs_first(self):
        s, order = Scheduler(gpus=1), []
        s.submit("low", recorder(order, "low"), priority=1)
        s.submit("high", recorder(order, "high"), priority=10)
        s.submit("mid", recorder(order, "mid"), priority=5)
        s.run_all()
        self.assertEqual(order, ["high", "mid", "low"])

    def test_fifo_among_equal_priority(self):
        s, order = Scheduler(gpus=1), []
        for name in ["a", "b", "c", "d"]:
            s.submit(name, recorder(order, name), priority=3)
        s.run_all()
        self.assertEqual(order, ["a", "b", "c", "d"])

    def test_dependent_runs_after_dependency_despite_higher_priority(self):
        s, order = Scheduler(gpus=1), []
        s.submit("dep", recorder(order, "dep"), priority=0)
        s.submit("child", recorder(order, "child"), priority=100, depends_on=["dep"])
        self.assertEqual(s.status("child"), "pending")
        s.run_all()
        self.assertEqual(order, ["dep", "child"])


class ResultAndRetryTests(unittest.TestCase):

    def test_result_and_status_after_success(self):
        s = Scheduler(gpus=1)
        s.submit("a", lambda: 42)
        self.assertEqual(s.status("a"), "ready")
        s.run_all()
        self.assertEqual(s.status("a"), "succeeded")
        self.assertEqual(s.result("a"), 42)

    def test_retries_until_success(self):
        s, attempts = Scheduler(gpus=1), []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("boom")
            return "ok"

        s.submit("flaky", flaky, max_retries=2)
        s.run_all()
        self.assertEqual(len(attempts), 3)
        self.assertEqual(s.result("flaky"), "ok")

    def test_fails_after_retries_exhausted(self):
        s, attempts = Scheduler(gpus=1), []

        def always_fails():
            attempts.append(1)
            raise ValueError("boom")

        s.submit("bad", always_fails, max_retries=2)
        s.run_all()
        self.assertEqual(len(attempts), 3)  # 1 try + 2 retries
        self.assertEqual(s.status("bad"), "failed")
        with self.assertRaises(RuntimeError) as cm:
            s.result("bad")
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_failure_skips_dependents_transitively(self):
        s, order = Scheduler(gpus=1), []
        s.submit("bad", lambda: 1 / 0)
        s.submit("child", recorder(order, "child"), depends_on=["bad"])
        s.submit("grandchild", recorder(order, "grandchild"), depends_on=["child"])
        s.submit("unrelated", recorder(order, "unrelated"))
        s.run_all()
        self.assertEqual(s.status("child"), "skipped")
        self.assertEqual(s.status("grandchild"), "skipped")
        self.assertEqual(order, ["unrelated"])

    def test_submit_after_dependency_failed_is_skipped(self):
        s = Scheduler(gpus=1)
        s.submit("bad", lambda: 1 / 0)
        s.run_all()
        s.submit("late", lambda: None, depends_on=["bad"])
        self.assertEqual(s.status("late"), "skipped")


class ValidationTests(unittest.TestCase):

    def setUp(self):
        self.s = Scheduler(gpus=2)
        self.s.submit("a", lambda: None)

    def test_rejects_duplicate_id(self):
        with self.assertRaises(ValueError):
            self.s.submit("a", lambda: None)

    def test_rejects_unknown_dependency(self):
        with self.assertRaises(ValueError):
            self.s.submit("b", lambda: None, depends_on=["nope"])

    def test_rejects_self_dependency(self):
        with self.assertRaises(ValueError):
            self.s.submit("b", lambda: None, depends_on=["b"])

    def test_rejects_gpus_required_out_of_range(self):
        for bad in (0, 3):
            with self.assertRaises(ValueError):
                self.s.submit(f"g{bad}", lambda: None, gpus_required=bad)

    def test_unknown_job_status_raises(self):
        with self.assertRaises(KeyError):
            self.s.status("nope")


class GpuTests(unittest.TestCase):

    def test_run_next_is_nonblocking_and_respects_free_gpus(self):
        s, release = Scheduler(gpus=2), threading.Event()
        for name in ["a", "b", "c"]:
            s.submit(name, lambda: release.wait(2))
        self.assertEqual(s.run_next(), "a")  # returns while "a" is still running
        self.assertEqual(s.run_next(), "b")
        self.assertIsNone(s.run_next())      # both GPUs busy
        self.assertEqual(s.status("a"), "running")
        self.assertEqual(s.status("c"), "ready")
        release.set()
        s.run_all()
        self.assertEqual(s.status("c"), "succeeded")

    def test_concurrency_never_exceeds_gpu_count(self):
        s, lock = Scheduler(gpus=3), threading.Lock()
        live, peak = [0], [0]

        def work():
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.05)
            with lock:
                live[0] -= 1

        for i in range(8):
            s.submit(f"j{i}", work)
        s.submit("big", work, gpus_required=2)
        s.run_all()
        self.assertEqual(peak[0], 3)

    def test_backfills_smaller_job_when_top_does_not_fit(self):
        s, release = Scheduler(gpus=4), threading.Event()
        s.submit("a", lambda: release.wait(2), priority=9, gpus_required=2)
        s.submit("big", lambda: None, priority=5, gpus_required=3)
        s.submit("small", lambda: release.wait(2), priority=1, gpus_required=1)
        self.assertEqual(s.run_next(), "a")
        self.assertEqual(s.run_next(), "small")  # "big" needs 3, only 2 free
        self.assertEqual(s.status("big"), "ready")
        release.set()
        s.run_all()
        self.assertEqual(s.status("big"), "succeeded")

    def test_starving_job_blocks_backfill(self):
        s = Scheduler(gpus=4, max_skips=1)
        release = {k: threading.Event() for k in ["a", "small1", "small2"]}
        s.submit("a", lambda: release["a"].wait(2), priority=9, gpus_required=2)
        s.submit("big", lambda: None, priority=5, gpus_required=3)
        s.submit("small1", lambda: release["small1"].wait(2), gpus_required=1)
        s.submit("small2", lambda: release["small2"].wait(2), gpus_required=1)

        self.assertEqual(s.run_next(), "a")
        self.assertEqual(s.run_next(), "small1")  # "big" passed over once
        self.assertIsNone(s.run_next())           # "small2" fits, but "big" is starving

        release["a"].set()
        release["small1"].set()
        wait_until(lambda: s.status("a") == "succeeded" and s.status("small1") == "succeeded")
        self.assertEqual(s.run_next(), "big")
        self.assertEqual(s.run_next(), "small2")
        release["small2"].set()
        s.run_all()


if __name__ == "__main__":
    unittest.main()
