"""
Requirements:

A job is ready only when all its dependencies have succeeded.
Among ready jobs, run the highest priority first, with FIFO order for ties.
If a job raises, retry up to max_retries times. After that it is failed, and everything depending on it (transitively) is skipped.
Reject duplicate IDs, unknown dependencies, and cycles at submit time.

The scheduler owns `gpus` devices and each job needs `gpus_required` of them.
run_next is non-blocking: it starts a job on a thread and returns. If the top
job doesn't fit in the free GPUs, a lower job that fits may start instead, but
once a job has been passed over `max_skips` times nothing may start ahead of
it, so the free GPUs accumulate until it fits (no starvation).
"""

import heapq
import itertools
import threading
from typing import Any, Callable

PENDING = "pending"
READY = "ready"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
SKIPPED = "skipped"
CANCELLED = "cancelled"


class Scheduler:
    def __init__(self, gpus: int, max_skips: int = 3) -> None:
        if gpus < 1:
            raise ValueError("gpus must be >= 1")
        if max_skips < 0:
            raise ValueError("max_skips must be >= 0")
        self.gpus = gpus
        self.max_skips = max_skips
        self._free_gpus = gpus
        self._running = 0

        self._fns: dict[str, Callable[[], Any]] = {}
        self._priority: dict[str, int] = {}
        self._gpus_required: dict[str, int] = {}
        self._status: dict[str, str] = {}
        self._results: dict[str, Any] = {}
        self._errors: dict[str, BaseException] = {}

        # job_id -> (retries used so far, max retries allowed)
        self._retries: dict[str, tuple[int, int]] = {}

        # job_id -> how many times a lower job was started while this one was
        # ready but didn't fit. Hitting max_skips stops backfilling past it.
        self._passed_over: dict[str, int] = {}

        # Dependency bookkeeping: how many deps each job is still waiting on,
        # and the reverse edges so a finished job can notify its dependents.
        self._waiting_on: dict[str, int] = {}
        self._dependents: dict[str, list[str]] = {}

        # Ready queue of (-priority, seq, job_id). heapq is a min-heap, so we
        # negate priority; seq is a monotonically increasing tiebreaker that
        # gives FIFO order among equal priorities.
        self._ready: list[tuple[int, int, str]] = []
        self._seq = itertools.count()

        # Jobs cancelled by the user. Cancellation is lazy: the heap entry is
        # left in place and dropped when the scan in run_next reaches it.
        self._cancelled: set[str] = set()

        # Guards all state above; job threads notify it when they finish.
        self._cond = threading.Condition()

    def submit(self, job_id: str, fn: Callable[[], Any], priority: int = 0,
               depends_on: list[str] | None = None, max_retries: int = 0,
               gpus_required: int = 1) -> None:
        deps = list(dict.fromkeys(depends_on or []))  # dedupe, keep order

        with self._cond:
            if job_id in self._status:
                raise ValueError(f"duplicate job id: {job_id!r}")
            if job_id in deps:
                raise ValueError(f"job {job_id!r} cannot depend on itself")
            # O(D)
            unknown = [d for d in deps if d not in self._status]
            if unknown:
                raise ValueError(f"job {job_id!r} has unknown dependencies: {unknown}")
            if max_retries < 0:
                raise ValueError("max_retries must be >= 0")
            # A job bigger than the whole machine would block the queue forever.
            if not 1 <= gpus_required <= self.gpus:
                raise ValueError(f"gpus_required must be between 1 and {self.gpus}")
            # Cycles: every dependency must already be submitted, and a job can't
            # be edited after submission, so no new edge can ever close a cycle.

            self._fns[job_id] = fn
            self._priority[job_id] = priority
            self._gpus_required[job_id] = gpus_required
            self._retries[job_id] = (0, max_retries)
            self._passed_over[job_id] = 0
            self._dependents[job_id] = []
            self._status[job_id] = PENDING

            if any(self._status[d] in (FAILED, SKIPPED) for d in deps):
                self._skip(job_id)
                return
            # O(D)
            unfinished = [d for d in deps if self._status[d] != SUCCEEDED]
            for d in unfinished:
                self._dependents[d].append(job_id)
            self._waiting_on[job_id] = len(unfinished)
            # O(logN)
            if not unfinished:
                self._enqueue(job_id)

    def cancel(self, job_id: str) -> None:
        with self._cond:
            status = self.status(job_id)
            if status == RUNNING:
                raise RuntimeError(f"cannot cancel running job {job_id!r}")
            if status in (PENDING, READY):
                self._cancelled.add(job_id)
            # Already finished (succeeded/failed/skipped/cancelled): no-op.

    def run_next(self) -> str | None:
        """Start the best job that fits on a thread and return its id, or None
        if nothing ready can start right now. Never waits for a job to finish."""
        with self._cond:
            return self._start_next()

    def run_all(self) -> None:
        """Keep the GPUs busy until every job has finished."""
        with self._cond:
            while True:
                while self._start_next() is not None:
                    pass
                if self._running == 0:
                    return
                self._cond.wait()  # a job finished; GPUs or new jobs freed up

    def status(self, job_id: str) -> str:
        with self._cond:
            if job_id not in self._status:
                raise KeyError(job_id)
            # Report cancellation immediately, even before the lazy cleanup runs.
            if job_id in self._cancelled:
                return CANCELLED
            return self._status[job_id]

    def result(self, job_id: str) -> Any:
        with self._cond:
            status = self.status(job_id)
            if status == FAILED:
                raise RuntimeError(f"job {job_id!r} failed") from self._errors[job_id]
            if status != SUCCEEDED:
                raise RuntimeError(f"job {job_id!r} has no result (status: {status})")
            return self._results[job_id]

    # The helpers below assume self._cond is held.

    def _start_next(self) -> str | None:
        # Walk the heap in priority order by popping, looking for the first job
        # that fits. Jobs that don't fit are set aside and pushed back after.
        passed: list[tuple[int, int, str]] = []
        chosen = None
        while self._ready:
            # O(logN)
            entry = heapq.heappop(self._ready)
            job_id = entry[-1]
            if job_id in self._cancelled:
                self._status[job_id] = CANCELLED
                for child in self._dependents[job_id]:
                    self._skip(child)
                continue
            if self._gpus_required[job_id] <= self._free_gpus:
                chosen = job_id
                break
            passed.append(entry)
            if self._passed_over[job_id] >= self.max_skips:
                break  # starving: hold the free GPUs for this job

        if chosen is not None:
            for _, _, j in passed:
                self._passed_over[j] += 1
        for entry in passed:
            heapq.heappush(self._ready, entry)
        if chosen is None:
            return None

        self._free_gpus -= self._gpus_required[chosen]
        self._running += 1
        self._status[chosen] = RUNNING
        threading.Thread(target=self._run_job, args=(chosen, self._fns[chosen])).start()
        return chosen

    def _run_job(self, job_id: str, fn: Callable[[], Any]) -> None:
        # Runs on the job's own thread; fn executes without holding the lock.
        try:
            result, error = fn(), None
        except Exception as e:
            result, error = None, e

        with self._cond:
            self._free_gpus += self._gpus_required[job_id]
            self._running -= 1

            if error is not None:
                used, allowed = self._retries[job_id]
                if used < allowed:
                    self._retries[job_id] = (used + 1, allowed)
                    self._passed_over[job_id] = 0
                    self._enqueue(job_id)  # back of the line for its priority
                else:
                    self._status[job_id] = FAILED
                    self._errors[job_id] = error
                    for child in self._dependents[job_id]:
                        self._skip(child)
            else:
                self._status[job_id] = SUCCEEDED
                self._results[job_id] = result
                for child in self._dependents[job_id]:
                    self._waiting_on[child] -= 1
                    if self._waiting_on[child] == 0 and self._status[child] == PENDING:
                        self._enqueue(child)

            self._cond.notify_all()

    def _enqueue(self, job_id: str) -> None:
        self._status[job_id] = READY
        heapq.heappush(self._ready, (-self._priority[job_id], next(self._seq), job_id))

    def _skip(self, job_id: str) -> None:
        # Iterative DFS so deep dependency chains don't hit the recursion limit.
        stack = [job_id]
        while stack:
            j = stack.pop()
            if self._status[j] == SKIPPED:
                continue
            self._status[j] = SKIPPED
            stack.extend(self._dependents[j])
