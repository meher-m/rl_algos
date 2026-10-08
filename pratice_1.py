"""
Requirements:

A job is ready only when all its dependencies have succeeded.
Among ready jobs, run the highest priority first, with FIFO order for ties.
If a job raises, retry up to max_retries times. After that it is failed, and everything depending on it (transitively) is skipped.
Reject duplicate IDs, unknown dependencies, and cycles at submit time.
"""

import heapq
import itertools
from typing import Any, Callable

PENDING = "pending"
READY = "ready"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
SKIPPED = "skipped"
CANCELLED = "cancelled"


class Scheduler:
    def __init__(self) -> None:
        self._fns: dict[str, Callable[[], Any]] = {}
        self._priority: dict[str, int] = {}
        self._status: dict[str, str] = {}
        self._results: dict[str, Any] = {}
        self._errors: dict[str, BaseException] = {}

        # job_id -> (retries used so far, max retries allowed)
        self._retries: dict[str, tuple[int, int]] = {}

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
        # left in place and dropped when it reaches the top in run_next.
        self._cancelled: set[str] = set()

    def submit(self, job_id: str, fn: Callable[[], Any], priority: int = 0,
               depends_on: list[str] | None = None, max_retries: int = 0) -> None:
        deps = list(dict.fromkeys(depends_on or []))  # dedupe, keep order

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
        # Cycles: every dependency must already be submitted, and a job can't
        # be edited after submission, so no new edge can ever close a cycle.
        
        self._fns[job_id] = fn
        self._priority[job_id] = priority
        self._retries[job_id] = (0, max_retries)
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
        status = self.status(job_id)
        if status == RUNNING:
            raise RuntimeError(f"cannot cancel running job {job_id!r}")
        if status in (PENDING, READY):
            self._cancelled.add(job_id)
        # Already finished (succeeded/failed/skipped/cancelled): no-op.

    def run_next(self) -> str | None:
        # Pop past any cancelled entries; each one is finalized here and its
        # dependents skipped. A pending job that was cancelled gets enqueued
        # normally once its deps succeed, then is dropped here.
        while self._ready:
            # O(logN)
            _, _, job_id = heapq.heappop(self._ready)
            if job_id not in self._cancelled:
                break
            self._status[job_id] = CANCELLED
            for child in self._dependents[job_id]:
                self._skip(child)
        else:
            return None

        self._status[job_id] = RUNNING
        try:
            result = self._fns[job_id]()
        except Exception as e:
            used, allowed = self._retries[job_id]
            if used < allowed:
                self._retries[job_id] = (used + 1, allowed)
                self._enqueue(job_id)  # back of the line for its priority
            else:
                self._status[job_id] = FAILED
                self._errors[job_id] = e
                for child in self._dependents[job_id]:
                    self._skip(child)
            return job_id

        self._status[job_id] = SUCCEEDED
        self._results[job_id] = result
        for child in self._dependents[job_id]:
            self._waiting_on[child] -= 1
            if self._waiting_on[child] == 0 and self._status[child] == PENDING:
                self._enqueue(child)
        return job_id

    def run_all(self) -> None:
        while self.run_next() is not None:
            pass

    def status(self, job_id: str) -> str:
        if job_id not in self._status:
            raise KeyError(job_id)
        # Report cancellation immediately, even before the lazy cleanup runs.
        if job_id in self._cancelled:
            return CANCELLED
        return self._status[job_id]

    def result(self, job_id: str) -> Any:
        status = self.status(job_id)
        if status == FAILED:
            raise RuntimeError(f"job {job_id!r} failed") from self._errors[job_id]
        if status != SUCCEEDED:
            raise RuntimeError(f"job {job_id!r} has no result (status: {status})")
        return self._results[job_id]

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
