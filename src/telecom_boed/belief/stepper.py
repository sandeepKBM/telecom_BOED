"""[telecom_boed] Parallel particle stepper (owner: W2-S5). Imports tau2 only through env.particles.

One serial `env.particles.apply_calls` costs about 0.9-1 ms cold (restore + tool call + snapshot; W1-S2: 2.06 s for the 1,984
mms particles; 2 ms/step measured on a host at load 85), against a 500 ms p95 controller budget.
`ParallelStepper.step_many(states, calls)` splits the states over a fork-started process pool; `step_pairs` does the same for a
heterogeneous wave of (state, calls, outputs_only) items (one dispatch per call depth for the filter) and `grade_many` for
env.particles.grade. apply_calls is a pure function of (state, calls), so the result is identical to the serial loop
(same order, same Snapshot fields including snapshot_id), which tests/test_filter.py::test_parallel_matches_serial asserts.

Per-worker needs:
- Start method is `fork`, so each worker inherits the already imported AND patched tau2 (env.tau2_patches.apply() must have
  run in the parent before the pool is created) and the parent's caches as of the fork. Create the stepper AFTER tau2 import,
  patching and TAU2_DATA_DIR setup. Do not create it before.
- A worker's own apply_calls memo is useless to the parent (different process), so workers clear it after every task to bound
  RSS; the parent installs results in its own caches (belief/filter.py `_step_cache`, env.particles `_GRADE_MEMO`). The parent re-interns agent-DB dicts
  (unpickling creates one copy per chunk).
- `persistent=True` (default): one pool, states are pickled to the workers (~1-8 KB each; agent_db is shared inside a chunk).
  `persistent=False`: fork a fresh pool per call and hand the states over through a module global (no state pickling, but the
  fork cost is paid on every call). tests/benchmarks pick the faster; the default is persistent.
- n_workers <= 1: plain serial loop in the calling process (no pool, no pickling).
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
from typing import Optional, Sequence

from telecom_boed.contracts import DeviceAction, Snapshot
from telecom_boed.env import particles as P
from telecom_boed.env.hypotheses import load_catalog

StepResult = tuple[Snapshot, tuple[str, ...]]

_JOB: Optional[list] = None  # work items inherited by fork-per-call workers
_WORKER_MEMO_CAP = 20_000


def _purge_worker_memo() -> None:
    if len(P._APPLY_MEMO) > _WORKER_MEMO_CAP:
        P._APPLY_MEMO.clear()


def _one(item: tuple) -> tuple:
    state, calls, outputs_only = item
    snap, outs = P.apply_calls(state, calls)
    return (None, outs) if outputs_only else (snap, outs)


def _run_chunk(items: list) -> list[StepResult]:
    out = [_one(it) for it in items]
    _purge_worker_memo()
    return out


def _run_chunk_perf(args: tuple) -> list[StepResult]:
    """Persistent-pool chunk that first adopts the parent's env.particles.PERF switches (the pool was forked before they were set).
    Only used when a switch is ON, so the default path is the unchanged _run_chunk."""
    perf, items = args
    P.PERF.update(perf)
    return _run_chunk(items)


def _run_range(args: tuple) -> list[StepResult]:
    lo, hi = args
    assert _JOB is not None
    out = [_one(_JOB[i]) for i in range(lo, hi)]
    _purge_worker_memo()
    return out


def _grade_chunk(items: list) -> list[float]:
    g = P.ParticleGrader()
    out = [g.grade(s, hk, tr) for s, hk, tr in items]
    if len(P._GRADE_MEMO) > _WORKER_MEMO_CAP:
        P._GRADE_MEMO.clear()
    return out


def _bounds(n: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, min(parts, n))
    q, r = divmod(n, parts)
    out, lo = [], 0
    for i in range(parts):
        hi = lo + q + (1 if i < r else 0)
        out.append((lo, hi))
        lo = hi
    return out


class ParallelStepper:
    """step_many(states, calls) -> [(new Snapshot, tool contents)] aligned with `states`; same as serial apply_calls."""

    def __init__(self, n_workers: int = 1, persistent: bool = True, chunks_per_worker: int = 4):
        self.n_workers = int(n_workers)
        self.persistent = persistent
        self.chunks_per_worker = chunks_per_worker
        self._pool = None
        self._perf_shipped = False
        if self.n_workers > 1 and persistent:
            load_catalog()  # so workers inherit the hypothesis catalog instead of each re-reading tasks.json
            self._pool = mp.get_context("fork").Pool(self.n_workers)
            atexit.register(self.close)

    # ------------------------------------------------------------------ API
    def step_many(self, states: Sequence[Snapshot], calls: tuple[DeviceAction, ...],
                  outputs_only: bool = False) -> list[StepResult]:
        """outputs_only=True returns (None, contents) and skips shipping snapshots (for read-only calls, whose state
        does not change)."""
        return self.step_pairs([(s, calls, outputs_only) for s in states])

    def step_pairs(self, items: Sequence[tuple]) -> list[StepResult]:
        """Heterogeneous batch: items = [(state, calls, outputs_only)] -> results aligned with items. One dispatch for a whole
        wave of independent (state, call) pairs; share the `calls` tuple object between items to keep pickles small."""
        n = len(items)
        if self.n_workers <= 1 or n < 2 * self.n_workers:
            return [_one(it) for it in items]
        bounds = _bounds(n, self.n_workers * self.chunks_per_worker)
        if self.persistent:
            assert self._pool is not None, "stepper was closed"
            if P.PERF != P.PERF_DEFAULT or self._perf_shipped:  # also ship the reset to defaults after a non-default run
                perf = dict(P.PERF)
                self._perf_shipped = P.PERF != P.PERF_DEFAULT
                parts = self._pool.map(_run_chunk_perf, [(perf, list(items[lo:hi])) for lo, hi in bounds], chunksize=1)
            else:
                parts = self._pool.map(_run_chunk, [list(items[lo:hi]) for lo, hi in bounds], chunksize=1)
        else:
            global _JOB
            _JOB = list(items)
            try:
                with mp.get_context("fork").Pool(self.n_workers) as pool:
                    parts = pool.map(_run_range, bounds, chunksize=1)
            finally:
                _JOB = None
        return [(snap if snap is None else P._intern(snap), outs) for part in parts for snap, outs in part]

    def grade_many(self, items: Sequence[tuple]) -> list[float]:
        """items = [(state, hyp_key, transferred)] -> env.particles.grade values, aligned. Serial below 2 x n_workers items;
        the pool path never touches the parent's grade memo (the caller installs results)."""
        n = len(items)
        g = P.ParticleGrader()
        if self.n_workers <= 1 or n < 2 * self.n_workers or not self.persistent:
            return [g.grade(s, hk, tr) for s, hk, tr in items]
        assert self._pool is not None, "stepper was closed"
        parts = self._pool.map(_grade_chunk, [list(items[lo:hi]) for lo, hi in _bounds(n, self.n_workers * self.chunks_per_worker)], chunksize=1)
        return [x for part in parts for x in part]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "ParallelStepper":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
