"""[telecom_boed] Simulator-as-likelihood: one exact tau2 env per hypothesis, stepped with the REAL tools (W1-S2).

Imports tau2 (via env.snapshot); the caller must set TAU2_DATA_DIR and apply tau2_patches first.
- build_state(hyp): fast_env + every fault-injection action -> Snapshot (cached per hypothesis key).
- apply_calls(state, calls): pure restore -> get_response per call -> (new Snapshot, tool contents). Memoized on
  (state.key, calls); the Snapshot wrapper (parent_id, step, snapshot_id) is rebuilt per call so lineage stays exact.
- step(particles, calls): apply_calls on every particle, then merge.
- grade(state, hyp, transferred): exact tau2 reward (evaluator.py:223-256) for a trajectory that ends with
  USER_STOP or AGENT_STOP; tau2 gives 0 for any other termination (evaluator.py:119-129), which grade does not model.
  reward = [all env assertions on the state == assert_value] * ([transferred] if "ACTION" in reward_basis else 1).
  ACTION is a name-only match of transfer_to_human_agents (compare_args=[]), a GENERIC tool invisible in env state.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall

from telecom_boed.contracts import DeviceAction, HypKey, Hypothesis, Particle, Snapshot
from telecom_boed.env import snapshot as S
from telecom_boed.env.hypotheses import hypothesis_by_key

_STATE_CACHE: dict[HypKey, Snapshot] = {}
_APPLY_MEMO: dict[tuple, tuple[Snapshot, tuple[str, ...]]] = {}
_GRADE_MEMO: dict[tuple, float] = {}
_INTERN: dict[str, dict] = {}  # agent-DB hash -> dump
STATS = {"apply_hit": 0, "apply_miss": 0, "grade_hit": 0, "grade_miss": 0}
MEMO_CAP = 300_000

# P0-2 (boed_v2): perf switches, all OFF by default so existing configs reproduce exactly. Set from cfg["perf"] by ParticleFilter.__init__
# (absolute: keys missing from cfg fall back to the defaults) and shipped to stepper workers with each chunk (belief/stepper.py).
PERF_DEFAULT = {"reuse_state_hash": False}
PERF: dict = dict(PERF_DEFAULT)


def configure_perf(**kw) -> None:
    bad = set(kw) - set(PERF_DEFAULT)
    if bad:
        raise ValueError(f"unknown perf switch(es) {sorted(bad)}; valid: {sorted(PERF_DEFAULT)}")
    PERF.update({k: bool(v) for k, v in kw.items()})


def clear_caches() -> None:
    _STATE_CACHE.clear(); _APPLY_MEMO.clear(); _GRADE_MEMO.clear(); _INTERN.clear()
    for k in STATS:
        STATS[k] = 0


def _intern(snap: Snapshot) -> Snapshot:
    """Share one agent-DB dict among snapshots with equal agent-DB hash (it rarely changes; never mutated).
    The user DB is not interned: reading it by attribute is what the leakage scanner forbids, and it is mostly unique."""
    a = _INTERN.setdefault(snap.key[0], snap.agent_db)
    return snap if a is snap.agent_db else replace(snap, agent_db=a)


def build_state(hyp: Hypothesis) -> Snapshot:
    """Initial env state of a hypothesis: fast_env + every init action (same order as tau2 set_state)."""
    snap = _STATE_CACHE.get(hyp.key)
    if snap is None:
        env = S.fast_env()
        for c in hyp.init_actions:
            env.run_env_function_call(EnvFunctionCall(env_type=c.env_type, func_name=c.func_name, arguments=c.args_dict()))
        snap = _STATE_CACHE[hyp.key] = _intern(S.snapshot(env))
    return snap


def initial_particles(hyps: Sequence[Hypothesis], weights: Optional[Sequence[float]] = None) -> tuple[Particle, ...]:
    """One particle per hypothesis (uniform weights unless given; given weights are used as-is)."""
    w = [1.0 / len(hyps)] * len(hyps) if weights is None else list(weights)
    assert len(w) == len(hyps)
    return tuple(Particle(h.key, build_state(h), wi) for h, wi in zip(hyps, w))


def _thaw(v):
    """Inverse of contracts._frozen_items for nested values: tuple of (str, x) pairs -> dict, other tuples -> list.
    (Only set_apn_settings takes a nested dict; the frozen form cannot tell a pair-list from a dict, so a list of
    (str, x) pairs would be misread as a dict. No telecom tool takes such a list.)"""
    if isinstance(v, tuple):
        if v and all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in v):
            return {k: _thaw(x) for k, x in v}
        return [_thaw(x) for x in v]
    return v


def _tool_calls(state_id: str, calls: Sequence[DeviceAction]) -> list[ToolCall]:
    return [ToolCall(id=f"{state_id}:{i}", name=c.tool, arguments={k: _thaw(v) for k, v in c.args_dict().items()},
                     requestor=c.requestor) for i, c in enumerate(calls)]


def apply_calls(state: Snapshot, calls: tuple[DeviceAction, ...]) -> tuple[Snapshot, tuple[str, ...]]:
    """Run `calls` in order on a restored copy of `state`. Pure: `state` is not modified.

    Returns (new snapshot with parent_id=state.snapshot_id, step=state.step+1; ToolMessage contents in call order).
    Tool errors come back as "Error: ..." contents exactly as tau2 reports them (state changes made before the
    exception stay, as in the live env). An empty `calls` returns the same state, still as a new step.
    """
    mk = (state.key, calls)
    hit = _APPLY_MEMO.get(mk)
    if hit is None:
        STATS["apply_miss"] += 1
        env = S.restore(state)
        outs = tuple(env.get_response(tc).content or "" for tc in _tool_calls(state.snapshot_id or "s", calls))
        hit = (_intern(S.snapshot_reusing(env, state) if PERF["reuse_state_hash"] else S.snapshot(env)), outs)
        if len(_APPLY_MEMO) >= MEMO_CAP:
            _APPLY_MEMO.clear()
        _APPLY_MEMO[mk] = hit
    else:
        STATS["apply_hit"] += 1
    base, outs = hit
    step_no = state.step + 1
    sid = S._sha([base.key, state.snapshot_id, step_no])[:12]
    return replace(base, snapshot_id=sid, parent_id=state.snapshot_id, step=step_no, messages=state.messages), outs


def step(particles: Sequence[Particle], calls: tuple[DeviceAction, ...], merge: bool = True
         ) -> tuple[tuple[Particle, ...], tuple[tuple[str, ...], ...]]:
    """Apply `calls` to every particle -> (particles', outputs'); outputs'[i] are the contents for particles'[i].

    merge=True merges particles with identical (post-state key, goal_id, outputs) and sums their weights; the merged
    particle keeps the hyp_key / exec_history of its first member (input order). `outputs` is part of the merge key
    on purpose: two different pre-states can reach the same post-state with different tool texts ("VPN already
    connected" vs "connected successfully"), and the filter needs the likelihood of the observed text.
    merge=False returns one output per input particle, aligned with the input order.
    """
    res = [(p, *apply_calls(p.state, calls)) for p in particles]
    if not merge:
        return (tuple(replace(p, state=ns) for p, ns, _ in res), tuple(o for _, _, o in res))
    groups: dict[tuple, int] = {}
    ps: list[Particle] = []
    outs: list[tuple[str, ...]] = []
    for p, ns, o in res:
        k = (ns.key, hypothesis_by_key(p.hyp_key).goal_id, o)
        i = groups.get(k)
        if i is None:
            groups[k] = len(ps)
            ps.append(replace(p, state=ns))
            outs.append(o)
        else:
            ps[i] = replace(ps[i], weight=ps[i].weight + p.weight)
    return tuple(ps), tuple(outs)


def grade(state: Snapshot, hyp: Hypothesis, transferred: bool) -> float:
    """Exact tau2 final reward of `state` under hypothesis `hyp` (1.0 or 0.0); see module docstring."""
    if "ACTION" in hyp.reward_basis and not transferred:
        return 0.0  # ACTION factor is 0, so the product is 0 whatever the assertions say
    mk = (state.key, hyp.goal_id)
    r = _GRADE_MEMO.get(mk)
    if r is None:
        STATS["grade_miss"] += 1
        env = S.restore(state)
        ok = all(env.run_env_assertion(EnvAssertion(env_type=a.env_type, func_name=a.func_name,
                                                    arguments=a.args_dict(), assert_value=bool(a.assert_value)),
                                       raise_assertion_error=False) for a in hyp.goal_assertions)
        r = _GRADE_MEMO[mk] = 1.0 if ok else 0.0
    else:
        STATS["grade_hit"] += 1
    return r


class ParticleGrader:
    """contracts.Grader backed by the real tau2 env."""

    def grade(self, state: Snapshot, hyp_key: HypKey, transferred: bool) -> float:
        return grade(state, hypothesis_by_key(hyp_key), transferred)

    def requires_transfer(self, hyp_key: HypKey) -> bool:
        return hypothesis_by_key(hyp_key).requires_transfer
