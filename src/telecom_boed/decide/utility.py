"""[telecom_boed] Utilities U(r, h') of the contract's "Utility semantics" (docs/proposal.md §3, eq. 2').

Pure functions over a Belief, a Grader and the nested cfg dict (configs/boed_default.yaml shape: cfg["cost"], cfg["utility"]).
No tau2 imports.

- U(claim_resolved, h') = grade(h', transferred) - c_FA * [grade == 0]
- U(transfer, h')       = grade(h', True) if h.requires_transfer else -c_esc
- EU(beta) = max_r sum_h b(h) U(r, h) = max(u_claim, u_transfer)   (the "max_r" term of eq. 2')
- Reach-avoid (eq. 4, probability semantics): q_claim_reach = P(repaired and v=1), q_transfer_reach = P(requires_transfer and
  grade(h', True) == 1). repaired(h') := grade(h', belief.transferred) == 1 and not requires_transfer(h)  (toy_cases.yaml header).

Performance (W2-P1): every belief-level sum below runs on a cached numpy `BeliefView` (weights + per-particle class ids; grade /
requires_transfer looked up once per (state.key, hyp_key) class and grader). The arithmetic is kept bit-identical to the plain
per-particle loops: sequential `np.cumsum` for the former `tot += ...` loops, builtin `sum` over `.tolist()` for the former `sum(...)`.

Action cost C(a): `action_cost` (turn units). A Predictor may expose `action_cost(cand) -> float` to override it (the toy
POMDP does, with the per-action `cost` of toy_cases.yaml); `cost_of` picks whichever applies.
"""
from __future__ import annotations

import hashlib
import weakref
from typing import Any, Mapping

import numpy as np

from telecom_boed.contracts import Belief, CandidateAction, DecisionKind, Grader

TERMINALS = (DecisionKind.CLAIM_RESOLVED, DecisionKind.TRANSFER)


def is_terminal(cand: CandidateAction) -> bool:
    return cand.kind in TERMINALS


def action_cost(cand: CandidateAction, cfg: Mapping[str, Any]) -> float:
    """C(a) in turn units: turn_cost for a message to the user (REQUEST_DEVICE / ASK), tool_cost for a backend call, 0 for terminals."""
    if cand.kind in (DecisionKind.REQUEST_DEVICE, DecisionKind.ASK):
        return float(cfg["cost"]["turn_cost"])
    if cand.kind == DecisionKind.AGENT_TOOL:
        return float(cfg["cost"]["tool_cost"])
    return 0.0


def cost_of(cand: CandidateAction, predictor: Any, cfg: Mapping[str, Any]) -> float:
    fn = getattr(predictor, "action_cost", None)
    return float(fn(cand)) if fn is not None else action_cost(cand, cfg)


# --------------------------------------------------------------------------- numpy belief views
TABLE_CAP = 1_500_000  # classes per grader before the table (and its views) is rebuilt


class BeliefView:
    """Array form of one (frozen) Belief for one grader table. `ids` index the table's grade arrays; `hk` are content hashes of
    the (state.key, hyp_key) class (stable across table rebuilds, used for memo keys). `memo` caches per-belief results."""
    __slots__ = ("n", "w", "ids", "hk", "memo", "__weakref__")

    def __init__(self, w: np.ndarray, ids: np.ndarray, hk: np.ndarray):
        self.n, self.w, self.ids, self.hk, self.memo = len(w), w, ids, hk, {}

    def digest(self, quantum: float) -> bytes:
        """Order-independent fingerprint of {(class, round(weight/quantum))} (the lookahead memo key's particle part)."""
        d = self.memo.get(("digest", quantum))
        if d is None:
            q = np.round(self.w / quantum).astype(np.int64)
            o = np.lexsort((q, self.hk))
            h = hashlib.blake2b(digest_size=16)
            h.update(self.hk[o].tobytes()); h.update(q[o].tobytes())
            d = self.memo[("digest", quantum)] = h.digest()
        return d


class GradeTable:
    """Per-grader class table: (state.key, hyp_key) -> id, with lazily filled grade(False) / grade(True) / requires_transfer arrays."""

    def __init__(self, grader: Grader):
        self.grader = grader
        self.reset()

    def reset(self) -> None:
        self.cid: dict = {}
        self.cap = 1024
        self.g = [np.full(self.cap, np.nan), np.full(self.cap, np.nan)]  # grade(., transferred=False / True); nan = not graded yet
        self.req = np.zeros(self.cap, dtype=bool)
        self.hk = np.zeros(self.cap, dtype=np.int64)
        self.views: "weakref.WeakKeyDictionary[Belief, BeliefView]" = weakref.WeakKeyDictionary()

    def _grow(self, n: int) -> None:
        while self.cap < n:
            self.cap *= 2
        for a in (0, 1):
            self.g[a] = np.concatenate([self.g[a], np.full(self.cap - len(self.g[a]), np.nan)])
        self.req = np.concatenate([self.req, np.zeros(self.cap - len(self.req), dtype=bool)])
        self.hk = np.concatenate([self.hk, np.zeros(self.cap - len(self.hk), dtype=np.int64)])

    def view(self, b: Belief) -> BeliefView:
        v = self.views.get(b)
        if v is not None:
            return v
        if len(self.cid) > TABLE_CAP:
            self.reset()
        ps, cid = b.particles, self.cid
        keys = [(p.state.key, p.hyp_key) for p in ps]
        ids = [cid.get(k, -1) for k in keys]
        if -1 in ids:
            for j, k in enumerate(keys):
                if ids[j] < 0:
                    i = cid.get(k)
                    if i is None:
                        i = cid[k] = len(cid)
                        if i >= self.cap:
                            self._grow(i + 1)
                        self.req[i] = bool(self.grader.requires_transfer(ps[j].hyp_key))
                        self.hk[i] = hash(k)
                    ids[j] = i
        v = self.views[b] = BeliefView(np.array([p.weight for p in ps], dtype=np.float64), np.array(ids, dtype=np.int64), self.hk[ids])
        return v

    def grades(self, b: Belief, v: BeliefView, flag: bool, only: Any = None) -> np.ndarray:
        """grade(state, hyp, flag) per particle (nan where `only` (a bool mask) excludes it and it was never graded)."""
        arr = self.g[flag]
        vals = arr[v.ids]
        miss = np.isnan(vals) if only is None else (np.isnan(vals) & only)
        if miss.any():
            ps, grade = b.particles, self.grader.grade
            for j in np.flatnonzero(miss).tolist():
                i = v.ids[j]
                if arr[i] != arr[i]:  # still nan (a class may repeat within the belief)
                    arr[i] = float(grade(ps[j].state, ps[j].hyp_key, flag))
            vals = arr[v.ids]
        return vals


_TABLES: "weakref.WeakKeyDictionary[Any, GradeTable]" = weakref.WeakKeyDictionary()


def table_for(grader: Any) -> GradeTable:
    try:
        t = _TABLES.get(grader)
        if t is None:
            t = _TABLES[grader] = GradeTable(grader)
        return t
    except TypeError:  # grader not weak-referenceable / hashable: no cross-call cache
        return GradeTable(grader)


def view_of(belief: Belief, grader: Grader) -> tuple[GradeTable, BeliefView]:
    t = table_for(grader)
    return t, t.view(belief)


def _seq_sum(x: np.ndarray) -> float:
    """Sequential left-to-right float sum (what `tot = 0.0; tot += x_i` computes)."""
    return float(np.cumsum(x)[-1]) if len(x) else 0.0


def u_claim(belief: Belief, grader: Grader, cfg: Mapping[str, Any]) -> float:
    c_fa = float(cfg["utility"]["c_FA"])
    t, v = view_of(belief, grader)
    r = v.memo.get(("uc", c_fa))
    if r is None:
        g = t.grades(belief, v, belief.transferred)
        r = v.memo[("uc", c_fa)] = _seq_sum(v.w * (g - np.where(g == 0, c_fa, 0.0)))
    return r


def u_transfer(belief: Belief, grader: Grader, cfg: Mapping[str, Any]) -> float:
    c_esc = float(cfg["utility"]["c_esc"])
    t, v = view_of(belief, grader)
    r = v.memo.get(("ut", c_esc))
    if r is None:
        req = t.req[v.ids]
        g1 = t.grades(belief, v, True, only=req)
        r = v.memo[("ut", c_esc)] = _seq_sum(v.w * np.where(req, g1, -c_esc))
    return r


def eu(belief: Belief, grader: Grader, cfg: Mapping[str, Any]) -> float:
    """max_r sum_h b(h) U(r, h) of eq. 2'."""
    return max(u_claim(belief, grader, cfg), u_transfer(belief, grader, cfg))


def p_repaired(belief: Belief, grader: Grader) -> float:
    """P(grade(state, h, belief.transferred) == 1 and not requires_transfer(h))."""
    t, v = view_of(belief, grader)
    r = v.memo.get("pr")
    if r is None:
        g = t.grades(belief, v, belief.transferred)
        r = v.memo["pr"] = sum(v.w[(g == 1) & ~t.req[v.ids]].tolist())
    return r


def q_claim_reach(belief: Belief, grader: Grader) -> float:
    """Eq. 4: Q(claim_resolved) = P(repaired and v = 1); 0 unless the belief is verified."""
    return p_repaired(belief, grader) if belief.v else 0.0


def q_transfer_reach_pen(belief: Belief, grader: Grader, c_esc: float) -> float:
    """reach_avoid.transfer_penalty (off by default, 2026-11-01): Q(transfer) = P(requires_transfer and grade(h', True) == 1)
    - c_esc * P(not requires_transfer), so escalating while a fixable fault is likely costs about c_esc * P(fixable)."""
    t, v = view_of(belief, grader)
    r = v.memo.get(("qtp", c_esc))
    if r is None:
        req = t.req[v.ids]
        r = v.memo[("qtp", c_esc)] = q_transfer_reach(belief, grader) - c_esc * sum(v.w[~req].tolist())
    return r


def q_transfer_reach(belief: Belief, grader: Grader) -> float:
    """Eq. 4: Q(transfer) = P(requires_transfer and grade(h', True) == 1). Transferring a fixable fault has value 0 (kappa)."""
    t, v = view_of(belief, grader)
    r = v.memo.get("qt")
    if r is None:
        req = t.req[v.ids]
        g1 = t.grades(belief, v, True, only=req)
        r = v.memo["qt"] = sum(v.w[req & (g1 == 1)].tolist())
    return r
