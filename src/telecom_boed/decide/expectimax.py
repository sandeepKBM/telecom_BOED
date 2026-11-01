"""[telecom_boed] k-step expectimax over beliefs (the "k-step lookahead DUG" baseline) + shared lookahead machinery.

V_0(b) = EU(b) (decide/utility.eu, the myopic U of eq. 2');
V_k(b) = max( EU(b), max_{a nonterminal [, cost(b)+C(a) <= B if budget_aware]} [ sum_o p(o) V_{k-1}(b'_o) - lambda*C(a) - rho*R(a) ] ).
Root Q: claim = u_claim, transfer = u_transfer, nonterminal a = sum_o p(o) V_{k-1}(b'_o) - lambda*C(a) - rho*R(a)
(k = cfg lookahead.depth_k, counted in nonterminal actions). With `budget_aware=True` actions whose cost would exceed B are
excluded (root Q = NEG). Default `budget_aware=True` (plan spec; expected_decisions.yaml v2 after Opus fixed its reference solver,
2026-10-16). `budget_aware=False` is kept only as an explicit ablation. Reach-avoid always honours B.

`LookaheadBase` is shared with decide/reach_avoid.py: memo (key = blake2b digest of the sorted (particle class, rounded weight) arrays
of the belief's cached numpy view, v, kappa, rounded cost, transferred, remaining depth, root-set signature), hit/miss stats, and a
pluggable inner-node action provider
`inner_actions(belief, root_nonterminals) -> Sequence[CandidateAction]`. Default provider: all root nonterminals when
<= lookahead.inner_m of them, else (first inner_m in heuristic/manual order) U (top inner_m by myopic DUG at that node).
The provider must be a deterministic function of (belief, root nonterminal set). The memo outlives a decision (LRU, MEMO_MAX entries;
`reset()` at episode start): its key carries a signature of the root set (`root_key(root_nt)`; default = the candidates' (kind, actions),
or the full candidate tuple for an unknown provider), so a later decision with an inner-action-equivalent root set (e.g. the successor
belief predict(b, a)[o], same candidates) re-uses every node value of the previous search, e.g. a depth-(k-1) decision on it is all hits.
Works through the Predictor/Grader protocols only; no tau2 imports.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from telecom_boed.contracts import Belief, CandidateAction, DecisionKind, Grader, Predictor
from telecom_boed.decide.dug import dug
from telecom_boed.decide.heuristics import MANUAL_ORDER, cand_tool
from telecom_boed.decide.risk import risk
from telecom_boed.decide.utility import cost_of, eu, is_terminal, u_claim, u_transfer, view_of

NEG = -1e9  # score of an infeasible root action (budget would be exceeded / no lookahead)
InnerActions = Callable[[Belief, Sequence[CandidateAction]], Sequence[CandidateAction]]
MEMO_MAX = 500_000


def actions_signature(root_nt: Sequence[CandidateAction]) -> tuple:
    """What the default providers read from the root candidates: kind and device actions, in order (not text / candidate_idx)."""
    return tuple((c.kind, c.actions) for c in root_nt)


def default_inner_actions(predictor: Predictor, grader: Grader, cfg: Mapping[str, Any]) -> InnerActions:
    m = int(cfg["lookahead"]["inner_m"])

    def provider(belief: Belief, root_nt: Sequence[CandidateAction]) -> Sequence[CandidateAction]:
        if len(root_nt) <= m:
            return root_nt
        def rank(i: int) -> tuple:
            t = cand_tool(root_nt[i])
            return (MANUAL_ORDER.index(t) if t in MANUAL_ORDER else len(MANUAL_ORDER), i)
        by_heur = sorted(range(len(root_nt)), key=rank)[:m]
        by_dug = sorted(range(len(root_nt)), key=lambda i: (-dug(belief, root_nt[i], predictor, grader, cfg), i))[:m]
        return [root_nt[i] for i in sorted(set(by_heur) | set(by_dug))]
    provider.root_key = actions_signature  # type: ignore[attr-defined]
    return provider


class LookaheadBase:
    """Memo + inner action set + stats shared by KStepExpectimax / ReachAvoid / ReachAvoidMargin."""

    name = "lookahead"

    def __init__(self, predictor: Predictor, grader: Grader, cfg: Mapping[str, Any],
                 inner_actions: Optional[InnerActions] = None, depth: Optional[int] = None):
        self.predictor, self.grader, self.cfg = predictor, grader, cfg
        self.depth = int(cfg["lookahead"]["depth_k"]) if depth is None else int(depth)
        self.B = float(cfg["cost"]["budget_B"])
        self.lam, self.rho = float(cfg["cost"]["lambda"]), float(cfg["cost"]["rho"])
        self.quantum = float(cfg["lookahead"]["memo_round"])
        self.inner_actions = inner_actions or default_inner_actions(predictor, grader, cfg)
        self._memo: "OrderedDict[tuple, tuple[float, int]]" = OrderedDict()  # key -> (value, stats window of last use)
        self._root_nt: tuple = ()
        # root_key(root_nt) -> hashable signature of everything the inner-action provider reads from the root candidates
        self.root_key: Callable[[Sequence[CandidateAction]], Any] = getattr(self.inner_actions, "root_key", tuple)
        self._sig_ids: dict = {}
        self._sig = 0
        self.hits = self.misses = self.reused = 0
        self._win = 0
        self.last_decision: dict = {}

    # ---- memo
    def clear_memo(self) -> None:
        self._memo.clear()

    def reset_stats(self) -> None:
        """Open a new stats window (the controller does this per deepening level). `misses` counts the nodes a memo-less run of the
        window would have evaluated: true misses plus hits on entries stamped by an EARLIER window (`reused`), because the controller
        projects the next level from it (frontier size) and a carried-over entry is not free one level deeper."""
        self.hits = self.misses = self.reused = 0
        self._win += 1

    @property
    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": self.hits / n if n else 0.0, "size": len(self._memo),
                "reused": self.reused, "computed": self.misses - self.reused}

    def reset(self) -> None:
        """New episode: forget the memo and the stats."""
        self._memo.clear(); self._sig_ids.clear(); self._root_nt, self._sig = (), 0
        self.reset_stats()

    def _key(self, b: Belief, d: int) -> tuple:
        return (view_of(b, self.grader)[1].digest(self.quantum), b.v, b.kappa, round(b.cost, 6), b.transferred, d, self._sig)

    def _memoised(self, b: Belief, d: int, fn: Callable[[], float]) -> float:
        k = self._key(b, d)
        memo = self._memo
        e = memo.get(k)
        if e is not None:
            if e[1] == self._win:
                self.hits += 1
            else:  # computed in an earlier window (previous level / decision): count it as a miss that was served from the memo
                self.misses += 1
                self.reused += 1
                memo[k] = (e[0], self._win)
            memo.move_to_end(k)
            return e[0]
        self.misses += 1
        val = fn()
        memo[k] = (val, self._win)
        if len(memo) > MEMO_MAX:
            memo.popitem(last=False)
        return val

    def _set_root(self, candidates: Sequence[CandidateAction]) -> list[CandidateAction]:
        nt = [c for c in candidates if not is_terminal(c)]
        if tuple(nt) != self._root_nt:
            self._root_nt = tuple(nt)
            self._sig = self._sig_ids.setdefault(self.root_key(nt), len(self._sig_ids) + 1)
        return nt

    def anytime(self, belief: Belief, candidates: Sequence[CandidateAction], budget_s: Optional[float] = None,
                catch: tuple = (TimeoutError,)) -> tuple[list[float], int]:
        """Iterative deepening 0..depth_target with a wall-clock budget: returns (Q values of the deepest COMPLETED level, that depth).
        Level 0 (terminal Qs only, no predict) always exists, level 1 is the myopic answer (kstep: DUG + EU(b)); a level that raises one
        of `catch` (e.g. a deadline proxy's timeout) is discarded. A level is not started when the time left is below the previous
        level's duration. `self.last_decision` records depth / target / per-level ms / why it stopped (also the depth actually used)."""
        target, t0 = self.depth, time.perf_counter()
        vals, best, levels, why = None, -1, [], "done"
        try:
            for k in range(target + 1):
                left = None if budget_s is None else budget_s - (time.perf_counter() - t0)
                if k > 1 and left is not None and left < levels[-1] / 1e3:
                    why = "deadline"
                    break
                self.depth, t1 = k, time.perf_counter()
                try:
                    q = self.q_values(belief, candidates)
                except catch:
                    why = "timeout"
                    break
                vals, best = q, k
                levels.append((time.perf_counter() - t1) * 1e3)
        finally:
            self.depth = target
        self.last_decision = {"depth": best, "target": target, "levels_ms": [round(x, 3) for x in levels], "stopped": why,
                              "elapsed_ms": (time.perf_counter() - t0) * 1e3, "hits": self.hits, "misses": self.misses, "reused": self.reused}
        return vals, best

    def _inner(self, b: Belief) -> Sequence[CandidateAction]:
        return self.inner_actions(b, self._root_nt)

    def _penalty(self, b: Belief, a: CandidateAction) -> tuple[float, float]:
        c = cost_of(a, self.predictor, self.cfg)
        return c, self.lam * c + self.rho * risk(a, b)[0]


class KStepExpectimax(LookaheadBase):
    name = "kstep"

    def __init__(self, *a, budget_aware: bool = True, **kw):
        super().__init__(*a, **kw)
        self.budget_aware = budget_aware

    def _eu(self, b: Belief) -> float:
        return eu(b, self.grader, self.cfg)

    def _q(self, b: Belief, a: CandidateAction, d: int) -> float:
        """sum_o p(o) V_{d-1}(b'_o) - lambda C(a) - rho R(a); NEG if b.cost + C(a) > B."""
        c, pen = self._penalty(b, a)
        if self.budget_aware and b.cost + c > self.B:
            return NEG
        return sum(p * self._v(b2, d - 1) for p, b2 in self.predictor.predict(b, a).values()) - pen

    def _v(self, b: Belief, d: int) -> float:
        return self._memoised(b, d, lambda: self._v_raw(b, d))

    def _v_raw(self, b: Belief, d: int) -> float:
        best = self._eu(b)
        if d > 0:
            for a in self._inner(b):
                best = max(best, self._q(b, a, d))
        return best

    def q_values(self, belief: Belief, candidates: Sequence[CandidateAction]) -> list[float]:
        self._set_root(candidates)
        out = []
        for c in candidates:
            if c.kind == DecisionKind.CLAIM_RESOLVED:
                out.append(u_claim(belief, self.grader, self.cfg))
            elif c.kind == DecisionKind.TRANSFER:
                out.append(u_transfer(belief, self.grader, self.cfg))
            else:
                out.append(self._q(belief, c, self.depth) if self.depth > 0 else NEG)
        return out

    score = q_values

    def value(self, belief: Belief, candidates: Sequence[CandidateAction]) -> float:
        """V_k at the root over the given candidates (max of the Q values)."""
        return max(self.q_values(belief, candidates))

    def decide(self, belief: Belief, candidates: Sequence[CandidateAction]) -> int:
        q = self.q_values(belief, candidates)
        return max(range(len(q)), key=lambda i: (q[i], candidates[i].kind == DecisionKind.CLAIM_RESOLVED, -i))


def bench_scoring(predictor: Predictor, belief: Belief, candidates: Sequence[CandidateAction], scorer: Any, reps: int = 20) -> dict:
    """Wall-clock latency of one cold scorer.score() (memo cleared each rep) and of one predictor.predict, in ms (p50/p95)."""
    def pct(xs: list) -> tuple:
        return float(np.percentile(xs, 50)), float(np.percentile(xs, 95))
    nt = [c for c in candidates if not is_terminal(c)]
    t_pred, t_score = [], []
    for _ in range(reps):
        if nt:
            t0 = time.perf_counter()
            predictor.predict(belief, nt[0])
            t_pred.append((time.perf_counter() - t0) * 1e3)
        if hasattr(scorer, "clear_memo"):
            scorer.clear_memo()
        t0 = time.perf_counter()
        scorer.score(belief, candidates)
        t_score.append((time.perf_counter() - t0) * 1e3)
    s50, s95 = pct(t_score)
    p50, p95 = pct(t_pred) if t_pred else (0.0, 0.0)
    return {"p50_ms": s50, "p95_ms": s95, "predict_p50_ms": p50, "predict_p95_ms": p95, "reps": reps}
