"""[telecom_boed] Reach-avoid value over beliefs, eq. 4 of docs/proposal.md §3 (the project's main decision rule).

V(b) = 0 if cost(b) > B or kappa(b);  else  max{ Q_claim(b), Q_transfer(b), max_a Q(b, a) }   (max_a only while depth remains)
Q_claim(b) = P_b(repaired and v=1);  Q_transfer(b) = P_b(requires_transfer and grade(h', True)=1)   (contracts.py "Utility semantics")
Q(b, a)    = gamma * sum_o p(o) V(b'_o) - lambda*C(a) - rho*R(a)   for nonterminal a; if cost(b)+C(a) > B the successor is failure,
             so Q = -(lambda*C(a) + rho*R(a))  (the toy `budget` case: -0.1).
Depth: k = lookahead.depth_k nonterminal actions; at depth 0 only the terminal Qs. Root Q(a) uses V at depth k-1.

Decision rule (`decide`): commit iff Q(claim) >= max_{a != claim} Q(a) and P_b(repaired) >= 1 - delta; otherwise escalate (transfer)
if max_{a != claim} Q(a) < eta; otherwise argmax over a != claim (a claim that failed the commit test is never the fallback argmax;
ties: first index). If there is no transfer candidate the escalation step is skipped; with only a claim candidate, claim.

`ReachAvoidMargin` ("reach_margin"): Hsu et al. 2021 discounted backup with the margins of proposal §3
  l(b) = max( min{ P(repaired) - p*, v - 1/2 },  P(requires_transfer and grade(True)=1) - p* )   [2nd term: transfer extension]
  g(b) = min{ B - cost(b), 1/2 - kappa(b) }
  V_d(b) = (1-gamma)*min{l,g} + gamma*min{ g, max{ l, max_a E_o V_{d-1}(b'_o) } },  V_0 = (1-gamma)*min{l,g} + gamma*min{g,l}
Sign convention: V > 0 iff reach-before-avoid is certain; a failed belief (kappa) has V <= g = -1/2 (not 0). lambda, rho are not used.
Q scores: nonterminal = E_o V_{d-1}(b'_o); claim = min{ g, min{P(repaired)-p*, v-1/2} }; transfer = min{ g, P(transfer ok) - p* }.
"""
from __future__ import annotations

from typing import Sequence

from telecom_boed.contracts import Belief, CandidateAction, DecisionKind
from telecom_boed.decide.expectimax import NEG, LookaheadBase, bench_scoring  # noqa: F401  (bench_scoring re-exported)
from telecom_boed.decide.utility import is_terminal, p_repaired, q_claim_reach, q_transfer_reach

TOL = 1e-12


class ReachAvoid(LookaheadBase):
    name = "reach"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gamma = float(self.cfg["reach_avoid"]["gamma"])
        self.delta = float(self.cfg["reach_avoid"]["delta"])
        self.eta = float(self.cfg["reach_avoid"]["eta"])

    def _q(self, b: Belief, a: CandidateAction, d: int) -> float:
        c, pen = self._penalty(b, a)
        if b.cost + c > self.B:
            return -pen
        return self.gamma * sum(p * self._v(b2, d - 1) for p, b2 in self.predictor.predict(b, a).values()) - pen

    def _v(self, b: Belief, d: int) -> float:
        if b.cost > self.B or b.kappa:
            return 0.0
        return self._memoised(b, d, lambda: self._v_raw(b, d))

    def _v_raw(self, b: Belief, d: int) -> float:
        best = max(q_claim_reach(b, self.grader), q_transfer_reach(b, self.grader))
        if d > 0:
            for a in self._inner(b):
                best = max(best, self._q(b, a, d))
        return best

    def value(self, belief: Belief) -> float:
        """V at the root over the given root nonterminal set (call q_values/decide first to set it), depth = depth_k."""
        return self._v(belief, self.depth)

    def q_values(self, belief: Belief, candidates: Sequence[CandidateAction]) -> list[float]:
        self._set_root(candidates)
        out = []
        for c in candidates:
            if c.kind == DecisionKind.CLAIM_RESOLVED:
                out.append(q_claim_reach(belief, self.grader))
            elif c.kind == DecisionKind.TRANSFER:
                out.append(q_transfer_reach(belief, self.grader))
            else:
                out.append(self._q(belief, c, self.depth) if self.depth > 0 else NEG)
        return out

    score = q_values

    def v_root(self, belief: Belief, candidates: Sequence[CandidateAction]) -> float:
        """V*(root) = 0 if failed else max over all root Q values (the depth-k eq. 4 value)."""
        if belief.cost > self.B or belief.kappa:
            return 0.0
        return max(self.q_values(belief, candidates))

    def decide(self, belief: Belief, candidates: Sequence[CandidateAction]) -> int:
        q = self.q_values(belief, candidates)
        claim = next((i for i, c in enumerate(candidates) if c.kind == DecisionKind.CLAIM_RESOLVED), None)
        others = [i for i in range(len(candidates)) if i != claim]
        if claim is not None:
            gate = p_repaired(belief, self.grader) >= 1.0 - self.delta - TOL
            if gate and (not others or q[claim] >= max(q[i] for i in others) - TOL):
                return claim
            if not others:
                return claim
        mx = max(q[i] for i in others)
        tr = next((i for i in others if candidates[i].kind == DecisionKind.TRANSFER), None)
        if mx < self.eta and tr is not None:
            return tr
        return max(others, key=lambda i: (q[i], -i))


class ReachAvoidMargin(ReachAvoid):
    name = "reach_margin"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.p_star = float(self.cfg["reach_avoid"]["p_star"])

    def _l_claim(self, b: Belief) -> float:
        return min(p_repaired(b, self.grader) - self.p_star, (1.0 if b.v else 0.0) - 0.5)

    def _l_transfer(self, b: Belief) -> float:
        return q_transfer_reach(b, self.grader) - self.p_star

    def _l(self, b: Belief) -> float:
        return max(self._l_claim(b), self._l_transfer(b))

    def _g(self, b: Belief) -> float:
        return min(self.B - b.cost, 0.5 - (1.0 if b.kappa else 0.0))

    def _q(self, b: Belief, a: CandidateAction, d: int) -> float:
        return sum(p * self._v(b2, d - 1) for p, b2 in self.predictor.predict(b, a).values())

    def _v(self, b: Belief, d: int) -> float:
        return self._memoised(b, d, lambda: self._v_raw(b, d))

    def _v_raw(self, b: Belief, d: int) -> float:
        l, g, gm = self._l(b), self._g(b), self.gamma
        inner = l
        if d > 0:
            for a in self._inner(b):
                inner = max(inner, self._q(b, a, d))
        return (1.0 - gm) * min(l, g) + gm * min(g, inner)

    def value(self, belief: Belief) -> float:
        return self._v(belief, self.depth)

    def q_values(self, belief: Belief, candidates: Sequence[CandidateAction]) -> list[float]:
        self._set_root(candidates)
        g = self._g(belief)
        out = []
        for c in candidates:
            if c.kind == DecisionKind.CLAIM_RESOLVED:
                out.append(min(g, self._l_claim(belief)))
            elif c.kind == DecisionKind.TRANSFER:
                out.append(min(g, self._l_transfer(belief)))
            else:
                out.append(self._q(belief, c, self.depth) if self.depth > 0 else NEG)
        return out

    score = q_values

    def decide(self, belief: Belief, candidates: Sequence[CandidateAction]) -> int:
        q = self.q_values(belief, candidates)
        return max(range(len(q)), key=lambda i: (q[i], candidates[i].kind == DecisionKind.CLAIM_RESOLVED, -i))
