"""[telecom_boed] Myopic decision-utility gain, eq. 2' of docs/proposal.md §3.

DUG(a) = sum_o p(o) EU(beta'_o) - EU(beta) - lambda*C(a) - rho*R(a)   (beta'_o from predictor.predict; p(o)*EU(beta'_o) equals
the unnormalised max_r sum_h b p(o|h) U(r, T(h,a)) of eq. 2'). `dug_gross` is the same without the two cost terms.

MyopicDUG terminal-score mapping (so that argmax over ALL candidates reproduces the rule "take argmax DUG if any DUG > DUG_EPS,
else the better terminal"): nonterminal -> DUG(a); terminal t -> U_t - max(u_claim, u_transfer) (regret: 0 for the better
terminal, negative for the worse). Ties at exactly 0 resolve in favour of the terminal in `decide`; a raw argmax picks the
first index. Use `decide` as the authoritative rule. Terminal tie (claim == transfer): claim (transfer of a fixable fault
violates policy, so the non-escalating terminal is the default).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from telecom_boed.contracts import Belief, CandidateAction, DecisionKind, Grader, Predictor
from telecom_boed.decide.risk import risk
from telecom_boed.decide.utility import cost_of, eu, is_terminal, u_claim, u_transfer

DUG_EPS = 1e-12  # a DUG must exceed this to beat committing (guards float noise)


def dug_gross(belief: Belief, cand: CandidateAction, predictor: Predictor, grader: Grader, cfg: Mapping[str, Any]) -> float:
    base = eu(belief, grader, cfg)
    return sum(p * eu(b2, grader, cfg) for p, b2 in predictor.predict(belief, cand).values()) - base


def dug(belief: Belief, cand: CandidateAction, predictor: Predictor, grader: Grader, cfg: Mapping[str, Any]) -> float:
    c = cfg["cost"]
    r, _kappa = risk(cand, belief)
    return dug_gross(belief, cand, predictor, grader, cfg) - float(c["lambda"]) * cost_of(cand, predictor, cfg) - float(c["rho"]) * r


class MyopicDUG:
    name = "dug"

    def __init__(self, predictor: Predictor, grader: Grader, cfg: Mapping[str, Any]):
        self.predictor, self.grader, self.cfg = predictor, grader, cfg

    def _terminal_utils(self, belief: Belief) -> dict[DecisionKind, float]:
        return {DecisionKind.CLAIM_RESOLVED: u_claim(belief, self.grader, self.cfg),
                DecisionKind.TRANSFER: u_transfer(belief, self.grader, self.cfg)}

    def score(self, belief: Belief, candidates: Sequence[CandidateAction]) -> list[float]:
        tu = self._terminal_utils(belief)
        best = max(tu.values())
        return [tu[c.kind] - best if is_terminal(c) else dug(belief, c, self.predictor, self.grader, self.cfg) for c in candidates]

    def decide(self, belief: Belief, candidates: Sequence[CandidateAction]) -> int:
        scores = self.score(belief, candidates)
        non = [i for i, c in enumerate(candidates) if not is_terminal(c)]
        if non:
            i = max(non, key=lambda j: (scores[j], -j))  # first index wins ties
            if scores[i] > DUG_EPS:
                return i
        terms = [i for i, c in enumerate(candidates) if is_terminal(c)]
        return max(terms, key=lambda j: (scores[j], candidates[j].kind == DecisionKind.CLAIM_RESOLVED, -j))
