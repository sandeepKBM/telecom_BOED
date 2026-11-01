"""[telecom_boed] Leaf values for the reach-avoid lookahead (2026-11-01, P3 pre-test). Off by default: cfg reach_avoid.leaf = "none".

The depth-k search scores a leaf belief b by its terminal value max{Q_claim(b), Q_transfer(b)} only. A fixable multi-fault task needs
more repairs than k, so every nonterminal looks worthless and the controller transfers (CPU check on the l2x train-20: 20/20 transfer
on decision 1, perfect user). The two leaf values below estimate what is still reachable from b.

leaf = "rollout" (Bertsekas rollout with the manual-order base policy, decide/heuristics.MANUAL_ORDER):
  V_leaf(b) = max{ Q_claim(b), Q_transfer(b), R(b) },  R(b) = sum_i w_i [ succ_i - lambda * cost_i ] over n_roll particles i ~ b.
  Per particle (state s, hypothesis h) the base policy walks the root's state-changing candidates in manual order (MANUAL_ORDER rank,
  then backend agent tools, then the rest in candidate order), skipping the ones already done (this episode: `mark_done`; this search
  path: the actions above the leaf) unless their target still fails in s (see `_unmet`; then the step is redone), applies each one to s (an `ensure` already met changes nothing) and stops once grade(s, h) = 1.
  Every walked step is paid, as the manual-order agent pays it: cost_i = sum of C(a) / q(a) over the walked steps plus one verify turn;
  (none if b is verified and s is already fixed); q(a) = user-model compliance for the tool (rollout_compliance: true) or 1, so 1/q is the expected number of asks. succ_i = 1 if h is
  fixable, s reached grade 1 and b.cost + cost_i <= B; for a transfer hypothesis succ_i = grade(s_end, h, transferred=True) (the base
  policy ends with a transfer). Optimistic in two ways: it stops at the step that fixes s, and the user's reports are not modelled.
leaf = "fraction": R(b) = fraction_scale * sum_i w_i f_i, f_i = share of h's goal assertions that hold in s (transfer hypotheses: their
  transfer grade). A partial-progress heuristic, the simpler variant asked for.

Particles i ~ b: all particles, weighted, when the belief has <= leaf_exact_max (64) of them; otherwise deterministic systematic
resampling (offset 0.5) over the weights, leaf_n_roll draws (duplicates share the memo). Sampling noise at n_roll = 8 is ~0.02, the
size of one turn's lambda cost, which is why small beliefs are summed exactly.
Grades go through the scorer's grader (the controller's deadline proxy), so a leaf never runs past the decision budget unnoticed.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from telecom_boed.contracts import Belief, CandidateAction, DecisionKind, DeviceAction, Grader
from telecom_boed.decide.heuristics import MANUAL_ORDER, cand_tool
from telecom_boed.decide.utility import action_cost

LEAVES = ("none", "rollout", "fraction")


def sample_indices(w: Sequence[float], n: int) -> list[int]:
    """Deterministic systematic resampling: n indices into w (weights need not be normalised)."""
    w = np.asarray(w, dtype=float)
    tot = float(w.sum())
    if n <= 0 or tot <= 0.0 or len(w) == 0:
        return []
    cum = np.cumsum(w / tot)
    cum[-1] = 1.0
    return [int(i) for i in np.searchsorted(cum, (np.arange(n) + 0.5) / n, side="left")]


def base_order(root_nt: Sequence[CandidateAction], mutating: Callable[[str], bool]) -> list[CandidateAction]:
    """State-changing root candidates in manual order: MANUAL_ORDER rank, then AGENT_TOOL candidates (backend fixes and payment
    requests must come before the user's make_payment), then the rest in candidate order."""
    rank = {t: i for i, t in enumerate(MANUAL_ORDER)}
    keep = [(pos, c) for pos, c in enumerate(root_nt)
            if c.kind in (DecisionKind.REQUEST_DEVICE, DecisionKind.AGENT_TOOL) and c.actions
            and any(mutating(a.tool) for a in c.actions)]

    def key(pc: tuple) -> tuple:
        pos, c = pc
        t = cand_tool(c)
        if t in rank:
            return (0, rank[t], pos)
        return (1 if c.kind == DecisionKind.AGENT_TOOL else 2, 0, pos)
    return [c for _, c in sorted(keep, key=key)]


class LeafValue:
    """Leaf value R(b) for ReachAvoid (see module docstring). One instance per scorer; memo keyed by particle class."""

    def __init__(self, kind: str, grader: Grader, cfg: Mapping[str, Any], user_model: Any = None,
                 apply_fn: Optional[Callable] = None, feat_fn: Optional[Callable] = None,
                 ensure_read: Optional[Mapping] = None, mutating: Optional[Callable[[str], bool]] = None,
                 goal_frac: Optional[Callable] = None):
        if kind not in LEAVES or kind == "none":
            raise ValueError(f"leaf kind {kind!r} not in {LEAVES[1:]}")
        ra = cfg["reach_avoid"]
        self.kind, self.grader, self.cfg = kind, grader, cfg
        self.n_roll = int(ra.get("leaf_n_roll", 8))
        self.exact_max = int(ra.get("leaf_exact_max", 64))
        self.use_q = bool(ra.get("rollout_compliance", True))
        self.scale = float(ra.get("leaf_fraction_scale", 0.5))
        self.lam, self.B = float(cfg["cost"]["lambda"]), float(cfg["cost"]["budget_B"])
        self.verify_cost = float(cfg["cost"]["turn_cost"])
        if apply_fn is None or feat_fn is None or ensure_read is None or mutating is None:
            from telecom_boed.belief.user_model import ENSURE_READ, UserModel, is_mutating_tool
            from telecom_boed.env.featurize import featurize
            from telecom_boed.env.particles import apply_calls
            apply_fn, feat_fn, ensure_read, mutating = apply_fn or apply_calls, feat_fn or featurize, ensure_read or ENSURE_READ, \
                mutating or is_mutating_tool
            user_model = user_model or UserModel.from_cfg(cfg)
        self.apply, self.feat, self.ensure_read, self.mutating, self.um = apply_fn, feat_fn, ensure_read, mutating, user_model
        self._goal_frac = goal_frac
        self._memo: dict = {}
        self.done: set = set()  # (kind, actions) of candidates executed this episode
        self.path: list = []  # (kind, actions) of the actions on the current search path (ReachAvoid._q pushes/pops)
        self._seq: list[CandidateAction] = []
        self._seq_key: tuple = ()
        self.n_rollouts = 0

    # ---- root candidate set (the base policy's action list)
    def set_root(self, root_nt: Sequence[CandidateAction]) -> None:
        k = tuple(root_nt)
        if k != self._seq_key:
            self._seq_key, self._seq = k, base_order(root_nt, self.mutating)

    def clear(self) -> None:
        self._memo.clear()

    def reset(self) -> None:
        self._memo.clear()
        self.done.clear()
        self.path.clear()

    def mark_done(self, cand: CandidateAction) -> None:
        self.done.add((cand.kind, cand.actions))

    def path_sig(self) -> frozenset:
        return frozenset(self.path)

    # ---- helpers
    def _q(self, tool: str) -> float:
        if not self.use_q or self.um is None:
            return 1.0
        q = self.um.q_for(tool) if hasattr(self.um, "q_for") else self.um.q
        return max(float(q), 1e-3)

    def _unmet(self, s: Any, c: CandidateAction) -> bool:
        """A done step is redone when its target still does not hold in s: an `ensure` value that is not met, or (no ensure) a call
        that would still change s. Assumes the user's reply shows the setting, which holds for the device tools in the action set."""
        for a in c.actions:
            if a.ensure is not None and a.ensure[0] in self.ensure_read:
                rtool, field = self.ensure_read[a.ensure[0]]
                _, outs = self.apply(s, (DeviceAction(a.requestor, rtool),))
                if self.feat(rtool, outs[0]).get(field) != a.ensure[1]:
                    return True
            elif a.ensure is None and self.apply(s, (DeviceAction(a.requestor, a.tool, a.args),))[0].key != s.key:
                return True
        return False

    def _step(self, s: Any, c: CandidateAction) -> tuple[Any, float]:
        """Apply candidate c to state s -> (s', expected cost of the step: C(a)/q for a device request, C(a) for a backend call)."""
        calls = []
        for a in c.actions:
            if a.ensure is not None and a.ensure[0] in self.ensure_read:
                rtool, field = self.ensure_read[a.ensure[0]]
                _, outs = self.apply(s, (DeviceAction(a.requestor, rtool),))
                if self.feat(rtool, outs[0]).get(field) == a.ensure[1]:
                    continue
            calls.append(DeviceAction(a.requestor, a.tool, a.args))
        c0 = action_cost(c, self.cfg)
        if c.kind == DecisionKind.REQUEST_DEVICE:
            c0 /= self._q(c.actions[0].tool)
        if not calls:
            return s, c0
        s2, _ = self.apply(s, tuple(calls))
        return (s if s2.key == s.key else s2), c0

    def _rollout(self, s: Any, hk: Any, budget_left: float, skip: frozenset, verified: bool = False) -> float:
        req = self.grader.requires_transfer(hk)
        mk = ("r", s.key, hk, round(budget_left, 2), self._seq_key, skip, verified)
        r = self._memo.get(mk)
        if r is not None:
            return r
        self.n_rollouts += 1
        cost = 0.0
        if not req and self.grader.grade(s, hk, False) == 1.0:
            vc = 0.0 if verified else self.verify_cost
            r = (1.0 if vc <= budget_left else 0.0) - self.lam * vc
        else:
            ok = False
            for c in self._seq:
                if (c.kind, c.actions) in skip and not self._unmet(s, c):
                    continue
                s2, dc = self._step(s, c)
                cost += dc
                changed, s = s2 is not s, s2
                if changed and not req and self.grader.grade(s, hk, False) == 1.0:
                    ok = True
                    break
            if req:
                r = self.grader.grade(s, hk, True) - self.lam * cost
            else:
                tot = cost + self.verify_cost
                r = (1.0 if ok and tot <= budget_left else 0.0) - self.lam * tot
        self._memo[mk] = r
        return r

    def _fraction(self, s: Any, hk: Any) -> float:
        mk = ("f", s.key, hk)
        r = self._memo.get(mk)
        if r is None:
            if self.grader.requires_transfer(hk):
                r = self.grader.grade(s, hk, True)
            else:
                fn = self._goal_frac or _default_goal_frac
                r = self.scale * fn(s, hk)
            self._memo[mk] = r
        return r

    def value(self, b: Belief) -> float:
        parts = b.particles
        if not parts:
            return 0.0
        if len(parts) <= self.exact_max:  # exact weighted sum: no sampling noise at the lambda scale of one turn
            ix = list(range(len(parts)))
            w = [p.weight for p in parts]
        else:
            ix = sample_indices([p.weight for p in parts], self.n_roll)
            w = [1.0] * len(ix)
        tot = float(sum(w))
        if not ix or tot <= 0.0:
            return 0.0
        left = self.B - b.cost
        if self.kind == "rollout":
            skip = frozenset(self.done) | self.path_sig()
            vals = [self._rollout(parts[i].state, parts[i].hyp_key, left, skip, bool(b.v)) for i in ix]
        else:
            vals = [self._fraction(parts[i].state, parts[i].hyp_key) for i in ix]
        return float(sum(wi * v for wi, v in zip(w, vals)) / tot)


_FRAC_MEMO: dict = {}


def _default_goal_frac(state: Any, hk: Any) -> float:
    """Share of hypothesis hk's goal assertions that hold in `state` (real tau2 env; memoised by (state key, goal id))."""
    from tau2.data_model.tasks import EnvAssertion

    from telecom_boed.env import snapshot as S
    from telecom_boed.env.hypotheses import hypothesis_by_key
    hyp = hypothesis_by_key(hk)
    mk = (state.key, hyp.goal_id)
    r = _FRAC_MEMO.get(mk)
    if r is None:
        al = hyp.goal_assertions
        if not al:
            r = 0.0
        else:
            env = S.restore(state)
            ok = sum(bool(env.run_env_assertion(EnvAssertion(env_type=a.env_type, func_name=a.func_name, arguments=a.args_dict(),
                                                             assert_value=bool(a.assert_value)), raise_assertion_error=False))
                     for a in al)
            r = ok / len(al)
        if len(_FRAC_MEMO) > 200_000:
            _FRAC_MEMO.clear()
        _FRAC_MEMO[mk] = r
    return r
