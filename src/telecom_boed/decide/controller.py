"""[telecom_boed] Decision controller of proposal §4 (owner: W2-S9): belief + pre-parsed candidates -> `Decision`.

Pure: no tau2 import; everything tau2-specific arrives through the Predictor / Grader objects. Candidates are PRE-PARSED
(`CandidateAction`s built from Qwen's N sampled messages, plus optional templates).

Scoring.  `scorer_name` in {dug, eig, heuristic, kstep, reach, reach_margin} selects decide/{dug,eig,heuristics,expectimax,reach_avoid}.
All candidates (terminals included) are scored; the pick is the scorer's OWN `decide()` applied to the very score vector that is
reported in `Decision.scores` ({candidate position: score}, positions index `candidates` first, then any templates appended in
override="full"). `HeuristicScorer` is stateful: the caller reports executed actions via `controller.scorer.mark_done(cand)`.

Override modes (`override=`):
- "none"          shadow mode: the scorer's choice is returned unchanged (`overridden=False`).
- "filter_commit" least-restrictive runtime filter (proposal §3, "In GRPO ... a least-restrictive runtime filter"). If the chosen
                  candidate is CLAIM_RESOLVED and the claim is unverified (belief.v is False OR P(repaired) < 1 - delta) the claim is
                  replaced by a verification request: the best-scoring candidate among the N that is a NON-MUTATING REQUEST_DEVICE and
                  can confirm repair (some predicted outcome sets v = True; this is the contract's v rule, so "primary group can
                  confirm" is decided by the Predictor itself); else a TEMPLATE verify candidate (candidate_idx=None, fixed polite
                  text, device action from VERIFY_BY_INTENT keyed by the belief's dominant intent). overridden=True. Never touches
                  TRANSFER or any non-claim choice. Two guards keep it least-restrictive: no override when one more turn would exceed
                  the budget B (the claim is then the best remaining move), and at most `max_overrides` consecutive overrides (then
                  the claim goes through, reason "verify_cap"), so an unresponsive user cannot trap the agent in a verify loop.
- "full"          "controller as policy" ablation. Templates built from the inner action set (default_inner_actions at the root, plus
                  the verify template) are appended to the candidates and scored with everything else; the scorer may pick ANY of
                  them (a template pick reports overridden=True because the chosen action is not one of the N). The filter_commit
                  rule is applied on top.

Anytime lookahead (`time_budget_ms`; default "cfg" = cfg latency.p95_scoring_ms, None = unlimited). kstep / reach / reach_margin run
iterative deepening k = 1..depth_k on ONE scorer instance (its memo is keyed by remaining depth, so level k reuses level k-1) and
return the deepest COMPLETED level. A cooperative deadline is enforced inside `predict` (a proxy raises before starting a new
predict once time is up), so a single predict call, which cannot be interrupted, can overshoot the budget (cold mms predict is
~0.5-2 s). Level k+1 is not started if the remaining time is below the last level's duration. If level 1 does not complete,
the controller falls back to MyopicDUG over the candidates whose one-step prediction had already finished (their predicts are
cached) plus the terminals, and says so in `reason`. dug / eig under a budget degrade the same way (fewer candidates scored).
`self.stats` (last decision) and `self.stats_log` hold {scorer, depth, depth_target, elapsed_ms, budget_ms, fallback, n_scored,
n_candidates, overran, override}.

Inner-node actions (`default_inner_actions`): root candidates' device actions + the intent-specific diagnostic / repair set of
INTENT_ACTIONS (tool names and `ensure`/args from tech_support_manual.md, lines cited there), de-duplicated, capped at cfg
lookahead.inner_m (rank "dug": top-m by myopic DUG at the node, the spec; rank "manual": first m in pool order, no predicts).

`prewarm(belief)`: runs predict once for every action of the default inner set, filling the filter's step / grade memos.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from telecom_boed.belief.user_model import is_mutating_tool
from telecom_boed.contracts import (Belief, CandidateAction, Decision, DecisionKind, DeviceAction, Grader, Predictor,
                                    _frozen_items)
from telecom_boed.decide.dug import MyopicDUG, dug
from telecom_boed.decide.eig import EIGScorer
from telecom_boed.decide.expectimax import KStepExpectimax
from telecom_boed.decide.heuristics import HeuristicScorer
from telecom_boed.decide.reach_avoid import ReachAvoid, ReachAvoidMargin
from telecom_boed.decide.utility import action_cost, is_terminal, p_repaired

SCORERS = ("dug", "eig", "heuristic", "kstep", "reach", "reach_margin")
LOOKAHEAD = ("kstep", "reach", "reach_margin")
OVERRIDES = ("none", "filter_commit", "full")
CFG_BUDGET = "cfg"  # time_budget_ms sentinel: read cfg latency.p95_scoring_ms


# --------------------------------------------------------------------------- verify templates
@dataclass(frozen=True)
class VerifySpec:
    tool: str
    text: str
    why: str  # manual citation (lines of data/tau2/domains/telecom/tech_support_manual.md)


VERIFY_BY_INTENT: dict[str, VerifySpec] = {
    "service_issue": VerifySpec(
        "check_status_bar",
        "Before we wrap up, could you please look at the status bar at the top of your phone and tell me whether it shows a "
        "signal-strength indicator and whether Airplane Mode is off?",
        "manual l.70-71: 'check_status_bar() can be used to check if the user is facing a service issue; if there is cellular "
        "service, the status bar will return a signal strength indicator'"),
    "mobile_data_issue": VerifySpec(
        "run_speed_test",
        "Before we wrap up, could you please run a speed test on your phone and tell me the result?",
        "manual l.122-126: 'run_speed_test() ... no connection when mobile data is unavailable; any speed below Excellent is slow'"),
    "mms_issue": VerifySpec(
        "can_send_mms",
        "Before we wrap up, could you please try sending a picture message (MMS) and tell me whether it goes through?",
        "manual l.181-182: 'can_send_mms() on the user's phone can be used to check if the user is facing an MMS issue'"),
}
GENERIC_VERIFY = VERIFY_BY_INTENT["service_issue"]  # unknown intent (toys): the status bar is the cheapest generic check


def _u(tool: str, ensure: Optional[tuple] = None, **args: Any) -> DeviceAction:
    return DeviceAction("user", tool, _frozen_items(args), ensure)


# Intent-specific diagnostic / repair set, in manual order (pool order = priority under rank="manual"). Line numbers: tech_support_manual.md.
INTENT_ACTIONS: dict[str, tuple[DeviceAction, ...]] = {
    "service_issue": (
        _u("check_status_bar"),                                  # l.70-71 diagnose; l.76 also shows airplane mode
        _u("toggle_airplane_mode", ("airplane_mode", False)),    # l.74-77 airplane mode must be OFF
        _u("check_sim_status"),                                  # l.83 'Missing' / 'Locked' / 'Active'
        _u("reseat_sim_card"),                                   # l.84 if 'Missing'
        _u("check_apn_settings"),                                # l.90 'Incorrect'
        _u("reset_apn_settings"),                                # l.90 then reboot_device (l.91)
        _u("reboot_device"),                                     # l.91 applies the APN reset
    ),
    "mobile_data_issue": (
        _u("run_speed_test"),                                    # l.123-126 diagnose
        _u("check_network_status"),                              # l.134 mobile data enabled? (also airplane, l.76)
        _u("toggle_airplane_mode", ("airplane_mode", False)),    # l.129-130 -> cellular-service section l.74-77
        _u("toggle_data", ("data", True)),                       # l.132-134 mobile data ON
        _u("toggle_roaming", ("roaming", True)),                 # l.136-138 roaming ON when abroad
        _u("check_data_restriction_status"),                     # l.144 'Data Saver mode is ON'
        _u("toggle_data_saver_mode", ("data_saver", False)),     # l.142-144 data saver OFF
        _u("check_vpn_status"),                                  # l.149 'VPN is ON and connected' + Poor
        _u("disconnect_vpn", ("vpn", False)),                    # l.146-149
        _u("check_network_mode_preference"),                     # l.162 '2G' / '3G'
        _u("set_network_mode_preference", mode="4g_5g_preferred"),  # l.159-162
    ),
    "mms_issue": (
        _u("can_send_mms"),                                      # l.181-182 diagnose
        _u("check_network_status"),                              # l.191-192 '2G' unsuitable; also data prerequisite l.170-172
        _u("set_network_mode_preference", mode="4g_5g_preferred"),  # l.192 mode with 3G/4G/5G
        _u("check_apn_settings"),                                # l.197 MMSC URL not set
        _u("reset_apn_settings"),                                # l.197-198
        _u("reboot_device"),                                     # l.198 applies the APN reset
        _u("check_wifi_calling_status"),                         # l.202 'Wi-Fi Calling is ON'
        _u("toggle_wifi_calling", ("wifi_calling", False)),      # l.200-202 OFF
        _u("toggle_data", ("data", True)),                       # l.170-172 prerequisite: mobile data
        _u("toggle_airplane_mode", ("airplane_mode", False)),    # l.170-172 prerequisite: cellular service
        _u("check_app_permissions"),                             # l.204-205 messaging app permissions
    ),
}
_TOOL_TEXT: dict[str, str] = {
    "check_status_bar": "could you please look at the status bar at the top of your screen and tell me what you see?",
    "check_network_status": "could you please check your phone's network status and tell me what it shows?",
    "run_speed_test": "could you please run a speed test on your phone and tell me the result?",
    "can_send_mms": "could you please try sending a picture message (MMS) and tell me whether it goes through?",
    "check_sim_status": "could you please check your SIM card status and tell me what it says?",
    "check_apn_settings": "could you please check your APN settings and tell me what they show?",
    "check_vpn_status": "could you please check whether a VPN is connected and tell me its status?",
    "check_data_restriction_status": "could you please check whether Data Saver mode is on?",
    "check_network_mode_preference": "could you please check your preferred network mode and tell me what it is?",
    "check_wifi_calling_status": "could you please check whether Wi-Fi Calling is on?",
    "check_app_permissions": "could you please check which permissions your messaging app has?",
    "reseat_sim_card": "could you please remove and re-insert your SIM card and tell me what the status bar shows?",
    "reset_apn_settings": "could you please reset your APN settings to default and tell me what happens?",
    "reboot_device": "could you please restart your phone and tell me what the status bar shows afterwards?",
    "disconnect_vpn": "could you please disconnect your VPN?",
}


def template_for(act: DeviceAction) -> CandidateAction:
    """A TEMPLATE candidate (candidate_idx=None) for one device action, with a fixed polite request."""
    t = act.tool
    if act.requestor == "assistant":
        return CandidateAction(DecisionKind.AGENT_TOOL, (act,), None, f"[backend call: {t}]")
    if t in _TOOL_TEXT:
        text = "Please follow this step: " + _TOOL_TEXT[t]
    elif t == "toggle_airplane_mode":
        text = "Could you please make sure Airplane Mode is turned off, then tell me what the status bar shows?"
    elif t == "toggle_data":
        text = "Could you please make sure Mobile Data is turned on, then tell me what you see?"
    elif t == "toggle_roaming":
        text = "Could you please make sure Data Roaming is turned on, then tell me what you see?"
    elif t == "toggle_data_saver_mode":
        text = "Could you please make sure Data Saver mode is turned off, then tell me what you see?"
    elif t == "toggle_wifi_calling":
        text = "Could you please make sure Wi-Fi Calling is turned off, then tell me what you see?"
    elif t == "set_network_mode_preference":
        text = f"Could you please set your preferred network mode to '{act.args_dict().get('mode', '4g_5g_preferred')}'?"
    else:
        text = f"Could you please run '{t}' on your phone and tell me what it shows?"
    return CandidateAction(DecisionKind.REQUEST_DEVICE, (act,), None, text)


def dominant_intent(belief: Belief) -> str:
    probs = dict(belief.intent_probs) if belief.intent_probs else {}
    if probs:
        return max(probs, key=lambda k: (probs[k], k))
    mass: dict = {}
    for p in belief.particles:
        mass[p.hyp_key[0]] = mass.get(p.hyp_key[0], 0.0) + p.weight
    return max(mass, key=lambda k: (mass[k], k)) if mass else ""


def verify_template(belief: Belief, verify_map: Mapping[str, VerifySpec] = VERIFY_BY_INTENT) -> CandidateAction:
    spec = verify_map.get(dominant_intent(belief), GENERIC_VERIFY)
    return CandidateAction(DecisionKind.REQUEST_DEVICE, (_u(spec.tool),), None, spec.text)


# --------------------------------------------------------------------------- inner-node actions
def default_inner_actions(belief: Belief, root_nonterminals: Sequence[CandidateAction], *, predictor: Predictor, grader: Grader,
                          cfg: Mapping[str, Any], rank: str = "dug") -> list[CandidateAction]:
    """Inner-node action set at `belief`: root candidates' device actions + INTENT_ACTIONS[dominant intent] (de-duplicated, roots
    first), capped at cfg lookahead.inner_m. rank="dug": top-m by myopic DUG at this node (the spec; costs one predict per pool
    member); rank="manual": first m in pool order (no predicts). Deterministic in `belief` (the lookahead memo relies on it)."""
    m = int(cfg["lookahead"]["inner_m"])
    pool: list[CandidateAction] = list(root_nonterminals)
    seen = {c.actions for c in pool}
    for act in INTENT_ACTIONS.get(dominant_intent(belief), ()):
        if (act,) not in seen:
            seen.add((act,))
            pool.append(template_for(act))
    if len(pool) <= m:
        return pool
    if rank == "manual":
        return pool[:m]
    order = sorted(range(len(pool)), key=lambda i: (-dug(belief, pool[i], predictor, grader, cfg), i))[:m]
    return [pool[i] for i in sorted(order)]


# --------------------------------------------------------------------------- deadline proxy
class _Timeout(Exception):
    """Raised inside predict() once the anytime deadline has passed."""


class _DeadlinePredictor:
    """Predictor proxy: refuses to START a predict after `deadline` (perf_counter seconds); records the actions predicted at
    the root belief and keeps the mean duration of its slow (cache-miss) calls. Everything else (e.g. toy `action_cost`) is forwarded."""

    def __init__(self, inner: Predictor):
        self.inner, self.deadline, self.root, self.done = inner, None, None, []
        self.n_slow, self.t_slow = 0, 0.0  # predicts that took > 1 ms (cache misses), kept across decisions

    def predict(self, belief: Belief, action: CandidateAction) -> dict:
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise _Timeout()
        t = time.perf_counter()
        out = self.inner.predict(belief, action)
        dt = time.perf_counter() - t
        if dt > 1e-3:
            self.n_slow, self.t_slow = self.n_slow + 1, self.t_slow + dt
        if belief is self.root:
            self.done.append(action)
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _DeadlineGrader:
    """Grader proxy sharing the predictor proxy's deadline: a grade miss runs the real tau2 env (ms each), and a utility sum
    over thousands of fresh particles must not be able to blow the budget unnoticed."""

    def __init__(self, inner: Grader, owner: _DeadlinePredictor):
        self.inner, self.owner = inner, owner

    def grade(self, state, hyp_key, transferred):
        d = self.owner.deadline
        if d is not None and time.perf_counter() > d:
            raise _Timeout()
        return self.inner.grade(state, hyp_key, transferred)

    def requires_transfer(self, hyp_key):
        return self.inner.requires_transfer(hyp_key)


def _decide_on(scorer: Any, fn: str, vals: list, belief: Belief, cands: Sequence[CandidateAction]) -> int:
    """scorer.decide() evaluated on a precomputed score vector (shadows score / q_values for the call), so the reported
    scores and the pick come from one evaluation."""
    setattr(scorer, fn, lambda *_a, **_k: vals)
    try:
        return scorer.decide(belief, cands)
    finally:
        delattr(scorer, fn)


# --------------------------------------------------------------------------- controller
class Controller:
    def __init__(self, scorer_name: str, predictor: Predictor, grader: Grader, cfg: Mapping[str, Any], *,
                 override: str = "filter_commit", time_budget_ms: Any = CFG_BUDGET,
                 inner_actions: Optional[Callable] = None, inner_rank: str = "dug", max_overrides: int = 2,
                 verify_map: Mapping[str, VerifySpec] = VERIFY_BY_INTENT):
        if scorer_name not in SCORERS:
            raise ValueError(f"scorer_name must be one of {SCORERS}, got {scorer_name!r}")
        if override not in OVERRIDES:
            raise ValueError(f"override must be one of {OVERRIDES}, got {override!r}")
        self.scorer_name, self.predictor, self.grader, self.cfg, self.override = scorer_name, predictor, grader, cfg, override
        self.time_budget_ms = cfg.get("latency", {}).get("p95_scoring_ms") if time_budget_ms == CFG_BUDGET else time_budget_ms
        self.max_overrides, self.verify_map = max_overrides, verify_map
        self.delta = float(cfg["reach_avoid"]["delta"])
        self.B = float(cfg["cost"]["budget_B"])
        self.proxy = _DeadlinePredictor(predictor)
        self.gproxy = _DeadlineGrader(grader, self.proxy)
        self.inner_rank = inner_rank
        self.inner_actions = inner_actions or (lambda b, root_nt: default_inner_actions(
            b, root_nt, predictor=self.proxy, grader=self.gproxy, cfg=cfg, rank=inner_rank))
        if inner_actions is None:
            # default_inner_actions depends on the root candidates only through their (kind, actions): key the lookahead
            # memo on that, so a later turn whose candidates differ only in wording / candidate_idx reuses subtrees (W2-P1).
            self.inner_actions.root_key = lambda nt: tuple((c.kind, c.actions) for c in nt)
        self.depth_target = int(cfg["lookahead"]["depth_k"])
        # P0-2 (boed_v2) knobs, all absent/off in older configs: cfg["perf"] = {bundle_cap_actions, predict_abort_checks, deadline_margin_ms}
        perf = cfg.get("perf") or {}
        self.bundle_cap = int(perf["bundle_cap_actions"]) if perf.get("bundle_cap_actions") else None
        self.abort_in_predict = bool(perf.get("predict_abort_checks", False))
        self.deadline_margin_ms = float(perf.get("deadline_margin_ms", 0.0) or 0.0)
        self.scorer = self._build(scorer_name)
        self.stats: dict = {}
        self.stats_log: list[dict] = []
        self._streak, self._root_intent = 0, ""

    def _build(self, name: str) -> Any:
        p, g, c = self.proxy, self.gproxy, self.cfg
        if name == "dug":
            return MyopicDUG(p, g, c)
        if name == "eig":
            return EIGScorer(p, g, c)
        if name == "heuristic":
            return HeuristicScorer(g, c)
        cls = {"kstep": KStepExpectimax, "reach": ReachAvoid, "reach_margin": ReachAvoidMargin}[name]
        return cls(p, g, c, inner_actions=self.inner_actions)

    def reset(self) -> None:
        """New episode: forget the consecutive-override counter and the scorer memo."""
        self._streak = 0
        if hasattr(self.scorer, "clear_memo"):
            self.scorer.clear_memo()

    # ---- helpers
    def _abort(self) -> None:
        """Installed as predictor.abort_check (perf.predict_abort_checks): polled by the filter between stepper waves."""
        d = self.proxy.deadline
        if d is not None and time.perf_counter() > d:
            raise _Timeout()

    def _capped(self, cands: list[CandidateAction]) -> list[CandidateAction]:
        """perf.bundle_cap_actions: a REQUEST_DEVICE candidate with more actions than the cap is SCORED on its first `cap` actions
        (bundles of 6 cost ~10 ms/particle, W2 replay). Same list object if nothing exceeds the cap. The decision still returns the
        original candidate (its full message is what gets emitted)."""
        cap = self.bundle_cap
        if cap is None or not any(c.kind == DecisionKind.REQUEST_DEVICE and len(c.actions) > cap for c in cands):
            return cands
        return [replace(c, actions=c.actions[:cap]) if c.kind == DecisionKind.REQUEST_DEVICE and len(c.actions) > cap else c for c in cands]

    def _score_fn(self) -> str:
        return "q_values" if self.scorer_name in LOOKAHEAD else "score"

    def _scores_all(self, belief: Belief, cands: list[CandidateAction], t0: float) -> tuple[int, dict, dict]:
        """(chosen position, {position: score}, info). Anytime: see module docstring."""
        sc, fn, name = self.scorer, self._score_fn(), self.scorer_name
        budget = self.time_budget_ms
        deadline = None if budget is None else t0 + max(budget - self.deadline_margin_ms, 0.0) / 1e3
        if self.abort_in_predict and hasattr(self.predictor, "abort_check"):
            self.predictor.abort_check = self._abort
        self.proxy.deadline, self.proxy.root, self.proxy.done = deadline, belief, []
        self._root_intent = dominant_intent(belief)
        info: dict = {"depth": None, "fallback": None, "n_scored": len(cands)}
        scorefn = getattr(sc, fn)
        try:
            if name in LOOKAHEAD:
                best = None
                for k in range(1, self.depth_target + 1):
                    if best is not None and deadline is not None and self._next_level_estimate(sc) > deadline - time.perf_counter():
                        info["skipped_depth"] = k  # projected cost of level k exceeds the time left: do not burn the budget on it
                        break
                    sc.depth = k
                    sc.reset_stats()
                    try:
                        vals = scorefn(belief, cands)
                    except _Timeout:
                        break
                    best = (k, vals)
                if best is not None:
                    k, vals = best
                    info["depth"] = k
                    self.proxy.deadline = None  # scoring done: the pick (cheap, cached) must not time out
                    return _decide_on(sc, fn, vals, belief, cands), {i: float(v) for i, v in enumerate(vals)}, info
                return self._fallback(belief, cands, info, "kstep/reach depth 1 exceeded budget")
            try:
                vals = scorefn(belief, cands)
            except _Timeout:
                return self._fallback(belief, cands, info, f"{name} exceeded budget")
            self.proxy.deadline = None
            return _decide_on(sc, fn, vals, belief, cands), {i: float(v) for i, v in enumerate(vals)}, info
        finally:
            self.proxy.deadline = None
            if self.abort_in_predict and hasattr(self.predictor, "abort_check"):
                self.predictor.abort_check = None
            if name in LOOKAHEAD:
                sc.depth = self.depth_target

    def _next_level_estimate(self, sc: Any) -> float:
        """Lower-bound-style projection (seconds) of the next deepening level: every belief evaluated at this level (memo
        misses) is expanded with the inner action set, one predict each at the mean predict time seen so far in this decision."""
        px = self.proxy
        if px.n_slow == 0:
            return 0.0
        mean = px.t_slow / px.n_slow  # cache hits (prewarm, repeated beliefs) are excluded: a NEW belief pays full price
        m = int(self.cfg["lookahead"]["inner_m"])
        if self.inner_rank == "dug":  # ranking by DUG predicts the whole pool at every node
            m = max(m, len(sc._root_nt) + len(INTENT_ACTIONS.get(self._root_intent, ())))
        return sc.misses * m * mean

    def _fallback(self, belief: Belief, cands: list[CandidateAction], info: dict, why: str) -> tuple[int, dict, dict]:
        """Budget exhausted before the scorer finished: MyopicDUG (EIG for eig) over the candidates whose one-step predict had
        completed (cached) plus the terminals. No new predict is started beyond what finished before the deadline... except that
        MyopicDUG re-reads those predicts, which are cached/cheap."""
        done = list(self.proxy.done)
        keep = [i for i, c in enumerate(cands) if is_terminal(c) or any(c is d for d in done)]
        sub = [cands[i] for i in keep]
        n_nt = sum(1 for c in cands if not is_terminal(c))
        info.update(fallback=f"MyopicDUG ({why}); scored {len(done)}/{n_nt} nonterminals", depth=0 if self.scorer_name in LOOKAHEAD else None,
                    n_scored=len(sub))
        if not sub:  # W3-F1: the agent offers no terminals and no predict finished in time -> nothing to compare: first candidate
            return 0, {}, info
        fb = EIGScorer(self.predictor, self.grader, self.cfg) if self.scorer_name == "eig" else MyopicDUG(self.predictor, self.grader, self.cfg)
        info["fallback"] = f"{fb.name} ({why}); scored {len(done)}/{n_nt} nonterminals"
        # W3-F1: the scorers' tie-break needs a terminal (dug.py:57 max() over none raised ValueError for agent-offered sets without
        # claim/transfer, the controller agent's case). Score with the missing terminals appended, then never pick a synthetic one.
        synth = [] if any(is_terminal(c) for c in sub) else [CandidateAction(DecisionKind.CLAIM_RESOLVED), CandidateAction(DecisionKind.TRANSFER)]
        full = sub + synth
        vals = fb.score(belief, full)
        j = _decide_on(fb, "score", vals, belief, full)
        if j >= len(sub):  # a synthetic terminal won: take the best offered candidate instead
            j = max(range(len(sub)), key=lambda i: (vals[i], -i))
        return keep[j], {keep[i]: float(vals[i]) for i in range(len(sub))}, info

    def _templates(self, belief: Belief, cands: list[CandidateAction]) -> list[CandidateAction]:
        """override="full": template candidates for the inner action set at the root + the verify template."""
        nt = [c for c in cands if not is_terminal(c)]
        have = {c.actions for c in cands}
        out: list[CandidateAction] = []
        for c in list(self.inner_actions(belief, nt)) + [verify_template(belief, self.verify_map)]:
            if c.actions not in have:
                have.add(c.actions)
                out.append(c if c.candidate_idx is None else template_for(c.actions[0]))
        return out

    def _non_mutating(self, c: CandidateAction) -> bool:
        if c.kind != DecisionKind.REQUEST_DEVICE:
            return False
        fn = getattr(self.predictor, "is_mutating", None)
        if fn is not None:
            return not fn(c)
        return bool(c.actions) and not any(is_mutating_tool(a.tool) for a in c.actions)

    def _confirms(self, belief: Belief, c: CandidateAction) -> bool:
        return any(b2.v for _p, b2 in self.predictor.predict(belief, c).values())

    def _filter_commit(self, belief: Belief, cands: list[CandidateAction], scores: dict, idx: int) -> tuple[int, CandidateAction, str]:
        """(idx, candidate, note). note == "" when untouched."""
        chosen = cands[idx]
        if chosen.kind != DecisionKind.CLAIM_RESOLVED:
            return idx, chosen, ""
        p_rep = p_repaired(belief, self.grader)
        if belief.v and p_rep >= 1.0 - self.delta:
            return idx, chosen, "claim verified"
        why = f"claim unverified (v={belief.v}, P(repaired)={p_rep:.3f}, need >= {1.0 - self.delta:.3f})"
        tpl = verify_template(belief, self.verify_map)
        if belief.cost + action_cost(tpl, self.cfg) > self.B:
            return idx, chosen, f"{why}; no override: a verify turn would exceed budget B={self.B:g}"
        if self._streak >= self.max_overrides:
            return idx, chosen, f"{why}; no override: verify_cap ({self.max_overrides} consecutive)"
        nm = [i for i, c in enumerate(cands) if self._non_mutating(c)]
        ver = [i for i in nm if self._confirms(belief, cands[i])]  # tier 1: some outcome sets v (contract v rule)
        tier = "confirming"
        if not ver:  # tier 2: the intent's own verifier tool (when no outcome can reach LR >= v_lr yet, e.g. P(repaired) ~ 0)
            ver = [i for i in nm if any(a.tool == tpl.actions[0].tool for a in cands[i].actions)]
            tier = "verifier-tool"
        if ver:
            j = max(ver, key=lambda i: (scores.get(i, float("-inf")), -i))
            return j, cands[j], f"{why}; replaced by {tier} verify candidate #{j}"
        return -1, tpl, f"{why}; replaced by TEMPLATE verify ({tpl.actions[0].tool})"

    # ---- API
    def decide(self, belief: Belief, candidates: Sequence[CandidateAction]) -> Decision:
        t0 = time.perf_counter()
        cands = list(candidates)
        if self.override == "full":
            cands = cands + self._templates(belief, cands)
        n_offered = len(cands)
        if self.scorer_name in ("dug", "eig"):  # W3-F1: their tie-break max()es over the terminals; an agent that offers none (the controller
            # agent) used to raise "max() iterable argument is empty". Score with the missing terminals appended; never pick a synthetic one.
            if not any(is_terminal(c) for c in cands):  # only when NONE is offered: otherwise the offered terminal keeps its old role
                cands = cands + [CandidateAction(DecisionKind.CLAIM_RESOLVED), CandidateAction(DecisionKind.TRANSFER)]
        orig = cands
        cands = self._capped(orig)  # scoring copies (identical list unless perf.bundle_cap_actions bites)
        idx, scores, info = self._scores_all(belief, cands, t0)
        if idx >= n_offered:
            idx = max(range(n_offered), key=lambda i: (scores.get(i, float("-inf")), -i))
        cands, scores = cands[:n_offered], {i: v for i, v in scores.items() if i < n_offered}
        chosen, notes = cands[idx], []
        chosen_idx = idx
        overridden = self.override == "full" and idx >= len(candidates)
        if overridden:
            notes.append(f"full: template #{idx} ({chosen.actions[0].tool if chosen.actions else chosen.kind.value}) chosen")
        if self.override in ("filter_commit", "full"):
            j, ch2, note = self._filter_commit(belief, cands, scores, idx)
            if note:
                notes.append("filter_commit: " + note)
            if ch2 is not chosen:
                chosen, overridden = ch2, True
                chosen_idx = j
                self._streak += 1
            else:
                self._streak = 0
        if self.bundle_cap is not None and chosen_idx >= 0:
            chosen = orig[chosen_idx]  # emit the full original candidate, not its scoring copy
        elapsed = (time.perf_counter() - t0) * 1e3
        budget = self.time_budget_ms
        self.stats = {"scorer": self.scorer_name, "depth": info["depth"], "depth_target": self.depth_target if self.scorer_name in LOOKAHEAD else None,
                      "elapsed_ms": elapsed, "budget_ms": budget, "fallback": info["fallback"], "n_scored": info["n_scored"],
                      "n_candidates": len(cands), "overran": budget is not None and elapsed > budget, "overridden": overridden,
                      "override": self.override}
        if self.bundle_cap is not None:
            self.stats["n_capped"] = sum(1 for a, b in zip(orig, cands) if a is not b)
        self.stats_log.append(self.stats)
        head = f"scorer={self.scorer_name}"
        if self.scorer_name in LOOKAHEAD:
            head += f" depth={info['depth']}/{self.depth_target}"
        head += f" elapsed={elapsed:.1f}ms budget={'none' if budget is None else f'{budget:g}ms'}"
        if info.get("skipped_depth"):
            head += f" (level {info['skipped_depth']} skipped: projected cost > time left)"
        if info["fallback"]:
            head += f" FALLBACK={info['fallback']}"
        return Decision(chosen, _frozen_items(scores), self.scorer_name, overridden, "; ".join([head] + notes))

    def prewarm(self, belief: Belief, root_nonterminals: Sequence[CandidateAction] = ()) -> dict:
        """Run predict once for every action of the default inner set at `belief` (fills the filter's step / grade memos; the
        per-belief predict cache only helps this exact belief object). Returns {n_actions, ms}."""
        t0 = time.perf_counter()
        pool = list(root_nonterminals)
        seen = {c.actions for c in pool}
        for act in INTENT_ACTIONS.get(dominant_intent(belief), ()):
            if (act,) not in seen:
                pool.append(template_for(act))
        for c in pool:
            self.predictor.predict(belief, c)
        return {"n_actions": len(pool), "ms": (time.perf_counter() - t0) * 1e3}
