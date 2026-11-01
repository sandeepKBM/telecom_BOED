"""[telecom_boed] W2-S9 tests: decision controller (decide/controller.py): scorer pick, filter_commit, anytime lookahead, latency.

Toy cases come from tests/fixtures/W2 (Opus' expected decisions). Real-support tests use only TRAIN-split hypotheses: the
"first observation" is a status-bar report built from the hypothesis' true initial state (as an honest user would give it) and the
controller itself never sees the task. `pytest tests/test_controller.py::test_latency_report` rewrites results/timings/W2-S9.json
(~1-2 min on a shared host; absolute numbers move with host load, loadavg is recorded).
"""
from __future__ import annotations

import copy
import json
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from loguru import logger

from telecom_boed import paths
from telecom_boed.belief.filter import ParticleFilter
from telecom_boed.belief.stepper import ParallelStepper
from telecom_boed.contracts import (Belief, CandidateAction, Decision, DecisionKind, DeviceAction, Observation, _frozen_items)
from telecom_boed.decide.controller import (INTENT_ACTIONS, VERIFY_BY_INTENT, Controller, VerifySpec, default_inner_actions,
                                            dominant_intent, template_for, verify_template)
from telecom_boed.decide.dug import MyopicDUG
from telecom_boed.decide.expectimax import KStepExpectimax
from telecom_boed.decide.reach_avoid import ReachAvoid
from telecom_boed.env import particles as P
from telecom_boed.env.featurize import featurize
from telecom_boed.env.hypotheses import load_support
from telecom_boed.testing.toy_pomdp import label, load_toy

logger.disable("tau2")
REPO = paths.PACKAGE_ROOT
FIX = Path(__file__).parent / "fixtures"
TIMINGS = REPO / "results" / "timings" / "W2-S9.json"
EXPECTED = yaml.safe_load((FIX / "W2" / "expected_decisions.yaml").read_text())
CASES = list(EXPECTED)
REQ, CLAIM = DecisionKind.REQUEST_DEVICE, DecisionKind.CLAIM_RESOLVED
SCORER_CLS = {"dug": MyopicDUG, "kstep": KStepExpectimax, "reach": ReachAvoid}
EXPECT_KEY = {"dug": "myopic", "kstep": "kstep", "reach": "reach"}


def U(tool: str, ensure=None, **args) -> DeviceAction:
    return DeviceAction("user", tool, _frozen_items(args), ensure)


def cand(act: DeviceAction, idx=None) -> CandidateAction:
    return CandidateAction(REQ, (act,), idx)


def own_decide(name, pred, grader, cfg, b, cands, **kw) -> int:
    sc = SCORER_CLS[name](pred, grader, cfg, **kw) if name != "dug" else MyopicDUG(pred, grader, cfg)
    return sc.decide(b, cands)


# --------------------------------------------------------------------------- toy POMDPs
@pytest.mark.parametrize("name", ["dug", "kstep", "reach"])
@pytest.mark.parametrize("case", CASES)
def test_toy_decisions_match_scorers(case, name):
    pred, grader, b0, cands, cfg = load_toy(case)
    ctl = Controller(name, pred, grader, cfg, override="none", time_budget_ms=None)
    d = ctl.decide(b0, cands)
    assert isinstance(d, Decision) and not d.overridden and d.scorer == name
    j = own_decide(name, pred, grader, cfg, b0, cands)
    assert d.chosen is cands[j]
    assert label(d.chosen) == EXPECTED[case]["initial"][EXPECT_KEY[name]]
    assert [k for k, _ in d.scores] == list(range(len(cands)))  # every candidate incl. terminals scored
    if name != "dug":
        assert ctl.stats["depth"] == cfg["lookahead"]["depth_k"]
    # a generous budget changes nothing and still reaches full depth on a toy
    ctl2 = Controller(name, pred, grader, cfg, override="none", time_budget_ms=60_000)
    d2 = ctl2.decide(b0, cands)
    assert d2.chosen is d.chosen and d2.scores == d.scores
    if name != "dug":
        assert ctl2.stats["depth"] == cfg["lookahead"]["depth_k"] and ctl2.stats["fallback"] is None


def _after_two_silent_repairs():
    """repair_verify after two R_air requests with a null report each: P(repaired) = 0.99, v = False (myopic claims here)."""
    pred, grader, b, cands, cfg = load_toy("repair_verify")
    r_air = next(c for c in cands if label(c) == "R_air")
    for _ in range(2):
        b = pred.predict(b, r_air)[("null",)][1]
    assert not b.v
    return pred, grader, b, cands, cfg


def test_filter_commit_replaces_unverified_claim():
    pred, grader, b, cands, cfg = _after_two_silent_repairs()
    none = Controller("dug", pred, grader, cfg, override="none", time_budget_ms=None).decide(b, cands)
    assert none.chosen.kind == CLAIM and not none.overridden
    fc = Controller("dug", pred, grader, cfg, override="filter_commit", time_budget_ms=None).decide(b, cands)
    assert fc.overridden and label(fc.chosen) == "K_status" and fc.chosen is next(c for c in cands if label(c) == "K_status")
    assert "filter_commit" in fc.reason and "unverified" in fc.reason and fc.scores == none.scores
    # reach already verifies on its own: the filter has nothing to replace
    r = Controller("reach", pred, grader, cfg, override="filter_commit", time_budget_ms=None).decide(b, cands)
    assert label(r.chosen) == "K_status" and not r.overridden
    # a verified belief (K_status saw 'repaired') is not touched
    k = next(c for c in cands if label(c) == "K_status")
    bv = pred.predict(b, k)[("repaired", True)][1]
    assert bv.v
    dv = Controller("dug", pred, grader, cfg, override="filter_commit", time_budget_ms=None).decide(bv, cands)
    assert dv.chosen.kind == CLAIM and not dv.overridden
    # full mode applies the same filter (K_status is already among the candidates, so no template is added)
    full = Controller("dug", pred, grader, cfg, override="full", time_budget_ms=None,
                      verify_map={"toy": VerifySpec("K_status", "x", "toy")}).decide(b, cands)
    assert label(full.chosen) == "K_status" and full.overridden


def test_filter_commit_template_when_no_verify_candidate():
    pred, grader, b, cands, cfg = _after_two_silent_repairs()
    no_verify = [c for c in cands if label(c) != "K_status"]
    vm = {"toy": VerifySpec("K_status", "Could you please check the status and tell me?", "toy")}
    ctl = Controller("dug", pred, grader, cfg, override="filter_commit", time_budget_ms=None, verify_map=vm)
    d = ctl.decide(b, no_verify)
    assert d.overridden and d.chosen.candidate_idx is None and d.chosen.kind == REQ
    assert d.chosen.actions == (U("K_status"),) and d.chosen.text == vm["toy"].text and "TEMPLATE" in d.reason
    # budget guard: one more turn would exceed B -> the claim goes through, reason says why
    tight = replace(b, cost=cfg["cost"]["budget_B"] - 0.5)
    dt = Controller("dug", pred, grader, cfg, override="filter_commit", time_budget_ms=None, verify_map=vm).decide(tight, no_verify)
    assert dt.chosen.kind == CLAIM and not dt.overridden and "exceed budget" in dt.reason
    # verify cap: after max_overrides consecutive overrides the claim passes
    cap = Controller("dug", pred, grader, cfg, override="filter_commit", time_budget_ms=None, verify_map=vm, max_overrides=2)
    outs = [cap.decide(b, no_verify) for _ in range(3)]
    assert [o.overridden for o in outs] == [True, True, False] and "verify_cap" in outs[2].reason
    cap.reset()
    assert cap.decide(b, no_verify).overridden
    # default map: dominant intent picks the template (manual-cited), unknown intent falls back to the status bar
    assert {k: v.tool for k, v in VERIFY_BY_INTENT.items()} == {
        "service_issue": "check_status_bar", "mobile_data_issue": "run_speed_test", "mms_issue": "can_send_mms"}
    assert all("manual l." in v.why for v in VERIFY_BY_INTENT.values())
    assert verify_template(b).actions[0].tool == "check_status_bar"
    b_mms = replace(b, intent_probs=_frozen_items({"service_issue": 0.2, "mms_issue": 0.8}))
    assert dominant_intent(b_mms) == "mms_issue" and verify_template(b_mms).actions[0].tool == "can_send_mms"


def test_never_overrides_transfer():
    pred, grader, b0, cands, cfg = load_toy("unfixable")
    for name in ("dug", "kstep", "reach"):
        for mode in ("filter_commit", "full"):
            d = Controller(name, pred, grader, cfg, override=mode, time_budget_ms=None,
                           verify_map={"toy": VerifySpec("K_sim", "x", "toy")}).decide(b0, cands)
            assert d.chosen.kind == DecisionKind.TRANSFER and not d.overridden, (name, mode)
    # a non-claim, non-transfer pick is never touched either (repair_verify prior: R_air)
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    d = Controller("reach", pred, grader, cfg, override="filter_commit", time_budget_ms=None).decide(b0, cands)
    assert label(d.chosen) == "R_air" and not d.overridden


def test_full_mode_may_pick_templates():
    """override="full": templates from the inner action set are scored with the candidates; a template pick is overridden=True."""
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    only_claim = [c for c in cands if c.kind in (CLAIM, DecisionKind.TRANSFER)]  # no device candidates among the N
    vm = {"toy": VerifySpec("K_status", "check please", "toy")}
    d = Controller("dug", pred, grader, cfg, override="full", time_budget_ms=None, verify_map=vm).decide(b0, only_claim)
    assert len(d.scores) == len(only_claim) + 1  # the verify template was appended and scored
    none = Controller("dug", pred, grader, cfg, override="none", time_budget_ms=None).decide(b0, only_claim)
    assert none.chosen.kind in (CLAIM, DecisionKind.TRANSFER) and not none.overridden
    assert d.chosen.kind in (CLAIM, DecisionKind.TRANSFER, REQ)
    if d.chosen.candidate_idx is None and d.chosen.kind == REQ:
        assert d.overridden


def test_anytime_fallback_and_budget_semantics():
    """A predictor that burns time: depth 1 cannot finish -> MyopicDUG over what finished (+ terminals), reported in reason/stats."""
    pred, grader, b0, cands, cfg = load_toy("repair_verify")

    class Slow:
        def __init__(self, inner, ms):
            self.inner, self.ms = inner, ms

        def predict(self, b, a):
            time.sleep(self.ms / 1e3)
            return self.inner.predict(b, a)

        def __getattr__(self, n):
            return getattr(self.inner, n)

    slow = Slow(pred, 60)
    ctl = Controller("reach", slow, grader, cfg, override="none", time_budget_ms=30)  # 1st predict starts in time (60 ms), the 2nd start is refused
    d = ctl.decide(b0, cands)
    assert ctl.stats["fallback"] and "dug" in ctl.stats["fallback"] and ctl.stats["depth"] == 0 and "FALLBACK" in d.reason
    assert d.chosen in cands and len(d.scores) == ctl.stats["n_scored"] < len(cands) + 1
    # a larger budget completes depth 1 but not the whole depth-3 search: achieved depth is reported, between 1 and depth_k
    ctl = Controller("reach", Slow(pred, 20), grader, cfg, override="none", time_budget_ms=100)
    ctl.decide(b0, cands)
    assert 1 <= ctl.stats["depth"] < cfg["lookahead"]["depth_k"] and ctl.stats["fallback"] is None
    # None = unlimited = full depth
    ctl = Controller("reach", Slow(pred, 1), grader, cfg, override="none", time_budget_ms=None)
    ctl.decide(b0, cands)
    assert ctl.stats["depth"] == cfg["lookahead"]["depth_k"] and ctl.stats["budget_ms"] is None
    # default budget = cfg latency.p95_scoring_ms (absent in toy cfgs -> None)
    cfg2 = {**cfg, "latency": {"p95_scoring_ms": 123}}
    assert Controller("dug", pred, grader, cfg2).time_budget_ms == 123 and Controller("dug", pred, grader, cfg).time_budget_ms is None
    assert Controller("dug", pred, grader, cfg2, time_budget_ms=None).time_budget_ms is None


def test_inner_actions_default_set():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    assert dominant_intent(b0) == "toy"
    root = [c for c in cands if c.kind == REQ]
    assert default_inner_actions(b0, root, predictor=pred, grader=grader, cfg=cfg) == root  # unknown intent: roots only
    # real-intent tables: tools exist in the manual-driven tables, ensure fields are the featurize toggle groups, cap by inner_m
    from telecom_boed.belief.user_model import ENSURE_READ
    for intent, acts in INTENT_ACTIONS.items():
        assert intent in VERIFY_BY_INTENT and (U(VERIFY_BY_INTENT[intent].tool) in acts)
        assert len({a for a in acts}) == len(acts)
        for a in acts:
            assert a.ensure is None or a.ensure[0] in ENSURE_READ
    b = replace(b0, intent_probs=_frozen_items({"mms_issue": 1.0}))
    cfg2 = {**cfg, "lookahead": {**cfg["lookahead"], "inner_m": 4}}
    out = default_inner_actions(b, root[:1], predictor=pred, grader=grader, cfg=cfg2, rank="manual")
    assert len(out) == 4 and out[0] is root[0] and out[1].actions == (U("can_send_mms"),)
    assert template_for(U("set_network_mode_preference", mode="4g_5g_preferred")).text.endswith("'4g_5g_preferred'?")
    assert template_for(DeviceAction("assistant", "get_details_by_id")).kind == DecisionKind.AGENT_TOOL


# --------------------------------------------------------------------------- real support
@pytest.fixture(scope="module")
def cfg_real():
    return yaml.safe_load((paths.CONFIG_DIR / "boed_default.yaml").read_text())


@pytest.fixture(scope="module")
def stepper():
    st = ParallelStepper(int(os.environ.get("W2S9_WORKERS", "8")))
    yield st
    st.close()


BAR = U("check_status_bar")
CANDS = {
    "mobile_data_issue": [U("toggle_airplane_mode", ("airplane_mode", False)), U("run_speed_test"), U("toggle_data", ("data", True))],
    "mms_issue": [U("can_send_mms"), U("check_network_status"), U("reset_apn_settings")],
}


def hand_candidates(intent: str) -> list[CandidateAction]:
    return [cand(a, i) for i, a in enumerate(CANDS[intent])] + [CandidateAction(CLAIM, (), 3)]


def train_hyps(intent: str, train_ids, n: int) -> list:
    """n hypotheses of `intent` that have a TRAIN task id, evenly spaced over the sorted support (deterministic)."""
    ts = set(train_ids)
    hs = sorted((h for h in load_support(intent) if any(t in ts for t in h.task_ids)), key=lambda h: (len(h.faults), sorted(h.faults)))
    step = max(1, len(hs) // n)
    return hs[::step][:n]


def first_obs_belief(flt: ParticleFilter, intent: str, hyp) -> Belief:
    """Prior on the intent support, then ONE scripted observation: the user reports the whole status bar of hyp's true state."""
    b0 = flt.init({intent: 1.0})
    f = featurize("check_status_bar", P.apply_calls(P.build_state(hyp), (BAR,))[1][0])
    f = {k: v for k, v in f.items() if k.startswith("status_bar.")}
    return flt.update(b0, cand(BAR), Observation((BAR,), "user_report", _frozen_items(f), frozenset(f)))


def test_real_support_controller_smoke(cfg_real, stepper, train_ids):
    intent = "mobile_data_issue"
    hyp = train_hyps(intent, train_ids, 1)[0]
    flt = ParticleFilter("deployable", cfg_real, stepper=stepper)
    grader = flt.grader
    b1 = first_obs_belief(flt, intent, hyp)
    assert len(b1.particles) >= 200 and b1.step == 1
    cands = hand_candidates(intent)
    seen = {}
    for name in ("dug", "reach"):
        for budget in (500, None):
            cfg = cfg_real if budget == 500 else {**cfg_real, "lookahead": {**cfg_real["lookahead"], "depth_k": 1}}  # None: depth 1 only
            ctl = Controller(name, flt, grader, cfg, override="none", time_budget_ms=budget)
            flt._pcache.clear(); flt._ecache.clear()
            d = ctl.decide(b1, cands)
            assert isinstance(d, Decision) and d.chosen in cands and d.scorer == name and not d.overridden
            assert len(d.scores) >= 1 and all(isinstance(v, float) for _k, v in d.scores)
            st = ctl.stats
            seen[(name, budget)] = {"depth": st["depth"], "fallback": st["fallback"], "ms": round(st["elapsed_ms"], 1), "pick": d.chosen.candidate_idx}
            if budget is None:
                j = own_decide(name, flt, grader, cfg, b1, cands)
                assert d.chosen is cands[j], (name, d.reason)
                assert st["fallback"] is None and (name == "dug" or st["depth"] == 1)
            elif name == "reach":
                assert st["depth"] in (0, 1, 2, 3) and (st["depth"] == 0) == bool(st["fallback"])
    print("real-support smoke", seen)


def test_real_support_filter_commit_templates(cfg_real, stepper, train_ids):
    """A claim forced on a real unverified belief is replaced: by the speed-test candidate if present, else by the manual template."""
    intent = "mobile_data_issue"
    flt = ParticleFilter("deployable", cfg_real, stepper=stepper)
    b1 = first_obs_belief(flt, intent, train_hyps(intent, train_ids, 1)[0])
    claim_only = [CandidateAction(CLAIM, (), 0)]
    ctl = Controller("dug", flt, flt.grader, cfg_real, override="filter_commit", time_budget_ms=None)
    d = ctl.decide(b1, claim_only)
    assert d.overridden and d.chosen.candidate_idx is None and d.chosen.actions == (U("run_speed_test"),)
    withv = [cand(U("run_speed_test"), 0), CandidateAction(CLAIM, (), 1)]
    d2 = Controller("dug", flt, flt.grader, cfg_real, override="filter_commit", time_budget_ms=None).decide(b1, withv)
    if d2.chosen.kind == CLAIM or d2.overridden:  # either dug verifies itself, or the filter picks the candidate (not a template)
        assert d2.chosen.kind == REQ and d2.chosen.actions == (U("run_speed_test"),)
        assert d2.chosen.candidate_idx in (0, None)


# --------------------------------------------------------------------------- latency report
def _pctl(xs, q=50):
    return float(statistics.median(xs)) if q == 50 else float(max(xs))


def test_latency_report(cfg_real, train_ids):
    """cold / warm decide() ms for dug, kstep, reach at budget 500 ms on the mobile_data (254) and mms (1,984) supports + prewarm."""
    workers = int(os.environ.get("W2S9_WORKERS", "8"))
    out = {"budget_ms": 500, "workers": workers, "cpu_count": os.cpu_count(), "loadavg_at_start": os.getloadavg(),
           "cfg": {"depth_k": cfg_real["lookahead"]["depth_k"], "inner_m": cfg_real["lookahead"]["inner_m"],
                   "eps_extra": cfg_real["user_model"]["eps_extra"]},
           "definitions": {
               "cold": "P.clear_caches + fresh filter + freshly forked worker pool + new Controller (first decision of an episode)",
               "warm": "same filter / pool after the cold run, per-call predict + expand caches and scorer memo cleared (median of 3)",
               "n_sampled": "warm_fraction rows: decisions on N train hypotheses, caches persisting between them (a long-lived process)"},
           "supports": {}}
    for intent in ("mobile_data_issue", "mms_issue"):
        hyps = train_hyps(intent, train_ids, 5)
        cands = hand_candidates(intent)
        rows: dict = {}
        with ParallelStepper(1) as st0:  # belief construction only (no timing)
            f0 = ParticleFilter("deployable", cfg_real, stepper=st0)
            b1s = [first_obs_belief(f0, intent, h) for h in hyps]
        n_support = len(load_support(intent))
        for name in ("dug", "kstep", "reach"):
            with ParallelStepper(workers) as st:
                P.clear_caches()
                flt = ParticleFilter("deployable", cfg_real, stepper=st)
                ctl = Controller(name, flt, flt.grader, cfg_real, override="filter_commit", time_budget_ms=500)
                t = time.perf_counter()
                d = ctl.decide(b1s[0], cands)
                cold = {"ms": round((time.perf_counter() - t) * 1e3, 1), "depth": ctl.stats["depth"], "fallback": ctl.stats["fallback"],
                        "n_scored": ctl.stats["n_scored"], "overran": ctl.stats["overran"]}
                warm = []
                for _ in range(3):
                    flt._pcache.clear(); flt._ecache.clear()
                    if hasattr(ctl.scorer, "clear_memo"):
                        ctl.scorer.clear_memo()
                    ctl.decide(b1s[0], cands)
                    warm.append({"ms": ctl.stats["elapsed_ms"], "depth": ctl.stats["depth"], "fallback": ctl.stats["fallback"]})
                # warm decisions over the sampled train hypotheses (persisting step / grade memos)
                frac = []
                for b in b1s:
                    flt._pcache.clear(); flt._ecache.clear()
                    if hasattr(ctl.scorer, "clear_memo"):
                        ctl.scorer.clear_memo()
                    ctl.decide(b, cands)
                    frac.append({"ms": round(ctl.stats["elapsed_ms"], 1), "depth": ctl.stats["depth"], "fallback": bool(ctl.stats["fallback"])})
            rows[name] = {
                "cold": cold,
                "warm_ms_p50": round(_pctl([w["ms"] for w in warm]), 1), "warm_depth": [w["depth"] for w in warm],
                "warm_fallback": [bool(w["fallback"]) for w in warm],
                "warm_sample": frac, "n_sample": len(frac),
                "frac_depth_ge1": (sum(1 for r in frac if (r["depth"] or 0) >= 1) / len(frac)) if name != "dug" else None,
                "frac_depth_ge2": (sum(1 for r in frac if (r["depth"] or 0) >= 2) / len(frac)) if name != "dug" else None,
                "frac_within_budget": sum(1 for r in frac if r["ms"] <= 500) / len(frac),
            }
            print(intent, name, rows[name]["cold"], rows[name]["warm_ms_p50"], rows[name]["warm_depth"], flush=True)
        # unbudgeted depth-1 and depth-2 cost (what the budget is up against), warm, manual-ranked inner set, reach
        with ParallelStepper(workers) as st:
            flt = ParticleFilter("deployable", cfg_real, stepper=st)
            unb = {}
            for k in ((1, 2) if intent == "mobile_data_issue" else (1,)):  # depth 2 on mms costs ~1 min: mobile_data only
                cfgk = {**cfg_real, "lookahead": {**cfg_real["lookahead"], "depth_k": k}}
                c = Controller("reach", flt, flt.grader, cfgk, override="none", time_budget_ms=None, inner_rank="manual")
                c.decide(b1s[0], cands)  # fills step / grade memos
                flt._pcache.clear(); flt._ecache.clear(); c.scorer.clear_memo()
                c.decide(b1s[0], cands)
                unb[f"reach_depth{k}_warm_ms"] = round(c.stats["elapsed_ms"], 1)
        # prewarm: cost, and what it saves on the first decision (reach @500 ms, cold otherwise)
        pw = {}
        for use in (False, True):
            with ParallelStepper(workers) as st:
                P.clear_caches()
                flt = ParticleFilter("deployable", cfg_real, stepper=st)
                ctl = Controller("reach", flt, flt.grader, cfg_real, override="none", time_budget_ms=500)
                key = "with_prewarm" if use else "without_prewarm"
                if use:
                    pw["prewarm"] = ctl.prewarm(b1s[0], [c for c in cands if c.kind == REQ])
                t = time.perf_counter()
                ctl.decide(b1s[0], cands)
                pw[key] = {"decide_ms": round((time.perf_counter() - t) * 1e3, 1), "depth": ctl.stats["depth"], "fallback": bool(ctl.stats["fallback"])}
        pw["prewarm"]["ms"] = round(pw["prewarm"]["ms"], 1)
        out["supports"][intent] = {"n_support": n_support, "n_particles_belief": len(b1s[0].particles), "results": rows,
                                   "unbudgeted": unb, "prewarm": pw}
        print(intent, unb, pw, flush=True)
    out["loadavg_at_end"] = os.getloadavg()
    TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    TIMINGS.write_text(json.dumps(out, indent=1))
    for intent, s in out["supports"].items():
        for name, r in s["results"].items():
            assert r["cold"]["ms"] > 0 and len(r["warm_depth"]) == 3
            # depth >= 1 or a declared fallback: the controller never returns silently without a completed level
            assert all((d or 0) >= 1 or fb or name == "dug" for d, fb in zip(r["warm_depth"], r["warm_fallback"]))


def test_fixture_inputs_are_pinned(train_ids):
    """tests/fixtures/W2-S9/inputs.json pins the train-only hypotheses used by the real-support tests (catalog / split drift is loud)."""
    fx = json.loads((FIX / "W2-S9" / "inputs.json").read_text())
    for intent in ("mobile_data_issue", "mms_issue"):
        got = [sorted(h.faults) for h in train_hyps(intent, train_ids, 5)]
        assert got == fx["train_hypotheses"][intent], intent
        assert all(any(t in set(train_ids) for t in h.task_ids) for h in train_hyps(intent, train_ids, 5))
