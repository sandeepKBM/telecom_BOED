"""[telecom_boed] W2-S5 tests: exact Bayes filter, user model, predict/update consistency, parallel stepper.

Only TRAIN-split material is used as input: hypotheses are picked among those with a train task id, and the 10 episodes of
test_oracle_tracks_truth are train tasks (the test uses env.oracle_hidden to play the user; the filter never sees it).
The brute-force posteriors below are written independently of belief/filter.py: they re-run env.particles.apply_calls branch by
branch and apply the report model formulas of the filter's module docstring by hand.

Run `python tests/test_filter.py` to regenerate results/timings/W2-S5.json (full mms support; ~2-3 min on a shared host).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

if __name__ == "__main__":  # benchmark entry: bootstrap TAU2_DATA_DIR + tau2 patches exactly as pytest's conftest does, BEFORE tau2 is imported
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import conftest  # noqa: F401

import numpy as np
import pytest
import yaml
from loguru import logger

from telecom_boed import paths
from telecom_boed.belief.filter import ParticleFilter
from telecom_boed.belief.stepper import ParallelStepper
from telecom_boed.belief.user_model import UserModel, available_personas
from telecom_boed.contracts import (
    Belief, CandidateAction, DecisionKind, DeviceAction, Observation, Particle, _frozen_items)
from telecom_boed.env import oracle_hidden as O
from telecom_boed.env import particles as P
from telecom_boed.env.featurize import featurize
from telecom_boed.env.hypotheses import hypothesis_by_key, load_support
from telecom_boed.testing.leakage_scan import check_deployable, scan_module_source

logger.disable("tau2")  # tau2 logs every tool response at DEBUG; that I/O is not part of what these tests measure
REPO = paths.PACKAGE_ROOT
BELIEF_SRC = REPO / "src" / "telecom_boed" / "belief"
TIMINGS = REPO / "results" / "timings" / "W2-S5.json"
TOL = 1e-9
REQ, AGT = DecisionKind.REQUEST_DEVICE, DecisionKind.AGENT_TOOL
BAR = "status_bar"


def D(req: str, tool: str, **args) -> DeviceAction:
    return DeviceAction(req, tool, _frozen_items(args))


def U(tool: str, **args) -> DeviceAction:
    return D("user", tool, **args)


def request(*acts: DeviceAction) -> CandidateAction:
    return CandidateAction(REQ, tuple(acts))


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def train_set(splits) -> frozenset:
    return frozenset(splits["train"])


@pytest.fixture(scope="module")
def train_hyp(train_set):
    """faults (iterable) -> mobile_data_issue Hypothesis that has a TRAIN task id."""
    def find(faults, intent="mobile_data_issue"):
        h = next(h for h in load_support(intent) if h.faults == frozenset(faults))
        assert any(t in train_set for t in h.task_ids), f"{faults} has no train task"
        return h
    return find


@pytest.fixture(scope="module")
def small(train_hyp):
    """Five real mobile_data hypotheses (all with train tasks), a hand-made non-uniform prior, and their states."""
    hyps = [train_hyp(f) for f in (["airplane_mode_on", "data_mode_off"], ["data_mode_off", "data_usage_exceeded"],
                                   ["airplane_mode_on", "bad_network_preference"], ["bad_network_preference", "user_abroad_roaming_enabled_off"],
                                   ["data_saver_mode_on", "data_usage_exceeded"])]
    w = [0.30, 0.25, 0.20, 0.15, 0.10]
    return Belief(P.initial_particles(hyps, w), intent_probs=_frozen_items({"mobile_data_issue": 1.0})), hyps


def make_filter(cfg, mode="deployable", persona="default", **kw) -> ParticleFilter:
    return ParticleFilter(mode, cfg, persona, **kw)


def bar_fields(state, flt_feat=featurize) -> dict:
    out = P.apply_calls(state, (U("check_status_bar"),))[1][0]
    return flt_feat("check_status_bar", out)


def merge_key(p: Particle) -> tuple:
    return (p.state.key, hypothesis_by_key(p.hyp_key).goal_id)


def post_map(b: Belief) -> dict:
    out: dict = {}
    for p in b.particles:
        out[merge_key(p)] = out.get(merge_key(p), 0.0) + p.weight
    return out


# --------------------------------------------------------------------------- independent brute force
def brute_posterior(belief: Belief, act: DeviceAction, um: UserModel, obs: Observation, mode: str) -> dict:
    """{(state key, goal_id): posterior weight} for ONE requested user action, by hand: every particle x {nd, d, dd} x {no extra,
    extra reboot}, post-state by apply_calls, likelihood by the module-docstring formulas (status-bar group)."""
    branches = [(n, pr) for n, pr in (("nd", um.p_not_done), ("d", um.q), ("dd", um.eps_double)) if pr > 0]
    extras = [(e, pr) for e, pr in ((False, 1 - um.eps_extra), (True, um.eps_extra)) if pr > 0]
    items = []
    for p in belief.particles:
        goal = hypothesis_by_key(p.hyp_key).goal_id
        once = featurize(act.tool, P.apply_calls(p.state, (act,))[1][0])  # what a single run would show (claims_done)
        for br, pb in branches:
            for ex, pe in extras:
                calls = {"nd": [], "d": [act], "dd": [act, act]}[br]
                st, outs = p.state, []
                for c in calls:
                    st, o = P.apply_calls(st, (c,))
                    outs.append(o[0])
                shown = featurize(act.tool, outs[-1]) if outs else None
                if ex:
                    st, _ = P.apply_calls(st, (U("reboot_device"),))
                items.append({"key": (st.key, goal), "w": p.weight * pb * pe, "shown": shown, "once": once})
    if mode == "oracle":
        for it in items:
            it["lik"] = 1.0 if _frozen_items(it["shown"] or {}) == obs.fields else 0.0
    else:
        fields = sorted({f for it in items for d in (it["shown"], it["once"]) if d for f in d if f.startswith(BAR + ".")})
        tup = lambda d: tuple(d.get(f) for f in fields)
        space = sorted({tup(it["shown"]) for it in items if it["shown"] is not None} | {tup(it["once"]) for it in items}, key=repr)
        eps, t = um.eps_conf, len(space)
        r = tuple(dict(obs.fields)[f] for f in fields)
        for it in items:
            if it["shown"] is not None:
                it["lik"] = (1 - eps) * (r == tup(it["shown"])) + eps / t
            elif um.report_when_not_done == "claims_done":
                it["lik"] = (1 - eps) * (r == tup(it["once"])) + eps / t
            else:
                it["lik"] = 1.0 / t
    post: dict = {}
    for it in items:
        post[it["key"]] = post.get(it["key"], 0.0) + it["w"] * it["lik"]
    z = sum(post.values())
    return {k: v / z for k, v in post.items() if v / z >= 1e-12}


def report_obs(act: DeviceAction, tup_source: dict) -> Observation:
    """A user_report that mentions the whole status-bar group with the values of `tup_source`."""
    f = {k: v for k, v in tup_source.items() if k.startswith(BAR + ".")}
    return Observation((act,), "user_report", _frozen_items(f), frozenset(f))


def tool_obs(act: DeviceAction, content: str) -> Observation:
    f = featurize(act.tool, content)
    return Observation((act,), "tool", _frozen_items(f), frozenset(f))


def assert_posterior_equal(b: Belief, expect: dict, tol: float = TOL) -> None:
    got = post_map(b)
    assert set(got) == set(expect), (len(got), len(expect))
    for k, v in expect.items():
        assert got[k] == pytest.approx(v, abs=tol)


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("persona", ["default", "noncompliant", "confused"])
@pytest.mark.parametrize("mode", ["deployable", "oracle"])
def test_bruteforce_bayes_small_support(boed_cfg, small, mode, persona):
    b0, hyps = small
    flt = make_filter(boed_cfg, mode, persona)
    um = flt.um
    for act in (U("toggle_airplane_mode"), U("check_status_bar")):
        cand = request(act)
        for pick in (0, 2, 4):  # observation derived from three different particles' single-run outputs
            once_out = P.apply_calls(b0.particles[pick].state, (act,))[1][0]
            obs = tool_obs(act, once_out) if mode == "oracle" else report_obs(act, featurize(act.tool, once_out))
            got = flt.update(b0, cand, obs)
            assert not got.degenerate
            assert_posterior_equal(got, brute_posterior(b0, act, um, obs, mode))
            assert sum(p.weight for p in got.particles) == pytest.approx(1.0, abs=1e-12)


def test_noncompliance_keeps_unfixed_mass(boed_cfg, small):
    """After 'turn airplane mode off' and a report that shows it OFF, the states that still have airplane mode ON keep mass:
    the user may not have done it (1 - q - eps_double) and just said something."""
    b0, hyps = small
    act = U("toggle_airplane_mode")
    fixed = P.apply_calls(b0.particles[0].state, (act,))[0]  # H1 after the toggle: airplane OFF
    obs = report_obs(act, bar_fields(fixed))
    assert obs.fields and dict(obs.fields)["status_bar.airplane_mode"] is False
    masses = {}
    for persona in ("default", "noncompliant"):
        flt = make_filter(boed_cfg, "deployable", persona)
        b1 = flt.update(b0, request(act), obs)
        expect = brute_posterior(b0, act, flt.um, obs, "deployable")
        assert_posterior_equal(b1, expect)
        unfixed = sum(p.weight for p in b1.particles if bar_fields(p.state)["status_bar.airplane_mode"] is True)
        assert unfixed > 0.0
        assert unfixed == pytest.approx(sum(v for (sk, g), v in expect.items()
                                            if any(merge_key(p) == (sk, g) and bar_fields(p.state)["status_bar.airplane_mode"] is True
                                                   for p in b1.particles)), abs=TOL)
        masses[persona] = unfixed
    # claims_done (noncompliant) believes a 'fixed' report even less: the unfixed mass is larger than under 'uniform'
    assert masses["noncompliant"] > masses["default"] > 0.0


def test_weights_normalised_and_degenerate_flag(boed_cfg, small):
    b0, hyps = small
    flt = make_filter(boed_cfg, "deployable")
    act = U("check_status_bar")
    obs = report_obs(act, bar_fields(b0.particles[1].state))
    b1 = flt.update(b0, request(act), obs)
    ws = np.array([p.weight for p in b1.particles])
    assert not b1.degenerate and (ws > 0).all() and np.isfinite(ws).all() and ws.sum() == pytest.approx(1.0, abs=1e-12)
    assert b1.step == b0.step + 1 and b1.cost == pytest.approx(boed_cfg["cost"]["turn_cost"])
    # impossible backend observation (agent_tool is exact): zero total weight -> degenerate, prior particles, no NaN
    bad = D("assistant", "get_details_by_id", id="L1002")
    impossible = Observation((bad,), "agent_tool", _frozen_items({"line.status": "no-such-status"}), frozenset({"line.status"}))
    bd = flt.update(b0, CandidateAction(AGT, (bad,)), impossible)
    assert bd.degenerate and bd.particles == b0.particles
    assert all(math.isfinite(p.weight) for p in bd.particles) and sum(p.weight for p in bd.particles) == pytest.approx(1.0)
    assert bd.cost == pytest.approx(boed_cfg["cost"]["tool_cost"])
    # oracle mode: a 'tool' observation no branch can produce
    fo = make_filter(boed_cfg, "oracle")
    od = fo.update(b0, request(act), Observation((act,), "tool", _frozen_items({"status_bar.battery": -1}), frozenset({"status_bar.battery"})))
    assert od.degenerate and od.particles == b0.particles
    # a deployable report whose value is outside the label space is uninformative, not degenerate
    weird = Observation((act,), "user_report", _frozen_items({"status_bar.battery": 7}), frozenset({"status_bar.battery"}))
    bw = flt.update(b0, request(act), weird)
    assert not bw.degenerate and post_map(bw) == pytest.approx(post_map(flt.update(b0, request(act), Observation((act,), "user_report"))))


def _predict_cases(small):
    b0, hyps = small
    return [
        request(U("check_status_bar")),
        request(U("toggle_airplane_mode")),
        request(DeviceAction("user", "toggle_airplane_mode", (), ("airplane_mode", False))),
        request(U("run_speed_test")),
        request(U("toggle_data"), U("check_status_bar")),
        CandidateAction(AGT, (D("assistant", "get_details_by_id", id="L1002"),)),
    ]


@pytest.mark.parametrize("mode", ["deployable", "oracle"])
def test_predict_total_probability(boed_cfg, small, mode):
    b0, _ = small
    flt = make_filter(boed_cfg, mode, "confused")
    for cand in _predict_cases(small):
        out = flt.predict(b0, cand)
        assert out and sum(p for p, _ in out.values()) == pytest.approx(1.0, abs=1e-9), cand
        for p, b in out.values():
            assert 0.0 < p <= 1.0 + 1e-12 and sum(q.weight for q in b.particles) == pytest.approx(1.0, abs=1e-9)
    for cand in (CandidateAction(DecisionKind.ASK), CandidateAction(DecisionKind.CLAIM_RESOLVED), CandidateAction(DecisionKind.TRANSFER)):
        out = flt.predict(b0, cand)
        assert len(out) == 1 and next(iter(out.values()))[0] == 1.0


@pytest.mark.parametrize("persona", ["default", "noncompliant"])
@pytest.mark.parametrize("mode", ["deployable", "oracle"])
def test_predict_update_consistency(boed_cfg, small, mode, persona):
    """update(b, c, obs_for_outcome_o) == predict(b, c)[o][1] for every outcome, including v / kappa / cost / step."""
    b0, _ = small
    flt = make_filter(boed_cfg, mode, persona)
    n_checked = 0
    for cand in _predict_cases(small):
        out = flt.predict(b0, cand)
        for key, (p, bp) in out.items():
            source, actions, fields, mentioned = key
            obs = Observation(actions, source, fields, mentioned)
            assert obs.key() == key
            bu = flt.update(b0, cand, obs)
            assert (bu.v, bu.kappa, bu.transferred, bu.cost, bu.step, bu.degenerate) == (bp.v, bp.kappa, bp.transferred, bp.cost, bp.step, bp.degenerate)
            gu, gp = post_map(bu), post_map(bp)
            assert set(gu) == set(gp)
            assert all(gu[k] == pytest.approx(gp[k], abs=1e-12) for k in gp)
            n_checked += 1
    assert n_checked >= 12
    # terminals: update ignores the observation and equals predict
    for cand in (CandidateAction(DecisionKind.CLAIM_RESOLVED), CandidateAction(DecisionKind.TRANSFER), CandidateAction(DecisionKind.ASK)):
        (key, (p, bp)), = flt.predict(b0, cand).items()
        bu = flt.update(b0, cand, Observation((), "user_report"))
        assert bu.particles == bp.particles and (bu.v, bu.kappa, bu.cost, bu.transferred, bu.step) == (bp.v, bp.kappa, bp.cost, bp.transferred, bp.step)


def test_transfer_kappa_and_terminals(boed_cfg, train_hyp):
    """kappa on transfer iff some mass is on a hypothesis that does not require transfer (contracts: transferring a fixable fault)."""
    flt = make_filter(boed_cfg, "deployable")
    fixable = train_hyp(["airplane_mode_on", "data_mode_off"])
    needs = next(h for h in load_support("service_issue") if h.requires_transfer)
    mixed = Belief(P.initial_particles([fixable, needs], [0.5, 0.5]))
    only = Belief(P.initial_particles([needs], [1.0]))
    tr = CandidateAction(DecisionKind.TRANSFER)
    (_, b_mixed), = flt.predict(mixed, tr).values()
    (_, b_only), = flt.predict(only, tr).values()
    assert b_mixed.transferred and b_mixed.kappa and b_only.transferred and not b_only.kappa
    cl = CandidateAction(DecisionKind.CLAIM_RESOLVED)
    assert flt.predict(mixed, cl)[("claim",)][1] is mixed
    ask = flt.predict(mixed, CandidateAction(DecisionKind.ASK))[("ask",)][1]
    assert ask.cost == pytest.approx(boed_cfg["cost"]["turn_cost"]) and ask.particles == mixed.particles


def test_agent_tool_cost_kappa_and_exactness(boed_cfg):
    """Backend tools run exactly once: cost += tool_cost, kappa from decide/risk (resume_line on an expired contract), outcomes exact."""
    h = next(h for h in load_support("service_issue") if "contract_end_suspension" in h.faults and "lock_sim_card_pin" in h.faults)
    other = next(h for h in load_support("service_issue") if h.faults == frozenset({"airplane_mode_on", "unseat_sim_card"}))
    b0 = Belief(P.initial_particles([h, other], [0.5, 0.5]))
    flt = make_filter(boed_cfg, "deployable")
    resume = CandidateAction(AGT, (D("assistant", "resume_line", customer_id="C1001", line_id="L1002"),))
    out = flt.predict(b0, resume)
    assert sum(p for p, _ in out.values()) == pytest.approx(1.0)
    for p, b in out.values():
        assert b.kappa is True and b.cost == pytest.approx(boed_cfg["cost"]["tool_cost"]) and b.step == 1
    read = CandidateAction(AGT, (D("assistant", "get_details_by_id", id="L1002"),))
    out = flt.predict(b0, read)
    assert len(out) == 2 and all(k[0] == "agent_tool" and not b.kappa for k, (_, b) in out.items())
    for key, (p, b) in out.items():  # each outcome is the exact featurized output of one hypothesis' state
        assert p == pytest.approx(0.5) and len(b.particles) == 1
    # the oracle filter accepts the same observation under source="tool"
    key = next(iter(out))
    obs = Observation(key[1], "tool", key[2], key[3])
    assert make_filter(boed_cfg, "oracle").update(b0, read, obs).particles[0].state.key == out[key][1].particles[0].state.key
    with pytest.raises(ValueError):
        flt.update(b0, read, obs)  # oracle observations are rejected in deployable mode


def test_ensure_semantics(boed_cfg, small):
    """ensure=(airplane_mode, False): a particle already OFF compiles to no state change; one that is ON runs the toggle."""
    b0, hyps = small
    ens = DeviceAction("user", "toggle_airplane_mode", (), ("airplane_mode", False))
    flt = make_filter(boed_cfg, "oracle")
    out = flt.predict(b0, request(ens))
    on_keys = {p.state.key for p in b0.particles if bar_fields(p.state)["status_bar.airplane_mode"]}
    off_keys = {p.state.key for p in b0.particles} - on_keys
    assert on_keys and off_keys
    # oracle outcome with fields==() = tool never ran: includes every OFF particle (no-op) and every ON particle's not_done
    nothing = Observation((ens,), "tool", (), frozenset())
    b_nothing = flt.update(b0, request(ens), nothing)
    keys_after = {p.state.key for p in b_nothing.particles}
    assert off_keys <= keys_after
    # an observation of a real toggle result can only come from ON particles
    on_state = next(p.state for p in b0.particles if p.state.key in on_keys)
    obs = tool_obs(ens, P.apply_calls(on_state, (U("toggle_airplane_mode"),))[1][0])
    b_ran = flt.update(b0, request(ens), obs)
    # by hand: no particle starting OFF can have produced the toggle's fields (the toggle never ran for it)
    assert_posterior_equal(b_ran, brute_ens(b0, ens, obs))
    assert_posterior_equal(b_nothing, brute_ens(b0, ens, Observation((ens,), "tool", (), frozenset())))


def brute_ens(belief: Belief, act: DeviceAction, obs: Observation) -> dict:
    """Oracle brute force for an ensure=(airplane_mode, False) toggle at the default persona."""
    um = UserModel.from_cfg(yaml.safe_load((paths.CONFIG_DIR / "boed_default.yaml").read_text()))
    post: dict = {}
    for p in belief.particles:
        goal = hypothesis_by_key(p.hyp_key).goal_id
        off = bar_fields(p.state)["status_bar.airplane_mode"] is False
        for br, pb in (("nd", um.p_not_done), ("d", um.q), ("dd", um.eps_double)):
            for ex, pe in ((False, 1 - um.eps_extra), (True, um.eps_extra)):
                calls = [] if (br == "nd" or off) else [U("toggle_airplane_mode")] * (2 if br == "dd" else 1)
                st, outs = p.state, []
                for c in calls:
                    st, o = P.apply_calls(st, (c,))
                    outs.append(o[0])
                if ex:
                    st, _ = P.apply_calls(st, (U("reboot_device"),))
                f = _frozen_items(featurize("toggle_airplane_mode", outs[-1])) if outs else ()
                if f == obs.fields:
                    k = (st.key, goal)
                    post[k] = post.get(k, 0.0) + p.weight * pb * pe
    z = sum(post.values())
    return {k: v / z for k, v in post.items() if v / z >= 1e-12}


def _v_belief(train_hyp):
    """A belief over states along H1's repair path: fixed (grade 1), only-airplane-fixed, and the untouched start."""
    h1 = train_hyp(["airplane_mode_on", "data_mode_off"])
    s0 = P.build_state(h1)
    s1, _ = P.apply_calls(s0, (U("toggle_airplane_mode"),))
    s2, _ = P.apply_calls(s1, (U("toggle_data"),))
    assert P.grade(s2, h1, False) == 1.0 and P.grade(s1, h1, False) == 0.0
    parts = (Particle(h1.key, s2, 0.4), Particle(h1.key, s1, 0.3), Particle(h1.key, s0, 0.3))
    return Belief(parts), (s0, s1, s2)


def test_v_rule(boed_cfg, train_hyp):
    b0, (s0, s1, s2) = _v_belief(train_hyp)
    flt = make_filter(boed_cfg, "deployable")
    chk = U("check_status_bar")
    null = Observation((chk,), "user_report")
    confirm = report_obs(chk, bar_fields(s2))     # shows the repaired phone
    deny = report_obs(chk, bar_fields(s1))        # shows airplane OFF but data still disabled
    # (1) a confirming check sets v = 1 (non-mutating); LR computed by hand
    b = flt.update(b0, request(chk), confirm)
    assert b.v is True
    um = flt.um
    tup = lambda s: tuple(sorted((k, v) for k, v in bar_fields(s).items()))
    space = sorted({tup(s) for s in (s0, s1, s2)})
    t, eps = len(space), um.eps_conf
    like = lambda s: (um.q + um.eps_double) * ((1 - eps) * (tup(s) == tup(s2)) + eps / t) + um.p_not_done / t
    lr = like(s2) / ((0.3 * like(s1) + 0.3 * like(s0)) / 0.6)
    assert lr >= boed_cfg["reach_avoid"]["v_lr"]
    # raising the threshold above the hand LR must keep v = 0 (the rule is a threshold on the likelihood ratio)
    hi = {**boed_cfg, "reach_avoid": {**boed_cfg["reach_avoid"], "v_lr": lr * 1.01}}
    assert make_filter(hi, "deployable").update(b0, request(chk), confirm).v is False
    # (2) a null report leaves v; a contradicting report does not set it
    assert flt.update(b0, request(chk), null).v is False
    assert flt.update(b0, request(chk), deny).v is False
    # (3) null report keeps an already-set v; a mutating request resets it before its observation is applied
    bv = Belief(b.particles, v=True)
    assert flt.update(bv, request(chk), null).v is True
    tog = U("toggle_airplane_mode")
    assert flt.update(bv, request(tog), Observation((tog,), "user_report")).v is False
    assert any(bp.v is False for p, bp in flt.predict(bv, request(tog)).values())
    # contract v rule (clarified 2026-10-16): a confirming/null check keeps an already-set v, a strongly contradicting
    # one (LR <= 1/v_lr) clears it — same as the toy POMDP's ("repaired", False)
    assert any(bp.v is True for p, bp in flt.predict(bv, request(chk)).values())
    assert flt.update(bv, request(chk), confirm).v is True
    assert flt.update(bv, request(chk), deny).v is False
    # (4) after a reset, a later confirming (non-mutating) check sets it again
    b_reset = flt.update(bv, request(tog), Observation((tog,), "user_report"))
    assert b_reset.v is False
    assert flt.update(b_reset, request(chk), confirm).v is True
    # (5) agent-side tool results are always seen: get_details_by_id(L1002) is identical in every particle here (no LR) -> v unchanged
    ag = D("assistant", "get_details_by_id", id="L1002")
    ((p_ag, b_ag),) = flt.predict(b0, CandidateAction(AGT, (ag,))).values()
    assert p_ag == pytest.approx(1.0) and b_ag.v is False


@pytest.fixture(scope="module")
def stepper8():
    with ParallelStepper(8) as st:
        yield st


def _train_episode_ids(train_ids) -> list:
    """10 deterministic TRAIN tasks across the three intents (catalog order within the train split)."""
    by_intent: dict = {}
    for i in train_ids:
        by_intent.setdefault(i.split("]")[0][1:], []).append(i)
    pick = []
    for intent, n in (("service_issue", 3), ("mobile_data_issue", 4), ("mms_issue", 3)):
        ids = by_intent[intent]
        pick += [ids[1 + k * (len(ids) // n)] for k in range(n)]
    assert len(set(pick)) == 10
    return pick


def test_fixture_inputs_are_pinned(train_ids, small):
    """tests/fixtures/W2-S5/inputs.json records the train-only inputs of this file so a catalog or split change is loud."""
    fx = json.loads((Path(__file__).parent / "fixtures" / "W2-S5" / "inputs.json").read_text())
    assert fx["oracle_episodes"] == _train_episode_ids(train_ids)
    _, hyps = small
    assert fx["small_support_faults"] == [sorted(h.faults) for h in hyps] and fx["small_prior"] == [0.30, 0.25, 0.20, 0.15, 0.10]


def test_oracle_tracks_truth(boed_cfg, catalog, train_ids, train_set, stepper8):
    """Scripted reporter in the test (reads the true task via env.oracle_hidden) feeds EXACT tool content along the gold
    actions. With likelihoods in {0, 1} the truth's merge class is never lost and its weight can fall only by the execution-branch factor."""
    by_id = {t["id"]: t for t in catalog}
    flt = make_filter(boed_cfg, "oracle", stepper=stepper8)
    rows = []
    for tid in _train_episode_ids(train_ids):
        assert tid in train_set
        task_row = by_id[tid]
        key = O.true_key(task_row)
        hyp = hypothesis_by_key(key)
        b = flt.init({key[0]: 1.0})
        truth = P.build_state(hyp)
        w_prev = max(p.weight for p in b.particles if merge_key(p) == (truth.key, hyp.goal_id))
        w_first = w_prev
        for a in O.gold_actions(task_row):
            if a.tool == "transfer_to_human_agents":
                b = flt.update(b, CandidateAction(DecisionKind.TRANSFER), Observation((a,), "agent_tool"))
                assert b.transferred
                continue
            truth, outs = P.apply_calls(truth, (a,))
            f = featurize(a.tool, outs[0])
            if a.requestor == "user":
                b = flt.update(b, request(a), Observation((a,), "tool", _frozen_items(f), frozenset(f)))
            else:
                b = flt.update(b, CandidateAction(AGT, (a,)), Observation((a,), "agent_tool", _frozen_items(f), frozenset(f)))
            assert not b.degenerate, (tid, a.tool)
            w = sum(p.weight for p in b.particles if merge_key(p) == (truth.key, hyp.goal_id))
            assert w > 0.0, (tid, a.tool)
            # Bayes bound: the truth item survives with the branch probability of the observed execution, and the evidence
            # (all weights summed) is at most 1, so the posterior is at least prior * pi_branch. Exact backend steps have pi = 1.
            pi = flt.um.q * (1 - flt.um.eps_extra) if a.requestor == "user" else 1.0
            assert w >= w_prev * pi - 1e-12, (tid, a.tool, w_prev, w)
            w_prev = w
        rows.append((tid, len(b.particles), w_first, w_prev))
        assert sum(p.weight for p in b.particles) == pytest.approx(1.0, abs=1e-9)
        if b.transferred is False:
            assert P.grade(truth, hyp, False) == 1.0 or O.gold_actions(task_row) == ()  # gold actions repair the truth
    print("\n[W2-S5] oracle tracking (task, final particles, truth mass: prior -> final):")
    for r in rows:
        print(f"  {r[0][:60]:60s} n={r[1]:5d}  {r[2]:.2e} -> {r[3]:.4f}")


def _same(r1, r2) -> bool:
    (s1, o1), (s2, o2) = r1, r2
    return (o1 == o2 and s1.key == s2.key and s1.snapshot_id == s2.snapshot_id and s1.parent_id == s2.parent_id and s1.step == s2.step
            and s1.agent_db == s2.agent_db and s1.user_db == s2.user_db and s1.vpn_default == s2.vpn_default)


def test_parallel_matches_serial(boed_cfg, train_hyp):
    hyps = load_support("mobile_data_issue")
    parts = P.initial_particles(hyps)
    states = [p.state for p in parts]
    for calls in ((U("check_status_bar"),), (U("toggle_airplane_mode"),), (U("toggle_data"), U("reboot_device"))):
        serial = [P.apply_calls(s, calls) for s in states]
        for persistent in (True, False):
            with ParallelStepper(4, persistent=persistent) as st:
                par = st.step_many(states, calls)
            assert len(par) == len(serial)
            assert all(_same(a, b) for a, b in zip(serial, par))
    with ParallelStepper(4) as st:  # outputs_only skips snapshots
        res = st.step_many(states, (U("check_status_bar"),), outputs_only=True)
    assert all(s is None for s, _ in res) and [o for _, o in res] == [P.apply_calls(s, (U("check_status_bar"),))[1] for s in states]
    # n_workers <= 1 is the plain serial loop
    one = ParallelStepper(1).step_many(states[:3], (U("check_status_bar"),))
    assert all(_same(a, P.apply_calls(s, (U("check_status_bar"),))) for a, s in zip(one, states[:3]))
    # the whole filter: identical beliefs with and without workers
    b0 = make_filter(boed_cfg).init({"mobile_data_issue": 1.0})
    for mode in ("deployable", "oracle"):
        f_ser = make_filter(boed_cfg, mode)
        with ParallelStepper(4) as st:
            f_par = make_filter(boed_cfg, mode, stepper=st)
            for cand in (request(U("toggle_airplane_mode")), request(DeviceAction("user", "toggle_data", (), ("data", True)))):
                o1, o2 = f_ser.predict(b0, cand), f_par.predict(b0, cand)
                assert list(o1) == list(o2)
                for k in o1:
                    assert o1[k][0] == o2[k][0]
                    assert [(p.state.key, p.weight) for p in o1[k][1].particles] == [(p.state.key, p.weight) for p in o2[k][1].particles]
                    assert o1[k][1].v == o2[k][1].v


def test_init_prior_and_support(boed_cfg):
    flt = make_filter(boed_cfg)
    b = flt.init({"service_issue": 0.5, "mobile_data_issue": 0.5, "mms_issue": 0.0})
    assert not b.degenerate and sum(p.weight for p in b.particles) == pytest.approx(1.0)
    by_intent: dict = {}
    for p in b.particles:
        by_intent[p.hyp_key[0]] = by_intent.get(p.hyp_key[0], 0.0) + p.weight
    assert set(by_intent) == {"service_issue", "mobile_data_issue"}
    assert all(v == pytest.approx(0.5) for v in by_intent.values())
    n_svc = len(load_support("service_issue"))
    w_svc = {p.weight for p in b.particles if p.hyp_key[0] == "service_issue"}
    assert len(w_svc) <= 2 and min(w_svc) == pytest.approx(0.5 / n_svc)  # uniform within the intent (merged duplicates sum)
    assert dict(b.intent_probs) == {"mobile_data_issue": 0.5, "service_issue": 0.5}


def test_user_model_personas_load(boed_cfg):
    names = set(available_personas())
    assert {"cooperative", "terse", "confused", "contradictory", "noncompliant", "delayed", "default"} <= names
    base = UserModel.from_cfg(boed_cfg)
    assert (base.q, base.eps_double, base.eps_extra, base.default_mention_p) == (0.9, 0.02, 0.02, 0.7)
    assert base.mention_p("speed_test") == pytest.approx(0.91) and base.mention_p("status_bar") == pytest.approx(0.7)
    for n in names - {"default"}:
        um = UserModel.from_cfg(boed_cfg, n)
        assert um.persona == n and sum(um.branch_probs().values()) == pytest.approx(1.0)
    assert UserModel.from_cfg(boed_cfg, "noncompliant").report_when_not_done == "claims_done"
    with pytest.raises(KeyError):
        UserModel.from_cfg(boed_cfg, "nope")


def test_prune_knob(boed_cfg, small):
    b0, _ = small
    act = U("check_status_bar")
    obs = report_obs(act, bar_fields(b0.particles[1].state))
    full = make_filter(boed_cfg).update(b0, request(act), obs)
    tight = make_filter({**boed_cfg, "filter": {"prune_eps": 0.05}})
    assert tight.prune_eps == 0.05 and make_filter(boed_cfg).prune_eps == 1e-12
    pr = tight.update(b0, request(act), obs)
    assert len(pr.particles) < len(full.particles) and sum(p.weight for p in pr.particles) == pytest.approx(1.0)


def test_filter_module_deployable_clean():
    mods = ["telecom_boed.belief.filter", "telecom_boed.belief.user_model", "telecom_boed.belief.stepper"]
    rep = check_deployable(mods)
    assert not rep.pending, rep.pending
    assert not rep.problems(), "\n".join(rep.problems())
    for m in ("filter", "user_model", "stepper"):
        assert scan_module_source(BELIEF_SRC / f"{m}.py") == []


# --------------------------------------------------------------------------- benchmark (python tests/test_filter.py)
def _pct(xs, q):
    return float(np.percentile(np.array(xs) * 1e3, q))


def run_benchmark(intent: str = "mms_issue", workers=((1, True, None), (4, True, None), (8, True, None), (16, True, None),
                                                      (8, False, None), (1, True, 0.0), (16, True, 0.0)),
                  cold_reps: int = 3, warm_reps: int = 30) -> dict:
    """predict() latency on the full `intent` support. COLD = every cache empty (env.particles memos, filter step cache, grade
    memo) and a freshly forked pool, so workers have no memo either; the first call of an episode. WARM = step cache and grade
    memo populated, per-call predict/expand caches cleared (what a repeated scorer call on a NEW belief with known states pays)."""
    cfg = yaml.safe_load((paths.CONFIG_DIR / "boed_default.yaml").read_text())
    cands = {"check_status_bar": request(U("check_status_bar")), "toggle_airplane_mode": request(U("toggle_airplane_mode"))}
    b0 = ParticleFilter("deployable", cfg).init({intent: 1.0})
    res: dict = {"intent": intent, "n_particles": len(b0.particles), "cpu_count": os.cpu_count(), "loadavg_at_start": os.getloadavg(),
                 "persona": "default", "user_model": cfg["user_model"], "results": {}}
    for nw, persistent, eps_extra in workers:  # eps_extra=None keeps the config value (0.02); 0.0 drops the reboot branch
        cfg_run = cfg if eps_extra is None else {**cfg, "user_model": {**cfg["user_model"], "eps_extra": eps_extra}}
        for name, cand in cands.items():
            cold, n_out, n_part = [], 0, 0
            for _ in range(cold_reps):
                P.clear_caches()  # apply memo, grade memo, state cache, agent-DB intern table
                with ParallelStepper(nw, persistent=persistent) as st:
                    flt = ParticleFilter("deployable", cfg_run, stepper=st)
                    t = time.perf_counter()
                    out = flt.predict(b0, cand)
                    cold.append(time.perf_counter() - t)
                n_out, n_part = len(out), max(len(bp.particles) for _, bp in out.values())
            warm = []
            for _ in range(warm_reps):
                flt._pcache.clear(); flt._ecache.clear()
                t = time.perf_counter()
                flt.predict(b0, cand)
                warm.append(time.perf_counter() - t)
            tag = f"{name}|workers={nw}" + ("" if persistent else "|fork_per_call") + ("" if eps_extra is None else f"|eps_extra={eps_extra}")
            res["results"][tag] = {
                "cold_ms": [round(x * 1e3, 1) for x in cold], "cold_p50_ms": round(_pct(cold, 50), 1), "cold_p95_ms": round(_pct(cold, 95), 1),
                "warm_p50_ms": round(_pct(warm, 50), 1), "warm_p95_ms": round(_pct(warm, 95), 1),
                "n_outcomes": n_out, "max_beta_prime_particles": n_part}
            print(tag, res["results"][tag], flush=True)
    res["loadavg_at_end"] = os.getloadavg()
    return res


def test_benchmark_function_smoke():
    out = run_benchmark("service_issue", workers=((1, True, None), (2, True, None)), cold_reps=1, warm_reps=2)
    assert out["n_particles"] == len(load_support("service_issue"))
    assert all(v["n_outcomes"] >= 2 for v in out["results"].values())


if __name__ == "__main__":  # pragma: no cover
    result = run_benchmark()
    TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    TIMINGS.write_text(json.dumps(result, indent=1))
    print("wrote", TIMINGS)
