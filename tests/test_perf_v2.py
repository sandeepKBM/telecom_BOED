"""P0-2 (boed_v2): the new perf knobs. Old configs must be untouched (test_perf_equivalence / test_filter_perf_equivalence / test_controller
/ test_lookahead are the golden checks); here: bit-identical state hashes, bundle cap, abort checks, deadline margin, config diff."""
from __future__ import annotations

import time
from dataclasses import fields, replace

import pytest
import yaml
from loguru import logger

from telecom_boed import paths
from telecom_boed.belief import stepper as ST
from telecom_boed.belief.filter import ParticleFilter
from telecom_boed.contracts import CandidateAction, DecisionKind, DeviceAction, _frozen_items
from telecom_boed.decide.controller import Controller, _Timeout
from telecom_boed.env import particles as P
from telecom_boed.env.hypotheses import load_support
from telecom_boed.testing.toy_pomdp import label, load_toy

logger.disable("tau2")
REQ = DecisionKind.REQUEST_DEVICE


def same(a, b) -> bool:
    """Snapshot has identity equality; compare every field."""
    return all(getattr(a, f.name) == getattr(b, f.name) for f in fields(a))


def U(tool, ensure=None, **a):
    return DeviceAction("user", tool, _frozen_items(a), ensure)


@pytest.fixture(autouse=True)
def _reset_perf():
    P.configure_perf(reuse_state_hash=False)
    yield
    P.configure_perf(reuse_state_hash=False)


def test_boed_v2_differs_from_default_only_by_name_agent_perf():
    a = yaml.safe_load((paths.CONFIG_DIR / "boed_default.yaml").read_text())
    b = yaml.safe_load((paths.CONFIG_DIR / "boed_v2.yaml").read_text())
    assert {k for k in b if k not in a} == {"agent", "perf"}
    assert [k for k in a if a[k] != b[k]] == ["name"] and b["name"] == "boed_v2"
    assert b["agent"]["intent_prune_below"] == 0.1 and b["perf"]["reuse_state_hash"] is True


def test_state_hash_reuse_is_bit_identical():
    hyps = load_support("mobile_data_issue")[:6] + load_support("mms_issue")[:6]
    seqs = [(U("check_status_bar"),), (U("toggle_airplane_mode"),), (U("toggle_data"),), (U("reboot_device"),),
            (U("reset_apn_settings"), U("reboot_device")), (U("run_speed_test"), U("toggle_data"), U("check_network_status"))]
    for h in hyps:
        s0 = P.build_state(h)
        for calls in seqs:
            P._APPLY_MEMO.clear()
            P.configure_perf(reuse_state_hash=False)
            ref = P.apply_calls(s0, calls)
            P._APPLY_MEMO.clear()
            P.configure_perf(reuse_state_hash=True)
            got = P.apply_calls(s0, calls)
            assert got[1] == ref[1] and same(got[0], ref[0]) and got[0].key == ref[0].key and got[0].snapshot_id == ref[0].snapshot_id
            # second step from the new state (parent != pristine)
            P._APPLY_MEMO.clear()
            P.configure_perf(reuse_state_hash=False)
            ref2 = P.apply_calls(ref[0], (U("toggle_airplane_mode"),))
            P._APPLY_MEMO.clear()
            P.configure_perf(reuse_state_hash=True)
            got2 = P.apply_calls(got[0], (U("toggle_airplane_mode"),))
            assert got2[1] == ref2[1] and same(got2[0], ref2[0])


def test_filter_init_sets_switches_absolutely_and_chunk_adopts_them(boed_cfg):
    cfg2 = {**boed_cfg, "perf": {"reuse_state_hash": True}}
    ParticleFilter("deployable", cfg2)
    assert P.PERF["reuse_state_hash"] is True
    ParticleFilter("deployable", boed_cfg)  # no perf block -> back to default
    assert P.PERF == P.PERF_DEFAULT
    ST._run_chunk_perf(({"reuse_state_hash": True}, []))
    assert P.PERF["reuse_state_hash"] is True


class _Rec:
    """Wraps the toy predictor; records the action count it was asked to predict; strips extras before delegating."""

    def __init__(self, inner):
        self.inner, self.seen = inner, []

    def predict(self, belief, cand):
        self.seen.append(len(cand.actions))
        return self.inner.predict(belief, replace(cand, actions=cand.actions[:1]) if cand.actions else cand)

    def __getattr__(self, n):
        return getattr(self.inner, n)


def _bundle(c):
    return replace(c, actions=c.actions + (U("check_network_status"), U("run_speed_test"))) if c.kind == REQ else c


def test_bundle_cap_scores_prefix_but_returns_original():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    big = [_bundle(c) for c in cands]
    base = Controller("dug", _Rec(pred), grader, cfg, override="none", time_budget_ms=None).decide(b0, [replace(c) for c in cands])
    rec = _Rec(pred)
    capped = {**cfg, "perf": {"bundle_cap_actions": 1}}
    ctl = Controller("dug", rec, grader, capped, override="none", time_budget_ms=None)
    d = ctl.decide(b0, big)
    assert max(rec.seen) == 1 and ctl.stats["n_capped"] == sum(c.kind == REQ for c in cands)
    assert d.chosen in big and len(d.chosen.actions) == len(big[big.index(d.chosen)].actions)  # the original (uncapped) object
    assert label(d.chosen) == label(base.chosen) and d.scores == base.scores
    # without the knob the bundle goes to the predictor whole, and no n_capped key appears
    rec2 = _Rec(pred)
    c2 = Controller("dug", rec2, grader, cfg, override="none", time_budget_ms=None)
    c2.decide(b0, big)
    assert max(rec2.seen) == 3 and "n_capped" not in c2.stats


class _Slow:
    """Predictor that polls abort_check in the middle of predict, like ParticleFilter does between stepper waves."""

    def __init__(self, inner):
        self.inner, self.abort_check, self.polled, self.aborted = inner, None, 0, 0

    def predict(self, belief, cand):
        for _ in range(5):
            time.sleep(0.03)
            if self.abort_check is not None:
                self.polled += 1
                try:
                    self.abort_check()
                except _Timeout:
                    self.aborted += 1
                    raise
        return self.inner.predict(belief, cand)

    def __getattr__(self, n):
        return getattr(self.inner, n)


def test_predict_abort_checks_fall_back_and_reset():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    slow = _Slow(pred)
    on = {**cfg, "perf": {"predict_abort_checks": True}}
    ctl = Controller("dug", slow, grader, on, override="none", time_budget_ms=100)
    d = ctl.decide(b0, cands)
    assert slow.aborted >= 1 and ctl.stats["fallback"] and d.chosen is not None
    assert slow.abort_check is None  # uninstalled after every decision
    # off (old behaviour): checks are never installed, predicts complete
    slow2 = _Slow(pred)
    ctl2 = Controller("dug", slow2, grader, cfg, override="none", time_budget_ms=100)
    ctl2.decide(b0, cands)
    assert slow2.polled == 0 and slow2.aborted == 0


def test_deadline_margin_shortens_the_deadline():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    seen = {}

    class Spy(_Rec):
        def predict(self, belief, cand):
            seen["deadline_left_ms"] = (ctl.proxy.deadline - time.perf_counter()) * 1e3
            return super().predict(belief, cand)

    ctl = Controller("dug", Spy(pred), grader, {**cfg, "perf": {"deadline_margin_ms": 400}}, override="none", time_budget_ms=500)
    ctl.decide(b0, cands)
    assert 0 < seen["deadline_left_ms"] <= 100
