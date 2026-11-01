"""[telecom_boed] Tests for the reach-avoid leaf values and the transfer penalty (decide/rollout_leaf.py, 2026-11-01). Both are off
by default; the fakes below stand in for the tau2 env (states = frozensets of open faults)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import yaml

from telecom_boed import paths
from telecom_boed.contracts import Belief, CandidateAction, DecisionKind, DeviceAction, Particle
from telecom_boed.decide.reach_avoid import ReachAvoid
from telecom_boed.decide.rollout_leaf import LeafValue, base_order, sample_indices

V2 = yaml.safe_load((paths.CONFIG_DIR / "boed_v2.yaml").read_text())


@dataclass(frozen=True)
class St:
    key: frozenset  # open faults


FIX = {"toggle_airplane_mode": "air", "toggle_data": "data", "set_network_mode_preference": "net", "refuel_data": "gb"}


def fake_apply(s, calls):
    open_ = set(s.key)
    for c in calls:
        open_.discard(FIX.get(c.tool, ""))
    return St(frozenset(open_)), tuple("ok" for _ in calls)


class FakeGrader:
    def __init__(self, req=()):
        self.req = set(req)

    def grade(self, state, hk, transferred):
        if hk in self.req:
            return 1.0 if transferred else 0.0
        return 1.0 if not state.key else 0.0

    def requires_transfer(self, hk):
        return hk in self.req


def dev(tool, **a):
    return CandidateAction(DecisionKind.REQUEST_DEVICE, (DeviceAction("user", tool, tuple(a.items())),))


def agent(tool):
    return CandidateAction(DecisionKind.AGENT_TOOL, (DeviceAction("assistant", tool),))


def leaf(kind="rollout", req=(), **ra):
    cfg = {**V2, "reach_avoid": {**V2["reach_avoid"], "leaf": kind, "rollout_compliance": False, **ra}}
    return LeafValue(kind, FakeGrader(req), cfg, apply_fn=fake_apply, feat_fn=lambda t, o: {}, ensure_read={},
                     mutating=lambda t: t not in ("check_status_bar", "get_data_usage"), goal_frac=lambda s, hk: 1.0 - len(s.key) / 2)


def belief(*states_w, v=False, cost=0.0):
    return Belief(tuple(Particle(hk, St(frozenset(f)), w) for hk, f, w in states_w), v=v, cost=cost)


ROOT = [agent("refuel_data"), dev("check_status_bar"), dev("toggle_data"), dev("toggle_airplane_mode"), agent("get_data_usage")]


# ------------------------------------------------------------------ helpers
def test_sample_indices_deterministic_and_weighted():
    assert sample_indices([0.5, 0.25, 0.25], 4) == [0, 0, 1, 2]
    assert sample_indices([0, 0, 1], 3) == [2, 2, 2]
    assert sample_indices([], 3) == [] and sample_indices([1.0], 0) == []


def test_base_order_manual_then_agent_then_rest():
    order = [c.actions[0].tool for c in base_order(ROOT, lambda t: t not in ("check_status_bar", "get_data_usage"))]
    assert order == ["toggle_airplane_mode", "toggle_data", "refuel_data"]  # reads dropped; MANUAL_ORDER rank; agent tools after


# ------------------------------------------------------------------ rollout value
def test_rollout_counts_every_walked_step_and_verify():
    L = leaf()
    L.set_root(ROOT)
    lam = V2["cost"]["lambda"]
    # needs toggle_data only: walks airplane (1 turn, no change) + data (1 turn), then 1 verify turn
    assert L.value(belief(("h", {"data"}, 1.0))) == pytest.approx(1.0 - lam * 3)
    # already fixed, not verified: verify only; verified: free
    assert L.value(belief(("h", set(), 1.0))) == pytest.approx(1.0 - lam * 1)
    assert L.value(belief(("h", set(), 1.0), v=True)) == pytest.approx(1.0)


def test_rollout_skips_done_unless_still_unmet_and_respects_budget():
    L = leaf()
    L.set_root(ROOT)
    lam, B = V2["cost"]["lambda"], V2["cost"]["budget_B"]
    L.mark_done(ROOT[3])  # airplane already asked this episode
    assert L.value(belief(("h", {"data"}, 1.0))) == pytest.approx(1.0 - lam * 2)  # airplane skipped (no-op here)
    assert L.value(belief(("h", {"air"}, 1.0))) == pytest.approx(1.0 - lam * 2)  # done but still open -> redone
    assert L.value(belief(("h", {"data"}, 1.0), cost=B - 1.5)) == pytest.approx(-lam * 2)  # cannot fit 2 turns


def test_rollout_transfer_hypothesis_and_weights():
    L = leaf(req=("t",))
    L.set_root(ROOT)
    lam = V2["cost"]["lambda"]
    walk = 1 + 1 + V2["cost"]["tool_cost"]  # airplane, data, refuel
    v = L.value(belief(("t", {"x"}, 0.5), ("h", set(), 0.5)))
    assert v == pytest.approx(0.5 * (1.0 - lam * walk) + 0.5 * (1.0 - lam))


def test_rollout_compliance_scales_device_steps():
    cfg_q = {**V2, "reach_avoid": {**V2["reach_avoid"], "leaf": "rollout", "rollout_compliance": True}}

    class UM:
        def q_for(self, tool):
            return 0.5
    L = LeafValue("rollout", FakeGrader(), cfg_q, user_model=UM(), apply_fn=fake_apply, feat_fn=lambda t, o: {}, ensure_read={},
                  mutating=lambda t: t not in ("check_status_bar", "get_data_usage"))
    L.set_root(ROOT)
    lam = V2["cost"]["lambda"]
    assert L.value(belief(("h", {"data"}, 1.0))) == pytest.approx(1.0 - lam * (2 + 2 + 1))  # two device steps at 1/q = 2 turns each


def test_fraction_leaf():
    L = leaf("fraction", leaf_fraction_scale=0.5)
    assert L.value(belief(("h", {"data"}, 1.0))) == pytest.approx(0.25)


# ------------------------------------------------------------------ switches are off by default
def test_reach_defaults_unchanged():
    ra = ReachAvoid(predictor=None, grader=FakeGrader(), cfg=V2)
    assert ra.leaf is None and ra.transfer_penalty is False


@pytest.mark.parametrize("name,kind", [("boed_v2_tmpl_rollout", "rollout"), ("boed_v2_tmpl_fraction", "fraction"),
                                       ("boed_v2_umfit_tmpl", None)])
def test_new_configs(name, kind):
    cfg = yaml.safe_load((paths.CONFIG_DIR / f"{name}.yaml").read_text())
    assert cfg["name"] == name
    ra = ReachAvoid(predictor=None, grader=FakeGrader(), cfg=cfg)
    assert (ra.leaf.kind if ra.leaf else None) == kind
    assert ra.transfer_penalty is (kind is not None)
    if kind:
        assert cfg["lookahead"]["depth_k"] == 1
    for k in ("cost", "utility", "user_model"):  # the rest of boed_v2 is untouched apart from compliance
        if k != "user_model":
            assert cfg[k] == V2[k]


def test_transfer_penalty_costs_c_esc_times_p_fixable():
    from telecom_boed.decide.utility import q_transfer_reach, q_transfer_reach_pen
    g = FakeGrader(req=("t",))
    b = belief(("t", {"x"}, 0.3), ("h", set(), 0.7))
    c = V2["utility"]["c_esc"]
    assert q_transfer_reach(b, g) == pytest.approx(0.3)
    assert q_transfer_reach_pen(b, g, c) == pytest.approx(0.3 - c * 0.7)
    on = ReachAvoid(predictor=None, grader=g, cfg={**V2, "reach_avoid": {**V2["reach_avoid"], "transfer_penalty": True}})
    tr = CandidateAction(DecisionKind.TRANSFER)
    assert on.q_values(b, [tr]) == [pytest.approx(0.3 - c * 0.7)]
    assert ReachAvoid(predictor=None, grader=g, cfg=V2).q_values(b, [tr]) == [pytest.approx(0.3)]
