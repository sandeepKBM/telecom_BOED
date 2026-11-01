"""[telecom_boed] W5-A2 tests: per-agent-turn H7 labels (internals/labels.py, scripts/h7_labels.py).

Layers: (1) fault-presence definition on the real catalog (turn 0 = the task's fault set, gold fix clears it, toggle direction);
(2) synthetic episodes (turn / msg_idx keying, claim and false-accept labels, MC return, transfer flag, sidecar alignment);
(3) integration on two real TRAIN runs (tests/fixtures/W5-A2/): validation invariants + golden rows; (4) the CLI.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from telecom_boed import paths
from telecom_boed.belief.parsers import RuleActionParser, RuleClaimClassifier
from telecom_boed.env import hypotheses as H
from telecom_boed.env import oracle_hidden as O
from telecom_boed.env import particles as P
from telecom_boed.env import snapshot as S
from telecom_boed.internals import labels as L

FX = Path(__file__).parent / "fixtures" / "W5-A2"
PL, CT = FX / "f1plrep_526576.35", FX / "f1ctrep_526576.32"
SCRIPT = paths.PACKAGE_ROOT / "scripts" / "h7_labels.py"
CLAIM = "Your service issue has been resolved. Is there anything else I can help you with today?"
GREET = L.GREETING

CLF, PARSER = RuleClaimClassifier(), RuleActionParser()


def _rows():
    with open(paths.telecom_catalog_paths()[0]) as f:
        return {r["id"]: r for r in json.load(f)}


def _tc(i, name, requestor, args=None):
    return {"id": f"c{i}", "name": name, "arguments": args or {}, "requestor": requestor}


def _msg(role, content=None, calls=None):
    return {"role": role, "content": content, "tool_calls": calls or None}


def _sim(task_id, msgs, term="user_stop", reward=1.0, trial=0, sid="s0"):
    return {"id": sid, "task_id": task_id, "trial": trial, "termination_reason": term, "reward_info": {"reward": reward},
            "messages": msgs}


def _task(task_id):
    return _rows()[task_id]


def _label(sim, gamma=1.0, sidecar=None):
    return L.label_episode(sim, _task(sim["task_id"]), run="r", persona="native", classifier=CLF, parser=PARSER,
                           gamma=gamma, sidecar=sidecar)


AIR = "[service_issue]airplane_mode_on[PERSONA:None]"


# --------------------------------------------------------------------------- (1) fault presence
def test_fault_names_cover_catalog():
    cat = {f for h in H.load_catalog() for f in h.faults}
    assert cat == set(L.FAULT_NAMES)


def test_turn0_presence_equals_fault_set_on_sampled_catalog():
    random.seed(1)
    for h in random.sample(H.load_catalog(), 120):
        pr = L.fault_presence(P.build_state(h))
        assert sorted(f for f, p in pr.items() if p) == sorted(h.faults), h.key


def test_every_single_fault_hypothesis_is_detected_alone():
    for h in H.load_catalog():
        if len(h.faults) == 1 and h.intent == "mobile_data_issue":
            pr = L.fault_presence(P.build_state(h))
            assert [f for f, p in pr.items() if p] == list(h.faults), h.key


def test_gold_fix_clears_all_faults_on_non_transfer_tasks():
    rows = _rows()
    random.seed(2)
    hs = [h for h in H.load_catalog() if not h.requires_transfer]
    for h in random.sample(hs, 40):
        acts = O.gold_actions(rows[h.task_ids[0]])
        env = S.restore(P.build_state(h))
        from tau2.data_model.message import ToolCall
        for i, a in enumerate(acts):
            env.get_response(ToolCall(id=f"g{i}", name=a.tool, arguments=a.args_dict(), requestor=a.requestor))
        snap = S.snapshot(env)
        assert not any(L.fault_presence(snap).values()), h.key
        assert P.grade(snap, h, False) == 1.0


def test_toggle_direction_toggling_airplane_twice():
    """The airplane fix is a toggle: a second toggle must bring the fault back (state-changing fix alone could not tell)."""
    from tau2.data_model.message import ToolCall
    h = H.hypothesis_by_key(H.parse_task_id(AIR))
    env = S.restore(P.build_state(h))
    assert L.fault_presence(S.snapshot(env))["airplane_mode_on"]
    env.get_response(ToolCall(id="a", name="toggle_airplane_mode", arguments={}, requestor="user"))
    assert not L.fault_presence(S.snapshot(env))["airplane_mode_on"]
    env.get_response(ToolCall(id="b", name="toggle_airplane_mode", arguments={}, requestor="user"))
    assert L.fault_presence(S.snapshot(env))["airplane_mode_on"]


def test_agent_induced_fault_appears_on_a_clean_state():
    """Toggling data off on a task that has no data fault creates data_mode_off (labels are state-based, not task-based)."""
    from tau2.data_model.message import ToolCall
    h = H.hypothesis_by_key(H.parse_task_id(AIR))
    env = S.restore(P.build_state(h))
    env.get_response(ToolCall(id="a", name="toggle_data", arguments={}, requestor="user"))
    pr = L.fault_presence(S.snapshot(env))
    assert pr["data_mode_off"] and pr["airplane_mode_on"]


def test_unfixable_faults_use_init_persistence():
    lock = H.hypothesis_by_key(("service_issue", frozenset({"lock_sim_card_pin"})))
    ce = H.hypothesis_by_key(("service_issue", frozenset({"contract_end_suspension"})))
    ob = H.hypothesis_by_key(("service_issue", frozenset({"overdue_bill_suspension"})))
    assert [f for f, p in L.fault_presence(P.build_state(lock)).items() if p] == ["lock_sim_card_pin"]
    assert [f for f, p in L.fault_presence(P.build_state(ce)).items() if p] == ["contract_end_suspension"]
    assert [f for f, p in L.fault_presence(P.build_state(ob)).items() if p] == ["overdue_bill_suspension"]


def test_overdue_suspension_clears_after_resume_even_when_fix_cannot_build():
    """After payment the tau2 fix raises (no overdue bill left); presence then falls back to the gate (line suspended)."""
    from tau2.data_model.message import ToolCall
    h = H.hypothesis_by_key(("service_issue", frozenset({"overdue_bill_suspension"})))
    env = S.restore(P.build_state(h))
    acts = O.gold_actions(_task(h.task_ids[0]))
    seen = []
    for i, a in enumerate(acts):
        env.get_response(ToolCall(id=f"g{i}", name=a.tool, arguments=a.args_dict(), requestor=a.requestor))
        seen.append(L.fault_presence(S.snapshot(env))["overdue_bill_suspension"])
    assert seen[0] is True and seen[-1] is False  # present until resume_line, gone after
    assert any(seen[:-1]) and seen.index(False) >= 2


def test_roaming_gates_partition():
    for faults, want in ((("user_abroad_roaming_enabled_off",), "user_abroad_roaming_enabled_off"),
                         (("user_abroad_roaming_disabled_on",), "user_abroad_roaming_disabled_on"),
                         (("user_abroad_roaming_disabled_off",), "user_abroad_roaming_disabled_off")):
        h = H.hypothesis_by_key(("mobile_data_issue", frozenset(faults)))
        assert [f for f, p in L.fault_presence(P.build_state(h)).items() if p] == [want]


# --------------------------------------------------------------------------- (2) synthetic episodes
def _gold_msgs():
    """greeting, assistant asks, user toggles airplane, assistant claims, user stops. Replay-consistent for AIR."""
    return [
        _msg("assistant", GREET),
        _msg("user", "My phone has no service."),
        _msg("assistant", "Please turn off airplane mode."),                                  # idx 2, turn 1: before the fix
        _msg("user", None, [_tc(1, "toggle_airplane_mode", "user")]),
        _msg("tool", "Airplane Mode is now OFF."),
        _msg("assistant", None, [_tc(2, "check_status_bar", "assistant")]),                   # idx 5, turn 2: tool call, after fix
        _msg("tool", "ok"),
        _msg("assistant", CLAIM),                                                              # idx 7, turn 3: true claim
        _msg("user", "Thanks ###STOP###"),
    ]


def test_keying_turn_msg_idx_and_counts():
    sim = _sim(AIR, _gold_msgs())
    rows = _label(sim)
    assert [(r["turn"], r["msg_idx"]) for r in rows] == [(0, 0), (1, 2), (2, 5), (3, 7)]
    assert all(r["n_assistant_msgs"] == 4 for r in rows)
    assert rows[0]["is_greeting"] and not rows[1]["is_greeting"]
    assert L.turn_of(sim["messages"], 7) == 3


def test_labels_are_on_the_state_before_the_message():
    rows = _label(_sim(AIR, _gold_msgs()))
    # idx 2 precedes the user's fix; idx 5 / 7 follow it
    assert [r["faults_remaining"] for r in rows] == [["airplane_mode_on"], ["airplane_mode_on"], [], []]
    assert [r["repaired_now"] for r in rows] == [False, False, True, True]
    assert [r["n_faults_remaining"] for r in rows] == [1, 1, 0, 0]


def test_claim_tool_and_device_request_labels():
    rows = _label(_sim(AIR, _gold_msgs()))
    assert [r["is_claim"] for r in rows] == [False, False, False, True]
    assert rows[3]["claim_prob"] >= 0.5 and rows[3]["false_accept_here"] is False
    assert [r["is_tool_call"] for r in rows] == [False, False, True, False]
    assert rows[2]["tool_name"] == "check_status_bar" and rows[2]["claim_prob"] is None
    assert rows[1]["is_device_request"] is True and rows[3]["is_device_request"] is False


def test_false_accept_when_claim_precedes_the_fix():
    msgs = [_msg("assistant", GREET), _msg("user", "No service."), _msg("assistant", CLAIM),
            _msg("user", "Still broken ###STOP###")]
    rows = _label(_sim(AIR, msgs, reward=0.0))
    assert rows[1]["is_claim"] and rows[1]["repaired_now"] is False and rows[1]["false_accept_here"] is True
    assert rows[1]["final_reward"] == 0.0 and rows[1]["mc_return"] == 0.0


def test_mc_return_discounts_by_remaining_agent_turns():
    rows = _label(_sim(AIR, _gold_msgs()), gamma=0.9)
    assert [r["remaining_agent_turns"] for r in rows] == [3, 2, 1, 0]
    assert [round(r["mc_return"], 6) for r in rows] == [round(0.9 ** k, 6) for k in (3, 2, 1, 0)]
    assert all(r["gamma"] == 0.9 for r in rows)


def test_gamma_from_config_and_end_class():
    assert L._gamma() == 1.0
    rows = _label(_sim(AIR, _gold_msgs()))
    assert {r["end_class"] for r in rows} == {"stop"} and {r["termination"] for r in rows} == {"user_stop"}


def test_transfer_flag_flips_after_the_transfer_message():
    tid = next(h.task_ids[0] for h in H.load_catalog() if h.requires_transfer)
    msgs = [_msg("assistant", GREET), _msg("user", "Help."),
            _msg("assistant", None, [_tc(1, "transfer_to_human_agents", "assistant", {"summary": "x"})]),
            _msg("tool", "Transfer successful"), _msg("assistant", "Done."), _msg("user", "ok ###TRANSFER###")]
    rows = L.label_episode(_sim(tid, msgs, reward=0.0), _task(tid), run="r", persona=None, classifier=CLF, parser=PARSER)
    assert [r["transferred_so_far"] for r in rows] == [False, False, True]
    assert all(r["requires_transfer"] for r in rows)


def test_sidecar_alignment_skips_the_greeting():
    sc = [{"turn": 1, "posterior_topk": [[["airplane_mode_on"], 0.5]], "parse": {"belief": {"p_repaired": 0.1, "v": False, "n_particles": 7,
           "intent_probs": {"service_issue": 1.0}}}} for _ in range(3)]
    for i, r in enumerate(sc):
        r["turn"] = i + 1
    rows = _label(_sim(AIR, _gold_msgs()), sidecar=sc)
    assert rows[0]["filter"] is None
    assert [r["filter"]["sidecar_turn"] for r in rows[1:]] == [1, 2, 3]
    assert rows[1]["filter"]["p_repaired"] == 0.1 and rows[1]["filter"]["n_particles"] == 7
    assert rows[1]["filter"]["topk"] == [[["airplane_mode_on"], 0.5]]
    # a sidecar of the wrong length is dropped, never misaligned
    assert all(r["filter"] is None for r in _label(_sim(AIR, _gold_msgs()), sidecar=sc[:2]))


def test_persona_from_meta():
    f = L.persona_from_meta
    assert f({"run_config": {"content": "name: x\nuser: user_simulator\n"}}) == "native"
    assert f({"run_config": {"content": "name: x\nuser: boed-user-terse\n"}}) == "terse"
    assert f({"run_config": {"content": ""}}) is None and f(None) is None


def test_balance_counts():
    rows = _label(_sim(AIR, _gold_msgs()))
    b = L.balance(rows)
    assert b["n_rows"] == 4 and b["n_episodes"] == 1
    assert b["repaired_now"] == {"true": 2, "false": 2, "frac_true": 0.5}
    assert b["fault_present"]["airplane_mode_on"]["true"] == 2 and b["fault_present"]["bad_vpn"]["true"] == 0


# --------------------------------------------------------------------------- (3) real TRAIN runs
@pytest.mark.parametrize("run", [PL, CT])
def test_validation_invariants_on_real_runs(run):
    res = L.label_run(run)
    assert res["errors"] == []
    v = L.validate(res)
    assert v["n_sims_labeled"] == v["n_sims"] == 5 and v["n_user_stop"] >= 4
    assert v["turn_count_mismatch"] == [] and v["turn0_fault_set_mismatch"] == []
    assert v["last_turn_repaired_ne_reward_user_stop"] == [] and v["replay_final_ne_reward_user_stop"] == []
    assert res["sidecar_matched"] == 5
    assert all(r["filter"] is not None for r in res["rows"] if not r["is_greeting"])


def test_golden_rows_f1plrep():
    got = L.label_run(PL)["rows"]
    exp = json.loads((FX / "expected_f1plrep.json").read_text())
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        for k, v in e.items():
            if k == "filter":
                assert (g["filter"] is None) == (v is None)
                if v is not None:
                    assert {kk: g["filter"][kk] for kk in v} == v
            elif k == "claim_prob":
                assert (g[k] is None) == (v is None) and (v is None or abs(g[k] - v) < 1e-3)
            else:
                assert g[k] == v, (k, g["task_id"], g["turn"])


# --------------------------------------------------------------------------- (4) CLI
def test_cli_writes_labels_and_summary(tmp_path):
    out = tmp_path / "o"
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), str(PL), "--out", str(out), "--strict"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(x) for x in (out / "labels.jsonl").read_text().splitlines()]
    doc = json.loads((out / "summary.json").read_text())
    assert len(rows) == doc["n_rows"] == 56 and doc["key"] == ["run", "task_id", "trial", "turn", "msg_idx"]
    assert len({(x["run"], x["task_id"], x["trial"], x["turn"], x["msg_idx"]) for x in rows}) == len(rows)
    assert doc["runs"]["f1plrep_526576.35"]["validation"]["turn0_fault_set_mismatch"] == []
    assert "repaired_now" in doc["all_runs_balance"]


def test_cli_refuses_common_users_output(tmp_path):
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), str(PL), "--out", "/common/users/ss5772/should_not_exist_h7"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and not Path("/common/users/ss5772/should_not_exist_h7").exists()
