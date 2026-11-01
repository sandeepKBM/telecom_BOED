"""[telecom_boed] Tests for boed-script (tau2_ext/belief_script.py) and the per-type user-model compliance (Test 1 / Test 2)."""
from __future__ import annotations

import json
import re

import pytest
from tau2.data_model.message import MultiToolMessage, ToolCall, ToolMessage, UserMessage

from telecom_boed import paths
from telecom_boed.belief.user_model import ENSURE_READ, UserModel, is_mutating_tool
from telecom_boed.env.featurize import featurize
from telecom_boed.contracts import DecisionKind
from telecom_boed.env import particles as P
from telecom_boed.env import snapshot as S
from telecom_boed.env.hypotheses import hypothesis_by_key
from telecom_boed.env.oracle_hidden import gold_actions, true_key  # test-only (the agent never imports it)
from telecom_boed.tau2_ext import belief_script as B
from telecom_boed.testing.leakage_scan import check_deployable

_CAT, _SPLIT = paths.telecom_catalog_paths()
BYID = {t["id"]: t for t in json.loads(_CAT.read_text())}
TRAIN = json.loads(_SPLIT.read_text())["train"]
V2 = str(paths.CONFIG_DIR / "boed_v2.yaml")


# ------------------------------------------------------------------ user model (Test 2 switch, default off)
def test_user_model_default_unchanged():
    um = UserModel()
    for tool in ("check_status_bar", "toggle_airplane_mode", "grant_app_permission", "run_speed_test"):
        assert um.q_for(tool) == um.q
        assert dict(um.branches_for(tool)) == {"nd": um.p_not_done, "d": um.q, "dd": um.eps_double}


def test_user_model_per_type():
    um = UserModel(q_read=0.67, q_mutating=0.34)
    assert um.q_for("check_status_bar") == 0.67 and not is_mutating_tool("check_status_bar")
    assert um.q_for("toggle_data") == 0.34 and is_mutating_tool("toggle_data")
    assert abs(sum(p for _, p in um.branches_for("toggle_data")) - 1.0) < 1e-12


def test_user_model_from_cfg_keys():
    base = {"user_model": {"compliance_q": 0.9, "eps_double": 0.02, "eps_extra": 0.02, "default_mention_p": 0.7}}
    assert UserModel.from_cfg(base).q_read is None
    cfg = {"user_model": {**base["user_model"], "compliance_q_read": 0.672, "compliance_q_mutating": 0.338}}
    um = UserModel.from_cfg(cfg)
    assert (um.q_read, um.q_mutating) == (0.672, 0.338)
    with pytest.raises(ValueError):
        UserModel(q_mutating=0.99, eps_double=0.02)


def test_umfit_config_loads():
    import yaml
    cfg = yaml.safe_load((paths.CONFIG_DIR / "boed_v2_umfit.yaml").read_text())
    um = UserModel.from_cfg(cfg)
    assert um.q_read == pytest.approx(0.672) and um.q_mutating == pytest.approx(0.338)


# ------------------------------------------------------------------ action set + leakage
def test_find_phone():
    assert B.find_phone("my number is 555-123-2002, thanks") == "555-123-2002"
    assert B.find_phone("it's (5551232002)") == "555-123-2002"
    assert B.find_phone("no number here") is None


def test_action_set_covers_every_train_gold_tool():
    """Every tool a train-74 gold repair needs is in the script's action set (device templates or backend set)."""
    sc = B._Script()
    sc.customer_id, sc.line_id, sc.overdue_bills = "C1", "L1", ["B1"]
    agent = B.BoedScriptAgent(tools=[], domain_policy="policy", llm="none", llm_args={"boed_config": V2})
    have = {a.tool for c in B.device_candidates(list(B.INTENT_ACTIONS)) + agent.backend_candidates(sc) for a in c.actions}
    have.add(B.TRANSFER_TOOL)
    need = {a.tool for t in TRAIN for a in gold_actions(BYID[t])}
    assert need <= have, sorted(need - have)


def test_deployable_scan_includes_script():
    rep = check_deployable()
    assert "telecom_boed.tau2_ext.belief_script" in str(rep) or rep.ok
    assert rep.ok, rep


# ------------------------------------------------------------------ full episode with a scripted, fully compliant fake user
def _phone(task):
    m = re.search(r"\d{3}-\d{3}-\d{4}", task["user_scenario"]["instructions"]["known_info"])
    return m.group(0)


def _episode(tid, scorer="dug", max_steps=80):
    """Fake user: gives the phone number when asked; otherwise runs the user-side actions of the agent's last candidate on
    the real tau2 env and replies with the tool outputs. Test-only access to the agent's candidate (the env is the truth)."""
    task = BYID[tid]
    env = S.restore(P.build_state(hypothesis_by_key(true_key(task))))
    agent = B.BoedScriptAgent(tools=[], domain_policy="policy", llm="none",
                              llm_args={"boed_config": V2, "boed_scorer": scorer, "boed_time_budget_ms": 2000})
    state = agent.get_init_state()
    msg = UserMessage(role="user", content=task["user_scenario"]["instructions"]["reason_for_call"])
    transferred, claimed, n = False, False, 0
    for _ in range(max_steps):
        out, state = agent.generate_next_message(msg, state)
        n += 1
        if out.tool_calls:
            tc = out.tool_calls[0]
            assert tc.requestor == "assistant"
            transferred |= tc.name == B.TRANSFER_TOOL
            tm = env.get_response(tc)
            msg = ToolMessage(id=tc.id, role="tool", content=tm.content, requestor="assistant", error=tm.error)
            continue
        if B.BoedScriptAgent.is_stop(out) or transferred:
            break
        if out.content == B.CLOSING:
            claimed = True
            msg = UserMessage(role="user", content="Thanks, bye.")
            continue
        if out.content in (B.ASK_PHONE, B.ASK_PHONE_AGAIN):
            msg = UserMessage(role="user", content=f"My number is {_phone(task)}.")
            continue
        cand = agent._last_ep.last
        replies = []
        for a in (cand.actions if cand is not None else ()):
            if a.requestor != "user":
                continue
            args = a.args_dict()
            tool = a.tool
            if a.ensure is not None:  # "make sure X is off": toggle only when needed (read via the real env)
                rtool, feat = ENSURE_READ[a.ensure[0]]
                cur = featurize(rtool, env.get_response(ToolCall(id=f"r{n}", name=rtool, arguments={}, requestor="user")).content)
                if cur.get(feat) == a.ensure[1]:
                    replies.append(f"It is already {'on' if a.ensure[1] else 'off'}.")
                    continue
            tm = env.get_response(ToolCall(id=f"u{n}", name=tool, arguments=args, requestor="user"))
            replies.append(tm.content)
        msg = UserMessage(role="user", content=" ".join(replies) or "ok")
    return env, transferred, claimed, n, agent


@pytest.mark.parametrize("tid", [t for t in TRAIN if "ACTION" not in BYID[t]["evaluation_criteria"]["reward_basis"]][:2])
def test_script_episode_runs_and_is_bounded(tid):
    env, transferred, claimed, n, agent = _episode(tid)
    assert n <= 80
    recs = agent.records
    assert recs and any(r.decision.get("phase") == "lookup" for r in recs)
    assert any(r.decision.get("phase") == "boed" for r in recs)
    assert all(r.decision.get("mode") == "script" for r in recs if r.decision)
    assert claimed or transferred or n == 80
