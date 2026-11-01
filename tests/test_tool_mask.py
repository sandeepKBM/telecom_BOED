"""[telecom_boed] Step 1 (2026-11-01) tests: phone-tool prompt mask and reject + resample (tau2_ext/tool_mask.py). CPU only, no LLM
(tau2.agent.llm_agent.generate patched). Both switches are off by default."""
from __future__ import annotations

import json
import re

import pytest
import yaml
from loguru import logger
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.telecom.environment import get_environment

from telecom_boed import paths
from telecom_boed.env.featurize import USER_TOOLS
from telecom_boed.tau2_ext import agents as A
from telecom_boed.tau2_ext import candidates as C
from telecom_boed.tau2_ext.tool_mask import MASK_NOTE, REJECT_TEXT, label, mask_policy, mask_tools
from test_agents import msg_dump, run_episode, small_factory  # noqa: E402

logger.disable("tau2")
USER = [UserMessage(role="user", content="My mobile data is not working at all.")]


@pytest.fixture(scope="module")
def env_tools():
    env = get_environment()
    return env.get_tools(), env.get_policy()


@pytest.fixture(autouse=True)
def _clean_defaults():
    A.reset_configuration()
    yield
    A.reset_configuration()


def tc_msg(name, args=None, i=0):
    return AssistantMessage(role="assistant", content=None, cost=0.0,
                            tool_calls=[ToolCall(id=f"call_{i}", name=name, arguments=args or {}, requestor="assistant")])


def recording_generate(seq):
    it, calls = iter(seq), []

    def gen(*a, **k):
        calls.append(k)
        return next(it)
    return gen, calls


def _plain(tools, policy, side, **extra):
    la = {"temperature": 0.0, "boed_scorer": "dug", "boed_sidecar": str(side), "boed_filter_factory": small_factory(),
          "boed_time_budget_ms": None, **extra}
    return A.FACTORIES["boed-plain"](tools=tools, domain_policy=policy, llm="m", llm_args=la, task=None)


def _names_left(text):
    return [t for t in USER_TOOLS if re.search(r"\b" + t + r"\b", text)]


# ------------------------------------------------------------------ prompt mask
def test_mask_policy_real_policy(env_tools):
    tools, policy = env_tools
    own = [t.name for t in tools]
    assert _names_left(policy)  # the manual names phone tools (why Qwen calls them)
    out, n = mask_policy(policy, own)
    assert n >= 55 and not _names_left(out) and out.endswith(MASK_NOTE)
    assert label("toggle_data_saver_mode") in out and label("toggle_data") in out  # longest name first
    assert out.count("transfer_to_human_agents") == policy.count("transfer_to_human_agents") > 0  # the one agent tool it names is untouched
    assert mask_tools(tools) == list(tools)  # tau2 telecom gives the agent no phone tools


def test_defaults_off_and_mask_changes_only_the_prompt(monkeypatch, env_tools, tmp_path):
    tools, policy = env_tools
    st, _ = A.resolve_settings({})
    assert st.mask_phone_tools is False and st.reject_phone_tools == 0
    ref = LLMAgent(tools=tools, domain_policy=policy, llm="m", llm_args={"temperature": 0.0})
    off = _plain(tools, policy, tmp_path / "off.jsonl")
    assert off.system_prompt == ref.system_prompt and off.mask_info == {}
    on = _plain(tools, policy, tmp_path / "on.jsonl", boed_mask_phone_tools=True)
    assert on.system_prompt != ref.system_prompt and not _names_left(on.system_prompt)
    assert [t.name for t in on.tools] == [t.name for t in tools]
    gen, calls = recording_generate([AssistantMessage.text("Could you check your status bar and tell me what you see?")])
    monkeypatch.setattr("tau2.agent.llm_agent.generate", gen)
    run_episode(on, USER)
    assert not _names_left(calls[0]["messages"][0].content)
    rec = json.loads((tmp_path / "on.jsonl").read_text().splitlines()[0])
    assert rec["parse"]["mask"]["names_replaced"] >= 55 and rec["parse"]["mask"]["tools_dropped"] == 0


# ------------------------------------------------------------------ reject + resample
def test_reject_resample_hidden_retries(monkeypatch, env_tools, tmp_path):
    tools, policy = env_tools
    text = AssistantMessage.text("Could you please toggle airplane mode and tell me what you see?")
    gen, calls = recording_generate([tc_msg("check_network_status"), tc_msg("toggle_airplane_mode", i=1), text])
    monkeypatch.setattr("tau2.agent.llm_agent.generate", gen)
    side = tmp_path / "rej.jsonl"
    got, st = run_episode(_plain(tools, policy, side, boed_reject_phone_tools=2), USER)
    assert got[0] is text and st.messages[-1] is text
    assert not any(getattr(m, "tool_calls", None) for m in st.messages)  # rejected calls never enter the history
    assert not any(isinstance(m, ToolMessage) for m in st.messages)
    # retry 2 sees both rejected calls and both errors after the real history (cumulative side context)
    side_ctx = calls[2]["messages"][len(st.system_messages) + len(st.messages) - 1:]
    assert [type(m).__name__ for m in side_ctx] == ["AssistantMessage", "ToolMessage", "AssistantMessage", "ToolMessage"]
    assert side_ctx[1].content == REJECT_TEXT.format(tool="check_network_status") and side_ctx[1].error
    rec = json.loads(side.read_text().splitlines()[0])["parse"]
    assert [r["from_tool"] for r in rec["rejected"]] == ["check_network_status", "toggle_airplane_mode"]
    assert rec["rejected"][-1]["new_tool"] is None and "repaired" not in rec


def test_reject_then_repair_template_and_off_is_identity(monkeypatch, env_tools, tmp_path):
    tools, policy = env_tools
    gen, calls = recording_generate([tc_msg("check_sim_status"), tc_msg("check_sim_status", i=1)])
    monkeypatch.setattr("tau2.agent.llm_agent.generate", gen)
    side = tmp_path / "rr.jsonl"
    got, st = run_episode(_plain(tools, policy, side, boed_reject_phone_tools=1, boed_repair_invalid_user_tools=True), USER)
    assert len(calls) == 2 and not got[0].tool_calls and got[0].content == C.repair_text("check_sim_status", {})
    rec = json.loads(side.read_text().splitlines()[0])["parse"]
    assert len(rec["rejected"]) == 1 and rec["repaired"][0]["from_tool"] == "check_sim_status"
    # a valid first message: no retry, no record
    ok = tc_msg("get_customer_by_phone", {"phone_number": "555-123-2002"})
    gen, calls = recording_generate([ok])
    monkeypatch.setattr("tau2.agent.llm_agent.generate", gen)
    side = tmp_path / "ok.jsonl"
    got, _ = run_episode(_plain(tools, policy, side, boed_reject_phone_tools=2), USER)
    assert len(calls) == 1 and msg_dump(got[0]) == msg_dump(ok)
    assert "rejected" not in json.loads(side.read_text().splitlines()[0])["parse"]


def test_controller_rejects_reject_switch(env_tools, tmp_path):
    tools, policy = env_tools
    with pytest.raises(ValueError, match="boed-plain only"):
        A.FACTORIES["boed-controller"](tools=tools, domain_policy=policy, llm="m", task=None,
                                        llm_args={"boed_reject_phone_tools": 1, "boed_sidecar": str(tmp_path / "c.jsonl")})


# ------------------------------------------------------------------ Step 1 run configs
ARMS = {"r3m_plain": {"scorer": "reach"}, "r3m_repair": {"scorer": "reach", "repair_invalid_user_tools": True},
        "r3m_mask_repair": {"scorer": "reach", "mask_phone_tools": True, "repair_invalid_user_tools": True},
        "r3m_reject": {"scorer": "reach", "reject_phone_tools": 2, "repair_invalid_user_tools": True}}


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("n", [5, 20])
def test_step1_configs(arm, n):
    cfg = yaml.safe_load((paths.CONFIG_DIR / "runs" / f"{arm}_native_train{n}.yaml").read_text())
    base = yaml.safe_load((paths.CONFIG_DIR / "runs" / "l2x_boedplain_native_train20.yaml").read_text())
    assert cfg["name"] == f"{arm}_native_train{n}" and cfg["agent"] == "boed-plain" and cfg["agent_kwargs"] == ARMS[arm]
    assert cfg["num_tasks"] == n and cfg["eval_final"] is False
    for k in ("user", "domain", "split", "num_trials", "seed", "max_steps", "max_concurrency", "llm", "boed_config"):
        assert cfg[k] == base[k], k
    A.resolve_settings({"_boed": {"agent_kwargs": cfg["agent_kwargs"]}})  # every key is a valid BoedSettings field
