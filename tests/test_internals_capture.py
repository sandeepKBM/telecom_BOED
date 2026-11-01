"""[telecom_boed] W5-A1: H7 prompt reconstruction + capture bookkeeping (CPU only; no torch, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from telecom_boed.internals import capture as C
from telecom_boed.internals import prompts as P

FIX = Path(__file__).parent / "fixtures" / "W5-A1" / "episode.json"


@pytest.fixture(scope="module")
def episode():
    return json.load(open(FIX))


@pytest.fixture(scope="module")
def env_inputs():
    return P.load_env_inputs()


@pytest.fixture(scope="module")
def rows(episode, env_inputs):
    policy, tools = env_inputs
    sim = {"task_id": episode["task_id"], "trial": episode["trial"], "messages": episode["messages"]}
    return list(P.episode_prompts(sim, policy, tools, run="fx"))


def test_copies_match_tau2():
    P.check_against_tau2()


def test_system_prompt_equals_tau2(rows, env_inputs):
    from tau2.agent.llm_agent import LLMAgent
    from tau2.domains.telecom.environment import get_environment

    env = get_environment()
    agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="x")
    for r in rows:
        assert r["messages"][0] == {"role": "system", "content": agent.system_prompt}


def test_tools_are_the_13_agent_tools(rows, env_inputs):
    from tau2.domains.telecom.environment import get_environment

    env = get_environment()
    want = [t.openai_schema for t in env.get_tools()]
    assert len(want) == P.N_AGENT_TOOLS == 13
    for r in rows:
        assert r["tools"] == want
        assert len(r["tools"]) == 13


def test_decision_points_and_history_filter(rows, episode):
    msgs = episode["messages"]
    # greeting (idx 0, no usage) is not a decision; every later assistant message with usage is
    want = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("usage")]
    assert [r["msg_idx"] for r in rows] == want == [2, 4, 6, 8, 10, 12, 16]
    assert [r["turn"] for r in rows] == list(range(len(rows)))
    assert [r["recorded_prompt_tokens"] for r in rows] == [msgs[i]["usage"]["prompt_tokens"] for i in want]
    last = rows[-1]["messages"][1:]  # after the system message
    # visible history before msg 16: user phone tool call (13) and its user-requested tool result (14) are excluded
    seen_ids = [m.get("tool_call_id") for m in last if m["role"] == "tool"]
    user_tool_ids = {tc["id"] for m in msgs if m["role"] == "user" for tc in (m.get("tool_calls") or [])}
    assert user_tool_ids and not (set(seen_ids) & user_tool_ids)
    assert len(last) == sum(P.is_agent_visible(m) for m in msgs[:16])
    assert sum(m["role"] == "tool" for m in last) == 3  # three assistant-requested tool results (idx 3, 7, 11)
    assert all(not (m["role"] == "user" and m.get("tool_calls")) for m in last)
    # roles alternate sensibly and the prompt for msg_idx k never contains msg k itself or anything later
    assert len(rows[0]["messages"]) == 1 + 2  # system + greeting + first user message


def test_openai_format(rows):
    m = [x for x in rows[-1]["messages"] if x["role"] == "assistant" and x.get("tool_calls")][0]
    tc = m["tool_calls"][0]
    assert tc["type"] == "function" and isinstance(tc["function"]["arguments"], str)
    json.loads(tc["function"]["arguments"])
    h = C.hf_messages(rows[-1]["messages"])
    tch = [x for x in h if x["role"] == "assistant" and x.get("tool_calls")][0]["tool_calls"][0]
    assert isinstance(tch["function"]["arguments"], dict)
    assert all(isinstance(x["content"], str) for x in h if x["role"] == "assistant")


def test_row_schema(rows):
    for r in rows:
        assert tuple(r) == P.PROMPT_FIELDS
        json.dumps(r)


def test_user_empty_toolcalls_list_is_not_visible():
    # tau2: is_tool_call() is True for [] -> such a user message is a phone-tool turn
    assert not P.is_agent_visible({"role": "user", "tool_calls": []})
    assert P.is_agent_visible({"role": "user", "tool_calls": None, "content": "hi"})
    assert not P.is_agent_visible({"role": "tool", "requestor": "user"})


def test_plan_anchors_prefix_chain_and_breaks():
    a = [1, 2, 3]
    b = [1, 2, 3, 4, 5]
    c = [1, 2, 3, 4, 5, 6, 7]
    d = [9, 9]
    e = [1, 2, 8]
    plan = C.plan_anchors([a, b, c, d, e])
    got = {x: m for x, m in plan}
    assert got[2] == [0, 1, 2]  # one forward over c serves a, b, c
    assert got[3] == [3] and got[4] == [4]
    assert sorted(i for _, m in plan for i in m) == [0, 1, 2, 3, 4]
    # equal sequences share an anchor; a longer *earlier* sequence is not a prefix of a shorter later one
    assert C.plan_anchors([[1, 2], [1, 2]]) == [(1, [0, 1])] or C.plan_anchors([[1, 2], [1, 2]]) == [(0, [0, 1])]
    assert len(C.plan_anchors([[1, 2, 3], [1, 2]])) == 1


def test_index_schema_and_shard_name():
    row = {"run": "r", "task_id": "[a]b|c[PERSONA:None]", "trial": 0, "turn": 3, "msg_idx": 8, "recorded_prompt_tokens": 100}
    stats = {"entropy": np.float32(1.5), "top1_prob": np.float32(0.9), "top1_id": np.int64(7)}
    rec = C.index_record(row, np.int64(100), "shards/x.safetensors", 2, 4, stats)
    for f in C.INDEX_FIELDS:
        assert f in rec
    assert rec["match_flag"] is True and rec["n_tokens"] == 100 and rec["row"] == 2
    assert C.index_record(row, 101, "s", 0, 0, stats)["match_flag"] is False
    json.dumps(rec)
    nm = C.shard_name("[a]b|c[PERSONA:None]", 1)
    assert nm.startswith("shards/") and nm.endswith("__t1.safetensors") and all(ch not in nm[7:] for ch in "[]|:")


def test_group_episodes_sorted():
    rows = [{"task_id": "t", "trial": 0, "msg_idx": 6}, {"task_id": "t", "trial": 0, "msg_idx": 2}, {"task_id": "u", "trial": 1, "msg_idx": 4}]
    g = C.group_episodes(rows)
    assert [r["msg_idx"] for r in g[("t", 0)]] == [2, 6] and list(g) == [("t", 0), ("u", 1)]


def test_shard_roundtrip(tmp_path):
    pytest.importorskip("safetensors")
    arrays = {"h": np.zeros((2, C.N_STATES, C.HIDDEN), np.float16), "entropy": np.zeros(2, np.float32),
              "top1_prob": np.zeros(2, np.float32), "top1_id": np.zeros(2, np.int64), "n_tokens": np.zeros(2, np.int64)}
    C.save_shard(tmp_path / "shards" / "a.safetensors", arrays)
    from safetensors.numpy import load_file

    got = load_file(str(tmp_path / "shards" / "a.safetensors"))
    assert set(got) == set(C.SHARD_TENSORS) and got["h"].shape == (2, 37, 4096) and got["h"].dtype == np.float16
