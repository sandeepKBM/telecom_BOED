"""[telecom_boed] H7 stage 1: rebuild the exact prompt the tau2 LLMAgent saw at every assistant message.

Sources of truth (tau2 pinned clone 4ce7c039):
  system prompt : agent/llm_agent.py:34-41 (SYSTEM_PROMPT.format(domain_policy, agent_instruction=AGENT_INSTRUCTION))
  visible history: agent/base_agent.py:38-44 (assistant msgs, user msgs WITHOUT tool calls, tool msgs requestor=="assistant")
  message format : utils/llm_utils.py:168-208 (to_litellm_messages); tool-call arguments stay a JSON *string* here
                   (OpenAI format); capture.py parses them to dicts for the HF chat template, as vLLM does.
  tools          : [t.openai_schema for t in environment.get_tools()] (13 backend tools, none of the 30 phone tools).

Pure-dict logic (no tau2 import) so it runs and tests anywhere; `load_env_inputs()` imports tau2 lazily and needs
TAU2_DATA_DIR set first (telecom_boed.paths.prepare_tau2_data_dir).

The scripted greeting (first assistant message, usage None) is not a model call and is skipped: there is no decision
there. Every assistant message with usage is a decision point.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Iterator, Optional

# Verbatim copies of tau2/agent/llm_agent.py (checked against the clone by `check_against_tau2`).
AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

N_AGENT_TOOLS = 13

PROMPT_FIELDS = ("run", "task_id", "trial", "turn", "msg_idx", "messages", "tools", "recorded_prompt_tokens")


def system_prompt(domain_policy: str) -> str:
    return SYSTEM_PROMPT.format(domain_policy=domain_policy, agent_instruction=AGENT_INSTRUCTION)


def is_agent_visible(m: dict) -> bool:
    """tau2 `is_valid_agent_history_message` on a results.json message dict.

    UserMessage.is_tool_call() is `tool_calls is not None` (True even for []), so only a None/missing list counts as
    a plain user message.
    """
    role = m.get("role")
    if role == "assistant":
        return True
    if role == "user":
        return m.get("tool_calls") is None
    if role == "tool":
        return m.get("requestor") == "assistant"
    return False


def to_openai(m: dict) -> dict:
    """One results.json message -> litellm/OpenAI chat message (tau2 to_litellm_messages)."""
    role = m["role"]
    if role == "user":
        return {"role": "user", "content": m.get("content")}
    if role == "assistant":
        tcs = m.get("tool_calls")
        out = {"role": "assistant", "content": m.get("content"), "tool_calls": None}
        if tcs:
            out["tool_calls"] = [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments") or {})},
                    "type": "function",
                }
                for tc in tcs
            ]
        return out
    if role == "tool":
        return {"role": "tool", "content": m.get("content"), "tool_call_id": m["id"]}
    raise ValueError(f"unexpected role {role!r}")


def usage_prompt_tokens(m: dict) -> Optional[int]:
    u = m.get("usage")
    if not u:
        return None
    return u.get("prompt_tokens")


def episode_prompts(
    sim: dict, domain_policy: str, tools: list, run: str = "", skip_unmodeled: bool = True
) -> Iterator[dict]:
    """Yield one prompt row per decision point (assistant message with usage) of one simulation."""
    sys_msg = {"role": "system", "content": system_prompt(domain_policy)}
    history: list[dict] = []
    turn = 0
    for idx, m in enumerate(sim["messages"]):
        if not is_agent_visible(m):
            continue
        if m["role"] == "assistant":
            rec = usage_prompt_tokens(m)
            if rec is not None or not skip_unmodeled:
                yield {
                    "run": run,
                    "task_id": sim["task_id"],
                    "trial": sim.get("trial", 0),
                    "turn": turn,
                    "msg_idx": idx,
                    "messages": [sys_msg] + list(history),
                    "tools": tools,
                    "recorded_prompt_tokens": rec,
                }
                turn += 1
        history.append(to_openai(m))


def run_prompts(results: dict, domain_policy: str, tools: list, run: str = "") -> Iterator[dict]:
    for sim in results["simulations"]:
        yield from episode_prompts(sim, domain_policy, tools, run=run)


def load_env_inputs(domain: str = "telecom") -> tuple[str, list]:
    """(domain_policy, [openai_schema of the agent tools]) from a fresh tau2 environment. Imports tau2 here.

    tau2 `runner/build.py:108-119` passes `environment.get_tools()` and `environment.get_policy()` to the agent
    (solo_mode False). Requires TAU2_DATA_DIR to be set before the first tau2 import.
    """
    if domain != "telecom":
        raise ValueError("only telecom is wired")
    from tau2.domains.telecom.environment import get_environment  # noqa: E402  (lazy: see telecom_boed.__init__)

    env = get_environment()
    tools = [t.openai_schema for t in env.get_tools()]
    return env.get_policy(), tools


def check_against_tau2() -> None:
    """Assert our verbatim copies equal tau2's (imports tau2; call from tests / the CLI)."""
    from tau2.agent import llm_agent

    assert llm_agent.AGENT_INSTRUCTION == AGENT_INSTRUCTION, "AGENT_INSTRUCTION drifted from tau2"
    assert llm_agent.SYSTEM_PROMPT == SYSTEM_PROMPT, "SYSTEM_PROMPT drifted from tau2"


def write_jsonl(rows: Iterable[dict], path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
