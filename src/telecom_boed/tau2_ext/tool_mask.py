"""[telecom_boed] Phone-tool masking for the LLM agent (Step 1, 2026-11-01). DEPLOYABLE module: it reads only the agent's own policy
text, its tool list and the message it just generated.

tau2 telecom never gives the agent phone tools, but the agent policy (tech_support_manual.md) names them, e.g. **check_status_bar**,
so Qwen calls them and tau2 answers "Tool not found". Two opt-in switches (BoedSettings; both off by default):

- `mask_phone_tools` (prompt mask): drop any phone tool from the agent's tool list (a no-op in tau2 telecom, kept as a guard) and,
  in the agent's system prompt only, rewrite each phone-tool name as "[customer action: check status bar]", plus one closing note
  saying these steps are done by the customer. The user simulator, the tasks and the grader are unchanged.
- `reject_phone_tools` = N > 0 (reject + resample, boed-plain only): when the agent emits a phone-tool call, the call and a short
  error are shown to the model in a hidden side context and it is asked again, up to N times. Hidden means the rejected calls never
  enter the chat history or reach the user. If it still calls a phone tool, the repair template (repair_invalid_user_tools) takes
  over when that switch is on; otherwise the last call goes out as tau2 would send it.
"""
from __future__ import annotations

import re
from typing import Iterable

from telecom_boed.env.featurize import USER_TOOLS

MASK_NOTE = ("\n\nNOTE ON DEVICE ACTIONS: every step written as [customer action: ...] in this policy is done by the customer on "
             "their own phone. It is not a tool you can call. Ask the customer in plain text to do it and to tell you what they see.\n")
REJECT_TEXT = ("Error: '{tool}' is an action the customer performs on their own phone. It is not a tool you can call. "
               "Ask the customer in plain text to do it and to tell you what they see.")


def phone_tools(agent_tools: Iterable[str] = ()) -> tuple[str, ...]:
    """Phone (user) tool names the agent does not own, longest first (so toggle_data_saver_mode wins over toggle_data)."""
    own = set(agent_tools)
    return tuple(sorted((t for t in USER_TOOLS if t not in own), key=len, reverse=True))


def label(tool: str) -> str:
    return f"[customer action: {tool.replace('_', ' ')}]"


def mask_policy(policy: str, agent_tools: Iterable[str] = ()) -> tuple[str, int]:
    """(masked policy, number of names replaced). Backticks around a name are dropped with it; markdown bold is kept."""
    names = phone_tools(agent_tools)
    if not names:
        return policy + MASK_NOTE, 0
    pat = re.compile(r"`?\b(" + "|".join(map(re.escape, names)) + r")\b`?")
    out, n = pat.subn(lambda m: label(m.group(1)), policy)
    return out + MASK_NOTE, n


def mask_tools(tools: list) -> list:
    """The agent's tools without phone tools (tau2 telecom: unchanged)."""
    ph = set(USER_TOOLS)
    return [t for t in tools if getattr(t, "name", None) not in ph]
