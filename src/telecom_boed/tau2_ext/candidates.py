"""[telecom_boed] Candidate sampling + parsing for the BOED controller agent (owner: W3-S11). DEPLOYABLE module.

Why not tau2's `generate()`: it reads `choices[0]` only and calls `json.loads` on tool-call arguments with no guard
(tau2/utils/llm_utils.py:423,440), so one malformed Qwen tool call kills the episode. Here we call `litellm.completion(n=N)`
directly, convert messages with tau2's own `to_litellm_messages` (llm_utils.py:168), and parse each choice defensively.

- `sample_candidates(...)` -> `Samples` (a list of AssistantMessage plus counters). Dropped choices: malformed tool-arg JSON
  (`n_malformed`), neither text nor tool call (`n_empty`). `Samples.indices[i]` = original choice index of kept message i.
  The whole completion's cost/usage are attached to `Samples` (the caller puts them on the message it emits).
- `to_candidate(msg, parsers, candidate_idx)` -> CandidateAction: transfer tool call -> TRANSFER; other backend tool call ->
  AGENT_TOOL (DeviceAction requestor="assistant"); text -> classify_agent_message: REQUEST_DEVICE (parsed DeviceActions) / ASK /
  CLAIM_RESOLVED (claim prob >= 0.5). Text that merely SAYS it will transfer is demoted to ASK: only the tool call transfers.
- `repair_user_tool_call(...)` (W3-F1, opt-in `repair_invalid_user_tools`): an agent tool call whose name is a PHONE tool the agent does not
  own (Qwen3-8B's dominant failure) becomes a text request to the user + a REQUEST_DEVICE candidate (DeviceAction requestor="user").
Never reads the episode's task or databases; only the agent-visible messages and the public tool schema.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Mapping, Optional, Sequence

import litellm

from tau2.config import DEFAULT_MAX_RETRIES
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.utils.llm_utils import get_response_cost, get_response_usage, to_litellm_messages, validate_message_history

from telecom_boed.belief.parsers import classify_agent_message
from telecom_boed.contracts import CandidateAction, DecisionKind, DeviceAction, _frozen_items
from telecom_boed.env.featurize import USER_TOOLS

TRANSFER_TOOL = "transfer_to_human_agents"


class Samples(list):
    """list[AssistantMessage] (kept candidates) + sampling bookkeeping."""

    def __init__(self, *a: Any) -> None:
        super().__init__(*a)
        self.n_requested = 0
        self.n_malformed = 0
        self.n_empty = 0
        self.n_truncated = 0  # finish_reason == 'length': a cut-off message is not an answer (bimanual, 2026-10-16)
        self.indices: list[int] = []  # original choice index of each kept message
        self.cost: Optional[float] = None
        self.usage: Optional[dict] = None
        self.seconds = 0.0

    def stats(self) -> dict:
        return {"n_requested": self.n_requested, "n_kept": len(self), "n_malformed": self.n_malformed,
                "n_empty": self.n_empty, "n_truncated": self.n_truncated, "indices": list(self.indices)}


def _choice_to_message(choice: Any) -> tuple[Optional[AssistantMessage], str]:
    """(message, "") or (None, reason in {"malformed", "empty", "role"})."""
    m = choice.message
    if getattr(m, "role", "assistant") != "assistant":
        return None, "role"
    content = m.content
    calls: list[ToolCall] = []
    for tc in (getattr(m, "tool_calls", None) or []):
        try:
            args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
        except (ValueError, TypeError):
            return None, "malformed"
        if not isinstance(args, dict) or not getattr(tc.function, "name", None):
            return None, "malformed"
        calls.append(ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:12]}", name=tc.function.name, arguments=args, requestor="assistant"))
    if not calls and not (content or "").strip():
        return None, "empty"
    return AssistantMessage(role="assistant", content=content, tool_calls=calls or None), ""


def sample_candidates(model: str, messages: Sequence[Any], tools: Optional[Sequence[Any]], n: int, temperature: float,
                      llm_args: Optional[Mapping[str, Any]] = None, max_tokens: Optional[int] = None) -> Samples:
    """N sampled assistant messages from ONE `litellm.completion(n=N)` call. `llm_args` are the agent's litellm kwargs (any
    `temperature`/`n` in them is overridden). Raises only if the completion call itself fails; per-choice problems drop the
    choice and are counted."""
    validate_message_history(list(messages))  # same check generate() runs (llm_utils.py)
    kw = {k: v for k, v in (llm_args or {}).items() if k not in ("n", "temperature")}
    if max_tokens is not None:
        kw["max_tokens"] = int(max_tokens)  # decode length dominates candidate latency; capped per agent_kwargs
    if kw.get("num_retries") is None:
        kw["num_retries"] = DEFAULT_MAX_RETRIES  # same default generate() applies
    schema = [t.openai_schema for t in tools] if tools else None
    t0 = time.perf_counter()
    resp = litellm.completion(model=model, messages=to_litellm_messages(list(messages)), tools=schema,
                              tool_choice="auto" if schema else None, n=int(n), temperature=float(temperature), **kw)
    out = Samples()
    out.seconds = time.perf_counter() - t0
    choices = list(resp.choices)
    out.n_requested = int(n)
    for i, ch in enumerate(choices):
        if getattr(ch, "finish_reason", None) == "length":
            out.n_truncated += 1
            continue
        msg, why = _choice_to_message(ch)
        if msg is None:
            if why == "malformed":
                out.n_malformed += 1
            else:
                out.n_empty += 1
            continue
        out.append(msg)
        out.indices.append(i)
    try:
        out.cost = float(get_response_cost(resp))
    except Exception:
        out.cost = None
    try:
        out.usage = get_response_usage(resp)
    except Exception:
        out.usage = None
    return out


def to_candidate_info(msg: AssistantMessage, parsers: Any, candidate_idx: Optional[int] = None,
                      tool_names: Optional[Sequence[str]] = None) -> tuple[CandidateAction, dict]:
    """(CandidateAction, parse info dict). `tool_names`: the agent's backend tool names; a tool call outside it is flagged
    `invalid_tool` (the call would only produce a tau2 error)."""
    info: dict = {"idx": candidate_idx}
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        info.update(tool=tc.name, n_tool_calls=len(msg.tool_calls))
        if tc.name == TRANSFER_TOOL:
            info["kind"] = DecisionKind.TRANSFER.value
            return CandidateAction(DecisionKind.TRANSFER, (), candidate_idx, msg.content), info
        if tool_names is not None and tc.name not in tool_names:
            info["invalid_tool"] = True
        act = DeviceAction("assistant", tc.name, _frozen_items(tc.arguments))
        info["kind"] = DecisionKind.AGENT_TOOL.value
        return CandidateAction(DecisionKind.AGENT_TOOL, (act,), candidate_idx, msg.content), info
    text = msg.content or ""
    kind = classify_agent_message(text, parsers.action, parsers.claim)
    if kind == DecisionKind.TRANSFER:  # words only: tau2 transfers on the tool call, so this is just a message
        kind = DecisionKind.ASK
        info["text_transfer_demoted"] = True
    acts = tuple(parsers.action.parse(text)) if kind == DecisionKind.REQUEST_DEVICE else ()
    info.update(kind=kind.value, claim_prob=round(float(parsers.claim.prob_claim(text)), 4), n_device_actions=len(acts))
    return CandidateAction(kind, acts, candidate_idx, text), info


def to_candidate(msg: AssistantMessage, parsers: Any, candidate_idx: Optional[int] = None) -> CandidateAction:
    return to_candidate_info(msg, parsers, candidate_idx)[0]


# --------------------------------------------------------------------------- W3-F1: invalid user-tool repair
# Imperative description of each phone tool (tau2 user tool docstrings, shortened); `{app_name}` / `{mode}` come from the call's args.
_REQUEST_PHRASE: dict[str, str] = {
    "check_status_bar": "look at the status bar at the top of your screen",
    "check_network_status": "check your phone's network status",
    "check_network_mode_preference": "check which network mode your phone prefers",
    "set_network_mode_preference": "set your preferred network mode to '{mode}'",
    "run_speed_test": "run a speed test on your phone",
    "toggle_airplane_mode": "switch Airplane Mode to the opposite setting (on if it is off, off if it is on)",
    "check_sim_status": "check your SIM card status",
    "reseat_sim_card": "remove and re-insert your SIM card",
    "toggle_data": "switch Mobile Data to the opposite setting (on if it is off, off if it is on)",
    "toggle_roaming": "switch Data Roaming to the opposite setting (on if it is off, off if it is on)",
    "check_data_restriction_status": "check whether Data Saver mode is on",
    "toggle_data_saver_mode": "switch Data Saver mode to the opposite setting (on if it is off, off if it is on)",
    "check_apn_settings": "check your APN settings",
    "set_apn_settings": "apply the APN settings I give you",
    "reset_apn_settings": "reset your APN settings to the default",
    "check_wifi_status": "check your Wi-Fi status",
    "toggle_wifi": "switch Wi-Fi to the opposite setting (on if it is off, off if it is on)",
    "check_wifi_calling_status": "check whether Wi-Fi Calling is on",
    "toggle_wifi_calling": "switch Wi-Fi Calling to the opposite setting (on if it is off, off if it is on)",
    "check_vpn_status": "check whether a VPN is connected",
    "connect_vpn": "connect to your VPN",
    "disconnect_vpn": "disconnect your VPN",
    "check_installed_apps": "list the apps installed on your phone",
    "check_app_status": "check the status of the app '{app_name}'",
    "check_app_permissions": "check which permissions the app '{app_name}' has",
    "grant_app_permission": "grant the app '{app_name}' the permission I name",
    "can_send_mms": "try sending a picture message (MMS)",
    "reboot_device": "restart your phone",
    "check_payment_request": "check whether you have received a payment request",
    "make_payment": "pay the bill I sent you",
}
REQUEST_TEMPLATE = "Could you please {phrase} and tell me exactly what you see?"


def repair_text(tool: str, args: Mapping[str, Any]) -> str:
    """The user-facing request for phone tool `tool` (fixed template; `{x}` slots filled from `args`, 'the relevant app' etc. if absent)."""
    phrase = _REQUEST_PHRASE.get(tool, f"run '{tool}' on your phone")
    fill = {"app_name": args.get("app_name") or "the relevant app", "mode": args.get("mode") or "4g_5g_preferred"}
    return REQUEST_TEMPLATE.format(phrase=phrase.format(**fill))


def is_invalid_user_tool_call(msg: AssistantMessage, tool_names: Optional[Sequence[str]]) -> bool:
    """First tool call is a phone tool the agent does not own (so tau2 would only answer 'Tool ... not found')."""
    if not msg.tool_calls or tool_names is None:
        return False
    n = msg.tool_calls[0].name
    return n in USER_TOOLS and n not in tool_names


def repair_user_tool_call(msg: AssistantMessage, candidate_idx: Optional[int], tool_names: Optional[Sequence[str]]
                          ) -> Optional[tuple[AssistantMessage, CandidateAction, dict]]:
    """None unless `msg` is an invalid user-tool call. Else (text message asking the user, its REQUEST_DEVICE candidate, sidecar
    entry {idx, from_tool, to_request}). The message carries the original's cost/usage metadata and drops its tool calls."""
    if not is_invalid_user_tool_call(msg, tool_names):
        return None
    tc = msg.tool_calls[0]
    text = repair_text(tc.name, tc.arguments or {})
    act = DeviceAction("user", tc.name, _frozen_items(tc.arguments or {}))
    new = msg.model_copy(update={"content": text, "tool_calls": None})
    return (new, CandidateAction(DecisionKind.REQUEST_DEVICE, (act,), candidate_idx, text),
            {"idx": candidate_idx, "from_tool": tc.name, "to_request": {"tool": tc.name, "args": dict(tc.arguments or {}), "text": text}})
