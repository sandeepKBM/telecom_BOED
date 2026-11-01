"""[telecom_boed] `boed-script`: an LLM-free, belief-driven scripted agent (Test 1, 2026-11-01). DEPLOYABLE module.

Same per-episode BOED state as boed-controller (agents._Episode: deployable ParticleFilter, parsers, Controller, sidecar), but
the candidates come from a FIXED ENUMERABLE ACTION SET instead of Qwen samples, and every message is a template:

    ask for the phone number -> get_customer_by_phone -> get_details_by_id(line) until the line's phone matches
    -> loop: candidates = device actions (decide/controller.INTENT_ACTIONS for every intent still in the belief, plus the
             permission grants, make_payment and an app-scoped permission check) + backend actions built from the ids the
             lookups returned (refuel_data, enable_roaming, resume_line, get_data_usage, bill lookups / payment requests)
             + CLAIM_RESOLVED + TRANSFER
             a = Controller.decide(belief, candidates)  (scorer from agent_kwargs: dug | reach | ...; override filter_commit
             replaces an unverified claim with a verification request)
             emit the template; the filter updates on the reply (agents._Episode.ingest)
    -> claim: closing text, then GOODBYE + ###STOP### after the user's answer. transfer: the tool call, then a closing text.

It never reads the task, the user DB or hidden state: ids come from the agent-visible tool results and the phone number from
the user's text. testing/leakage_scan lists this module as deployable.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Optional

from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage

from telecom_boed.contracts import CandidateAction, DecisionKind, DeviceAction, _frozen_items
from telecom_boed.decide.controller import INTENT_ACTIONS, template_for
from telecom_boed.tau2_ext.agents import BoedAgentState, _BoedBase, _Episode, cand_dict, decision_dict

SCRIPT = "boed-script"
TRANSFER_TOOL = "transfer_to_human_agents"
STOP_TOKEN = "###STOP###"
ASK_PHONE = ("Thank you for contacting us. To look up your account, could you please tell me the phone number of the line "
             "that has the problem?")
ASK_PHONE_AGAIN = "I could not find that number. Could you please give me the full phone number, in the format 555-123-4567?"
CLOSING = ("Everything on your phone should be working now, so I consider your issue fixed. "
           "Is there anything else I can help you with?")
TRANSFER_SAY = "I have transferred you to a human agent who can take this further. Thank you for your patience."
GOODBYE = "Thank you for contacting us. Goodbye! " + STOP_TOKEN
TRANSFER_SUMMARY = "Technical issue the agent could not resolve with the available troubleshooting steps."
PHONE_RE = re.compile(r"\b(\d{3})[-. ]?(\d{3})[-. ]?(\d{4})\b")
MAX_PHONE_ASKS = 2
MAX_LINE_LOOKUPS = 4
MAX_BILL_LOOKUPS = 3
MAX_DECISIONS = 60  # hard stop: transfer after this many BOED decisions (tau2 max_steps is the outer bound)
APP = "messaging"


def _u(tool: str, ensure: Optional[tuple] = None, **args: Any) -> DeviceAction:
    return DeviceAction("user", tool, _frozen_items(args), ensure)


def _b(tool: str, **args: Any) -> DeviceAction:
    return DeviceAction("assistant", tool, _frozen_items(args))


# Device actions the controller's INTENT_ACTIONS lacks but the catalog's fixes need (grants, payment) or need an argument
# (check_app_permissions without app_name is a tau2 error).
_EXTRA_TEXT = {
    ("grant_app_permission", "sms"): f"Could you please grant the '{APP}' app the 'sms' permission and tell me what you see?",
    ("grant_app_permission", "storage"): f"Could you please grant the '{APP}' app the 'storage' permission and tell me what you see?",
    ("make_payment", None): "I have sent a payment request to your account. Could you please check it and make the payment?",
    ("check_app_permissions", None): f"Could you please check which permissions the '{APP}' app has and read me the list?",
}
EXTRA_DEVICE = (
    (_u("check_app_permissions", app_name=APP), _EXTRA_TEXT[("check_app_permissions", None)]),
    (_u("grant_app_permission", app_name=APP, permission="sms"), _EXTRA_TEXT[("grant_app_permission", "sms")]),
    (_u("grant_app_permission", app_name=APP, permission="storage"), _EXTRA_TEXT[("grant_app_permission", "storage")]),
    (_u("make_payment"), _EXTRA_TEXT[("make_payment", None)]),
)


def find_phone(text: Optional[str]) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _json(content: Optional[str]) -> Optional[dict]:
    try:
        d = json.loads(content or "")
    except (ValueError, TypeError):
        return None
    return d if isinstance(d, dict) else None


def device_candidates(intents: list[str]) -> list[CandidateAction]:
    """Templates for INTENT_ACTIONS of `intents` (pool order, de-duplicated) + EXTRA_DEVICE. The arg-less check_app_permissions
    of INTENT_ACTIONS is replaced by the app-scoped one."""
    out, seen = [], set()
    for it in intents:
        for a in INTENT_ACTIONS.get(it, ()):
            if a.tool == "check_app_permissions" and not a.args:
                continue
            if a not in seen:
                seen.add(a)
                out.append(template_for(a))
    for a, text in EXTRA_DEVICE:
        if a not in seen:
            seen.add(a)
            out.append(CandidateAction(DecisionKind.REQUEST_DEVICE, (a,), None, text))
    return out


class _Script:
    """Per-episode script state (ids learnt from tool results, lookup phase, counters)."""

    def __init__(self) -> None:
        self.phone: Optional[str] = None
        self.phone_asks = 0
        self.tried: set = set()
        self.customer_id: Optional[str] = None
        self.line_queue: list[str] = []
        self.line_id: Optional[str] = None
        self.bill_queue: list[str] = []
        self.overdue_bills: list[str] = []
        self.awaiting: Optional[str] = None  # backend tool whose result is pending
        self.lookups_done = False
        self.line_lookups = 0
        self.bill_lookups = 0
        self.decisions = 0
        self.closed = False  # CLOSING sent: next turn says GOODBYE
        self.transferred = False


class BoedScriptAgent(_BoedBase):
    """LLM-free belief-driven agent. agent_kwargs: the BoedSettings fields (scorer, override, time_budget_ms, persona, config)."""

    def get_init_state(self, message_history: Optional[list] = None) -> BoedAgentState:
        st = super().get_init_state(message_history)
        st.boed.script = _Script()
        self._last_ep = st.boed
        return st

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        return bool(getattr(message, "content", None)) and STOP_TOKEN in message.content

    # ---- reading tool results (agent-visible)
    @staticmethod
    def _read_tool(sc: _Script, m: ToolMessage, info: dict) -> None:
        name, sc.awaiting = sc.awaiting, None
        d = None if m.error else _json(m.content)
        if name == "get_customer_by_phone":
            if d and d.get("customer_id"):
                sc.customer_id = d["customer_id"]
                sc.line_queue = [x for x in (d.get("line_ids") or [])][:MAX_LINE_LOOKUPS]
                sc.bill_queue = [x for x in (d.get("bill_ids") or [])][-MAX_BILL_LOOKUPS:]
            else:
                sc.phone = None  # ask again
        elif name == "get_details_by_id" and d:
            if d.get("line_id") and sc.line_id is None:
                if (d.get("phone_number") or "") == sc.phone or not sc.line_queue:
                    sc.line_id = d["line_id"]
            elif d.get("bill_id") and str(d.get("status", "")).lower() == "overdue":
                if d["bill_id"] not in sc.overdue_bills:
                    sc.overdue_bills.append(d["bill_id"])
        info["script"] = {"tool": name, "customer_id": sc.customer_id, "line_id": sc.line_id, "overdue_bills": list(sc.overdue_bills)}

    def _lookup_step(self, sc: _Script, user_texts: list[str]) -> Optional[tuple[str, Any]]:
        """Next lookup message, or None when the lookups are finished. -> ("ask", text) | ("call", DeviceAction)."""
        if sc.customer_id is None:
            if sc.phone is None:  # latest number the user has written (a failed lookup clears sc.phone)
                sc.phone = next((p for p in (find_phone(t) for t in reversed(user_texts)) if p), None)
            if sc.phone is None or sc.phone in sc.tried:
                sc.phone = None
                if sc.phone_asks >= MAX_PHONE_ASKS:
                    sc.lookups_done = True  # carry on with device actions only (no backend ids)
                    return None
                sc.phone_asks += 1
                return "ask", (ASK_PHONE if sc.phone_asks == 1 else ASK_PHONE_AGAIN)
            sc.tried.add(sc.phone)  # one lookup per number
            return "call", _b("get_customer_by_phone", phone_number=sc.phone)
        if sc.line_id is None and sc.line_queue:
            sc.line_lookups += 1
            return "call", _b("get_details_by_id", id=sc.line_queue.pop(0))
        sc.lookups_done = True
        return None

    def backend_candidates(self, sc: _Script) -> list[CandidateAction]:
        out: list[CandidateAction] = []
        if sc.customer_id and sc.line_id:
            c, l = sc.customer_id, sc.line_id
            for a in (_b("get_data_usage", customer_id=c, line_id=l), _b("refuel_data", customer_id=c, line_id=l, gb_amount=2.0),
                      _b("enable_roaming", customer_id=c, line_id=l), _b("resume_line", customer_id=c, line_id=l)):
                out.append(template_for(a))
        if sc.customer_id:
            for bid in sc.bill_queue:
                out.append(template_for(_b("get_details_by_id", id=bid)))
            for bid in sc.overdue_bills:
                out.append(template_for(_b("send_payment_request", customer_id=sc.customer_id, bill_id=bid)))
        return out

    def candidates(self, ep: _Episode, sc: _Script) -> list[CandidateAction]:
        probs = dict(ep.belief.intent_probs) if ep.belief is not None and ep.belief.intent_probs else {}
        intents = [k for k, v in sorted(probs.items(), key=lambda kv: -kv[1]) if v > 0.0] or list(INTENT_ACTIONS)
        cands = device_candidates(intents) + self.backend_candidates(sc)
        cands.append(CandidateAction(DecisionKind.CLAIM_RESOLVED, (), None, CLOSING))
        cands.append(CandidateAction(DecisionKind.TRANSFER, (), None, TRANSFER_SUMMARY))
        return cands

    # ---- message builders
    @staticmethod
    def _call(a: DeviceAction) -> AssistantMessage:
        tc = ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=a.tool, arguments=a.args_dict(), requestor="assistant")
        return AssistantMessage(role="assistant", content=None, tool_calls=[tc])

    def _emit(self, ch: CandidateAction, sc: _Script) -> AssistantMessage:
        if ch.kind == DecisionKind.TRANSFER:
            sc.transferred = True
            return self._call(_b(TRANSFER_TOOL, summary=ch.text or TRANSFER_SUMMARY))
        if ch.kind == DecisionKind.AGENT_TOOL:
            sc.awaiting = ch.actions[0].tool
            return self._call(ch.actions[0])
        if ch.kind == DecisionKind.CLAIM_RESOLVED:
            sc.closed = True
        return AssistantMessage(role="assistant", content=ch.text or "")

    # ---- main loop
    def generate_next_message(self, message: Any, state: Any) -> tuple[AssistantMessage, Any]:
        ep = self._episode(state)
        self._last_ep = ep
        sc: _Script = getattr(ep, "script", None) or _Script()
        ep.script = sc
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)
        parse: dict = {"error": [], "mode": "script"}
        tm = {"llm": 0.0, "filter": 0.0, "parse": 0.0, "score": 0.0}
        scores: dict = {}
        decision: dict = {}
        t = time.perf_counter()
        for m in (message.tool_messages if isinstance(message, MultiToolMessage) else [message]):
            if isinstance(m, ToolMessage) and m.requestor == "assistant" and sc.awaiting:
                try:
                    self._read_tool(sc, m, parse)
                except Exception as e:
                    parse["error"].append(f"read_tool: {type(e).__name__}: {e}")
        if not ep.init_error:
            ep.ingest(message, parse, parse["error"])
        tm["filter"] = time.perf_counter() - t
        out, cand = self._next(ep, sc, state, parse, tm, scores, decision)
        ep.note_emitted(out, cand)
        if not parse["error"]:
            parse.pop("error")
        ep.record(scores=scores, decision=decision, parse=parse, timings=tm)
        state.messages.append(out)
        return out, state

    def _next(self, ep: _Episode, sc: _Script, state: Any, parse: dict, tm: dict, scores: dict, decision: dict):
        """-> (message, the candidate it stands for or None)."""
        if sc.transferred:
            return AssistantMessage(role="assistant", content=TRANSFER_SAY), None
        if sc.closed:
            decision.update(mode="script", phase="goodbye")
            return AssistantMessage(role="assistant", content=GOODBYE), None
        if not sc.lookups_done:
            texts = [m.content or "" for m in state.messages if isinstance(m, UserMessage)]
            step = self._lookup_step(sc, texts)
            if step is not None:
                kind, x = step
                decision.update(mode="script", phase="lookup", step=kind)
                if kind == "ask":
                    return AssistantMessage(role="assistant", content=x), CandidateAction(DecisionKind.ASK, (), None, x)
                ch = template_for(x)
                return self._emit(ch, sc), ch
        if ep.init_error or ep.belief is None:
            parse["error"].append(ep.init_error or "no belief")
            ch = CandidateAction(DecisionKind.TRANSFER, (), None, TRANSFER_SUMMARY)
            decision.update(mode="script", phase="no_belief", chosen=cand_dict(ch))
            return self._emit(ch, sc), ch
        sc.decisions += 1
        cands = self.candidates(ep, sc)
        if sc.decisions > MAX_DECISIONS:
            ch = cands[-1]  # TRANSFER
            decision.update(mode="script", phase="cap", chosen=cand_dict(ch))
            return self._emit(ch, sc), ch
        ctrl = ep.controller(ep.st.override or ep.cfg["controller"]["override"])
        t = time.perf_counter()
        dec = ctrl.decide(ep.belief, cands)
        tm["score"] = time.perf_counter() - t
        scores[dec.scorer] = {str(k): v for k, v in dec.scores}
        pos = next((i for i, x in enumerate(cands) if x is dec.chosen), None)
        decision.update(decision_dict(dec, pos))
        decision.update(mode="script", phase="boed", override=ctrl.override, stats=dict(ctrl.stats), n_candidates=len(cands))
        return self._emit(dec.chosen, sc), dec.chosen


def _make(tools: list, domain_policy: str, llm: Optional[str] = None, llm_args: Optional[dict] = None, task: Any = None,
          **kwargs: Any) -> BoedScriptAgent:
    del task, kwargs  # never read the episode's task
    return BoedScriptAgent(tools=tools, domain_policy=domain_policy, llm=llm or "none", llm_args=llm_args)


FACTORIES = {SCRIPT: _make}


def register(registry: Any = None) -> list[str]:
    """Register `boed-script` (idempotent)."""
    if registry is None:
        from tau2.registry import registry
    names: list[str] = []
    for name, fac in FACTORIES.items():
        try:
            registry.register_agent_factory(fac, name)
        except ValueError:
            if registry.get_agent_factory(name) is not fac:
                raise
        names.append(name)
    return names
