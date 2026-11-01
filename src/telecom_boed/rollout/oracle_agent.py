"""[telecom_boed] PRIVILEGED scripted teacher agent `oracle-teacher` (P0-5). NOT deployable.

Reads the hidden task (env/oracle_hidden: true_key, gold_actions) and plays the reference repair as a tau2 agent:

    look up the customer -> diagnose (status bar) -> for each gold action in gold order:
        requestor "assistant": call the backend tool itself with the gold arguments
        requestor "user":      ask the user, in plain text, to run that phone tool (the agent never calls phone tools)
    -> verify (one read-only phone check chosen by intent) -> claim "fixed".
    ACTION tasks (reward_basis has ACTION; the gold action is `transfer_to_human_agents`): look up, diagnose, call
    transfer_to_human_agents(summary) and say so. Nothing is "repaired": the grader wants the transfer (name-only match).

Imperfect LLM user: after each ask the reply is checked with a cheap lexical test (`reply_ok`). A confused / empty reply is
re-asked up to `max_reasks` times (default 2) with a fixed rephrase; then the script proceeds regardless (bounded, no loops).
The script is a pure function of the task, so identical tasks give identical agent messages given identical user replies.

Use: train-safe tasks only. `register.check_oracle_allowed` (parse + check_names) and this factory both refuse otherwise, and
`testing/leakage_scan` forbids deployable modules from importing this module. Its output is teacher data (P1 ceiling, P5 SFT).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import Field
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall

from telecom_boed.contracts import DeviceAction, _frozen_items
from telecom_boed.env.oracle_hidden import gold_actions, true_key
from telecom_boed.tau2_ext import register as _register
from telecom_boed.tau2_ext.candidates import repair_text

NAME = "oracle-teacher"
TRANSFER_TOOL = "transfer_to_human_agents"
DEFAULT_PHONE = "555-123-2002"
DEFAULT_CUSTOMER, DEFAULT_LINE = "C1001", "L1002"
DIAGNOSE_TOOL = "check_status_bar"
VERIFY_TOOL = {"mms_issue": "can_send_mms", "mobile_data_issue": "run_speed_test", "service_issue": "check_status_bar"}
CLOSING = ("Everything on your phone should be working now, so I consider your issue fixed. "
           "Is there anything else I can help you with?")
TRANSFER_SAY = "I have transferred you to a human agent who can take this further. Thank you for your patience."
# stop_after_close: the user sim often says goodbye without ###STOP###; ending the chat ourselves avoids a max_steps loop.
STOP_TOKEN = "###STOP###"  # tau2 LLMSoloAgent.STOP_TOKEN; LLMAgent itself never stops
GOODBYE = "Thank you for contacting us. Goodbye! " + STOP_TOKEN
_CONFUSED = re.compile(
    r"(not sure|don'?t know|do not know|how do i|where (do|is|can)|what do you mean|can'?t find|cannot find|unable to|"
    r"confus|which (one|setting)|could you (explain|clarify)|i don'?t see|no idea|didn'?t understand|not able to)", re.I)


def ask_text(tool: str, args: dict) -> str:
    """User-facing request for phone tool `tool`. candidates.repair_text, plus the grant_app_permission argument the shared
    phrase omits (the teacher knows which permission; the user cannot guess)."""
    if tool == "grant_app_permission" and args.get("permission"):
        return (f"Could you please grant the app '{args.get('app_name') or 'the relevant app'}' the "
                f"'{args['permission']}' permission and tell me exactly what you see?")
    return repair_text(tool, args)


def reask_text(text: str, attempt: int) -> str:
    lead = ("Let me put that another way. If you have already done this, just tell me what your phone shows now. "
            if attempt == 1 else "Just this one step, please. If it is already done, tell me what you see now. ")
    return lead + text


def reply_ok(text: Optional[str]) -> bool:
    """Cheap lexical test that the user answered the request (not that they did the right thing)."""
    t = (text or "").strip()
    return bool(t) and (("###" in t) or not _CONFUSED.search(t))


def bill_paid(content: Optional[str]) -> bool:
    """True if a get_details_by_id bill reply shows status Paid. Unreadable replies count as paid (never loop on them)."""
    try:
        return json.loads(content or "").get("status") == "Paid"
    except (ValueError, AttributeError):
        return True


PERM_CHECK_TOOL = "check_app_permissions"
_NEG_BEFORE = r"(n'?t|not|no|never|missing|without|lacks?)\b[^.!?]{0,40}"


def perm_check_text(app: str) -> str:
    return f"Could you please check which permissions the '{app}' app has now and read me the exact list?"


def perm_regrant_text(app: str, perm: str) -> str:
    return (f"It looks like the '{perm}' permission is not granted yet. It will not show in the list until you grant it. "
            f"Could you please use the option to grant a permission and grant the '{app}' app the '{perm}' permission?")


def perm_granted(text: Optional[str], perm: str) -> bool:
    """Cheap lexical test that the user's reply shows `perm` as granted: the word appears and is not negated just before it
    ("I don't see the 'sms' permission"). Visible text only. False on doubt; re-asking a grant is harmless (it is idempotent)."""
    t, p = (text or "").lower(), re.escape(perm.lower())
    if not re.search(rf"\b{p}\b", t):
        return False
    return not re.search(_NEG_BEFORE + rf"\b{p}\b", t)


@dataclass(frozen=True)
class Step:
    kind: str  # "call" (backend tool call) | "ask" (text request for a phone tool) | "say" (closing / transfer text)
    tool: str = ""
    args: tuple = ()
    text: str = ""
    why: str = ""  # lookup | diagnose | gold | verify | transfer | claim

    def args_dict(self) -> dict:
        return dict(self.args)

    @property
    def action(self) -> Optional[DeviceAction]:
        """The env action this step stands for (None for plain text)."""
        if self.kind == "say":
            return None
        return DeviceAction("assistant" if self.kind == "call" else "user", self.tool, self.args)


def _call(tool: str, args: dict, why: str) -> Step:
    return Step("call", tool, _frozen_items(args), why=why)


def _ask(tool: str, args: dict, why: str) -> Step:
    return Step("ask", tool, _frozen_items(args), ask_text(tool, args), why)


def _task_dict(task: Any) -> dict:
    return task.model_dump() if hasattr(task, "model_dump") else task


def build_plan(task: Any, verify_payment: bool = False, verify_permissions: bool = False) -> list[Step]:
    """The teacher's full script for `task` (tau2 Task or tasks.json row). Pure and deterministic.
    verify_payment: after the make_payment request, read the bill back (get_details_by_id) so the agent can re-ask if unpaid.
    verify_permissions: after each grant_app_permission request, ask the user to read the app's permissions back, so the
    agent can re-ask the grant if the permission is not listed."""
    d = _task_dict(task)
    key = true_key(d)
    gold = gold_actions(d)
    basis = d["evaluation_criteria"].get("reward_basis") or []
    scen = d.get("user_scenario") or {}
    instr = scen.get("instructions") or {}
    known = instr.get("known_info") if isinstance(instr, dict) else str(instr)
    m = re.search(r"\d{3}-\d{3}-\d{4}", known or "")
    phone = m.group(0) if m else DEFAULT_PHONE
    cid, lid = DEFAULT_CUSTOMER, DEFAULT_LINE
    for a in gold:
        if a.requestor == "assistant":
            ad = a.args_dict()
            cid, lid = ad.get("customer_id", cid), ad.get("line_id", lid)
    plan = [_call("get_customer_by_phone", {"phone_number": phone}, "lookup"),
            _call("get_details_by_id", {"id": lid}, "lookup"),
            _ask(DIAGNOSE_TOOL, {}, "diagnose")]
    if "ACTION" in basis:  # transfer task: do what the grader expects and nothing else
        t = next((a for a in gold if a.tool == TRANSFER_TOOL), None)
        summary = (t.args_dict().get("summary") if t else None) or "I cannot fix the issue."
        plan += [_call(TRANSFER_TOOL, {"summary": summary}, "transfer"), Step("say", text=TRANSFER_SAY, why="transfer")]
        return plan
    bill = next((a.args_dict().get("bill_id") for a in gold if a.tool == "send_payment_request"), None)
    for a in gold:
        plan.append(_call(a.tool, a.args_dict(), "gold") if a.requestor == "assistant" else _ask(a.tool, a.args_dict(), "gold"))
        if verify_payment and a.tool == "make_payment" and (a.args_dict().get("bill_id") or bill):
            plan.append(_call("get_details_by_id", {"id": a.args_dict().get("bill_id") or bill}, "verify_payment"))
        if verify_permissions and a.tool == "grant_app_permission" and a.requestor == "user":
            app = a.args_dict().get("app_name") or "messaging"
            plan.append(Step("ask", PERM_CHECK_TOOL, _frozen_items({"app_name": app}), perm_check_text(app), "verify_permission"))
    plan += [_ask(VERIFY_TOOL.get(key[0], DIAGNOSE_TOOL), {}, "verify"), Step("say", text=CLOSING, why="claim")]
    return plan


def plan_actions(plan: list[Step]) -> tuple[DeviceAction, ...]:
    """Every env action the plan stands for, in script order (lookups and read-only checks included)."""
    return tuple(s.action for s in plan if s.action is not None)


# --------------------------------------------------------------------------- the tau2 agent
class OracleAgentState(LLMAgentState):
    """LLMAgentState + the script pointer (not serialised)."""
    oracle: Any = Field(default=None, exclude=True)


@dataclass
class _Run:
    plan: list
    max_reasks: int
    i: int = -1  # index of the last emitted step (-1: nothing yet)
    reasks: int = 0
    pay_retries: int = 0
    perm_retries: dict = field(default_factory=dict)  # verify_permission step index -> grant re-asks so far
    trace: list = field(default_factory=list)


class OracleTeacherAgent(LLMAgent):
    """Scripted teacher. Subclasses LLMAgent only for tau2's stop/transfer plumbing and state; it never calls an LLM."""

    def __init__(self, tools: list, domain_policy: str, task: Any, llm: Optional[str] = None, llm_args: Optional[dict] = None,
                 max_reasks: int = 2, stop_after_close: bool = False, no_toggle_reask: bool = False,
                 verify_payment: bool = False, verify_permissions: bool = False):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm or "none", llm_args={})
        d = _task_dict(task)
        if not _register.train_safe([d["id"]]):
            raise _register.ConfigError(f"{NAME}: task {d['id']!r} is not train-safe (test-40 id or fault set)")
        self.task_id, self.max_reasks = d["id"], int(max_reasks)
        self.stop_after_close = bool(stop_after_close)
        # no_toggle_reask: a re-asked toggle_* step is often done twice by the user sim, which undoes it.
        self.no_toggle_reask, self.verify_payment = bool(no_toggle_reask), bool(verify_payment)
        self.verify_permissions = bool(verify_permissions)
        self.plan = build_plan(d, verify_payment=self.verify_payment, verify_permissions=self.verify_permissions)
        self.last_run: Optional[_Run] = None

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        """True only for GOODBYE, which is sent only when stop_after_close is on (tau2 then ends with AGENT_STOP)."""
        return bool(getattr(message, "content", None)) and STOP_TOKEN in message.content

    def get_init_state(self, message_history: Optional[list] = None) -> OracleAgentState:
        base = super().get_init_state(message_history)
        st = OracleAgentState.model_construct(system_messages=base.system_messages, messages=base.messages)
        st.oracle = self.last_run = _Run(self.plan, self.max_reasks)
        return st

    def _emit(self, run: _Run, i: int, text: Optional[str] = None, *, reask: int = 0) -> AssistantMessage:
        s = run.plan[i]
        meta = {"oracle_step": i, "kind": s.kind, "why": s.why, "reask": reask}
        run.i = i
        run.trace.append(meta)
        if s.kind == "call":
            tc = ToolCall(id=f"oracle_{i}", name=s.tool, arguments=s.args_dict(), requestor="assistant")
            return AssistantMessage(role="assistant", content=None, tool_calls=[tc], raw_data=meta)
        return AssistantMessage(role="assistant", content=text if text is not None else s.text, raw_data=meta)

    def generate_next_message(self, message: Any, state: Any) -> tuple[AssistantMessage, Any]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        run: _Run = state.oracle
        last = run.plan[run.i] if run.i >= 0 else None
        if last is None:
            nxt = 0
        elif last.why == "verify_permission":  # replaces the generic re-ask: "I don't see sms" means re-grant, not re-check
            grant = run.plan[run.i - 1].args_dict()
            tries = run.perm_retries.get(run.i, 0)
            if tries < run.max_reasks and not perm_granted(getattr(message, "content", None), grant.get("permission", "")):
                run.perm_retries[run.i] = tries + 1
                text = perm_regrant_text(grant.get("app_name") or "messaging", grant.get("permission", ""))
                msg = self._emit(run, run.i - 1, text, reask=tries + 1)
                state.messages.append(msg)
                return msg, state
            nxt = run.i + 1
        elif (last.kind == "ask" and not reply_ok(getattr(message, "content", None)) and run.reasks < run.max_reasks
              and not (self.no_toggle_reask and last.tool.startswith("toggle_"))):
            run.reasks += 1
            msg = self._emit(run, run.i, reask_text(last.text, run.reasks), reask=run.reasks)
            state.messages.append(msg)
            return msg, state
        elif (last.kind == "call" and last.why == "verify_payment" and run.pay_retries < run.max_reasks
              and not bill_paid(getattr(message, "content", None))):  # user said they paid but the bill is unpaid
            run.pay_retries += 1
            ask = run.plan[run.i - 1]
            msg = self._emit(run, run.i - 1, reask_text(ask.text, run.pay_retries), reask=run.pay_retries)
            state.messages.append(msg)
            return msg, state
        elif self.stop_after_close and run.i == len(run.plan) - 1:  # user answered the final text: end the chat
            meta = {"oracle_step": run.i + 1, "kind": "stop", "why": "stop_after_close", "reask": 0}
            run.trace.append(meta)
            msg = AssistantMessage(role="assistant", content=GOODBYE, raw_data=meta)
            state.messages.append(msg)
            return msg, state
        else:
            nxt = min(run.i + 1, len(run.plan) - 1)  # past the end: repeat the final text
        run.reasks = 0
        msg = self._emit(run, nxt)
        state.messages.append(msg)
        return msg, state


def _make(tools: list, domain_policy: str, llm: Optional[str] = None, llm_args: Optional[dict] = None, task: Any = None,
          **kwargs: Any) -> OracleTeacherAgent:
    del kwargs
    if task is None:
        raise _register.ConfigError(f"{NAME} needs the episode's task")
    _, boed = _register.split_boed_args(llm_args)
    kw = dict(boed.get("agent_kwargs") or {})
    bad = set(kw) - {"max_reasks", "stop_after_close", "no_toggle_reask", "verify_payment", "verify_permissions"}
    if bad:
        raise _register.ConfigError(f"{NAME}: unknown agent_kwargs {sorted(bad)}")
    return OracleTeacherAgent(tools, domain_policy, task, llm, max_reasks=kw.get("max_reasks", 2),
                              stop_after_close=kw.get("stop_after_close", False),
                              no_toggle_reask=kw.get("no_toggle_reask", False), verify_payment=kw.get("verify_payment", False),
                              verify_permissions=kw.get("verify_permissions", False))


FACTORIES = {NAME: _make}


def register(registry: Any = None) -> list[str]:
    """Register `oracle-teacher` (idempotent). Refusal under eval_final / non-train-safe splits is in register.check_oracle_allowed
    (run config parse + check_names) and in the factory (per-task)."""
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
