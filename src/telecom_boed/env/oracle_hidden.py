"""[telecom_boed] ORACLE ONLY (owner: W1-S2). The single module allowed to read a *current* task.

Used by tests and, later, for oracle labels / evaluation bookkeeping; never imported by deployable code
(leakage scanner: any `oracle_*` import in a deployable module is a violation). Imports tau2.
Accepts a tau2 `Task` or its JSON dict (a tasks.json row).
"""
from __future__ import annotations

from typing import Any

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage

from telecom_boed.contracts import DeviceAction, HypKey, _frozen_items
from telecom_boed.env import snapshot as S
from telecom_boed.env.hypotheses import hypothesis_by_key, parse_task_id
from telecom_boed.env.particles import build_state


def _as_dict(task_or_dict: Any) -> dict:
    return task_or_dict.model_dump() if hasattr(task_or_dict, "model_dump") else task_or_dict


def true_key(task_or_dict: Any) -> HypKey:
    """The episode's hypothesis key (intent, fault set), parsed from the task id."""
    return parse_task_id(_as_dict(task_or_dict)["id"])


def gold_actions(task_or_dict: Any) -> tuple[DeviceAction, ...]:
    """The task's reference repair sequence (evaluation_criteria.actions) as DeviceActions, in order."""
    acts = _as_dict(task_or_dict)["evaluation_criteria"].get("actions") or []
    return tuple(DeviceAction(a["requestor"], a["name"], _frozen_items(a.get("arguments"))) for a in acts)


def messages_for(key: HypKey, actions: tuple[DeviceAction, ...] | list[DeviceAction]) -> list:
    """tau2 message list for `actions` run in order on hypothesis `key`'s initial env: per action one tool-call
    message (UserMessage for requestor "user", AssistantMessage for "assistant") followed by its ToolMessage from the
    real env (init actions replayed first, as tau2's set_state does; init itself is NOT a message). Tool outputs are
    recomputed, so edited action lists stay replay-consistent for tau2's strict evaluator."""
    env = S.restore(build_state(hypothesis_by_key(key)))
    msgs: list = []
    for i, a in enumerate(actions):
        tc = ToolCall(id=f"gold_{i}", name=a.tool, arguments=a.args_dict(), requestor=a.requestor)
        cls = UserMessage if a.requestor == "user" else AssistantMessage
        msgs.append(cls(role=a.requestor, content=None, tool_calls=[tc]))
        msgs.append(env.get_response(tc))
    return msgs


def gold_trajectory_messages(task_or_dict: Any) -> list:
    """messages_for() on the task's own gold actions."""
    d = _as_dict(task_or_dict)
    return messages_for(true_key(d), gold_actions(d))
