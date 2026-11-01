"""[telecom_boed] Agent-claim false-acceptance replay (owner: W4-S15). EVALUATION code: it reads the episode's task and the
oracle on purpose, so it must never be imported by a deployable module (leakage scanner stays clean).

For an episode's transcript (a results.json simulation):
1. find the first agent TEXT message i whose claim probability (RuleClaimClassifier by default) is >= threshold;
2. build the task's initial state (fast_env + the task's own initialization_actions, as tau2's set_state does);
3. replay every tool call in messages[:i] in order with the real tools (user-side and agent-side), exactly like tau2's
   EnvironmentEvaluator, and read `particles.grade(state, hyp, transferred)`: repaired = grade == 1.
   transferred = an assistant `transfer_to_human_agents` call occurs in messages[:i] (ACTION is name-only).
The claim message itself changes no state. The rest of the transcript is replayed too and graded as a self-check
against tau2's own reward (`final_replay_agrees`; only meaningful for user_stop / agent_stop endings).

tau2 is imported lazily (needs TAU2_DATA_DIR before the first import; `ensure_tau2()` sets it up like conftest does).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

TRANSFER_TOOL = "transfer_to_human_agents"
_SCRATCH_CANDIDATES = (Path("/data/local/ss5772/tmp"), Path("/freespace/local/ss5772/tmp"))
_MODS: dict[str, Any] = {}


def ensure_tau2() -> dict[str, Any]:
    """Set TAU2_DATA_DIR (a private copy outside the clone), apply the process patches and import the env modules once."""
    if _MODS:
        return _MODS
    try:  # tau2 logs every tool response at DEBUG through loguru; a replay makes thousands of calls
        import sys
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass
    from telecom_boed import paths
    if not os.environ.get("TAU2_DATA_DIR"):
        root = os.environ.get("TELECOM_SCRATCH")
        base = Path(root) if root else next((p / "telecom_boed" for p in _SCRATCH_CANDIDATES if p.is_dir() and os.access(p, os.W_OK)),
                                            paths.PACKAGE_ROOT / ".scratch")
        paths.prepare_tau2_data_dir(base / f"tau2_data_{paths.TAU2_SHA[:8]}")
    from telecom_boed.env import tau2_patches
    tau2_patches.apply()
    from tau2.data_model.message import ToolCall
    from tau2.data_model.tasks import EnvFunctionCall
    from telecom_boed.contracts import EnvCall, Hypothesis, _frozen_items
    from telecom_boed.env import hypotheses, particles
    from telecom_boed.env import snapshot as S
    _MODS.update(ToolCall=ToolCall, EnvFunctionCall=EnvFunctionCall, EnvCall=EnvCall, Hypothesis=Hypothesis, S=S,
                 frozen=_frozen_items, hypotheses=hypotheses, particles=particles)
    return _MODS


def first_claim(messages: Sequence[dict], classifier: Any, threshold: float = 0.5) -> Optional[tuple[int, float, str]]:
    """(message index, claim prob, text) of the first agent text message scored >= threshold, else None."""
    for i, m in enumerate(messages):
        c = m.get("content")
        if m.get("role") == "assistant" and isinstance(c, str) and c.strip() and not m.get("tool_calls"):
            p = float(classifier.prob_claim(c))
            if p >= threshold:
                return i, p, c
    return None


def _hyp_for_task(task: Optional[dict], task_id: str):
    """Hypothesis built from the episode's OWN task row (init actions, goal assertions, reward basis) so the replay does not
    depend on the catalog file; falls back to the catalog entry when the run has no task rows."""
    M = ensure_tau2()
    hyps = M["hypotheses"]
    key = hyps.parse_task_id(task_id)
    if task is None:
        return hyps.hypothesis_by_key(key)
    init_k, eval_k = "initial" + "_state", "evaluation" + "_criteria"
    ini, crit = task.get(init_k) or {}, task.get(eval_k) or {}
    if ini.get("initialization_data") or ini.get("message_history"):
        raise ValueError("task has initialization_data / message_history: replay does not model it")
    EC = M["EnvCall"]
    fz = M["frozen"]
    init = tuple(EC(a["env_type"], a["func_name"], fz(a.get("arguments"))) for a in (ini.get("initialization_actions") or []))
    goal = tuple(EC(a["env_type"], a["func_name"], fz(a.get("arguments")), a.get("assert_value", True))
                 for a in (crit.get("env_assertions") or []))
    basis = tuple(crit.get("reward_basis") or ())
    return M["Hypothesis"](key=key, task_ids=(task_id,), init_actions=init, goal_assertions=goal, reward_basis=basis,
                           requires_transfer="ACTION" in basis, goal_id=hyps.goal_id_of(goal, basis), is_test40=False)


def _initial_env(hyp):
    M = ensure_tau2()
    env = M["S"].fast_env()
    for c in hyp.init_actions:
        env.run_env_function_call(M["EnvFunctionCall"](env_type=c.env_type, func_name=c.func_name, arguments=c.args_dict()))
    return env


def _calls(m: dict) -> list:
    M = ensure_tau2()
    out = []
    for tc in m.get("tool_calls") or []:
        req = tc.get("requestor") or ("user" if m.get("role") == "user" else "assistant")
        out.append(M["ToolCall"](id=str(tc.get("id")), name=tc["name"], arguments=tc.get("arguments") or {}, requestor=req))
    return out


def replay_episode(sim: dict, task: Optional[dict], classifier: Any, threshold: float = 0.5) -> dict:
    """Claim-FA record for one episode (see module doc). status 'ok' or 'error' (never raises)."""
    try:
        M = ensure_tau2()
        S, grade = M["S"], M["particles"].grade
        msgs = sim.get("messages") or []
        hyp = _hyp_for_task(task, sim["task_id"])
        claim = first_claim(msgs, classifier, threshold)
        env = _initial_env(hyp)
        transferred = False
        at_claim: Optional[dict] = None
        for i, m in enumerate(msgs):
            if claim is not None and i == claim[0]:
                at_claim = {"repaired": grade(S.snapshot(env), hyp, transferred) == 1.0, "transferred": transferred}
            for tc in _calls(m):
                if tc.requestor == "assistant" and tc.name == TRANSFER_TOOL:
                    transferred = True
                env.get_response(tc)
        final = grade(S.snapshot(env), hyp, transferred)
        term = sim.get("termination_reason")
        tau2_reward = (sim.get("reward_info") or {}).get("reward")
        out = {"status": "ok", "has_claim": claim is not None, "claim_msg_idx": None, "claim_turn": None, "claim_prob": None,
               "repaired_at_claim": None, "transferred_at_claim": None, "claim_fa": False, "claim_text": None,
               "replay_final_grade": final, "tau2_reward": tau2_reward,
               "final_replay_agrees": (final == float(tau2_reward)) if term in ("user_stop", "agent_stop") and tau2_reward is not None else None}
        if claim is not None and at_claim is not None:
            idx, p, text = claim
            turn = sum(1 for j, mm in enumerate(msgs[:idx + 1]) if mm.get("role") == "assistant") - (
                1 if msgs and msgs[0].get("role") == "assistant" and not msgs[0].get("tool_calls") else 0)
            out.update(claim_msg_idx=idx, claim_turn=turn, claim_prob=p, repaired_at_claim=at_claim["repaired"],
                       transferred_at_claim=at_claim["transferred"], claim_fa=not at_claim["repaired"], claim_text=text[:200])
        return out
    except Exception as e:  # one bad transcript must not kill the evaluation; counted in metrics as a replay error
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "has_claim": False, "claim_fa": False}


def replay_run(sims: Sequence[dict], tasks: dict[str, dict], classifier: Any = None, threshold: float = 0.5) -> dict[str, dict]:
    """sim id -> replay_episode() record for every simulation."""
    if classifier is None:
        from telecom_boed.belief.parsers import RuleClaimClassifier
        classifier = RuleClaimClassifier()
    ensure_tau2()
    return {s["id"]: replay_episode(s, tasks.get(s.get("task_id")), classifier, threshold) for s in sims}
