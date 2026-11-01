"""P0-5: privileged scripted teacher (rollout/oracle_agent.py). CPU only, no LLM."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from tau2.data_model.message import ToolMessage, UserMessage

from telecom_boed import paths
from telecom_boed.contracts import DeviceAction
from telecom_boed.env import particles as P
from telecom_boed.env import snapshot as S
from telecom_boed.env.hypotheses import hypothesis_by_key
from telecom_boed.env.oracle_hidden import true_key
from telecom_boed.rollout import oracle_agent as O
from telecom_boed.tau2_ext import register as R
from telecom_boed.testing.leakage_scan import branch_pool_task_ids, check_deployable

_CAT, _SPLIT = paths.telecom_catalog_paths()
CATALOG = json.loads(_CAT.read_text())
BYID = {t["id"]: t for t in CATALOG}
SPLITS = json.loads(_SPLIT.read_text())
TRAIN = SPLITS["train"]
SAFE_POOL = branch_pool_task_ids(CATALOG, SPLITS["test"])
SAMPLE = TRAIN + SAFE_POOL[::40]  # all train-74 + a stratified slice of the train-safe catalog


def _final_grade(tid: str, state, transferred: bool) -> float:
    return P.grade(state, hypothesis_by_key(true_key(BYID[tid])), transferred)


# ------------------------------------------------------------------ (a) plan reaches the gold end state
@pytest.mark.parametrize("tid", SAMPLE)
def test_plan_reaches_gold_state(tid):
    task = BYID[tid]
    plan = O.build_plan(task)
    hyp = hypothesis_by_key(true_key(task))
    acts = O.plan_actions(plan)
    state, outs = P.apply_calls(P.build_state(hyp), acts)
    for a, o in zip(acts, outs):
        assert not o.startswith("Error"), (tid, a.tool, o)
    transferred = any(a.tool == O.TRANSFER_TOOL for a in acts)
    assert _final_grade(tid, state, transferred) == 1.0
    assert all(s.action.requestor == "assistant" for s in plan if s.kind == "call")  # phone tools are only ever asked in text
    assert plan[-1].kind == "say"
    if "ACTION" in task["evaluation_criteria"]["reward_basis"]:
        assert [s.tool for s in plan if s.kind != "say" and s.why in ("gold", "transfer")] == [O.TRANSFER_TOOL]
    assert plan == O.build_plan(task)  # deterministic


def test_gold_trajectory_state_matches_plan_state():
    """Cross-check with oracle_hidden: gold-only replay and plan replay end in the same state hash (non-transfer task)."""
    from telecom_boed.env.oracle_hidden import gold_actions
    tid = next(t for t in TRAIN if "ACTION" not in BYID[t]["evaluation_criteria"]["reward_basis"])
    hyp = hypothesis_by_key(true_key(BYID[tid]))
    s0 = P.build_state(hyp)
    a, _ = P.apply_calls(s0, tuple(gold_actions(BYID[tid])))
    b, _ = P.apply_calls(s0, O.plan_actions(O.build_plan(BYID[tid])))
    assert a.key == b.key


# ------------------------------------------------------------------ (b) full episode with a scripted fake user
class FakeUser:
    """Executes each request verbatim on the real tau2 env (no LLM). `mode`: "ok" | "confused_first" | "never"."""

    def __init__(self, env, task, mode="ok"):
        self.env, self.mode, self.seen = env, mode, {}
        self.table = {}
        for a in O.plan_actions(O.build_plan(task)):  # user->tool map from the request text (what a user would infer)
            if a.requestor == "user":
                self.table.setdefault(O.ask_text(a.tool, a.args_dict()), []).append(a)
        self.queue = {k: list(v) for k, v in self.table.items()}
        self.executed = []

    def reply(self, text: str) -> str:
        if "Could you" in text:
            text = text[text.index("Could you"):]  # strip a re-ask lead-in
        if "fixed" in text:
            return "###STOP###"
        if "transferred you" in text:
            return "###TRANSFER###"
        self.seen[text] = self.seen.get(text, 0) + 1
        if self.mode == "never" or (self.mode == "confused_first" and self.seen[text] == 1):
            return "I'm not sure how to do that, where do I find it?"
        q = self.queue.get(text)
        a = q.pop(0) if q else (self.table[text][-1] if text in self.table else None)
        if a is None:
            return "ok"
        from tau2.data_model.message import ToolCall
        tm = self.env.get_response(ToolCall(id=f"u{len(self.executed)}", name=a.tool, arguments=a.args_dict(), requestor="user"))
        self.executed.append(a.tool)
        return f"Done. Result: {tm.content}"


def _episode(tid, mode="ok", max_steps=120, max_reasks=2):
    from tau2.registry import registry  # noqa: F401  (import check only)
    task = BYID[tid]
    hyp = hypothesis_by_key(true_key(task))
    env = S.restore(P.build_state(hyp))
    agent = O.OracleTeacherAgent([], "policy", task, max_reasks=max_reasks)
    state = agent.get_init_state()
    user = FakeUser(env, task, mode)
    msg_in = UserMessage(role="user", content=task["user_scenario"]["instructions"]["reason_for_call"])
    transferred, n_agent = False, 0
    for _ in range(max_steps):
        out, state = agent.generate_next_message(msg_in, state)
        n_agent += 1
        if out.tool_calls:
            tc = out.tool_calls[0]
            assert tc.requestor == "assistant"
            assert tc.name in {"get_customer_by_phone", "get_details_by_id", "refuel_data", "enable_roaming",
                               "send_payment_request", "resume_line", O.TRANSFER_TOOL}, tc.name  # never a phone tool
            transferred |= tc.name == O.TRANSFER_TOOL
            tm = env.get_response(tc)
            msg_in = ToolMessage(id=tc.id, role="tool", content=tm.content, requestor="assistant", error=tm.error)
            continue
        r = user.reply(out.content)
        if r.startswith("###"):
            return env, hyp, transferred, user, n_agent
        msg_in = UserMessage(role="user", content=r)
    return env, hyp, transferred, user, n_agent


def _pick(pred, n=3):
    return [t for t in TRAIN if pred(BYID[t]["evaluation_criteria"])][:n]


@pytest.mark.parametrize("tid", _pick(lambda ec: "ACTION" not in ec["reward_basis"], 4) + _pick(lambda ec: "ACTION" in ec["reward_basis"], 2))
def test_full_episode_cooperative_user_passes(tid):
    env, hyp, transferred, user, _ = _episode(tid)
    assert P.grade(S.snapshot(env), hyp, transferred) == 1.0
    if "ACTION" in BYID[tid]["evaluation_criteria"]["reward_basis"]:
        assert transferred and user.executed == ["check_status_bar"]  # nothing repaired before the transfer


def test_confused_user_is_reasked_then_passes():
    tid = _pick(lambda ec: "ACTION" not in ec["reward_basis"], 1)[0]
    env, hyp, transferred, user, _ = _episode(tid, mode="confused_first")
    assert P.grade(S.snapshot(env), hyp, transferred) == 1.0


def test_never_complying_user_is_bounded():
    tid = _pick(lambda ec: "ACTION" not in ec["reward_basis"], 1)[0]
    plan = O.build_plan(BYID[tid])
    n_asks = sum(s.kind == "ask" for s in plan)
    _, _, _, user, n_agent = _episode(tid, mode="never")
    assert user.executed == []
    assert n_agent <= len(plan) + 2 * n_asks + 2  # 2 re-asks per ask, then the script moves on and finishes


def _drive_past_end(stop_after_close, tid):
    """Answer every step with 'ok', then say goodbye (no ###STOP###) after the final text. Returns the replies."""
    agent = O.OracleTeacherAgent([], "policy", BYID[tid], stop_after_close=stop_after_close)
    state = agent.get_init_state()
    outs = []
    for _ in range(len(agent.plan) + 3):
        out, state = agent.generate_next_message(UserMessage(role="user", content="ok, done. Thanks, have a great day!"), state)
        outs.append(out)
    return agent, outs


@pytest.mark.parametrize("pred", [lambda ec: "ACTION" not in ec["reward_basis"], lambda ec: "ACTION" in ec["reward_basis"]])
def test_stop_after_close_ends_chat(pred):
    tid = _pick(pred, 1)[0]
    agent, outs = _drive_past_end(True, tid)
    n = len(agent.plan)
    assert outs[n - 1].content == agent.plan[-1].text  # final text sent once
    assert O.OracleTeacherAgent.is_stop(outs[n])  # then the teacher ends the chat
    assert agent.last_run.trace[n]["kind"] == "stop"


def test_stop_after_close_off_keeps_old_behaviour():
    tid = _pick(lambda ec: "ACTION" not in ec["reward_basis"], 1)[0]
    agent, outs = _drive_past_end(False, tid)
    n = len(agent.plan)
    assert all(o.content == agent.plan[-1].text and not O.OracleTeacherAgent.is_stop(o) for o in outs[n - 1:])


def test_factory_reads_stop_after_close():
    tid = _pick(lambda ec: "ACTION" not in ec["reward_basis"], 1)[0]
    a = O._make([], "policy", task=BYID[tid], llm_args={O._register.BOED_KEY: {"agent_kwargs": {"stop_after_close": True}}})
    assert a.stop_after_close and not O._make([], "policy", task=BYID[tid]).stop_after_close


# ------------------------------------------------------------------ (c) registration + refusal
def test_registered_and_idempotent():
    names = R.register_all()
    assert O.NAME in names or O.NAME in R.registered_names()["agents"]
    assert O.register() == [O.NAME]  # idempotent on the global registry


def _raw(**kw):
    base = dict(name="x_oracle", llm={"model": "hosted_vllm/Qwen/Qwen3-8B"}, agent=O.NAME, split="train")
    base.update(kw)
    return base


def test_config_allows_train_and_train_safe_ids():
    R.parse_run_config(_raw())
    R.parse_run_config(_raw(num_tasks=3))
    R.parse_run_config(_raw(split="base", task_ids=TRAIN[:2]))


def test_config_refuses_eval_final():
    with pytest.raises(R.ConfigError, match="eval_final"):
        R.parse_run_config(_raw(eval_final=True))
    with pytest.raises(R.ConfigError):
        R.parse_run_config(_raw(split="test", eval_final=True, task_ids=SPLITS["test"][:1]))


@pytest.mark.parametrize("split", ["test", "base", "small", "full"])
def test_config_refuses_test_bearing_split(split):
    with pytest.raises(R.ConfigError):
        R.parse_run_config(_raw(split=split))


def test_config_refuses_test_task_ids_even_with_train_split_name():
    t = SPLITS["test"][0]
    with pytest.raises(R.ConfigError):
        R.parse_run_config(_raw(split="train", task_ids=[t]))
    # a catalog task sharing a test-40 fault set under another intent is also refused
    fs = t.split("]", 1)[1].split("[", 1)[0]
    twins = [c["id"] for c in CATALOG if c["id"] not in SPLITS["test"] and c["id"].split("]", 1)[1].split("[", 1)[0] == fs]
    if twins:
        with pytest.raises(R.ConfigError):
            R.parse_run_config(_raw(split="train", task_ids=twins[:1]))


def test_check_names_refuses_even_if_spec_built_directly():
    R.register_all()
    spec = R.parse_run_config(_raw())
    R.check_names(spec)
    import dataclasses
    with pytest.raises(R.ConfigError):
        R.check_names(dataclasses.replace(spec, eval_final=True))
    with pytest.raises(R.ConfigError):
        R.check_names(dataclasses.replace(spec, split="test", task_ids=None))


def test_factory_refuses_test_task():
    t = BYID[SPLITS["test"][0]]
    with pytest.raises(R.ConfigError):
        O._make([], "policy", task=t)
    with pytest.raises(R.ConfigError):
        O._make([], "policy", task=None)
    O._make([], "policy", task=BYID[TRAIN[0]])


def test_other_agents_unaffected():
    R.parse_run_config(dict(name="y", llm={"model": "m"}, agent="llm_agent", split="train"))


# ------------------------------------------------------------------ (d) leakage rule
def test_leakage_scan_flags_deployable_importing_teacher(tmp_path):
    root = tmp_path / "src" / "telecom_boed"
    (root / "rollout").mkdir(parents=True)
    for d in (root, root / "rollout"):
        (d / "__init__.py").write_text("")
    (root / "rollout" / "oracle_agent.py").write_text("X = 1\n")
    (root / "rollout" / "privileged_thing.py").write_text("X = 1\n")
    (root / "bad_direct.py").write_text("from telecom_boed.rollout import oracle_agent\n")
    (root / "bad_import.py").write_text("import telecom_boed.rollout.oracle_agent as o\n")
    (root / "helper.py").write_text("from telecom_boed.rollout.oracle_agent import X\n")
    (root / "bad_transitive.py").write_text("from telecom_boed import helper\n")
    (root / "good.py").write_text("import json\n")
    rep = check_deployable([f"telecom_boed.{n}" for n in ("bad_direct", "bad_import", "bad_transitive", "good")], tmp_path / "src")
    assert rep.modules["telecom_boed.good"].clean
    for n in ("bad_direct", "bad_import"):
        assert any(v.kind == "oracle_import" for v in rep.modules[f"telecom_boed.{n}"].violations), n
    chains = rep.modules["telecom_boed.bad_transitive"].oracle_chains
    assert "telecom_boed.rollout.oracle_agent" in chains


def test_real_deployables_do_not_reach_teacher():
    rep = check_deployable()
    assert rep.ok, rep.problems()
    for m in rep.modules.values():
        assert not any("rollout" in k for k in m.oracle_chains)
