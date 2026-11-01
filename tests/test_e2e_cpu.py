"""[telecom_boed] W3-S10 CPU end-to-end: launch.run_config -> register -> tau2 run_domain -> grading, with NO LLM.

Test-only participants (not in src): GoldAgent replays the task's gold actions (oracle_hidden is allowed in tests only)
and ScriptedUser executes the user-side ones, then sends ###STOP### (###TRANSFER### after a transfer task).
They share one per-episode script through a thread-local set by the agent factory (it is the only builder that sees
`task`; tau2 builds agent then user in the same worker thread, runner/build.py:393-413).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import pytest

from telecom_boed import paths
from telecom_boed.contracts import DeviceAction
from telecom_boed.env import oracle_hidden
from telecom_boed.tau2_ext import launch, register

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.registry import registry
from tau2.user.user_simulator_base import STOP, TRANSFER, HalfDuplexUser, UserState

FIX = Path(__file__).parent / "fixtures" / "W3-S10" / "e2e_task_ids.json"
AGENT_NAME, USER_NAME = "test-gold-agent", "test-scripted-user"
_TL = threading.local()
SEEN_BOED: list[dict] = []  # `_boed` payloads our factory received (side-channel check)
SKIP: dict[str, int] = {}  # task_id -> index (among that task's user actions) to skip


class _Script:
    def __init__(self, task) -> None:
        acts = list(oracle_hidden.gold_actions(task))
        k = SKIP.get(task.id)
        if k is not None:
            users = [i for i, a in enumerate(acts) if a.requestor == "user"]
            del acts[users[k]]
        self.pending: list[DeviceAction] = acts
        self.transfer = any(a.tool == "transfer_to_human_agents" for a in acts)
        self.instructions = str(task.user_scenario)
        self.n = 0

    def peek(self) -> DeviceAction | None:
        return self.pending[0] if self.pending else None

    def pop_call(self) -> list[ToolCall]:
        a = self.pending.pop(0)
        self.n += 1
        return [ToolCall(id=f"t{self.n}", name=a.tool, arguments=a.args_dict(), requestor=a.requestor)]


class GoldAgent(HalfDuplexAgent):
    def __init__(self, tools, domain_policy, script: _Script) -> None:
        super().__init__(tools, domain_policy)
        self.script = script

    def get_init_state(self, message_history=None):
        return list(message_history or [])

    def generate_next_message(self, message, state):
        a = self.script.peek()
        if a is not None and a.requestor == "assistant":
            msg = AssistantMessage(role="assistant", content=None, tool_calls=self.script.pop_call())
        elif a is not None:
            msg = AssistantMessage(role="assistant", content=f"Please run {a.tool} on your phone.")
        else:
            msg = AssistantMessage(role="assistant", content="Everything is done.")
        return msg, state

    def set_seed(self, seed: int) -> None:
        pass


class ScriptedUser(HalfDuplexUser):
    def __init__(self, tools=None, instructions=None, llm=None, llm_args=None) -> None:
        super().__init__(instructions=instructions, tools=tools)
        self.script: _Script = _TL.script
        assert self.script.instructions == instructions, "agent/user built for different tasks"

    def get_init_state(self, message_history=None):
        return UserState(system_messages=[], messages=list(message_history or []))

    def generate_next_message(self, message, state):
        a = self.script.peek()
        if a is not None and a.requestor == "user":
            msg = UserMessage(role="user", content=None, tool_calls=self.script.pop_call())
        elif a is not None:
            msg = UserMessage(role="user", content="Please go ahead.")
        else:
            msg = UserMessage(role="user", content=TRANSFER if self.script.transfer else STOP)
        state.messages.append(msg)
        return msg, state

    def set_seed(self, seed: int) -> None:
        pass


def _gold_agent_factory(tools, domain_policy, **kw):
    la, boed = register.split_boed_args(kw.get("llm_args"))
    assert "extra_body" in la and "_boed" not in la
    SEEN_BOED.append(boed)
    _TL.script = _Script(kw["task"])  # test-only: real agents must never read `task`
    return GoldAgent(tools, domain_policy, _TL.script)


@pytest.fixture(scope="module", autouse=True)
def _participants():
    if AGENT_NAME not in registry.get_agents():
        registry.register_agent_factory(_gold_agent_factory, AGENT_NAME)
    if USER_NAME not in registry.get_users():
        registry.register_user(ScriptedUser, USER_NAME)


def _write_cfg(tmp: Path, name: str, ids: list[str]) -> Path:
    p = tmp / f"{name}.yaml"
    p.write_text(
        f"name: {name}\nagent: {AGENT_NAME}\nuser: {USER_NAME}\ndomain: telecom\nsplit: train\nnum_trials: 1\nseed: 300\n"
        f"max_steps: 200\nmax_concurrency: 2\ntask_ids: {json.dumps(ids)}\n"
        "llm: {model: test/none, temperature: 0.0, extra_body: {chat_template_kwargs: {enable_thinking: false}}}\n"
        "boed_config: configs/boed_default.yaml\n")
    return p


@pytest.fixture
def run(tmp_path, monkeypatch):
    saved: list[Path] = []

    def go(name: str, ids: list[str]):
        save_to = f"w3s10_{name}_{uuid.uuid4().hex[:8]}"
        run_dir = tmp_path / f"run_{name}"
        monkeypatch.setenv("TAU2_SAVE_TO", save_to)
        monkeypatch.setenv("TELECOM_RUN_DIR", str(run_dir))
        res = launch.run_config(_write_cfg(tmp_path, name, ids))
        out = Path(launch.os.environ["TAU2_DATA_DIR"]) / "simulations" / save_to
        saved.append(out)
        return res, out, run_dir

    yield go
    for d in saved:
        shutil.rmtree(d, ignore_errors=True)


def _rewards(res) -> dict[str, float]:
    return {s.task_id: s.reward_info.reward for s in res.simulations}


def test_gold_replay_reward_one_on_five_train_tasks(run, splits):
    fx = json.loads(FIX.read_text())
    ids = fx["task_ids"]
    assert len(ids) == 5 and set(ids) <= set(splits["train"]) and not set(ids) & set(splits["test"])
    assert fx["transfer"][0] in ids
    SEEN_BOED.clear()
    res, out, run_dir = run("e2e_gold", ids)
    rew = _rewards(res)
    assert set(rew) == set(ids) and all(r == 1.0 for r in rew.values()), rew
    term = {s.task_id: getattr(s.termination_reason, "value", str(s.termination_reason)) for s in res.simulations}
    assert all("user_stop" in t for t in term.values()), term
    # results land under TAU2_DATA_DIR/simulations/<save_to>/results.json
    assert (out / "results.json").is_file() and str(out).startswith(launch.os.environ["TAU2_DATA_DIR"])
    # side channel: every agent build got the `_boed` payload, stripped of litellm kwargs by split_boed_args
    assert len(SEEN_BOED) == 5 and all(b["run"] == "e2e_gold" and b["agent_kwargs"] == {} for b in SEEN_BOED)
    # run_meta.json
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["mode"] == "config" and meta["run_config"]["name"] == "e2e_gold"
    assert meta["run_config"]["sha256"] == register.sha256_text(meta["run_config"]["content"])
    assert meta["patches"] and meta["tau2_sha"] == paths.TAU2_SHA and meta["results_path"] == str(out / "results.json")
    assert AGENT_NAME in meta["registry"]["agents"] and USER_NAME in meta["registry"]["users"]
    assert meta["boed_config"]["sha256"]


def test_skipping_a_gold_user_action_gives_reward_zero(run):
    fx = json.loads(FIX.read_text())
    tid = fx["task_ids"][-1]  # the longest mixed user/assistant task
    SKIP[tid] = 0
    try:
        res, _, _ = run("e2e_skip", [tid])
    finally:
        SKIP.clear()
    assert _rewards(res) == {tid: 0.0}


def test_transfer_task_scores_via_transfer_stop(run):
    tid = json.loads(FIX.read_text())["transfer"][0]
    res, _, _ = run("e2e_transfer", [tid])
    (sim,) = res.simulations
    assert sim.reward_info.reward == 1.0
    names = [tc.name for m in sim.messages for tc in (getattr(m, "tool_calls", None) or [])]
    assert names == ["transfer_to_human_agents"]


def test_existing_results_are_not_overwritten(run, monkeypatch, tmp_path):
    tid = json.loads(FIX.read_text())["transfer"][0]
    _, out, _ = run("e2e_exists", [tid])
    monkeypatch.setenv("TAU2_SAVE_TO", out.name)  # same save_to again -> refuse (immutable runs)
    with pytest.raises(SystemExit, match="exists"):
        launch.run_config(_write_cfg(tmp_path, "e2e_exists_again", [tid]))


def test_clone_stays_clean():
    out = subprocess.run(["git", "-C", str(paths.tau2_root()), "status", "--porcelain"], capture_output=True, text=True).stdout
    assert out.strip() == ""
