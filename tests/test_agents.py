"""[telecom_boed] W3-S11 tests: boed-plain / boed-controller agents + candidate sampling (CPU only, no LLM: generate / litellm patched).

Support: five real mobile_data hypotheses that all have a TRAIN task id (same set as test_filter's `small`), injected through
`boed_filter_factory`, so no test builds the 1,000+ particle full support. Only TRAIN material is used.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import litellm
import pytest
import yaml
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message
from loguru import logger

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.telecom.environment import get_environment
from tau2.registry import Registry

from telecom_boed import paths
from telecom_boed.belief.filter import ParticleFilter
from telecom_boed.belief.parsers import build_parsers
from telecom_boed.contracts import (CONTRACT_VERSION, Belief, DecisionKind, DeviceAction, _frozen_items)
from telecom_boed.env import particles as P
from telecom_boed.env.hypotheses import load_support
from telecom_boed.tau2_ext import agents as A
from telecom_boed.tau2_ext import candidates as C
from telecom_boed.testing.leakage_scan import check_deployable
from telecom_boed.testing.poison import LEAK_LOG, PoisonDB, PoisonTask, poisoned

logger.disable("tau2")
FIX = Path(__file__).parent / "fixtures" / "W3-S11"
FAULTS = (["airplane_mode_on", "data_mode_off"], ["data_mode_off", "data_usage_exceeded"],
          ["airplane_mode_on", "bad_network_preference"], ["bad_network_preference", "user_abroad_roaming_enabled_off"],
          ["data_saver_mode_on", "data_usage_exceeded"])
WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]
VERIFY_TEXT = "Could you please run a speed test on your phone and tell me the result?"
CLAIM_TEXT = "Great news, your issue is resolved now! Is there anything else I can help you with?"


# --------------------------------------------------------------------------- helpers
class SmallFilter(ParticleFilter):
    """Real ParticleFilter whose prior is the 5-hypothesis train support above (any intent)."""

    def init(self, intent_probs):
        by = {h.faults: h for h in load_support("mobile_data_issue")}
        hyps = [by[frozenset(f)] for f in FAULTS]
        return Belief(P.initial_particles(hyps, WEIGHTS), intent_probs=_frozen_items({"mobile_data_issue": 1.0}))


class UpdateBroken(SmallFilter):
    def update(self, belief, action, obs):
        raise RuntimeError("forced filter failure")


def small_factory(cls=SmallFilter):
    return lambda cfg, persona: cls("deployable", cfg, persona)


@pytest.fixture(scope="module")
def env_tools():
    env = get_environment()
    return env.get_tools(), env.get_policy()


@pytest.fixture(autouse=True)
def _clean_defaults():
    A.reset_configuration()
    yield
    A.reset_configuration()


@pytest.fixture(scope="module")
def train_ok(splits):
    by = {h.faults: h for h in load_support("mobile_data_issue")}
    ts = set(splits["train"])
    for f in FAULTS:
        assert any(t in ts for t in by[frozenset(f)].task_ids), f"{f} has no train task"


def msg_dump(m) -> dict:
    return m.model_dump(exclude={"timestamp", "generation_time_seconds"})


def canned(env_content: str):
    """(incoming messages, factory of canned generate() outputs) from the fixture; the ToolMessage carries `env_content`."""
    spec = json.loads((FIX / "canned_plain.json").read_text())
    incoming = [UserMessage(role="user", content=spec["users"][0]),
                UserMessage(role="user", content=spec["users"][1]),
                ("tool", env_content)]

    def outs():
        res = []
        for a in spec["agent"]:
            if "text" in a:
                res.append(AssistantMessage.text(a["text"], cost=0.0))
            else:
                res.append(AssistantMessage(role="assistant", content=None, cost=0.0,
                                            tool_calls=[ToolCall(id="call_fix", name=a["tool"], arguments=a["arguments"], requestor="assistant")]))
        return res

    return incoming, outs


def fake_generate(seq):
    it = iter(seq)

    def gen(*a, **k):
        return next(it)

    return gen


def run_episode(agent, incoming):
    """Feed `incoming` (UserMessage | ("tool", content) bound to the previous tool call) to an agent; return its emitted messages."""
    st = agent.get_init_state([AssistantMessage.text("Hi! How can I help you today?")])
    out, last = [], None
    for m in incoming:
        if isinstance(m, tuple):
            m = ToolMessage(id=last.tool_calls[0].id, role="tool", content=m[1], requestor="assistant")
        last, st = agent.generate_next_message(m, st)
        out.append(last)
    return out, st


def mk_resp(choices: list[dict]) -> litellm.ModelResponse:
    cs = []
    for i, c in enumerate(choices):
        tcs = [ChatCompletionMessageToolCall(id=f"t{i}", function=Function(name=t[0], arguments=t[1])) for t in c.get("tools", [])]
        cs.append(Choices(index=i, finish_reason="stop", message=Message(role="assistant", content=c.get("text"), tool_calls=tcs or None)))
    return litellm.ModelResponse(model="qwen-test", choices=cs)


def patch_completion(monkeypatch, responses: list[litellm.ModelResponse], calls: list | None = None):
    it = iter(responses)

    def comp(**kw):
        if calls is not None:
            calls.append(kw)
        return next(it)

    monkeypatch.setattr(litellm, "completion", comp)


def env_content(tool: str, args: dict) -> str:
    st = next(iter(SmallFilter("deployable", yaml.safe_load(open(paths.CONFIG_DIR / "boed_default.yaml")), "default").init({}).particles)).state
    return P.apply_calls(st, (DeviceAction("assistant", tool, _frozen_items(args)),))[1][0]


# --------------------------------------------------------------------------- tests
def test_boed_plain_matches_llm_agent(monkeypatch, env_tools, tmp_path, train_ok):
    from tau2.agent.llm_agent import LLMAgent
    tools, policy = env_tools
    incoming, outs = canned(env_content("get_customer_by_id", {"customer_id": "C1001"}))
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate(outs()))
    ref, _ = run_episode(LLMAgent(tools=tools, domain_policy=policy, llm="m", llm_args={"temperature": 0.0}), incoming)
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate(outs()))
    side = tmp_path / "sidecar.jsonl"
    la = {"temperature": 0.0, "boed_scorer": "dug", "boed_sidecar": str(side), "boed_filter_factory": small_factory(),
          "boed_run_id": "r1", "boed_time_budget_ms": None}
    agent = A.FACTORIES["boed-plain"](tools=tools, domain_policy=policy, llm="m", llm_args=la, task=None)
    assert "boed_scorer" not in agent.llm_args and agent.llm_args == {"temperature": 0.0}  # stripped before litellm
    got, st = run_episode(agent, incoming)
    assert [msg_dump(m) for m in got] == [msg_dump(m) for m in ref]
    assert [m.timestamp is not None for m in got]
    # the shadow produced one record per turn, written as JSONL
    lines = [json.loads(x) for x in side.read_text().splitlines()]
    assert len(lines) == 3 == len(agent.records)
    for i, d in enumerate(lines, 1):
        assert d["turn"] == i and d["run_id"] == "r1" and d["contract_version"] == CONTRACT_VERSION and d["prompt_hash"]
        assert d["posterior_topk"] and abs(sum(p for _, p in d["posterior_topk"]) - 1.0) < 1e-6
        assert "error" not in d["parse"], d["parse"].get("error")
        assert d["decision"]["mode"] == "shadow" and d["scores"]["dug"] and d["timings"]["llm"] >= 0
    assert lines[0]["parse"]["emitted"]["kind"] == "request_device"
    assert lines[1]["parse"]["emitted"]["kind"] == "agent_tool" and "obs" in lines[1]["parse"]  # user report of turn 2 was ingested
    assert lines[2]["parse"]["obs"]["source"] == "agent_tool"
    assert lines[2]["parse"]["emitted"]["kind"] == "claim_resolved"
    assert st.boed.belief.step == 2  # request + backend call were filtered; the claim has no update yet


def test_boed_plain_shadow_survives_errors(monkeypatch, env_tools, tmp_path, train_ok):
    tools, policy = env_tools
    incoming, outs = canned(env_content("get_customer_by_id", {"customer_id": "C1001"}))
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate(outs()))
    ref, _ = run_episode(A.FACTORIES["boed-plain"](tools=tools, domain_policy=policy, llm="m", llm_args={"boed_filter_factory": small_factory()}), incoming)
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate(outs()))
    side = tmp_path / "s.jsonl"
    agent = A.FACTORIES["boed-plain"](tools=tools, domain_policy=policy, llm="m",
                                      llm_args={"boed_filter_factory": small_factory(UpdateBroken), "boed_sidecar": str(side)})
    got, _ = run_episode(agent, incoming)
    assert [msg_dump(m) for m in got] == [msg_dump(m) for m in ref]
    errs = [d["parse"]["error"] for d in map(json.loads, side.read_text().splitlines()) if "error" in d["parse"]]
    assert errs and any("forced filter failure" in e for es in errs for e in es)
    # a filter factory that raises disables BOED for the episode but never the agent
    def boom(cfg, persona):
        raise ValueError("no filter")
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate(outs()))
    agent2 = A.FACTORIES["boed-plain"](tools=tools, domain_policy=policy, llm="m", llm_args={"boed_filter_factory": boom})
    got2, _ = run_episode(agent2, incoming)
    assert [msg_dump(m) for m in got2] == [msg_dump(m) for m in ref]
    assert all("no filter" in r.parse["error"][0] for r in agent2.records)


def test_candidates_guarded_json_and_conversion(monkeypatch, env_tools):
    tools, _ = env_tools
    calls: list = []
    resp = mk_resp([{"text": "Could you please check your status bar?"},
                    {"tools": [("transfer_to_human_agents", '{"summary": "needs a human"}')]},
                    {"tools": [("get_customer_by_id", '{"customer_id": "C10')]},  # truncated JSON -> dropped + counted
                    {"text": CLAIM_TEXT}, ])
    patch_completion(monkeypatch, [resp], calls)
    msgs = [UserMessage(role="user", content="My data is slow.")]
    out = C.sample_candidates("m", msgs, tools, 4, 0.7, {"temperature": 0.0, "seed": 7, "n": 99})
    assert len(out) == 3 and out.n_malformed == 1 and out.n_empty == 0 and out.indices == [0, 1, 3]
    assert out.stats()["n_requested"] == 4
    kw = calls[0]
    assert kw["n"] == 4 and kw["temperature"] == 0.7 and kw["seed"] == 7 and kw["tool_choice"] == "auto"
    assert kw["messages"] == [{"role": "user", "content": "My data is slow."}] and len(kw["tools"]) == len(tools)
    assert out[1].tool_calls[0].name == "transfer_to_human_agents" and out[1].tool_calls[0].arguments == {"summary": "needs a human"}
    assert out[1].tool_calls[0].requestor == "assistant"
    # an empty choice is dropped too, and a non-object JSON argument is malformed
    patch_completion(monkeypatch, [mk_resp([{"text": ""}, {"tools": [("get_customer_by_id", "[1]")]}, {"text": "ok"}])])
    o2 = C.sample_candidates("m", msgs, None, 3, 0.7)
    assert len(o2) == 1 and o2.n_empty == 1 and o2.n_malformed == 1


def test_to_candidate_kinds():
    pb = build_parsers("rule")
    K = DecisionKind
    tc = lambda name, args, i=0: AssistantMessage(role="assistant", content=None, tool_calls=[ToolCall(id=f"c{i}", name=name, arguments=args)])
    c = C.to_candidate(tc("transfer_to_human_agents", {"summary": "x"}), pb, 2)
    assert (c.kind, c.actions, c.candidate_idx) == (K.TRANSFER, (), 2)
    c = C.to_candidate(tc("get_customer_by_id", {"customer_id": "C1001"}), pb, 1)
    assert c.kind == K.AGENT_TOOL and c.candidate_idx == 1
    assert c.actions == (DeviceAction("assistant", "get_customer_by_id", _frozen_items({"customer_id": "C1001"})),)
    c = C.to_candidate(AssistantMessage.text("Please turn Airplane Mode off and tell me what the status bar shows."), pb, 0)
    assert c.kind == K.REQUEST_DEVICE and any(a.tool == "toggle_airplane_mode" and a.ensure == ("airplane_mode", False) and a.requestor == "user" for a in c.actions)
    c = C.to_candidate(AssistantMessage.text("Which phone number is the line registered under?"), pb, 3)
    assert c.kind == K.ASK and c.actions == ()
    c = C.to_candidate(AssistantMessage.text(CLAIM_TEXT), pb, 0)
    assert c.kind == K.CLAIM_RESOLVED and c.text == CLAIM_TEXT
    # words alone do not transfer; an unknown backend tool is flagged
    c, info = C.to_candidate_info(AssistantMessage.text("I will transfer you to a human agent now."), pb)
    assert c.kind == K.ASK and info["text_transfer_demoted"]
    c, info = C.to_candidate_info(tc("toggle_airplane_mode", {}), pb, 0, tool_names=("get_customer_by_id",))
    assert c.kind == K.AGENT_TOOL and info["invalid_tool"]


def _cfg_file(tmp_path: Path, delta: float) -> str:
    cfg = yaml.safe_load((paths.CONFIG_DIR / "boed_default.yaml").read_text())
    cfg["name"] = "boed_test_delta"
    cfg["reach_avoid"]["delta"] = delta
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_controller_agent_chooses_and_overrides(monkeypatch, env_tools, tmp_path, train_ok):
    tools, policy = env_tools
    # delta=1.0 makes the heuristic scorer pick the claim although nothing is verified: the runtime filter must catch it.
    base = {"boed_scorer": "heuristic", "boed_config": _cfg_file(tmp_path, 1.0), "boed_filter_factory": small_factory(),
            "boed_n_candidates": 2, "boed_candidate_temperature": 0.7, "boed_time_budget_ms": None}
    two = [{"text": CLAIM_TEXT}, {"text": VERIFY_TEXT}]

    def episode(override: str, side: Path):
        patch_completion(monkeypatch, [mk_resp(two)])
        ag = A.FACTORIES["boed-controller"](tools=tools, domain_policy=policy, llm="m", task=PoisonTask(),
                                            llm_args={**base, "boed_override": override, "boed_sidecar": str(side)})
        st = ag.get_init_state([AssistantMessage.text("Hi! How can I help you today?")])
        m, st = ag.generate_next_message(UserMessage(role="user", content="My mobile data is very slow."), st)
        return ag, m, st

    ag, m, st = episode("filter_commit", tmp_path / "fc.jsonl")
    assert m.content == VERIFY_TEXT and not m.tool_calls
    rec = json.loads((tmp_path / "fc.jsonl").read_text().splitlines()[0])
    assert rec["decision"]["overridden"] is True and rec["decision"]["chosen"]["kind"] == "request_device"
    assert rec["decision"]["chosen"]["candidate_idx"] == 1 and rec["decision"]["mode"] == "controller"
    assert "filter_commit" in rec["decision"]["reason"] and rec["scores"]["heuristic"]
    assert [c["kind"] for c in rec["parse"]["candidates"]] == ["claim_resolved", "request_device"]
    assert rec["parse"]["sampling"]["n_kept"] == 2 and "error" not in rec["parse"] and "fallback" not in rec["parse"]
    assert m.cost is not None and st.messages[-1] is m
    # next turn: the user's reply is parsed over the particles' label space and the verify request is filtered
    patch_completion(monkeypatch, [mk_resp(two)])
    m2, st = ag.generate_next_message(UserMessage(role="user", content="The speed test says the connection is excellent."), st)
    r2 = json.loads((tmp_path / "fc.jsonl").read_text().splitlines()[1])
    assert "error" not in r2["parse"], r2["parse"].get("error")
    assert r2["parse"]["obs"]["source"] == "user_report" and r2["parse"]["belief"]["step"] == 1
    # shadow control: the same samples with override "none" emit the claim
    _, m0, _ = episode("none", tmp_path / "none.jsonl")
    assert m0.content == CLAIM_TEXT
    assert json.loads((tmp_path / "none.jsonl").read_text().splitlines()[0])["decision"]["overridden"] is False


def test_controller_agent_failure_ladder(monkeypatch, env_tools, tmp_path, train_ok):
    tools, policy = env_tools
    mk = lambda **kw: A.FACTORIES["boed-controller"](tools=tools, domain_policy=policy, llm="m",
                                                     llm_args={"boed_filter_factory": small_factory(), "boed_n_candidates": 2, **kw})
    user = UserMessage(role="user", content="My mobile data is very slow.")
    # every sampled tool call malformed -> no sample -> one greedy generate() call
    patch_completion(monkeypatch, [mk_resp([{"tools": [("get_customer_by_id", "{")]}, {"tools": [("get_customer_by_id", "{")]}])])
    greedy = AssistantMessage.text("greedy hello", cost=0.0)
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate([greedy]))
    ag = mk()
    m, _ = ag.generate_next_message(user, ag.get_init_state())
    assert m is greedy and ag.records[0].parse["fallback"] == "greedy_generate" and ag.records[0].parse["sampling"]["n_malformed"] == 2
    # the controller raising -> first sampled candidate is emitted
    patch_completion(monkeypatch, [mk_resp([{"text": VERIFY_TEXT}, {"text": CLAIM_TEXT}])])
    monkeypatch.setattr(A.Controller, "decide", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("scorer exploded")))
    ag = mk()
    m, _ = ag.generate_next_message(user, ag.get_init_state())
    assert m.content == VERIFY_TEXT and ag.records[0].parse["fallback"] == "first_candidate"
    assert "scorer exploded" in ag.records[0].parse["error"][0]


def test_agents_never_read_task(monkeypatch, env_tools, tmp_path, train_ok):
    tools, policy = env_tools
    incoming, outs = canned(env_content("get_customer_by_id", {"customer_id": "C1001"}))
    base = {"boed_scorer": "dug", "boed_filter_factory": small_factory(), "boed_n_candidates": 2, "boed_time_budget_ms": None}
    n0 = len(LEAK_LOG)

    def plain(task, **extra):
        monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate(outs()))
        ag = A.FACTORIES["boed-plain"](tools=tools, domain_policy=policy, llm="m", llm_args=dict(base), task=task, **extra)
        got, _ = run_episode(ag, incoming)
        return [msg_dump(m) for m in got], [(r.decision, r.scores, r.posterior_topk) for r in ag.records]

    def controller(task, **extra):
        patch_completion(monkeypatch, [mk_resp([{"text": VERIFY_TEXT}, {"text": CLAIM_TEXT}])])
        ag = A.FACTORIES["boed-controller"](tools=tools, domain_policy=policy, llm="m", llm_args=dict(base), task=task, **extra)
        st = ag.get_init_state()
        m, _ = ag.generate_next_message(incoming[0], st)
        return msg_dump(m), [(r.decision, r.scores, r.posterior_topk) for r in ag.records]

    real = types.SimpleNamespace(id="[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]")
    ref_p, ref_c = plain(real), controller(real)
    with poisoned(task=PoisonTask, user_db=PoisonDB) as p:
        got_p = plain(p["task"], user_db=p["user_db"], audio_native_config=p["user_db"])
        got_c = controller(p["task"], user_db=p["user_db"])
    strip = lambda x: json.loads(json.dumps(x, default=str, sort_keys=True))
    assert strip(got_p[0]) == strip(ref_p[0]) and strip(got_c[0]) == strip(ref_c[0])
    # decisions are deterministic given the same belief; timings/stat times differ so compare the decision payloads minus stats
    drop = lambda recs: [({k: v for k, v in d.items() if k not in ("stats", "reason")}, s, t) for d, s, t in recs]  # reason embeds elapsed ms
    assert strip(drop(got_p[1])) == strip(drop(ref_p[1])) and strip(drop(got_c[1])) == strip(drop(ref_c[1]))
    assert len(LEAK_LOG) == n0
    rep = check_deployable(("telecom_boed.tau2_ext.agents", "telecom_boed.tau2_ext.candidates"))
    assert not rep.pending and rep.ok, rep.problems()


def test_register_idempotent():
    reg = Registry()
    assert A.register(reg) == ["boed-plain", "boed-controller"]
    assert A.register(reg) == ["boed-plain", "boed-controller"]
    assert reg.get_agent_factory("boed-plain") is A.FACTORIES["boed-plain"]
    assert reg.get_agent_factory("boed-controller") is A.FACTORIES["boed-controller"]
    other = Registry()
    other.register_agent_factory(lambda **k: None, "boed-plain")
    with pytest.raises(ValueError):
        A.register(other)
    with pytest.raises(ValueError):
        A.configure(not_a_setting=1)


def test_settings_via_launcher_payload(env_tools, tmp_path):
    from telecom_boed.tau2_ext import register as R
    tools, policy = env_tools
    A.configure(filter_factory=small_factory(), scorer="eig")  # process default, overridden by the payload below
    payload = {"run": "run_x", "boed_config": _cfg_file(tmp_path, 0.2), "boed_config_sha256": "abc123",
               "agent_kwargs": {"scorer": "dug", "n_candidates": 3, "candidate_temperature": 0.5, "persona": "default",
                                "sidecar": str(tmp_path / "x.jsonl"), "time_budget_ms": 250}}
    ag = A.FACTORIES["boed-controller"](tools=tools, domain_policy=policy, llm="m", task=None,
                                        llm_args={"temperature": 0.0, "_boed": payload})
    assert ag.llm_args == {"temperature": 0.0}  # `_boed` never reaches litellm
    s = ag.boed_settings
    assert (s.scorer, s.n_candidates, s.candidate_temperature, s.time_budget_ms, s.run_id, s.config_sha256) == \
        ("dug", 3, 0.5, 250, "run_x", "abc123") and s.config == payload["boed_config"]
    ep = ag.get_init_state().boed
    assert ep.n_candidates == 3 and ep.cfg["name"] == "boed_test_delta" and ep.init_error is None
    with pytest.raises(ValueError):
        A.resolve_settings({"_boed": {"agent_kwargs": {"objective": "reach"}}})
    reg = Registry()
    proxy = R._IdempotentRegistry(reg)
    assert A.register(proxy) == A.register(proxy) == ["boed-plain", "boed-controller"]
