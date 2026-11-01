"""[telecom_boed] W3-F1 tests: fixes from the first real GPU smoke (rlab2, 2026-10-16; 5 TRAIN tasks, boed-plain l2plain_526576.20 and
boed-controller l2ctrl_526576.21). CPU only, no LLM (generate / litellm patched). Fixtures: tests/fixtures/W3-F1/smoke_first_turns.json
(real first user messages + the agent's first tool calls of those runs).

1. empty-intent / `max() iterable argument is empty`: root cause = Controller._fallback handing MyopicDUG a candidate set without terminals.
2. intent at step 0: the rule parser scored every real opener uniform; agents.parse_intent has fallback cues; informative -> pruned support.
3. `repair_invalid_user_tools`: default off (boed-plain == LLMAgent), on -> text request + REQUEST_DEVICE candidate, recorded in the sidecar.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from loguru import logger

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.domains.telecom.environment import get_environment

from telecom_boed import paths
from telecom_boed.belief.filter import ParticleFilter
from telecom_boed.belief.parsers import INTENTS, build_parsers
from telecom_boed.contracts import CandidateAction, DecisionKind, DeviceAction, _frozen_items
from telecom_boed.decide.controller import Controller, dominant_intent
from telecom_boed.env.featurize import USER_TOOLS
from telecom_boed.tau2_ext import agents as A
from telecom_boed.tau2_ext import candidates as C
from telecom_boed.testing.leakage_scan import check_deployable

from test_agents import (SmallFilter, fake_generate, mk_resp, msg_dump, patch_completion, run_episode, small_factory)  # noqa: E402

logger.disable("tau2")
FIX = Path(__file__).parent / "fixtures" / "W3-F1" / "smoke_first_turns.json"
REC = json.loads(FIX.read_text())["records"]
GREETING = AssistantMessage.text("Hi! How can I help you today?")
K = DecisionKind
USER_LIKE = ("check_network_status", "check_sim_status", "toggle_airplane_mode", "run_speed_test", "reboot_device")


@pytest.fixture(scope="module")
def env_tools():
    env = get_environment()
    return env.get_tools(), env.get_policy()


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((paths.CONFIG_DIR / "boed_default.yaml").read_text())


@pytest.fixture(scope="module")
def full_filter(cfg) -> ParticleFilter:
    return ParticleFilter("deployable", cfg, "default")


@pytest.fixture(autouse=True)
def _clean_defaults():
    A.reset_configuration()
    yield
    A.reset_configuration()


def tc_msg(name: str, args: dict | None = None, i: int = 0) -> AssistantMessage:
    return AssistantMessage(role="assistant", content=None, cost=0.0,
                            tool_calls=[ToolCall(id=f"call_{i}", name=name, arguments=args or {}, requestor="assistant")])


# --------------------------------------------------------------------------- 1. empty max()
def test_fixture_is_train_and_complete(splits):
    assert len(REC) == 10
    for r in REC:
        assert r["task_id"] in splits["train"], r["task_id"]  # TRAIN only
        assert r["intent"] == "mobile_data_issue" and r["first_user"]


def test_dominant_intent_is_robust(full_filter):
    b = full_filter.init({"mobile_data_issue": 1.0})
    assert dominant_intent(b) == "mobile_data_issue"
    import dataclasses
    e = dataclasses.replace(b, intent_probs=())  # empty intent_probs -> particle intent mass
    assert dominant_intent(e) == "mobile_data_issue"
    assert dominant_intent(dataclasses.replace(e, particles=())) == ""  # nothing at all -> "" and no exception


@pytest.mark.parametrize("n_done", [0, 1])
def test_fallback_without_terminals_does_not_raise(full_filter, cfg, n_done):
    """Real failing case: the agent offers only backend calls (no claim/transfer) and the budget ends before/after one predict."""
    r = next(x for x in REC if x["first_agent_calls"][0][0] == "get_customer_by_phone")
    name, args = r["first_agent_calls"][0]
    cand = CandidateAction(K.AGENT_TOOL, (DeviceAction("assistant", name, _frozen_items(args)),), 0, None)
    b = full_filter.init({"mobile_data_issue": 1.0})
    c = Controller("reach", full_filter, full_filter.grader, cfg, override="none", time_budget_ms=0.0)  # deadline already passed
    if n_done:
        c.proxy.done = [cand]
        idx, scores, info = c._fallback(b, [cand], {"depth": None, "fallback": None, "n_scored": 1}, "test")
        assert idx == 0 and set(scores) == {0} and "scored 1/1" in info["fallback"]
    d = c.decide(b, [cand])
    assert d.chosen is cand and "FALLBACK" in d.reason  # first candidate, no exception
    d2 = c.decide(b, [cand, CandidateAction(K.AGENT_TOOL, (DeviceAction("assistant", "get_details_by_id", ()),), 1, None)])
    assert d2.chosen.kind == K.AGENT_TOOL


@pytest.mark.parametrize("scorer", ["dug", "eig", "heuristic", "kstep", "reach"])
def test_controller_without_terminal_candidates_never_raises(full_filter, cfg, scorer):
    """The controller agent offers only sampled messages (no claim/transfer); dug/eig used to raise `max() iterable argument is empty`
    whenever no nonterminal had positive gain (an ASK always has gain 0). Synthetic terminals are scored but never chosen."""
    b = full_filter.init({"mobile_data_issue": 1.0})
    ask = CandidateAction(K.ASK, (), 0, "Which phone number is on the account?")
    c = Controller(scorer, full_filter, full_filter.grader, cfg, override="none", time_budget_ms=None)
    d = c.decide(b, [ask])
    assert d.chosen is ask and all(k < 1 for k, _ in d.scores)  # scores only for the offered candidate
    d = c.decide(b, [ask, CandidateAction(K.ASK, (), 1, "Could you tell me your name?")])
    assert d.chosen.kind == K.ASK


def test_controller_agent_real_failing_turn_has_no_error(monkeypatch, env_tools, tmp_path):
    """Replays turn 1 of a real failing episode (uniform-intent 2,285-particle belief that the budget cannot score): the old agent logged
    `boed: ValueError: max() iterable argument is empty`; now the turn completes (fallback inside the controller, no error)."""
    tools, policy = env_tools
    r = next(x for x in REC if x["run"].startswith("l2ctrl") and x["first_agent_calls"][0][0] == "get_customer_by_phone")
    name, args = r["first_agent_calls"][0]
    patch_completion(monkeypatch, [mk_resp([{"tools": [(name, json.dumps(args))]}] * 4)])
    side = tmp_path / "s.jsonl"
    # the opener is replaced by an uninformative one so the belief really is the 2,285-particle uniform prior of the failing run
    ag = A.FACTORIES["boed-controller"](tools=tools, domain_policy=policy, llm="m",
                                        llm_args={"boed_sidecar": str(side), "boed_time_budget_ms": 0.0, "boed_scorer": "reach"})
    m, _ = ag.generate_next_message(UserMessage(role="user", content="Hello, I need some help."), ag.get_init_state([GREETING]))
    rec = json.loads(side.read_text().splitlines()[0])
    assert "error" not in rec["parse"], rec["parse"].get("error")
    assert rec["parse"]["belief"]["n_particles"] == 2285 and rec["parse"]["intent_informative"] is False
    assert rec["decision"]["mode"] == "controller" and rec["decision"]["stats"]["fallback"]
    assert m.tool_calls and m.tool_calls[0].name == name


# --------------------------------------------------------------------------- 2. intent at step 0
@pytest.mark.parametrize("r", REC, ids=[f"{r['run'][:7]}-{i}" for i, r in enumerate(REC)])
def test_first_user_message_intent_matches_task(r):
    probs, informative = A.parse_intent(build_parsers("rule"), r["first_user"])
    assert informative and set(probs) == set(INTENTS)
    assert max(probs, key=probs.get) == r["intent"] and probs[r["intent"]] > 0.9
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_parse_intent_uninformative_and_rule_parser_untouched():
    pb = build_parsers("rule")
    probs, inf = A.parse_intent(pb, "Hello, I need some help.")
    assert not inf and all(abs(v - 1 / 3) < 1e-9 for v in probs.values())
    probs, inf = A.parse_intent(pb, "I can't send picture messages to my friend.")  # the rule parser's own cue still wins
    assert inf and max(probs, key=probs.get) == "mms_issue"
    probs, inf = A.parse_intent(pb, "I have no service at all, zero bars.")
    assert inf and max(probs, key=probs.get) == "service_issue"


def test_first_message_inits_pruned_filter(monkeypatch, env_tools, tmp_path):
    tools, policy = env_tools
    r = REC[0]
    patch_completion(monkeypatch, [mk_resp([{"text": "Could you please run a speed test?"}] * 2)])
    side = tmp_path / "s.jsonl"
    ag = A.FACTORIES["boed-controller"](tools=tools, domain_policy=policy, llm="m",
                                        llm_args={"boed_sidecar": str(side), "boed_n_candidates": 2, "boed_scorer": "dug"})
    ag.generate_next_message(UserMessage(role="user", content=r["first_user"]), ag.get_init_state([GREETING]))
    b = json.loads(side.read_text().splitlines()[0])["parse"]
    assert b["intent_informative"] is True and b["belief"]["n_particles"] == 254  # mobile_data support only (was 2,285)
    ip = b["belief"]["intent_probs"]
    assert set(ip) == {"mobile_data_issue"} and abs(ip["mobile_data_issue"] - 1.0) < 1e-9
    assert "error" not in b


def test_late_informative_message_reweights_uniform_prior(monkeypatch, env_tools, tmp_path):
    """Greeting-like first user turn (no cue) -> uniform prior; the first informative message afterwards reweights and prunes it."""
    tools, policy = env_tools
    patch_completion(monkeypatch, [mk_resp([{"text": "Which phone number is on the account?"}] * 2)] * 2)
    side = tmp_path / "s.jsonl"
    ag = A.FACTORIES["boed-controller"](tools=tools, domain_policy=policy, llm="m",
                                        llm_args={"boed_sidecar": str(side), "boed_n_candidates": 2, "boed_scorer": "dug"})
    st = ag.get_init_state([GREETING])
    _, st = ag.generate_next_message(UserMessage(role="user", content="Hi, are you there?"), st)
    _, st = ag.generate_next_message(UserMessage(role="user", content=REC[0]["first_user"]), st)
    r1, r2 = (json.loads(x)["parse"] for x in side.read_text().splitlines())
    assert r1["belief"]["n_particles"] == 2285 and r1["intent_informative"] is False
    assert r2.get("intent_reweighted") is True and "error" not in r2
    assert r2["belief"]["n_particles"] == 254 and set(r2["belief"]["intent_probs"]) == {"mobile_data_issue"}
    assert abs(sum(p.weight for p in st.boed.belief.particles) - 1.0) < 1e-9


# --------------------------------------------------------------------------- 3. repair
def test_repair_unit():
    pb = build_parsers("rule")
    names = ("get_customer_by_phone", "get_customer_by_id")
    assert C.repair_user_tool_call(tc_msg("get_customer_by_phone", {"phone_number": "1"}), 0, names) is None  # own tool
    assert C.repair_user_tool_call(AssistantMessage.text("hello"), 0, names) is None
    assert C.repair_user_tool_call(tc_msg("check_network_status"), 0, None) is None
    msg, cand, entry = C.repair_user_tool_call(tc_msg("check_network_status"), 2, names)
    assert msg.tool_calls is None and msg.cost == 0.0
    assert msg.content == "Could you please check your phone's network status and tell me exactly what you see?"
    assert cand.kind == K.REQUEST_DEVICE and cand.candidate_idx == 2 and cand.text == msg.content
    assert cand.actions == (DeviceAction("user", "check_network_status", ()),)
    assert entry["from_tool"] == "check_network_status" and entry["to_request"]["tool"] == "check_network_status"
    # args are carried; every phone tool has a template and is repairable
    _, cand, _ = C.repair_user_tool_call(tc_msg("check_app_permissions", {"app_name": "messaging"}), 0, names)
    assert "'messaging'" in cand.text and cand.actions[0].args_dict() == {"app_name": "messaging"}
    for t in USER_TOOLS:
        assert t in C._REQUEST_PHRASE, t
        r = C.repair_user_tool_call(tc_msg(t), 0, names)
        assert r is not None and r[0].content.startswith("Could you please ") and r[0].content.endswith("and tell me exactly what you see?")
    # the plain text parser reads the diagnostic templates back as the same phone action
    for t in ("check_network_status", "check_sim_status", "run_speed_test", "check_status_bar", "can_send_mms"):
        txt = C.repair_text(t, {})
        assert [a.tool for a in C.to_candidate(AssistantMessage.text(txt), pb).actions] == [t], txt


def test_settings_default_off():
    st, _ = A.resolve_settings({})
    assert st.repair_invalid_user_tools is False
    st, _ = A.resolve_settings({"_boed": {"agent_kwargs": {"repair_invalid_user_tools": True}}})
    assert st.repair_invalid_user_tools is True


def _canned_invalid():
    """USER, USER, TOOL(result of own call) with agent outputs: invalid phone-tool call, own backend call, text."""
    outs = lambda: [tc_msg("check_network_status", {}, 0), tc_msg("get_customer_by_id", {"customer_id": "C1001"}, 1),
                    AssistantMessage.text("Could you please run a speed test?", cost=0.0)]
    user = [UserMessage(role="user", content=REC[0]["first_user"]), UserMessage(role="user", content="It says airplane mode is on."),
            UserMessage(role="user", content="Okay.")]
    return user, outs


def _plain(tools, policy, side, **extra):
    la = {"temperature": 0.0, "boed_scorer": "dug", "boed_sidecar": str(side), "boed_filter_factory": small_factory(),
          "boed_time_budget_ms": None, **extra}
    return A.FACTORIES["boed-plain"](tools=tools, domain_policy=policy, llm="m", llm_args=la, task=None)


def test_boed_plain_identity_repair_off_and_repair_on(monkeypatch, env_tools, tmp_path):
    from tau2.agent.llm_agent import LLMAgent
    tools, policy = env_tools
    user, outs = _canned_invalid()
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate(outs()))
    ref, _ = run_episode(LLMAgent(tools=tools, domain_policy=policy, llm="m", llm_args={"temperature": 0.0}), user)
    # explicit False == default == LLMAgent, byte for byte, and no `repaired` key
    for extra in ({}, {"boed_repair_invalid_user_tools": False}):
        monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate(outs()))
        side = tmp_path / f"off{len(extra)}.jsonl"
        got, _ = run_episode(_plain(tools, policy, side, **extra), user)
        assert [msg_dump(m) for m in got] == [msg_dump(m) for m in ref]
        assert all("repaired" not in json.loads(x)["parse"] for x in side.read_text().splitlines())
    assert ref[0].tool_calls[0].name == "check_network_status"
    # on: the invalid call is replaced by the request; own backend calls and text are untouched; the history holds the repaired message
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate(outs()))
    side = tmp_path / "on.jsonl"
    ag = _plain(tools, policy, side, boed_repair_invalid_user_tools=True)
    got, st = run_episode(ag, user[:2])  # turn 2 calls get_customer_by_id (a result would follow); two turns suffice here
    assert not got[0].tool_calls and got[0].content == C.repair_text("check_network_status", {})
    assert msg_dump(got[1]) == msg_dump(ref[1])
    assert any(m is got[0] for m in st.messages) and not any(getattr(m, "tool_calls", None) and m.tool_calls[0].name == "check_network_status" for m in st.messages)
    recs = [json.loads(x) for x in side.read_text().splitlines()]
    assert recs[0]["parse"]["repaired"] == [{"idx": 0, "from_tool": "check_network_status", "to_request": {
        "tool": "check_network_status", "args": {}, "text": got[0].content}}]
    assert recs[0]["parse"]["emitted"]["kind"] == "request_device" and "repaired" not in recs[1]["parse"]
    assert "error" not in recs[0]["parse"], recs[0]["parse"].get("error")


def _ctrl(tools, policy, side, **extra):
    la = {"boed_scorer": "heuristic", "boed_filter_factory": small_factory(), "boed_n_candidates": 4, "boed_candidate_temperature": 0.7,
          "boed_time_budget_ms": None, "boed_sidecar": str(side), **extra}
    return A.FACTORIES["boed-controller"](tools=tools, domain_policy=policy, llm="m", llm_args=la, task=None)


def test_controller_repair_off_drops_all_on_scores_requests(monkeypatch, env_tools, tmp_path):
    tools, policy = env_tools
    four = [{"tools": [("check_network_status", "{}")]}] * 4
    user = UserMessage(role="user", content=REC[0]["first_user"])
    # off: the 42-turn failure of the real run -- every candidate is an invalid phone-tool call and is dropped
    patch_completion(monkeypatch, [mk_resp(four)])
    side = tmp_path / "off.jsonl"
    ag = _ctrl(tools, policy, side)
    m, _ = ag.generate_next_message(user, ag.get_init_state([GREETING]))
    p = json.loads(side.read_text().splitlines()[0])["parse"]
    assert p["fallback"] == "no_scorable_candidate" and p["dropped"]["invalid_tool"] == 4 and "repaired" not in p
    assert m.tool_calls[0].name == "check_network_status"  # the bad call goes out unchanged (first candidate)
    # on: repaired before scoring -> scored as a REQUEST_DEVICE (duplicates deduped), emitted as the template text
    patch_completion(monkeypatch, [mk_resp(four)])
    side = tmp_path / "on.jsonl"
    ag = _ctrl(tools, policy, side, boed_repair_invalid_user_tools=True)
    m, st = ag.generate_next_message(user, ag.get_init_state([GREETING]))
    rec = json.loads(side.read_text().splitlines()[0])
    p = rec["parse"]
    assert "error" not in p and "fallback" not in p, p
    assert len(p["repaired"]) == 4 and {r["from_tool"] for r in p["repaired"]} == {"check_network_status"}
    assert p["dropped"] == {"invalid_tool": 0, "duplicate": 3}
    assert rec["decision"]["mode"] == "controller" and rec["decision"]["chosen"]["kind"] == "request_device"
    assert not m.tool_calls and m.content == C.repair_text("check_network_status", {}) and st.messages[-1] is m


def test_controller_repair_mixed_candidates_and_greedy_fallback(monkeypatch, env_tools, tmp_path):
    tools, policy = env_tools
    mixed = [{"tools": [("check_sim_status", "{}")]}, {"tools": [("get_customer_by_id", '{"customer_id": "C1001"}')]},
             {"text": "Could you please run a speed test on your phone and tell me the result?"}, {"tools": [("check_sim_status", "{}")]}]
    user = UserMessage(role="user", content=REC[0]["first_user"])
    patch_completion(monkeypatch, [mk_resp(mixed)])
    side = tmp_path / "mix.jsonl"
    ag = _ctrl(tools, policy, side, boed_repair_invalid_user_tools=True)
    m, _ = ag.generate_next_message(user, ag.get_init_state([GREETING]))
    p = json.loads(side.read_text().splitlines()[0])["parse"]
    assert [r["idx"] for r in p["repaired"]] == [0, 3] and "error" not in p
    assert all(c.get("tool") != "check_sim_status" or c.get("repaired") for c in p["candidates"])
    assert not (m.tool_calls and m.tool_calls[0].name == "check_sim_status")  # a phone tool never leaves the agent
    # greedy-generate fallback (all samples malformed) is repaired too
    patch_completion(monkeypatch, [mk_resp([{"tools": [("get_customer_by_id", "{")]}] * 4)])
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate([tc_msg("run_speed_test")]))
    ag = _ctrl(tools, policy, tmp_path / "g.jsonl", boed_repair_invalid_user_tools=True)
    m, _ = ag.generate_next_message(user, ag.get_init_state([GREETING]))
    assert not m.tool_calls and "speed test" in m.content
    assert ag.records[0].parse["fallback"] == "greedy_generate" and ag.records[0].parse["repaired"][0]["from_tool"] == "run_speed_test"


def test_leakage_scan_still_clean():
    rep = check_deployable(("telecom_boed.tau2_ext.agents", "telecom_boed.tau2_ext.candidates"))
    assert not rep.pending and rep.ok, rep.problems()


def test_controller_pick_first_new_skips_repeated_request(monkeypatch, env_tools, tmp_path):
    """pick="first_new": no scoring; a device request already emitted this episode is skipped in favour of the next new one."""
    tools, policy = env_tools
    net, speed = {"tools": [("check_network_status", "{}")]}, {"tools": [("run_speed_test", "{}")]}
    patch_completion(monkeypatch, [mk_resp([net, net]), mk_resp([net, speed]), mk_resp([net, net])])
    side = tmp_path / "fn.jsonl"
    ag = _ctrl(tools, policy, side, boed_repair_invalid_user_tools=True, boed_pick="first_new", boed_n_candidates=2)
    st = ag.get_init_state([GREETING])
    m1, st = ag.generate_next_message(UserMessage(role="user", content=REC[0]["first_user"]), st)
    m2, st = ag.generate_next_message(UserMessage(role="user", content="It says connected, no data."), st)
    m3, st = ag.generate_next_message(UserMessage(role="user", content="Speed test failed."), st)
    recs = [json.loads(x) for x in side.read_text().splitlines()]
    assert m1.content == C.repair_text("check_network_status", {})
    assert m2.content == C.repair_text("run_speed_test", {})  # the repeat (sample 0) is skipped
    assert m3.content == C.repair_text("check_network_status", {})  # all candidates repeat -> first one
    assert [r["decision"]["mode"] for r in recs] == ["first_new"] * 3
    assert [r["decision"]["repeat"] for r in recs] == [False, False, True]
    assert all("error" not in r["parse"] for r in recs), [r["parse"].get("error") for r in recs]
