"""[telecom_boed] W2-S7 tests: rule parsers (action / observation / claim / intent), LLM stubs, accuracy harness, leakage.

All fixtures under tests/fixtures/W2-S7/ are hand-written by the author of the parsers. Accuracy measured on them is a
regression guard, NOT evidence of real parser accuracy (that is measured on logged train-split runs in W4).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from telecom_boed.belief import parsers as P
from telecom_boed.contracts import (INTENTS, ActionParser, ClaimClassifier, DecisionKind, DeviceAction, IntentParser,
                                    Observation, ObservationParser)
from telecom_boed.env.featurize import USER_TOOLS
from telecom_boed.eval.parser_accuracy import evaluate_parsers
from telecom_boed.testing.leakage_scan import check_deployable, scan_module_source

FX = Path(__file__).parent / "fixtures" / "W2-S7"
PARSERS_PY = Path(P.__file__)


def _load(name: str):
    return json.loads((FX / name).read_text())


def _same(a, b) -> bool:
    return type(a) is type(b) and a == b if isinstance(a, bool) or isinstance(b, bool) else a == b


# --------------------------------------------------------------------------- action parser
def test_action_parser_lexicon_covers_all_user_tools():
    from tau2.domains.telecom.environment import get_environment
    from tau2.domains.telecom.user_data_model import NetworkModePreference

    tau2_names = sorted(t.name for t in get_environment().get_user_tools())
    assert len(tau2_names) == 30 and sorted(P.USER_TOOL_NAMES) == tau2_names == sorted(USER_TOOLS)
    assert P.NETWORK_MODES == tuple(m.value for m in NetworkModePreference)

    fx = {k: v for k, v in _load("action_phrasings.json").items() if not k.startswith("_")}
    assert sorted(fx) == tau2_names
    parser = P.RuleActionParser()
    table = []
    for tool in tau2_names:
        cases = fx[tool]
        assert len(cases) >= 2, f"{tool}: need >= 2 phrasings"
        reached = 0
        for c in cases:
            hit = [a for a in parser.parse(c["text"]) if a.tool == tool]
            ok = bool(hit)
            if ok and "ensure" in c:
                ok = hit[0].ensure == tuple(c["ensure"])
            if ok and "args" in c:
                ok = all(hit[0].args_dict().get(k) == v for k, v in c["args"].items())
            assert ok, f"{tool}: not reached from {c['text']!r} -> {[(a.tool, a.ensure, a.args_dict()) for a in parser.parse(c['text'])]}"
            reached += 1
        # third route: the literal function name, which agents sometimes paste
        lit = [a.tool for a in parser.parse(f"Please run {tool}() on your phone.")]
        assert tool in lit, f"{tool}: literal name not recognised ({lit})"
        table.append((tool, reached + 1))
    print("lexicon coverage (tool, routes):", table)
    assert all(n >= 3 for _, n in table)
    # every returned action is a well-formed user DeviceAction
    for a in parser.parse("Please restart your phone."):
        assert isinstance(a, DeviceAction) and a.requestor == "user" and a.tool in tau2_names


def test_action_parser_multi_and_conditional():
    p = P.RuleActionParser()
    tools = lambda t: [a.tool for a in p.parse(t)]  # noqa: E731

    # several firm requests in one message -> all of them, in textual order
    assert tools("Please toggle Airplane Mode off and then reboot your device.") == ["toggle_airplane_mode", "reboot_device"]
    assert tools("Let's restart your phone. After it reboots, run a speed test.") == ["reboot_device", "run_speed_test"]
    got = p.parse("Please grant the messaging app the storage and sms permissions.")
    assert [(a.tool, a.args_dict()["permission"]) for a in got] == [("grant_app_permission", "storage"), ("grant_app_permission", "sms")]
    assert p.parse("Please turn off airplane mode and data saver.") == (
        DeviceAction("user", "toggle_airplane_mode", (), ("airplane_mode", False)),
        DeviceAction("user", "toggle_data_saver_mode", (), ("data_saver", False)))
    # ensure targets for the toggle family
    assert p.parse("Make sure Wi-Fi calling is off.")[0].ensure == ("wifi_calling", False)
    assert p.parse("Please turn mobile data back on.")[0].ensure == ("data", True)
    assert p.parse("Please toggle airplane mode.")[0].ensure is None  # no direction stated
    assert p.parse("Switch it off and on again, i.e. turn off airplane mode. Then turn it back on.")[0].tool == "toggle_airplane_mode"

    # conditionals: parse() keeps only firm requests; parse_all() flags the rest
    t = "Try disabling data saver mode, and if that doesn't work, restart your phone."
    assert tools(t) == ["toggle_data_saver_mode"]
    flags = [(x.action.tool, x.conditional) for x in p.parse_all(t)]
    assert flags == [("toggle_data_saver_mode", False), ("reboot_device", True)]

    t = "Check the status bar. If airplane mode is on, turn it off. Otherwise, check your SIM card status."
    assert [(x.action.tool, x.conditional) for x in p.parse_all(t)] == [
        ("check_status_bar", False), ("toggle_airplane_mode", True), ("check_sim_status", True)]
    assert tools(t) == ["check_status_bar"]
    # the condition clause itself is not a request ("if you see airplane mode ...")
    assert [x.action.tool for x in p.parse_all("If you see the airplane icon on the status bar, turn it off.")] == ["toggle_airplane_mode"]
    # "check if/whether ..." is a question, not a condition
    assert tools("Please check if there is a payment request on your account.") == ["check_payment_request"]
    # only conditional requests: the first one is returned (the user will most likely act on it)
    assert tools("If your SIM is missing, please reseat it, otherwise restart your phone.") == ["reseat_sim_card"]

    # negations are dropped; explanatory mentions are not requests
    assert tools("You don't need to restart your phone.") == []
    assert tools("Do not turn off airplane mode yet.") == []
    assert tools("A reboot can sometimes help refresh the connection.") == []
    assert tools("I will check your account and send you a payment request.") == []

    # classify_agent_message
    D = DecisionKind
    assert p.classify_agent_message("Please turn off airplane mode.") is D.REQUEST_DEVICE
    assert p.classify_agent_message("Which phone model do you have?") is D.ASK
    assert p.classify_agent_message("Your issue should now be resolved. Is there anything else I can help with?") is D.CLAIM_RESOLVED
    assert p.classify_agent_message("I'm going to transfer you to a human agent who can unlock your SIM.") is D.TRANSFER
    assert p.classify_agent_message("Let's verify. Please run a speed test.") is D.REQUEST_DEVICE
    # longer, policy-style agent messages
    assert tools("It looks like Airplane Mode is on, which disables cellular. Please toggle it off, then tell me whether you now have signal.") == ["toggle_airplane_mode"]
    assert tools("Please go to settings, open Apps, tap Messaging, then Permissions, and enable Storage and SMS.") == ["grant_app_permission", "grant_app_permission"]
    assert tools("Let's try changing the network mode to 4G/5G preferred, since 2G is very slow.") == ["set_network_mode_preference"]
    assert tools("Could you try power cycling the phone?") == ["reboot_device"]
    assert p.parse("Please make sure your phone is not in Airplane Mode.")[0].ensure == ("airplane_mode", False)
    assert tools("I've sent you a payment request for your overdue bill. Please check your phone and accept it.") == ["make_payment"]
    assert p.parse("Please set the network mode to 4g_only.")[0].args_dict() == {"mode": "4g_only"}
    # tool-less duplicates collapse
    assert len(p.parse("Restart your phone. Please reboot your device.")) == 1


# --------------------------------------------------------------------------- observation parser
def _in_space(v, allowed) -> bool:
    return any(_same(v, a) for a in allowed)


def test_observation_parser_closed_label_space():
    fx = _load("observation_cases.json")
    full = fx["label_space"]
    cases = fx["cases"]
    assert len(cases) >= 25
    op = P.RuleObservationParser()
    assert isinstance(op, ObservationParser)
    n_ok = 0
    for c in cases:
        space = {k: v for k, v in full.items() if not c.get("space") or any(k.startswith(x) for x in c["space"])}
        req = tuple(DeviceAction("user", t) for t in c["requested"])
        ob = op.parse(c["text"], req, space)
        assert isinstance(ob, Observation) and ob.source == "user_report" and ob.actions == req and ob.raw == c["text"]
        got = dict(ob.fields)
        # closed space: only fields of the label space, only values inside it
        for k, v in got.items():
            assert k in space and _in_space(v, space[k]), (c["text"], k, v)
        assert ob.mentioned == frozenset(got)  # mentioned is exactly the emitted fields
        hash(ob.key())  # hashable, usable as a filter key
        for k, v in c["expect"].items():
            assert k in got and _same(got[k], v), (c["text"], k, got.get(k), v)
        for k in c.get("forbid", []):
            assert k not in got, (c["text"], k)
        if "conflict" in c:
            assert c["conflict"] in dict(ob.conflicts), (c["text"], ob.conflicts)
        n_ok += 1
    assert n_ok == len(cases)

    # a value outside the label space is never emitted, even when the text is unambiguous; it is reported instead
    narrow = {"status_bar.signal": ["none", "poor"], "speed_test.mbps": [2.0]}
    ob = op.parse("I have three bars, it measured 7.3 Mbps.", (), narrow)
    assert dict(ob.fields) == {} and ob.mentioned == frozenset()
    assert dict(ob.out_of_space) == {"status_bar.signal": "good", "speed_test.mbps": 7.3}
    # fields absent from the label space are never touched; empty space -> empty observation
    assert op.parse("Airplane mode is on.", (), {}).fields == ()
    # self-contradiction: nothing emitted, conflict surfaced on the Observation subclass (contract unchanged)
    ob = op.parse("The airplane icon is there. Actually no, the airplane icon is not there.", (), full)
    assert "status_bar.airplane_mode" not in dict(ob.fields) and "status_bar.airplane_mode" not in ob.mentioned
    assert dict(ob.conflicts)["status_bar.airplane_mode"] == (True, False)
    # the first word of a clause in the past tense is ignored, the post-action state wins
    ob = op.parse("Airplane mode was on, but I turned it off and now I have two bars.", (), full)
    assert dict(ob.fields)["status_bar.airplane_mode"] is False and dict(ob.fields)["status_bar.signal"] == "fair"
    # bool vs int confusion cannot slip through the membership test
    assert P.RuleObservationParser._in_space(True, [0, 1]) is P._MISS


# --------------------------------------------------------------------------- claim / intent
def test_claim_classifier_fixtures():
    cases = _load("claim_cases.json")["cases"]
    assert len(cases) >= 20
    cc = P.RuleClaimClassifier()
    assert isinstance(cc, ClaimClassifier)
    ok = 0
    for c in cases:
        p = cc.prob_claim(c["text"])
        assert 0.0 < p < 1.0
        ok += (p >= 0.5) == c["claim"]
    acc = ok / len(cases)
    print(f"claim classifier accuracy on the author's own {len(cases)} fixtures: {acc:.3f} (not evidence of real accuracy)")
    assert acc >= 0.9
    assert cc.prob_claim("") < 0.1
    # a verification request lowers the claim probability of the same assertion
    assert cc.prob_claim("That should fix it. Can you run a speed test to confirm?") < cc.prob_claim("That should fix it.")


def test_intent_parser_fixtures():
    ip = P.RuleIntentParser()
    assert isinstance(ip, IntentParser)
    for c in _load("intent_cases.json")["cases"]:
        pr = ip.parse(c["text"])
        assert set(pr) == set(INTENTS)
        assert abs(sum(pr.values()) - 1.0) < 1e-9 and min(pr.values()) >= ip.floor / 1.1  # floor survives renormalisation
        if "intent" in c:
            assert max(pr, key=pr.get) == c["intent"] and pr[c["intent"]] > 0.8, (c, pr)
        elif "mixed" in c:  # mixed message: split between the two cues, third intent small
            a, b = c["mixed"]
            third = ({*INTENTS} - {a, b}).pop()
            assert abs(pr[a] - pr[b]) < 0.1 and pr[a] > 0.35 and pr[third] < 0.1, (c, pr)
        elif c.get("uniform"):
            assert max(pr.values()) - min(pr.values()) < 1e-9


# --------------------------------------------------------------------------- stubs + factory
def test_llm_stubs_and_factory():
    b = P.build_parsers("llm")
    with pytest.raises(NotImplementedError, match="parser: rule"):
        b.action.parse("restart")
    with pytest.raises(NotImplementedError):
        b.observation.parse("x", (), {})
    with pytest.raises(NotImplementedError):
        b.claim.prob_claim("x")
    with pytest.raises(NotImplementedError):
        b.intent.parse("x")
    assert isinstance(b.action, ActionParser) and isinstance(b.observation, ObservationParser)
    r = P.build_parsers("rule")
    assert isinstance(r.action, P.RuleActionParser) and isinstance(r.claim, P.RuleClaimClassifier)
    with pytest.raises(ValueError):
        P.build_parsers("nope")


# --------------------------------------------------------------------------- harness
def test_parser_accuracy_harness_synthetic():
    d = _load("synthetic_results.json")
    res = evaluate_parsers(d, claim_labels=d["claim_labels"])  # whole results dict accepted
    assert res["n_sims"] == 4
    a = res["action"]
    # 10 agent turns that have a user reaction with a request + 8 without; one request is missed by design
    assert a["request_detect"] == {"tp": 9, "fp": 0, "fn": 1, "tn": 8, "accuracy": pytest.approx(17 / 18)}
    assert a["tool_set"]["precision"] == 1.0 and a["tool_set"]["recall"] == pytest.approx(9 / 10)
    assert a["first_action_acc"] == pytest.approx(9 / 10) and a["n_first"] == 10
    assert a["arg_acc"] == 1.0 and a["n_args"] == 2
    o = res["observation"]
    assert o["n_turns"] == 10 and o["n_emitted"] > 0
    # exactly one deliberately wrong report (three bars vs Poor) among the emitted fields
    assert o["value_precision"] == pytest.approx((o["n_emitted"] - 1) / o["n_emitted"])
    assert o["turns_all_correct"] == 9 and o["precision_by_group"]["status_bar"] < 1.0
    assert o["precision_by_group"]["speed_test"] == 1.0 and 0 < o["coverage"] < 1
    assert res["claim"]["n"] == 5 and res["claim"]["accuracy@0.5"] == 1.0
    assert res["intent"]["n"] == 4 and res["intent"]["accuracy"] == 1.0
    # end-to-end mode (requested = parsed actions) and a caller-supplied narrow label space both run
    e2e = evaluate_parsers(d["simulations"], requested_source="parsed")
    assert e2e["observation"]["requested_source"] == "parsed" and e2e["claim"] is None
    # a parser that returns nothing scores zero recall, not a crash
    class Silent:
        def parse(self, text):
            return ()
    z = evaluate_parsers(d["simulations"], action_parser=Silent())
    assert z["action"]["tool_set"]["recall"] == 0.0 and z["action"]["first_action_acc"] == 0.0


# --------------------------------------------------------------------------- leakage
def test_parsers_deployable_clean():
    assert scan_module_source(PARSERS_PY) == []
    rep = check_deployable(("telecom_boed.belief.parsers",))
    assert "telecom_boed.belief.parsers" in rep.modules and rep.ok, rep.problems()
    src = PARSERS_PY.read_text()
    assert "import parser_accuracy" not in src and "eval.parser_accuracy import" not in src  # harness reads ToolMessages


def test_intent_parser_real_tau2_first_messages():
    """Regression 2026-10-28: real tau2 telecom first user messages (from l2plain results.json) all scored uniform (cue sentence-split by '.')."""
    ip = P.RuleIntentParser()
    mdi = ("I'm having trouble with my mobile data. It's either not working or very slow. I need excellent internet speed on my phone, "
           "and I'm not willing to accept anything less. Can you help me fix this?")
    pr = ip.parse(mdi)
    assert max(pr, key=pr.get) == "mobile_data_issue" and pr["mobile_data_issue"] > 0.8, pr
    pr = ip.parse("I'm having some issues with my mobile data. It's either not working or very slow.")
    assert max(pr, key=pr.get) == "mobile_data_issue", pr
    assert max(ip.parse("I'm unable to send MMS messages using my messaging app.").items(), key=lambda kv: kv[1])[0] == "mms_issue"
    assert max(ip.parse("My phone has been showing 'No Service' for the past few hours.").items(), key=lambda kv: kv[1])[0] == "service_issue"
