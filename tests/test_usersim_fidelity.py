"""CPU unit tests for scripts/usersim_fidelity.py on synthetic transcripts (no tau2 import)."""
import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "usersim_fidelity.py"
_s = importlib.util.spec_from_file_location("usersim_fidelity", _p)
uf = importlib.util.module_from_spec(_s)
_s.loader.exec_module(uf)


def A(c, i, tc=None): return {"role": "assistant", "content": c, "tool_calls": tc, "turn_idx": i}
def U(c, i, tc=None): return {"role": "user", "content": c, "tool_calls": [{"name": n, "arguments": {}} for n in (tc or [])] or None, "turn_idx": i}
def T(c, i, req="user"): return {"role": "tool", "content": c, "requestor": req, "turn_idx": i}


def sim(msgs, term="user_stop", reward=0.0):
    return {"id": "abcdef012345", "messages": msgs, "termination_reason": term, "reward_info": {"reward": reward}}


def test_find_requests():
    r = uf.find_requests("Please turn off Airplane Mode on your phone. Then run a speed test.")
    assert [(x["kind"], x["topic"]) for x in r] == [("act", "airplane"), ("read", "speed_test")]
    assert r[0]["expected"] == {"toggle_airplane_mode"}
    assert uf.find_requests("I have enabled roaming on your line.") == []


def test_compliance_and_reasons():
    m = [A("Hi! How can I help you today?", 0), U("data broken", 1),
         A("Please toggle airplane mode.", 2), U("", 3, ["toggle_airplane_mode"]), T("ok", 4), U("Done, airplane mode is off.", 5),
         A("Please turn on data roaming.", 6), U("How do I do that? I can't find it.", 7),
         A("Please run a speed test.", 8), U("I've run the speed test, it shows 50 Mbps.", 9)]
    c = uf.analyze_sim(sim(m), [])["counts"]
    assert c["compliance"] == [1, 3]
    assert c["comp_reason_clarify_or_refuse"][0] == 1 and c["comp_reason_claims_done_no_tool"][0] == 1


def test_invented_and_template():
    m = [A("Hi! How can I help you today?", 0), U("My data is slow.", 1),
         A("Anything else?", 2), U("The status bar shows 5G with full bars. <think>x</think>", 3),
         A("ok", 4), U("I don't understand 'no_service'.", 5)]
    c = uf.analyze_sim(sim(m, "max_steps"), [])["counts"]
    assert c["invented_state"][0] == 1
    assert c["think_leak"][0] == 1 and c["template_artifact"][0] == 1
    assert c["confused_dont_understand"][0] == 1 and c["confused_token_echo"][0] == 1


def test_grounded_state_not_flagged_and_contradiction():
    m = [A("Hi! How can I help you today?", 0), U("slow", 1),
         A("Please check the network status.", 2), U("", 3, ["check_network_status"]), T("Airplane Mode: ON\nCellular Signal: none", 4),
         U("Airplane mode is off and speed is 90 Mbps.", 5)]
    c = uf.analyze_sim(sim(m, "max_steps"), [])["counts"]
    assert c["invented_state"][0] == 0 and c["contradicts_tool"] == [1, 1]


def test_premature_stop():
    m = [A("Hi! How can I help you today?", 0), U("slow", 1), A("Anything?", 2), U("###STOP###", 3)]
    rows = [{"msg_idx": 2, "n_faults_remaining": 1, "faults_remaining": ["x"], "is_claim": False}]
    c = uf.analyze_sim(sim(m), rows)["counts"]
    assert c["stop_premature_both"] == [1, 1] and c["stops"] == [1, 1]


def test_wilson():
    lo, hi = uf.wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
