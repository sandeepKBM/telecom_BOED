"""[telecom_boed] W4-S15 tests: H6 evaluation metrics (eval/metrics.py, eval/fa_replay.py, scripts/telecom_eval.py).

Layers: (1) synthetic fixtures, one per metric; (2) the oracle claim replay on gold trajectories; (3) integration on two
real TRAIN runs (tests/fixtures/W4-S15/, slimmed copies of results/runs/f1ctrep_526576.32 and f1plrep_526576.35).
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from telecom_boed import paths
from telecom_boed.eval import fa_replay
from telecom_boed.eval import metrics as M

FX = Path(__file__).parent / "fixtures" / "W4-S15"
CT, PL = FX / "f1ctrep_526576.32", FX / "f1plrep_526576.35"
SCRIPTS = paths.PACKAGE_ROOT / "scripts"
CLAIM_TEXT = "Your mobile data issue has been resolved. Is there anything else I can help you with today?"


def _status_module():
    spec = importlib.util.spec_from_file_location("telecom_status", SCRIPTS / "telecom_status.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- synthetic builders
def _call(i, name, requestor="assistant"):
    return {"id": f"c{i}", "name": name, "arguments": {}, "requestor": requestor}


def make_sim(task_id, term="user_stop", reward=1.0, last_user="Thanks ###STOP###", steps=(), usage=(100, 10), sid=None,
             trial=0, texts=("Let me look into that.",)):
    """Transcript: greeting, user, then `steps` = [("tool", name, requestor, error)] or [("text", str)], final user message."""
    msgs = [{"role": "assistant", "content": M.GREETING}, {"role": "user", "content": "My data is broken."}]
    n = 0
    for st in steps:
        if st[0] == "text":
            msgs.append({"role": "assistant", "content": st[1], "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}})
        else:
            _, name, req, err = st
            n += 1
            holder = "assistant" if req == "assistant" else "user"
            msgs.append({"role": holder, "content": None, "tool_calls": [_call(n, name, req)],
                         **({"usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}} if holder == "assistant" else {})})
            msgs.append({"role": "tool", "id": f"c{n}", "content": "Error: x" if err else "ok", "requestor": req, "error": err})
    msgs.append({"role": "user", "content": last_user})
    return {"id": sid or f"s{random.Random(task_id + str(trial) + str(term) + str(reward)).randrange(10**9)}", "task_id": task_id,
            "trial": trial, "termination_reason": term, "reward_info": {"reward": reward}, "messages": msgs}


def run_of(sims, name="syn", sidecar=(), tasks=None):
    return {"name": name, "path": "-", "results": {"simulations": list(sims), "tasks": []}, "tasks": tasks or {},
            "sidecar": list(sidecar), "meta": None, "events": []}


def tid(intent="mobile_data_issue", faults="airplane_mode_on|bad_vpn", persona="None"):
    return f"[{intent}]{faults}[PERSONA:{persona}]"


def ev(sims, **kw):
    return M.evaluate_run("-", replay=False, n_boot=200, run=run_of(sims, **kw))


# --------------------------------------------------------------------------- pass^k
def test_pass_hat_k_matches_telecom_status_and_tau2_definition():
    st = _status_module()
    rng = random.Random(1)
    per = {f"t{i}": [float(rng.random() < 0.5) for _ in range(4)] for i in range(30)}
    per["short"] = [1.0, 0.0]  # n < k is skipped for k > 2 in both implementations
    for k in (1, 2, 3, 4):
        assert M.pass_hat_k(per, k) == pytest.approx(st.pass_hat_k(per, k))
    assert M.pass_hat_k({"a": [1.0, 1.0, 0.0, 0.0]}, 2) == pytest.approx(1 / 6)  # C(2,2)/C(4,2)
    assert M.pass_hat_k({"a": [1.0, 0.0]}, 3) is None


def test_run_pass_hat_k_with_ci():
    sims = [make_sim(tid(faults=f"f{t}"), reward=float(r), trial=i, sid=f"{t}-{i}")
            for t, rs in enumerate([(1, 1), (1, 0), (0, 0)]) for i, r in enumerate(rs)]
    m = ev(sims)["metrics"]
    assert m["pass_hat_k"]["1"]["value"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert m["pass_hat_k"]["2"]["value"] == pytest.approx((1 + 0 + 0) / 3)
    lo, hi = m["pass_hat_k"]["1"]["ci95"]
    assert 0 <= lo <= m["pass_hat_k"]["1"]["value"] <= hi <= 1


# --------------------------------------------------------------------------- episode FA and endings
def test_episode_fa_and_harness_failures_are_separate():
    cases = [  # (term, reward, last_user) -> (end_class, fa)
        (("user_stop", 0.0, "bye ###STOP###"), ("stop", True)),
        (("user_stop", 1.0, "bye ###STOP###"), ("stop", False)),
        (("user_stop", 0.0, "please ###TRANSFER###"), ("user_transfer", False)),
        (("user_stop", 0.0, "no ###OUT-OF-SCOPE###"), ("user_oos", False)),
        (("agent_stop", 0.0, "x"), ("agent_stop", False)),
        (("max_steps", 0.0, "still broken"), ("max_steps", False)),
        (("too_many_errors", 0.0, "still broken ###STOP###"), ("too_many_errors", False)),  # STOP text is irrelevant off user_stop
        (("timeout", 0.0, "x"), ("harness", False)),
        (("infrastructure_error", 0.0, "x"), ("harness", False)),
    ]
    sims = [make_sim(tid(faults=f"f{i}"), term=t, reward=r, last_user=lu) for i, ((t, r, lu), _) in enumerate(cases)]
    r = ev(sims)
    for e, (_, (cls, fa)) in zip(r["episodes"], cases):
        assert (e["end_class"], e["fa_episode"]) == (cls, fa), e["termination"]
    m = r["metrics"]
    assert m["episode_fa"]["count"] == 1 and m["episode_fa"]["n_user_stop_stop"] == 2
    assert m["episode_fa"]["rate_given_stop"]["value"] == pytest.approx(0.5)
    assert m["episode_fa"]["rate_all"]["value"] == pytest.approx(1 / len(cases))
    assert m["end_classes"]["max_steps"] == 1 and m["end_classes"]["too_many_errors"] == 1 and m["end_classes"]["harness"] == 2
    assert m["harness_failures"]["reasons"] == {"timeout": 1, "infrastructure_error": 1}


def test_missing_reward_counts_as_zero_but_is_flagged():
    s = make_sim(tid())
    s["reward_info"] = None
    m = ev([s])["metrics"]
    assert m["reward_missing"] == 1 and m["pass_hat_k"]["1"]["value"] == 0.0


# --------------------------------------------------------------------------- turns, tools, tokens, verified
def test_turns_tool_calls_tokens_and_verified_transcript_proxy():
    steps = [("text", "Hello, which phone?"), ("tool", "get_customer_by_phone", "assistant", False),
             ("tool", "toggle_airplane_mode", "user", False), ("tool", "check_status_bar", "user", False),
             ("text", CLAIM_TEXT)]
    s = make_sim(tid(), reward=1.0, steps=steps, usage=(100, 10))
    s2 = make_sim(tid(faults="x"), reward=1.0, steps=steps[:3] + [("text", CLAIM_TEXT)], sid="b")  # no check after the fix
    s3 = make_sim(tid(faults="y"), reward=0.0, steps=steps, sid="c")  # verified but not a success
    r = ev([s, s2, s3])
    e = r["episodes"]
    assert e[0]["turns"] == 3 and e[0]["agent_tool_calls"] == 1 and e[0]["user_tool_calls"] == 2
    assert (e[0]["prompt_tokens"], e[0]["completion_tokens"]) == (300, 30)
    assert [x["verified_transcript"] for x in e] == [True, False, True]
    assert [x["verified_success"] for x in e] == [True, False, False]
    m = r["metrics"]
    assert m["verified_success"]["count"] == 1 and m["verified_success"]["source_counts"] == {"transcript": 3}
    assert m["tokens"]["total"] == 990 and m["tokens"]["per_verified_success_total"] == pytest.approx(m["tokens"]["total"] / 1)


def test_verified_ignores_write_own_output_errors_and_reads_before_the_fix():
    read_first = [("tool", "check_status_bar", "user", False), ("tool", "toggle_data", "user", False), ("text", CLAIM_TEXT)]
    assert not M.transcript_verified(make_sim(tid(), steps=read_first)["messages"])
    errored_write = [("tool", "toggle_data", "user", False), ("tool", "check_status_bar", "user", False),
                     ("tool", "reboot_device", "user", True)]  # a failed write does not move the "last mutation"
    assert M.transcript_verified(make_sim(tid(), steps=errored_write)["messages"])
    only_transfer = [("tool", "toggle_data", "user", False), ("tool", "transfer_to_human_agents", "assistant", False)]
    assert not M.transcript_verified(make_sim(tid(), steps=only_transfer)["messages"])


def test_write_tools_match_tau2_tool_types():
    from tau2.environment.toolkit import ToolType, get_tool_types
    from telecom_boed.env import snapshot as S
    env = S.fast_env()
    types = {**get_tool_types(env.tools), **get_tool_types(env.user_tools)}
    assert {n for n, t in types.items() if t == ToolType.WRITE} == set(M.WRITE_TOOLS)


# --------------------------------------------------------------------------- sidecar matching
def _rec(sim_id, turn, kind, text=None, tool=None, v=False, claim_prob=None):
    chosen = {"kind": kind, "text": text, "actions": [{"requestor": "assistant", "tool": tool, "args": {}}] if tool else [],
              "candidate_idx": 0}
    return {"run_id": "r", "sim_id": sim_id, "turn": turn, "decision": {"chosen": chosen},
            "parse": {"belief": {"v": v}, "candidates": [{"idx": 0, "kind": kind, "claim_prob": claim_prob}]}}


def test_sidecar_matching_uses_text_and_tool_signature_and_v_flag():
    a = make_sim(tid(faults="a"), steps=[("text", "Please check your status bar."), ("text", CLAIM_TEXT)], sid="A")
    b = make_sim(tid(faults="b"), steps=[("tool", "get_customer_by_phone", "assistant", False), ("text", CLAIM_TEXT)], sid="B")
    sc = [_rec("zz", 1, "agent_tool", tool="get_customer_by_phone"), _rec("zz", 2, "claim_resolved", CLAIM_TEXT, v=True),
          _rec("yy", 1, "request_device", "Please check your status bar."), _rec("yy", 2, "claim_resolved", CLAIM_TEXT, v=False)]
    got = M.match_sidecar([a, b], sc)
    assert [r["turn"] for r in got["A"]] == [1, 2] and got["A"][0]["sim_id"] == "yy" and got["B"][0]["sim_id"] == "zz"
    r = ev([a, b], sidecar=sc)
    e = {x["sim_id"]: x for x in r["episodes"]}
    assert e["A"]["verified_sidecar"] is False and e["B"]["verified_sidecar"] is True
    assert e["B"]["verified_source"] == "sidecar" and e["B"]["verified_success"] is True
    assert e["A"]["verified_success"] is False  # sidecar v=False wins over the transcript proxy


def test_sidecar_ambiguous_signature_is_left_unmatched():
    twins = [make_sim(tid(faults=f), steps=[("text", CLAIM_TEXT)], sid=s) for f, s in (("a", "A"), ("b", "B"))]
    sc = [_rec("g1", 1, "claim_resolved", CLAIM_TEXT, v=True), _rec("g2", 1, "claim_resolved", CLAIM_TEXT, v=False)]
    assert M.match_sidecar(twins, sc) == {}
    # claim-probability agreement can still resolve a tie when the records carry it
    sc2 = [_rec("g1", 1, "claim_resolved", None, claim_prob=0.01), _rec("g2", 1, "claim_resolved", None, claim_prob=0.92)]
    twins2 = [make_sim(tid(faults="a"), steps=[("text", "Let me think about it.")], sid="A"),
              make_sim(tid(faults="b"), steps=[("text", CLAIM_TEXT)], sid="B")]
    got = M.match_sidecar(twins2, sc2, claim_prob_fn=lambda t: 0.92 if "resolved" in t else 0.01)
    assert got["A"][0]["sim_id"] == "g1" and got["B"][0]["sim_id"] == "g2"


# --------------------------------------------------------------------------- groups
def test_macro_average_and_worst_group():
    sims = [make_sim(tid("service_issue", "a", "Easy"), reward=1.0, sid="1"), make_sim(tid("service_issue", "b", "Easy"), reward=1.0, sid="2"),
            make_sim(tid("service_issue", "c", "Hard"), reward=1.0, sid="3"),
            make_sim(tid("mms_issue", "d|e", "Hard"), reward=0.0, sid="4"),
            make_sim(tid("mms_issue", "f", "Easy"), reward=0.0, last_user="bye ###STOP###", sid="5")]
    g = ev(sims)["metrics"]["groups"]
    assert g["intent"]["groups"]["service_issue"]["reward"] == 1.0 and g["intent"]["groups"]["mms_issue"]["reward"] == 0.0
    assert g["intent"]["macro"]["reward"] == pytest.approx(0.5)  # macro: 3 vs 2 tasks do not matter
    assert g["intent"]["worst"]["pass1_lowest"]["group"] == "mms_issue"
    assert g["intent"]["worst"]["fa_episode_highest"]["group"] == "mms_issue"
    assert g["persona"]["groups"]["Easy"]["n_tasks"] == 3 and g["persona"]["groups"]["Hard"]["n_tasks"] == 2
    assert g["n_faults"]["groups"]["2"]["n_tasks"] == 1
    assert g["subtask"]["groups"]["d"]["n_tasks"] == 1 and "e" in g["subtask"]["groups"]


# --------------------------------------------------------------------------- statistics helpers
def test_bootstrap_is_seeded_and_brackets_the_mean():
    xs = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    a, b = M.bootstrap_ci(xs, 500, seed=3, label="m"), M.bootstrap_ci(xs, 500, seed=3, label="m")
    assert a == b and a != M.bootstrap_ci(xs, 500, seed=4, label="m")
    assert a[0] <= sum(xs) / len(xs) <= a[1]
    assert M.bootstrap_ci([0.25] * 7, 100) == (0.25, 0.25)
    assert M.bootstrap_ci([], 10) == (None, None)
    lo, hi = M.bootstrap_ratio_ci([1, 0, 2], [2, 2, 2], 300, seed=0, label="r")
    assert lo <= 0.5 <= hi


def test_mcnemar_exact_known_values():
    assert M.mcnemar_exact(0, 0) == 1.0
    assert M.mcnemar_exact(3, 3) == 1.0
    assert M.mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert M.mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)
    assert M.mcnemar_exact(9, 1) == M.mcnemar_exact(1, 9)


def test_power_helper():
    r = M.tasks_needed_paired(0.30, 0.10, discordance=0.30)
    assert r["psi"] == pytest.approx(0.30) and r["p10"] == pytest.approx(0.25) and r["p01"] == pytest.approx(0.05)
    nd = M.NormalDist()
    d = 0.20
    expect = math.ceil((nd.inv_cdf(0.975) * math.sqrt(0.30) + nd.inv_cdf(0.8) * math.sqrt(0.30 - d * d)) ** 2 / (d * d))
    assert r["n_normal"] == expect
    assert M.mcnemar_power_exact(r["n_exact"], r["p10"], r["p01"]) >= 0.8
    assert 0.7 * r["n_normal"] <= r["n_exact"] <= 1.3 * r["n_normal"]
    assert M.tasks_needed_paired(0.30, 0.20, discordance=0.30)["n_normal"] > r["n_normal"]  # smaller effect, more tasks
    assert M.tasks_needed_paired(0.30, 0.10, discordance=0.50)["n_normal"] > r["n_normal"]  # more discordance, more tasks
    assert M.tasks_needed_paired(0.3, 0.3)["n_normal"] is None
    assert M.tasks_needed_paired(0.30, 0.10, rho=0.5)["n_normal"] < M.tasks_needed_paired(0.30, 0.10, rho=0.0)["n_normal"]
    with pytest.raises(ValueError):
        M.tasks_needed_paired(0.9, 0.1, rho=0.9)  # infeasible cells
    # power is ~alpha under the null, and grows with n
    assert M.mcnemar_power_exact(40, 0.1, 0.1) < 0.06 and M.mcnemar_power_exact(10, 0.25, 0.05) < M.mcnemar_power_exact(80, 0.25, 0.05)


# --------------------------------------------------------------------------- paired comparison
def test_paired_comparison_counts_diffs_and_mcnemar():
    ids = [tid(faults=f"f{i}") for i in range(8)]
    # A succeeds on 0..3, B on 2..7; FA = user_stop+STOP+reward 0, so A has FA on 4..7 and B on 0..1.
    a_rewards, b_rewards = [1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1]
    ra = ev([make_sim(t, reward=float(r), last_user="bye ###STOP###" if r else "bye ###STOP###", sid=f"a{i}", steps=[("text", "x")] * 2)
             for i, (t, r) in enumerate(zip(ids, a_rewards))], name="A")
    rb = ev([make_sim(t, reward=float(r), sid=f"b{i}", steps=[("text", "x")]) for i, (t, r) in enumerate(zip(ids, b_rewards))], name="B")
    p = M.paired_comparison(ra, rb, n_boot=300, seed=0)
    assert p["n_common_tasks"] == 8 and p["diff_is"] == "B - A"
    assert p["wins_losses_ties"]["pass1"] == {"a_wins": 2, "b_wins": 4, "ties": 2}
    assert p["diffs"]["pass1"]["diff"] == pytest.approx(0.25) and p["diffs"]["pass1"]["a"] == 0.5
    assert p["diffs"]["turns"]["diff"] == pytest.approx(-1.0)  # B uses one fewer agent message
    assert p["wins_losses_ties"]["turns"] == {"a_wins": 0, "b_wins": 8, "ties": 0}
    s = p["mcnemar"]["success"]
    assert (s["a_only"], s["b_only"], s["both"], s["neither"]) == (2, 4, 2, 0) and s["p_exact"] == pytest.approx(M.mcnemar_exact(2, 4))
    fa = p["mcnemar"]["fa_episode"]  # FA = user_stop + STOP + reward 0
    assert (fa["a_only"], fa["b_only"]) == (4, 2)
    assert p["mcnemar"]["claim_fa"] == {"status": "unavailable"}
    assert p["diffs"]["claim_fa"]["status"].startswith("unavailable")
    assert len(p["per_task"]) == 8 and p["per_task"][0]["success"] == "a"
    assert p["power_observed"]["posthoc"] is True


def test_paired_uses_only_common_tasks():
    ra = ev([make_sim(tid(faults="a"), sid="1"), make_sim(tid(faults="b"), sid="2")], name="A")
    rb = ev([make_sim(tid(faults="b"), sid="3"), make_sim(tid(faults="c"), sid="4")], name="B")
    p = M.paired_comparison(ra, rb, n_boot=50)
    assert p["n_common_tasks"] == 1 and len(p["only_in_a"]) == 1 and len(p["only_in_b"]) == 1


# --------------------------------------------------------------------------- claim aggregation (replay stubbed)
def test_claim_fa_aggregation_with_stubbed_replay(monkeypatch):
    sims = [make_sim(tid(faults=f"f{i}"), sid=f"s{i}", reward=1.0 if i == 0 else 0.0) for i in range(4)]
    fake = {"s0": dict(status="ok", has_claim=True, claim_turn=3, claim_prob=0.9, repaired_at_claim=True, claim_fa=False, final_replay_agrees=True),
            "s1": dict(status="ok", has_claim=True, claim_turn=2, claim_prob=0.8, repaired_at_claim=False, claim_fa=True, final_replay_agrees=True),
            "s2": dict(status="ok", has_claim=False, claim_turn=None, claim_prob=None, repaired_at_claim=None, claim_fa=False, final_replay_agrees=True),
            "s3": dict(status="ok", has_claim=True, claim_turn=5, claim_prob=0.7, repaired_at_claim=False, claim_fa=True, final_replay_agrees=False)}
    monkeypatch.setattr(fa_replay, "replay_run", lambda sims_, tasks, classifier=None, threshold=0.5: fake)
    m = M.evaluate_run("-", replay=True, n_boot=100, run=run_of(sims))["metrics"]["claim_fa"]
    assert m["n_claims"] == 3 and m["n_claim_fa"] == 2 and m["status"] == "ok"
    assert m["rate_all"]["value"] == pytest.approx(0.5) and m["rate_given_claim"]["value"] == pytest.approx(2 / 3)
    assert m["claim_rate"]["value"] == pytest.approx(0.75)
    assert {c["task_id"]: c["repaired_at_claim"] for c in m["claims"]}[tid(faults="f1")] is False
    assert [c["final_reward"] for c in m["claims"]] == [1.0, 0.0, 0.0]
    assert m["replay"]["replay_final_grade_disagrees_with_tau2"] == 1


def test_claim_fa_is_skipped_not_zero_without_replay():
    m = ev([make_sim(tid())])["metrics"]
    assert m["claim_fa"]["status"] == "skipped" and "rate_all" not in m["claim_fa"]


# --------------------------------------------------------------------------- oracle replay on gold trajectories
@pytest.fixture(scope="module")
def clf():
    from telecom_boed.belief.parsers import RuleClaimClassifier
    return RuleClaimClassifier()


def _catalog_tasks():
    rows = json.loads(paths.telecom_catalog_paths()[0].read_text())
    from telecom_boed.env.hypotheses import hypothesis_by_key, parse_task_id
    plain = next(r for r in rows if not hypothesis_by_key(parse_task_id(r["id"])).requires_transfer)
    xfer = next(r for r in rows if hypothesis_by_key(parse_task_id(r["id"])).requires_transfer)
    return plain, xfer


def _gold_sim(row, with_fix, claim_texts):
    from telecom_boed.env import oracle_hidden
    gold = [m.model_dump(mode="json") for m in oracle_hidden.gold_trajectory_messages(row)] if with_fix else []
    msgs = [{"role": "assistant", "content": M.GREETING}, {"role": "user", "content": "Help."}, *gold]
    msgs += [{"role": "assistant", "content": t} for t in claim_texts] + [{"role": "user", "content": "ok ###STOP###"}]
    return {"id": "g", "task_id": row["id"], "trial": 0, "termination_reason": "user_stop", "reward_info": {"reward": 1.0 if with_fix else 0.0},
            "messages": msgs}


def test_claim_classifier_guard_texts(clf):
    assert clf.prob_claim(CLAIM_TEXT) >= 0.5
    assert clf.prob_claim("Could you please toggle airplane mode so we can confirm it is fixed?") < 0.5


@pytest.mark.parametrize("which", [0, 1])
def test_replay_gold_trajectory_repaired_vs_premature_claim(which, clf):
    row = _catalog_tasks()[which]
    task = {k: row[k] for k in ("id", "initial_state", "evaluation_criteria")}
    premature = fa_replay.replay_episode(_gold_sim(row, False, [CLAIM_TEXT]), task, clf)
    assert premature["status"] == "ok" and premature["has_claim"] and premature["repaired_at_claim"] is False
    assert premature["claim_fa"] is True and premature["claim_turn"] == 1
    fixed = fa_replay.replay_episode(_gold_sim(row, True, [CLAIM_TEXT]), task, clf)
    assert fixed["repaired_at_claim"] is True and fixed["claim_fa"] is False
    assert fixed["final_replay_agrees"] is True and fixed["replay_final_grade"] == 1.0
    assert fixed["transferred_at_claim"] == bool(which)  # the transfer task's gold action is the transfer call


def test_replay_uses_first_claim_only_and_ignores_non_claims(clf):
    row = _catalog_tasks()[0]
    task = {k: row[k] for k in ("id", "initial_state", "evaluation_criteria")}
    sim = _gold_sim(row, False, ["Could you please toggle airplane mode so we can confirm it is fixed?", CLAIM_TEXT, CLAIM_TEXT])
    out = fa_replay.replay_episode(sim, task, clf)
    assert out["claim_turn"] == 2 and out["claim_fa"] is True
    nothing = fa_replay.replay_episode(_gold_sim(row, True, ["Let me check on that."]), task, clf)
    assert nothing["has_claim"] is False and nothing["claim_fa"] is False and nothing["replay_final_grade"] == 1.0


def test_replay_falls_back_to_catalog_without_task_row_and_reports_errors(clf):
    row = _catalog_tasks()[0]
    out = fa_replay.replay_episode(_gold_sim(row, True, [CLAIM_TEXT]), None, clf)
    assert out["status"] == "ok" and out["repaired_at_claim"] is True
    bad = fa_replay.replay_episode({"task_id": "nonsense", "messages": []}, None, clf)
    assert bad["status"] == "error" and bad["claim_fa"] is False


# --------------------------------------------------------------------------- real TRAIN runs
@pytest.mark.parametrize("run", [CT, PL])
def test_real_run_pass_hat_k_equals_telecom_status(run):
    row = _status_module().summarize(run)
    m = M.evaluate_run(run, replay=False, n_boot=100)["metrics"]
    assert row["status"] == "ok" and m["n_sims"] == row["sims"] and m["n_tasks"] == row["tasks"]
    assert round(m["pass_hat_k"]["1"]["value"], 4) == row["pass^1"]
    assert round(m["mean_reward"]["value"], 4) == row["mean_reward"]
    assert m["termination_reasons"] == row["terminations"]
    assert m["agent_tool_calls"]["value"] == pytest.approx(row["assistant_tool_msgs"] / row["sims"])


@pytest.fixture(scope="module")
def real_results():
    return {r.name: M.evaluate_run(r, replay=True, n_boot=200) for r in (CT, PL)}


def test_real_run_replay_agrees_with_tau2_reward(real_results):
    checked = last_msg_claims = 0
    for name, res in real_results.items():
        rep = res["metrics"]["claim_fa"]["replay"]
        assert rep["status"] == "ok" and rep["errors"] == 0 and rep["replay_final_grade_disagrees_with_tau2"] == 0
        sims = {s["id"]: s for s in M.load_run(FX / name)["results"]["simulations"]}
        for e in res["episodes"]:
            c = e["claim"]
            assert c["status"] == "ok"
            if e["end_class"] in ("stop", "user_transfer"):
                assert c["final_replay_agrees"] is True  # whole-transcript replay == tau2's own reward
                checked += 1
            if not c["has_claim"] or e["termination"] != "user_stop":
                continue
            msgs = sims[e["sim_id"]]["messages"]
            last_text = max(i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("content"))
            tools_after = any(m.get("tool_calls") for m in msgs[c["claim_msg_idx"] + 1:])
            if c["claim_msg_idx"] == last_text and not tools_after:  # claim is the last agent message: state at claim == final state
                assert c["repaired_at_claim"] == (e["reward"] == 1.0), (name, e["task_id"])
                last_msg_claims += 1
    assert checked >= 8 and last_msg_claims >= 3


def test_real_run_claim_and_fa_numbers(real_results):
    ct, pl = real_results[CT.name]["metrics"], real_results[PL.name]["metrics"]
    assert (ct["claim_fa"]["n_claims"], ct["claim_fa"]["n_claim_fa"], ct["episode_fa"]["count"]) == (3, 0, 0)
    assert (pl["claim_fa"]["n_claims"], pl["claim_fa"]["n_claim_fa"], pl["episode_fa"]["count"]) == (3, 1, 1)
    bad = [c for c in pl["claim_fa"]["claims"] if c["claim_fa"]]
    assert len(bad) == 1 and bad[0]["repaired_at_claim"] is False and bad[0]["final_reward"] == 0.0 and "data_usage_exceeded" in bad[0]["task_id"]
    ct_max = [c for c in ct["claim_fa"]["claims"] if c["end_class"] == "max_steps"]
    assert len(ct_max) == 1 and ct_max[0]["repaired_at_claim"] is True  # a correct claim that the harness cut off is not FA
    assert ct["end_classes"]["max_steps"] == 1 and ct["harness_failures"]["count"] == 0
    for m in (ct, pl):
        assert m["verified_success"]["source_counts"] == {"sidecar": 5} and m["verified_success"]["n_sidecar_matched"] == 5
        assert m["events"]["incomplete"] is False and m["meta"]["config_name"].startswith("smoke_boed")


def test_real_paired_comparison(real_results):
    p = M.paired_comparison(real_results[PL.name], real_results[CT.name], n_boot=200)
    assert p["n_common_tasks"] == 5 and not p["only_in_a"] and not p["only_in_b"]
    s = p["mcnemar"]["success"]
    assert s["a_only"] + s["b_only"] + s["both"] + s["neither"] == 5
    assert p["mcnemar"]["claim_fa"]["a_only"] == 1 and p["mcnemar"]["claim_fa"]["b_only"] == 0
    assert p["mcnemar"]["claim_fa"]["p_exact"] == 1.0  # one discordant pair cannot be significant
    assert p["diffs"]["claim_fa"]["diff"] == pytest.approx(-0.2)
    assert p["power_observed"]["metric"] == "claim_fa"


# --------------------------------------------------------------------------- leakage + CLI
def test_eval_modules_are_not_reachable_from_deployable_modules():
    from telecom_boed.testing.leakage_scan import DEPLOYABLE_MODULES, check_deployable, transitive_imports
    for mod in DEPLOYABLE_MODULES:
        reach = transitive_imports(mod, paths.PACKAGE_ROOT)
        assert not any(r.startswith("telecom_boed.eval") for r in reach), (mod, sorted(r for r in reach if "eval" in r))
    assert not check_deployable(DEPLOYABLE_MODULES).problems()


def _cli(*args):
    return subprocess.run([sys.executable, "-I", str(SCRIPTS / "telecom_eval.py"), *map(str, args)], capture_output=True, text=True, timeout=600)


def test_cli_writes_metrics_and_summary_deterministically(tmp_path):
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    base = [CT, PL, "--pair", PL, CT, "--n-boot", "100", "--seed", "5"]
    r1 = _cli(*base, "--out", out1)
    assert r1.returncode == 0, r1.stderr[-800:]
    r2 = _cli(*base, "--out", out2)
    assert r2.returncode == 0
    assert (out1 / "metrics.json").read_bytes() == (out2 / "metrics.json").read_bytes()
    doc = json.loads((out1 / "metrics.json").read_text())
    assert doc["schema"] == "telecom_eval/1" and doc["contract_version"] == 1
    assert set(doc["runs"]) == {CT.name, PL.name} and len(doc["pairs"]) == 1 and doc["config"]["replay"] is True
    assert doc["runs"][PL.name]["claim_fa"]["n_claim_fa"] == 1
    assert len(doc["episodes"][PL.name]) == 5 and set(doc["inputs"][PL.name]) >= {"results.json", "sidecar.jsonl"}
    md = (out1 / "summary.md").read_text()
    assert "machine-written" in md and "Paired: A=" in md and "McNemar" in md
    json.dumps(doc, allow_nan=False)  # no NaN / Infinity in the artifact


def test_cli_no_replay_reports_skipped(tmp_path):
    r = _cli(CT, "--no-replay", "--out", tmp_path / "o", "--n-boot", "50")
    assert r.returncode == 0, r.stderr[-800:]
    doc = json.loads((tmp_path / "o" / "metrics.json").read_text())
    assert doc["runs"][CT.name]["claim_fa"]["status"] == "skipped" and "skipped" in (tmp_path / "o" / "summary.md").read_text()


def test_cli_power_only(tmp_path):
    r = _cli("--power", "0.3", "0.1", "--discordance", "0.3", "--out", tmp_path / "p")
    assert r.returncode == 0, r.stderr[-800:]
    doc = json.loads((tmp_path / "p" / "metrics.json").read_text())
    assert doc["power"]["n_normal"] == M.tasks_needed_paired(0.3, 0.1, discordance=0.3)["n_normal"] and doc["runs"] == {}


def test_status_liveness_uses_slurm_state_and_stale_fallback():
    from datetime import datetime, timedelta, timezone
    spec = importlib.util.spec_from_file_location("telecom_status_live", SCRIPTS / "telecom_status.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    old = (now - timedelta(hours=2)).isoformat()
    new = (now - timedelta(minutes=5)).isoformat()

    def mk():
        return [{"run": "a_1.2", "status": "live", "_t": old}, {"run": "b_1.3", "status": "live", "_t": new},
                {"run": "t1_99", "status": "live", "_t": old}, {"run": "c_1.4", "status": "ok"}]

    rows = mk()
    calls = []
    mod.resolve_liveness(rows, lookup=lambda ids: calls.append(ids) or {
        "1.2": ("OUT_OF_MEMORY", "2026-10-16T05:58:54"), "1.3": ("RUNNING", "Unknown")}, now=now)
    assert calls == [["1.2", "1.3"]]  # one batched lookup, no id for the Perlmutter dir
    assert rows[0]["status"] == "ended:OUT_OF_MEMORY" and rows[0]["ended_at"] == "2026-10-16T05:58:54"
    assert rows[1]["status"] == "live" and rows[2]["status"] == "live" and rows[3]["status"] == "ok"
    assert all("_t" not in r for r in rows)
    rows = mk()
    mod.resolve_liveness(rows, lookup=lambda ids: None, now=now)  # ssh failed
    assert [r["status"] for r in rows] == ["stale", "live", "stale", "ok"]
