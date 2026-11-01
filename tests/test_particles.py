"""[telecom_boed] W1-S2 tests: catalog hypotheses, particle build/step/merge, exact grader vs tau2's evaluator,
featurizer coverage. Only TRAIN-split tasks are test inputs (split key "train"); the catalog-wide counts are the
one exception (they read every shipped row but never run a test-40 task)."""
from __future__ import annotations

import copy
import json
import random
import resource
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from loguru import logger
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import EnvFunctionCall, Task
from tau2.domains.telecom.environment import get_environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

from telecom_boed import paths
from telecom_boed.contracts import DeviceAction, Grader, Particle, _frozen_items
from telecom_boed.env import featurize as F
from telecom_boed.env import oracle_hidden as O
from telecom_boed.env import particles as P
from telecom_boed.env import snapshot as S
from telecom_boed.env.hypotheses import hypothesis_by_key, load_catalog, load_support, parse_task_id
from telecom_boed.testing.leakage_scan import scan_module_source, transitive_imports

TIMINGS = paths.PACKAGE_ROOT / "results" / "timings" / "W1-S2.json"
SRC = paths.PACKAGE_ROOT / "src" / "telecom_boed" / "env"


def D(req: str, tool: str, **args) -> DeviceAction:
    return DeviceAction(req, tool, _frozen_items(args))


def U(tool: str, **args) -> DeviceAction:
    return D("user", tool, **args)


@pytest.fixture(scope="module")
def train_tasks(catalog, train_ids) -> dict[str, dict]:
    by_id = {t["id"]: t for t in catalog}
    return {i: by_id[i] for i in train_ids}  # never touches the test split


def intent_of(task_id: str) -> str:
    return parse_task_id(task_id)[0]


def is_transfer(task: dict) -> bool:
    return "ACTION" in task["evaluation_criteria"]["reward_basis"]


# --------------------------------------------------------------------------- catalog
def test_catalog_keys_unique_and_action_tasks_transfer_only(catalog):
    hyps = load_catalog()
    assert len(catalog) == 2285 and len(hyps) == 2285
    assert len({h.key for h in hyps}) == 2285
    assert all(len(h.task_ids) == 1 for h in hyps), "a key maps to >1 task"
    assert Counter(h.intent for h in hyps) == {"mms_issue": 1984, "mobile_data_issue": 254, "service_issue": 47}
    assert {len(load_support(i)) for i in ("mms_issue",)} == {1984}
    assert sum(h.requires_transfer for h in hyps) == 32
    by_id = {t["id"]: t for t in catalog}
    n_action = 0
    for h in hyps:
        t = by_id[h.task_ids[0]]
        crit = t["evaluation_criteria"]
        assert h.requires_transfer == ("ACTION" in crit["reward_basis"])
        assert crit["reward_basis"] in (["ENV_ASSERTION"], ["ENV_ASSERTION", "ACTION"])
        if h.requires_transfer:
            n_action += 1
            assert [(a["name"], a["compare_args"]) for a in crit["actions"]] == [("transfer_to_human_agents", [])]
        assert len(h.init_actions) == len(t["initial_state"]["initialization_actions"]) and h.init_actions[0].func_name == "set_user_info"
        assert all(a.assert_value is True for a in h.goal_assertions)
    assert n_action == 32
    # 32 fault sets occur under >1 intent: the intent in the key keeps them apart (and their goals differ)
    by_faults: dict = {}
    for h in hyps:
        by_faults.setdefault(h.faults, []).append(h)
    multi = [v for v in by_faults.values() if len(v) > 1]
    assert len(multi) == 32 and all(len({x.goal_id for x in v}) == len(v) for v in multi)
    assert sum(h.is_test40 for h in hyps) == 40


def test_modules_pass_leakage_scan():
    for m in ("hypotheses", "particles", "featurize"):
        assert scan_module_source(SRC / f"{m}.py") == [], m
        closure = transitive_imports(f"telecom_boed.env.{m}", SRC.parents[1])
        assert not any("oracle_" in x for x in closure), (m, closure)


# --------------------------------------------------------------------------- build_state
def test_build_state_matches_tau2_init(train_tasks):
    rng = random.Random(0)
    ids = sorted(train_tasks)
    pick = []
    for it, n in (("mms_issue", 20), ("mobile_data_issue", 15), ("service_issue", 14)):
        pick += rng.sample([i for i in ids if intent_of(i) == it], n)
    pick.append([i for i in ids if is_transfer(train_tasks[i])][0])
    assert len(pick) == 50
    for tid in pick:
        hyp = hypothesis_by_key(parse_task_id(tid))
        snap = P.build_state(hyp)
        task = Task.model_validate(train_tasks[tid])
        env = get_environment()
        env.set_state(initialization_data=None, initialization_actions=task.initial_state.initialization_actions,
                      message_history=[])
        assert snap.key == S.state_key(env), tid
        assert (snap.key[0], snap.key[1]) == (env.get_db_hash(), env.get_user_db_hash()), tid
        assert P.build_state(hyp) is snap  # cached


# --------------------------------------------------------------------------- grader vs tau2
def _sim(task_id: str, msgs: list, reason: TerminationReason = TerminationReason.USER_STOP) -> SimulationRun:
    return SimulationRun(id="t", task_id=task_id, start_time="2026-01-01T00:00:00", end_time="2026-01-01T00:00:01",
                         duration=1.0, termination_reason=reason, messages=msgs)


def _tau2_reward(task_row: dict, msgs: list, reason: TerminationReason = TerminationReason.USER_STOP) -> float:
    task = Task.model_validate(task_row)
    return evaluate_simulation(_sim(task.id, msgs, reason), task, EvaluationType.ALL, solo_mode=False,
                               domain="telecom").reward


def _mine(task_row: dict, k_actions: list[DeviceAction]) -> float:
    """grade() of the state reached by replaying `k_actions` through apply_calls (one call at a time)."""
    hyp = hypothesis_by_key(O.true_key(task_row))
    st = P.build_state(hyp)
    for a in k_actions:
        st, _ = P.apply_calls(st, (a,))
    return P.grade(st, hyp, any(a.tool == "transfer_to_human_agents" for a in k_actions))


def _pick_20(train_tasks) -> list[str]:
    rng = random.Random(1)
    ids = sorted(train_tasks)
    tr = [i for i in ids if is_transfer(train_tasks[i])]
    assert len(tr) == 8
    rest = [i for i in ids if i not in tr]
    return sorted(rng.sample(tr, 4) + rng.sample(rest, 16))


def test_grade_matches_tau2_evaluator(train_tasks):
    ids = _pick_20(train_tasks)
    n_gold1 = n_transfer = n_neg_equal = n_prefix = n_prefix_pos = n_drop_pos = 0
    max_steps_zero = 0
    for tid in ids:
        row = train_tasks[tid]
        msgs = O.gold_trajectory_messages(row)
        acts = list(O.gold_actions(row))
        assert len(msgs) == 2 * len(acts)
        r_tau2, r_mine = _tau2_reward(row, msgs), _mine(row, acts)
        assert r_tau2 == r_mine, (tid, r_tau2, r_mine)
        n_gold1 += r_tau2 == 1.0
        n_transfer += is_transfer(row)
        # every prefix of the gold trajectory (covers transfer-missing and partial-repair cases)
        for k in range(len(acts)):
            a, b = _tau2_reward(row, msgs[: 2 * k]), _mine(row, acts[:k])
            assert a == b, (tid, k, a, b)
            n_prefix += 1
            n_prefix_pos += a == 1.0
        # (i) drop one gold user action
        user_ix = [i for i, a in enumerate(acts) if a.requestor == "user"]
        if user_ix:
            j = random.Random(tid).choice(user_ix)
            acts_d = acts[:j] + acts[j + 1:]
            a, b = _tau2_reward(row, O.messages_for(O.true_key(row), acts_d)), _mine(row, acts_d)
            assert a == b, (tid, "drop", j)
            n_neg_equal += 1
            n_drop_pos += a == 1.0
        # (ii) transfer task without the transfer call
        if is_transfer(row):
            assert _tau2_reward(row, []) == 0.0 and _mine(row, []) == 0.0
        # (iii) tau2 gives 0 for a non-STOP ending, regardless of state; grade() assumes USER/AGENT_STOP
        if _tau2_reward(row, msgs, TerminationReason.MAX_STEPS) == 0.0:
            max_steps_zero += 1
    assert n_transfer >= 3 and max_steps_zero == len(ids)
    print(f"\n[W1-S2] gold trajectories with tau2 reward 1: {n_gold1}/{len(ids)}; transfer tasks {n_transfer}; "
          f"prefixes checked {n_prefix} (reward 1 in {n_prefix_pos}); drop-one controls {n_neg_equal} (reward 1 in {n_drop_pos})")
    assert n_gold1 >= 1


def test_particle_grader_protocol(train_tasks):
    g = P.ParticleGrader()
    assert isinstance(g, Grader)
    tid = next(i for i in sorted(train_tasks) if is_transfer(train_tasks[i]))
    key = parse_task_id(tid)
    assert g.requires_transfer(key)
    st = P.build_state(hypothesis_by_key(key))
    assert g.grade(st, key, False) == 0.0
    # service transfer tasks: assert_service_status(no_service) is already true in the broken state
    assert g.grade(st, key, True) == 1.0
    nkey = parse_task_id(next(i for i in sorted(train_tasks) if not is_transfer(train_tasks[i])))
    assert not g.requires_transfer(nkey) and g.grade(P.build_state(hypothesis_by_key(nkey)), nkey, True) == 0.0


# --------------------------------------------------------------------------- featurize
USER_ARGS: dict[str, list[dict]] = {
    "set_network_mode_preference": [{"mode": "4g_5g_preferred"}, {"mode": "bogus"}],
    "set_apn_settings": [{"apn_settings": {"apn_name": "internet", "reset_at_reboot": False, "mms_apn": "mms"}}],
    "check_app_status": [{"app_name": "messaging"}, {"app_name": "browser"}, {"app_name": "nope"}],
    "check_app_permissions": [{"app_name": "messaging"}, {"app_name": "nope"}],
    "grant_app_permission": [{"app_name": "messaging", "permission": "sms"}, {"app_name": "messaging", "permission": "bogus"},
                             {"app_name": "nope", "permission": "sms"}],
}
AGENT_ARGS: dict[str, list[dict]] = {
    "get_customer_by_phone": [{"phone_number": "555-123-2002"}],
    "get_customer_by_id": [{"customer_id": "C1001"}],
    "get_customer_by_name": [{"full_name": "John Smith", "dob": "1985-06-15"}],
    "get_details_by_id": [{"id": i} for i in ("C1001", "L1002", "B1001", "P1002", "D1002")],
    "suspend_line": [{"customer_id": "C1001", "line_id": "L1002", "reason": "x"}],
    "resume_line": [{"customer_id": "C1001", "line_id": "L1002"}],
    "get_bills_for_customer": [{"customer_id": "C1001"}],
    "send_payment_request": [{"customer_id": "C1001", "bill_id": "B1001"}],
    "get_data_usage": [{"customer_id": "C1001", "line_id": "L1002"}],
    "enable_roaming": [{"customer_id": "C1001", "line_id": "L1002"}],
    "disable_roaming": [{"customer_id": "C1001", "line_id": "L1002"}],
    "transfer_to_human_agents": [{"summary": "x"}],
    "refuel_data": [{"customer_id": "C1001", "line_id": "L1002", "gb_amount": 2.0}],
}


def _all_calls(tools: tuple[str, ...], args: dict, req: str) -> list[DeviceAction]:
    return [D(req, t, **a) for t in tools for a in (args.get(t) or [{}])]


def _states(train_tasks, n_per_intent=3) -> list:
    out = []
    for it in ("mms_issue", "mobile_data_issue", "service_issue"):
        for tid in sorted(i for i in train_tasks if intent_of(i) == it)[:n_per_intent]:
            st = P.build_state(hypothesis_by_key(parse_task_id(tid)))
            out.append(st)
            out.append(P.apply_calls(st, (U("toggle_airplane_mode"), U("toggle_data"), U("toggle_roaming"), U("connect_vpn")))[0])
    out.append(P.apply_calls(out[0], (D("assistant", "suspend_line", customer_id="C1001", line_id="L1002", reason="x"),))[0])
    return out


def test_featurize_covers_all_user_tools(train_tasks):
    env = S.fast_env()
    assert set(F.USER_TOOLS) == set(env.user_tools.get_tools()) and len(F.USER_TOOLS) == 30
    assert set(F.AGENT_TOOLS) == set(env.tools.get_tools()) and len(F.AGENT_TOOLS) == 13
    seen_fields: dict[str, set] = {}
    calls = _all_calls(F.USER_TOOLS, USER_ARGS, "user") + _all_calls(F.AGENT_TOOLS, AGENT_ARGS, "assistant")
    n = 0
    for st in _states(train_tasks):
        for c in calls:
            _, (out,) = P.apply_calls(st, (c,))
            f = F.featurize(c.tool, out)
            assert f, (c.tool, out)
            assert not any(k.endswith(".raw") for k in f), (c.tool, out)
            assert F.featurize(c.tool, out) == f  # deterministic
            if "error" in f:  # state-dependent tool errors (resume an active line, ...) and the deliberate bad inputs
                assert f == {"error": True, "error.msg": out[len("Error: "):]}, (c.tool, out)
            else:
                seen_fields.setdefault(c.tool, set()).update(f)
            n += 1
    for t in F.USER_TOOLS + F.AGENT_TOOLS:
        assert seen_fields.get(t), f"no non-error featurization of {t}"
    # known outputs
    st = P.build_state(hypothesis_by_key(parse_task_id(sorted(train_tasks)[0])))
    f = F.featurize("check_status_bar", "Status Bar: 📶³ Good | 5G | 📱 Data Enabled | 🔽 Data Saver | 🔒 VPN Connected | 🔋 80%")
    assert (f["status_bar.signal"], f["status_bar.tech"], f["status_bar.data"], f["status_bar.data_saver"], f["status_bar.vpn"]) == ("good", "5G", "enabled", True, True)
    assert F.featurize("check_status_bar", "Status Bar: ✈️ Airplane Mode | 🔋 80%")["status_bar.airplane_mode"] is True
    assert F.featurize("x_unknown", "hello") == {"x_unknown.raw": "hello"}
    assert F.featurize("get_customer_by_id", "Error: Customer with ID C9 not found") == {"error": True, "error.msg": "Customer with ID C9 not found"}
    assert F.featurize("can_send_mms", "Your messaging app can send MMS messages.") == {"mms.can_send": True}
    d = F.featurize("run_speed_test", "Speed Test Result: 27.50 Mbps (Good). Connection is good for most activities, including HD streaming.")
    assert d == {"speed_test.ok": True, "speed_test.mbps": 27.5, "speed_test.desc": "Good"}
    print(f"\n[W1-S2] featurize calls: {n}; groups: " + ", ".join(
        f"{g}:{len({k for ks in seen_fields.values() for k in ks if k.split('.')[0] == g})}" for g in sorted({k.split('.')[0] for ks in seen_fields.values() for k in ks})))
    (Path(__file__).parent / "fixtures" / "W1-S2" / "field_inventory.json").write_text(
        json.dumps({t: sorted(v) for t, v in sorted(seen_fields.items())}, indent=1, ensure_ascii=False) + "\n")


def test_label_space_closed():
    DOM = {
        "check_status_bar": {"status_bar.signal": {"hidden", "none", "poor", "fair", "good", "excellent"},
                             "status_bar.tech": {"hidden", "none", "2G", "3G", "4G", "5G"},
                             "status_bar.data": {"hidden", "enabled", "disabled"}, "status_bar.airplane_mode": {True, False}},
        "check_network_status": {"network.sim_status": {"active", "missing", "locked_pin", "locked_puk"},
                                 "network.connection": {"connected", "no_service", "searching"},
                                 "network.tech": {"none", "2G", "3G", "4G", "5G"}},
        "run_speed_test": {"speed_test.desc": {"Very Poor", "Poor", "Fair", "Good", "Excellent", "No Connection"}},
        "can_send_mms": {"mms.can_send": {True, False}},
        "check_apn_settings": {"apn.name": {"internet", "broken", None}, "apn.mmsc_set": {True, False}},
    }
    sup = load_catalog()
    parts = P.initial_particles(sup)
    for tool, dom in DOM.items():
        ps2, outs = P.step(parts, (U(tool),), merge=False)
        assert len(outs) == len(parts)
        contents = [o[0] for o in outs]
        ls = F.label_space(contents, tool)
        assert ls and ls == F.label_space(reversed(contents), tool)  # order independent
        for field, vals in ls.items():
            assert vals and len(vals) == len(set(vals))  # no duplicates, non-empty
            assert vals == sorted(vals, key=F._sortkey)
            seen = {F.featurize(tool, c).get(field) for c in contents if field in F.featurize(tool, c)}
            assert set(vals) == seen  # exactly the union over particles
        for field, allowed in dom.items():
            assert set(ls[field]) <= allowed, (tool, field, ls[field])
    # closed under one more action: outputs after toggling airplane mode stay inside the same domains
    p2, _ = P.step(parts, (U("toggle_airplane_mode"),), merge=False)
    for tool in ("check_status_bar", "check_network_status", "run_speed_test"):
        _, outs = P.step(p2, (U(tool),), merge=False)
        ls = F.label_space([o[0] for o in outs], tool)
        for field, allowed in DOM[tool].items():
            assert set(ls[field]) <= allowed, (tool, field, ls[field])
    assert F.label_space(["Error: x"], "check_status_bar") == {"error": [True], "error.msg": ["x"]}


# --------------------------------------------------------------------------- step: merge, purity, memo
def test_step_merge_and_purity(train_tasks):
    hyps = load_support("mms_issue")[:400]
    parts = P.initial_particles(hyps)
    before = [(copy.deepcopy(p.state.agent_db), copy.deepcopy(p.state.user_db), copy.deepcopy(p.state.vpn_default), p.state.key,
               p.state.snapshot_id) for p in parts[:20]]
    P.clear_caches()
    parts = P.initial_particles(hyps)  # rebuilt after clear: still equal weights
    calls = (U("toggle_airplane_mode"), U("check_status_bar"))
    m0 = dict(P.STATS)
    merged, outs = P.step(parts, calls)
    unmerged, outs_u = P.step(parts, calls, merge=False)
    # purity: inputs untouched, outputs reproducible
    for p, (a, u, v, k, sid) in zip(parts[:20], before):
        assert (p.state.agent_db, p.state.user_db, p.state.vpn_default, p.state.key) == (a, u, v, k)
    assert [p.state.key for p in parts] == [P.build_state(h).key for h in hyps]
    # lineage
    for p, q in zip(parts, unmerged):
        assert q.state.parent_id == p.state.snapshot_id and q.state.step == p.state.step + 1 and q.hyp_key == p.hyp_key
    # weights are conserved and merging only reduces the count
    assert len(unmerged) == len(parts) == len(outs_u) and 1 <= len(merged) <= len(parts) == 400
    assert len(merged) == len(outs)
    assert abs(sum(p.weight for p in merged) - 1.0) < 1e-12 and abs(sum(p.weight for p in unmerged) - 1.0) < 1e-12
    assert len({(p.state.key, hypothesis_by_key(p.hyp_key).goal_id, o) for p, o in zip(merged, outs)}) == len(merged)
    # every output of the merged set also occurs unmerged (aligned semantics)
    assert set(outs) == set(outs_u)
    assert all(len(o) == 2 for o in outs)
    # explicit merge: the same particle twice with different weights collapses and sums
    h = hyps[0]
    dup = (Particle(h.key, P.build_state(h), 0.3), Particle(h.key, P.build_state(h), 0.2), Particle(hyps[1].key, P.build_state(hyps[1]), 0.5))
    mp, mo = P.step(dup, (U("check_status_bar"),))
    w = sorted(p.weight for p in mp)
    assert len(mp) in (1, 2) and abs(sum(w) - 1.0) < 1e-12
    assert max(w) >= 0.5 - 1e-12 and any(abs(x - 0.5) < 1e-12 or abs(x - 1.0) < 1e-12 for x in w)
    mp2, _ = P.step(dup[:2], (U("check_status_bar"),))
    assert len(mp2) == 1 and abs(mp2[0].weight - 0.5) < 1e-12 and len(mo) == len(mp)
    # memo: the second identical step is all hits and yields identical results
    m1 = dict(P.STATS)
    again, outs_again = P.step(parts, calls)
    m2 = dict(P.STATS)
    assert m2["apply_miss"] == m1["apply_miss"] and m2["apply_hit"] - m1["apply_hit"] == len(parts)
    assert outs_again == outs and [p.state.key for p in again] == [p.state.key for p in merged]
    assert m1["apply_miss"] > m0["apply_miss"]
    # apply_calls is pure and lineage-correct on a memo hit from a different parent
    s0 = parts[0].state
    ns1, o1 = P.apply_calls(s0, calls)
    other = replace(s0, snapshot_id="other-parent", step=5)
    h0 = P.STATS["apply_hit"]
    ns2, o2 = P.apply_calls(other, calls)
    assert P.STATS["apply_hit"] == h0 + 1  # same state key -> memo hit, yet lineage follows the caller's snapshot
    assert (ns1.key, o1) == (ns2.key, o2) and ns1.parent_id == s0.snapshot_id and ns1.step == s0.step + 1
    assert ns2.parent_id == "other-parent" and ns2.step == 6 and ns2.snapshot_id != ns1.snapshot_id
    # a multi-call batch equals its calls applied one by one
    one, oa = P.apply_calls(s0, (calls[0],))
    two, ob = P.apply_calls(one, (calls[1],))
    assert two.key == ns1.key and oa + ob == o1


# --------------------------------------------------------------------------- timings
def test_support_timings():
    logger.disable("tau2")  # tau2 logs every tool response at DEBUG; that I/O is not part of the simulator cost
    try:
        _support_timings()
    finally:
        logger.enable("tau2")


def _support_timings():
    P.clear_caches()
    hy = load_support("mms_issue")
    assert len(hy) == 1984
    t0 = time.perf_counter()
    parts = P.initial_particles(hy)
    t_build = time.perf_counter() - t0
    rss_build = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.perf_counter()
    ps2, outs = P.step(parts, (U("check_status_bar"),))
    t_step = time.perf_counter() - t0
    t0 = time.perf_counter()
    P.step(parts, (U("check_status_bar"),))
    t_step_memo = time.perf_counter() - t0
    t0 = time.perf_counter()
    gr = [P.grade(p.state, hypothesis_by_key(p.hyp_key), False) for p in parts]
    t_grade = time.perf_counter() - t0
    n = len(parts)
    rec = {"n_particles": n, "build_total_s": round(t_build, 3), "build_us_per_particle": round(t_build / n * 1e6, 1),
           "step_check_status_bar_total_s": round(t_step, 3), "step_us_per_particle": round(t_step / n * 1e6, 1),
           "step_memo_hit_total_s": round(t_step_memo, 4), "step_memo_hit_us_per_particle": round(t_step_memo / n * 1e6, 1),
           "merged_particles_after_step": len(ps2), "distinct_initial_state_keys": len({p.state.key for p in parts}),
           "grade_all_total_s": round(t_grade, 3), "grade_initial_mean": sum(gr) / n,
           "distinct_goal_ids": len({h.goal_id for h in hy}), "maxrss_after_build_mb": round(rss_build, 1),
           "maxrss_end_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
           "apply_memo_entries": len(P._APPLY_MEMO), "python": "3.12", "cpu_only": True, "tau2_logging_disabled": True}
    TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    TIMINGS.write_text(json.dumps(rec, indent=1) + "\n")
    print("\n[W1-S2] timings:", rec)
    assert gr == [0.0] * n  # a fresh fault is never already repaired (broken MMS state fails assert_can_send_mms)
