"""[telecom_boed] W2-S8 tests: k-step expectimax + reach-avoid value (eq. 4) vs Opus' expected values and a brute-force reference."""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from telecom_boed.contracts import DecisionKind
from telecom_boed.decide.dug import dug
from telecom_boed.decide.expectimax import KStepExpectimax, bench_scoring
from telecom_boed.decide.reach_avoid import ReachAvoid, ReachAvoidMargin
from telecom_boed.decide.utility import eu
from telecom_boed.testing.toy_pomdp import label, load_spec, load_toy, run_path

EXPECTED = yaml.safe_load((Path(__file__).parent / "fixtures" / "W2" / "expected_decisions.yaml").read_text())
CASES = list(EXPECTED)
TOL = 1e-9


def _nt(cands):
    return [c for c in cands if c.kind == DecisionKind.REQUEST_DEVICE]


@pytest.mark.parametrize("case", CASES)
def test_toy_kstep_matches_expected(case):
    pred, grader, b0, cands, cfg = load_toy(case)
    exp = EXPECTED[case]["initial"]
    sc = KStepExpectimax(pred, grader, cfg)
    q = sc.q_values(b0, cands)
    for c, v in zip(cands, q):
        assert v == pytest.approx(exp["kstep_q"][label(c)], abs=TOL), (case, label(c))
    assert label(cands[sc.decide(b0, cands)]) == exp["kstep"]


@pytest.mark.parametrize("case", CASES)
def test_toy_reach_matches_expected(case):
    pred, grader, b0, cands, cfg = load_toy(case)
    exp = EXPECTED[case]["initial"]
    sc = ReachAvoid(pred, grader, cfg)
    q = sc.q_values(b0, cands)
    for c, v in zip(cands, q):
        assert v == pytest.approx(exp["reach_q"][label(c)], abs=TOL), (case, label(c))
    assert sc.v_root(b0, cands) == pytest.approx(exp["V_reach_root"], abs=TOL)
    assert label(cands[sc.decide(b0, cands)]) == exp["reach"]
    assert sc.stats["hits"] + sc.stats["misses"] > 0


def test_kstep_budget_aware_variant():
    """kstep honours B by default (expected v2); budget_aware=False is the unbudgeted ablation."""
    pred, grader, b0, cands, cfg = load_toy("budget")
    q = dict(zip(map(label, cands), KStepExpectimax(pred, grader, cfg, budget_aware=True).q_values(b0, cands)))
    assert q["R_air"] == pytest.approx(0.735, abs=TOL) and q["K_status"] == pytest.approx(-0.4, abs=TOL)  # nothing affordable below the root
    q_unb = dict(zip(map(label, cands), KStepExpectimax(pred, grader, cfg, budget_aware=False).q_values(b0, cands)))
    assert q_unb["R_air"] == pytest.approx(0.8357625, abs=TOL) and q_unb["K_status"] == pytest.approx(0.7285, abs=TOL)


def test_toy_paths():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    path = load_spec()["repair_verify"]["paths"]["user_silent_then_fixed"]
    exp = EXPECTED["repair_verify"]["user_silent_then_fixed"]
    ks = run_path(pred, b0, cands, KStepExpectimax(pred, grader, cfg).decide, path)
    ra = run_path(pred, b0, cands, ReachAvoid(pred, grader, cfg).decide, path)
    assert ks == exp["kstep"] == ["R_air", "R_air", "claim"]
    assert ra == exp["reach"] == ["R_air", "K_status", "claim"]


# ----------------------------------------------------------------------------- independent brute-force reference
def _bf_u(grader, cfg, b):
    c_fa, c_esc = cfg["utility"]["c_FA"], cfg["utility"]["c_esc"]
    cl = sum(p.weight * (grader.grade(p.state, p.hyp_key, b.transferred) - (c_fa if grader.grade(p.state, p.hyp_key, b.transferred) == 0 else 0))
             for p in b.particles)
    tr = sum(p.weight * (grader.grade(p.state, p.hyp_key, True) if grader.requires_transfer(p.hyp_key) else -c_esc) for p in b.particles)
    return cl, tr


def _bf_kstep(pred, grader, cfg, b, cands, d):
    """V_d without memo/pruning over all nonterminal candidates; returns (V, {name: Q}) with Q for depth d at b."""
    lam, B = cfg["cost"]["lambda"], cfg["cost"]["budget_B"]
    cl, tr = _bf_u(grader, cfg, b)
    best, qs = max(cl, tr), {}
    if d > 0:
        for a in _nt(cands):
            if b.cost + pred.action_cost(a) > B:  # budget-aware (KStepExpectimax default); unaffordable → not a choice
                continue
            q = sum(p * _bf_kstep(pred, grader, cfg, b2, cands, d - 1)[0] for p, b2 in pred.predict(b, a).values()) - lam * pred.action_cost(a)
            qs[label(a)] = q
            best = max(best, q)
    return best, qs


def _bf_reach(pred, grader, cfg, b, cands, d):
    lam, B, gam = cfg["cost"]["lambda"], cfg["cost"]["budget_B"], cfg["reach_avoid"]["gamma"]
    if b.cost > B or b.kappa:
        return 0.0, {}
    qc = sum(p.weight for p in b.particles if b.v and grader.grade(p.state, p.hyp_key, b.transferred) == 1 and not grader.requires_transfer(p.hyp_key))
    qt = sum(p.weight for p in b.particles if grader.requires_transfer(p.hyp_key) and grader.grade(p.state, p.hyp_key, True) == 1)
    best, qs = max(qc, qt), {}
    if d > 0:
        for a in _nt(cands):
            c = pred.action_cost(a)
            if b.cost + c > B:
                q = -lam * c
            else:
                q = gam * sum(p * _bf_reach(pred, grader, cfg, b2, cands, d - 1)[0] for p, b2 in pred.predict(b, a).values()) - lam * c
            qs[label(a)] = q
            best = max(best, q)
    return best, qs


def _priors(rng, case, n):
    ids = [h["id"] for h in load_spec()[case]["hypotheses"]]
    return [dict(zip(ids, rng.dirichlet(np.ones(len(ids)) * 0.7).tolist())) for _ in range(n)]


def test_bruteforce_reference_depth3():
    rng = np.random.default_rng(2026)
    work = [(c, None) for c in CASES] + [(c, pr) for c in CASES for pr in _priors(rng, c, 4)]
    assert len(work) == 25
    for case, prior in work:
        pred, grader, b0, cands, cfg = load_toy(case)
        b = b0 if prior is None else pred.make_belief(prior)
        for d in (1, 2, 3):
            cfg_d = copy.deepcopy(cfg)
            cfg_d["lookahead"]["depth_k"] = d
            ks, ra = KStepExpectimax(pred, grader, cfg_d), ReachAvoid(pred, grader, cfg_d)
            vk, qk = _bf_kstep(pred, grader, cfg_d, b, cands, d)
            vr, qr = _bf_reach(pred, grader, cfg_d, b, cands, d)
            gk, gr = dict(zip(map(label, cands), ks.q_values(b, cands))), dict(zip(map(label, cands), ra.q_values(b, cands)))
            for n, v in qk.items():
                assert gk[n] == pytest.approx(v, abs=TOL), (case, d, n, "kstep")
            for n, v in qr.items():
                assert gr[n] == pytest.approx(v, abs=TOL), (case, d, n, "reach")
            assert max(gk.values()) == pytest.approx(vk, abs=TOL) and ks.value(b, cands) == pytest.approx(vk, abs=TOL)
            assert ra.v_root(b, cands) == pytest.approx(vr, abs=TOL)
            # repeated call on a warm memo gives the same numbers
            assert ra.q_values(b, cands) == pytest.approx(list(gr.values()), abs=TOL)


def test_memo_stats_and_clear():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    sc = ReachAvoid(pred, grader, cfg)
    q1 = sc.q_values(b0, cands)
    s1 = sc.stats
    assert s1["hits"] > 0 and s1["misses"] == s1["size"]
    sc.q_values(b0, cands)
    assert sc.stats["misses"] == s1["misses"]  # second call is all hits
    sc.clear_memo()
    assert sc.q_values(b0, cands) == q1


def test_alpha0_equivalence():
    pred, grader, b0, cands, cfg = load_toy("alpha0_equivalence")
    exp = EXPECTED["alpha0_equivalence"]["initial"]
    sc = ReachAvoid(pred, grader, cfg)
    assert label(cands[sc.decide(b0, cands)]) == exp["myopic"] == exp["reach"]
    assert cfg["lookahead"]["depth_k"] == 1 and cfg["reach_avoid"]["gamma"] == 1.0
    rng = np.random.default_rng(7)
    for prior in [None] + _priors(rng, "alpha0_equivalence", 10):
        b = b0 if prior is None else pred.make_belief(prior)
        q = sc.q_values(b, cands)
        diffs = [qi - dug(b, c, pred, grader, cfg) for c, qi in zip(cands, q) if c.kind == DecisionKind.REQUEST_DEVICE]
        assert max(diffs) - min(diffs) < TOL
        assert diffs[0] == pytest.approx(eu(b, grader, cfg), abs=TOL)  # the constant is eu(beta0)
        nt = [i for i, c in enumerate(cands) if c.kind == DecisionKind.REQUEST_DEVICE]
        assert max(nt, key=lambda i: (q[i], -i)) == max(nt, key=lambda i: (dug(b, cands[i], pred, grader, cfg), -i))


def test_budget_failure():
    pred, grader, b0, cands, cfg = load_toy("budget")
    sc = ReachAvoid(pred, grader, cfg)
    q = dict(zip(map(label, cands), sc.q_values(b0, cands)))
    assert q["K_status"] == pytest.approx(-0.1, abs=TOL) and q["R_air"] == pytest.approx(0.35, abs=TOL)
    over = pred.make_belief(cost=11.0)  # cost > B
    assert sc._v(over, 3) == 0.0 and sc.v_root(over, cands) == 0.0
    fail = dataclasses.replace(b0, kappa=True)
    assert sc._v(fail, 3) == 0.0 and sc.v_root(fail, cands) == 0.0
    # exactly at B is still alive, one more action is not affordable
    at_b = pred.make_belief(cost=10.0)
    assert sc.q_values(at_b, cands)[0] == pytest.approx(-0.1, abs=TOL)
    assert sc._v(at_b, 3) == 0.0  # no verified mass and nothing affordable -> 0


def test_decision_rule_gates():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    verified = dataclasses.replace(b0, particles=pred.make_belief().particles, v=True)
    # P(repaired) = 0 at the prior (air is on): never claim, even with v=1
    assert cands[ReachAvoid(pred, grader, cfg).decide(verified, cands)].kind != DecisionKind.CLAIM_RESOLVED
    out = pred.predict(b0, cands[0])[("repaired", True)][1]
    assert out.v and label(cands[ReachAvoid(pred, grader, cfg).decide(out, cands)]) == "claim"
    # escalation: eta above every Q -> transfer
    cfg2 = copy.deepcopy(cfg)
    cfg2["reach_avoid"]["eta"] = 2.0
    assert cands[ReachAvoid(pred, grader, cfg2).decide(b0, cands)].kind == DecisionKind.TRANSFER


def test_inner_action_provider_pluggable():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    full = ReachAvoid(pred, grader, cfg).q_values(b0, cands)
    only_k = ReachAvoid(pred, grader, cfg, inner_actions=lambda b, root: [c for c in root if label(c) == "K_status"])
    q = only_k.q_values(b0, cands)
    assert q[0] < full[0] - 1e-6  # R_air cannot be repeated below the root -> strictly worse
    # default provider with inner_m < #actions: heuristic/DUG union, never larger than inner_m*2, values <= full lookahead
    cfg1 = copy.deepcopy(cfg)
    cfg1["lookahead"]["inner_m"] = 1
    sc = ReachAvoid(pred, grader, cfg1)
    sc._set_root(cands)
    assert 1 <= len(sc._inner(b0)) <= 2
    assert all(a <= b + 1e-12 for a, b in zip(sc.q_values(b0, cands), full))


def test_margin_variant_properties():
    rng = np.random.default_rng(3)
    for case in CASES:
        pred, grader, b0, cands, cfg = load_toy(case)
        ids = [h["id"] for h in pred.spec["hypotheses"]]
        beliefs = [b0] + [pred.make_belief(dict(zip(ids, rng.dirichlet(np.ones(len(ids))).tolist()))) for _ in range(3)]
        for b in beliefs:
            prev = None
            for B in (0.5, 1.5, 3.0, 10.0):  # monotone non-decreasing in the budget
                cfgB = copy.deepcopy(cfg)
                cfgB["cost"]["budget_B"] = B
                vals = []
                for gamma in (1.0, 0.9):
                    cfgB["reach_avoid"]["gamma"] = gamma
                    sc = ReachAvoidMargin(pred, grader, cfgB)
                    sc._set_root(cands)
                    v = sc.value(dataclasses.replace(b, cost=0.0))
                    assert -1.0 <= v <= 0.5, (case, B, gamma, v)
                    vals.append(v)
                if prev is not None:
                    assert all(v >= p - 1e-12 for v, p in zip(vals, prev)), (case, B)
                prev = vals
            sc = ReachAvoidMargin(pred, grader, cfg)
            sc._set_root(cands)
            failed = dataclasses.replace(b, kappa=True)
            assert sc.value(failed) <= -0.5 + 1e-12  # sign convention: failure <= g = -1/2 (negative, not 0)
            assert sc.value(dataclasses.replace(b, cost=cfg["cost"]["budget_B"] + 1.0)) <= -1.0 + 1e-12
            q = sc.q_values(b, cands)
            assert len(q) == len(cands) and all(np.isfinite(q))
            assert sc.decide(b, cands) in range(len(cands))
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    sc = ReachAvoidMargin(pred, grader, cfg)
    sc._set_root(cands)
    assert label(cands[sc.decide(b0, cands)]) in ("R_air", "K_status")


def test_bench_scoring_smoke():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    r = bench_scoring(pred, b0, cands, ReachAvoid(pred, grader, cfg), reps=5)
    assert 0 <= r["p50_ms"] <= r["p95_ms"] and r["reps"] == 5
