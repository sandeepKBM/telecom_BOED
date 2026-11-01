"""[telecom_boed] W2-S6 tests: myopic scorers vs Opus' independent expected values (tests/fixtures/W2/expected_decisions.yaml)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from telecom_boed.contracts import Belief, CandidateAction, DecisionKind, DeviceAction, Particle, Snapshot
from telecom_boed.decide.dug import MyopicDUG, dug, dug_gross
from telecom_boed.decide.eig import EIGScorer, eig
from telecom_boed.decide.heuristics import HeuristicScorer
from telecom_boed.decide.risk import risk
from telecom_boed.decide.utility import cost_of
from telecom_boed.testing.toy_pomdp import label, load_toy, run_path

EXPECTED = yaml.safe_load((Path(__file__).parent / "fixtures" / "W2" / "expected_decisions.yaml").read_text())
CASES = list(EXPECTED)
TOL = 1e-9


def _nonterminals(cands):
    return [c for c in cands if c.kind == DecisionKind.REQUEST_DEVICE]


@pytest.mark.parametrize("case", CASES)
def test_toy_myopic_dug_matches_expected(case):
    pred, grader, b0, cands, cfg = load_toy(case)
    exp = EXPECTED[case]["initial"]
    for c in _nonterminals(cands):
        assert dug(b0, c, pred, grader, cfg) == pytest.approx(exp["myopic_dug"][label(c)], abs=TOL), (case, label(c))
        assert dug_gross(b0, c, pred, grader, cfg) == pytest.approx(exp["dug_gross"][label(c)], abs=TOL), (case, label(c))
    scorer = MyopicDUG(pred, grader, cfg)
    assert label(cands[scorer.decide(b0, cands)]) == exp["myopic"]


@pytest.mark.parametrize("case", CASES)
def test_toy_eig_matches_expected(case):
    pred, grader, b0, cands, cfg = load_toy(case)
    for c in _nonterminals(cands):
        assert eig(b0, c, pred) == pytest.approx(EXPECTED[case]["initial"]["eig"][label(c)], abs=TOL), (case, label(c))
    sc = EIGScorer(pred, grader, cfg)
    assert len(sc.score(b0, cands)) == len(cands)
    # stop rule: only irrelevant_uncertainty / alpha0 have EIG above the stop threshold
    informative = max(eig(b0, c, pred) for c in _nonterminals(cands)) >= sc.stop
    assert (cands[sc.decide(b0, cands)].kind == DecisionKind.REQUEST_DEVICE) == informative


def test_toy_myopic_path_repair_verify():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    from telecom_boed.testing.toy_pomdp import load_spec
    path = load_spec()["repair_verify"]["paths"]["user_silent_then_fixed"]
    seq = run_path(pred, b0, cands, MyopicDUG(pred, grader, cfg).decide, path)
    assert seq == EXPECTED["repair_verify"]["user_silent_then_fixed"]["myopic"] == ["R_air", "R_air", "claim"]


@pytest.mark.parametrize("case", CASES)
def test_voi_lower_bound_property(case):
    pred, grader, _, cands, cfg = load_toy(case)
    rng = np.random.default_rng(12345)
    ids = [h["id"] for h in pred.spec["hypotheses"]]
    lam = cfg["cost"]["lambda"]
    obs_only = [c for c in _nonterminals(cands) if not pred.is_mutating(c)]
    assert obs_only
    for _ in range(200):
        w = rng.dirichlet(np.ones(len(ids)))
        b = pred.make_belief(dict(zip(ids, w.tolist())))
        for c in obs_only:
            r, _k = risk(c, b)
            assert dug(b, c, pred, grader, cfg) >= -lam * cost_of(c, pred, cfg) - cfg["cost"]["rho"] * r - 1e-12


def _snap(agent_db):
    return Snapshot(agent_db=agent_db, user_db={}, vpn_default=None, key=("x", "", ""))


def _belief(agent_db):
    return Belief((Particle(("t", frozenset({"h"})), _snap(agent_db), 1.0),))


def _bc(tool, **args):
    return CandidateAction(DecisionKind.AGENT_TOOL, (DeviceAction("assistant", tool, tuple(sorted(args.items()))),))


def test_risk_table():
    empty = _belief({})
    assert risk(CandidateAction(DecisionKind.CLAIM_RESOLVED), empty) == (0.0, False)
    assert risk(CandidateAction(DecisionKind.REQUEST_DEVICE, (DeviceAction("user", "reset_apn_settings"),)), empty) == (0.0, False)
    assert risk(_bc("mystery_tool", customer_id="C1001"), empty) == (0.0, False)
    assert risk(_bc("suspend_line", customer_id="C1001", line_id="L1002", reason="x"), empty) == (1.0, False)
    assert risk(_bc("refuel_data", customer_id="C9999", line_id="L1002", gb_amount=1.0), empty) == (1.0, True)
    expired = _belief({"lines": [{"line_id": "L1002", "status": "Suspended", "contract_end_date": "2024-12-01"}]})
    overdue = _belief({"lines": [{"line_id": "L1002", "status": "Suspended", "contract_end_date": "2026-01-01"}]})
    res = _bc("resume_line", customer_id="C1001", line_id="L1002")
    assert risk(res, expired) == (1.0, True)  # tools.py:298-330 resumes anyway; policy forbids it (main_policy.md:126)
    assert risk(res, overdue) == (0.0, False)


def test_heuristic_follows_order():
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    h = HeuristicScorer(grader, cfg, order=[label(c) for c in _nonterminals(cands)])
    seq = []
    for _ in range(4):
        i = h.decide(b0, cands)
        seq.append(label(cands[i]))
        h.mark_done(cands[i])
    assert seq == ["R_air", "K_status", "transfer", "transfer"]
