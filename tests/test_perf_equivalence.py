"""[telecom_boed] W2-P1 tests: the scorer speed-ups must not change any value or decision.

`tests/fixtures/W2-P1/golden.json` was captured from the UNOPTIMISED scorers (provenance: md5s of the owned files at capture time are
stored inside it). Regenerate only deliberately: `W2P1_CAPTURE=1 pytest tests/test_perf_equivalence.py::test_golden_equivalence`.
Covers: utilities / entropy on beliefs, dug / eig / kstep / reach / reach_margin score vectors + decisions on the 5 toys and on the
W2-S9 pinned real-support beliefs (mobile_data depth 1+2, mms depth 1), budget None. Plus the memo-reuse tests of the lookahead.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from telecom_boed import paths
from telecom_boed.belief.filter import ParticleFilter
from telecom_boed.belief.stepper import ParallelStepper
from telecom_boed.contracts import CandidateAction, DecisionKind
from telecom_boed.decide.controller import Controller
from telecom_boed.decide.dug import MyopicDUG, dug, dug_gross
from telecom_boed.decide.eig import EIGScorer, eig, hyp_entropy
from telecom_boed.decide.expectimax import KStepExpectimax
from telecom_boed.decide.reach_avoid import ReachAvoid, ReachAvoidMargin
from telecom_boed.decide.utility import eu, p_repaired, q_claim_reach, q_transfer_reach, u_claim, u_transfer
from telecom_boed.testing.toy_pomdp import label, load_toy, load_spec
from test_controller import hand_candidates, first_obs_belief, train_hyps  # same belief construction as W2-S9

FIX = Path(__file__).parent / "fixtures" / "W2-P1"
GOLDEN = FIX / "golden.json"
SRC = paths.PACKAGE_ROOT / "src" / "telecom_boed" / "decide"
OWNED = ("expectimax.py", "reach_avoid.py", "utility.py", "dug.py", "eig.py")
TOL = 1e-9
TERM = [CandidateAction(DecisionKind.TRANSFER, (), 4)]


def _utils(b, pred, grader, cfg, cands) -> dict:
    out = {"u_claim": u_claim(b, grader, cfg), "u_transfer": u_transfer(b, grader, cfg), "eu": eu(b, grader, cfg),
           "p_repaired": p_repaired(b, grader), "q_claim_reach": q_claim_reach(b, grader),
           "q_transfer_reach": q_transfer_reach(b, grader), "H": hyp_entropy(b)}
    nt = [c for c in cands if c.kind not in (DecisionKind.CLAIM_RESOLVED, DecisionKind.TRANSFER)]
    out["dug"] = [dug(b, c, pred, grader, cfg) for c in nt]
    out["dug_gross"] = [dug_gross(b, c, pred, grader, cfg) for c in nt]
    out["eig"] = [eig(b, c, pred) for c in nt]
    return out


def _scorers(pred, grader, cfg, cands, b, depths, mk=None) -> dict:
    """{name: {"scores": [...], "pick": i}} for dug / eig (myopic) and for each lookahead scorer at each depth."""
    out: dict = {}
    for nm, sc in (("dug", MyopicDUG(pred, grader, cfg)), ("eig", EIGScorer(pred, grader, cfg))):
        out[nm] = {"scores": sc.score(b, cands), "pick": sc.decide(b, cands)}
    for nm, cls, kw in (("kstep", KStepExpectimax, {}), ("kstep_nobudget", KStepExpectimax, {"budget_aware": False}),
                        ("reach", ReachAvoid, {}), ("reach_margin", ReachAvoidMargin, {})):
        for d in depths:
            if nm == "kstep_nobudget" and d > 2:
                continue
            sc = (mk or (lambda c, **k: c(pred, grader, cfg, **k)))(cls, **kw)
            sc.depth = d
            out[f"{nm}@{d}"] = {"scores": sc.q_values(b, cands), "pick": sc.decide(b, cands)}
    return out


def _toy_golden() -> dict:
    out: dict = {}
    for case in load_spec():
        pred, grader, b0, cands, cfg = load_toy(case)
        beliefs = {"b0": b0}
        for c in cands:
            if c.kind == DecisionKind.REQUEST_DEVICE:
                for k, (_p, b2) in pred.predict(b0, c).items():
                    beliefs[f"{label(c)}/{k}"] = b2
        out[case] = {"utils": {n: _utils(b, pred, grader, cfg, cands) for n, b in beliefs.items()},
                     "scorers": _scorers(pred, grader, cfg, cands, b0, (1, 2, 3))}
        mid = [b for n, b in beliefs.items() if n != "b0"][-1]  # one non-root belief through the scorers too
        out[case]["scorers_mid"] = _scorers(pred, grader, cfg, cands, mid, (1, 2))
    return out


REAL_CASES = (("mobile_data_issue", "dug", (1,)), ("mobile_data_issue", "manual", (1, 2)), ("mms_issue", "dug", (1,)),
              ("mms_issue", "manual", (1,)))


def _real_golden(cfg_real, train_ids, workers: int) -> dict:
    out: dict = {}
    for intent in ("mobile_data_issue", "mms_issue"):
        hyp = train_hyps(intent, train_ids, 5)[0]
        with ParallelStepper(workers) as st:
            flt = ParticleFilter("deployable", cfg_real, stepper=st)
            b1 = first_obs_belief(flt, intent, hyp)
            base = hand_candidates(intent)
            res: dict = {"n_particles": len(b1.particles), "utils": {}, "scorers": {}}
            for tag, cands in (("claim", base), ("claim+transfer", base + TERM)):
                res["utils"][tag] = _utils(b1, flt, flt.grader, cfg_real, cands)
            for it, rank, depths in REAL_CASES:
                if it != intent:
                    continue
                cands = base + TERM
                ctl = Controller("reach", flt, flt.grader, cfg_real, override="none", time_budget_ms=None, inner_rank=rank)

                def mk(cls, **kw):
                    return cls(ctl.proxy, ctl.gproxy, cfg_real, inner_actions=ctl.inner_actions, **kw)
                for d in depths:
                    for nm in ("kstep", "reach") + (("reach_margin",) if rank == "manual" and d == 1 else ()):
                        cls = {"kstep": KStepExpectimax, "reach": ReachAvoid, "reach_margin": ReachAvoidMargin}[nm]
                        sc = mk(cls)
                        sc.depth = d
                        res["scorers"][f"{rank}/{nm}@{d}"] = {"scores": sc.q_values(b1, cands), "pick": sc.decide(b1, cands)}
                if rank == "dug":
                    res["scorers"].update({k: v for k, v in _scorers(flt, flt.grader, cfg_real, cands, b1, ()).items()})
            out[intent] = res
    return out


def _md5s() -> dict:
    return {f: hashlib.md5((SRC / f).read_bytes()).hexdigest() for f in OWNED}


def _flat(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flat(v, f"{path}/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _flat(v, f"{path}[{i}]")
    else:
        yield path, x


def _assert_equal(got: dict, want: dict, what: str) -> None:
    g, w = dict(_flat(got)), dict(_flat(want))
    assert g.keys() == w.keys(), (what, sorted(set(g) ^ set(w))[:5])
    for k, v in w.items():
        if k.endswith("/pick"):
            assert g[k] == v, (what, k, g[k], v)  # same decision
        elif os.environ.get("W2P1_EXACT") == "1":  # stricter, optional: bit-identical floats
            assert g[k] == v, (what, k, g[k], v)
        else:
            assert g[k] == pytest.approx(v, abs=TOL, rel=0), (what, k, g[k], v)


@pytest.fixture(scope="module")
def cfg_real():
    return yaml.safe_load((paths.CONFIG_DIR / "boed_default.yaml").read_text())


def test_golden_equivalence(cfg_real, train_ids):
    workers = int(os.environ.get("W2P1_WORKERS", "8"))
    got = {"toys": _toy_golden(), "real": _real_golden(cfg_real, train_ids, workers)}
    if os.environ.get("W2P1_CAPTURE") == "1":
        assert not GOLDEN.exists(), "golden.json exists: delete it deliberately to re-capture"
        FIX.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({"provenance": {"note": "captured from the unoptimised code (no git; md5 of owned files)",
                                                     "md5": _md5s(), "inputs": "tests/fixtures/W2-S9/inputs.json first train hyp per intent"},
                                      **got}, indent=0))
        return
    want = json.loads(GOLDEN.read_text())
    _assert_equal(got["toys"], want["toys"], "toys")
    _assert_equal(got["real"], want["real"], "real")


# --------------------------------------------------------------------------- memo reuse across decisions / anytime / LRU
def _cold(cls, pred, grader, cfg, cands, b, depth, **kw):
    sc = cls(pred, grader, cfg, **kw)
    sc.depth = depth
    return sc.q_values(b, cands)


@pytest.mark.parametrize("cls", [KStepExpectimax, ReachAvoid, ReachAvoidMargin])
@pytest.mark.parametrize("case", ["repair_verify", "budget"])
def test_memo_reuse_receding_horizon(cls, case):
    """Decision 1 at depth 3 on b; decision 2 on b' = predict(b, a)[o] (same candidates) at depth 2 is ALL memo hits and equals a cold run."""
    pred, grader, b0, cands, cfg = load_toy(case)
    sc = cls(pred, grader, cfg)
    sc.depth = 3
    q1 = sc.q_values(b0, cands)
    assert q1 == _cold(cls, pred, grader, cfg, cands, b0, 3)
    a = next(c for c in cands if c.kind == DecisionKind.REQUEST_DEVICE)
    n_hits = 0
    for _o, (_p, b1) in pred.predict(b0, a).items():
        sc.depth = 2
        sc.reset_stats()
        q2 = sc.q_values(b1, cands)
        assert q2 == _cold(cls, pred, grader, cfg, cands, b1, 2)  # exactly what a cold computation returns
        assert sc.stats["computed"] == 0, (case, _o, sc.stats)  # every V_{<=1} node was computed by decision 1
        n_hits += sc.hits
        sc.depth = 3
        assert sc.q_values(b1, cands) == _cold(cls, pred, grader, cfg, cands, b1, 3)  # deepening past the cached levels is still exact
    assert n_hits > 0 or case == "budget"
    # anytime: level 0 (terminals only) then 1..3, exact, depth recorded
    sc.reset_stats()
    vals, d = sc.anytime(b0, cands)
    assert d == 3 == sc.last_decision["depth"] and vals == q1 and len(sc.last_decision["levels_ms"]) == 4
    vals, d = sc.anytime(b0, cands, budget_s=0.0)
    assert d == 1 and sc.last_decision["stopped"] == "deadline" and vals == _cold(cls, pred, grader, cfg, cands, b0, 1)
    sc.reset()
    assert sc.stats["size"] == 0 and sc.q_values(b0, cands) == q1


def test_memo_not_shared_across_different_root_sets():
    """The default inner set depends on the root candidates: a different root set must not read the old values."""
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    sc = ReachAvoid(pred, grader, cfg)
    sc.q_values(b0, cands)
    sub = [c for c in cands if c.kind != DecisionKind.REQUEST_DEVICE or label(c) != "R_air"]
    assert sc.q_values(b0, sub) == _cold(ReachAvoid, pred, grader, cfg, sub, b0, cfg["lookahead"]["depth_k"])
    # same actions, other text / candidate_idx (what changes every turn with sampled messages): same inner set -> reuse is allowed
    from dataclasses import replace
    re = [replace(c, text="x", candidate_idx=9) if c.kind == DecisionKind.REQUEST_DEVICE else c for c in cands]
    sc.reset_stats()
    assert sc.q_values(b0, re) == _cold(ReachAvoid, pred, grader, cfg, re, b0, cfg["lookahead"]["depth_k"]) and sc.stats["computed"] == 0


def test_memo_lru_cap(monkeypatch):
    import telecom_boed.decide.expectimax as ex
    pred, grader, b0, cands, cfg = load_toy("repair_verify")
    want = _cold(ReachAvoid, pred, grader, cfg, cands, b0, 3)
    monkeypatch.setattr(ex, "MEMO_MAX", 5)
    sc = ReachAvoid(pred, grader, cfg)
    assert sc.q_values(b0, cands) == want and sc.stats["size"] <= 5


def test_real_successor_belief_reuse(cfg_real, train_ids):
    """Real support (mobile_data): decision on b, then the successor predict(b, a)[o] one level shallower is all memo hits and exact."""
    intent = "mobile_data_issue"
    with ParallelStepper(int(os.environ.get("W2P1_WORKERS", "8"))) as st:
        flt = ParticleFilter("deployable", cfg_real, stepper=st)
        b = first_obs_belief(flt, intent, train_hyps(intent, train_ids, 5)[0])
        cands = hand_candidates(intent)
        ctl = Controller("reach", flt, flt.grader, cfg_real, override="none", time_budget_ms=None, inner_rank="manual")
        sc = ReachAvoid(ctl.proxy, ctl.gproxy, cfg_real, inner_actions=ctl.inner_actions)
        sc.depth = 2
        sc.q_values(b, cands)
        a = next(c for c in cands if c.kind == DecisionKind.REQUEST_DEVICE)
        outs = sorted(flt.predict(b, a).values(), key=lambda t: -t[0])[:2]
        for _p, b1 in outs:
            sc.depth = 1
            sc.reset_stats()
            q = sc.q_values(b1, cands)
            assert sc.stats["computed"] == 0 and sc.stats["reused"] > 0
            cold = ReachAvoid(ctl.proxy, ctl.gproxy, cfg_real, inner_actions=ctl.inner_actions)
            cold.depth = 1
            assert q == cold.q_values(b1, cands)
