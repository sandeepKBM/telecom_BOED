"""[telecom_boed] W2-P2 tests: the ParticleFilter speed-ups must not change any belief, probability or decision.

`tests/fixtures/W2-P2/golden.json` was captured from the UNOPTIMISED filter (provenance: md5s of the owned files at capture time are stored
inside it; the originals are kept in tests/fixtures/W2-P2/orig/). Regenerate only deliberately:
`W2P2_CAPTURE=1 pytest tests/test_filter_perf_equivalence.py::test_filter_golden_equivalence` (refuses if golden.json exists).
Inputs: the W2-S9 / W2-P1 pinned real-support beliefs (first train hypothesis per intent, scripted status-bar report) on the mobile_data
(254) and mms (1,984) supports, 2 modes x 2 personas, ~14 actions (checks, toggles with ensure, reseat SIM, reset APN, network mode,
speed test, MMS, 2-action request, agent backend tool, ASK, CLAIM, TRANSFER). For every action: `predict` (outcome order, prob, posterior
digest) and `update` on three of its outcomes. Posterior digest = structure (particle order, merge keys, hyp_keys, exec_history: exact)
+ v/kappa/cost/transferred/step/degenerate/intent_probs (exact) + weights sorted by merge key (n, sum, sum of squares, a 48-point
subsample: 1e-12). `W2P2_EXACT=1` additionally requires the md5 of the weight bytes (bit-identical weights).
Order: the dict order of `predict` and the particle order of every posterior are compared EXACTLY (scorers sum over them).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from telecom_boed import paths
from telecom_boed.belief.filter import ParticleFilter
from telecom_boed.belief.stepper import ParallelStepper
from telecom_boed.contracts import Belief, CandidateAction, DecisionKind, DeviceAction, Observation, _frozen_items
from telecom_boed.env.hypotheses import hypothesis_by_key
from test_controller import U, cand, first_obs_belief, train_hyps  # same belief construction as W2-S9 / W2-P1

FIX = Path(__file__).parent / "fixtures" / "W2-P2"
GOLDEN = FIX / "golden.json"
SRC = paths.PACKAGE_ROOT / "src" / "telecom_boed" / "belief"
OWNED = ("filter.py", "stepper.py", "user_model.py")
TOL = 1e-12
REQ = DecisionKind.REQUEST_DEVICE


def _req(*acts: DeviceAction, idx=None) -> CandidateAction:
    return CandidateAction(REQ, tuple(acts), idx)


ACTIONS = {
    "check_status_bar": _req(U("check_status_bar")),
    "toggle_airplane_ensure_off": _req(U("toggle_airplane_mode", ("airplane_mode", False))),
    "toggle_data_ensure_on": _req(U("toggle_data", ("data", True))),
    "toggle_roaming_ensure_on": _req(U("toggle_roaming", ("roaming", True))),
    "reseat_sim": _req(U("reseat_sim_card")),
    "reset_apn": _req(U("reset_apn_settings")),
    "set_network_mode": _req(U("set_network_mode_preference", mode="4g_5g_preferred")),
    "check_network_status": _req(U("check_network_status")),
    "run_speed_test": _req(U("run_speed_test")),
    "can_send_mms": _req(U("can_send_mms")),
    "two_actions": _req(U("toggle_airplane_mode", ("airplane_mode", False)), U("toggle_data", ("data", True))),
    "agent_get_details": CandidateAction(DecisionKind.AGENT_TOOL, (DeviceAction("assistant", "get_details_by_id", _frozen_items({"id": "L1002"})),)),
    "ask": CandidateAction(DecisionKind.ASK, (), 7),
    "claim": CandidateAction(DecisionKind.CLAIM_RESOLVED, (), 3),
    "transfer": CandidateAction(DecisionKind.TRANSFER, (), 4),
}


# --------------------------------------------------------------------------- digests
def _okey(k: tuple) -> str:
    return repr((k[0], k[1], k[2], tuple(sorted(k[3])))) if len(k) == 4 else repr(k)


def _digest(b: Belief) -> dict:
    gid = {}
    rows = []
    for p in b.particles:
        g = gid.get(p.hyp_key)
        if g is None:
            g = gid[p.hyp_key] = hypothesis_by_key(p.hyp_key).goal_id
        rows.append(((p.state.key, g), (p.hyp_key[0], tuple(sorted(p.hyp_key[1]))), p.exec_history, p.state.step))
    order = sorted(range(len(rows)), key=lambda i: rows[i][0])
    w = np.array([b.particles[i].weight for i in order], dtype=np.float64)
    n = len(w)
    stride = max(1, n // 48)
    return {
        "n": n, "struct_md5": hashlib.md5(repr(rows).encode()).hexdigest(),
        "v": b.v, "kappa": b.kappa, "cost": b.cost, "transferred": b.transferred, "step": b.step, "degenerate": b.degenerate,
        "intent_probs": repr(b.intent_probs), "wsum": float(w.sum()) if n else 0.0, "wsq": float((w * w).sum()) if n else 0.0,
        "wsub": w[::stride].tolist(), "wmd5": hashlib.md5(w.tobytes()).hexdigest(),
    }


def _case(flt: ParticleFilter, b: Belief, c: CandidateAction) -> dict:
    out = flt.predict(b, c)
    rec = {"keys": [_okey(k) for k in out], "outcomes": [{"p": p, "b": _digest(b2)} for p, b2 in out.values()]}
    items = list(out.items())
    if c.kind in (REQ, DecisionKind.AGENT_TOOL):
        pick = sorted({0, max(range(len(items)), key=lambda i: items[i][1][0]), len(items) - 1}) if items else []
        rec["updates"] = {}
        for i in pick:
            k, (_p, _b2) = items[i]
            rec["updates"][str(i)] = _digest(flt.update(b, c, Observation(k[1], k[0], k[2], k[3])))
    return rec


def _golden_run(cfg_real, train_ids, workers: int) -> dict:
    out: dict = {}
    for intent in ("mobile_data_issue", "mms_issue"):
        hyp = train_hyps(intent, train_ids, 5)[0]
        with ParallelStepper(workers) as st:
            f0 = ParticleFilter("deployable", cfg_real, stepper=st)
            beliefs = {"b1": first_obs_belief(f0, intent, hyp)}
            if intent == "mobile_data_issue":
                beliefs["b0"] = f0.init({intent: 1.0})
            for mode in ("deployable", "oracle"):
                for persona in ("default", "noncompliant"):
                    flt = ParticleFilter(mode, cfg_real, persona=persona, stepper=st)
                    for bn, b in beliefs.items():
                        if bn == "b0" and (persona != "default" or mode != "deployable"):
                            continue
                        for an, c in ACTIONS.items():
                            if bn == "b0" and an not in ("check_status_bar", "toggle_airplane_ensure_off", "run_speed_test", "claim", "transfer"):
                                continue
                            out[f"{intent}/{mode}/{persona}/{bn}/{an}"] = _case(flt, b, c)
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


def _assert_equal(got: dict, want: dict) -> None:
    assert list(got) == list(want), sorted(set(got) ^ set(want))[:5]
    exact = os.environ.get("W2P2_EXACT") == "1"
    for case in want:
        g, w = dict(_flat(got[case])), dict(_flat(want[case]))
        assert list(g) == list(w), (case, sorted(set(g) ^ set(w))[:3])  # includes outcome ORDER (keys list is compared below)
        for k, v in w.items():
            if isinstance(v, float) and not isinstance(v, bool):
                assert g[k] == pytest.approx(v, abs=TOL, rel=0), (case, k, g[k], v)
            elif k.endswith("/wmd5") and not exact:
                continue
            else:
                assert g[k] == v, (case, k, g[k], v)


@pytest.fixture(scope="module")
def cfg_real():
    return yaml.safe_load((paths.CONFIG_DIR / "boed_default.yaml").read_text())


def test_filter_golden_equivalence(cfg_real, train_ids):
    got = _golden_run(cfg_real, train_ids, int(os.environ.get("W2P2_WORKERS", "8")))
    if os.environ.get("W2P2_CAPTURE") == "1":
        assert not GOLDEN.exists(), "golden.json exists: delete it deliberately to re-capture"
        FIX.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({"provenance": {"note": "captured from the unoptimised filter (no git; md5 of owned files; originals in orig/)",
                                                     "md5": _md5s(), "inputs": "tests/fixtures/W2-S9/inputs.json first train hyp per intent"},
                                      "cases": got}))
        return
    want = json.loads(GOLDEN.read_text())["cases"]
    _assert_equal(got, want)


# --------------------------------------------------------------------------- helpers of the speed-ups
def test_fast_particles_equal_dataclass_init():
    import dataclasses
    import gc

    from telecom_boed.belief.filter import _gc_paused, _particles
    from telecom_boed.contracts import Particle
    hk = ("mms_issue", frozenset({"a", "b"}))
    got = _particles([hk, hk], ["s1", "s2"], [0.25, 0.75], [(), ("done",)])
    want = (Particle(hk, "s1", 0.25, ()), Particle(hk, "s2", 0.75, ("done",)))
    assert [repr(p) for p in got] == [repr(p) for p in want]
    assert [p.__dict__ for p in got] == [p.__dict__ for p in want]
    with pytest.raises(dataclasses.FrozenInstanceError):
        got[0].weight = 1.0
    assert dataclasses.replace(got[0], weight=0.5).weight == 0.5
    # the GC pause restores the previous state (also when nested or disabled by the caller)
    assert gc.isenabled()
    with _gc_paused():
        assert not gc.isenabled()
        with _gc_paused():
            assert not gc.isenabled()
        assert not gc.isenabled()
    assert gc.isenabled()
    gc.disable()
    try:
        with _gc_paused():
            pass
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_tables_survive_cache_clears_and_order(cfg_real, train_ids):
    """The transition tables / structure registry are pure caches: predict after clear_caches() and on a second filter is identical."""
    intent = "mobile_data_issue"
    with ParallelStepper(1) as st:
        f1 = ParticleFilter("deployable", cfg_real, stepper=st)
        b = first_obs_belief(f1, intent, train_hyps(intent, train_ids, 5)[0])
        c = ACTIONS["toggle_data_ensure_on"]
        d1 = _case(f1, b, c)
        f1.clear_caches()
        assert _case(f1, b, c) == d1
        assert _case(ParticleFilter("deployable", cfg_real, stepper=st), b, c) == d1
        # belief chaining (successor of a successor) goes through the same tables
        k, (_p, b2) = next(iter(f1.predict(b, c).items()))
        assert _case(f1, b2, ACTIONS["run_speed_test"]) == _case(ParticleFilter("deployable", cfg_real, stepper=st), b2, ACTIONS["run_speed_test"])
