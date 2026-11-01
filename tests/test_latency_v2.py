"""P0-2: CPU latency comparison boed_default vs boed_v2 with the test_controller::test_latency_report machinery (cold + warm decisions on
the train-only mobile_data (254) / mms (1,984) supports at the 500 ms budget). Adds, per candidate set, a 6-action request_device BUNDLE
(the 2026-10-28 replay found all 36 slowest live decisions were bundles). Writes results/timings/boed_v2_latency.json and PRINTS
p50/p95 per (config, intent, scorer). Absolute numbers move with host load (loadavg recorded); workers via W2S9_WORKERS (default 4 here)."""
from __future__ import annotations

import json
import os
import time

import pytest
import yaml

from telecom_boed import paths
from telecom_boed.belief.filter import ParticleFilter
from telecom_boed.belief.stepper import ParallelStepper
from telecom_boed.contracts import CandidateAction
from telecom_boed.decide.controller import Controller
from telecom_boed.env import particles as P
from telecom_boed.eval.profiling import percentile
from tests.test_controller import REQ, U, first_obs_belief, hand_candidates, train_hyps

OUT = paths.PACKAGE_ROOT / "results" / "timings" / "boed_v2_latency.json"
BUNDLE = [U("check_status_bar"), U("check_network_status"), U("run_speed_test"), U("check_vpn_status"),
          U("check_data_restriction_status"), U("check_network_mode_preference")]


def _cands(intent):
    base = hand_candidates(intent)
    return {"plain": base, "with_bundle": base + [CandidateAction(REQ, tuple(BUNDLE), 4)]}


@pytest.mark.skipif(os.environ.get("RUN_LATENCY_V2") != "1", reason="slow CPU benchmark (hours with boed_default bundles): opt in with RUN_LATENCY_V2=1")
def test_latency_old_vs_v2(train_ids):
    workers = int(os.environ.get("W2S9_WORKERS", "4"))
    cfgs = {n: yaml.safe_load((paths.CONFIG_DIR / f"{n}.yaml").read_text()) for n in ("boed_default", "boed_v2")}
    out = {"budget_ms": 500, "workers": workers, "loadavg_at_start": os.getloadavg(), "rows": []}
    for intent in ("mobile_data_issue", "mms_issue"):
        hyps = train_hyps(intent, train_ids, 5)
        with ParallelStepper(1) as st0:
            f0 = ParticleFilter("deployable", cfgs["boed_default"], stepper=st0)
            b1s = [first_obs_belief(f0, intent, h) for h in hyps]
        for cname, cfg in [(k, v) for k, v in cfgs.items() if k in os.environ.get("LATENCY_CFGS", "boed_default,boed_v2").split(",")]:
            for name in ("dug", "kstep", "reach"):
                for cset, cands in _cands(intent).items():
                    with ParallelStepper(workers) as st:
                        P.clear_caches()
                        flt = ParticleFilter("deployable", cfg, stepper=st)
                        ctl = Controller(name, flt, flt.grader, cfg, override="filter_commit", time_budget_ms=500)
                        t = time.perf_counter()
                        ctl.decide(b1s[0], cands)
                        cold = (time.perf_counter() - t) * 1e3
                        warm = []
                        for b in b1s:
                            flt._pcache.clear(); flt._ecache.clear()
                            if hasattr(ctl.scorer, "clear_memo"):
                                ctl.scorer.clear_memo()
                            ctl.decide(b, cands)
                            warm.append(ctl.stats["elapsed_ms"])
                        depth = [s["depth"] for s in ctl.stats_log[1:]]
                    ms = [cold] + warm
                    row = {"cfg": cname, "intent": intent, "scorer": name, "cands": cset, "n": len(ms),
                           "p50_ms": round(percentile(ms, 50), 1), "p95_ms": round(percentile(ms, 95), 1), "max_ms": round(max(ms), 1),
                           "cold_ms": round(cold, 1), "overran": sum(s["overran"] for s in ctl.stats_log), "depths_warm": depth}
                    out["rows"].append(row)
                    print("LAT", row, flush=True)
    out["loadavg_at_end"] = os.getloadavg()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    assert out["rows"]
