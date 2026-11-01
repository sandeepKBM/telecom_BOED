"""[telecom_boed] W4-S17: systems profile (eval/profiling.py + scripts/telecom_profile.py). CPU only, no tau2 import.

Synthetic fixtures are built inline (every computation has hand-checkable numbers); the integration tests run on two
trimmed copies of real runs under tests/fixtures/W4-S17/ (f1ctrep_526576.32 controller+repair, sdA_526576.29 deterministic)."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from telecom_boed.eval import profiling as P

FIX = Path(__file__).parent / "fixtures" / "W4-S17"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "telecom_profile.py"


def _iso(t: float) -> str:
    return f"2026-01-01T00:{int(t) // 60:02d}:{t % 60:06.3f}"


def _sim(start: float, msgs, end: float, dur=None):
    """msgs: list of (t, role, gen_time_or_None, tool_call, (prompt, completion) or None)."""
    out = []
    for t, role, gen, tc, usage in msgs:
        m = {"role": role, "timestamp": _iso(t), "generation_time_seconds": gen, "tool_calls": tc}
        if usage:
            m["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1]}
        out.append(m)
    return {"start_time": _iso(start), "end_time": _iso(end), "duration": end - start if dur is None else dur,
            "messages": out, "termination_reason": "user_stop"}


# ------------------------------------------------------------------------------------------- helpers


def test_percentile():
    assert P.percentile([], 50) is None
    assert P.percentile([7], 95) == 7
    assert P.percentile([1, 2, 3, 4], 50) == 2.5
    assert P.percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert P.percentile([4, None, 1, float("nan")], 100) == 4


PROM = """# HELP x
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="m"} 1.5e+06
vllm:generation_tokens_total{engine="0",model_name="m"} 3000.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 1.5e+06
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 1.2e+06
vllm:request_queue_time_seconds_sum{engine="0",model_name="m"} 4.0
vllm:request_queue_time_seconds_count{engine="0",model_name="m"} 100.0
vllm:num_requests_waiting 0.0
vllm:bad_line notanumber
"""


def test_parse_prom_sums_labels_and_skips_junk():
    d = P.parse_prom(PROM + 'vllm:prompt_tokens_total{engine="1"} 500000.0\n')
    assert d["vllm:prompt_tokens_total"] == 2.0e6
    assert d["vllm:prefix_cache_hits_total"] == 1.2e6
    assert d["vllm:num_requests_waiting"] == 0.0
    assert "vllm:bad_line" not in d


def test_parse_job_info():
    j = P.parse_job_info("run=r job=1 conc=3 trials=4 host=h\nvllm_env: A=1 B=2 \ntau2_rc=0 tau2_wall_s=214\n")
    assert j["tau2_wall_s"] == 214.0 and j["conc"] == 3 and j["trials"] == 4 and j["vllm_env"] == "A=1 B=2"
    assert P.parse_job_info("run=r\n")["vllm_env"] is None


def test_detect_serving():
    det = P.detect_serving({"vllm_env": "VLLM_BATCH_INVARIANT=1 VLLM_EXTRA_ARGS=--enforce-eager"}, "whatever")
    assert det["mode"] == "deterministic" and det["source"] == "job_info.vllm_env"
    assert P.detect_serving({"vllm_env": "VLLM_PREFIX_CACHE_FLAG=--no-enable-prefix-caching"}, "x")["mode"] == "deterministic"
    # an empty vllm_env line is authoritative: default serving, even for a determinism-looking tag
    assert P.detect_serving({"vllm_env": ""}, "sdA_1.2")["mode"] == "default"
    # no line at all: fall back to the run tag
    assert P.detect_serving(None, "sdB_5.6") == {"mode": "deterministic", "source": "run_tag", "vllm_env": None}
    assert P.detect_serving({"vllm_env": None}, "f1ctrl_5.3")["mode"] == "default"
    assert P.detect_serving(None, "p3_plain_native_5.16")["source"].startswith("assumed")


# ------------------------------------------------------------------------------------------- episodes


def test_episode_breakdown_partitions_timeline():
    sim = _sim(0.0, [
        (0.0, "assistant", None, False, None),              # greeting, gap 0
        (4.0, "user", None, False, (100, 10)),               # user gap 4
        (5.0, "assistant", 0.9, False, (200, 20)),           # agent gap 1
        (7.0, "user", None, True, (120, 5)),                 # user gap 2 (tool-calling user)
        (7.5, "tool", None, False, None),                    # tool gap 0.5
        (8.5, "assistant", 0.8, True, (300, 30)),            # agent gap 1
    ], end=10.0)
    e = P.episode_breakdown(sim)
    assert e["gap"] == pytest.approx({"assistant": 2.0, "user": 6.0, "tool": 0.5, "other": 1.5})
    assert sum(e["gap"].values()) == pytest.approx(e["duration_s"])
    assert e["gen"] == pytest.approx({"assistant": 1.7, "user": 0.0})
    assert e["tokens"] == {"prompt": 720, "completion": 65}
    assert e["user_gen_msgs"] == 0 and e["termination"] == "user_stop"


def test_episode_duration_falls_back_to_start_end():
    sim = _sim(1.0, [(2.0, "assistant", 0.5, False, None)], end=3.0)
    sim.pop("duration")
    assert P.episode_breakdown(sim)["duration_s"] == pytest.approx(2.0)


def _ep(dur, agent_gap, user_gap, tool_gap=0.0, agent_gen=0.0, user_gen=0.0, user_msgs=0):
    return {"duration_s": dur, "gap": {"assistant": agent_gap, "user": user_gap, "tool": tool_gap, "other": 0.0},
            "gen": {"assistant": agent_gen, "user": user_gen}, "user_gen_msgs": user_msgs}


def test_time_budget_no_sidecar():
    b = P.time_budget([_ep(100, 40, 50, 5, agent_gen=38), _ep(100, 30, 60, 5, agent_gen=29)], None)
    t = b["time_s"]
    assert t["agent_llm_s"] == 67 and t["user_llm_s"] == 110 and t["boed_s"] == 0
    assert t["env_other_s"] == pytest.approx(23)
    assert b["share"]["agent_llm"] == pytest.approx(0.335) and b["share"]["user_llm"] == pytest.approx(0.55)
    assert sum(b["share"].values()) == pytest.approx(1.0)
    assert b["user_llm_source"] == "timestamp_gap_before_user_msg" and b["boed_overlap_s"] == 0
    assert not b["residual_clipped"]


def test_time_budget_boed_carved_out_of_agent_gap():
    # gen 60 + boed 20 = 80 > agent gap 70 -> overlap 10 removed from agent LLM
    b = P.time_budget([_ep(200, 70, 100, 5, agent_gen=60)], {"total_s": 20.0})
    assert b["boed_overlap_s"] == pytest.approx(10)
    t = b["time_s"]
    assert t["agent_llm_s"] == pytest.approx(50) and t["boed_s"] == 20 and t["user_llm_s"] == 100
    assert t["env_other_s"] == pytest.approx(30)
    assert sum(b["share"].values()) == pytest.approx(1.0)
    # BOED fits in the slack: no carve-out
    b2 = P.time_budget([_ep(200, 90, 100, 5, agent_gen=60)], {"total_s": 20.0})
    assert b2["boed_overlap_s"] == 0 and b2["time_s"]["agent_llm_s"] == 60


def test_time_budget_prefers_recorded_user_generation_time():
    b = P.time_budget([_ep(100, 40, 50, 0, agent_gen=40, user_gen=30, user_msgs=3)], None)
    assert b["time_s"]["user_llm_s"] == 30 and b["user_llm_source"] == "generation_time_seconds"
    assert b["time_s"]["env_other_s"] == pytest.approx(30)  # 100 - 40 - 30


def test_time_budget_clips_negative_remainder():
    b = P.time_budget([_ep(10, 9, 9, agent_gen=9)], {"total_s": 1.0})
    assert b["time_s"]["env_other_s"] == 0 and b["residual_clipped"]


# ------------------------------------------------------------------------------------------- sidecar


def _row(llm=1.0, flt=0.0, parse=0.0, score=0.0, stats=None, reason="", mode="controller", sampling=None):
    r = {"timings": {"llm": llm, "filter": flt, "parse": parse, "score": score},
         "decision": {"mode": mode, "reason": reason, "stats": stats} if (stats is not None or reason or mode) else None}
    if sampling is not None:
        r["parse"] = {"sampling": sampling}
    return r


def test_decision_stats_from_stats_and_from_reason():
    d = P._decision_stats(_row(stats={"elapsed_ms": 571.8, "depth": 1, "depth_target": 3, "budget_ms": 500, "fallback": None,
                                       "overran": True, "scorer": "reach"}))
    assert d["elapsed_ms"] == 571.8 and d["depth"] == 1 and d["depth_target"] == 3 and d["overran"] is True
    d = P._decision_stats(_row(stats={}, reason="scorer=reach depth=2/3 elapsed=120.5ms budget=500ms (x)"))
    assert (d["elapsed_ms"], d["depth"], d["depth_target"], d["budget_ms"], d["overran"]) == (120.5, 2, 3, 500.0, False)
    d = P._decision_stats(_row(stats=None, reason="scorer=reach depth=0/3 elapsed=609.3ms budget=500ms FALLBACK=dug (a); scored 1/4"))
    assert d["fallback"].startswith("dug") and d["overran"] is True
    assert P._decision_stats({"timings": {}, "decision": None}) is None
    assert P._decision_stats(_row(stats={}, reason="", mode="controller")) is None


def test_sidecar_profile_sums_rates_and_controller():
    rows = [
        _row(1.0, 0.5, 0.1, 0.4, {"elapsed_ms": 100, "depth": 3, "depth_target": 3, "budget_ms": 500, "overran": False},
             sampling={"n_requested": 4, "n_kept": 3, "n_malformed": 1, "n_empty": 0, "n_truncated": 0}),
        _row(2.0, 0.0, 0.1, 0.2, {"elapsed_ms": 600, "depth": 1, "depth_target": 3, "budget_ms": 500, "overran": True,
                                    "fallback": "dug"},
             sampling={"n_requested": 4, "n_kept": 2, "n_malformed": 0, "n_empty": 1, "n_truncated": 1}),
        _row(3.0, 0.0, 0.0, 0.0, mode=None),  # no stats row
    ]
    sp = P.sidecar_profile(rows)
    bo = sp["boed"]
    assert (bo["filter_s"], bo["parse_s"], bo["score_s"], bo["llm_s"]) == (0.5, pytest.approx(0.2), pytest.approx(0.6), 6.0)
    assert bo["total_s"] == pytest.approx(1.3) and bo["n_turns"] == 3
    c = sp["controller"]
    assert c["n_decisions"] == 2 and c["latency_ms"]["max"] == 600 and c["latency_ms"]["p50"] == 350
    assert c["overran_rate"] == 0.5 and c["fallback_rate"] == 0.5 and c["budget_ms"] == 500
    assert c["depth_mean"] == 2 and c["depth_max"] == 3 and c["depth_hist"] == {"1": 1, "3": 1}
    assert c["depth_reached_target_rate"] == 0.5 and c["depth_target_max"] == 3 and c["modes"] == {"controller": 2}
    s = sp["sampling"]
    assert (s["n_requested"], s["n_kept"]) == (8, 5) and s["kept_rate"] == 5 / 8
    assert s["malformed_rate"] == 1 / 8 and s["truncated_rate"] == 1 / 8 and s["empty_rate"] == 1 / 8
    assert P.sidecar_profile([]) is None
    assert P.sidecar_profile([_row(mode=None)])["sampling"] is None


# ------------------------------------------------------------------------------------------- vllm


def test_vllm_profile_rates_and_queue():
    v = P.vllm_profile(P.parse_prom(PROM), 100.0, 400.0,
                       [{"phase": "progress", "num_requests_waiting": 2.0, "num_requests_running": 4.0},
                        {"phase": "progress", "num_requests_waiting": 0.0, "num_requests_running": 1.0}, {"phase": "x"}])
    assert v["prompt_tok_per_s"] == 15000 and v["generation_tok_per_s"] == 30
    assert v["prefix_cache_hit_rate"] == pytest.approx(0.8) and v["prefix_cache_note"] is None
    assert v["queue_time_mean_ms"] == pytest.approx(40) and v["queue_share_of_episode_time"] == pytest.approx(0.01)
    assert v["waiting_max_sampled"] == 2.0 and v["running_max_sampled"] == 4.0


def test_vllm_profile_cache_off_and_missing():
    off = P.parse_prom("vllm:prefix_cache_queries_total 0.0\nvllm:prefix_cache_hits_total 0.0\nvllm:prompt_tokens_total 100.0\n")
    v = P.vllm_profile(off, 10.0, 10.0, [])
    assert v["prefix_cache_hit_rate"] is None and v["prefix_cache_note"] and v["queue_time_mean_ms"] is None
    assert P.vllm_profile(None, 10.0, 10.0, []) is None and P.vllm_profile({}, 10.0, 10.0, []) is None
    assert P.vllm_profile(off, None, 10.0, [])["prompt_tok_per_s"] is None


# ------------------------------------------------------------------------------------------- whole run (synthetic dir)


def _write_run(tmp_path, *, sidecar=True, prom=True, job=True):
    r = tmp_path / "x_1.2"
    r.mkdir()
    sims = [
        _sim(0.0, [(0.0, "assistant", None, False, None), (2.0, "user", None, False, (10, 2)),
                   (3.0, "assistant", 0.9, False, (50, 5))], end=3.5),
        _sim(1.0, [(1.0, "assistant", None, False, None), (4.0, "user", None, False, (10, 2)),
                   (6.0, "assistant", 1.8, False, (60, 6))], end=7.0),
    ]
    (r / "results.json").write_text(json.dumps({"simulations": sims}))
    if sidecar:
        rows = [_row(0.9, 0.1, 0.0, 0.1, {"elapsed_ms": 50, "depth": 1, "depth_target": 1, "budget_ms": 500, "overran": False},
                     sampling={"n_requested": 2, "n_kept": 2, "n_malformed": 0, "n_empty": 0, "n_truncated": 0})]
        (r / "sidecar.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\ntorn-line{\n")
    if prom:
        (r / "vllm_metrics.prom").write_text(PROM)
    if job:
        (r / "job_info.txt").write_text("run=x_1.2 conc=2\nvllm_env: \ntau2_wall_s=10\n")
    return r


def test_profile_run_synthetic_full(tmp_path):
    p = P.profile_run(_write_run(tmp_path))
    assert p["n_episodes"] == 2 and p["wall_s"] == 10.0 and p["wall_source"] == "job_info.tau2_wall_s"
    assert p["episode_s"]["p50"] == pytest.approx(4.75) and p["episode_s"]["total"] == pytest.approx(9.5)
    assert p["serving"]["mode"] == "default" and p["termination"] == {"user_stop": 2}
    assert p["vllm"]["prompt_tok_per_s"] == pytest.approx(150000) and p["client_tokens"]["prompt_tokens"] == 130
    assert p["controller"]["n_decisions"] == 1 and p["boed"]["n_turns"] == 1  # torn last line ignored
    b = p["time_budget"]
    assert sum(b["share"].values()) == pytest.approx(1.0)
    assert p["artifacts"] == {"results.json": True, "sidecar.jsonl": True, "vllm_metrics.prom": True, "events.jsonl": False, "job_info.txt": True}
    json.dumps(p)  # fully serialisable


def test_profile_run_missing_artifacts(tmp_path):
    p = P.profile_run(_write_run(tmp_path, sidecar=False, prom=False, job=False))
    assert p["controller"] is None and p["boed"] is None and p["sampling"] is None and p["vllm"] is None
    assert p["wall_source"] == "first_start_to_last_end" and p["wall_s"] == pytest.approx(7.0)
    assert p["time_budget"]["time_s"]["boed_s"] == 0
    assert p["client_tokens"]["prompt_tok_per_s"] == pytest.approx(130 / 7.0)
    assert not p["artifacts"]["sidecar.jsonl"]
    empty = tmp_path / "empty_1.1"
    empty.mkdir()
    pe = P.profile_run(empty)
    assert pe["n_episodes"] == 0 and pe["time_budget"] is None and pe["episode_s"]["p50"] is None


# ------------------------------------------------------------------------------------------- microbenchmarks and rendering


def test_scorer_microbenchmarks(tmp_path):
    (tmp_path / "W2-S5.json").write_text(json.dumps({"n_particles": 10, "results": {
        "a|workers=1": {"cold_p50_ms": 5.0, "warm_p50_ms": 1.0, "warm_p95_ms": 2.0}, "junk": {"x": 1}}}))
    (tmp_path / "W2-S9.json").write_text(json.dumps({"supports": {"mms": {"n_particles_belief": 7, "results": {
        "dug": {"cold": {"ms": 9.0}, "warm_ms_p50": 3.0}}}}}))
    (tmp_path / "W2-P1.json").write_text(json.dumps({
        "latency_before": {"supports": {"mms": {"n_particles": 7, "scorers": {"dug": {"cold_ms": {"p50": 8.0}, "warm_ms": {"p50": 4.0, "p95": 6.0}}}}}},
        "latency_after": {"supports": {"mms": {"n_particles": 7, "scorers": {"dug": {"cold_ms": {"p50": 7.0}, "warm_ms": {"p50": 2.0, "p95": 3.0}}}}}}}))
    rows = P.scorer_microbenchmarks(tmp_path)
    by = {(r["source"], r["case"]): r for r in rows}
    assert len(rows) == 4
    assert by[("W2-S5", "a ; workers=1")]["warm_p95_ms"] == 2.0  # '|' replaced so markdown tables stay intact
    assert by[("W2-S9", "mms/dug")]["cold_p50_ms"] == 9.0 and by[("W2-S9", "mms/dug")]["n_particles"] == 7
    assert by[("W2-P1:before", "mms/dug")]["warm_p50_ms"] == 4.0 and by[("W2-P1:after", "mms/dug")]["warm_p50_ms"] == 2.0
    assert P.scorer_microbenchmarks(tmp_path / "nope") == []


def test_render_markdown_has_caveat_and_every_run(tmp_path):
    full = P.profile_run(_write_run(tmp_path))
    d2 = tmp_path / "b"
    d2.mkdir()
    bare = P.profile_run(_write_run(d2, sidecar=False, prom=False, job=False))
    md = P.render_markdown([full, bare], [{"source": "W2-S5", "case": "c", "cold_p50_ms": 1.0, "warm_p50_ms": 2.0,
                                           "warm_p95_ms": None, "n_particles": 3}])
    assert "per-episode sums" in md and "not of wall-clock" in md
    assert md.count("x_1.2") >= 5
    assert "n/a (no sidecar)" in md and "Scorer microbenchmarks" in md and "no prom" in md


# ------------------------------------------------------------------------------------------- integration on trimmed real runs


@pytest.fixture(scope="module")
def real_ctrl():
    return P.profile_run(FIX / "real_f1ctrep")


@pytest.fixture(scope="module")
def real_sd():
    return P.profile_run(FIX / "real_sdA")


def test_real_controller_run(real_ctrl):
    p = real_ctrl
    assert p["n_episodes"] == 3 and p["wall_s"] == 214.0 and p["serving"]["mode"] == "default"
    durs = [s["duration"] for s in json.loads((FIX / "real_f1ctrep" / "results.json").read_text())["simulations"]]
    assert p["episode_s"]["total"] == pytest.approx(sum(durs)) and p["episode_s"]["p50"] == pytest.approx(sorted(durs)[1])
    b = p["time_budget"]
    assert sum(b["share"].values()) == pytest.approx(1.0) and not b["residual_clipped"]
    assert all(v >= 0 for v in b["time_s"].values())
    assert b["agent_generation_raw_s"] > 0 and b["boed_overlap_s"] >= 0
    # prom: prefix-cache hit rate and tokens/s recomputed independently from the trimmed .prom text
    prom = P.parse_prom((FIX / "real_f1ctrep" / "vllm_metrics.prom").read_text())
    v = p["vllm"]
    assert v["prefix_cache_hit_rate"] == pytest.approx(prom["vllm:prefix_cache_hits_total"] / prom["vllm:prefix_cache_queries_total"])
    assert v["prompt_tok_per_s"] == pytest.approx(8427276 / 214) and v["generation_tok_per_s"] == pytest.approx(35668 / 214)
    assert 0.98 < v["prefix_cache_hit_rate"] < 1.0
    c = p["controller"]
    assert c["n_decisions"] >= 30 and c["modes"].keys() == {"controller"} and c["budget_ms"] == 500
    assert 0 <= c["depth_mean"] <= 3 and c["depth_target_max"] == 3
    assert c["latency_ms"]["p50"] <= c["latency_ms"]["p95"] <= c["latency_ms"]["max"]
    s = p["sampling"]
    assert s["n_kept"] <= s["n_requested"] and s["kept_rate"] <= 1.0
    assert 0 <= s["truncated_rate"] <= 1 and 0 <= s["malformed_rate"] <= 1
    assert p["client_tokens"]["prompt_tokens"] > 0


def test_real_deterministic_run(real_sd):
    p = real_sd
    assert p["serving"]["mode"] == "deterministic" and p["serving"]["source"] == "job_info.vllm_env"
    assert p["controller"] is None and p["sampling"] is None and p["boed"] is None
    assert p["time_budget"]["time_s"]["boed_s"] == 0
    assert p["vllm"]["prefix_cache_hit_rate"] is None and p["vllm"]["prefix_cache_note"]  # prefix caching disabled
    assert p["vllm"]["prompt_tok_per_s"] == pytest.approx(786733 / 285)
    assert p["termination"]  # non-empty


# ------------------------------------------------------------------------------------------- CLI


def _load_script():
    spec = importlib.util.spec_from_file_location("telecom_profile_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_writes_profile_and_summary(tmp_path, capsys):
    out = tmp_path / "prof"
    rc = _load_script().main([str(FIX / "real_f1ctrep"), str(FIX / "real_sdA"), str(tmp_path / "missing"), "--out", str(out),
                              "--no-timings", "--quiet"])
    assert rc == 0 and capsys.readouterr().out == ""
    prof = json.loads((out / "profile.json").read_text())
    assert [r["run"] for r in prof["runs"]] == ["real_f1ctrep", "real_sdA"] and prof["scorer_microbenchmarks"] == []
    md = (out / "summary.md").read_text()
    assert "per-episode sums" in md and "deterministic" in md and "real_f1ctrep" in md
    assert not any(math.isnan(x) for x in [prof["runs"][0]["episode_s"]["p50"]])


def test_cli_no_runs_is_an_error(tmp_path):
    assert _load_script().main([str(tmp_path / "nope"), "--out", str(tmp_path / "o")]) == 2
