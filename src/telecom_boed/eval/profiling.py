"""[telecom_boed] Systems-side run profile (W4-S17), machine-written from existing run artifacts. Read-only, stdlib only.

Per run directory (results/runs/<run>/) it reads: results.json (tau2 messages, timestamps, per-simulation duration,
assistant generation_time_seconds, usage), sidecar.jsonl (per agent turn timings / decision stats / parse.sampling),
vllm_metrics.prom (final counters), events.jsonl (queue depth samples), job_info.txt (tau2_wall_s, vllm_env).
Any artifact may be missing (e.g. the baseline has no sidecar and no prom); the matching section is then None.

Accounting model (what the numbers mean; repeated in summary.md):
  * tau2 runs episodes concurrently in threads, so every "time" below is a sum over episodes of per-episode time.
    Shares are fractions of the summed episode durations, NOT of wall-clock.
  * The episode timeline is partitioned exactly by message timestamps: the gap before an assistant message is agent
    time, before a user message is user-simulator time, before a tool message is env time (tail and system gaps: other).
  * agent LLM = sum of assistant generation_time_seconds. tau2 does not record generation_time_seconds on user
    messages, so user-sim LLM = the timestamp gap before each user message (an upper bound: it includes user-side
    tool execution and queueing). If a user message does carry generation_time_seconds it is used instead.
  * BOED = sidecar filter + parse + score. The sidecar `llm` time equals generation_time_seconds to within 1%, and BOED
    time is not visible as its own gap in the timestamps, so BOED is carved out of the agent-turn gap: when
    generation_time + BOED exceeds the agent-turn gap, the excess (`boed_overlap_s`) is removed from agent LLM.
  * env/tool/other = the remainder (tool-message gaps, queueing between turns, setup, tail).
  * Sidecar sim ids are not tau2 simulation ids, so BOED cannot be joined per episode; it is a run-level sum.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------------------------- small helpers


def percentile(values, q: float):
    """Linear-interpolation percentile (q in [0, 100]); None on empty input."""
    xs = sorted(float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _div(a, b):
    return None if a is None or b in (None, 0) else a / b


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # a live run can leave a torn last line
    except OSError:
        pass
    return rows


# ---------------------------------------------------------------------------------------------- text artifacts


def parse_prom(text: str) -> dict[str, float]:
    """Prometheus text -> {metric_name: value summed over label sets} (HELP/TYPE lines skipped).

    Labelled variants that split one counter by a label (e.g. prompt_tokens_by_source_total) are summed too, which is
    right for engine/model labels and is never used for the headline counters below."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eEinfNa]+)\s*$", line)
        if not m:
            continue
        try:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
        except ValueError:
            continue
    return out


def parse_job_info(text: str) -> dict:
    """Pull the few provenance fields the profile needs out of job_info.txt."""
    info: dict = {"vllm_env": None}
    m = re.search(r"^vllm_env:(.*)$", text, re.M)
    if m:
        info["vllm_env"] = m.group(1).strip()
    for key in ("tau2_wall_s", "step_wall_s", "gpu_hours_est", "vllm_ready_s", "conc", "trials", "host", "split", "model"):
        m = re.search(rf"(?:^|\s){key}=(\S+)", text)
        if m:
            val = m.group(1)
            try:
                info[key] = float(val) if re.fullmatch(r"[0-9.]+", val) and key not in ("conc", "trials") else (
                    int(val) if re.fullmatch(r"[0-9]+", val) else val)
            except ValueError:
                info[key] = val
    return info


def detect_serving(job_info: dict | None, tag: str) -> dict:
    """Deterministic vs default vLLM serving. Source of truth is job_info `vllm_env:`; the run tag (sdA/sdB) is only a
    fallback for old runs that wrote no vllm_env line."""
    env = (job_info or {}).get("vllm_env")
    if env is not None:
        det = "VLLM_BATCH_INVARIANT=1" in env or "--no-enable-prefix-caching" in env
        return {"mode": "deterministic" if det else "default", "source": "job_info.vllm_env", "vllm_env": env}
    if re.match(r"^sd[A-Za-z](?:_|$)", tag):
        return {"mode": "deterministic", "source": "run_tag", "vllm_env": None}
    return {"mode": "default", "source": "assumed (no vllm_env line)", "vllm_env": None}


# ---------------------------------------------------------------------------------------------- episodes


def episode_breakdown(sim: dict) -> dict:
    """Exact timestamp partition of one tau2 simulation plus the recorded LLM times."""
    msgs = sim.get("messages") or []
    start = _ts(sim["start_time"]) if sim.get("start_time") else None
    end = _ts(sim["end_time"]) if sim.get("end_time") else None
    dur = sim.get("duration")
    if dur is None and start is not None and end is not None:
        dur = end - start
    gap = {"assistant": 0.0, "user": 0.0, "tool": 0.0, "other": 0.0}
    gen = {"assistant": 0.0, "user": 0.0}
    n_user_gen = 0
    prev = start
    for m in msgs:
        t = _ts(m["timestamp"]) if m.get("timestamp") else None
        if t is not None and prev is not None:
            role = m.get("role")
            gap[role if role in gap else "other"] += max(0.0, t - prev)
        if t is not None:
            prev = t
        g = m.get("generation_time_seconds")
        if g is not None and m.get("role") in gen:
            gen[m["role"]] += float(g)
            n_user_gen += m["role"] == "user"
    if end is not None and prev is not None:
        gap["other"] += max(0.0, end - prev)
    tok = {"prompt": 0, "completion": 0}
    for m in msgs:
        u = m.get("usage") or {}
        tok["prompt"] += int(u.get("prompt_tokens") or 0)
        tok["completion"] += int(u.get("completion_tokens") or 0)
    return {
        "duration_s": dur, "start": start, "end": end, "gap": gap, "gen": gen, "user_gen_msgs": n_user_gen,
        "n_msgs": len(msgs), "tokens": tok, "termination": sim.get("termination_reason"),
    }


def time_budget(eps: list[dict], boed: dict | None) -> dict:
    """Run-level sums of per-episode time (not wall-clock) and their shares of the summed episode duration."""
    total = sum(e["duration_s"] or 0.0 for e in eps)
    agent_gap = sum(e["gap"]["assistant"] for e in eps)
    user_gap = sum(e["gap"]["user"] for e in eps)
    agent_gen = sum(e["gen"]["assistant"] for e in eps)
    user_gen = sum(e["gen"]["user"] for e in eps)
    user_from_gen = sum(e["user_gen_msgs"] for e in eps) > 0
    user_llm = user_gen if user_from_gen else user_gap
    boed_total = (boed or {}).get("total_s", 0.0) if boed else 0.0
    overlap = max(0.0, agent_gen + boed_total - agent_gap) if boed else 0.0
    agent_llm = max(0.0, agent_gen - overlap)
    other = total - agent_llm - user_llm - boed_total
    clipped = other < 0
    other = max(0.0, other)
    t = {
        "episode_total_s": total, "agent_llm_s": agent_llm, "user_llm_s": user_llm, "boed_s": boed_total,
        "env_other_s": other,
    }
    return {
        "time_s": t,
        "share": {k.replace("_s", ""): _div(v, total) for k, v in t.items() if k != "episode_total_s"},
        "agent_generation_raw_s": agent_gen, "agent_turn_gap_s": agent_gap, "user_turn_gap_s": user_gap,
        "tool_gap_s": sum(e["gap"]["tool"] for e in eps), "boed_overlap_s": overlap,
        "user_llm_source": "generation_time_seconds" if user_from_gen else "timestamp_gap_before_user_msg",
        "residual_clipped": clipped,
    }


# ---------------------------------------------------------------------------------------------- sidecar


def _decision_stats(row: dict) -> dict | None:
    """(elapsed_ms, depth, depth_target, budget_ms, fallback, overran) from decision.stats, else parsed from `reason`."""
    dec = row.get("decision") or {}
    st = dec.get("stats") or {}
    reason = dec.get("reason") or ""
    elapsed = st.get("elapsed_ms")
    depth = st.get("depth")
    target = st.get("depth_target")
    budget = st.get("budget_ms")
    if elapsed is None:
        m = re.search(r"elapsed=([0-9.]+)ms", reason)
        elapsed = float(m.group(1)) if m else None
    if depth is None or target is None:
        m = re.search(r"depth=(\d+)/(\d+)", reason)
        if m:
            depth = int(m.group(1)) if depth is None else depth
            target = int(m.group(2)) if target is None else target
    if budget is None:
        m = re.search(r"budget=([0-9.]+)ms", reason)
        budget = float(m.group(1)) if m else None
    fallback = st.get("fallback")
    if fallback is None and "FALLBACK=" in reason:
        fallback = reason.split("FALLBACK=", 1)[1].split(";")[0].strip() or "unknown"
    if elapsed is None and depth is None:
        return None
    overran = st.get("overran")
    if overran is None and elapsed is not None and budget is not None:
        overran = elapsed > budget
    return {"elapsed_ms": elapsed, "depth": depth, "depth_target": target, "budget_ms": budget,
            "fallback": fallback, "overran": overran, "mode": dec.get("mode"), "scorer": st.get("scorer") or dec.get("scorer")}


def sidecar_profile(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    comp = {"llm": 0.0, "filter": 0.0, "parse": 0.0, "score": 0.0}
    per_turn = {k: [] for k in comp}
    for r in rows:
        tm = r.get("timings") or {}
        for k in comp:
            v = tm.get(k)
            if v is not None:
                comp[k] += float(v)
                per_turn[k].append(float(v))
    boed = {
        "n_turns": len(rows), "filter_s": comp["filter"], "parse_s": comp["parse"], "score_s": comp["score"],
        "total_s": comp["filter"] + comp["parse"] + comp["score"], "llm_s": comp["llm"],
        "per_turn_p50_s": {k: percentile(v, 50) for k, v in per_turn.items()},
        "per_turn_p95_s": {k: percentile(v, 95) for k, v in per_turn.items()},
    }

    decs = [d for d in (_decision_stats(r) for r in rows) if d is not None]
    lat = [d["elapsed_ms"] for d in decs if d["elapsed_ms"] is not None]
    depths = [d["depth"] for d in decs if d["depth"] is not None]
    reached = [d for d in decs if d["depth"] is not None and d["depth_target"] is not None]
    budgets = Counter(d["budget_ms"] for d in decs if d["budget_ms"] is not None)
    controller = {
        "n_turns": len(rows), "n_decisions": len(decs),
        "modes": dict(Counter(d["mode"] or "none" for d in decs)),
        "scorers": dict(Counter(d["scorer"] for d in decs if d["scorer"])),
        "latency_ms": {"p50": percentile(lat, 50), "p95": percentile(lat, 95), "max": max(lat) if lat else None, "n": len(lat)},
        "budget_ms": budgets.most_common(1)[0][0] if budgets else None,
        "overran_rate": _div(sum(1 for d in decs if d["overran"]), sum(1 for d in decs if d["overran"] is not None)),
        "fallback_rate": _div(sum(1 for d in decs if d["fallback"]), len(decs)),
        "depth_mean": (sum(depths) / len(depths)) if depths else None,
        "depth_max": max(depths) if depths else None,
        "depth_hist": {str(k): v for k, v in sorted(Counter(depths).items())},
        "depth_target_max": max((d["depth_target"] for d in reached), default=None),
        "depth_reached_target_rate": _div(sum(1 for d in reached if d["depth"] >= d["depth_target"]), len(reached)),
    }

    req = kept = mal = emp = trunc = nrows = 0
    for r in rows:
        s = ((r.get("parse") or {}).get("sampling")) or None
        if not s:
            continue
        nrows += 1
        req += int(s.get("n_requested") or 0)
        kept += int(s.get("n_kept") or 0)
        mal += int(s.get("n_malformed") or 0)
        emp += int(s.get("n_empty") or 0)
        trunc += int(s.get("n_truncated") or 0)
    sampling = None
    if nrows:
        sampling = {
            "n_turns_sampled": nrows, "n_requested": req, "n_kept": kept, "n_malformed": mal, "n_empty": emp,
            "n_truncated": trunc, "kept_rate": _div(kept, req), "malformed_rate": _div(mal, req),
            "empty_rate": _div(emp, req), "truncated_rate": _div(trunc, req),
        }
    return {"boed": boed, "controller": controller, "sampling": sampling}


# ---------------------------------------------------------------------------------------------- vLLM


def vllm_profile(prom: dict[str, float] | None, wall_s: float | None, total_episode_s: float, events: list[dict]) -> dict | None:
    if not prom:
        return None
    q = prom.get("vllm:prefix_cache_queries_total")
    h = prom.get("vllm:prefix_cache_hits_total")
    pt = prom.get("vllm:prompt_tokens_total")
    gt = prom.get("vllm:generation_tokens_total")
    qn = prom.get("vllm:request_queue_time_seconds_count")
    qs = prom.get("vllm:request_queue_time_seconds_sum")
    waiting = [e.get("num_requests_waiting") for e in events if e.get("phase") == "progress" and e.get("num_requests_waiting") is not None]
    running = [e.get("num_requests_running") for e in events if e.get("phase") == "progress" and e.get("num_requests_running") is not None]
    return {
        "wall_s": wall_s, "prompt_tokens": pt, "generation_tokens": gt,
        "prompt_tok_per_s": _div(pt, wall_s), "generation_tok_per_s": _div(gt, wall_s),
        "prefix_cache_queries": q, "prefix_cache_hits": h,
        "prefix_cache_hit_rate": _div(h, q) if q else None,
        "prefix_cache_note": "cache off or no queries" if not q else None,
        "n_requests": qn, "queue_time_s_sum": qs, "queue_time_mean_ms": (1000 * qs / qn) if qs is not None and qn else None,
        "queue_share_of_episode_time": _div(qs, total_episode_s),
        "prefill_time_s_sum": prom.get("vllm:request_prefill_time_seconds_sum"),
        "decode_time_s_sum": prom.get("vllm:request_decode_time_seconds_sum"),
        "waiting_max_sampled": max(waiting) if waiting else None,
        "running_max_sampled": max(running) if running else None,
    }


# ---------------------------------------------------------------------------------------------- one run


def profile_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    tag = run_dir.name
    job_txt = ""
    try:
        job_txt = (run_dir / "job_info.txt").read_text()
    except OSError:
        pass
    job = parse_job_info(job_txt) if job_txt else {}
    res = _read_json(run_dir / "results.json") or {}
    sims = res.get("simulations") or []
    eps = [episode_breakdown(s) for s in sims]
    durs = [e["duration_s"] for e in eps if e["duration_s"] is not None]

    wall = job.get("tau2_wall_s") if isinstance(job.get("tau2_wall_s"), (int, float)) else None
    wall_src = "job_info.tau2_wall_s"
    if wall is None and eps:
        starts = [e["start"] for e in eps if e["start"] is not None]
        ends = [e["end"] for e in eps if e["end"] is not None]
        if starts and ends:
            wall, wall_src = max(ends) - min(starts), "first_start_to_last_end"
    side = sidecar_profile(_read_jsonl(run_dir / "sidecar.jsonl"))
    budget = time_budget(eps, side["boed"] if side else None) if eps else None

    prom = None
    try:
        prom = parse_prom((run_dir / "vllm_metrics.prom").read_text())
    except OSError:
        pass
    total_ep = sum(durs)
    vllm = vllm_profile(prom, wall, total_ep, _read_jsonl(run_dir / "events.jsonl"))
    client_tok = {"prompt": sum(e["tokens"]["prompt"] for e in eps), "completion": sum(e["tokens"]["completion"] for e in eps)}
    client = {
        "prompt_tokens": client_tok["prompt"], "completion_tokens": client_tok["completion"],
        "prompt_tok_per_s": _div(client_tok["prompt"], wall), "completion_tok_per_s": _div(client_tok["completion"], wall),
        "note": "sum of per-message usage; each request re-sends the full history, so this tracks vLLM prompt_tokens_total",
    }
    return {
        "run": tag,
        "serving": detect_serving(job if job_txt else None, tag),
        "n_episodes": len(eps),
        "conc": job.get("conc"),
        "wall_s": wall, "wall_source": wall_src if wall is not None else None,
        "episode_s": {"p50": percentile(durs, 50), "p95": percentile(durs, 95), "mean": (total_ep / len(durs)) if durs else None,
                      "max": max(durs) if durs else None, "total": total_ep},
        "termination": dict(Counter(e["termination"] or "none" for e in eps)),
        "time_budget": budget,
        "vllm": vllm,
        "client_tokens": client if eps else None,
        "controller": side["controller"] if side else None,
        "boed": side["boed"] if side else None,
        "sampling": side["sampling"] if side else None,
        "artifacts": {n: (run_dir / n).exists() for n in ("results.json", "sidecar.jsonl", "vllm_metrics.prom", "events.jsonl", "job_info.txt")},
    }


# ---------------------------------------------------------------------------------------------- scorer microbenchmarks


def scorer_microbenchmarks(timings_dir) -> list[dict]:
    """Flatten results/timings/W2-*.json into rows {source, case, cold_p50_ms, warm_p50_ms, warm_p95_ms, n_particles}.

    W2-S5: per (action, worker count) outcome-distribution timings. W2-S9: per intent x scorer, warm p50 only.
    W2-P1: before/after scorer speed-up (cold and warm p50/p95 per intent x scorer). Unknown shapes are skipped."""
    rows: list[dict] = []
    tdir = Path(timings_dir)
    d = _read_json(tdir / "W2-S5.json")
    if isinstance(d, dict):
        for case, v in (d.get("results") or {}).items():
            if isinstance(v, dict) and "warm_p50_ms" in v:
                rows.append({"source": "W2-S5", "case": case.replace("|", " ; "), "cold_p50_ms": v.get("cold_p50_ms"), "warm_p50_ms": v.get("warm_p50_ms"),
                             "warm_p95_ms": v.get("warm_p95_ms"), "n_particles": d.get("n_particles")})
    d = _read_json(tdir / "W2-S9.json")
    if isinstance(d, dict):
        for intent, v in (d.get("supports") or {}).items():
            for scorer, r in ((v or {}).get("results") or {}).items():
                if isinstance(r, dict):
                    rows.append({"source": "W2-S9", "case": f"{intent}/{scorer}", "cold_p50_ms": (r.get("cold") or {}).get("ms"),
                                 "warm_p50_ms": r.get("warm_ms_p50"), "warm_p95_ms": None, "n_particles": v.get("n_particles_belief")})
    d = _read_json(tdir / "W2-P1.json")
    if isinstance(d, dict):
        for key in ("latency_before", "latency_after"):
            for intent, v in ((d.get(key) or {}).get("supports") or {}).items():
                for scorer, r in ((v or {}).get("scorers") or {}).items():
                    if isinstance(r, dict):
                        rows.append({"source": f"W2-P1:{key.split('_')[1]}", "case": f"{intent}/{scorer}",
                                     "cold_p50_ms": (r.get("cold_ms") or {}).get("p50"), "warm_p50_ms": (r.get("warm_ms") or {}).get("p50"),
                                     "warm_p95_ms": (r.get("warm_ms") or {}).get("p95"), "n_particles": v.get("n_particles")})
    return rows


# ---------------------------------------------------------------------------------------------- rendering


def _f(x, nd=1, pct=False, suffix=""):
    if x is None:
        return "n/a"
    if pct:
        return f"{100 * x:.{nd}f}%"
    return f"{x:.{nd}f}{suffix}"


def _ppair(a, b, nd=1):
    return f"{_f(a, nd)}/{_f(b, nd)}"


def render_markdown(profiles: list[dict], microbench: list[dict] | None = None) -> str:
    L: list[str] = []
    L.append("# Systems profile (machine-written by `scripts/telecom_profile.py`)\n")
    L.append("How to read this. tau2 runs episodes concurrently in threads, so the time columns are **per-episode sums**, "
             "and shares are fractions of the summed episode durations, not of wall-clock. Agent LLM = assistant "
             "`generation_time_seconds`. tau2 records no `generation_time_seconds` on user messages, so user-sim LLM is the "
             "timestamp gap before each user message (upper bound: includes user-side tool execution and queueing). BOED = "
             "sidecar filter + parse + score; it is not visible as its own timestamp gap, so it is carved out of the agent-turn "
             "gap (excess of generation_time + BOED over the gap is removed from agent LLM). env/tool/other is the remainder. "
             "Sidecar sim ids do not match tau2 simulation ids, so BOED is a run-level sum. Tokens/s = vLLM counters over the "
             "tau2 wall time (cold start excluded; includes idle gaps).\n")

    L.append("## Time budget per run\n")
    L.append("| run | serving | eps | tau2 wall s | episode s p50/p95 | agent LLM | user-sim LLM | BOED | env/tool/other |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p in profiles:
        b = p["time_budget"]
        sh = b["share"] if b else {}
        e = p["episode_s"]
        L.append(f"| {p['run']} | {p['serving']['mode']} | {p['n_episodes']} | {_f(p['wall_s'], 0)} | {_ppair(e['p50'], e['p95'])} | "
                 f"{_f(sh.get('agent_llm'), 1, True)} | {_f(sh.get('user_llm'), 1, True)} | {_f(sh.get('boed'), 1, True)} | "
                 f"{_f(sh.get('env_other'), 1, True)} |")
    L.append("")
    L.append("Absolute per-episode sums (s): agent LLM / user-sim LLM / BOED (filter+parse+score) / env-other / episodes total.\n")
    L.append("| run | agent LLM | user-sim LLM | BOED | (filter / parse / score) | env/other | total | BOED overlap carved |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p in profiles:
        b = p["time_budget"]
        if not b:
            L.append(f"| {p['run']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        t = b["time_s"]
        bo = p["boed"]
        parts = f"{_f(bo['filter_s'])} / {_f(bo['parse_s'], 2)} / {_f(bo['score_s'])}" if bo else "n/a"
        L.append(f"| {p['run']} | {_f(t['agent_llm_s'])} | {_f(t['user_llm_s'])} | {_f(t['boed_s'] if bo else None)} | {parts} | "
                 f"{_f(t['env_other_s'])} | {_f(t['episode_total_s'])} | {_f(b['boed_overlap_s'] if bo else None)} |")
    L.append("")

    L.append("## vLLM throughput and prefix cache\n")
    L.append("| run | serving | prompt tok/s | gen tok/s | prefix-cache hit | queue ms mean | queue share of ep time | waiting max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p in profiles:
        v = p["vllm"]
        if not v:
            ct = p["client_tokens"]
            alt = f"client: {_f(ct['prompt_tok_per_s'], 0)} prompt / {_f(ct['completion_tok_per_s'], 1)} gen (no prom)" if ct else "n/a"
            L.append(f"| {p['run']} | {p['serving']['mode']} | {alt} | | n/a | n/a | n/a | n/a |")
            continue
        hit = _f(v["prefix_cache_hit_rate"], 1, True) if v["prefix_cache_hit_rate"] is not None else "off"
        L.append(f"| {p['run']} | {p['serving']['mode']} | {_f(v['prompt_tok_per_s'], 0)} | {_f(v['generation_tok_per_s'], 1)} | {hit} | "
                 f"{_f(v['queue_time_mean_ms'], 1)} | {_f(v['queue_share_of_episode_time'], 2, True)} | {_f(v['waiting_max_sampled'], 0)} |")
    L.append("")

    L.append("## Controller decision latency and candidate sampling\n")
    L.append("| run | decisions | mode | latency ms p50/p95/max | budget ms | over budget | fallback | depth mean (max) / target | at target | kept | truncated | malformed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in profiles:
        c, s = p["controller"], p["sampling"]
        if not c:
            L.append(f"| {p['run']} | n/a (no sidecar) | | | | | | | | | | |")
            continue
        lat = c["latency_ms"]
        modes = ",".join(f"{k}:{v}" for k, v in c["modes"].items())
        depth = f"{_f(c['depth_mean'], 2)} ({c['depth_max']}) / {c['depth_target_max']}"
        sk = (_f(s["kept_rate"], 1, True), _f(s["truncated_rate"], 1, True), _f(s["malformed_rate"], 1, True)) if s else ("n/a",) * 3
        L.append(f"| {p['run']} | {c['n_decisions']} | {modes} | {_f(lat['p50'], 0)}/{_f(lat['p95'], 0)}/{_f(lat['max'], 0)} | "
                 f"{_f(c['budget_ms'], 0)} | {_f(c['overran_rate'], 0, True)} | {_f(c['fallback_rate'], 1, True)} | {depth} | "
                 f"{_f(c['depth_reached_target_rate'], 0, True)} | {sk[0]} | {sk[1]} | {sk[2]} |")
    L.append("")

    L.append("## Serving mode\n")
    for p in profiles:
        sv = p["serving"]
        L.append(f"- {p['run']}: {sv['mode']} [source: {sv['source']}]{' `' + sv['vllm_env'] + '`' if sv['vllm_env'] else ''}")
    L.append("")

    if microbench:
        L.append("## Scorer microbenchmarks (results/timings/W2-*.json, shared 72-core host, absolute ms are noisy)\n")
        L.append("| source | case | particles | cold p50 ms | warm p50 ms | warm p95 ms |")
        L.append("|---|---|---|---|---|---|")
        for r in microbench:
            L.append(f"| {r['source']} | {r['case']} | {r['n_particles'] if r['n_particles'] is not None else 'n/a'} | "
                     f"{_f(r['cold_p50_ms'])} | {_f(r['warm_p50_ms'])} | {_f(r['warm_p95_ms'])} |")
        L.append("")
    return "\n".join(L)
