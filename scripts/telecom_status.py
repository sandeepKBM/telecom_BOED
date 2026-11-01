#!/usr/bin/env python
"""[telecom_boed] Machine-written run status (pattern from jenga's tower_eval_ledger.py): read-only over the run
artifacts the launchers write, so status numbers are never hand-typed.

Sources per run dir (results/runs/<tag>_<job>.<step>/ locally, results/runs/t1_<job>/ for Perlmutter):
  job_info.txt (provenance + vllm_ready_s, tau2_wall_s, step wall, prefix-cache counters), results.json (tau2),
  run_meta.json (launch.py: patches, pins), vllm_metrics.prom.
Writes results/status/runs.jsonl (one line per run) and results/status/summary.md; prints the markdown table.

    python3 -I telecom_boed/scripts/telecom_status.py [--runs-dir telecom_boed/results/runs] [--only <substr>]
Live runs (no results.json yet) are shown from the last progress event in their events.jsonl; `live` is kept only if
Slurm (one batched sacct over ssh) says the job.step is RUNNING, else `ended:<STATE>`; no ssh -> `stale` if the
last event is >30 min old.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parents[1]  # .../telecom_boed


def _load_systems():
    """eval/systems.py is stdlib-only; load it by path so this script keeps working under `python3 -I` without the package."""
    spec = importlib.util.spec_from_file_location("telecom_systems", HERE / "src" / "telecom_boed" / "eval" / "systems.py")
    mod = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules["telecom_systems"] = mod  # dataclass-free module, but keep import semantics normal
    spec.loader.exec_module(mod)
    return mod


def _kv(text: str, key: str) -> str | None:
    m = re.search(rf"(?:^|\s){re.escape(key)}=(\S+)", text)
    return m.group(1) if m else None


def _counter(text: str, name: str) -> float | None:
    m = re.search(rf"^vllm:{name}(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)", text, re.M)
    return float(m.group(1)) if m else None


def pass_hat_k(per_task: dict[str, list[float]], k: int) -> float | None:
    """τ² pass^k: mean over tasks of C(c, k) / C(n, k) with c successes in n trials (tasks with n < k skipped)."""
    vals = [comb(sum(r == 1.0 for r in rs), k) / comb(len(rs), k) for rs in per_task.values() if len(rs) >= k]
    return sum(vals) / len(vals) if vals else None


def summarize(run: Path) -> dict:
    info = (run / "job_info.txt").read_text(errors="replace") if (run / "job_info.txt").exists() else ""
    prom = (run / "vllm_metrics.prom").read_text(errors="replace") if (run / "vllm_metrics.prom").exists() else info
    meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else None
    row: dict = {"run": run.name, "host": _kv(info, "host") or _kv(info, "node"), "mode": _kv(info, "mode"),
                 "patched": bool(meta and meta.get("patches")), "vllm_ready_s": _kv(info, "vllm_ready_s"),
                 "tau2_rc": _kv(info, "tau2_rc"), "tau2_wall_s": _kv(info, "tau2_wall_s"),
                 "gpu_h": _kv(info, "gpu_hours_est") or _kv(info, "gpu_hours"), "step_wall_s": _kv(info, "step_wall_s")}
    ev = run / "events.jsonl"  # live progress written by scripts/run_events.py (launcher)
    events = [json.loads(x) for x in ev.read_text(errors="replace").splitlines() if x.strip()] if ev.exists() else []
    if events:
        row["last_event"] = f"{events[-1]['phase']}@{events[-1]['t'][11:19]}"
    q, h = _counter(prom, "prefix_cache_queries_total"), _counter(prom, "prefix_cache_hits_total")
    row["prefix_hit"] = round(h / q, 4) if q else None
    res = run / "results.json"
    if not res.exists():
        prog = [e for e in events if e.get("phase") in ("progress", "watch_end")]
        if prog and events[-1].get("phase") not in ("step_end",):
            p = prog[-1]
            row["_t"] = events[-1]["t"]
            row.update(status="live", sims=p.get("sims_done"), mean_reward=p.get("mean_reward"),
                       terminations=p.get("terminations"), prefix_hit=p.get("prefix_hit", row["prefix_hit"]))
        else:
            row["status"] = "no results.json"
        return row
    sims = json.loads(res.read_text()).get("simulations", [])
    per_task: dict[str, list[float]] = defaultdict(list)
    term, tool_a, tool_u, think = Counter(), 0, 0, 0
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        per_task[s["task_id"]].append(float(r) if r is not None else 0.0)
        term[str(s.get("termination_reason"))] += 1
        for m in s.get("messages", []):
            if m.get("tool_calls"):
                tool_a += m.get("role") == "assistant"
                tool_u += m.get("role") == "user"
            if m.get("role") == "assistant" and "<think>" in (m.get("content") or ""):
                think += 1
    trials = max((len(v) for v in per_task.values()), default=0)
    rewards = [x for v in per_task.values() for x in v]
    row.update(status="ok", sims=len(sims), tasks=len(per_task), trials=trials,
               mean_reward=round(sum(rewards) / len(rewards), 4) if rewards else None,
               **{f"pass^{k}": (round(p, 4) if (p := pass_hat_k(per_task, k)) is not None else None)
                  for k in range(1, max(trials, 1) + 1)},
               terminations=dict(term), assistant_tool_msgs=tool_a, user_tool_msgs=tool_u, think_msgs=think)
    return row


STALE_S = 1800


def jobstep(name: str) -> str | None:
    """Slurm job.step id from a run dir name like h7nat_526576.38 (None for Perlmutter t1_<job> dirs)."""
    m = re.search(r"_(\d+\.\d+)$", name)
    return m.group(1) if m else None


def sacct_lookup(ids: list[str], timeout: int = 15) -> dict[str, tuple[str, str]] | None:
    """ONE ssh+sacct call for all ids -> {job.step: (STATE, End)}; None if the lookup fails."""
    if not ids:
        return {}
    cmd = ["ssh", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "rlab1",
           f"sacct -j {','.join(ids)} -n -P -o JobID,State,End"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    res: dict[str, tuple[str, str]] = {}
    for ln in out.stdout.splitlines():
        f = ln.split("|")
        if len(f) >= 3:
            res[f[0]] = (f[1].split()[0] if f[1] else "UNKNOWN", f[2])
    return res


def runid(name: str) -> str | None:
    """Slurm id for sacct: job.step for local dirs (tag_<job>.<step>), bare job for Perlmutter dirs (..._<job>)."""
    return jobstep(name) or (m.group(1) if (m := re.search(r"_(\d{6,})$", name)) else None)


def sacct_alloc_lookup(ids: list[str], timeout: int = 20, sacct_file: Path | None = None) -> dict[str, tuple[str, float, str]]:
    """ONE ssh+sacct call -> {id: (AllocTRES, ElapsedRaw_s, QOS)}. `sacct_file` = pre-fetched `sacct -n -P -o JobID,AllocTRES,ElapsedRaw,QOS`
    output (needed for Perlmutter jobs, which this host cannot query). Failure -> {} (columns stay empty)."""
    text = ""
    if sacct_file is not None:
        text = sacct_file.read_text(errors="replace")
    elif ids:
        cmd = ["ssh", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "rlab1",
               f"sacct -j {','.join(ids)} -n -P -o JobID,AllocTRES,ElapsedRaw,QOS"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return {}
        text = out.stdout if out.returncode == 0 else ""
    res: dict[str, tuple[str, float, str]] = {}
    for ln in text.splitlines():
        f = ln.split("|")
        if len(f) >= 4 and f[2].isdigit():
            res[f[0]] = (f[1], float(f[2]), f[3])
    return res


def add_alloc_gpu_h(rows: list[dict], lookup=sacct_alloc_lookup, sacct_file: Path | None = None) -> None:
    """gpu_h_reserved = reserved (allocated) gres/gpu (sacct AllocTRES) x elapsed, under the Perlmutter QOS charge rule for t1_*/bare-job dirs
    (full-node QOS: 4 GPU-h per node-h; shared: per GPU; see eval/systems.py). Differs from `gpu_h` (1 GPU x wall estimate)
    when a job holds more GPUs than the step uses."""
    sysm = _load_systems()
    ids = sorted({i for r in rows if (i := runid(r["run"]))})
    sq = lookup(ids, sacct_file=sacct_file)
    for r in rows:
        i = runid(r["run"])
        r["n_gpus_reserved"], r["gpu_h_reserved"] = None, None
        if i in sq:
            tres, el, qos = sq[i]
            r["n_gpus_reserved"] = sysm.gpu_count(tres) or None
            r["gpu_h_reserved"] = sysm.charged_gpu_h(tres, el, qos, perlmutter=jobstep(r["run"]) is None)


def resolve_liveness(rows: list[dict], lookup=sacct_lookup, now: datetime | None = None) -> None:
    """A progress-only run is `live` only if its Slurm step is RUNNING; else ended:<STATE> (+ended_at), or `stale`/`live`
    by event age (30 min) when Slurm cannot be queried."""
    cand = [r for r in rows if r.get("status") == "live"]
    ids = sorted({j for r in cand if (j := jobstep(r["run"]))})
    sq = lookup(ids) if ids else {}
    for r in cand:
        t = r.pop("_t", None)
        j = jobstep(r["run"])
        if sq is not None and j in sq:
            state, end = sq[j]
            if state not in ("RUNNING", "PENDING"):
                r["status"], r["ended_at"] = f"ended:{state}", end
        elif sq is None and t:
            age = ((now or datetime.now().astimezone()) - datetime.fromisoformat(t)).total_seconds()
            if age > STALE_S:
                r["status"] = "stale"
    for r in rows:
        r.pop("_t", None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=HERE / "results" / "runs")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "status")
    ap.add_argument("--sacct-file", type=Path, default=None, help="pre-fetched sacct output (JobID|AllocTRES|ElapsedRaw|QOS), e.g. from Perlmutter")
    ap.add_argument("--only", default=None, help="substring filter on run dir names")
    a = ap.parse_args()
    runs = sorted(p for p in a.runs_dir.iterdir() if p.is_dir() and (a.only is None or a.only in p.name))
    rows = [summarize(r) for r in runs]
    resolve_liveness(rows)
    add_alloc_gpu_h(rows, sacct_file=a.sacct_file)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cols = ["run", "host", "patched", "sims", "tasks", "trials", "mean_reward", "pass^1", "pass^2", "pass^4",
            "prefix_hit", "vllm_ready_s", "tau2_wall_s", "gpu_h", "gpu_h_reserved", "status", "last_event"]
    lines = ["| " + " | ".join(cols) + " | terminations |", "|" + "---|" * (len(cols) + 1)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + f" | {r.get('terminations', '')} |")
    md = "\n".join(lines) + "\n"
    (a.out / "summary.md").write_text("# telecom_boed run status (machine-written by scripts/telecom_status.py)\n\n" + md)
    print(md)


if __name__ == "__main__":
    main()
