#!/usr/bin/env python3
"""[telecom_boed] Live machine-written progress events for launcher runs (stdlib only; run with python3 -I).

One JSONL line per event in <RESULT_DIR>/events.jsonl (on NFS, so it can be read from westeros while the job runs):
  {"t": iso, "run": name, "phase": ..., ...fields}
Phases written by slurm/local_tau2_vllm.sh: step_start, vllm_ready, tau2_start, progress (watcher), tau2_end, step_end.

    run_events.py emit  --out F --run NAME --phase P [key=value ...]
    Optional async/systems fields (P0-7; all backward compatible, free-form key=value also still works):
      policy_version=<int> gen_start=<epoch s|now> gen_end=<epoch s|now> enqueue=<epoch s|now> dequeue=<epoch s|now>
      (+ item=<id> to pair enqueue/dequeue; read by eval/systems.py)
    run_events.py watch --out F --run NAME --results <tau2 results.json> --metrics <url> --gpu G --ppid PID \
                        [--interval 60] [--heartbeat 300]
The watcher emits `progress` when the number of finished simulations changes (tau2 checkpoints results.json as each
simulation finishes) and a heartbeat otherwise; it exits when --ppid dies or it receives SIGTERM.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write(out: str, run: str, phase: str, **fields) -> None:
    rec = {"t": _now(), "run": run, "phase": phase, **fields}
    with open(out, "a") as f:  # one short line per write; appends from one process at a time
        f.write(json.dumps(rec, sort_keys=False) + "\n")


EPOCH_FIELDS = ("gen_start", "gen_end", "enqueue", "dequeue")  # value `now` -> time.time()
OPTIONAL_FIELDS = ("policy_version",) + EPOCH_FIELDS


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def results_stats(path: str) -> dict | None:
    try:
        with open(path) as f:
            sims = json.load(f).get("simulations", [])
    except (OSError, ValueError):
        return None  # absent, or being rewritten right now: skip this tick
    rewards = [((s.get("reward_info") or {}).get("reward") or 0.0) for s in sims]
    return {"sims_done": len(sims), "reward1": sum(r == 1.0 for r in rewards),
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
            "terminations": dict(Counter(str(s.get("termination_reason")) for s in sims))}


def metrics_stats(url: str) -> dict:
    try:
        text = urllib.request.urlopen(url, timeout=5).read().decode()
    except Exception:
        return {}
    out = {}
    for key in ("prefix_cache_queries_total", "prefix_cache_hits_total", "prompt_tokens_total", "generation_tokens_total",
                "num_requests_running", "num_requests_waiting"):
        m = re.search(rf"^vllm:{key}(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)", text, re.M)
        if m:
            out[key] = float(m.group(1))
    if out.get("prefix_cache_queries_total"):
        out["prefix_hit"] = round(out.get("prefix_cache_hits_total", 0.0) / out["prefix_cache_queries_total"], 4)
    return out


def gpu_mem(gpu: str) -> int | None:
    try:
        r = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def watch(a: argparse.Namespace) -> None:
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    last_n, last_emit, t0 = -1, 0.0, time.time()
    while not stop["flag"]:
        try:
            os.kill(a.ppid, 0)
        except OSError:
            break
        st = results_stats(a.results)
        n = st["sims_done"] if st else last_n
        if (st and n != last_n) or time.time() - last_emit >= a.heartbeat:
            write(a.out, a.run, "progress", elapsed_s=int(time.time() - t0), **(st or {"sims_done": max(last_n, 0)}),
                  **metrics_stats(a.metrics), gpu_mem_MiB=gpu_mem(a.gpu))
            last_n, last_emit = n, time.time()
        for _ in range(a.interval):
            if stop["flag"]:
                break
            time.sleep(1)
    st = results_stats(a.results)
    write(a.out, a.run, "watch_end", elapsed_s=int(time.time() - t0), **(st or {}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--out", required=True)
    e.add_argument("--run", required=True)
    e.add_argument("--phase", required=True)
    e.add_argument("fields", nargs="*", help="key=value")
    w = sub.add_parser("watch")
    for k in ("--out", "--run", "--results", "--metrics", "--gpu"):
        w.add_argument(k, required=True)
    w.add_argument("--ppid", type=int, required=True)
    w.add_argument("--interval", type=int, default=60)
    w.add_argument("--heartbeat", type=int, default=300)
    a = ap.parse_args()
    if a.cmd == "emit":
        fields = dict(kv.split("=", 1) for kv in a.fields if "=" in kv)
        out = {k: (round(time.time(), 3) if k in EPOCH_FIELDS and v == "now" else _coerce(v)) for k, v in fields.items()}
        write(a.out, a.run, a.phase, **out)
    else:
        watch(a)


if __name__ == "__main__":
    sys.exit(main())
