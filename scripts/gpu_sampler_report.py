#!/usr/bin/env python
"""[telecom_boed] GPU / serving utilisation summary of one launcher job (2026-11-01). Machine-written; nothing typed by hand.

    python -I scripts/gpu_sampler_report.py RUN_DIR [--out OUT.md]

RUN_DIR = <tag>_<jobid>/ from slurm/tau2_launch_l2_v2.sbatch, holding job_info.txt, vllm.log and (GPU_SAMPLER=1) gpu_samples.csv.
- GPU rows (gpu_samples.csv, nvidia-smi every GPU_SAMPLER_INTERVAL_S): util % = utilization.gpu ("a kernel ran in the sample
  window", not SM occupancy), memory used MiB, power W; busy share = samples with util > 0. Phases: start-up = samples before
  first sample time + vllm_ready_s (job_info.txt), serving = the rest (start-up boundary is approximate: the sampler starts a few
  seconds after the job).
- vLLM rows (vllm.log "Engine 000:" stats lines, every VLLM_LOG_STATS_INTERVAL = 10 s by default; vLLM writes a line only for
  intervals with activity, so idle gaps are not in these rows): running requests, waiting requests, GPU KV-cache usage %,
  generation and prompt throughput tokens/s, prefix-cache hit rate %.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telecom_boed.eval.systems import read_gpu_samples  # noqa: E402

STATS = re.compile(r"Avg prompt throughput: ([0-9.]+) tokens/s, Avg generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs, "
                   r"Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%, Prefix cache hit rate: ([0-9.]+)%")


def stat(xs):
    xs = sorted(xs)
    if not xs:
        return None
    return {"n": len(xs), "mean": sum(xs) / len(xs), "p50": xs[len(xs) // 2], "max": xs[-1]}


def ts(s):
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.run)
    ji = (d / "job_info.txt").read_text() if (d / "job_info.txt").exists() else ""
    ready = int(m.group(1)) if (m := re.search(r"vllm_ready_s=(\d+)", ji)) else None
    wall = int(m.group(1)) if (m := re.search(r"job_wall_s=(\d+)", ji)) else None
    f = lambda x, k=1: "n/a" if x is None else f"{x:.{k}f}"  # noqa: E731
    L = [f"# GPU / serving utilisation, {d.name} (machine-written by scripts/gpu_sampler_report.py)", "",
         f"vllm_ready_s={ready} job_wall_s={wall} start-up share={f(ready / wall, 2) if ready and wall else 'n/a'} (job_info.txt)", ""]
    rows = read_gpu_samples(d / "gpu_samples.csv")
    if rows:
        t0 = ts(rows[0]["ts"])
        phase = {"all": rows, "start-up": [], "serving": []}
        for r in rows:
            t = ts(r["ts"])
            k = "start-up" if (ready is not None and t0 and t and (t - t0).total_seconds() < ready) else "serving"
            phase[k].append(r)
        L += ["| phase | samples | util % mean / p50 | busy share (util > 0) | memory used MiB mean / p50 / max | power W mean / p50 / max |",
              "|---|---|---|---|---|---|"]
        for k, rs in phase.items():
            if not rs:
                continue
            u, m_, p = stat([r["util_gpu"] for r in rs]), stat([r["mem_MiB"] for r in rs]), stat([r["power_W"] for r in rs])
            busy = sum(r["util_gpu"] > 0 for r in rs) / len(rs)
            L.append(f"| {k} | {len(rs)} | {f(u['mean'])} / {f(u['p50'], 0)} | {f(busy, 3)} | {f(m_['mean'], 0)} / {f(m_['p50'], 0)} / "
                     f"{f(m_['max'], 0)} | {f(p['mean'])} / {f(p['p50'])} / {f(p['max'])} |")
    else:
        L.append("gpu_samples.csv: absent or empty (GPU_SAMPLER was off).")
    L.append("")
    vl = d / "vllm.log"
    ticks = [tuple(float(x) for x in m.groups()) for line in (vl.read_text(errors="replace").splitlines() if vl.exists() else [])
             if (m := STATS.search(line))]
    if ticks:
        L += [f"vLLM stats lines: {len(ticks)} (10 s each when active)", "",
              "| quantity | mean | p50 | max |", "|---|---|---|---|"]
        for i, name in ((2, "running requests"), (3, "waiting requests"), (4, "GPU KV-cache usage %"), (1, "generation tokens/s"),
                        (0, "prompt tokens/s"), (5, "prefix-cache hit rate %")):
            s = stat([t[i] for t in ticks])
            L.append(f"| {name} | {f(s['mean'], 2)} | {f(s['p50'], 2)} | {f(s['max'], 2)} |")
    else:
        L.append("vllm.log: no stats lines.")
    txt = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
