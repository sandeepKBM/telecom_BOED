#!/usr/bin/env python
"""[telecom_boed] H7 stage 2 (GPU): capture last-token hidden states for every decision point in a prompts.jsonl.

  python scripts/h7_capture.py --prompts DIR/prompts.jsonl --out DIR [--limit-eps N] [--verify-alone K]
  python scripts/h7_capture.py --job P1 OUT1 --job P2 OUT2 ...   (several runs, one model load)

Run it under a GPU lock (slurm/h7_capture.sh). Needs the ddl-vllm env (torch+transformers+safetensors), HF offline.
Prints one JSON summary line (match rate, anchors, turns/s, size) and writes DIR/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from telecom_boed.internals import capture as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--job", nargs=2, action="append", default=[], metavar=("PROMPTS", "OUT"),
                    help="repeatable (prompts.jsonl, out dir) pair; all jobs share one model load")
    ap.add_argument("--mode", default="prefix", choices=["prefix", "full"])
    ap.add_argument("--limit-eps", type=int, default=None)
    ap.add_argument("--verify-alone", type=int, default=0, help="per episode: recompute K gathered turns alone and diff")
    a = ap.parse_args()
    jobs = list(a.job)
    if a.prompts or a.out:
        if not (a.prompts and a.out):
            ap.error("--prompts and --out go together")
        jobs.insert(0, (a.prompts, a.out))
    if not jobs:
        ap.error("give --prompts/--out or at least one --job")
    t0 = time.perf_counter()
    cap = C.Capturer()
    t_load = time.perf_counter() - t0
    for prompts, out_dir in jobs:
        t1 = time.perf_counter()
        st = C.capture_run(prompts, out_dir, mode=a.mode, limit_eps=a.limit_eps, verify_alone=a.verify_alone, cap=cap)
        wall = time.perf_counter() - t1
        out = Path(out_dir)
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        st.update(load_s=round(t_load, 1), wall_s=round(wall, 1), size_mb=round(size / 1e6, 1),
                  turns_per_s_forward=round(st["turns"] / st["seconds"], 3) if st["seconds"] else None,
                  turns_per_s_wall=round(st["turns"] / wall, 3) if wall else None, gpu_hours=round(wall / 3600, 4),
                  match_rate=round(st["match"] / st["turns"], 5) if st["turns"] else None)
        (out / "summary.json").write_text(json.dumps(st, indent=1))
        print(f"H7_SUMMARY run={out.name} " + json.dumps({k: v for k, v in st.items() if k not in ("mismatch", "verify")}), flush=True)
        if st["mismatch"]:
            print(f"H7_MISMATCH run={out.name} n={len(st['mismatch'])} first={st['mismatch'][:5]}", flush=True)
        for v in st["verify"][:6]:
            print(f"H7_VERIFY run={out.name} " + json.dumps(v), flush=True)
    print(f"H7_ALL_DONE jobs={len(jobs)} load_s={round(t_load, 1)} wall_s={round(time.perf_counter() - t0, 1)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
