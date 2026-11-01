#!/usr/bin/env python
"""[telecom_boed] H7 stage 1 (CPU): rebuild every agent prompt of a logged tau2 run -> prompts.jsonl.

  python scripts/h7_build_prompts.py --run-dir results/runs/f1plrep_526576.35 [--out DIR/prompts.jsonl]

Default out: $H7_ROOT (default /data/local/ss5772/telecom_h7 on westeros, /freespace/local/ss5772/h7 on a node)/<run>/prompts.jsonl.
Needs the ddl-telecom env (tau2). Row schema: telecom_boed.internals.prompts.PROMPT_FIELDS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from telecom_boed import paths  # noqa: E402


def default_root() -> Path:
    for p in (os.environ.get("H7_ROOT"), "/data/local/ss5772/telecom_h7", "/freespace/local/ss5772/h7"):
        if p and Path(p).parent.is_dir():
            return Path(p)
    raise SystemExit("set H7_ROOT")


def tau2_scratch() -> Path:
    for p in (os.environ.get("TELECOM_SCRATCH"), "/data/local/ss5772/tmp/telecom_boed", "/freespace/local/ss5772/tmp/telecom_boed"):
        if p and Path(p).parent.is_dir():
            return Path(p)
    return HERE / ".scratch"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    run = run_dir.name
    out = Path(a.out) if a.out else default_root() / run / "prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    paths.prepare_tau2_data_dir(tau2_scratch() / f"tau2_data_{paths.TAU2_SHA[:8]}")
    from telecom_boed.internals import prompts as P

    P.check_against_tau2()
    policy, tools = P.load_env_inputs()
    assert len(tools) == P.N_AGENT_TOOLS, len(tools)
    results = json.load(open(run_dir / "results.json"))
    tmp = out.with_suffix(".jsonl.tmp")
    n = P.write_jsonl(P.run_prompts(results, policy, tools, run=run), str(tmp))
    os.replace(tmp, out)
    print(f"{run}: {len(results['simulations'])} sims, {n} decision points -> {out} ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
