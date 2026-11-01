#!/usr/bin/env python
"""[telecom_boed] W5-A2 CLI: per-agent-turn labels for the H7 probes (definitions: src/telecom_boed/internals/labels.py).

    python3 -I scripts/h7_labels.py <run_dir>... [--out /data/local/ss5772/telecom_h7/labels/] [--threshold 0.5] [--gamma G] [--strict]

<run_dir> = results/runs/<run>/ or just the run name. Writes <out>/labels.jsonl (one row per assistant message, keyed
(run, task_id, trial, turn, msg_idx) exactly like the capture index) and <out>/summary.json (label balance per field, validation
invariants per run, replay errors). The oracle replay needs tau2: when this interpreter lacks it the script re-executes itself
under the ddl-telecom env ($TELECOM_LABELS_PYTHON, default /common/users/ss5772/conda_envs/ddl-telecom/bin/python).
CPU only. Output goes to westeros /data/local (NOT /common/users). --strict exits 1 when a validation invariant fails.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # .../telecom_boed
sys.path.insert(0, str(HERE / "src"))
DEFAULT_PY = "/common/users/ss5772/conda_envs/ddl-telecom/bin/python"
DEFAULT_OUT = Path("/data/local/ss5772/telecom_h7/labels")


def _resolve(x: str) -> Path:
    p = Path(x)
    if p.is_dir():
        return p
    q = HERE / "results" / "runs" / x
    if q.is_dir():
        return q
    raise SystemExit(f"run dir not found: {x} (tried {p} and {q})")


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs (results/runs/<run>) or run names")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output dir (default {DEFAULT_OUT})")
    ap.add_argument("--threshold", type=float, default=0.5, help="claim probability threshold (default 0.5)")
    ap.add_argument("--gamma", type=float, default=None, help="override reach_avoid.gamma of configs/boed_default.yaml")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any validation invariant or replay fails")
    a = ap.parse_args(argv)

    if importlib.util.find_spec("tau2") is None and not os.environ.get("TELECOM_LABELS_REEXEC"):
        py = os.environ.get("TELECOM_LABELS_PYTHON", DEFAULT_PY)
        if Path(py).exists():
            os.environ["TELECOM_LABELS_REEXEC"] = "1"
            os.execv(py, [py, "-I", str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])])
        raise SystemExit("the oracle replay needs tau2 (not importable here); set TELECOM_LABELS_PYTHON")

    out = a.out.resolve()
    if str(out).startswith("/common/users"):
        raise SystemExit(f"refusing to write under /common/users ({out}); use westeros /data/local")
    out.mkdir(parents=True, exist_ok=True)

    from telecom_boed.internals import labels as L

    dirs = []
    for x in a.runs:
        d = _resolve(x)
        if d not in dirs:
            dirs.append(d)
    all_rows, runs_doc, ok = [], {}, True
    tmp = out / "labels.jsonl.tmp"
    with tmp.open("w") as fh:
        for d in dirs:
            print(f"[h7_labels] {d.name} ...", file=sys.stderr, flush=True)
            res = L.label_run(d, threshold=a.threshold, gamma=a.gamma)
            for r in res["rows"]:
                fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
            v = L.validate(res)
            ok = ok and not (res["errors"] or v["turn_count_mismatch"] or v["turn0_fault_set_mismatch"]
                             or v["last_turn_repaired_ne_reward_user_stop"] or v["replay_final_ne_reward_user_stop"])
            runs_doc[d.name] = {"n_sims": res["n_sims"], "n_rows": len(res["rows"]), "sidecar_matched_episodes": res["sidecar_matched"],
                                "errors": res["errors"], "validation": v, "balance": L.balance(res["rows"]),
                                "inputs": {f.name: _md5(f) for f in sorted(d.iterdir())
                                           if f.name in ("results.json", "sidecar.jsonl", "run_meta.json")}}
            all_rows.extend(res["rows"])
    tmp.replace(out / "labels.jsonl")
    doc = {"schema": L.SCHEMA, "key": ["run", "task_id", "trial", "turn", "msg_idx"], "threshold": a.threshold,
           "gamma": a.gamma if a.gamma is not None else L._gamma(), "fault_names": list(L.FAULT_NAMES),
           "n_rows": len(all_rows), "presence_stats": dict(L.STATS), "all_runs_balance": L.balance(all_rows), "runs": runs_doc}
    (out / "summary.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[h7_labels] {len(all_rows)} rows -> {out}/labels.jsonl; validation {'OK' if ok else 'FAILED'} (see summary.json)", file=sys.stderr)
    return 0 if (ok or not a.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
