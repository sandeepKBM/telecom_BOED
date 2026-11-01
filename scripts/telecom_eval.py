#!/usr/bin/env python
"""[telecom_boed] H6 evaluation CLI (owner: W4-S15): metrics.json + summary.md from run artifacts. Machine-written, no hand numbers.

    python3 -I scripts/telecom_eval.py <run_dir>... [--pair A B] [--out results/eval/<name>]
        [--no-replay] [--threshold 0.5] [--n-boot 2000] [--seed 0] [--ks 1,2,4] [--power PA PB [--discordance PSI | --rho R]]

<run_dir> = results/runs/<run>/ (or just the run name). --pair A B adds a paired comparison (diff = B - A) and loads A/B when they
are not already listed. The agent-claim FA replay needs tau2 + the oracle: when this interpreter has no tau2 the script re-executes
itself under the ddl-telecom env ($TELECOM_EVAL_PYTHON, default /common/users/ss5772/conda_envs/ddl-telecom/bin/python); pass
--no-replay to skip claim FA (it is then reported as skipped, never as 0). CPU only; reads run artifacts, writes only under --out.
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
    ap.add_argument("runs", nargs="*", help="run dirs (results/runs/<run>) or run names")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), default=[], help="paired comparison, diff = B - A (repeatable)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default results/eval/<run names>)")
    ap.add_argument("--no-replay", action="store_true", help="skip the agent-claim FA replay (reported as skipped)")
    ap.add_argument("--threshold", type=float, default=0.5, help="claim probability threshold (default 0.5)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", default=None, help="comma-separated k for pass^k (default 1..max trials)")
    ap.add_argument("--power", nargs=2, type=float, metavar=("PA", "PB"), default=None, help="tasks needed to detect FA rates PA vs PB")
    ap.add_argument("--discordance", type=float, default=None, help="power: discordant fraction psi = P(A!=B)")
    ap.add_argument("--rho", type=float, default=0.0, help="power: phi correlation of paired outcomes when --discordance is not given")
    a = ap.parse_args(argv)

    if not a.no_replay and importlib.util.find_spec("tau2") is None and not os.environ.get("TELECOM_EVAL_REEXEC"):
        py = os.environ.get("TELECOM_EVAL_PYTHON", DEFAULT_PY)
        if Path(py).exists():
            os.environ["TELECOM_EVAL_REEXEC"] = "1"
            os.execv(py, [py, "-I", str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])])
        raise SystemExit("claim-FA replay needs tau2 (not importable here); set TELECOM_EVAL_PYTHON or pass --no-replay")

    from telecom_boed.contracts import CONTRACT_VERSION
    from telecom_boed.eval import metrics as M

    dirs: dict[str, Path] = {}
    for x in [*a.runs, *[y for pr in a.pair for y in pr]]:
        d = _resolve(x)
        dirs.setdefault(d.name, d)
    if not dirs and a.power is None:
        ap.error("give at least one run dir (or --power)")
    ks = [int(k) for k in a.ks.split(",")] if a.ks else None

    results = {}
    for name, d in dirs.items():
        print(f"[telecom_eval] {name} ...", file=sys.stderr, flush=True)
        results[name] = M.evaluate_run(d, replay=not a.no_replay, threshold=a.threshold, n_boot=a.n_boot, seed=a.seed, ks=ks)
    pairs = [M.paired_comparison(results[_resolve(x).name], results[_resolve(y).name], n_boot=a.n_boot, seed=a.seed) for x, y in a.pair]
    power = None
    if a.power is not None:
        power = M.tasks_needed_paired(a.power[0], a.power[1], discordance=a.discordance, rho=a.rho)

    name = "+".join(dirs)[:120] if dirs else "power"
    out = a.out or (HERE / "results" / "eval" / name)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "telecom_eval/1", "contract_version": CONTRACT_VERSION,
        "config": {"threshold": a.threshold, "n_boot": a.n_boot, "seed": a.seed, "replay": not a.no_replay, "ks": ks,
                   "pairs": a.pair, "claim_classifier": "belief.parsers.RuleClaimClassifier"},
        "inputs": {n: {f.name: _md5(f) for f in sorted(d.iterdir()) if f.name in
                       ("results.json", "sidecar.jsonl", "run_meta.json", "events.jsonl")} for n, d in dirs.items()},
        "runs": {n: r["metrics"] for n, r in results.items()},
        "episodes": {n: r["episodes"] for n, r in results.items()},
        "pairs": pairs, "power": power,
    }
    (out / "metrics.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    md = M.render_summary(list(results.values()), pairs)
    if power is not None:
        md += (f"\n## Power\n\nFA {power['p_a']} vs {power['p_b']}, alpha {power['alpha']}, power {power['power']}, "
               f"discordance psi {power['psi']:.4f}: n_normal={power['n_normal']}, n_exact={power['n_exact']} paired tasks\n")
    (out / "summary.md").write_text(md)
    print(md)
    print(f"[telecom_eval] wrote {out}/metrics.json and summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
