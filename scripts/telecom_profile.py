#!/usr/bin/env python
"""[telecom_boed] Machine-written systems profile of run artifacts (W4-S17). Read-only over results/runs/<run>/.

    python3 -I telecom_boed/scripts/telecom_profile.py <run_dir>... --out results/profile/<name>

Writes <out>/profile.json (all numbers) and <out>/summary.md (comparison tables + the accounting caveats) and prints
the summary. Stdlib only; the logic lives in telecom_boed.eval.profiling (src layout, so this script adds src/ to the
path because `-I` ignores PYTHONPATH).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # .../telecom_boed
sys.path.insert(0, str(HERE / "src"))

from telecom_boed.eval import profiling  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="+", help="results/runs/<run> directories")
    ap.add_argument("--out", required=True, help="output directory (profile.json + summary.md)")
    ap.add_argument("--timings-dir", default=str(HERE / "results" / "timings"), help="W2-*.json scorer microbenchmarks")
    ap.add_argument("--no-timings", action="store_true", help="skip the scorer microbenchmark section")
    ap.add_argument("--quiet", action="store_true", help="do not print summary.md")
    args = ap.parse_args(argv)

    profiles = []
    for d in args.run_dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"skip (not a directory): {d}", file=sys.stderr)
            continue
        profiles.append(profiling.profile_run(p))
    if not profiles:
        print("no runs to profile", file=sys.stderr)
        return 2
    micro = [] if args.no_timings or not Path(args.timings_dir).is_dir() else profiling.scorer_microbenchmarks(args.timings_dir)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps({"runs": profiles, "scorer_microbenchmarks": micro}, indent=1))
    md = profiling.render_markdown(profiles, micro)
    (out / "summary.md").write_text(md)
    if not args.quiet:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
