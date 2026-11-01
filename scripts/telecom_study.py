#!/usr/bin/env python3
"""[telecom_boed] W4-S16 study CLI (needs the ddl-telecom python: PyYAML + telecom_boed; imports no tau2 unless --check-names).

  telecom_study.py plan   <study.yaml> [--check-names]   expanded steps, episode count, GPU-h estimate; launches nothing, writes nothing
  telecom_study.py gen-configs <study.yaml>              write the derived per-persona run configs (NEW files only; validated)
  telecom_study.py run    <study.yaml> [--dry-run] [--poll-s N] [--check-names]
                                                         queue the steps inside the GPU hold via `cdx-agent --gpu-lock` (blocks until
                                                         done; use nohup/tmux). --dry-run prints the commands and runs nothing.
  telecom_study.py status <study name | study.yaml>      latest manifest state per step
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telecom_boed.eval import runner  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "gen-configs", "run"):
        p = sub.add_parser(name)
        p.add_argument("study")
        p.add_argument("--check-names", action="store_true", help="also check agent/user registration (imports tau2)")
        if name == "run":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--poll-s", type=float, default=20.0)
    sub.add_parser("status").add_argument("study")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "status":
            name = Path(a.study).stem if a.study.endswith(".yaml") else a.study
            print(runner.status(name))
            return 0
        study = runner.load_study(a.study)
        steps = runner.expand(study)
        if a.check_names:
            runner.check_step_names(steps)
        if a.cmd == "plan":
            print(runner.format_plan(study, steps))
        elif a.cmd == "gen-configs":
            for p in runner.materialize(steps):
                print("wrote", p)
            print(f"{sum(s.derived_text is not None for s in steps)} derived configs present")
        else:
            res = runner.run_study(study, steps, dry_run=a.dry_run, poll_s=a.poll_s)
            print(res)
            return 1 if res.get("failed") else 0
    except runner.StudyError as e:
        print(f"telecom_study: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
