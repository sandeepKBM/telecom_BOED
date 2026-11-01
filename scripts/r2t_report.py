#!/usr/bin/env python
"""[telecom_boed] Per-arm table for the pre-P3 r2t runs (2026-11-01). EVALUATION code (reads tasks + oracle via eval/fa_replay.py).

    python -I scripts/r2t_report.py RUN_DIR [RUN_DIR ...] [--out OUT.md]

RUN_DIR = a job folder holding <config>/{results.json, sidecar.jsonl}. Per arm (config folder):
- pass = tau2 reward 1; agent turns / success = mean number of assistant messages in passing episodes;
- claims / decisions = sidecar decisions whose chosen kind is claim_resolved, over all controller decisions (boed-script: phase
  "boed" only; its lookup / goodbye records are not decisions);
- end P(repaired) = mean over episodes (sidecar sim_id) of the last logged belief p_repaired;
- claimed / FA = eval/fa_replay.py: first agent text the claim classifier scores >= 0.5; FA = the replayed state was not repaired
  at that claim (claims a transfer-requiring task cannot satisfy count as FA too, as in fa_replay).
Machine-written; nothing typed by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telecom_boed import paths  # noqa: E402
from telecom_boed.eval.fa_replay import replay_run  # noqa: E402


def arm_row(d: Path, tasks: dict) -> dict:
    sims = json.loads((d / "results.json").read_text())["simulations"]
    ok = [s for s in sims if ((s.get("reward_info") or {}).get("reward") or 0) >= 1.0]
    turns = [sum(1 for m in s.get("messages") or [] if m.get("role") == "assistant") for s in ok]
    rep = replay_run(sims, tasks)
    dec = claims = 0
    last_p: dict = {}
    sc = d / "sidecar.jsonl"
    if sc.exists():
        for line in sc.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            dd = r.get("decision") or {}
            if dd and dd.get("phase", "boed") == "boed" and "chosen" in dd:
                dec += 1
                claims += (dd.get("chosen") or {}).get("kind") == "claim_resolved"
            p = ((r.get("parse") or {}).get("belief") or {}).get("p_repaired")
            if p is not None:
                last_p[r.get("sim_id")] = p
    term: dict = {}
    for s in sims:
        term[s.get("termination_reason")] = term.get(s.get("termination_reason"), 0) + 1
    return {"arm": d.name, "n": len(sims), "pass": len(ok), "claimed": sum(1 for v in rep.values() if v.get("has_claim")),
            "fa": sum(1 for v in rep.values() if v.get("claim_fa")), "replay_err": sum(1 for v in rep.values() if v.get("status") != "ok"),
            "decisions": dec, "claims": claims, "p_end": sum(last_p.values()) / len(last_p) if last_p else None,
            "turns_succ": sum(turns) / len(turns) if turns else None, "term": term}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cat, _ = paths.telecom_catalog_paths()
    tasks = {t["id"]: t for t in json.loads(cat.read_text())}
    rows = []
    for rd in a.runs:
        for d in sorted(Path(rd).iterdir()):
            if (d / "results.json").exists():
                rows.append(arm_row(d, tasks))
    f = lambda x, k: "n/a" if x is None else f"{x:.{k}f}"  # noqa: E731
    L = [f"# r2t per-arm table (machine-written by scripts/r2t_report.py; runs: {', '.join(Path(r).name for r in a.runs)})", "",
         "| arm | n | pass | claimed | false accept | claims / decisions | end P(repaired) | agent turns / success | endings | replay errors |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['n']} | {r['pass']} | {r['claimed']} | {r['fa']} | {r['claims']}/{r['decisions']} | "
                 f"{f(r['p_end'], 3)} | {f(r['turns_succ'], 1)} | {r['term']} | {r['replay_err']} |")
    txt = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
