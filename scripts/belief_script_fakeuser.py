#!/usr/bin/env python
"""[telecom_boed] Test 1 CPU proxy (2026-11-01): boed-script on the first N train tasks against a SCRIPTED, fully compliant user
that runs each requested phone action on the real tau2 env and replies with the raw tool output. No LLM, no GPU.

    python -I scripts/belief_script_fakeuser.py --scorer dug --n 20 --out OUT.jsonl [--ids-file IDS.json] [--config configs/boed_v2.yaml] [--budget-ms 2000]
    python -I scripts/belief_script_fakeuser.py --summarize A.jsonl B.jsonl ...   (markdown table per file, machine-written)

EVALUATION code (reads the task to build the env and grade it); the agent under test never does. Per episode it writes: task id,
reward (P.grade on the final env state + transfer flag), claimed, transferred, decisions, claims chosen, end-of-episode
P(repaired), agent turns, false acceptance (claimed and reward 0). The upper bound this gives: the user model's compliance and
report noise are absent, so a low score here points at the filter / action set, not at the user simulator.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tau2.data_model.message import ToolCall, ToolMessage, UserMessage  # noqa: E402

from telecom_boed import paths  # noqa: E402
from telecom_boed.belief.user_model import ENSURE_READ  # noqa: E402
from telecom_boed.env import particles as P  # noqa: E402
from telecom_boed.env import snapshot as S  # noqa: E402
from telecom_boed.env.featurize import featurize  # noqa: E402
from telecom_boed.env.hypotheses import hypothesis_by_key  # noqa: E402
from telecom_boed.env.oracle_hidden import true_key  # noqa: E402
from telecom_boed.tau2_ext import belief_script as B  # noqa: E402


def run(task: dict, scorer: str, config: str, budget_ms: int, max_steps: int = 120) -> dict:
    hyp = hypothesis_by_key(true_key(task))
    env = S.restore(P.build_state(hyp))
    agent = B.BoedScriptAgent(tools=[], domain_policy="policy", llm="none",
                              llm_args={"boed_config": config, "boed_scorer": scorer, "boed_time_budget_ms": budget_ms})
    state = agent.get_init_state()
    phone = re.search(r"\d{3}-\d{3}-\d{4}", task["user_scenario"]["instructions"]["known_info"]).group(0)
    msg = UserMessage(role="user", content=task["user_scenario"]["instructions"]["reason_for_call"])
    transferred = claimed = False
    n = 0
    t0 = time.time()
    for _ in range(max_steps):
        out, state = agent.generate_next_message(msg, state)
        n += 1
        if out.tool_calls:
            tc = out.tool_calls[0]
            transferred |= tc.name == B.TRANSFER_TOOL
            tm = env.get_response(tc)
            msg = ToolMessage(id=tc.id, role="tool", content=tm.content, requestor="assistant", error=tm.error)
            continue
        if B.BoedScriptAgent.is_stop(out) or (transferred and out.content == B.TRANSFER_SAY):
            break
        if out.content == B.CLOSING:
            claimed = True
            msg = UserMessage(role="user", content="Thanks, bye.")
            continue
        if out.content in (B.ASK_PHONE, B.ASK_PHONE_AGAIN):
            msg = UserMessage(role="user", content=f"My number is {phone}.")
            continue
        cand = agent._last_ep.last
        replies = []
        for a in (cand.actions if cand is not None else ()):
            if a.requestor != "user":
                continue
            if a.ensure is not None:
                rtool, feat = ENSURE_READ[a.ensure[0]]
                cur = featurize(rtool, env.get_response(ToolCall(id=f"r{n}", name=rtool, arguments={}, requestor="user")).content)
                if cur.get(feat) == a.ensure[1]:
                    replies.append(f"It is already {'on' if a.ensure[1] else 'off'}.")
                    continue
            tm = env.get_response(ToolCall(id=f"u{n}", name=a.tool, arguments=a.args_dict(), requestor="user"))
            replies.append(tm.content)
        msg = UserMessage(role="user", content=" ".join(replies) or "ok")
    reward = P.grade(S.snapshot(env), hyp, transferred)
    recs = agent.records
    boed = [r for r in recs if r.decision.get("phase") == "boed"]
    claims = sum(1 for r in boed if r.decision.get("chosen", {}).get("kind") == "claim_resolved")
    last_p = next((r.parse.get("belief", {}).get("p_repaired") for r in reversed(recs) if r.parse.get("belief")), None)
    lat = sorted(float(r.timings.get("score", 0.0)) * 1000 for r in boed)
    return {"task_id": task["id"], "scorer": scorer, "config": Path(config).name, "reward": reward, "claimed": claimed,
            "transferred": transferred, "false_accept": bool(claimed and reward < 1.0), "decisions": len(boed),
            "claims_chosen": claims, "p_repaired_end": last_p, "agent_turns": n, "wall_s": round(time.time() - t0, 1),
            "score_ms_p50": lat[len(lat) // 2] if lat else None, "score_ms_max": lat[-1] if lat else None,
            "chosen_tools": [a["tool"] for r in boed for a in r.decision.get("chosen", {}).get("actions", [])]}


def summarize(files: list) -> str:
    rows = ["| file | n | pass | claimed | transfer | false accept | decisions | claims chosen / decision | mean end P(repaired) "
            "| agent turns / success | errors |", "|" + "---|" * 11]
    for fn in files:
        R = [json.loads(l) for l in Path(fn).read_text().splitlines() if l.strip()]
        err = sum(1 for r in R if "error" in r)
        R = [r for r in R if "error" not in r]
        n, ok = len(R), [r for r in R if r["reward"] >= 1.0]
        dec = sum(r["decisions"] for r in R)
        cl = sum(r["claims_chosen"] for r in R)
        pr = [r["p_repaired_end"] for r in R if r["p_repaired_end"] is not None]
        rows.append(f"| {Path(fn).name} | {n} | {len(ok)} | {sum(r['claimed'] for r in R)} | {sum(r['transferred'] for r in R)} | "
                    f"{sum(r['false_accept'] for r in R)} | {dec} | {cl}/{dec} | {sum(pr) / len(pr) if pr else float('nan'):.3f} | "
                    f"{sum(r['agent_turns'] for r in ok) / len(ok) if ok else float('nan'):.1f} | {err} |")
    return "\n".join(rows)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        print(summarize(sys.argv[2:]))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="dug")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--config", default=str(paths.CONFIG_DIR / "boed_v2.yaml"))
    ap.add_argument("--budget-ms", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-file", default=None, help="JSON list of task ids (e.g. the l2x train-20 set); overrides --n")
    a = ap.parse_args()
    cat, sp = paths.telecom_catalog_paths()
    byid = {t["id"]: t for t in json.loads(cat.read_text())}
    ids = json.loads(Path(a.ids_file).read_text()) if a.ids_file else json.loads(sp.read_text())["train"][: a.n]
    done = set()
    out = Path(a.out)
    if out.exists():
        done = {json.loads(l)["task_id"] for l in out.read_text().splitlines() if l.strip()}
    for tid in ids:
        if tid in done:
            continue
        try:
            row = run(byid[tid], a.scorer, a.config, a.budget_ms)
        except Exception as e:  # keep going; the row records the error
            row = {"task_id": tid, "scorer": a.scorer, "error": f"{type(e).__name__}: {e}"}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row.get(k) for k in ("task_id", "reward", "claimed", "decisions", "p_repaired_end", "wall_s", "error")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
