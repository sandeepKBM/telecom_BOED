#!/usr/bin/env python
"""[telecom_boed] P0-3 memory-growth diagnosis: replay recorded episodes through the real boed-plain agent (full particle filter + controller in
shadow mode, recorded LLM outputs) on CPU under tracemalloc, and write a machine-written report with per-episode growth and the top
allocation sites. Process-global caches (env.particles memos) persist across episodes exactly as in a live run; the filter is per episode.

    python scripts/memory_tracemalloc.py --run-dir results/runs/l2x_boedplain_native_train20_60455429 [--run-dir ...] \
        [--max-episodes N] [--max-turns N] [--out results/memory/tracemalloc_<date>.md]

Replay fidelity: the agent is fed the user / assistant-requested tool messages of results.json in order and `tau2.agent.llm_agent.generate` is
patched to return the recorded assistant message, so the BOED path sees the live inputs (it did not diverge from transcripts in the 2026-10-28
replays). A controller-arm transcript is replayed with boed-plain (shadow scoring uses the same Controller, override none), so the Controller
call path and its memory are exercised, but not the N-sample candidate loop. Runs without results.json are listed and skipped (h7nat: events only).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def incoming_for(msgs: list, lo: int, hi: int, UserMessage, ToolMessage, MultiToolMessage):
    """What tau2 hands the agent before the assistant message at index hi: user texts and assistant-requested tool results since lo."""
    got = []
    for m in msgs[lo + 1:hi]:
        if m["role"] == "user" and not m.get("tool_calls"):
            got.append(UserMessage.model_validate(m))
        elif m["role"] == "tool" and (m.get("requestor") or "assistant") == "assistant":
            got.append(ToolMessage.model_validate(m))
    if not got:
        return None
    tools = [g for g in got if isinstance(g, ToolMessage)]
    if len(tools) > 1 and got[-1] is tools[-1]:
        return MultiToolMessage(role="tool", tool_messages=tools)
    return got[-1]


def replay_episode(sim: dict, tools, policy, llm_args: dict, max_turns: int | None) -> dict:
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage
    from telecom_boed.tau2_ext import agents as A

    msgs = sim["messages"]
    agent = A.FACTORIES["boed-plain"](tools=tools, domain_policy=policy, llm="m", llm_args=dict(llm_args))
    st = agent.get_init_state([AssistantMessage.model_validate(msgs[0])] if msgs and msgs[0]["role"] == "assistant" else None)
    cur = {}
    la.generate = lambda *a, **k: cur["m"].model_copy(deep=True)
    prev, turns = 0, 0
    for i, m in enumerate(msgs):
        if i == 0 or m["role"] != "assistant":
            continue
        inc = incoming_for(msgs, prev, i, UserMessage, ToolMessage, MultiToolMessage)
        prev = i
        if inc is None:
            continue
        cur["m"] = AssistantMessage.model_validate(m)
        agent.generate_next_message(inc, st)
        turns += 1
        if max_turns and turns >= max_turns:
            break
    ep = st.boed
    recs = ep.records
    stats = [r.decision.get("stats") for r in recs if r.decision and r.decision.get("stats")]
    flt = ep.flt
    return {"turns": turns, "n_records": len(recs), "decision_ms": [round(s["elapsed_ms"], 1) for s in stats],
            "filter_cache_sizes": {k: len(getattr(flt, k)) for k in dir(flt) if k.startswith("_") and isinstance(getattr(flt, k, None), (dict,))
                                   and len(getattr(flt, k)) > 0} if flt is not None else {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--nframes", type=int, default=6)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "memory" / f"tracemalloc_{date.today().isoformat()}.md")
    a = ap.parse_args()

    from telecom_boed.eval.fa_replay import ensure_tau2
    ensure_tau2()
    from tau2.domains.telecom.environment import get_environment
    from telecom_boed.env import particles as P

    env = get_environment()
    tools, policy = env.get_tools(), env.get_policy()
    tracemalloc.start(a.nframes)
    sections, skipped, results = [], [], {}
    for rd in a.run_dir:
        res = rd / "results.json"
        if not res.exists():
            skipped.append(f"{rd.name}: no results.json ({', '.join(p.name for p in rd.iterdir())})")
            continue
        meta = json.loads((rd / "run_meta.json").read_text()) if (rd / "run_meta.json").exists() else {}
        spec = (meta.get("run_config") or {}).get("spec") or {}
        kw = dict(spec.get("agent_kwargs") or {})
        llm_args = {"temperature": 0.0, "boed_config": str(HERE / "configs" / "boed_default.yaml"), "boed_sidecar": "/dev/null",
                    **{f"boed_{k}": v for k, v in kw.items()}}
        sims = json.loads(res.read_text())["simulations"]
        sims = sims[: a.max_episodes] if a.max_episodes else sims
        rows, base_snap = [], None
        gc.collect()
        for n, sim in enumerate(sims):
            t = time.perf_counter()
            try:
                out = replay_episode(sim, tools, policy, llm_args, a.max_turns or None)
                err = None
            except Exception as e:  # noqa: BLE001 - one bad transcript must not end the diagnosis
                out, err = {"turns": 0, "decision_ms": [], "filter_cache_sizes": {}}, f"{type(e).__name__}: {e}"
            gc.collect()
            cur, peak = tracemalloc.get_traced_memory()
            rows.append({"ep": n, "sim": sim["id"][:8], "task": sim["task_id"][:40], "msgs": len(sim["messages"]), "turns": out["turns"],
                         "wall_s": round(time.perf_counter() - t, 1), "traced_MB": round(cur / 2**20, 1), "peak_MB": round(peak / 2**20, 1),
                         "rss_MB": round(rss_mb(), 1), "max_decision_ms": max(out["decision_ms"], default=None),
                         "memo": len(P._APPLY_MEMO), "grade_memo": len(P._GRADE_MEMO), "state_cache": len(P._STATE_CACHE),
                         "intern": len(P._INTERN), "err": err})
            print(json.dumps(rows[-1]), flush=True)
            if n == 0:
                base_snap = tracemalloc.take_snapshot()
        end = tracemalloc.take_snapshot()
        top = []
        if base_snap is not None and len(rows) > 1:
            flt = [tracemalloc.Filter(False, tracemalloc.__file__), tracemalloc.Filter(False, "<frozen importlib._bootstrap>")]
            for st in end.filter_traces(flt).compare_to(base_snap.filter_traces(flt), "lineno")[: a.top]:
                fr = st.traceback[0]
                top.append({"site": f"{fr.filename.replace(str(HERE) + '/', '')}:{fr.lineno}", "size_diff_MB": round(st.size_diff / 2**20, 2),
                            "count_diff": st.count_diff, "size_MB": round(st.size / 2**20, 2)})
        results[rd.name] = {"rows": rows, "top_sites": top, "final_filter_caches": out["filter_cache_sizes"]}
        tracemalloc.clear_traces()
        P.clear_caches()  # next run starts cold (separate launcher process in the live setting)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# tracemalloc replay ({date.today().isoformat()}; machine-written by scripts/memory_tracemalloc.py)", "",
             f"args: runs={[p.name for p in a.run_dir]} max_episodes={a.max_episodes or 'all'} max_turns={a.max_turns or 'all'} nframes={a.nframes}",
             "traced_MB = tracemalloc current after gc at the end of each episode (process-global memos persist, filters do not); rss_MB = /proc statm;",
             "growth/ep = least-squares slope of traced_MB over episodes. Top sites = size diff between the end of episode 0 and the final snapshot.", ""]
    for name, r in results.items():
        rows = r["rows"]
        xs = [x["traced_MB"] for x in rows]
        n = len(xs)
        slope = (sum((i - (n - 1) / 2) * (x - sum(xs) / n) for i, x in enumerate(xs)) / sum((i - (n - 1) / 2) ** 2 for i in range(n))) if n > 1 else 0.0
        lines += [f"## {name}", f"episodes {n}; traced_MB first {xs[0]} last {xs[-1]}; growth {slope:+.2f} MB/episode (fit); rss_MB last {rows[-1]['rss_MB']}", "",
                  "| ep | sim | turns | wall_s | traced_MB | peak_MB | rss_MB | max_decision_ms | apply_memo | grade_memo | state_cache | intern | err |", "|" + "---|" * 13]
        for x in rows:
            lines.append(f"| {x['ep']} | {x['sim']} | {x['turns']} | {x['wall_s']} | {x['traced_MB']} | {x['peak_MB']} | {x['rss_MB']} | {x['max_decision_ms']} | {x['memo']} | {x['grade_memo']} | {x['state_cache']} | {x['intern']} | {x['err'] or ''} |")
        lines += ["", "Top allocation sites by growth (episode 0 -> last):", "", "| site | size_diff_MB | count_diff | size_MB |", "|---|---|---|---|"]
        lines += [f"| {t['site']} | {t['size_diff_MB']} | {t['count_diff']} | {t['size_MB']} |" for t in r["top_sites"]]
        lines += ["", f"final-episode filter dict sizes: {r['final_filter_caches']}", ""]
    if skipped:
        lines += ["## skipped", *[f"- {s}" for s in skipped], ""]
    a.out.write_text("\n".join(lines))
    a.out.with_suffix(".json").write_text(json.dumps({"results": results, "skipped": skipped}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
