#!/usr/bin/env python
"""[telecom_boed] H7-B/C stage 2 (CPU): tables from scripts/h7_steer.py outputs.

  python -I scripts/h7_steer_report.py --steer OUT --prep PREP --capture-root CAP [--main-layer 28]

B1 (how much can we steer):  per (layer, direction, alpha): mean change in logit P(<tool_call>) on tool-call turns and on
    text turns, the fraction of turns whose decision flips to the other action type, median KL(base || steered).
B2 (what the steered agent writes): per generation condition: tool-call rate, switch rate vs unsteered, JSON / tool
    name / argument-key validity of the tool calls it writes.
C  (can the state tell how steerable a turn is): per turn at the main layer with --c-dir (default dom_top1: the direction of the model's own top-1 choice),
    slope = least-squares d logit_tc / d alpha over alpha in [-1, 1]  (how hard the push lands),
    flip_alpha = smallest |alpha| (pushing toward the other action) that flips the decision, inf if none <= 4,
    margin = |base logit_tc| (the policy's own confidence, the main confound).
    Spearman correlations of slope and flip_alpha with: margin, capture entropy, top-1 prob, projection of h_L on that direction
    (signed toward the other action), turn, prompt length, filter n_particles / p_repaired (sidecar turns only).
    Also R^2 of flip_alpha ~ margin vs ~ margin + each predictor (rank-transformed linear fits).
Writes STEER/report.md and STEER/report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h7_probe_eval import key, load_capture, read_jsonl  # noqa: E402


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    # average ties
    out = np.empty_like(o)
    for v in np.unique(x):
        m = x == v
        out[m] = o[m].mean()
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan"), int(ok.sum())
    ra, rb = rank(a[ok]), rank(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return float(1 - res.var() / y.var())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--main-layer", type=int, default=28)
    ap.add_argument("--c-dir", default="dom_top1", help="direction used for the per-turn steerability analysis (C)")
    a = ap.parse_args()
    S, P = Path(a.steer), Path(a.prep)
    b1 = [r for p in sorted(S.glob("**/b1.jsonl")) for r in read_jsonl(p)]  # S may hold shard subdirs full_s0..
    b2 = [r for p in sorted(S.glob("**/b2.jsonl")) for r in read_jsonl(p)]
    ev = {key(r): r for r in read_jsonl(P / "eval_turns.jsonl")}
    d = np.load(P / "directions.npz")
    layers = [int(x) for x in d["layers"]]
    md = ["# H7-B steering report (machine-written by scripts/h7_steer_report.py)", "",
          f"turns with B1: {len(b1)}; with B2 generations: {len(b2)}. Units: alpha x |direction| added at the decision token; "
          "rand*/probe_tool are rescaled to |dom_tool|. logit = log P(<tool_call>) - log P(anything else) for the first token.", ""]
    rep = {"n_b1": len(b1), "n_b2": len(b2)}

    # ---------------- B1
    agg = defaultdict(lambda: defaultdict(list))
    base_match = []
    for r in b1:
        b = r["base"]["logit_tc"]
        tc = bool(r["is_tool_call"])
        base_tc = b > 0
        base_match.append(base_tc == tc)
        for L, n, al, lg, ent, top1, kl in r["grid"]:
            g = agg[(L, n, al)]
            g["d_tc" if base_tc else "d_text"].append(lg - b)  # grouped by the model's own unsteered choice
            g["flip"].append((lg > 0) != base_tc)
            g["kl"].append(kl)
    md += [f"Unsteered decision (P(<tool_call>) > 0.5) agrees with the logged action on {np.mean(base_match):.3f} of turns.", ""]
    rep["base_agrees_logged"] = float(np.mean(base_match))
    md += ["## B1: shift in logit P(<tool_call>) per alpha", "",
           "| layer | direction | alpha | mean d logit (model chose tool call) | mean d logit (model chose text) | flip rate | median KL |",
           "|---|---|---|---|---|---|---|"]
    rows_b1 = []
    for (L, n, al), g in sorted(agg.items(), key=lambda kv: (kv[0][0] != a.main_layer, kv[0][0], kv[0][1], kv[0][2])):
        row = {"layer": L, "dir": n, "alpha": al, "d_tc": float(np.mean(g["d_tc"])) if g["d_tc"] else None,
               "d_text": float(np.mean(g["d_text"])) if g["d_text"] else None, "flip": float(np.mean(g["flip"])),
               "kl_median": float(np.median(g["kl"]))}
        rows_b1.append(row)
        fs = lambda v: "-" if v is None else f"{v:+.2f}"
        md.append(f"| {L} | {n} | {al:+g} | {fs(row['d_tc'])} | {fs(row['d_text'])} | {row['flip']:.3f} | {row['kl_median']:.3g} |")
    rep["b1"] = rows_b1

    # ---------------- B2
    if b2:
        md += ["", f"## B2: greedy generations under steering at layer {b2[0]['layer']} (first {len(b2)} turns)", "",
               "| direction | alpha | starts with tool call | switched vs unsteered | JSON ok (of tool calls) | tool name ok | arg keys ok | stop=length |",
               "|---|---|---|---|---|---|---|---|"]
        conds = defaultdict(list)
        for r in b2:
            base = r["gens"][0]
            for g in r["gens"]:
                conds[(g["dir"], g["alpha"])].append((g, base))
        rows_b2 = []
        for (n, al), lst in conds.items():
            tcs = [g for g, _ in lst if g["starts_tc"]]
            row = {"dir": n, "alpha": al, "n": len(lst), "tc_rate": float(np.mean([g["starts_tc"] for g, _ in lst])),
                   "switch": float(np.mean([g["starts_tc"] != b["starts_tc"] for g, b in lst])),
                   "json_ok": float(np.mean([bool(g["json_ok"]) for g in tcs if g["json_ok"] is not None])) if [g for g in tcs if g["json_ok"] is not None] else None,
                   "tc_truncated": float(np.mean([g["json_ok"] is None for g in tcs])) if tcs else None,
                   "name_ok": float(np.mean([bool(g["name_ok"]) for g in tcs])) if tcs else None,
                   "args_ok": float(np.mean([bool(g["args_ok"]) for g in tcs])) if tcs else None,
                   "stop_length": float(np.mean([g["stop"] == "length" for g, _ in lst]))}
            rows_b2.append(row)
            f = lambda v: "-" if v is None else f"{v:.2f}"
            md.append(f"| {n or 'none'} | {al:+g} | {row['tc_rate']:.2f} | {row['switch']:.2f} | {f(row['json_ok'])} | {f(row['name_ok'])} | "
                      f"{f(row['args_ok'])} | {row['stop_length']:.2f} |")
        rep["b2"] = rows_b2

    # ---------------- C
    ML = a.main_layer
    j = layers.index(ML)
    u = d[f"vec_{a.c_dir}"][j].astype(np.float64); u /= np.linalg.norm(u)
    keys_c, H, ent, top1, ntok, turn = load_capture(Path(a.capture_root), [ML])
    hpos = {k: i for i, k in enumerate(keys_c)}
    feats = defaultdict(list)
    for r in b1:
        k = key(r)
        e = ev[k]
        b = r["base"]["logit_tc"]
        toward = -1.0 if b > 0 else 1.0  # push toward the other action
        pts = {al: lg for L, n, al, lg, *_ in r["grid"] if L == ML and n == a.c_dir}
        pts[0.0] = b
        xs = np.array([x for x in sorted(pts) if abs(x) <= 1.0]); ys = np.array([pts[x] for x in xs])
        slope = float(np.polyfit(xs, ys, 1)[0])
        flip = [abs(al) for al, lg in pts.items() if al * toward > 0 and (lg > 0) != (b > 0)]
        flip_alpha = min(flip) if flip else np.inf
        h = H[hpos[k], 0].astype(np.float64)
        filt = e.get("filter") or {}
        feats["slope"].append(slope); feats["flip_alpha"].append(flip_alpha)
        feats["margin"].append(abs(b)); feats["entropy"].append(e["cap_entropy"]); feats["top1_prob"].append(e["cap_top1_prob"])
        feats["proj_toward_other"].append(float(h @ u) * toward)
        feats["turn"].append(e["turn"]); feats["log_n_tokens"].append(float(np.log(e["n_tokens"])))
        feats["filter_n_particles"].append(float(filt["n_particles"]) if filt.get("n_particles") is not None else np.nan)
        feats["filter_p_repaired"].append(float(filt["p_repaired"]) if filt.get("p_repaired") is not None else np.nan)
        feats["is_tool_call"].append(float(e["is_tool_call"]))
    F = {k: np.asarray(v, float) for k, v in feats.items()}
    fa = F["flip_alpha"]
    md += ["", f"## C: per-turn steerability ({a.c_dir}, layer {ML})", "",
           f"turns {len(fa)}; flipped within |alpha| <= 4: {np.isfinite(fa).mean():.3f}; "
           f"median slope {np.median(F['slope']):.2f} logit per alpha (IQR {np.quantile(F['slope'], .25):.2f}-{np.quantile(F['slope'], .75):.2f})", "",
           "flip_alpha quantiles (finite only): " + (", ".join(f"q{int(q * 100)} {np.quantile(fa[np.isfinite(fa)], q):.2f}" for q in (.1, .25, .5, .75, .9)) if np.isfinite(fa).any() else "none"), "",
           "| predictor | Spearman with slope | Spearman with flip_alpha (inf = max rank) | n |", "|---|---|---|---|"]
    fa_rank = np.where(np.isfinite(fa), fa, 1e9)
    rc = {}
    for p in ["margin", "entropy", "top1_prob", "proj_toward_other", "turn", "log_n_tokens", "filter_n_particles", "filter_p_repaired", "is_tool_call"]:
        s1, n1 = spearman(F[p], F["slope"]); s2, n2 = spearman(F[p], fa_rank)
        rc[p] = {"slope": s1, "flip_alpha": s2, "n": n2}
        md.append(f"| {p} | {s1:+.3f} | {s2:+.3f} | {n2} |")
    rep["c_spearman"] = rc
    y = rank(fa_rank)
    base_r2 = r2(y, [rank(F["margin"])])
    md += ["", f"R^2 of rank(flip_alpha) ~ rank(margin): {base_r2:.3f}. Added R^2 from one more predictor:", "",
           "| + predictor | R^2 | added |", "|---|---|---|"]
    add = {}
    for p in ["entropy", "top1_prob", "proj_toward_other", "turn", "log_n_tokens", "is_tool_call"]:
        v = r2(y, [rank(F["margin"]), rank(F[p])])
        add[p] = v
        md.append(f"| {p} | {v:.3f} | {v - base_r2:+.3f} |")
    rep["c_r2"] = {"margin": base_r2, **{f"margin+{k}": v for k, v in add.items()}}
    (S / "report.md").write_text("\n".join(md) + "\n")
    (S / "report.json").write_text(json.dumps(rep, indent=1))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
