#!/usr/bin/env python
"""[telecom_boed] H7 stage 3 (CPU): linear readout probes on captured Qwen3-8B hidden states vs baselines.

  python -I scripts/h7_probe_eval.py --capture-root CAP --labels LABELS.jsonl --prompts-root PR --out OUT [--layers 0,8,...] [--folds 5]

CAP/<run>/{index.jsonl,shards/*.safetensors} (capture.py), LABELS = scripts/h7_labels.py output, PR/<run>/prompts.jsonl (for the
text baseline). Rows join on (run, task_id, trial, msg_idx). Needs numpy, scikit-learn, safetensors:
env /data/local/ss5772/conda_envs/ddl-probe (westeros; pins + rebuild command in constraints/ddl-probe.txt).

Every number is out-of-fold under GroupKFold by task_id (the same task recurs across runs/arms/trials, so episode-level splits leak).
Per target, AUROC (binary) or R^2 (continuous), averaged over folds (within-fold metric; see metric_fm), for:
  probe@L    standardised logistic / ridge on the layer-L last-prompt-token state, L in --layers
  text       TF-IDF (1-2 grams) of the visible conversation (no system prompt) + logistic / ridge  -> "the text already says it"
  position   turn index + prompt length                                                           -> "later turns are different"
  filter     the deployable Bayes filter's own score, where one exists (sidecar rows only)
  shuffled   best-layer probe refit on labels permuted across tasks (selectivity control, Hewitt & Liang 2019)
Writes OUT/results.json, OUT/table.md and OUT/oof.npz (out-of-fold probe scores per row, best layer, for the steering work).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KEY = ("run", "task_id", "trial", "msg_idx")
MIN_CLASS = 30   # binary target needs >= this many rows in each class
MIN_TASKS = 10   # ... spread over >= this many tasks (with ~6 tasks, task-grouped folds push random probes far below 0.5)


def read_jsonl(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def key(r):
    return (r["run"], str(r["task_id"]), int(r.get("trial", 0)), int(r["msg_idx"]))


def load_capture(root: Path, layers):
    from safetensors.numpy import load_file
    keys, H, ent, top1, ntok, turn = [], [], [], [], [], []
    for run_dir in sorted(p for p in root.iterdir() if (p / "index.jsonl").exists()):
        idx = list(read_jsonl(run_dir / "index.jsonl"))
        by_shard = defaultdict(list)
        for r in idx:
            by_shard[r["shard"]].append(r)
        for shard, rows in by_shard.items():
            a = load_file(str(run_dir / shard))
            for r in rows:
                keys.append(key(r))
                H.append(a["h"][r["row"]][layers])
                ent.append(r["entropy"]); top1.append(r["top1_prob"]); ntok.append(r["n_tokens"]); turn.append(r["turn"])
    return keys, np.stack(H), np.asarray(ent), np.asarray(top1), np.asarray(ntok), np.asarray(turn)


def visible_text(row):
    parts = []
    for m in row["messages"][1:]:
        c = m.get("content") or ""
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        parts.append(f"[{m.get('role')}] {c}")
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {})
            parts.append(f"[call] {f.get('name')} {f.get('arguments')}")
    return "\n".join(parts)


def load_text(root: Path, wanted: set):
    out = {}
    for run_dir in sorted(root.iterdir()):
        p = run_dir / "prompts.jsonl"
        if not p.exists():
            continue
        for r in read_jsonl(p):
            k = key(r)
            if k in wanted:
                out[k] = visible_text(r)
    return out


def targets(L):
    """name -> (kind, y array with nan = not applicable). L = label rows aligned with the capture rows."""
    T = {}
    n = len(L)
    def b(f):
        return np.asarray([np.nan if v is None else float(v) for v in map(f, L)], float)
    T["is_tool_call"] = ("bin", b(lambda r: r["is_tool_call"]))
    T["is_device_request"] = ("bin", b(lambda r: r["is_device_request"]))
    T["any_fault_remaining"] = ("bin", b(lambda r: r["n_faults_remaining"] > 0))
    T["repaired_now"] = ("bin", b(lambda r: r["repaired_now"]))
    T["episode_success"] = ("bin", b(lambda r: r["final_reward"] > 0))
    T["n_faults_remaining"] = ("reg", b(lambda r: r["n_faults_remaining"]))
    T["filter_p_repaired"] = ("reg", b(lambda r: (r.get("filter") or {}).get("p_repaired")))
    faults = sorted({f for r in L for f in r["faults"]})
    for f in faults:  # "is this task fault still unfixed?" on rows of tasks that contain f
        T[f"present:{f}"] = ("bin", b(lambda r, f=f: r["fault_present"].get(f) if f in r["faults"] else None))
    return T


def filter_score(name, L):
    """The filter's own score for a target (higher = more likely positive), or None."""
    def p_rep(r):
        v = (r.get("filter") or {}).get("p_repaired")
        return np.nan if v is None else float(v)
    if name in ("repaired_now",):
        return np.asarray([p_rep(r) for r in L])
    if name == "any_fault_remaining":
        return np.asarray([1 - p_rep(r) for r in L])
    if name.startswith("present:"):
        f = name.split(":", 1)[1]
        out = []
        for r in L:
            tk = (r.get("filter") or {}).get("topk")
            if not tk:
                out.append(np.nan); continue
            tot = sum(p for _, p in tk)
            out.append(sum(p for fs, p in tk if f in fs) / tot if tot else np.nan)  # top-k marginal (approximate)
        return np.asarray(out)
    return None


def cv_scores(X, y, groups, kind, folds, C=0.05, alpha=100.0, text=False, seed=0):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(folds, len(set(groups))))
    for tr, te in gkf.split(np.zeros(len(y)), y, groups):
        if kind == "bin" and len(set(y[tr])) < 2:
            continue
        if text:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
            Xtr = vec.fit_transform([X[i] for i in tr]); Xte = vec.transform([X[i] for i in te])
            m = LogisticRegression(C=1.0, max_iter=2000) if kind == "bin" else Ridge(alpha=1.0)
        else:
            Xtr, Xte = X[tr], X[te]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000) if kind == "bin" else Ridge(alpha=alpha))
        m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1] if kind == "bin" else m.predict(Xte)
    return oof


def fold_ids(groups, folds):
    """The GroupKFold test-fold index of every row (same split cv_scores uses)."""
    from sklearn.model_selection import GroupKFold
    f = np.full(len(groups), -1)
    for k, (_, te) in enumerate(GroupKFold(n_splits=min(folds, len(set(groups)))).split(np.zeros(len(groups)), None, groups)):
        f[te] = k
    return f


def metric_fm(kind, y, s, fold):
    """Mean over folds of the within-fold metric. Pooled out-of-fold AUROC mixes per-fold intercepts (each fold's model sees a
    different base rate), which pushes even uninformative scores far below 0.5 under task-grouped folds; within-fold ranking does not."""
    vals = []
    for k in np.unique(fold):
        m = fold == k
        v = metric(kind, y[m], s[m])
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def metric(kind, y, s):
    ok = ~np.isnan(s) & ~np.isnan(y)
    if ok.sum() < 10:
        return None
    y, s = y[ok], s[ok]
    if kind == "bin":
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None
    from sklearn.metrics import r2_score
    return float(r2_score(y, s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--prompts-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,32,36")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only", default=None, help="comma list of target names (default all)")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)

    keys, H, ent, top1, ntok, turn = load_capture(a.capture_root, layers)
    lab = {key(r): r for r in read_jsonl(a.labels)}
    keep = [i for i, k in enumerate(keys) if k in lab]
    print(f"[probe] captured rows {len(keys)}, with labels {len(keep)}", flush=True)
    keys = [keys[i] for i in keep]; H = H[keep]; ent, top1, ntok, turn = ent[keep], top1[keep], ntok[keep], turn[keep]
    L = [lab[k] for k in keys]
    groups = np.asarray([k[1] for k in keys])
    text = load_text(a.prompts_root, set(keys))
    T = [text.get(k, "") for k in keys]
    pos = np.c_[turn, np.log1p(ntok)].astype(float)

    res, oof_out = {}, {"keys": np.asarray([json.dumps(k) for k in keys])}
    tg = targets(L)
    only = set(a.only.split(",")) if a.only else None
    rng = np.random.default_rng(0)
    for name, (kind, y) in tg.items():
        if only and name not in only:
            continue
        m = ~np.isnan(y)
        ym, gm = y[m], groups[m]
        info = {"kind": kind, "n": int(m.sum()), "n_tasks": int(len(set(gm)))}
        if kind == "bin":
            info["n_pos"] = int(ym.sum())
            if min(ym.sum(), (1 - ym).sum()) < MIN_CLASS or info["n_tasks"] < MIN_TASKS:
                info["skipped"] = "too few rows per class/tasks"; res[name] = info; print(f"[probe] {name}: {info}", flush=True); continue
        fd = fold_ids(gm, a.folds)
        M = lambda yy, ss: metric_fm(kind, yy, ss, fd)
        per, pooled = {}, {}
        best = (None, -np.inf, None)
        for li, Lyr in enumerate(layers):
            s = cv_scores(H[m][:, li].astype(np.float32), ym, gm, kind, a.folds)
            per[Lyr] = M(ym, s); pooled[Lyr] = metric(kind, ym, s)
            if per[Lyr] is not None and per[Lyr] > best[1]:
                best = (Lyr, per[Lyr], s)
        info["probe_by_layer"] = per
        info["probe_by_layer_pooled"] = pooled
        info["n_folds_scored"] = int(sum(len(set(ym[fd == k])) == 2 for k in np.unique(fd))) if kind == "bin" else int(len(np.unique(fd)))
        info["probe_best_layer"], info["probe_best"] = best[0], best[1]
        info["text"] = M(ym, cv_scores([T[i] for i in np.flatnonzero(m)], ym, gm, kind, a.folds, text=True))
        info["position"] = M(ym, cv_scores(pos[m], ym, gm, kind, a.folds, C=1.0, alpha=1.0))
        fs = filter_score(name, L)
        if fs is not None:
            info["filter"] = M(ym, fs[m]); info["filter_n"] = int((~np.isnan(fs[m])).sum())
        # selectivity control: permute labels across tasks (task-level shuffle keeps within-task structure)
        tasks = np.unique(gm); perm = dict(zip(tasks, rng.permutation(tasks)))
        first = {t: ym[gm == t] for t in tasks}
        ysh = np.empty_like(ym)
        for t in tasks:
            src = first[perm[t]]; idx = np.flatnonzero(gm == t)
            ysh[idx] = src[rng.integers(0, len(src), len(idx))]
        li = layers.index(best[0]) if best[0] is not None else len(layers) - 1
        info["shuffled"] = M(ysh, cv_scores(H[m][:, li].astype(np.float32), ysh, gm, kind, a.folds))
        if best[2] is not None:
            full = np.full(len(y), np.nan); full[m] = best[2]; oof_out[name] = full
        res[name] = info
        print(f"[probe] {name}: n={info['n']} best L{info['probe_best_layer']}={info['probe_best']:.3f} text={info['text']} "
              f"pos={info['position']} filter={info.get('filter')} shuf={info['shuffled']}", flush=True)
    oof_out.update(entropy=ent, top1_prob=top1, n_tokens=ntok, turn=turn)
    np.savez_compressed(a.out / "oof.npz", **oof_out)
    (a.out / "results.json").write_text(json.dumps({"layers": layers, "folds": a.folds, "n_rows": len(keys), "targets": res}, indent=1))
    f3 = lambda v: "" if v is None else f"{v:.3f}"
    lines = ["Metric: mean over task-held-out folds of the within-fold AUROC (bin) / R^2 (reg); folds = folds with both classes.", "",
             "| target | kind | n (pos) | tasks | probe best (layer) | text | position | filter | shuffled | folds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, i in res.items():
        if "skipped" in i:
            lines.append(f"| {name} | {i['kind']} | {i['n']} ({i.get('n_pos','')}) | {i['n_tasks']} | skipped | | | | | |"); continue
        lines.append(f"| {name} | {i['kind']} | {i['n']} ({i.get('n_pos','')}) | {i['n_tasks']} | {f3(i['probe_best'])} (L{i['probe_best_layer']}) | "
                     f"{f3(i['text'])} | {f3(i['position'])} | {f3(i.get('filter'))} | {f3(i['shuffled'])} | {i['n_folds_scored']} |")
    (a.out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
