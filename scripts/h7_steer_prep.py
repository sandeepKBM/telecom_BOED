#!/usr/bin/env python
"""[telecom_boed] H7-B stage 0 (CPU): steering directions + held-out evaluation turns for scripts/h7_steer.py.

  python -I scripts/h7_steer_prep.py --capture-root CAP --labels LABELS.jsonl --out OUT [--layers 12,16,20,24,28,32] [--n-per-stratum 100]

Tasks are split in two by md5(task_id) (the same task recurs across runs, so the split is by task). Directions come from
split A only; evaluation turns come from split B only, so a direction never saw the turns it is tested on.

Directions per layer L (L = HF hidden_states index: output of decoder layer L, i.e. model.layers[L-1]):
  dom_tool      mean(h | next msg is a tool call) - mean(h | text message)                 (Marks & Tegmark diff-of-means)
  probe_tool    standardised logistic-probe weights mapped back to raw space, rescaled to |dom_tool|
  rand0..2      Gaussian directions rescaled to |dom_tool| (magnitude-matched random control)
  dom_device    on text turns only: mean(h | asks the user for a device action) - mean(h | other text)
  dom_repaired  mean(h | all faults fixed at this turn) - mean(h | not)   (confounded with turn position; see report)
  dom_top1      mean(h | the model's own top-1 next token is <tool_call>) - mean(h | not). Differs from dom_tool because the
                repair arms (repair_invalid_user_tools) log an agent call to a phone tool as a text request to the user.
Writes OUT/directions.npz (layers, names, vec[name] [n_layers, 4096] fp32, layer stats), OUT/eval_turns.jsonl
(strata = is_tool_call x (top1_prob < 0.99), n-per-stratum each; ordered so the first 4k rows are balanced) and OUT/prep.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h7_probe_eval import key, load_capture, read_jsonl  # noqa: E402

DUP_RUNS = {"sdB_526576.30"}  # identical transcripts to sdA (same forward tokens); kept out of the evaluation sample


def split_of(task_id: str) -> str:
    return "A" if int(hashlib.md5(task_id.encode()).hexdigest(), 16) % 2 == 0 else "B"


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="12,16,20,24,28,32")
    ap.add_argument("--n-per-stratum", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    keys, H, ent, top1, ntok, turn = load_capture(Path(a.capture_root), layers)
    idx_rows = {}
    for run_dir in Path(a.capture_root).iterdir():
        if (run_dir / "index.jsonl").exists():
            for r in read_jsonl(run_dir / "index.jsonl"):
                idx_rows[key(r)] = r
    lab = {key(r): r for r in read_jsonl(a.labels)}
    keep = [i for i, k in enumerate(keys) if k in lab and idx_rows[k]["match_flag"]]
    keys = [keys[i] for i in keep]; H = H[keep].astype(np.float32); ent, top1, ntok, turn = ent[keep], top1[keep], ntok[keep], turn[keep]
    L = [lab[k] for k in keys]
    split = np.array([split_of(k[1]) for k in keys])
    tc = np.array([bool(r["is_tool_call"]) for r in L])
    dev = np.array([bool(r["is_device_request"]) for r in L])
    rep = np.array([bool(r["repaired_now"]) for r in L])
    t1 = np.array([idx_rows[k]["top1_id"] == 151657 for k in keys])  # <tool_call>
    A = split == "A"
    print(f"rows {len(keys)}  split A {A.sum()} ({len(set(k[1] for k, s in zip(keys, split) if s == 'A'))} tasks)  "
          f"B {(~A).sum()} ({len(set(k[1] for k, s in zip(keys, split) if s == 'B'))} tasks)", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(a.seed)
    rand = [unit(rng.standard_normal(H.shape[-1]).astype(np.float32)) for _ in range(3)]
    names = ["dom_tool", "probe_tool", "rand0", "rand1", "rand2", "dom_device", "dom_repaired", "dom_top1"]
    vec = {n: np.zeros((len(layers), H.shape[-1]), np.float32) for n in names}
    stats = []
    for j, Lyr in enumerate(layers):
        X = H[:, j]
        d_tool = X[A & tc].mean(0) - X[A & ~tc].mean(0)
        n_tool = float(np.linalg.norm(d_tool))
        sc = StandardScaler().fit(X[A])
        lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(X[A]), tc[A])
        w = lr.coef_[0] / sc.scale_
        vec["dom_tool"][j] = d_tool
        vec["probe_tool"][j] = unit(w) * n_tool
        for r_i in range(3):
            vec[f"rand{r_i}"][j] = rand[r_i] * n_tool
        vec["dom_device"][j] = X[A & ~tc & dev].mean(0) - X[A & ~tc & ~dev].mean(0)
        vec["dom_repaired"][j] = X[A & rep].mean(0) - X[A & ~rep].mean(0)
        vec["dom_top1"][j] = X[A & t1].mean(0) - X[A & ~t1].mean(0)
        u = unit(d_tool)
        proj = X @ u
        st = {"layer": Lyr, "mean_h_norm": float(np.linalg.norm(X[A], axis=1).mean()), "dom_tool_norm": n_tool,
              "dom_device_norm": float(np.linalg.norm(vec["dom_device"][j])), "dom_repaired_norm": float(np.linalg.norm(vec["dom_repaired"][j])),
              "proj_std_within_tool": float(proj[A & tc].std()), "proj_std_within_text": float(proj[A & ~tc].std()),
              "cos_probe_dom": float(unit(w) @ u),
              "cos_device_tool": float(unit(vec["dom_device"][j]) @ u), "cos_repaired_tool": float(unit(vec["dom_repaired"][j]) @ u),
              "dom_top1_norm": float(np.linalg.norm(vec["dom_top1"][j])), "cos_top1_tool": float(unit(vec["dom_top1"][j]) @ u),
              "heldout_acc_dom_threshold": float(np.mean((proj[~A] > 0.5 * (proj[A & tc].mean() + proj[A & ~tc].mean())) == tc[~A])),
              "heldout_acc_probe": float(lr.score(sc.transform(X[~A]), tc[~A]))}
        stats.append(st)
        print(json.dumps(st), flush=True)
    np.savez(out / "directions.npz", layers=np.asarray(layers), names=np.asarray(names), **{f"vec_{n}": v for n, v in vec.items()})

    # held-out evaluation turns (split B), strata = is_tool_call x uncertain(top1 < 0.99)
    cand = [i for i in range(len(keys)) if split[i] == "B" and keys[i][0] not in DUP_RUNS]
    strata = {}
    for i in cand:
        strata.setdefault((int(tc[i]), int(top1[i] < 0.99)), []).append(i)
    picked = {s: list(rng.permutation(v)[: a.n_per_stratum]) for s, v in sorted(strata.items())}
    order = []
    for t in range(max(len(v) for v in picked.values())):  # round-robin so any prefix is balanced across strata
        for s in sorted(picked):
            if t < len(picked[s]):
                order.append((s, picked[s][t]))
    with open(out / "eval_turns.jsonl", "w") as f:
        for s, i in order:
            r = L[i]
            f.write(json.dumps({"run": keys[i][0], "task_id": keys[i][1], "trial": keys[i][2], "msg_idx": keys[i][3],
                                "stratum": f"tc{s[0]}_unc{s[1]}", "is_tool_call": bool(tc[i]), "is_device_request": bool(dev[i]),
                                "repaired_now": bool(rep[i]), "turn": int(turn[i]), "n_tokens": int(ntok[i]),
                                "cap_entropy": float(ent[i]), "cap_top1_prob": float(top1[i]), "cap_top1_id": int(idx_rows[keys[i]]["top1_id"]),
                                "filter": r.get("filter")}) + "\n")
    info = {"layers": layers, "names": names, "n_rows": len(keys), "n_A": int(A.sum()), "n_B": int((~A).sum()),
            "strata_available": {f"tc{s[0]}_unc{s[1]}": len(v) for s, v in sorted(strata.items())},
            "n_eval": len(order), "layer_stats": stats, "seed": a.seed}
    (out / "prep.json").write_text(json.dumps(info, indent=1))
    print("PREP_DONE " + json.dumps({k: info[k] for k in ("n_rows", "n_A", "n_B", "strata_available", "n_eval")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
