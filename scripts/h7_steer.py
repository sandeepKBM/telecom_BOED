#!/usr/bin/env python
"""[telecom_boed] H7-B stage 1 (GPU): activation steering of the Qwen3-8B telecom agent at its decision token.

  python -I scripts/h7_steer.py --prep PREP_DIR --prompts-root PR --out OUT [--limit N] [--gen-n 48] [--max-minutes 80]

For each held-out turn in PREP_DIR/eval_turns.jsonl (scripts/h7_steer_prep.py) the logged prompt is rebuilt exactly as
the capture did (capture.Capturer.tokenize, enable_thinking=False), prompt[:-1] is prefilled once, and the last prompt
token is re-run with  h_L <- h_L + alpha * |v| * unit(v)  added to the output of decoder layer L (model.layers[L-1], the
same state the probes read). Steering touches only the decision token (and, in generation, every generated token);
the prompt's KV cache is never steered.

B1 grid (one token, cheap): for every layer, dirs dom_tool and rand0; at --main-layer also probe_tool, rand1, rand2,
  dom_device, dom_repaired; alpha in ALPHAS (units of the direction's own norm; rand*/probe_tool have |dom_tool|).
  Per condition: logit of P(first token = <tool_call>) vs everything else, entropy, top-1 id, KL(base || steered).
B2 generation (first --gen-n turns, greedy, <= --max-new tokens; a tool call cut off by the limit gets json_ok=None): base + GEN_CONDS at --main-layer. Per output: does it
  start with <tool_call>, is the JSON valid, is the name one of the prompt's tools, are the argument keys in the schema.
Outputs OUT/b1.jsonl and OUT/b2.jsonl (one line per turn, resumable: done turns are skipped), OUT/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from telecom_boed.internals import capture as C  # noqa: E402

ALPHAS = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]
MAIN_ONLY = ["probe_tool", "rand1", "rand2", "dom_device", "dom_repaired", "dom_top1"]
GEN_CONDS = [("dom_tool", -2.0), ("dom_tool", -1.0), ("dom_tool", 1.0), ("dom_tool", 2.0), ("rand0", -2.0), ("rand0", 2.0),
             ("dom_device", 2.0), ("dom_repaired", 2.0)]
TOOL_CALL, TOOL_CALL_END, IM_END = "<tool_call>", "</tool_call>", "<|im_end|>"


def tkey(r):
    return (r["run"], str(r["task_id"]), int(r["trial"]), int(r["msg_idx"]))


def read_jsonl(p):
    out = []
    with open(p) as f:
        for l in f:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:  # partial last line from a killed job: that turn is redone
                pass
    return out


class Steerer:
    def __init__(self, cap: C.Capturer, vec: dict, layers: list[int]):
        self.cap, self.torch = cap, cap.torch
        t = self.torch
        self.layers = layers
        self.unit = {}   # (name, layer) -> unit vector (bf16, device); scale = |v| (fp32 python float)
        for n, arr in vec.items():
            for j, L in enumerate(layers):
                v = t.as_tensor(arr[j], dtype=t.float32)
                nv = float(v.norm())
                self.unit[(n, L)] = ((v / max(nv, 1e-12)).to(cap.device, t.bfloat16), nv)
        self.W = cap.model.lm_head.weight.float()  # fp32 readout: bf16 logits near 20-30 step by 0.125-0.25
        self.tc_id = cap.tok.convert_tokens_to_ids(TOOL_CALL)
        self.stop_ids = {cap.tok.convert_tokens_to_ids(IM_END), cap.tok.convert_tokens_to_ids(TOOL_CALL_END), cap.tok.eos_token_id}

    def _hook(self, add):
        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + add,) + tuple(out[1:])
            return out + add
        return hook

    def _branch(self, cache, n):
        from transformers import DynamicCache
        b = DynamicCache()
        for li, layer in enumerate(cache.layers):
            b.update(layer.keys[:, :, :n], layer.values[:, :, :n], li)
        return b

    def _steer(self, name, L, alpha):
        if name is None or alpha == 0:
            return None
        u, nv = self.unit[(name, L)]
        return self.cap.layers[L - 1].register_forward_hook(self._hook((alpha * nv) * u))

    def last_token(self, cache, n_prefix, last_id, name=None, L=None, alpha=0.0):
        t = self.torch
        h = self._steer(name, L, alpha)
        try:
            with t.inference_mode():
                out = self.cap.base(input_ids=t.as_tensor([[last_id]], device=self.cap.device),
                                    past_key_values=self._branch(cache, n_prefix), use_cache=True)
                return (out.last_hidden_state[0, -1].float() @ self.W.T).log_softmax(-1)
        finally:
            if h is not None:
                h.remove()

    def read(self, lp, base_lp=None):
        t = self.torch
        other = lp.clone(); other[self.tc_id] = -float("inf")
        p = lp.exp()
        rec = {"logit_tc": float(lp[self.tc_id] - t.logsumexp(other, -1)), "entropy": float(-(p * lp).sum()),
               "top1": int(lp.argmax())}
        if base_lp is not None:
            rec["kl"] = float((base_lp.exp() * (base_lp - lp)).sum())
        return rec

    def generate(self, cache, n_prefix, last_id, max_new, name=None, L=None, alpha=0.0):
        t = self.torch
        h = self._steer(name, L, alpha)
        branch = self._branch(cache, n_prefix)
        ids, cur, stop = [], last_id, "length"
        try:
            with t.inference_mode():
                for _ in range(max_new):
                    out = self.cap.model(input_ids=t.as_tensor([[cur]], device=self.cap.device), past_key_values=branch, use_cache=True)
                    cur = int(out.logits[0, -1].argmax())
                    ids.append(cur)
                    if cur in self.stop_ids:
                        stop = self.cap.tok.convert_ids_to_tokens(cur)
                        break
        finally:
            if h is not None:
                h.remove()
            del branch
        return ids, stop


def judge(text: str, tools: list[dict], stop: str = "") -> dict:
    schema = {t["function"]["name"]: set((t["function"].get("parameters") or {}).get("properties", {}).keys()) for t in tools}
    r = {"starts_tc": text.lstrip().startswith(TOOL_CALL), "json_ok": None, "name_ok": None, "args_ok": None, "name": None}
    if r["starts_tc"]:
        body = text.lstrip()[len(TOOL_CALL):].split(TOOL_CALL_END)[0].strip()
        try:
            obj = json.loads(body)
            r["json_ok"] = isinstance(obj, dict)
        except json.JSONDecodeError:
            r["json_ok"] = None if stop == "length" else False  # truncated by --max-new is not invalid JSON
            obj = None
        if r["json_ok"]:
            r["name"] = obj.get("name")
            r["name_ok"] = r["name"] in schema
            args = obj.get("arguments", {})
            r["args_ok"] = bool(r["name_ok"]) and isinstance(args, dict) and set(args) <= schema.get(r["name"], set())
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--prompts-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--main-layer", type=int, default=28)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gen-n", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--shard", default="0/1", help="I/N: this process takes eval turns with index %% N == I (separate --out per shard)")
    ap.add_argument("--max-minutes", type=float, default=75.0, help="stop cleanly (resumable) after this much wall time")
    a = ap.parse_args()
    t_start = time.perf_counter()
    prep, out = Path(a.prep), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(prep / "directions.npz")
    layers = [int(x) for x in d["layers"]]
    names = [str(x) for x in d["names"]]
    vec = {n: d[f"vec_{n}"] for n in names}
    assert a.main_layer in layers
    si, sn = (int(x) for x in a.shard.split("/"))
    turns = read_jsonl(prep / "eval_turns.jsonl")[: a.limit]
    gen_keys = {tkey(r) for r in turns[: a.gen_n]}  # generation subset is fixed by global order, not by shard
    turns = [r for i, r in enumerate(turns) if i % sn == si]
    want = {tkey(r) for r in turns}
    rows = {}
    for run in sorted({r["run"] for r in turns}):
        for r in read_jsonl(Path(a.prompts_root) / run / "prompts.jsonl"):
            if tkey(r) in want:
                rows[tkey(r)] = r
    missing = [k for k in want if k not in rows]
    assert not missing, f"{len(missing)} eval turns not found in prompts, e.g. {missing[:2]}"
    done1 = {tkey(r) for r in read_jsonl(out / "b1.jsonl")} if (out / "b1.jsonl").exists() else set()
    done2 = {tkey(r) for r in read_jsonl(out / "b2.jsonl")} if (out / "b2.jsonl").exists() else set()

    t0 = time.perf_counter()
    cap = C.Capturer()
    st = Steerer(cap, vec, layers)
    t_load = time.perf_counter() - t0
    print(f"[steer] model loaded {t_load:.0f}s; turns {len(turns)} (done b1 {len(done1)}, b2 {len(done2)}); tc_id {st.tc_id}", flush=True)
    grid = [(n, L) for L in layers for n in (["dom_tool", "rand0"] + (MAIN_ONLY if L == a.main_layer else []))]
    n1 = n2 = 0
    match_top1 = []
    stopped = False
    for i, tr in enumerate(turns):
        k = tkey(tr)
        need1, need2 = k not in done1, (k in gen_keys and k not in done2)
        if not (need1 or need2):
            continue
        if (time.perf_counter() - t_start) / 60 > a.max_minutes:
            stopped = True
            print(f"[steer] max-minutes reached at turn {i}; rerun to resume", flush=True)
            break
        row = rows[k]
        ids = cap.tokenize(row)
        assert len(ids) == tr["n_tokens"], (k, len(ids), tr["n_tokens"])
        tt = time.perf_counter()
        cache = cap.prefill(ids[:-1])
        n_pre = len(ids) - 1
        base_lp = st.last_token(cache, n_pre, ids[-1])
        base = st.read(base_lp)
        match_top1.append(base["top1"] == tr["cap_top1_id"])
        if need1:
            res = []
            for n, L in grid:
                for al in ALPHAS:
                    r = st.read(st.last_token(cache, n_pre, ids[-1], n, L, al), base_lp)
                    res.append([L, n, al, round(r["logit_tc"], 4), round(r["entropy"], 4), r["top1"], round(r["kl"], 5)])
            with open(out / "b1.jsonl", "a") as f:
                f.write(json.dumps({**{kk: tr[kk] for kk in ("run", "task_id", "trial", "msg_idx", "stratum", "is_tool_call")},
                                    "base": base, "cols": ["layer", "dir", "alpha", "logit_tc", "entropy", "top1", "kl"], "grid": res,
                                    "sec": round(time.perf_counter() - tt, 2)}) + "\n")
            n1 += 1
        if need2:
            gens = []
            for n, al in [(None, 0.0)] + GEN_CONDS:
                g_ids, stop = st.generate(cache, n_pre, ids[-1], a.max_new, n, a.main_layer, al)
                text = cap.tok.decode(g_ids, skip_special_tokens=False)
                gens.append({"dir": n, "alpha": al, "n_new": len(g_ids), "stop": stop, "text": text, **judge(text, row["tools"], stop)})
            with open(out / "b2.jsonl", "a") as f:
                f.write(json.dumps({**{kk: tr[kk] for kk in ("run", "task_id", "trial", "msg_idx", "stratum", "is_tool_call")},
                                    "layer": a.main_layer, "gens": gens}) + "\n")
            n2 += 1
        del cache
        cap.torch.cuda.empty_cache()
        if (n1 + n2) and i % 10 == 0:
            el = time.perf_counter() - t_start
            print(f"[steer] turn {i + 1}/{len(turns)} n_tok {len(ids)} b1 {n1} b2 {n2} elapsed {el / 60:.1f} min "
                  f"base_top1_match {np.mean(match_top1):.3f}", flush=True)
    summ = {"shard": a.shard, "turns": len(turns), "b1_new": n1, "b2_new": n2, "stopped_early": stopped, "load_s": round(t_load, 1),
            "wall_min": round((time.perf_counter() - t_start) / 60, 2), "base_top1_match_capture": float(np.mean(match_top1)) if match_top1 else None,
            "layers": layers, "main_layer": a.main_layer, "alphas": ALPHAS, "gen_conds": GEN_CONDS, "max_new": a.max_new}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("STEER_DONE " + json.dumps(summ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
