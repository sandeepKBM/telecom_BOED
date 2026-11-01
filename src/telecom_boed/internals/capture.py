"""[telecom_boed] H7 stage 2 (GPU): last-prompt-token hidden states of Qwen3-8B at every agent decision point.

Layout of one stored row (one decision point): h[37, 4096] fp16 = HF `output_hidden_states` convention, i.e.
  h[0]      embedding output,
  h[1..35]  decoder layer 1..35 outputs (pre final norm),
  h[36]     decoder layer 36 output AFTER the final RMSNorm (HF's last hidden state),
taken at the LAST prompt token (the final token of the generation prompt `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`),
plus next-token entropy (nats, fp32 from the lm_head logits) and top-1 probability / id at that position.

Tokenization: apply_chat_template(messages, tools=tools, add_generation_prompt=True, enable_thinking=False). Its length is
compared to the vLLM-recorded usage.prompt_tokens (`match_flag`).

Efficiency: Qwen3's template re-renders earlier assistant turns differently once a new user query arrives (the
<think></think> block is dropped for turns before the last query), so token prefixes are stable only within a
"tool run". `plan_anchors` finds, per episode, the minimal set of prompts such that every turn's tokens are a strict
prefix of some anchor's tokens; one causal forward per anchor serves all of its turns (causality makes the prefix
states identical up to numerics). Only gathered positions are kept (forward hooks), so no [L, T, 4096] tensors and no
[T, vocab] logits are materialised.

No torch/transformers/tau2 import at module import time.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
N_STATES = 37
HIDDEN = 4096
INDEX_FIELDS = ("run", "task_id", "trial", "turn", "msg_idx", "n_tokens", "match_flag", "shard", "row")
EXTRA_INDEX_FIELDS = ("recorded_prompt_tokens", "anchor", "entropy", "top1_prob", "top1_id")
SHARD_TENSORS = ("h", "entropy", "top1_prob", "top1_id", "n_tokens")


# ---------------------------------------------------------------- pure helpers (CPU-testable) ---

def hf_messages(messages: Sequence[dict]) -> list[dict]:
    """OpenAI-format messages (arguments as JSON strings) -> HF chat-template format.

    Arguments become dicts (vLLM does the same before templating); None content on an assistant tool call becomes "".
    """
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                new = []
                for tc in tcs:
                    f = dict(tc["function"])
                    args = f.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    f["arguments"] = args
                    new.append({"id": tc.get("id"), "type": "function", "function": f})
                m["tool_calls"] = new
            else:
                m.pop("tool_calls", None)
            if m.get("content") is None:
                m["content"] = ""
        out.append(m)
    return out


def group_episodes(rows: Sequence[dict]) -> dict[tuple, list[dict]]:
    eps: dict[tuple, list[dict]] = {}
    for r in rows:
        eps.setdefault((r["task_id"], r["trial"]), []).append(r)
    for v in eps.values():
        v.sort(key=lambda r: r["msg_idx"])
    return eps


def plan_anchors(tokens: Sequence[Sequence[int]]) -> list[tuple[int, list[int]]]:
    """Cover every sequence by the minimal-ish set of anchors that have it as a prefix.

    Returns [(anchor_index, [member indices incl. the anchor itself])]. Greedy, longest first: a sequence joins the first existing anchor whose tokens start with it.
    """
    arrs = [np.asarray(t, dtype=np.int64) for t in tokens]
    anchors: list[int] = []
    members: dict[int, list[int]] = {}
    for i in sorted(range(len(arrs)), key=lambda k: (-len(arrs[k]), -k)):  # longest first
        n = len(arrs[i])
        for a in anchors:
            if len(arrs[a]) >= n and np.array_equal(arrs[a][:n], arrs[i]):
                members[a].append(i)
                break
        else:
            anchors.append(i)
            members[i] = [i]
    return [(a, sorted(members[a])) for a in sorted(anchors)]


def index_record(row: dict, n_tokens: int, shard: str, shard_row: int, anchor: int, stats: dict) -> dict:
    rec = {
        "run": row["run"], "task_id": row["task_id"], "trial": row["trial"], "turn": row["turn"], "msg_idx": row["msg_idx"],
        "n_tokens": int(n_tokens),
        "match_flag": bool(row.get("recorded_prompt_tokens") == n_tokens),
        "shard": shard, "row": int(shard_row),
        "recorded_prompt_tokens": row.get("recorded_prompt_tokens"), "anchor": int(anchor),
    }
    rec.update({k: float(stats[k]) if k != "top1_id" else int(stats[k]) for k in ("entropy", "top1_prob", "top1_id")})
    return rec


def shard_name(task_id: str, trial: int) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(task_id))
    return f"shards/{safe}__t{int(trial)}.safetensors"


# ---------------------------------------------------------------- GPU side ---

class Capturer:
    """Loads Qwen3-8B (bf16, offline) and extracts last-token states at requested positions."""

    def __init__(self, model: str = MODEL, revision: str = REVISION, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16).to(device).eval()
        self.base = self.model.model
        self.layers = self.base.layers
        assert len(self.layers) == N_STATES - 1, len(self.layers)
        self.revision = revision

    def _render(self, row: dict, gen: bool) -> list[int]:
        text = self.tok.apply_chat_template(
            hf_messages(row["messages"]), tools=row["tools"], add_generation_prompt=gen, enable_thinking=False, tokenize=False
        )
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def tokenize(self, row: dict) -> list[int]:
        """Full prompt ids (what vLLM counts as prompt_tokens)."""
        return self._render(row, True)

    def tokenize_split(self, row: dict) -> tuple[list[int], Optional[list[int]], list[int]]:
        """(full ids, prefix ids, suffix ids): prefix = template without generation prompt, suffix = the generation prompt
        (`<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`). Later turns re-render earlier assistant turns WITHOUT the
        think block, so the prefix (not the full prompt) is what is shared between turns. prefix is None if
        full != prefix + suffix cleanly (such rows get their own plain forward)."""
        full = self._render(row, True)
        pre = self._render(row, False)
        if len(pre) < len(full) and full[: len(pre)] == pre:
            return full, pre, full[len(pre):]
        return full, None, []

    @staticmethod
    def _first(o: Any):
        return o[0] if isinstance(o, (tuple, list)) else o

    def _hooks(self, buf, pos):
        def mk(k):
            def hook(_m, _i, out):
                buf[:, k] = self._first(out)[0, pos]
            return hook

        hs = [self.base.embed_tokens.register_forward_hook(mk(0))]
        for k in range(N_STATES - 2):  # layers 1..35 -> slots 1..35; layer 36 comes post-norm from last_hidden_state
            hs.append(self.layers[k].register_forward_hook(mk(k + 1)))
        return hs

    def _readout(self, ids_len_pos, buf, last):
        torch = self.torch
        buf[:, N_STATES - 1] = last
        logits = self.model.lm_head(last).float()
        lp = torch.log_softmax(logits, dim=-1)
        p = lp.exp()
        ent = -(p * lp).sum(-1)
        top1_p, top1_i = p.max(-1)
        amax = float(buf.float().abs().max())
        h16 = buf.float().clamp(-65504, 65504).half()
        return {"h": h16.cpu().numpy(), "entropy": ent.cpu().numpy(), "top1_prob": top1_p.cpu().numpy(),
                "top1_id": top1_i.cpu().numpy(), "amax": amax}

    def states_at(self, ids: Sequence[int], positions: Sequence[int]) -> dict:
        """One plain causal forward over `ids`; states at `positions` (0-based token indices).

        Returns h [P, 37, H] fp16 (HF hidden_states layout, see module docstring), entropy [P], top1_prob [P], top1_id [P].
        """
        torch = self.torch
        pos = torch.as_tensor(list(positions), device=self.device, dtype=torch.long)
        buf = torch.empty(len(pos), N_STATES, HIDDEN, dtype=torch.bfloat16, device=self.device)
        handles = self._hooks(buf, pos)
        try:
            with torch.inference_mode():
                x = torch.as_tensor([list(ids)], device=self.device, dtype=torch.long)
                last = self.base(input_ids=x, use_cache=False).last_hidden_state[0, pos]  # post final norm
                return self._readout(None, buf, last)
        finally:
            for h in handles:
                h.remove()

    def prefill(self, prefix_ids: Sequence[int]):
        """Causal forward over the shared prefix, keeping the KV cache (no state capture)."""
        from transformers import DynamicCache

        torch = self.torch
        cache = DynamicCache()
        with torch.inference_mode():
            x = torch.as_tensor([list(prefix_ids)], device=self.device, dtype=torch.long)
            self.base(input_ids=x, past_key_values=cache, use_cache=True)
        return cache

    def suffix_states(self, cache, n_prefix: int, suffix_ids: Sequence[int]) -> dict:
        """State at the last suffix token given the first `n_prefix` cached positions (cache itself is not modified).

        Equivalent to the last-token state of a full causal pass over prefix+suffix (what vLLM prefix-caches too);
        a fresh DynamicCache is filled with [:n_prefix] views so several suffixes can branch off one prefill.
        """
        from transformers import DynamicCache

        torch = self.torch
        branch = DynamicCache()
        for li, layer in enumerate(cache.layers):
            branch.update(layer.keys[:, :, :n_prefix], layer.values[:, :, :n_prefix], li)
        pos = torch.as_tensor([len(suffix_ids) - 1], device=self.device, dtype=torch.long)
        buf = torch.empty(1, N_STATES, HIDDEN, dtype=torch.bfloat16, device=self.device)
        handles = self._hooks(buf, pos)
        try:
            with torch.inference_mode():
                x = torch.as_tensor([list(suffix_ids)], device=self.device, dtype=torch.long)
                last = self.base(input_ids=x, past_key_values=branch, use_cache=True).last_hidden_state[0, pos]
                return self._readout(None, buf, last)
        finally:
            for h in handles:
                h.remove()
            del branch

    def capture_episode(self, rows: Sequence[dict], mode: str = "prefix") -> tuple[dict, dict]:
        """All decision points of one episode. mode "prefix": share one prefill per anchor (+ per-turn suffix pass);
        mode "full": one plain forward per turn (reference). Returns (arrays aligned with rows, info)."""
        splits = [self.tokenize_split(r) for r in rows]
        fulls = [s[0] for s in splits]
        n = len(rows)
        H = np.zeros((n, N_STATES, HIDDEN), np.float16)
        ent = np.zeros(n, np.float32); t1p = np.zeros(n, np.float32); t1i = np.zeros(n, np.int64)
        anchor_of = np.zeros(n, np.int64)
        amax = 0.0
        fwd_tokens = 0
        splittable = [i for i in range(n) if splits[i][1] is not None] if mode == "prefix" else []
        plain = [i for i in range(n) if i not in set(splittable)]
        plan = []
        if splittable:
            sub = plan_anchors([splits[i][1] for i in splittable])
            plan = [(splittable[a], [splittable[m] for m in mem]) for a, mem in sub]
        for a, mem in plan:
            pa = splits[a][1]
            cache = self.prefill(pa)
            fwd_tokens += len(pa)
            for i in mem:
                out = self.suffix_states(cache, len(splits[i][1]), splits[i][2])
                fwd_tokens += len(splits[i][2])
                H[i], ent[i], t1p[i], t1i[i] = out["h"][0], out["entropy"][0], out["top1_prob"][0], out["top1_id"][0]
                anchor_of[i] = a
                amax = max(amax, out["amax"])
            del cache
            self.torch.cuda.empty_cache()
        for i in plain:
            out = self.states_at(fulls[i], [len(fulls[i]) - 1])
            fwd_tokens += len(fulls[i])
            H[i], ent[i], t1p[i], t1i[i] = out["h"][0], out["entropy"][0], out["top1_prob"][0], out["top1_id"][0]
            anchor_of[i] = i
            amax = max(amax, out["amax"])
            plan.append((i, [i]))
        arrays = {"h": H, "entropy": ent, "top1_prob": t1p, "top1_id": t1i,
                  "n_tokens": np.asarray([len(t) for t in fulls], np.int64)}
        return arrays, {"toks": fulls, "plan": plan, "anchor_of": anchor_of, "amax": amax, "fwd_tokens": fwd_tokens}


def save_shard(path: Path, arrays: dict) -> None:
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file({k: np.ascontiguousarray(arrays[k]) for k in SHARD_TENSORS}, str(tmp))
    os.replace(tmp, path)


def capture_run(prompts_path: str, out_dir: str, *, mode: str = "prefix", limit_eps: Optional[int] = None, verify_alone: int = 0,
                cap: Optional[Capturer] = None, log=print) -> dict:
    """Capture every episode in prompts.jsonl into out_dir (shards/ + index.jsonl). Resumable per episode."""
    from .prompts import read_jsonl

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.jsonl"
    done = set()
    if index_path.exists():
        for r in read_jsonl(str(index_path)):
            done.add((r["task_id"], r["trial"]))
    eps = group_episodes(list(read_jsonl(prompts_path)))
    cap = cap or Capturer()
    stats = {"turns": 0, "match": 0, "episodes": 0, "anchors": 0, "forward_tokens": 0, "seconds": 0.0, "amax": 0.0,
             "skipped_eps": 0, "mismatch": [], "verify": []}
    todo = [(k, v) for k, v in sorted(eps.items()) if k not in done]
    stats["skipped_eps"] = len(eps) - len(todo)
    if limit_eps:
        todo = todo[:limit_eps]
    for (task_id, trial), rows in todo:
        t0 = time.perf_counter()
        arrays, info = cap.capture_episode(rows, mode=mode)
        dt = time.perf_counter() - t0
        shard = shard_name(task_id, trial)
        save_shard(out / shard, arrays)
        recs = [index_record(r, arrays["n_tokens"][i], shard, i, info["anchor_of"][i],
                             {k: arrays[k][i] for k in ("entropy", "top1_prob", "top1_id")}) for i, r in enumerate(rows)]
        with open(index_path, "a") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        stats["turns"] += len(rows); stats["episodes"] += 1; stats["anchors"] += len(info["plan"])
        stats["forward_tokens"] += info["fwd_tokens"]; stats["seconds"] += dt
        stats["amax"] = max(stats["amax"], info["amax"])
        nm = sum(r["match_flag"] for r in recs); stats["match"] += nm
        stats["mismatch"] += [(r["task_id"], r["trial"], r["msg_idx"], r["n_tokens"], r["recorded_prompt_tokens"]) for r in recs if not r["match_flag"]]
        if verify_alone:
            stats["verify"] += verify_alone_vs_gathered(cap, rows, info, arrays, k=verify_alone)
        log(f"[h7] {task_id} t{trial}: {len(rows)} turns, {len(info['plan'])} fwd, match {nm}/{len(rows)}, {dt:.1f}s")
    return stats


def verify_alone_vs_gathered(cap: Capturer, rows, info, arrays, k: int = 2) -> list[dict]:
    """Recompute k turns that were served by a *different* (longer) anchor with their own forward; compare."""
    res = []
    cand = [i for a, mem in info["plan"] for i in mem if i != a] or [a for a, _ in info["plan"]]
    for i in cand[:k]:
        alone = cap.states_at(info["toks"][i], [len(info["toks"][i]) - 1])
        g = arrays["h"][i].astype(np.float32); s = alone["h"][0].astype(np.float32)
        d = np.abs(g - s)
        res.append({"msg_idx": rows[i]["msg_idx"], "n_tokens": len(info["toks"][i]), "anchor_tokens": len(info["toks"][info["anchor_of"][i]]),
                    "max_abs_last_layer": float(d[-1].max()), "max_abs_all_layers": float(d.max()),
                    "rel_last": float(np.linalg.norm(g[-1] - s[-1]) / max(np.linalg.norm(s[-1]), 1e-9)),
                    "last_layer_absmax": float(np.abs(s[-1]).max()),
                    "entropy_diff": float(abs(arrays["entropy"][i] - alone["entropy"][0]))})
    return res
