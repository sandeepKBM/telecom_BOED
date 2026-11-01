"""[telecom_boed] P8 systems-study metrics, computed from run artifacts (stdlib only, loadable by scripts under python -I).

Inputs (all optional; a metric whose input is missing is None, never a guess):
  events      list of dicts from <run>/events.jsonl (scripts/run_events.py); optional fields policy_version, gen_start,
              gen_end, enqueue, dequeue (epoch seconds) are written by the async rollout/learner once it exists.
  prom        dict from profiling.parse_prom(vllm_metrics.prom)
  sims        tau2 results.json "simulations"
  gpu_samples rows parsed from scripts/gpu_sampler.sh output (gpu_samples.csv)

Charge rule (NERSC Perlmutter GPU, verified in the perlmutter-ddl skill, 2026-10-16): full-node QOS (regular, debug,
preempt, ...) charge 4 GPU-h per node-hour regardless of GPUs actually used; `shared` charges per GPU requested
(1 GPU-h per GPU-hour). Local ilab/rlab Slurm has no charge rule: GPU-h = AllocTRES gres/gpu count x elapsed.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

FULL_NODE_QOS_GPU_PER_NODE = 4
SHARED_QOS = ("shared", "gpu_shared")


def parse_alloc_tres(s: str | None) -> dict[str, str]:
    """'billing=4,cpu=128,gres/gpu=4,node=1' -> {'billing': '4', ...}."""
    out: dict[str, str] = {}
    for part in (s or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def gpu_count(alloc_tres: str | None) -> int:
    """gres/gpu count from sacct AllocTRES ('gres/gpu:a100=2' style typed keys are summed too); 0 if none."""
    n = 0
    for k, v in parse_alloc_tres(alloc_tres).items():
        if k == "gres/gpu" or k.startswith("gres/gpu:"):
            if k == "gres/gpu" and any(kk.startswith("gres/gpu:") for kk in parse_alloc_tres(alloc_tres)):
                continue  # untyped total duplicates the typed entries
            try:
                n += int(v)
            except ValueError:
                pass
    return n


def node_count(alloc_tres: str | None) -> int:
    try:
        return int(parse_alloc_tres(alloc_tres).get("node", "1"))
    except ValueError:
        return 1


def charged_gpu_h(alloc_tres: str | None, elapsed_s: float | None, qos: str | None = None, perlmutter: bool = False) -> float | None:
    """GPU-h charged for one job/step. Perlmutter full-node QOS: nodes x 4 x hours; shared (and local Slurm): gres/gpu x hours."""
    if elapsed_s is None:
        return None
    g = gpu_count(alloc_tres)
    if perlmutter and (qos or "").lower() not in SHARED_QOS:
        g = node_count(alloc_tres) * FULL_NODE_QOS_GPU_PER_NODE if g else 0
    return round(g * elapsed_s / 3600.0, 3) if g else None


# ---------------------------------------------------------------------------------------------- gpu_sampler.sh output

def read_gpu_samples(path: str | Path) -> list[dict]:
    """gpu_samples.csv: timestamp,index,util_gpu,util_mem,mem_used_MiB,power_W (no header; nvidia-smi nounits)."""
    rows: list[dict] = []
    try:
        with open(path, newline="") as fh:
            for r in csv.reader(fh):
                if len(r) < 6:
                    continue
                try:
                    rows.append({"ts": r[0].strip(), "gpu": int(r[1]), "util_gpu": float(r[2]), "util_mem": float(r[3]),
                                 "mem_MiB": float(r[4]), "power_W": float(r[5])})
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def actor_utilisation(gpu_samples: list[dict]) -> float | None:
    """Mean nvidia-smi utilization.gpu (0-1) over all samples/GPUs. Note: it is 'a kernel ran in the interval', not SM occupancy."""
    return round(sum(r["util_gpu"] for r in gpu_samples) / len(gpu_samples) / 100.0, 4) if gpu_samples else None


# ---------------------------------------------------------------------------------------------- metrics

def _stat(xs: list[float]) -> dict | None:
    xs = sorted(xs)
    if not xs:
        return None
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 4), "p50": xs[len(xs) // 2], "max": xs[-1]}


def tokens_per_s(prom: dict | None, wall_s: float | None) -> dict | None:
    if not prom or not wall_s:
        return None
    gen, prompt = prom.get("vllm:generation_tokens_total"), prom.get("vllm:prompt_tokens_total")
    return {"gen_tok_s": round(gen / wall_s, 2) if gen is not None else None,
            "prompt_tok_s": round(prompt / wall_s, 2) if prompt is not None else None}


def candidates_per_s(n_candidates: int | None, wall_s: float | None) -> float | None:
    """Candidates (= finished simulations/rollouts) per second of wall. Caller defines 'candidate' (episode now, branch later)."""
    return round(n_candidates / wall_s, 5) if n_candidates is not None and wall_s else None


def verified_per_gpu_h(n_verified: int | None, gpu_h: float | None) -> float | None:
    """Verified episodes (graded by the env, reward in results.json) per GPU-h."""
    return round(n_verified / gpu_h, 3) if n_verified is not None and gpu_h else None


def verified_count(sims: list[dict] | None) -> int | None:
    if sims is None:
        return None
    return sum(1 for s in sims if (s.get("reward_info") or {}).get("reward") is not None)


def queue_delay(events: list[dict]) -> dict | None:
    """dequeue - enqueue per item (events carry an `item` key to pair them; unpaired ones are ignored)."""
    enq = {e["item"]: e["enqueue"] for e in events if "enqueue" in e and "item" in e}
    d = [e["dequeue"] - enq[e["item"]] for e in events if "dequeue" in e and e.get("item") in enq]
    return _stat(d)


def freshness_lag(events: list[dict]) -> dict | None:
    """Policy-version lag of each consumed rollout: learner_version at dequeue minus the version that generated it.
    Events: phase 'consume' rows carry policy_version (generator) and learner_version (learner at dequeue)."""
    lag = []
    for e in events:
        if e.get("phase") == "consume" and "policy_version" in e and "learner_version" in e:
            lag.append(e["learner_version"] - e["policy_version"])
    return _stat([float(x) for x in lag])


def learner_starvation(events: list[dict]) -> float | None:
    """Fraction of learner wall spent waiting for data. TODO(P8): needs learner `wait_start`/`wait_end` events from the
    async harness (verl or rollout/queue.py); returns None until they exist."""
    w = sum(e["wait_end"] - e["wait_start"] for e in events if "wait_start" in e and "wait_end" in e)
    span = [e["t_epoch"] for e in events if "t_epoch" in e and e.get("phase") in ("learn_start", "learn_end")]
    return round(w / (max(span) - min(span)), 4) if w and len(span) >= 2 and max(span) > min(span) else None


def time_to_quality(curve: list[tuple[float, float]] | None, target: float) -> float | None:
    """First wall time (s) at which quality >= target on a [(t_s, quality)] curve. Callers supply the eval curve (TODO P8: written by the eval queue)."""
    for t, q in sorted(curve or []):
        if q >= target:
            return t
    return None


def node_h_per_improvement(node_h: float | None, quality_gain: float | None) -> float | None:
    """Node-hours per +0.01 quality (pass^1). None if no gain (a non-positive gain has no meaningful cost)."""
    return round(node_h / (quality_gain * 100), 3) if node_h is not None and quality_gain and quality_gain > 0 else None


def run_systems_metrics(events: list[dict] | None = None, prom: dict | None = None, sims: list[dict] | None = None,
                        wall_s: float | None = None, gpu_h: float | None = None, gpu_samples: list[dict] | None = None) -> dict:
    events = events or []
    return {
        "candidates_per_s": candidates_per_s(len(sims) if sims is not None else None, wall_s),
        "tokens_per_s": tokens_per_s(prom, wall_s),
        "verified_per_gpu_h": verified_per_gpu_h(verified_count(sims), gpu_h),
        "actor_utilisation": actor_utilisation(gpu_samples or []),
        "queue_delay_s": queue_delay(events),
        "freshness_lag": freshness_lag(events),
        "learner_starvation": learner_starvation(events),
        "time_to_quality_s": None,        # TODO(P8): needs an eval-quality curve
        "node_h_per_improvement": None,   # TODO(P8): needs a baseline pass^1 and a node-h total across a comparison
    }
