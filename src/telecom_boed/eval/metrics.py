"""[telecom_boed] H6 evaluation metrics from run artifacts (owner: W4-S15). EVALUATION code, never deployable.

H6: reach-avoid lowers false acceptance (FA) at matched turns. Everything is computed from
`results/runs/<run>/{results.json, sidecar.jsonl, run_meta.json, events.jsonl}`; no number is typed by hand.
This module is pure stdlib and does not import tau2. The agent-claim replay (needs tau2 + the oracle) lives in
`eval/fa_replay.py` and is called lazily by `evaluate_run(..., replay=True)`.

Definitions (all rates are per EPISODE unless said otherwise; per-task value = mean over that task's trials, and the
headline number is the mean over tasks, with a seeded percentile bootstrap over tasks):
- pass^k          tau2: mean over tasks of C(c,k)/C(n,k), c = successes (reward == 1) in n trials (tasks with n < k skipped).
                  Same formula as scripts/telecom_status.py (a test pins them together).
- end class       user_stop is split by the user's token: ###STOP### (`stop`), ###TRANSFER### (`user_transfer`),
                  ###OUT-OF-SCOPE### (`user_oos`); then agent_stop, max_steps, too_many_errors, and `harness` (timeout,
                  infrastructure_error, agent_error, user_error, context_window_exceeded, unexpected_error).
- episode FA      user_stop with ###STOP### in the last user message and reward 0. max_steps / too_many_errors / harness
                  endings are NEVER FA (they are reported as their own rates). `rate_given_stop` = FA / #(user_stop+STOP).
- claim FA        (needs the replay) first agent text message the claim classifier scores >= threshold (default 0.5); the oracle
                  replays messages[:i] on the task's initial state and `repaired` = particles.grade(state, hyp, transferred).
                  FA = a claim was made and repaired == 0. `rate_all` is per episode (no claim = not FA);
                  `rate_given_claim` is over episodes that made a claim.
- verified success  reward == 1 AND the agent had a confirming observation after the last state-changing action. Source: the
                  sidecar `parse.belief.v` of the episode's last agent turn when the sidecar matches the episode, else a transcript
                  proxy: a non-error READ-type tool result (agent or user side) after the last non-error WRITE-type call
                  (the write's own output does not count). The proxy is not an LR test; it only says "something was checked".
- turns           number of agent messages (text or tool call) excluding the fixed greeting (= sidecar records per episode).
- tokens          sums of assistant `usage` (prompt_tokens, completion_tokens); tokens/verified success = total tokens of ALL
                  episodes / #verified successes (cost of a verified success, failures included).
- groups          macro-average over groups of task-level means, for intent, persona, fault count, subtask (each fault of the
                  task; overlapping), requires_transfer (from the task's reward_basis). worst = lowest pass^1 / highest FA group.
- paired          two runs on the same task ids, difference = B - A (A baseline, B treatment); wins/losses/ties per task;
                  paired bootstrap CI over tasks; exact two-sided McNemar on trial-matched binary pairs.
- power           tasks needed to detect a paired FA difference at alpha=.05, power=.8 (normal approx and exact conditional
                  binomial test), from the two FA rates plus a discordance fraction or a phi correlation.
"""
from __future__ import annotations

import json
import math
import random
import re
import zlib
from collections import Counter, defaultdict
from math import comb, lgamma, log, exp
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Optional, Sequence

STOP, TRANSFER, OOS = "###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"
GREETING = "Hi! How can I help you today?"
TRANSFER_TOOL = "transfer_to_human_agents"
HARNESS_REASONS = frozenset({"timeout", "infrastructure_error", "agent_error", "user_error", "context_window_exceeded",
                             "unexpected_error"})
# tau2 @is_tool(ToolType.WRITE) names in the pinned telecom domain (tools.py, user_tools.py); a test checks this list against tau2.
WRITE_TOOLS = frozenset({
    "suspend_line", "resume_line", "send_payment_request", "enable_roaming", "disable_roaming", "refuel_data",
    "set_network_mode_preference", "toggle_airplane_mode", "reseat_sim_card", "toggle_data", "toggle_roaming",
    "toggle_data_saver_mode", "set_apn_settings", "reset_apn_settings", "toggle_wifi", "toggle_wifi_calling",
    "connect_vpn", "disconnect_vpn", "grant_app_permission", "reboot_device", "make_payment",
})
END_CLASSES = ("stop", "user_transfer", "user_oos", "user_stop_other", "agent_stop", "max_steps", "too_many_errors", "harness")
_ID_RE = re.compile(r"\[([^\]]*)\](.*)\[PERSONA:([^\]]*)\]")


# --------------------------------------------------------------------------- statistics
def pass_hat_k(per_task: dict[str, list[float]], k: int) -> Optional[float]:
    """τ² pass^k: mean over tasks of C(c, k) / C(n, k) with c successes in n trials (tasks with n < k skipped).
    Identical to scripts/telecom_status.py:pass_hat_k."""
    vals = [comb(sum(r == 1.0 for r in rs), k) / comb(len(rs), k) for rs in per_task.values() if len(rs) >= k]
    return sum(vals) / len(vals) if vals else None


def pass_hat_k_terms(per_task: dict[str, list[float]], k: int) -> dict[str, float]:
    return {t: comb(sum(r == 1.0 for r in rs), k) / comb(len(rs), k) for t, rs in per_task.items() if len(rs) >= k}


def _seed(seed: int, label: str) -> int:
    return (int(seed) * 1_000_003 + zlib.crc32(label.encode())) & 0x7FFFFFFF


def mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, label: str = "", alpha: float = 0.05
                 ) -> tuple[Optional[float], Optional[float]]:
    """Seeded percentile bootstrap CI of the mean over `values` (one value per task)."""
    n = len(values)
    if n == 0:
        return None, None
    rng = random.Random(_seed(seed, label))
    ms = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return ms[int(alpha / 2 * n_boot)], ms[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def bootstrap_ratio_ci(nums: Sequence[float], dens: Sequence[float], n_boot: int = 2000, seed: int = 0, label: str = "",
                       alpha: float = 0.05) -> tuple[Optional[float], Optional[float]]:
    """Seeded cluster bootstrap CI of sum(nums)/sum(dens) over tasks (resamples tasks; resamples with zero den are dropped)."""
    n = len(nums)
    if n == 0 or sum(dens) == 0:
        return None, None
    rng = random.Random(_seed(seed, label))
    rs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(dens[i] for i in idx)
        if d > 0:
            rs.append(sum(nums[i] for i in idx) / d)
    if not rs:
        return None, None
    rs.sort()
    return rs[int(alpha / 2 * len(rs))], rs[min(len(rs) - 1, int((1 - alpha / 2) * len(rs)))]


def _est(values: Sequence[float], n_boot: int, seed: int, label: str) -> dict:
    """{value, ci95, n} for a per-task value list (None when empty)."""
    if not values:
        return {"value": None, "ci95": [None, None], "n": 0}
    lo, hi = bootstrap_ci(values, n_boot, seed, label)
    return {"value": mean(values), "ci95": [lo, hi], "n": len(values)}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant counts b, c (binomial(b+c, 1/2); p = min(1, 2 P(X <= min(b, c))))."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


# --------------------------------------------------------------------------- power
def _binom_logpmf(n: int, k: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p)


def _mcnemar_critical(d: int, alpha: float) -> int:
    """Largest k with exact two-sided p(min = k | d discordant) <= alpha (-1 if none)."""
    k_best, cum = -1, 0.0
    for k in range(d // 2 + 1):
        cum += exp(_binom_logpmf(d, k, 0.5))
        if min(1.0, 2 * cum) <= alpha:
            k_best = k
        else:
            break
    return k_best


def paired_cells(p_a: float, p_b: float, discordance: Optional[float] = None, rho: float = 0.0) -> tuple[float, float]:
    """(p10, p01): P(A=1,B=0), P(A=0,B=1) for event rates p_a, p_b. Give the discordant fraction psi = p10 + p01, or a phi
    correlation rho between the paired outcomes (rho=0: independent, which is the largest psi and so the most conservative)."""
    if not (0.0 <= p_a <= 1.0 and 0.0 <= p_b <= 1.0):
        raise ValueError("rates must be in [0, 1]")
    if discordance is not None:
        psi = float(discordance)
        d = p_a - p_b
        p10, p01 = (psi + d) / 2, (psi - d) / 2
    else:
        p11 = p_a * p_b + rho * math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
        p10, p01 = p_a - p11, p_b - p11
    if p10 < -1e-12 or p01 < -1e-12 or p10 + p01 > 1 + 1e-12:
        raise ValueError(f"infeasible paired cells p10={p10:.4g} p01={p01:.4g} for p_a={p_a} p_b={p_b}")
    return max(p10, 0.0), max(p01, 0.0)


def mcnemar_power_exact(n: int, p10: float, p01: float, alpha: float = 0.05) -> float:
    """Exact power of the conditional exact McNemar test with n pairs: D ~ Bin(n, psi), b | D ~ Bin(D, p10/psi)."""
    psi = p10 + p01
    if psi <= 0 or n <= 0:
        return 0.0
    q = p10 / psi
    total = 0.0
    for d in range(1, n + 1):
        pd = exp(_binom_logpmf(n, d, psi))
        if pd < 1e-14:
            continue
        kc = _mcnemar_critical(d, alpha)
        if kc < 0:
            continue
        # reject if min(b, d-b) <= kc  <=>  b <= kc or b >= d-kc
        low = sum(exp(_binom_logpmf(d, k, q)) for k in range(kc + 1))
        high = sum(exp(_binom_logpmf(d, k, q)) for k in range(d - kc, d + 1)) if d - kc > kc else 0.0
        total += pd * min(1.0, low + high)
    return min(1.0, total)


def tasks_needed_paired(p_a: float, p_b: float, alpha: float = 0.05, power: float = 0.8, discordance: Optional[float] = None,
                        rho: float = 0.0, exact: bool = True, n_max: int = 20000) -> dict:
    """Number of paired tasks needed to detect FA rates p_a (baseline) vs p_b at two-sided `alpha` and `power`.
    Normal approximation (Connor 1987): n = (z_{a/2} sqrt(psi) + z_b sqrt(psi - d^2))^2 / d^2, d = p10 - p01 = p_a - p_b.
    `n_exact` = smallest n whose exact conditional McNemar power is >= `power` (searched upward from the normal estimate)."""
    p10, p01 = paired_cells(p_a, p_b, discordance, rho)
    psi, delta = p10 + p01, p10 - p01
    out: dict[str, Any] = {"p_a": p_a, "p_b": p_b, "alpha": alpha, "power": power, "psi": psi, "p10": p10, "p01": p01,
                           "rho": rho if discordance is None else None, "n_normal": None, "n_exact": None}
    if abs(delta) < 1e-12:
        return out | {"note": "no difference to detect (p_a == p_b)"}
    nd = NormalDist()
    za, zb = nd.inv_cdf(1 - alpha / 2), nd.inv_cdf(power)
    n_norm = math.ceil((za * math.sqrt(psi) + zb * math.sqrt(max(psi - delta ** 2, 0.0))) ** 2 / delta ** 2)
    out["n_normal"] = n_norm
    if exact:
        n = max(2, int(0.8 * n_norm))
        while n <= n_max:
            if mcnemar_power_exact(n, p10, p01, alpha) >= power:
                out["n_exact"] = n
                break
            n += max(1, n // 50) if n > 200 else 1
        if out["n_exact"] is not None and out["n_exact"] > 200:  # coarse stride above 200: refine downward
            n = out["n_exact"]
            while n > 2 and mcnemar_power_exact(n - 1, p10, p01, alpha) >= power:
                n -= 1
            out["n_exact"] = n
    return out


# --------------------------------------------------------------------------- loading
def parse_task_id(task_id: str) -> tuple[str, list[str], str]:
    m = _ID_RE.fullmatch(task_id or "")
    if not m:
        return "unknown", [], "unknown"
    intent, faults, persona = m.groups()
    return intent, sorted(f for f in faults.split("|") if f), persona


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_run(run_dir: Path | str) -> dict:
    run_dir = Path(run_dir)
    res_p = run_dir / "results.json"
    if not res_p.exists():
        raise FileNotFoundError(f"{res_p} missing: run {run_dir.name} has no results.json")
    results = json.loads(res_p.read_text())
    meta_p = run_dir / "run_meta.json"
    return {"name": run_dir.name, "path": str(run_dir), "results": results,
            "tasks": {t["id"]: t for t in results.get("tasks", []) if isinstance(t, dict) and "id" in t},
            "sidecar": _read_jsonl(run_dir / "sidecar.jsonl"),
            "meta": json.loads(meta_p.read_text()) if meta_p.exists() else None,
            "events": _read_jsonl(run_dir / "events.jsonl")}


# --------------------------------------------------------------------------- per-episode analysis
def _tool_calls(m: dict) -> list[dict]:
    return m.get("tool_calls") or []


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"]:
            return m["content"]
    return ""


def end_class(term: Optional[str], last_user: str) -> str:
    if term == "user_stop":
        if STOP in last_user:
            return "stop"
        if TRANSFER in last_user:
            return "user_transfer"
        if OOS in last_user:
            return "user_oos"
        return "user_stop_other"
    if term == "agent_stop":
        return "agent_stop"
    if term in ("max_steps", "too_many_errors"):
        return term
    return "harness"


def agent_turn_indices(messages: list[dict]) -> list[int]:
    """Indices of assistant messages that are agent turns (the fixed first-message greeting is excluded)."""
    idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if idx and idx[0] == 0 and not _tool_calls(messages[0]) and (messages[0].get("content") or "").strip() == GREETING:
        idx = idx[1:]
    return idx


def transcript_verified(messages: list[dict]) -> bool:
    """Proxy for verified (see module doc): a non-error READ result after the last non-error WRITE call."""
    name_of: dict[str, str] = {}
    err_of: dict[str, bool] = {}
    for m in messages:
        for tc in _tool_calls(m):
            name_of[tc.get("id")] = tc.get("name", "")
        if m.get("role") == "tool":
            err_of[m.get("id")] = bool(m.get("error"))
    last_write = -1
    for i, m in enumerate(messages):
        for tc in _tool_calls(m):
            if tc.get("name") in WRITE_TOOLS and not err_of.get(tc.get("id"), False):
                last_write = max(last_write, i)
    for i, m in enumerate(messages):
        if i > last_write and m.get("role") == "tool" and not m.get("error"):
            nm = name_of.get(m.get("id"), "")
            if nm and nm not in WRITE_TOOLS and nm != TRANSFER_TOOL:
                return True
    return False


def _usage(messages: list[dict]) -> tuple[int, int]:
    p = c = 0
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("usage"), dict):
            p += int(m["usage"].get("prompt_tokens") or 0)
            c += int(m["usage"].get("completion_tokens") or 0)
    return p, c


def analyze_episode(sim: dict, task: Optional[dict]) -> dict:
    msgs = sim.get("messages") or []
    term = sim.get("termination_reason")
    if hasattr(term, "value"):
        term = term.value
    ri = sim.get("reward_info") or {}
    reward_missing = ri.get("reward") is None
    reward = 0.0 if reward_missing else float(ri["reward"])
    last_user = _last_user_text(msgs)
    ec = end_class(term, last_user)
    intent, faults, persona = parse_task_id(sim.get("task_id", ""))
    turn_idx = agent_turn_indices(msgs)
    p_tok, c_tok = _usage(msgs)
    basis = ((task or {}).get("evaluation_criteria") or {}).get("reward_basis")
    ep = {
        "sim_id": sim.get("id"), "task_id": sim.get("task_id"), "trial": sim.get("trial", 0), "intent": intent,
        "faults": faults, "persona": persona,
        "requires_transfer": (None if basis is None else "ACTION" in basis),
        "termination": term, "end_class": ec, "reward": reward, "reward_missing": reward_missing,
        "success": reward == 1.0,
        "fa_episode": bool(term == "user_stop" and STOP in last_user and reward == 0.0),
        "turns": len(turn_idx),
        "agent_text_msgs": sum(1 for i in turn_idx if not _tool_calls(msgs[i])),
        "agent_tool_calls": sum(len(_tool_calls(msgs[i])) for i in turn_idx),
        "user_tool_calls": sum(len(_tool_calls(m)) for m in msgs if m.get("role") == "user"),
        "prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok,
        "verified_transcript": transcript_verified(msgs),
        "verified_sidecar": None, "claim": None,
    }
    return ep


# --------------------------------------------------------------------------- sidecar <-> episode matching
def _sc_signature(rec: dict) -> tuple[Optional[str], Optional[str]]:
    """(tool name or None, emitted text or None) of one sidecar record, for matching to an assistant message."""
    parse, dec = rec.get("parse") or {}, rec.get("decision") or {}
    em = parse.get("emitted")
    if isinstance(em, dict):  # shadow mode (boed-plain): what was actually emitted
        kind, tool, text = em.get("kind"), em.get("tool"), None
    else:
        ch = dec.get("chosen") or {}
        kind, text = ch.get("kind"), ch.get("text")
        acts = ch.get("actions") or []
        tool = acts[0].get("tool") if kind == "agent_tool" and acts else None
    if kind == "transfer":
        tool = TRANSFER_TOOL
    elif kind != "agent_tool":
        tool = None
    return tool, (" ".join(text.split()) if isinstance(text, str) else None)


def _sc_claim_prob(rec: dict) -> Optional[float]:
    parse, dec = rec.get("parse") or {}, rec.get("decision") or {}
    em = parse.get("emitted")
    if isinstance(em, dict):
        return em.get("claim_prob")
    idx = (dec.get("chosen") or {}).get("candidate_idx")
    for c in parse.get("candidates") or []:
        if c.get("idx") == idx:
            return c.get("claim_prob")
    return None


def match_sidecar(sims: list[dict], sidecar: list[dict], claim_prob_fn: Optional[Callable[[str], float]] = None) -> dict[str, list[dict]]:
    """sim id -> that episode's sidecar records (ordered by turn). The sidecar sim_id is a random tag, so episodes are matched by
    signature: same number of agent turns, same tool name at each tool turn, same text where the record carries it. Ties
    (several episodes with the same signature) are broken by claim-probability agreement when `claim_prob_fn` is given; pairs
    that stay ambiguous are left unmatched (the verified flag then uses the transcript proxy)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in sidecar:
        groups[r.get("sim_id")].append(r)
    for g in groups.values():
        g.sort(key=lambda r: r.get("turn", 0))
    cand: dict[str, dict[str, float]] = {}
    for s in sims:
        msgs = s.get("messages") or []
        turns = [msgs[i] for i in agent_turn_indices(msgs)]
        sig = [((_tool_calls(m)[0].get("name") if _tool_calls(m) else None),
                (" ".join(m["content"].split()) if isinstance(m.get("content"), str) and not _tool_calls(m) else None))
               for m in turns]
        for gid, recs in groups.items():
            if len(recs) != len(sig):
                continue
            ok = True
            for (tool, text), rec in zip(sig, recs):
                rt, rtext = _sc_signature(rec)
                if tool != rt or (rtext is not None and text != rtext):
                    ok = False
                    break
            if not ok:
                continue
            sec = 0.0
            if claim_prob_fn is not None and turns:
                hits = tot = 0
                for m, rec in zip(turns, recs):
                    cp = _sc_claim_prob(rec)
                    if cp is not None and isinstance(m.get("content"), str):
                        tot += 1
                        hits += abs(cp - claim_prob_fn(m["content"])) < 0.011
                sec = hits / tot if tot else 0.0
            cand.setdefault(s["id"], {})[gid] = sec
    out: dict[str, list[dict]] = {}
    used: set[str] = set()
    for sid, gs in cand.items():
        best = max(gs.values())
        top = [g for g, v in gs.items() if v == best]
        if len(top) != 1:
            continue
        g = top[0]
        rivals = [s2 for s2, gs2 in cand.items() if g in gs2 and s2 != sid and gs2[g] >= best]
        if rivals or g in used:
            continue
        used.add(g)
        out[sid] = groups[g]
    return out


# --------------------------------------------------------------------------- aggregation
def _task_rows(episodes: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for e in episodes:
        by[e["task_id"]].append(e)
    rows = {}
    for t, es in by.items():
        n = len(es)
        claims = [e for e in es if e.get("claim") and e["claim"].get("status") == "ok"]
        rows[t] = {
            "task_id": t, "n": n, "c": sum(e["success"] for e in es), "intent": es[0]["intent"], "faults": es[0]["faults"],
            "persona": es[0]["persona"], "requires_transfer": es[0]["requires_transfer"],
            "rewards": [e["reward"] for e in es],
            "reward": mean([e["reward"] for e in es]),
            "fa_episode": mean([float(e["fa_episode"]) for e in es]),
            "fa_stop_num": sum(e["fa_episode"] for e in es), "fa_stop_den": sum(e["end_class"] == "stop" for e in es),
            "claim_fa": mean([float(e["claim"]["claim_fa"]) for e in claims]) if len(claims) == n else None,
            "claim_made": mean([float(e["claim"]["has_claim"]) for e in claims]) if len(claims) == n else None,
            "claim_fa_num": sum(e["claim"]["claim_fa"] for e in claims), "claim_den": sum(e["claim"]["has_claim"] for e in claims),
            "verified_success": mean([float(e["verified_success"]) for e in es]),
            "turns": mean([float(e["turns"]) for e in es]),
            "agent_tool_calls": mean([float(e["agent_tool_calls"]) for e in es]),
            "total_tokens": sum(e["total_tokens"] for e in es),
            "end_classes": dict(Counter(e["end_class"] for e in es)),
        }
    return rows


_GROUP_METRICS = ("reward", "fa_episode", "claim_fa", "verified_success", "turns")


def _group_stats(rows: dict[str, dict]) -> dict:
    dims: dict[str, Callable[[dict], list]] = {
        "intent": lambda r: [r["intent"]], "persona": lambda r: [r["persona"]],
        "n_faults": lambda r: [str(len(r["faults"]))], "subtask": lambda r: list(r["faults"]) or ["none"],
        "requires_transfer": lambda r: [] if r["requires_transfer"] is None else [str(r["requires_transfer"]).lower()],
    }
    out = {}
    for dim, fn in dims.items():
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in rows.values():
            for g in fn(r):
                buckets[g].append(r)
        if not buckets:
            continue
        groups = {}
        for g, rs in sorted(buckets.items()):
            d = {"n_tasks": len(rs)}
            for m in _GROUP_METRICS:
                vals = [r[m] for r in rs if r[m] is not None]
                d[m] = mean(vals) if len(vals) == len(rs) else None
            groups[g] = d
        macro = {}
        for m in _GROUP_METRICS:
            vals = [d[m] for d in groups.values() if d[m] is not None]
            macro[m] = mean(vals)
        worst = {}
        for m, key, hi in (("reward", "pass1_lowest", False), ("fa_episode", "fa_episode_highest", True),
                           ("claim_fa", "claim_fa_highest", True)):
            cand = [(d[m], g) for g, d in groups.items() if d[m] is not None]
            if cand:
                v, g = (max(cand, key=lambda x: (x[0], x[1])) if hi else min(cand, key=lambda x: (x[0], x[1])))
                worst[key] = {"group": g, "value": v, "n_tasks": groups[g]["n_tasks"]}
        out[dim] = {"groups": groups, "macro": macro, "worst": worst}
    return out


def _events_summary(events: list[dict], n_sims: int) -> dict:
    ph = {e.get("phase") for e in events}
    end = [e for e in events if e.get("phase") == "tau2_end"]
    step_end = [e for e in events if e.get("phase") == "step_end"]
    prog = [e for e in events if e.get("phase") in ("progress", "watch_end") and "sims_done" in e]
    rc = end[-1].get("rc") if end else None
    timeout_ev = any("timeout" in str(p) for p in ph) or rc == 124 or any(e.get("exit_rc") == 124 for e in step_end)
    return {"n_events": len(events), "tau2_rc": rc, "step_exit_rc": step_end[-1].get("exit_rc") if step_end else None,
            "sims_done_last": prog[-1].get("sims_done") if prog else None, "timeout_event": bool(timeout_ev),
            "incomplete": bool(events and (not end or rc not in (0, None) or
                                           (prog and prog[-1].get("sims_done") not in (None, n_sims))))}


def evaluate_run(run_dir: Path | str, *, replay: bool = True, threshold: float = 0.5, claim_classifier: Any = None,
                 n_boot: int = 2000, seed: int = 0, ks: Optional[Iterable[int]] = None, run: Optional[dict] = None) -> dict:
    """All metrics for one run. Returns {"metrics": <json-able dict>, "episodes": [...], "task_rows": {...}}."""
    run = run or load_run(run_dir)
    sims = run["results"].get("simulations") or []
    episodes = [analyze_episode(s, run["tasks"].get(s.get("task_id"))) for s in sims]

    clf = claim_classifier
    if clf is None:
        from telecom_boed.belief.parsers import RuleClaimClassifier
        clf = RuleClaimClassifier()
    claim_prob_fn = getattr(clf, "prob_claim", None)

    # sidecar -> episode
    matched = match_sidecar(sims, run["sidecar"], claim_prob_fn) if run["sidecar"] else {}
    for e in episodes:
        recs = matched.get(e["sim_id"])
        if recs:
            v = None
            for r in recs:
                b = (r.get("parse") or {}).get("belief")
                if isinstance(b, dict) and "v" in b:
                    v = bool(b["v"])
            e["verified_sidecar"] = v
        e["verified"] = e["verified_sidecar"] if e["verified_sidecar"] is not None else e["verified_transcript"]
        e["verified_source"] = "sidecar" if e["verified_sidecar"] is not None else "transcript"
        e["verified_success"] = bool(e["success"] and e["verified"])

    # claim FA replay
    replay_info: dict[str, Any] = {"status": "skipped", "errors": 0, "threshold": threshold}
    if replay:
        from telecom_boed.eval import fa_replay
        res = fa_replay.replay_run(sims, run["tasks"], classifier=clf, threshold=threshold)
        by_id = {e["sim_id"]: e for e in episodes}
        for sid, c in res.items():
            by_id[sid]["claim"] = c
        replay_info = {"status": "ok", "errors": sum(c.get("status") != "ok" for c in res.values()), "threshold": threshold,
                       "classifier": type(clf).__name__,
                       "replay_final_grade_disagrees_with_tau2": sum(c.get("final_replay_agrees") is False for c in res.values())}

    rows = _task_rows(episodes)
    per_task_rewards = {t: r["rewards"] for t, r in rows.items()}
    trials = max((r["n"] for r in rows.values()), default=0)
    ks = list(ks) if ks else list(range(1, max(trials, 1) + 1))
    n = len(episodes)
    ecount = Counter(e["end_class"] for e in episodes)
    tcount = Counter(str(e["termination"]) for e in episodes)
    tvals = lambda key: [r[key] for r in rows.values() if r[key] is not None]  # noqa: E731

    m: dict[str, Any] = {
        "run": run["name"], "path": run["path"], "n_sims": n, "n_tasks": len(rows), "trials_max": trials,
        "meta": _meta(run), "events": _events_summary(run["events"], n),
        "termination_reasons": dict(tcount),
        "end_classes": {k: ecount.get(k, 0) for k in END_CLASSES},
        "end_class_rates": {k: (ecount.get(k, 0) / n if n else None) for k in END_CLASSES},
        "harness_failures": {"count": ecount.get("harness", 0), "reasons": {k: v for k, v in tcount.items() if k in HARNESS_REASONS},
                             "timeout_event_in_events_jsonl": _events_summary(run["events"], n)["timeout_event"]},
        "reward_missing": sum(e["reward_missing"] for e in episodes),
        "pass_hat_k": {str(k): _est(list(pass_hat_k_terms(per_task_rewards, k).values()), n_boot, seed, f"pass^{k}") for k in ks},
        "mean_reward": _est(tvals("reward"), n_boot, seed, "reward"),
    }
    # episode FA
    fa_n = sum(e["fa_episode"] for e in episodes)
    stop_n = ecount.get("stop", 0)
    num, den = [r["fa_stop_num"] for r in rows.values()], [r["fa_stop_den"] for r in rows.values()]
    lo, hi = bootstrap_ratio_ci(num, den, n_boot, seed, "fa_given_stop")
    m["episode_fa"] = {"count": fa_n, "n_user_stop_stop": stop_n, "rate_all": _est(tvals("fa_episode"), n_boot, seed, "fa_episode"),
                       "rate_given_stop": {"value": (fa_n / stop_n if stop_n else None), "ci95": [lo, hi], "n": stop_n}}
    # claim FA
    claims = [e for e in episodes if e.get("claim") and e["claim"].get("status") == "ok"]
    if replay_info["status"] == "ok":
        cn = [r["claim_fa_num"] for r in rows.values()]
        cd = [r["claim_den"] for r in rows.values()]
        lo, hi = bootstrap_ratio_ci(cn, cd, n_boot, seed, "claim_fa_given_claim")
        n_cl = sum(e["claim"]["has_claim"] for e in claims)
        n_fa = sum(e["claim"]["claim_fa"] for e in claims)
        m["claim_fa"] = {"status": "ok" if len(claims) == n else "partial", "replay": replay_info, "n_replayed": len(claims),
                         "n_claims": n_cl, "n_claim_fa": n_fa, "claim_rate": _est(tvals("claim_made"), n_boot, seed, "claim_made"),
                         "rate_all": _est(tvals("claim_fa"), n_boot, seed, "claim_fa"),
                         "rate_given_claim": {"value": (n_fa / n_cl if n_cl else None), "ci95": [lo, hi], "n": n_cl},
                         "claims": [{"task_id": e["task_id"], "trial": e["trial"], "sim_id": e["sim_id"],
                                     "claim_prob": e["claim"]["claim_prob"], "claim_turn": e["claim"]["claim_turn"],
                                     "repaired_at_claim": e["claim"]["repaired_at_claim"], "claim_fa": e["claim"]["claim_fa"],
                                     "final_reward": e["reward"], "end_class": e["end_class"]}
                                    for e in claims if e["claim"]["has_claim"]]}
    else:
        m["claim_fa"] = {"status": "skipped", "replay": replay_info}
    # verified success
    vs = sum(e["verified_success"] for e in episodes)
    m["verified_success"] = {"count": vs, "rate": _est(tvals("verified_success"), n_boot, seed, "verified_success"),
                             "success_count": sum(e["success"] for e in episodes),
                             "source_counts": dict(Counter(e["verified_source"] for e in episodes)),
                             "n_sidecar_matched": len(matched),
                             "count_transcript_proxy_only": sum(e["success"] and e["verified_transcript"] for e in episodes)}
    # cost
    tot_tok = sum(e["total_tokens"] for e in episodes)
    comp_tok = sum(e["completion_tokens"] for e in episodes)
    m["turns"] = _est(tvals("turns"), n_boot, seed, "turns")
    m["agent_tool_calls"] = _est(tvals("agent_tool_calls"), n_boot, seed, "agent_tool_calls")
    m["tokens"] = {"prompt": sum(e["prompt_tokens"] for e in episodes), "completion": comp_tok, "total": tot_tok,
                   "per_episode_mean_total": (tot_tok / n if n else None),
                   "per_verified_success_total": (tot_tok / vs if vs else None),
                   "per_verified_success_completion": (comp_tok / vs if vs else None)}
    m["groups"] = _group_stats(rows)
    return {"metrics": m, "episodes": episodes, "task_rows": rows}


def _meta(run: dict) -> dict:
    meta = run.get("meta") or {}
    side = run.get("sidecar") or []
    rc = meta.get("run_config") or {}
    return {"run_id_sidecar": side[0].get("run_id") if side else None,
            "config_name": rc.get("name"), "config_sha256": rc.get("sha256"), "tau2_sha": meta.get("tau2_sha"),
            "contract_version": meta.get("contract_version"), "mode": meta.get("mode"), "host": meta.get("host"),
            "model_rev": side[0].get("model_rev") if side else None, "n_sidecar_records": len(side),
            "sidecar_contract_versions": sorted({r.get("contract_version") for r in side if r.get("contract_version") is not None})}


# --------------------------------------------------------------------------- paired comparison
def _pairs_by_trial(ea: list[dict], eb: list[dict], task_ids: Sequence[str]) -> list[tuple[dict, dict]]:
    ga, gb = defaultdict(list), defaultdict(list)
    for e in ea:
        ga[e["task_id"]].append(e)
    for e in eb:
        gb[e["task_id"]].append(e)
    out = []
    for t in task_ids:
        for x, y in zip(sorted(ga[t], key=lambda e: e["trial"]), sorted(gb[t], key=lambda e: e["trial"])):
            out.append((x, y))
    return out


def paired_comparison(ra: dict, rb: dict, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of two evaluate_run() results on their common task ids. diff = B - A."""
    A, B = ra["task_rows"], rb["task_rows"]
    common = sorted(set(A) & set(B))
    out: dict[str, Any] = {"a": ra["metrics"]["run"], "b": rb["metrics"]["run"], "n_common_tasks": len(common),
                           "only_in_a": sorted(set(A) - set(B)), "only_in_b": sorted(set(B) - set(A)), "diff_is": "B - A"}
    if not common:
        return out | {"note": "no common task ids"}
    metrics = (("pass1", "reward", False), ("fa_episode", "fa_episode", True), ("claim_fa", "claim_fa", True),
               ("verified_success", "verified_success", False), ("turns", "turns", True))
    table = []
    for t in common:
        a, b = A[t], B[t]
        table.append({"task_id": t, "intent": a["intent"],
                      **{f"a_{k}": a[src] for k, src, _ in metrics}, **{f"b_{k}": b[src] for k, src, _ in metrics},
                      "a_end": a["end_classes"], "b_end": b["end_classes"],
                      "success": _wlt(a["reward"], b["reward"], higher_better=True)})
    out["per_task"] = table
    diffs, wlt = {}, {}
    for key, src, lower_better in metrics:
        pairs = [(A[t][src], B[t][src]) for t in common]
        if any(x is None or y is None for x, y in pairs):
            diffs[key] = {"status": "unavailable (metric missing for some tasks)"}
            continue
        d = [y - x for x, y in pairs]
        lo, hi = bootstrap_ci(d, n_boot, seed, f"pair:{key}")
        diffs[key] = {"a": mean([x for x, _ in pairs]), "b": mean([y for _, y in pairs]), "diff": mean(d), "ci95": [lo, hi],
                      "n_tasks": len(d), "ci_excludes_0": bool(lo is not None and (lo > 0 or hi < 0))}
        w = Counter(_wlt(x, y, higher_better=not lower_better) for x, y in pairs)
        wlt[key] = {"a_wins": w.get("a", 0), "b_wins": w.get("b", 0), "ties": w.get("tie", 0)}
    out["diffs"] = diffs
    out["wins_losses_ties"] = wlt
    pairs = _pairs_by_trial(ra["episodes"], rb["episodes"], common)
    mc = {}
    binaries = {"success": lambda e: bool(e["success"]), "fa_episode": lambda e: bool(e["fa_episode"]),
                "claim_fa": (lambda e: bool(e["claim"]["claim_fa"]) if e.get("claim") and e["claim"].get("status") == "ok" else None),
                "verified_success": lambda e: bool(e["verified_success"])}
    for k, f in binaries.items():
        xs = [(f(x), f(y)) for x, y in pairs]
        if any(p is None or q is None for p, q in xs):
            mc[k] = {"status": "unavailable"}
            continue
        a_only = sum(p and not q for p, q in xs)
        b_only = sum(q and not p for p, q in xs)
        mc[k] = {"n_pairs": len(xs), "a_only": a_only, "b_only": b_only, "both": sum(p and q for p, q in xs),
                 "neither": sum((not p) and (not q) for p, q in xs), "p_exact": mcnemar_exact(a_only, b_only)}
    out["mcnemar"] = mc
    # post-hoc power from the observed FA rates (flagged as such)
    fa = mc.get("claim_fa") if mc.get("claim_fa", {}).get("n_pairs") else mc.get("fa_episode")
    which = "claim_fa" if mc.get("claim_fa", {}).get("n_pairs") else "fa_episode"
    if fa and fa.get("n_pairs"):
        n = fa["n_pairs"]
        pa, pb = (fa["a_only"] + fa["both"]) / n, (fa["b_only"] + fa["both"]) / n
        psi = (fa["a_only"] + fa["b_only"]) / n
        try:
            out["power_observed"] = {"metric": which, "posthoc": True,
                                     **tasks_needed_paired(pa, pb, discordance=psi if psi > abs(pa - pb) - 1e-12 else None)}
        except ValueError as e:
            out["power_observed"] = {"metric": which, "posthoc": True, "error": str(e)}
    return out


def _wlt(x: float, y: float, higher_better: bool) -> str:
    if abs(x - y) < 1e-12:
        return "tie"
    return "a" if ((x > y) == higher_better) else "b"


# --------------------------------------------------------------------------- markdown
def _f(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _c(x: Any) -> str:
    """Markdown table cell: task ids contain '|'."""
    return str(x).replace("|", "\\|")


def _ci(est: dict, nd: int = 3) -> str:
    if not est or est.get("value") is None:
        return "n/a"
    lo, hi = est["ci95"]
    return f"{_f(est['value'], nd)} [{_f(lo, nd)}, {_f(hi, nd)}]"


def render_summary(runs: list[dict], pairs: list[dict], title: str = "telecom_boed H6 evaluation") -> str:
    L = [f"# {title} (machine-written by scripts/telecom_eval.py; do not hand-edit)", "",
         "95% CIs are seeded percentile bootstraps over tasks. FA = false acceptance. harness endings are never FA.", ""]
    hdr = ["run", "sims", "tasks", "pass^1", "pass^2", "episode FA (all)", "FA|user STOP", "claim FA (all)", "FA|claim", "claim rate",
           "verified succ.", "turns", "agent tool calls", "tok/verif.succ"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in runs:
        m = r["metrics"]
        cf = m["claim_fa"]
        L.append("| " + " | ".join([
            m["run"], str(m["n_sims"]), str(m["n_tasks"]), _ci(m["pass_hat_k"].get("1")), _ci(m["pass_hat_k"].get("2")),
            _ci(m["episode_fa"]["rate_all"]), _ci(m["episode_fa"]["rate_given_stop"]),
            _ci(cf.get("rate_all")) if cf["status"] != "skipped" else "skipped",
            _ci(cf.get("rate_given_claim")) if cf["status"] != "skipped" else "skipped",
            _ci(cf.get("claim_rate")) if cf["status"] != "skipped" else "skipped",
            _ci(m["verified_success"]["rate"]), _ci(m["turns"], 1), _ci(m["agent_tool_calls"], 1),
            _f(m["tokens"]["per_verified_success_total"], 0)]) + " |")
    L += ["", "## Endings (episodes)", "", "| run | " + " | ".join(END_CLASSES) + " | harness reasons |", "|" + "---|" * (len(END_CLASSES) + 2)]
    for r in runs:
        m = r["metrics"]
        L.append(f"| {m['run']} | " + " | ".join(str(m["end_classes"][k]) for k in END_CLASSES) + f" | {m['harness_failures']['reasons'] or '-'} |")
    for r in runs:
        m = r["metrics"]
        L += ["", f"## {m['run']}", "",
              f"- config: {m['meta']['config_name']} (sha {str(m['meta']['config_sha256'])[:12]}), mode {m['meta']['mode']}, host {m['meta']['host']}",
              f"- verified success: {m['verified_success']['count']} of {m['verified_success']['success_count']} successes "
              f"(sources {m['verified_success']['source_counts']}, sidecar-matched episodes {m['verified_success']['n_sidecar_matched']})",
              f"- tokens: prompt {m['tokens']['prompt']}, completion {m['tokens']['completion']}",
              f"- events: {m['events']}"]
        cf = m["claim_fa"]
        if cf["status"] == "skipped":
            L.append("- claim FA: skipped (replay disabled)")
        else:
            L.append(f"- claim FA: {cf['n_claim_fa']} of {cf['n_claims']} claims unrepaired; replay errors {cf['replay']['errors']}, "
                     f"replay final-grade disagreements {cf['replay'].get('replay_final_grade_disagrees_with_tau2')}")
            if cf.get("claims"):
                L += ["", "| task | trial | claim prob | turn | repaired at claim | final reward | end |", "|---|---|---|---|---|---|---|"]
                for c in cf["claims"]:
                    L.append(f"| {_c(c['task_id'])} | {c['trial']} | {_f(c['claim_prob'], 2)} | {c['claim_turn']} | {c['repaired_at_claim']} | "
                             f"{_f(c['final_reward'], 1)} | {c['end_class']} |")
        for dim, g in m["groups"].items():
            w = g["worst"]
            L += ["", f"Groups by {dim} (macro: pass^1 {_f(g['macro']['reward'])}, FA {_f(g['macro']['fa_episode'])}, "
                      f"claim FA {_f(g['macro']['claim_fa'])}); worst pass^1: {w.get('pass1_lowest', {}).get('group')} "
                      f"({_f(w.get('pass1_lowest', {}).get('value'))}), worst FA: {w.get('fa_episode_highest', {}).get('group')} "
                      f"({_f(w.get('fa_episode_highest', {}).get('value'))})", "",
                  "| group | tasks | pass^1 | FA | claim FA | verified | turns |", "|---|---|---|---|---|---|---|"]
            for name, d in g["groups"].items():
                L.append(f"| {_c(name)} | {d['n_tasks']} | {_f(d['reward'])} | {_f(d['fa_episode'])} | {_f(d['claim_fa'])} | "
                         f"{_f(d['verified_success'])} | {_f(d['turns'], 1)} |")
    for p in pairs:
        L += ["", f"## Paired: A={p['a']} vs B={p['b']} (diff = B - A, {p['n_common_tasks']} common tasks)", ""]
        if "diffs" not in p:
            L.append(p.get("note", "n/a"))
            continue
        L += ["| metric | A | B | diff [95% CI] | A wins | B wins | ties |", "|---|---|---|---|---|---|---|"]
        for k, d in p["diffs"].items():
            if "diff" not in d:
                L.append(f"| {k} | {d.get('status')} | | | | | |")
                continue
            w = p["wins_losses_ties"][k]
            L.append(f"| {k} | {_f(d['a'])} | {_f(d['b'])} | {_f(d['diff'])} [{_f(d['ci95'][0])}, {_f(d['ci95'][1])}] | "
                     f"{w['a_wins']} | {w['b_wins']} | {w['ties']} |")
        L += ["", "| McNemar (exact) | pairs | A only | B only | both | neither | p |", "|---|---|---|---|---|---|---|"]
        for k, d in p["mcnemar"].items():
            if "p_exact" not in d:
                L.append(f"| {k} | unavailable | | | | | |")
                continue
            L.append(f"| {k} | {d['n_pairs']} | {d['a_only']} | {d['b_only']} | {d['both']} | {d['neither']} | {_f(d['p_exact'], 4)} |")
        if p.get("power_observed"):
            po = p["power_observed"]
            L += ["", f"Post-hoc power ({po['metric']}, observed rates; not a pre-registered number): "
                      f"n_normal={po.get('n_normal')}, n_exact={po.get('n_exact')} tasks for alpha=.05, power=.8"
                      + (f" ({po['error']})" if po.get("error") else "")]
        L += ["", "| task | A reward | B reward | A FA | B FA | A claim FA | B claim FA | A turns | B turns | success |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in p["per_task"]:
            L.append(f"| {_c(r['task_id'])} | {_f(r['a_pass1'], 2)} | {_f(r['b_pass1'], 2)} | {_f(r['a_fa_episode'], 2)} | "
                     f"{_f(r['b_fa_episode'], 2)} | {_f(r['a_claim_fa'], 2)} | {_f(r['b_claim_fa'], 2)} | {_f(r['a_turns'], 1)} | "
                     f"{_f(r['b_turns'], 1)} | {r['success']} |")
    return "\n".join(L) + "\n"
