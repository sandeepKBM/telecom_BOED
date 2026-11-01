"""[telecom_boed] W5-A2: per-agent-turn labels for the H7 probes. EVALUATION / ORACLE code (reads the episode's task and replays
the oracle): never import it from a deployable module. tau2 is imported lazily (through eval.fa_replay.ensure_tau2()).

Row key = the capture index: (run, task_id, trial, turn, msg_idx). `msg_idx` is the index of the ASSISTANT message in the
simulation's `messages`; `turn` is the number of assistant messages before it (the greeting, when present, is turn 0).
Every label is computed on the state BEFORE that message = the oracle replay of every tool call (user and agent side) in
messages[:msg_idx], with the same replay machinery as W4-S15 (fa_replay: initial env from the episode's own task row, real
tools, name-only transfer flag). The message's own tool calls are NOT applied.

Fault presence ("is fault f still in the state?") per catalog subtask f, `fault_present[f]`:
    present(f) = gate_f(state) AND (f has no fix OR the fix cannot be built OR applying f's tau2 fix_funcs to a copy of the
                 state changes its state key)
  * gate_f is the fault's own cause field(s), read off the state (the leaves its init function writes). It is needed because
    some tau2 fixes are TOGGLES or are not idempotent (toggle_airplane_mode, toggle_data, toggle_roaming,
    toggle_data_saver_mode, toggle_wifi_calling, refuel_data, reboot_device): they change the state even when the fault is
    already gone, so "the fix would change the state" alone cannot say which direction it moves.
  * the fix_funcs are the BaseTask.fix_funcs of tau2/domains/telecom/tasks/{service,mobile_data,mms}_issues.py, called on the
    current state exactly as tau2's task generator does and applied with the real tools.
  * Faults with no fix_funcs (lock_sim_card_pin, contract_end_suspension) are labeled by their init function's persistence (gate
    only). overdue_bill_suspension (fix builds only while a bill is OVERDUE) falls back to the gate once the bill is paid.
  * Roaming and app-permission faults overlap in the fields they write, so their gates partition the states:
    roaming (abroad only): line roaming x device roaming = (on, off) enabled_off, (off, on) disabled_on, (off, off) disabled_off;
    permissions: sms-only, storage-only, both. A half-fixed composite therefore shows as the residual single fault.
  `gate_fix_disagree` counts states where a gate is True but its fix would not change anything (diagnostic, in summary.json).
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

SCHEMA = "h7_labels/1"
GREETING = "Hi! How can I help you today?"
TRANSFER_TOOL = "transfer_to_human_agents"
CLAIM_THRESHOLD = 0.5

FAULT_NAMES: tuple[str, ...] = (
    "airplane_mode_on", "bad_network_preference", "bad_vpn", "bad_wifi_calling", "break_apn_mms_setting",
    "break_apn_settings", "break_app_both_permissions", "break_app_sms_permission", "break_app_storage_permission",
    "contract_end_suspension", "data_mode_off", "data_saver_mode_on", "data_usage_exceeded", "lock_sim_card_pin",
    "overdue_bill_suspension", "unseat_sim_card", "user_abroad_roaming_disabled_off", "user_abroad_roaming_disabled_on",
    "user_abroad_roaming_enabled_off",
)  # the 19 subtask names that occur in the shipped catalog (data_usage_exceeded_no_refuel is never shipped)


# --------------------------------------------------------------------------- state readers (gates)
def _v(x: Any) -> Any:
    """Enum -> value; everything else unchanged (model_dump keeps enums in python mode)."""
    return getattr(x, "value", x)


def _line(agent_db: dict, user_db: dict) -> dict:
    phone = (user_db.get("surroundings") or {}).get("phone_number")
    lines = agent_db.get("lines") or []
    for ln in lines:
        if ln.get("phone_number") == phone:
            return ln
    for ln in lines:
        if ln.get("line_id") == "L1002":
            return ln
    return {}


def _perm(dev: dict, which: str) -> bool:
    perms = ((dev.get("app_statuses") or {}).get("messaging") or {}).get("permissions") or {}
    return bool(perms.get(which, True))


def _contract_ended(ln: dict) -> bool:
    """The contract-end variant writes contract_end_date 2025-01-31, before the injected bill/suspension date 2025-02-25;
    the plain overdue suspension keeps the default end date (2026-12-31)."""
    end = ln.get("contract_end_date")
    return end is not None and str(end) < "2025-02-25"


def _gates() -> dict[str, Callable[[dict, dict], bool]]:
    def roam(a: dict, u: dict) -> Optional[tuple[bool, bool]]:
        if not (u.get("surroundings") or {}).get("is_abroad"):
            return None
        return bool(_line(a, u).get("roaming_enabled")), bool((u.get("device") or {}).get("roaming_enabled"))

    def susp(a: dict, u: dict) -> bool:
        return _v(_line(a, u).get("status")) == "Suspended"

    def apn(u: dict) -> dict:
        return (u.get("device") or {}).get("active_apn_settings") or {}

    def perms(u: dict) -> tuple[bool, bool]:
        d = u.get("device") or {}
        return _perm(d, "sms"), _perm(d, "storage")

    return {
        "airplane_mode_on": lambda a, u: bool(u["device"].get("airplane_mode")),
        "unseat_sim_card": lambda a, u: bool(u["device"].get("sim_card_missing")),
        "lock_sim_card_pin": lambda a, u: _v(u["device"].get("sim_card_status")) == "locked_pin",
        "break_apn_settings": lambda a, u: _v(apn(u).get("apn_name")) == "broken",
        "break_apn_mms_setting": lambda a, u: apn(u).get("mmsc_url") in (None, ""),
        "overdue_bill_suspension": lambda a, u: susp(a, u) and not _contract_ended(_line(a, u)),
        "contract_end_suspension": lambda a, u: susp(a, u) and _contract_ended(_line(a, u)),
        "data_mode_off": lambda a, u: not u["device"].get("data_enabled"),
        "data_saver_mode_on": lambda a, u: bool(u["device"].get("data_saver_mode")),
        "bad_network_preference": lambda a, u: _v(u["device"].get("network_mode_preference")) == "2g_only",
        "bad_vpn": lambda a, u: bool(u["device"].get("vpn_connected")) and _v(
            (u["device"].get("vpn_details") or {}).get("server_performance")) == "poor",
        "data_usage_exceeded": lambda a, u: bool((u.get("surroundings") or {}).get("mobile_data_usage_exceeded")),
        "bad_wifi_calling": lambda a, u: bool(u["device"].get("wifi_calling_enabled")) and bool(
            u["device"].get("wifi_calling_mms_over_wifi")),
        "break_app_sms_permission": lambda a, u: (not perms(u)[0]) and perms(u)[1],
        "break_app_storage_permission": lambda a, u: perms(u)[0] and not perms(u)[1],
        "break_app_both_permissions": lambda a, u: (not perms(u)[0]) and not perms(u)[1],
        "user_abroad_roaming_enabled_off": lambda a, u: roam(a, u) == (True, False),
        "user_abroad_roaming_disabled_on": lambda a, u: roam(a, u) == (False, True),
        "user_abroad_roaming_disabled_off": lambda a, u: roam(a, u) == (False, False),
    }


_GATES: Optional[dict] = None
_FIXES: Optional[dict[str, list]] = None
_PRESENCE_MEMO: dict[Any, tuple[dict, dict]] = {}
STATS = {"presence_hit": 0, "presence_miss": 0, "gate_fix_disagree": 0}


def fix_registry() -> dict[str, list]:
    """subtask name -> its non-None BaseTask.fix_funcs, from tau2's own task modules (read-only clone)."""
    global _FIXES
    if _FIXES is None:
        from telecom_boed.eval import fa_replay
        fa_replay.ensure_tau2()
        from tau2.domains.telecom.tasks import mobile_data_issues, mms_issues, service_issues
        from tau2.domains.telecom.tasks.utils import BaseTask
        reg: dict[str, list] = {}
        for mod in (service_issues, mobile_data_issues, mms_issues):
            for obj in vars(mod).values():
                if isinstance(obj, BaseTask):
                    reg[obj.name] = [f for f in obj.fix_funcs if f is not None]
        _FIXES = reg
    return _FIXES


def _fix_would_change(snap: Any, name: str) -> Optional[bool]:
    """True/False: applying `name`'s fix_funcs (built on `snap`, applied in order with the real tools) changes / keeps the
    state key. None: no fix_funcs, or the fix cannot be built on this state (tau2 raises, e.g. no overdue bill left)."""
    funcs = fix_registry().get(name) or []
    if not funcs:
        return None
    from telecom_boed.env import snapshot as S
    env = S.restore(snap)
    try:
        calls = [c for f in funcs for c in f(env)]
    except Exception:
        return None
    for i, c in enumerate(calls):
        if not getattr(c, "id", None):
            c.id = f"fix{i}"
        env.get_response(c)
    return S.state_key(env) != snap.key


def fault_presence(snap: Any) -> dict[str, bool]:
    """{subtask: still present in `snap`} for all FAULT_NAMES (definition in the module docstring). Memoized on snap.key."""
    global _GATES
    hit = _PRESENCE_MEMO.get(snap.key)
    if hit is not None:
        STATS["presence_hit"] += 1
        return dict(hit[0])
    STATS["presence_miss"] += 1
    if _GATES is None:
        _GATES = _gates()
    a, u = snap.agent_db, snap.user_db
    out: dict[str, bool] = {}
    gates: dict[str, bool] = {}
    for f in FAULT_NAMES:
        g = bool(_GATES[f](a, u))
        gates[f] = g
        if not g:
            out[f] = False
            continue
        ch = _fix_would_change(snap, f)
        if ch is False:
            STATS["gate_fix_disagree"] += 1
        out[f] = ch is not False
    _PRESENCE_MEMO[snap.key] = (out, gates)
    return dict(out)


# --------------------------------------------------------------------------- helpers shared with the evaluation code
def assistant_indices(messages: Sequence[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def turn_of(messages: Sequence[dict], msg_idx: int) -> int:
    """Capture-index turn: number of assistant messages before `msg_idx` (greeting = turn 0)."""
    return sum(1 for m in messages[:msg_idx] if m.get("role") == "assistant")


def persona_from_meta(meta: Optional[dict]) -> Optional[str]:
    """User persona of the run from run_meta: user class `boed-user-<p>` -> p, `user_simulator` -> 'native'. None if unknown."""
    if not meta:
        return None
    cfg = (meta.get("run_config") or {})
    user = None
    try:
        import yaml
        user = (yaml.safe_load(cfg.get("content") or "") or {}).get("user")
    except Exception:
        user = None
    if user is None:
        for ln in (cfg.get("content") or "").splitlines():
            if ln.startswith("user:"):
                user = ln.split(":", 1)[1].strip().strip("'\"")
    if not isinstance(user, str):
        return None
    return user[len("boed-user-"):] if user.startswith("boed-user-") else ("native" if user == "user_simulator" else user)


def _gamma(path: Optional[Path] = None) -> float:
    import yaml
    from telecom_boed import paths
    p = path or (paths.CONFIG_DIR / "boed_default.yaml")
    return float((yaml.safe_load(Path(p).read_text()).get("reach_avoid") or {}).get("gamma", 1.0))


def _filter_row(rec: dict) -> dict:
    """Sidecar record -> the filter labels (posterior top-k, p_repaired, v, intent_probs, n_particles)."""
    parse = rec.get("parse") or {}
    b = parse.get("belief") or {}
    ip = b.get("intent_probs") if isinstance(b.get("intent_probs"), dict) else parse.get("intent_probs")
    return {"sidecar_turn": rec.get("turn"), "topk": rec.get("posterior_topk"), "p_repaired": b.get("p_repaired"),
            "v": b.get("v"), "intent_probs": ip, "n_particles": b.get("n_particles"), "degenerate": b.get("degenerate")}


def label_episode(sim: dict, task: Optional[dict], *, run: str, persona: Optional[str], classifier: Any, parser: Any,
                  gamma: float = 1.0, sidecar: Optional[list[dict]] = None, threshold: float = CLAIM_THRESHOLD) -> list[dict]:
    """One label row per assistant message of `sim` (see module docstring). Raises on a broken replay (caller counts it)."""
    from telecom_boed.eval import fa_replay, metrics
    M = fa_replay.ensure_tau2()
    S, grade = M["S"], M["particles"].grade
    msgs = sim.get("messages") or []
    hyp = fa_replay._hyp_for_task(task, sim["task_id"])
    intent, faults = hyp.key[0], sorted(hyp.key[1])
    term = sim.get("termination_reason")
    term = getattr(term, "value", term)
    ri = sim.get("reward_info") or {}
    reward = 0.0 if ri.get("reward") is None else float(ri["reward"])
    ec = metrics.end_class(term, metrics._last_user_text(msgs))
    a_idx = assistant_indices(msgs)
    agent_turn = metrics.agent_turn_indices(msgs)
    sc_by_msg = {mi: rec for mi, rec in zip(agent_turn, sidecar)} if sidecar and len(sidecar) == len(agent_turn) else {}
    task_persona = metrics.parse_task_id(sim["task_id"])[2]

    env = fa_replay._initial_env(hyp)
    transferred = False
    snap = None
    rows: list[dict] = []
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            if snap is None:
                snap = S.snapshot(env)
            pres = fault_presence(snap)
            repaired = grade(snap, hyp, transferred) == 1.0
            calls = m.get("tool_calls") or []
            text = m.get("content") if isinstance(m.get("content"), str) else None
            pure_text = bool(text and text.strip()) and not calls
            cp = float(classifier.prob_claim(text)) if pure_text else None
            is_claim = bool(cp is not None and cp >= threshold)
            n_dev = len(parser.parse(text)) if pure_text else 0
            remaining = sum(1 for j in a_idx if j > i)
            rem_task = [f for f in faults if pres.get(f)]
            rows.append({
                "run": run, "task_id": sim["task_id"], "trial": sim.get("trial", 0), "turn": turn_of(msgs, i), "msg_idx": i,
                "sim_id": sim.get("id"), "n_assistant_msgs": len(a_idx), "is_greeting": bool(i == 0 and not calls and (text or "").strip() == GREETING),
                "intent": intent, "faults": faults, "n_faults": len(faults), "persona": persona, "task_persona": task_persona,
                "requires_transfer": bool(hyp.requires_transfer),
                "fault_present": pres, "faults_remaining": rem_task, "n_faults_remaining": len(rem_task),
                "n_faults_present": sum(pres.values()),
                "repaired_now": bool(repaired), "transferred_so_far": bool(transferred),
                "final_reward": reward, "termination": term, "end_class": ec,
                "is_claim": is_claim, "claim_prob": cp, "false_accept_here": bool(is_claim and not repaired),
                "is_tool_call": bool(calls), "tool_name": calls[0].get("name") if calls else None,
                "n_tool_calls": len(calls), "is_device_request": bool(n_dev > 0), "n_device_actions": n_dev,
                "filter": _filter_row(sc_by_msg[i]) if i in sc_by_msg else None,
                "remaining_agent_turns": remaining, "gamma": gamma, "mc_return": reward * (gamma ** remaining),
            })
        # apply this message's tool calls AFTER labelling it (labels are on the state before the message)
        cs = fa_replay._calls(m)
        for tc in cs:
            if tc.requestor == "assistant" and tc.name == TRANSFER_TOOL:
                transferred = True
            env.get_response(tc)
        if cs:
            snap = None
    final = grade(S.snapshot(env), hyp, transferred)
    for r in rows:
        r["replay_final_grade"] = final
    return rows


def label_run(run_dir: Path | str, *, classifier: Any = None, parser: Any = None, threshold: float = CLAIM_THRESHOLD,
              gamma: Optional[float] = None) -> dict:
    """Labels for every simulation of a run dir -> {"rows": [...], "errors": [...], "n_sims": int, "sidecar_matched": int}."""
    from telecom_boed.eval import fa_replay, metrics
    fa_replay.ensure_tau2()
    if classifier is None or parser is None:
        from telecom_boed.belief.parsers import RuleActionParser, RuleClaimClassifier
        classifier = classifier or RuleClaimClassifier()
        parser = parser or RuleActionParser()
    g = _gamma() if gamma is None else gamma
    run = metrics.load_run(run_dir)
    sims = run["results"].get("simulations") or []
    matched = metrics.match_sidecar(sims, run["sidecar"], getattr(classifier, "prob_claim", None)) if run["sidecar"] else {}
    persona = persona_from_meta(run["meta"])
    rows: list[dict] = []
    errors: list[dict] = []
    for s in sims:
        try:
            rows.extend(label_episode(s, run["tasks"].get(s.get("task_id")), run=run["name"], persona=persona, classifier=classifier,
                                      parser=parser, gamma=g, sidecar=matched.get(s.get("id")), threshold=threshold))
        except Exception as e:  # one bad transcript must not kill the run; reported in summary.json
            errors.append({"run": run["name"], "task_id": s.get("task_id"), "trial": s.get("trial"), "error": f"{type(e).__name__}: {e}"})
    return {"rows": rows, "errors": errors, "n_sims": len(sims), "sidecar_matched": len(matched), "run": run["name"],
            "sims": sims}


# --------------------------------------------------------------------------- validation + balance
def validate(label: dict) -> dict:
    """Checks on one label_run() result (the three W5-A2 invariants plus the replay self-check)."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in label["rows"]:
        by[(r["task_id"], r["trial"], r["sim_id"])].append(r)
    count_bad, t0_bad, last_bad, final_bad = [], [], [], []
    n_user_stop = 0
    for s in label["sims"]:
        k = (s["task_id"], s.get("trial", 0), s.get("id"))
        rs = by.get(k)
        if rs is None:
            continue
        n_asst = sum(1 for m in s.get("messages") or [] if m.get("role") == "assistant")
        if len(rs) != n_asst:
            count_bad.append(k[:2])
        r0 = rs[0]
        if sorted(f for f, p in r0["fault_present"].items() if p) != r0["faults"]:
            t0_bad.append((k[:2], sorted(f for f, p in r0["fault_present"].items() if p), r0["faults"]))
        if s.get("termination_reason") in ("user_stop", "agent_stop") or getattr(s.get("termination_reason"), "value", None) == "user_stop":
            n_user_stop += 1
            if float(rs[-1]["repaired_now"]) != rs[-1]["final_reward"]:
                last_bad.append(k[:2])
            if float(rs[-1]["replay_final_grade"]) != rs[-1]["final_reward"]:
                final_bad.append(k[:2])
    return {"n_sims": len(label["sims"]), "n_sims_labeled": len(by), "n_rows": len(label["rows"]), "n_user_stop": n_user_stop,
            "turn_count_mismatch": count_bad, "turn0_fault_set_mismatch": t0_bad,
            "last_turn_repaired_ne_reward_user_stop": last_bad, "replay_final_ne_reward_user_stop": final_bad}


_BOOL_FIELDS = ("repaired_now", "transferred_so_far", "requires_transfer", "is_claim", "false_accept_here", "is_tool_call",
                "is_device_request", "is_greeting")
_CAT_FIELDS = ("intent", "persona", "task_persona", "n_faults", "end_class", "termination", "tool_name", "final_reward",
               "n_faults_remaining", "n_faults_present")
_NUM_FIELDS = ("claim_prob", "mc_return", "remaining_agent_turns")


def balance(rows: Sequence[dict]) -> dict:
    """Label balance per field over all rows (bool -> counts, categorical -> counts, numeric -> n/mean/min/max)."""
    out: dict[str, Any] = {"n_rows": len(rows), "n_episodes": len({(r["run"], r["task_id"], r["trial"]) for r in rows})}
    for f in _BOOL_FIELDS:
        c = Counter(bool(r[f]) for r in rows)
        out[f] = {"true": c[True], "false": c[False], "frac_true": (c[True] / len(rows)) if rows else None}
    for f in _CAT_FIELDS:
        out[f] = dict(sorted(Counter(str(r[f]) for r in rows).items()))
    for f in _NUM_FIELDS:
        v = [float(r[f]) for r in rows if r[f] is not None]
        out[f] = {"n": len(v), "mean": (sum(v) / len(v)) if v else None, "min": min(v) if v else None, "max": max(v) if v else None}
    out["fault_present"] = {f: {"true": sum(bool(r["fault_present"][f]) for r in rows),
                                "frac_true": (sum(bool(r["fault_present"][f]) for r in rows) / len(rows)) if rows else None}
                            for f in FAULT_NAMES}
    fr = [r for r in rows if r.get("filter")]
    out["filter"] = {"n_rows_with_filter": len(fr),
                     "v": dict(Counter(str(r["filter"].get("v")) for r in fr)),
                     "p_repaired_mean": (sum(float(r["filter"]["p_repaired"]) for r in fr if r["filter"].get("p_repaired") is not None)
                                         / max(1, sum(r["filter"].get("p_repaired") is not None for r in fr))) if fr else None}
    return out
