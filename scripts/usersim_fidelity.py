#!/usr/bin/env python3
"""[telecom_boed] P0-4b: Qwen3-8B user-simulator fidelity from existing transcripts (CPU only, evaluation code).

Reads results/runs/*/results.json (+ run_meta.json, + oracle labels) and writes results/usersim/fidelity_8b.{md,json}.
Every number in the md is rendered here; nothing is typed by hand. All detectors are regex heuristics (see HEURISTICS /
the md "What each heuristic can miss" section) and every category writes seeded audit samples (flagged + unflagged).

Usage:  nice -n 19 python scripts/usersim_fidelity.py [--runs DIR] [--labels JSONL] [--out DIR] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
STOP, TRANSFER, OOS = "###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"
GREETING = "Hi! How can I help you today?"
USER_MODEL = "Qwen/Qwen3-8B"
PLAIN_USER_IMPL = "user_simulator"   # boed-user-noncompliant is a perturbed persona wrapper -> excluded
N_SAMPLES = 5
RECENT_TURNS = 3  # a state report is 'grounded' if a user tool ran in this or the previous 2 user turns

# user tools of the pinned tau2 telecom domain (user_tools.py @is_tool names); agent-only tools used by the user = role confusion
USER_TOOLS = frozenset("""check_status_bar check_network_status check_network_mode_preference set_network_mode_preference
run_speed_test toggle_airplane_mode check_sim_status reseat_sim_card toggle_data toggle_roaming check_data_restriction_status
toggle_data_saver_mode check_apn_settings set_apn_settings reset_apn_settings check_wifi_status toggle_wifi
check_wifi_calling_status toggle_wifi_calling check_vpn_status connect_vpn disconnect_vpn check_installed_apps check_app_status
check_app_permissions grant_app_permission can_send_mms reboot_device check_payment_request make_payment""".split())

# ----------------------------------------------------------------------------- 1. request -> expected user tool
_ACT = re.compile(r"\b(turn(?:ing)?\s+(?:it\s+)?(?:on|off)|toggle|enable|disable|switch|set\b|change|reseat|re-?seat|re-?insert|eject|"
                  r"reboot|restart|reset|grant|allow|connect|disconnect|make the payment|pay\b)", re.I)
_READ = re.compile(r"\b(check|verify|confirm|see (?:if|what|whether)|look|run|open|tell me|let me know|test|report|read)\b", re.I)
_CUE = re.compile(r"\b(please|could you|can you|would you|will you|try|let'?s|let us|go ahead|need you to|like you to|you (?:can|should|may|need|will)|"
                  r"kindly|make sure|first|next)\b|^\s*(?:(?:then|now|next|after that|also),?\s+)?(?:check|run|turn|toggle|enable|disable|switch|open|go|reboot|restart|reset|reseat|grant|set|change|connect|disconnect|make)\b", re.I)
_DONE_BY_AGENT = re.compile(r"\bi(?:'ve| have| had)\s+\w+ed\b|\bi(?:'m| am) (?:going to|about to)\b|\bi will\b|\bi'll\b", re.I)
# (name, keyword regex, act tools, read tools); order matters (more specific first)
REQ_TABLE: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = [
    ("wifi_calling", r"wi-?fi calling", ("toggle_wifi_calling",), ("check_wifi_calling_status",)),
    ("airplane", r"airplane mode|flight mode", ("toggle_airplane_mode",), ("check_status_bar", "check_network_status")),
    ("speed_test", r"speed ?test|internet speed|download speed", ("run_speed_test",), ("run_speed_test",)),
    ("status_bar", r"status bar", (), ("check_status_bar",)),
    ("data_saver", r"data saver|data restriction", ("toggle_data_saver_mode",), ("check_data_restriction_status",)),
    ("apn", r"\bapn\b|access point", ("set_apn_settings", "reset_apn_settings"), ("check_apn_settings",)),
    ("vpn", r"\bvpn\b", ("connect_vpn", "disconnect_vpn"), ("check_vpn_status",)),
    ("wifi", r"wi-?fi", ("toggle_wifi",), ("check_wifi_status",)),
    ("net_mode", r"network mode|network preference|preferred network|network type", ("set_network_mode_preference",), ("check_network_mode_preference",)),
    ("roaming", r"roaming", ("toggle_roaming",), ("check_network_status", "check_status_bar")),
    ("mobile_data", r"mobile data|cellular data|\bdata (?:is|switch|toggle|connection)\b", ("toggle_data",), ("check_network_status", "check_status_bar", "check_data_restriction_status")),
    ("sim", r"\bsim\b", ("reseat_sim_card",), ("check_sim_status",)),
    ("reboot", r"\b(?:reboot|restart|power cycle)\b", ("reboot_device",), ()),
    ("permission", r"permission", ("grant_app_permission",), ("check_app_permissions",)),
    ("mms", r"\bmms\b|picture message", (), ("can_send_mms",)),
    ("payment", r"payment|pay the bill|overdue bill", ("make_payment",), ("check_payment_request",)),
    ("signal", r"network status|cellular (?:signal|connection)|signal strength|\bsignal\b", (), ("check_network_status", "check_status_bar")),
]
_REQ_RE = [(n, re.compile(k, re.I), a, r) for n, k, a, r in REQ_TABLE]
_SENT = re.compile(r"(?<=[.!?\n])\s+")


def find_requests(text: str) -> list[dict]:
    """Agent text -> [{'kind': 'act'|'read', 'topic', 'expected': set(tool names), 'sentence'}]. One request per (sentence, first matching topic)."""
    out = []
    for s in _SENT.split(text or ""):
        if not s.strip() or not _CUE.search(s) or _DONE_BY_AGENT.search(s):
            continue
        for name, rx, act, read in _REQ_RE:
            if not rx.search(s):
                continue
            is_act = bool(act) and bool(_ACT.search(s))
            if name == "reboot" and not is_act:
                is_act = True
            if is_act:
                out.append({"kind": "act", "topic": name, "expected": set(act), "sentence": s.strip()})
            elif read and _READ.search(s):
                out.append({"kind": "read", "topic": name, "expected": set(read), "sentence": s.strip()})
            else:
                continue
            break
    return out


# ----------------------------------------------------------------------------- user text heuristics
_CLARIFY = re.compile(r"\b(how do i|how to|where (?:do|can|is|are)|i (?:can'?t|cannot|am unable|'m unable|don'?t know how)|unable to|not sure|"
                      r"what do you mean|could you (?:explain|clarify|guide|walk)|can you (?:explain|clarify|guide|walk|help me)|"
                      r"i (?:don'?t|do not) (?:understand|see|have)|i'?m not (?:able|comfortable)|rather not|refuse)\b", re.I)
_CLAIM_ACT = re.compile(r"\bi(?:'ve| have| just| already)\s+(?:just\s+)?(?:turned|toggled|enabled|disabled|switched|reseated|re-seated|rebooted|restarted|reset|"
                        r"granted|connected|disconnected|changed|set|run|checked|tried|done|made|paid)\b", re.I)
_STATE_CLAIM = re.compile(r"(status bar (?:shows?|says?|displays?|is)|(?:signal|bars?)\b.{0,25}\b(?:none|low|poor|full|one|two|three|four|\d)|"
                          r"\b(?:5g|4g|3g|lte|edge)\b|speed ?test (?:shows?|says?|result|came|completed|gave)|\b\d+(?:\.\d+)?\s*mbps\b|"
                          r"airplane mode (?:is|was|shows?)|(?:wi-?fi|data|mobile data|roaming|data roaming|vpn|sim(?: card)?|data saver) (?:is|was|shows?|appears?)\s+(?:currently\s+)?"
                          r"(?:on|off|enabled|disabled|active|connected|disconnected|locked|missing|not detected|working)|"
                          r"network (?:mode|preference|type) (?:is|was|set)|apn (?:is|settings? (?:are|is|show))|permission (?:is|are) (?:granted|denied|off|on)|"
                          r"no (?:service|signal|connection)|(?:cellular )?connection (?:is|shows)|battery)", re.I)
_DONT_UNDERSTAND = re.compile(r"\b(i (?:don'?t|do not|didn'?t) (?:really )?(?:understand|get)|(?:not sure|unsure|confused) (?:what|about what) you (?:mean|are)|"
                              r"what do you mean|i'?m (?:a bit |really |so )?confused|doesn'?t make sense|what does [\"'“]?\w+[\"'”]? mean|i don'?t know what .{1,40} (?:means|is))\b", re.I)
_SNAKE = re.compile(r"(?<![\w/])[a-z]{2,}(?:_[a-z0-9]+)+(?![\w/])")
_ROLE = re.compile(r"\b(how (?:can|may) i (?:help|assist)|thank you for (?:contacting|calling|reaching)|i(?:'ll| will) (?:look up|check your account|transfer you|"
                   r"pull up|process)|as (?:your|the) (?:customer service|support|agent|representative)|i can help you with|is there anything else i can|"
                   r"let me (?:look up|check your account|pull up) (?:your|the) (?:account|customer)|your account (?:shows|indicates)|"
                   r"i have (?:enabled|refueled|resumed|suspended) (?:roaming|your|the) )", re.I)
_TEMPLATE = re.compile(r"</?think>|<\|im_(?:start|end)\||<\|endoftext\||</?tool_call>|<\|[a-z_]+\|>|\[/?INST\]|^\s*(?:assistant|user|human|agent)\s*:", re.I | re.M)
_AIRPLANE_STATE = re.compile(r"airplane mode (?:is|was)\s+(?:currently\s+)?(on|off|enabled|disabled)", re.I)
_MBPS = re.compile(r"(\d+(?:\.\d+)?)\s*mbps", re.I)


def _norm_onoff(s: str) -> bool:
    return s.lower() in ("on", "enabled")


def user_turns(messages: list[dict]) -> list[dict]:
    """Split a transcript into agent-text messages and the user turn that follows each assistant message.
    user turn = every non-assistant message after an assistant message until the next assistant message."""
    turns = []
    cur: Optional[dict] = None
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            if cur:
                turns.append(cur)
            cur = {"agent_idx": i, "agent_msg": m, "msgs": [], "idxs": []}
        else:
            if cur is None:
                cur = {"agent_idx": -1, "agent_msg": None, "msgs": [], "idxs": []}
            cur["msgs"].append(m)
            cur["idxs"].append(i)
    if cur:
        turns.append(cur)
    for t in turns:
        ms = t["msgs"]
        t["user_tool_calls"] = [c["name"] for m in ms if m["role"] == "user" for c in (m.get("tool_calls") or [])]
        t["user_texts"] = [(ix, m.get("content") or "") for ix, m in zip(t["idxs"], ms)
                           if m["role"] == "user" and not m.get("tool_calls") and (m.get("content") or "").strip()]
        t["tool_outputs"] = [m.get("content") or "" for m in ms if m["role"] == "tool" and m.get("requestor") == "user"]
        t["stopped"] = any(STOP in x for _, x in t["user_texts"])
    return turns


def _ex(s: str, n: int = 200) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def analyze_sim(sim: dict, label_rows: list[dict]) -> dict:
    """-> {'counts': {metric: [num, den]}, 'events': {category: {'flag': [ex], 'ok': [ex]}}}. Pure function of the transcript (+ labels)."""
    msgs = sim["messages"]
    sid = (sim.get("id") or "")[:8]
    turns = user_turns(msgs)
    C: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    EV: dict[str, dict[str, list]] = defaultdict(lambda: {"flag": [], "ok": []})

    def ev(cat: str, flagged: bool, idx: int, text: str):
        EV[cat]["flag" if flagged else "ok"].append({"sim": sid, "turn": idx, "excerpt": _ex(text)})

    recent: list[bool] = []   # user-tool-call flag of the previous RECENT_TURNS user turns
    for ti, t in enumerate(turns):
        am = t["agent_msg"]
        # ---- 1. compliance (agent plain-text request -> next user turn)
        if am is not None and not am.get("tool_calls") and (am.get("content") or "").strip():
            for r in find_requests(am["content"]):
                tools = set(t["user_tool_calls"])
                ok = bool(r["expected"] & tools)
                key = "compliance_" + r["kind"]
                for k in ("compliance", key):
                    C[k][1] += 1
                    C[k][0] += ok
                utext = " ".join(x for _, x in t["user_texts"])
                if ok:
                    reason = "complied"
                elif t["stopped"]:
                    reason = "stopped"
                elif _CLARIFY.search(utext) or _DONT_UNDERSTAND.search(utext):
                    reason = "clarify_or_refuse"
                elif _CLAIM_ACT.search(utext) and not tools:
                    reason = "claims_done_no_tool"
                elif tools:
                    reason = "other_tool"
                else:
                    reason = "other_text"
                C["comp_reason_" + reason][0] += 1
                C["comp_reason_" + reason][1] = 0  # denominator filled below
                ev("compliance", not ok, am.get("turn_idx", t["agent_idx"]),
                   f"[{r['kind']}:{r['topic']} want {sorted(r['expected'])}] AGENT: {_ex(r['sentence'], 110)} || USER tools={sorted(tools)} text={_ex(utext, 90)} ({reason})")
        # ---- 3-5 on user text messages
        for ix, txt in t["user_texts"]:
            first_user = ix <= 1 or (am is not None and am.get("content") == GREETING)
            body = txt.replace(STOP, "").replace(TRANSFER, "").replace(OOS, "")
            has_tool_here = bool(t["user_tool_calls"])
            has_tool_recent = has_tool_here or any(recent[-(RECENT_TURNS - 1):])
            C["user_text_msgs"][0] += 1
            C["user_text_msgs"][1] += 1
            # 3a invented state
            state = bool(_STATE_CLAIM.search(body)) and not first_user
            inv_state = state and not has_tool_recent
            if not first_user:
                C["invented_state"][1] += 1
                C["invented_state"][0] += inv_state
                ev("invented_state", inv_state, ix, body)
            # 3b action claimed without tool this turn
            if not first_user:
                claim = bool(_CLAIM_ACT.search(body)) and not has_tool_here
                C["invented_action"][1] += 1
                C["invented_action"][0] += claim
                ev("invented_action", claim, ix, body)
            # 3c contradiction with tool output of this turn
            if t["tool_outputs"]:
                out = "\n".join(t["tool_outputs"])
                contra = False
                for mm in _AIRPLANE_STATE.finditer(body):
                    mo = re.search(r"Airplane Mode:\s*(ON|OFF)", out)
                    if mo and _norm_onoff(mm.group(1)) != (mo.group(1) == "ON"):
                        contra = True
                outn = set(_MBPS.findall(out)) | set(re.findall(r"(\d+(?:\.\d+)?)\s*Mbps", out))
                for mm in _MBPS.finditer(body):
                    if mm.group(1) not in outn and "Speed test" in out + "Speed test":  # number not present in tool output
                        contra = True
                if "failed" in out.lower() and re.search(r"speed ?test (?:shows|result|completed|gave|is)\s+\d", body, re.I):
                    contra = True
                C["contradicts_tool"][1] += 1
                C["contradicts_tool"][0] += contra
                ev("contradicts_tool", contra, ix, f"TOOL: {_ex(out, 80)} || USER: {body}")
            # 4 confused
            dont = bool(_DONT_UNDERSTAND.search(body))
            snake = _SNAKE.findall(body)
            role = bool(_ROLE.search(body))
            for k, f in (("confused_dont_understand", dont), ("confused_token_echo", bool(snake)), ("confused_role_swap_text", role)):
                C[k][1] += 1
                C[k][0] += f
            ev("confused_dont_understand", dont, ix, body)
            ev("confused_token_echo", bool(snake), ix, f"tokens={snake[:3]} :: {body}")
            ev("confused_role_swap_text", role, ix, body)
            # 5 template artefacts
            tmpl = bool(_TEMPLATE.search(txt))
            C["template_artifact"][1] += 1
            C["template_artifact"][0] += tmpl
            C["think_leak"][1] += 1
            C["think_leak"][0] += bool(re.search(r"</?think>", txt, re.I))
            ev("template_artifact", tmpl, ix, txt)
        # user used a tool that is not a user tool (role confusion by tool) / empty user reply
        for m in t["msgs"]:
            if m["role"] == "user":
                for c in (m.get("tool_calls") or []):
                    C["user_tool_calls"][1] += 1
                    bad = c["name"] not in USER_TOOLS
                    C["user_tool_calls"][0] += bad
                    ev("user_calls_agent_tool", bad, m.get("turn_idx", -1), f"{c['name']}({_ex(json.dumps(c.get('arguments')), 60)})")
                if not m.get("tool_calls") and not (m.get("content") or "").strip():
                    C["empty_user_msg"][0] += 1
                    C["empty_user_msg"][1] += 1
        recent.append(bool(t["user_tool_calls"]))

    # ---- 2. premature STOP (episode level)
    last_user = next((m for m in reversed(msgs) if m["role"] == "user" and not m.get("tool_calls")), None)
    stop = sim.get("termination_reason") == "user_stop" and last_user is not None and STOP in (last_user.get("content") or "")
    other_end = sim.get("termination_reason") == "user_stop" and not stop
    C["episodes"] = [1, 1]
    C["user_stop_other_token"] = [int(other_end), 1]
    if stop:
        agent_claim_rx = re.compile(r"\b(should (?:now )?be (?:working|resolved|fixed|restored)|(?:is|are|has been|have been) (?:now )?(?:resolved|fixed|restored|working)|"
                                    r"issue (?:is|has been) (?:resolved|fixed)|all set|you(?:'re| are) good to go)\b", re.I)
        last_agent_txt = [m.get("content") or "" for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")]
        regex_claim = any(agent_claim_rx.search(x) for x in last_agent_txt)
        C["stops"] = [1, 1]
        C["stop_reward_zero"] = [int((sim.get("reward_info") or {}).get("reward") == 0.0), 1]
        C["stop_no_claim_regex"] = [int(not regex_claim), 1]
        if label_rows:
            lr = sorted(label_rows, key=lambda r: r["msg_idx"])
            last = lr[-1]
            any_claim = any(r.get("is_claim") for r in lr)
            rem = last.get("n_faults_remaining") or 0
            C["stops_labelled"] = [1, 1]
            C["stop_faults_remaining"] = [int(rem > 0), 1]
            C["stop_no_claim_label"] = [int(not any_claim), 1]
            C["stop_premature_both"] = [int(rem > 0 and not any_claim), 1]
            prem = rem > 0
            ev("premature_stop", prem, len(msgs) - 1,
               f"faults_remaining={last.get('faults_remaining')} claim_seen={any_claim} reward={(sim.get('reward_info') or {}).get('reward')} LAST USER: {_ex(last_user.get('content'), 100)}")
        else:
            ev("premature_stop_unlabelled", not regex_claim, len(msgs) - 1,
               f"claim_regex={regex_claim} LAST USER: {_ex(last_user.get('content'), 100)}")
    return {"counts": {k: v for k, v in C.items()}, "events": {k: v for k, v in EV.items()}}


# ----------------------------------------------------------------------------- stats
def wilson(k: int, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def boot_task(per_task: dict[str, list[int]], n_boot: int = 1000, seed: int = 0) -> tuple[Optional[float], Optional[float]]:
    tasks = [v for v in per_task.values() if v[1] > 0]
    if len(tasks) < 2:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        k = n = 0
        for _ in range(len(tasks)):
            a, b = tasks[rng.randrange(len(tasks))]
            k += a
            n += b
        if n:
            vals.append(k / n)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


# ----------------------------------------------------------------------------- driver
HEADLINE = [  # (key, label)
    ("compliance", "1. compliance: agent request -> user executes matching tool in next turn (higher=better)"),
    ("compliance_act", "   1a. state-changing requests"),
    ("compliance_read", "   1b. check/read requests"),
    ("comp_noncompliant_clarify", "   1c. non-complied requests where user asks/refuses (share of ALL requests)"),
    ("stop_premature_both", "2. premature STOP: ###STOP### with faults remaining (labels) AND no agent claim"),
    ("stop_faults_remaining", "   2a. ###STOP### with >=1 fault remaining at last labelled agent turn"),
    ("stop_no_claim_label", "   2b. ###STOP### with no agent claim in the episode (claim classifier labels)"),
    ("stop_no_claim_regex", "   2c. ###STOP### with no agent 'fixed/working' phrase (regex; all stops)"),
    ("invented_state", "3a. invented output: user reports device state with no user tool call this or previous 2 turns"),
    ("invented_action", "3b. invented action: user says 'I have toggled/enabled/...' with no user tool call this turn"),
    ("contradicts_tool", "3c. contradicts tool output in same turn (airplane state / Mbps number)"),
    ("confused_dont_understand", "4a. confused: 'I don't understand / what do you mean'"),
    ("confused_token_echo", "4b. confused: echoes snake_case internal token (e.g. no_service)"),
    ("confused_role_swap_text", "4c. role confusion: user text sounds like the agent"),
    ("user_tool_calls", "4d. role confusion: user calls a non-user tool"),
    ("think_leak", "5a. <think> tag in user text"),
    ("template_artifact", "5b. any template artefact in user text (<think>, <|im_*|>, <tool_call>, 'assistant:' ...)"),
]


def derive(c: dict[str, list[int]]) -> dict[str, list[int]]:
    d = dict(c)
    d["comp_noncompliant_clarify"] = [c.get("comp_reason_clarify_or_refuse", [0, 0])[0], c.get("compliance", [0, 0])[1]]
    for k in [k for k in d if k.startswith("comp_reason_")]:
        d[k] = [d[k][0], c.get("compliance", [0, 0])[1]]
    return d


def pct(k: int, n: int) -> str:
    return "n/a" if n == 0 else f"{100 * k / n:.1f}%"


def cell(k: int, n: int, ci=None) -> str:
    lo, hi = ci if ci else wilson(k, n)
    if n == 0:
        return "0/0"
    return f"{k}/{n} = {pct(k, n)} [{100 * lo:.1f}, {100 * hi:.1f}]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--labels", default="/data/local/ss5772/telecom_h7/labels/all26/labels.jsonl")
    ap.add_argument("--out", default=str(ROOT / "results" / "usersim"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan-count", type=int, default=358)
    a = ap.parse_args(argv)
    runs_dir, out_dir = Path(a.runs), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[tuple[str, str], list[dict]] = defaultdict(list)
    lp = Path(a.labels)
    if lp.exists():
        for line in lp.open():
            r = json.loads(line)
            labels[(r["run"], r["sim_id"])].append(r)

    included, excluded = [], []
    per_run: dict[str, dict[str, list[int]]] = {}
    per_task_pooled: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    pooled: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    events: dict[str, dict[str, list]] = defaultdict(lambda: {"flag": [], "ok": []})
    n_labelled_sims = 0

    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        rp = d / "results.json"
        if not rp.exists():
            excluded.append((d.name, 0, "no results.json (events-only / incomplete run)"))
            continue
        res = json.loads(rp.read_text())
        info = res.get("info", {})
        ui = info.get("user_info", {}) or {}
        llm, impl = str(ui.get("llm", "")), ui.get("implementation")
        sims = res.get("simulations", [])
        if USER_MODEL not in llm:
            excluded.append((d.name, len(sims), f"user model {llm!r} is not {USER_MODEL}"))
            continue
        if impl != PLAIN_USER_IMPL:
            excluded.append((d.name, len(sims), f"user implementation {impl!r} (perturbed-persona wrapper, not the plain simulator)"))
            continue
        meta_note = "run_meta.json present" if (d / "run_meta.json").exists() else "no run_meta.json; user model taken from results.json info.user_info"
        rc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for s in sims:
            lr = labels.get((d.name, s.get("id")), [])
            n_labelled_sims += bool(lr)
            out = analyze_sim(s, lr)
            for k, (x, y) in out["counts"].items():
                rc[k][0] += x
                rc[k][1] += y
                pooled[k][0] += x
                pooled[k][1] += y
                per_task_pooled[k][s.get("task_id")][0] += x
                per_task_pooled[k][s.get("task_id")][1] += y
            for cat, e in out["events"].items():
                for kind in ("flag", "ok"):
                    for x in e[kind]:
                        events[cat][kind].append({"run": d.name, **x})
        per_run[d.name] = derive(rc)
        included.append((d.name, len(sims), meta_note))

    # reasons denominators + derived
    pooled_d = derive(pooled)
    for k in list(per_task_pooled):
        pass
    n_inc = sum(n for _, n, _ in included)

    rng = random.Random(a.seed)
    samples = {}
    for cat, e in sorted(events.items()):
        samples[cat] = {"flagged_total": len(e["flag"]), "unflagged_total": len(e["ok"]),
                        "flagged": rng.sample(e["flag"], min(N_SAMPLES, len(e["flag"]))),
                        "unflagged": rng.sample(e["ok"], min(N_SAMPLES, len(e["ok"])))}

    # ---------------- JSON
    pooled_ci = {}
    for key, _ in HEADLINE:
        k, n = pooled_d.get(key, [0, 0])
        pooled_ci[key] = {"num": k, "den": n, "rate": (k / n if n else None), "wilson95": wilson(k, n),
                          "boot_task95": boot_task(per_task_pooled.get(key, {}) if key in per_task_pooled else {}, seed=a.seed)}
    raw = {"included_runs": [{"run": r, "episodes": n, "note": m} for r, n, m in included],
           "excluded_runs": [{"run": r, "episodes": n, "why": w} for r, n, w in excluded],
           "n_episodes_included": n_inc, "plan_count": a.plan_count, "labelled_episodes": n_labelled_sims,
           "pooled": {k: v for k, v in pooled_d.items()}, "pooled_ci": pooled_ci,
           "per_run": per_run, "samples": samples, "seed": a.seed}
    (out_dir / "fidelity_8b.json").write_text(json.dumps(raw, indent=1, default=list))

    # ---------------- MD
    L = []
    L.append("# User-simulator fidelity (Qwen3-8B, non-thinking) from existing transcripts")
    L.append("")
    L.append("Machine-written by `scripts/usersim_fidelity.py` (seed %d). Do not hand-edit. CPU only, heuristics, see audit samples and the limits section." % a.seed)
    L.append("")
    L.append("## Episode set")
    L.append("")
    L.append(f"- Included: **{n_inc} episodes** in {len(included)} runs (plan said ~{a.plan_count}; {'matches' if n_inc == a.plan_count else 'DIFFERS by %+d' % (n_inc - a.plan_count)}). "
             f"Labelled with oracle per-turn labels: {n_labelled_sims}.")
    L.append("- Inclusion rule: `results.json` exists, `info.user_info.llm` contains `Qwen/Qwen3-8B` and `info.user_info.implementation == user_simulator`. Agent type is NOT a filter (the user side is what is measured).")
    L.append("")
    L.append("| run | episodes | note |\n|---|---:|---|")
    for r, n, m in included:
        L.append(f"| {r} | {n} | {m} |")
    L.append("")
    L.append("Excluded:\n")
    L.append("| run | episodes | why |\n|---|---:|---|")
    for r, n, w in excluded:
        L.append(f"| {r} | {n} | {w} |")
    L.append("")
    L.append("## Headline (pooled over included episodes; cell = k/n = rate [Wilson 95%] ; boot = task-cluster bootstrap 95%)")
    L.append("")
    L.append("| metric | k/n = rate [Wilson 95%] | boot-by-task 95% |\n|---|---|---|")
    for key, label in HEADLINE:
        k, n = pooled_d.get(key, [0, 0])
        b = pooled_ci[key]["boot_task95"]
        bs = "n/a" if not b or b[0] is None else f"[{100 * b[0]:.1f}, {100 * b[1]:.1f}]"
        L.append(f"| {label} | {cell(k, n)} | {bs} |")
    L.append("")
    L.append("Denominators: compliance = detected agent plain-text requests; stops = episodes ending with `###STOP###` (2a-2b labelled stops only, 2c all stops); "
             "3a/3b = user text messages after the first user message; 3c = user text messages in a turn that had a user tool output; 4a/4b/4c/5 = all user text messages; 4d = user tool calls.")
    L.append("")
    L.append("Non-compliance breakdown (share of ALL detected requests):\n")
    L.append("| outcome | k/n |\n|---|---|")
    for rk in ("complied", "clarify_or_refuse", "claims_done_no_tool", "other_tool", "other_text", "stopped"):
        k, n = pooled_d.get("comp_reason_" + rk, [0, pooled_d.get("compliance", [0, 0])[1]])
        L.append(f"| {rk} | {cell(k, n)} |")
    L.append("")
    L.append("Other counts: episodes with `user_stop` but no `###STOP###` token (transfer/out-of-scope): "
             f"{pooled_d.get('user_stop_other_token', [0, 0])[0]}/{pooled_d.get('user_stop_other_token', [0, 0])[1]}; "
             f"empty user messages: {pooled_d.get('empty_user_msg', [0, 0])[0]}; "
             f"`###STOP###` episodes with reward 0: {cell(*pooled_d.get('stop_reward_zero', [0, 0]))}.")
    L.append("")
    L.append("## Per run (k/n = rate; Wilson CIs in the json-free cells omitted for width, see fidelity_8b.json for counts)")
    L.append("")
    keys = ["compliance", "stop_premature_both", "stop_faults_remaining", "invented_state", "invented_action", "contradicts_tool",
            "confused_dont_understand", "confused_token_echo", "confused_role_swap_text", "user_tool_calls", "think_leak", "template_artifact"]
    L.append("| run | eps | " + " | ".join(keys) + " |")
    L.append("|---|---:|" + "---|" * len(keys))
    for r, n, _ in included:
        row = per_run[r]
        L.append(f"| {r} | {n} | " + " | ".join(f"{row.get(k, [0, 0])[0]}/{row.get(k, [0, 0])[1]}" for k in keys) + " |")
    L.append("")
    L.append("## Heuristics (exact rules) and what each can miss")
    L.append("")
    L.append("- **Compliance.** Agent plain-text message (no tool call) is split into sentences; a sentence is a request if it has a request cue (please / could you / try / imperative verb) "
             "and no first-person-done phrase, and a topic keyword (table `REQ_TABLE` in the script). Kind `act` if an action verb (turn on/off, toggle, enable, set, reseat, reboot, grant, connect...) is present, else `read` if a check verb is present. "
             "Expected user tool(s): act -> the matching toggle/set tool (e.g. roaming -> `toggle_roaming`); read -> matching check tool (generic network/signal/roaming/data checks accept `check_network_status` or `check_status_bar`). "
             "Complied = any expected tool is called in the user turn (all user messages until the next agent message). "
             "Misses/errors: requests phrased as questions without cues, multi-step instructions (only one topic per sentence), requests the agent withdraws or that are inapplicable because the setting is already in the right state (a correct user may skip an inapplicable toggle -> counted as non-compliant, so rate is a LOWER bound for act); "
             "read requests accept loose tools; the user may satisfy via an earlier turn (not counted).")
    L.append("- **Premature STOP.** `###STOP###` in the last user text and `termination_reason=user_stop`. Labels: last labelled agent turn's `n_faults_remaining>0` (state at that agent message, so a fix applied by the final user action is not visible -> upper bound), and no `is_claim` row in the episode. Regex fallback for unlabelled stops is the 2c row. `###TRANSFER###` / `###OUT-OF-SCOPE###` are not counted as STOP.")
    L.append("- **Invented outputs.** 3a: a state-claim regex (status bar shows, 5G/4G/LTE, Mbps, 'airplane mode is on', 'wifi is off', 'no service', battery...) on a user text message with no user tool call in that user turn or the previous two, first user message excluded (scenario text). Misses: claims that restate what the agent just said, claims after a tool older than 1 turn, paraphrases outside the regex; false positives: persona facts ('I'm in France' is not matched but 'my data is off' is) and the user quoting the agent. "
             "3b: 'I have turned/enabled/checked ...' with no user tool call in that turn (misses claims in other wording). 3c: only airplane mode ON/OFF and Mbps numbers are compared with the same-turn tool output.")
    L.append("- **Confused.** 4a regex of don't-understand phrases; 4b any snake_case token in user text (could be a legit identifier such as an app name or id -> false positives; hyphenated or capitalised variants missed); 4c agent-sounding phrases ('how can I help', 'I'll look up your account'); 4d user tool call whose name is not in the pinned user toolset.")
    L.append("- **Template artefacts.** `<think>`, `</think>`, `<|im_start|>`-style specials, `<tool_call>`, `[INST]`, line-initial `assistant:`/`user:`/`human:`. tau2 may strip some artefacts before the message is stored, so a 0 here means none survived into the transcript, not that the model never emitted them.")
    L.append("")
    L.append("## Audit samples (seeded; flagged = detector fired, unflagged = detector silent; check precision/recall by eye)")
    L.append("")
    for cat, s in samples.items():
        L.append(f"### {cat}  (flagged {s['flagged_total']}, unflagged {s['unflagged_total']})")
        for kind in ("flagged", "unflagged"):
            L.append(f"- {kind}:")
            if not s[kind]:
                L.append("  - (none)")
            for x in s[kind]:
                L.append(f"  - `{x['run']}` sim `{x['sim']}` turn {x['turn']}: {x['excerpt']}")
        L.append("")
    (out_dir / "fidelity_8b.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out_dir/'fidelity_8b.md'}  episodes={n_inc} runs={len(included)}")


if __name__ == "__main__":
    main()
