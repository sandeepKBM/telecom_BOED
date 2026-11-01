"""[telecom_boed] Parser accuracy harness (owner: W2-S7). EVALUATION code: unlike belief/parsers.py it may read user
ToolMessages, because they are the gold labels. Never import this from deployable code.

Input: tau2 SimulationRun-like dicts (the "simulations" entries of results.json; or the whole results dict).
Per agent TEXT message the "window" is everything up to the next assistant message: the user's tool calls (gold
actions), their tool results (gold observations, featurized by env.featurize) and the user's final text reply.

What is measured (all of it is a proxy; see the caveats)
- action:       RuleActionParser(agent text) vs the names of the user's tool calls in the window.
                request_detect (did the agent ask for a phone action at all), set precision/recall/F1 over tool
                names, first-action accuracy, and arg accuracy (mode / app_name / permission) where gold has args.
- observation:  ObservationParser(user text, requested, closed label space) vs featurize(tool, content) of the user's
                tool results. Reports value precision of EMITTED fields (the number that matters for the filter: a wrong
                value is worse than a missing one), coverage of gold fields, conflict / out-of-space counts, per group.
- claim:        prob_claim vs a caller-supplied label list (optional): accuracy@0.5 and Brier.
- intent:       first user message vs the intent in the task id `[intent]faults[PERSONA:..]`.

Gold-label caveats (do not read these numbers as parser-only error)
- The user may run a tool without being asked, skip a requested one (the simulator is told to do ONE action at a time and
  to ask for the rest), or run only part of a multi-request message: action recall is biased low, precision of
  "request_detect" is biased low too. `first_action_acc` is the least biased.
- The user's text may omit fields the tool printed (coverage is therefore a lower bound on parser recall) or report
  fields from an earlier call in the window; gold is the merge of the window's tool results (later call wins).
- The closed label space here is built from the values seen in the supplied runs (or `label_space=`), restricted to the
  fields the window's tools printed. Real label spaces come from the particle set, so absolute precision differs.
- With requested_source="gold" the observation parser is given the actually executed tools (isolates it from action
  parsing); "parsed" feeds it the action parser's output (end-to-end).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Optional, Sequence

from telecom_boed.belief.parsers import (RuleActionParser, RuleClaimClassifier, RuleIntentParser, RuleObservationParser)
from telecom_boed.contracts import INTENTS, DeviceAction, _frozen_items
from telecom_boed.env.featurize import featurize

_SKIP_FIELD = re.compile(r"(?:\.msg|\.raw|\.battery|ssid|^error)")


def _msgs(sim: Mapping[str, Any]) -> list[dict]:
    out: list[dict] = []
    for m in sim.get("messages") or []:
        if m.get("role") == "tool" and "tool_messages" in m:  # MultiToolMessage
            out.extend(m["tool_messages"])
        else:
            out.append(m)
    return out


def _windows(sim: Mapping[str, Any]) -> list[dict]:
    """One window per assistant TEXT message: gold calls (name, args), gold feature dict, final user text."""
    msgs = _msgs(sim)
    wins: list[dict] = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or not (m.get("content") or "").strip() or m.get("tool_calls"):
            continue
        w: dict[str, Any] = {"agent_text": m["content"], "calls": [], "gold": {}, "user_text": None, "tools": []}
        names: dict[str, tuple[str, dict]] = {}
        for n in msgs[i + 1:]:
            role = n.get("role")
            if role == "assistant":
                break
            if role == "user":
                for tc in n.get("tool_calls") or []:
                    if tc.get("requestor", "user") == "user":
                        names[tc.get("id", "")] = (tc["name"], dict(tc.get("arguments") or {}))
                        w["calls"].append((tc["name"], dict(tc.get("arguments") or {})))
                if not n.get("tool_calls") and (n.get("content") or "").strip():
                    w["user_text"] = n["content"]
            elif role == "tool" and n.get("requestor", "user") == "user":
                nm = names.get(n.get("id", ""), (None, {}))[0]
                if nm and n.get("content") is not None:
                    w["gold"].update({k: v for k, v in featurize(nm, n["content"]).items() if not _SKIP_FIELD.search(k)})
                    w["tools"].append(nm)
        wins.append(w)
    return wins


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), abs_tol=0.005)
    if isinstance(a, str) and isinstance(b, str):
        return a.casefold() == b.casefold()
    return a == b


def _ratio(n: float, d: float) -> Optional[float]:
    return None if not d else n / d


def evaluate_parsers(simulation_runs: Any, *, action_parser=None, observation_parser=None, claim_classifier=None,
                     intent_parser=None, claim_labels: Optional[Sequence[Mapping[str, Any]]] = None,
                     label_space: Optional[Mapping[str, Sequence[Any]]] = None,
                     requested_source: str = "gold") -> dict:
    """Score the parsers on logged runs. `claim_labels` = [{"text": agent message, "claim": bool}, ...] (optional)."""
    if isinstance(simulation_runs, Mapping):
        simulation_runs = simulation_runs.get("simulations", [])
    sims = list(simulation_runs)
    ap = action_parser or RuleActionParser()
    op = observation_parser or RuleObservationParser()
    cc = claim_classifier or RuleClaimClassifier()
    ip = intent_parser or RuleIntentParser()
    wins_by_sim = [_windows(s) for s in sims]

    # ---- closed label space: values seen in the supplied runs, unless given
    universe: dict[str, set] = defaultdict(set)
    for wins in wins_by_sim:
        for w in wins:
            for k, v in w["gold"].items():
                universe[k].add(v if not isinstance(v, list) else tuple(v))
    uni = ({k: sorted(vs, key=lambda x: (type(x).__name__, str(x))) for k, vs in universe.items()}
           if label_space is None else {k: list(v) for k, v in label_space.items()})

    # ---- action parser
    tp = fp = fn = 0
    det = Counter()
    first_ok = first_n = 0
    arg_ok = arg_n = 0
    n_turns = 0
    for wins in wins_by_sim:
        for w in wins:
            if w["user_text"] is None and not w["calls"]:
                continue  # last agent message of the episode: no user reaction to compare against
            n_turns += 1
            pred = ap.parse(w["agent_text"])
            pt, gt = [a.tool for a in pred], [c[0] for c in w["calls"]]
            det[("pred" if pt else "none", "gold" if gt else "none")] += 1
            tp += len(set(pt) & set(gt))
            fp += len(set(pt) - set(gt))
            fn += len(set(gt) - set(pt))
            if gt:
                first_n += 1
                first_ok += bool(pt) and pt[0] == gt[0]
            for name, args in w["calls"]:
                for a in pred:
                    if a.tool == name and args:
                        arg_n += 1
                        pa = a.args_dict()
                        arg_ok += all(_same(pa.get(k), v) for k, v in args.items() if k in pa)
                        break
    prec, rec = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    action = {
        "n_turns": n_turns,
        "request_detect": {"tp": det[("pred", "gold")], "fp": det[("pred", "none")], "fn": det[("none", "gold")],
                           "tn": det[("none", "none")],
                           "accuracy": _ratio(det[("pred", "gold")] + det[("none", "none")], sum(det.values()))},
        "tool_set": {"precision": prec, "recall": rec,
                     "f1": None if not (prec and rec) else 2 * prec * rec / (prec + rec)},
        "first_action_acc": _ratio(first_ok, first_n), "n_first": first_n,
        "arg_acc": _ratio(arg_ok, arg_n), "n_args": arg_n,
    }

    # ---- observation parser
    emitted = correct = gold_n = conflicts = oos = 0
    turns = turns_any = turns_clean = 0
    by_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for wins in wins_by_sim:
        for w in wins:
            if w["user_text"] is None or not w["gold"]:
                continue
            turns += 1
            req = (tuple(DeviceAction("user", n, _frozen_items(a)) for n, a in w["calls"])
                   if requested_source == "gold" else ap.parse(w["agent_text"]))
            space = {k: uni.get(k, [v]) for k, v in w["gold"].items()}
            ob = op.parse(w["user_text"], req, space)
            f = dict(ob.fields)
            gold_n += len(w["gold"])
            emitted += len(f)
            conflicts += len(getattr(ob, "conflicts", ()))
            oos += len(getattr(ob, "out_of_space", ()))
            ok = 0
            for k, v in f.items():
                hit = k in w["gold"] and _same(v, w["gold"][k])
                ok += hit
                g = by_group[k.split(".", 1)[0]]
                g[0] += 1
                g[1] += hit
            correct += ok
            turns_any += bool(f)
            turns_clean += bool(f) and ok == len(f)
    observation = {
        "n_turns": turns, "n_gold_fields": gold_n, "n_emitted": emitted,
        "value_precision": _ratio(correct, emitted), "coverage": _ratio(emitted, gold_n),
        "turns_with_emission": turns_any, "turns_all_correct": turns_clean,
        "n_conflicts": conflicts, "n_out_of_space": oos,
        "precision_by_group": {g: _ratio(v[1], v[0]) for g, v in sorted(by_group.items())},
        "requested_source": requested_source,
    }

    # ---- claim classifier (optional)
    claim = None
    if claim_labels:
        ps = [(cc.prob_claim(c["text"]), bool(c["claim"])) for c in claim_labels]
        claim = {"n": len(ps), "accuracy@0.5": sum((p >= 0.5) == y for p, y in ps) / len(ps),
                 "brier": sum((p - float(y)) ** 2 for p, y in ps) / len(ps),
                 "mean_p_pos": _ratio(sum(p for p, y in ps if y), sum(1 for _, y in ps if y)),
                 "mean_p_neg": _ratio(sum(p for p, y in ps if not y), sum(1 for _, y in ps if not y))}

    # ---- intent parser
    ok_i = n_i = 0
    p_true = 0.0
    ll = 0.0
    for s in sims:
        m = re.match(r"\[([a-z_]+)\]", str(s.get("task_id", "")))
        first = next((x.get("content") for x in _msgs(s) if x.get("role") == "user" and (x.get("content") or "").strip()
                      and not x.get("tool_calls")), None)
        if not m or m.group(1) not in INTENTS or first is None:
            continue
        pr = ip.parse(first)
        n_i += 1
        ok_i += max(pr, key=pr.get) == m.group(1)
        p_true += pr[m.group(1)]
        ll += -math.log(max(pr[m.group(1)], 1e-9))
    intent = {"n": n_i, "accuracy": _ratio(ok_i, n_i), "mean_p_true": _ratio(p_true, n_i), "log_loss": _ratio(ll, n_i)}

    return {"n_sims": len(sims), "action": action, "observation": observation, "claim": claim, "intent": intent,
            "caveats": "gold labels are the user's logged tool calls/results; see module docstring"}


def load_simulations(path: str) -> list[dict]:
    """results.json -> its `simulations` list."""
    with open(path) as fh:
        d = json.load(fh)
    return d["simulations"] if isinstance(d, dict) else d
