"""[telecom_boed] Toy POMDP Predictor + Grader built from tests/fixtures/W2/toy_cases.yaml (semantics: that file's header).

`load_toy(case) -> (predictor, grader, belief0, candidates, cfg)`; candidates = nonterminals in spec order, then claim, transfer.
- Particle: hyp_key=("toy", frozenset({hyp id})), state=Snapshot(agent_db=<state dict>, user_db={}, vpn_default=None,
  key=(md5 of canonical state json, "", "")). Identical (hyp, state) particles merge after every action; weights renormalise per obs.
- Action: one `effects` entry is applied with its prob. A hypothesis flagged `unfixable` ignores every effect. An action is
  "mutating" (resets v) iff some effect has a non-empty `set`, whatever outcome occurred.
- Observation on the post-state: with prob mention_p the agent sees (field, value) -- field "repaired" = all goal factors hold --
  else ("null",). v: mutating -> 0; ("repaired", True) -> 1; ("repaired", False) -> 0; anything else leaves v unchanged.
- Cost: the spec's `cost` is in turn units. The predictor accumulates it in beta'.cost AND exposes `action_cost(cand)`, which
  decide.utility.cost_of prefers over cfg turn_cost, so lambda*C(a) uses the toy cost. Terminals cost 0.
- Terminals (not used by the myopic scorers): claim -> unchanged belief; transfer -> transferred=True, kappa=True iff some mass
  is on a hypothesis that does not require transfer.
- Grader: hypotheses with requires_transfer have reward_basis ACTION only: grade = 1 iff transferred (env state ignored, because an
  unfixable hypothesis never reaches `goal`). Others: grade = 1 iff all goal factors hold (ENV only; transfer does not change it).
- cfg: nested like configs/boed_default.yaml (cost / utility / reach_avoid / lookahead), values from the case `params`.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import yaml

from telecom_boed.contracts import (Belief, CandidateAction, DecisionKind, DeviceAction, Particle, Snapshot, _frozen_items)

SPEC_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "W2" / "toy_cases.yaml"
NULL = ("null",)


def load_spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text())["cases"]


def _snap(state: Mapping[str, Any]) -> Snapshot:
    h = hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest()
    return Snapshot(agent_db=dict(state), user_db={}, vpn_default=None, key=(h, "", ""))


def hyp_key(hid: str) -> tuple:
    return ("toy", frozenset({hid}))


def label(c: CandidateAction) -> str:
    if c.kind == DecisionKind.CLAIM_RESOLVED:
        return "claim"
    if c.kind == DecisionKind.TRANSFER:
        return "transfer"
    return c.actions[0].tool


class ToyGrader:
    def __init__(self, goal: Mapping[str, Any], req: set):
        self.goal, self.req = dict(goal), set(req)

    def goal_holds(self, state: Snapshot) -> bool:
        return all(state.agent_db.get(k) == v for k, v in self.goal.items())

    def grade(self, state: Snapshot, hyp_key: tuple, transferred: bool) -> float:
        if hyp_key in self.req:
            return 1.0 if transferred else 0.0
        return 1.0 if self.goal_holds(state) else 0.0

    def requires_transfer(self, hyp_key: tuple) -> bool:
        return hyp_key in self.req


class ToyPredictor:
    def __init__(self, spec: Mapping[str, Any], grader: ToyGrader):
        self.spec, self.grader = spec, grader
        self.hyps = {hyp_key(h["id"]): h for h in spec["hypotheses"]}
        self.actions = {n: a for n, a in spec["actions"].items() if a["kind"] == "request_device"}

    def is_mutating(self, cand: CandidateAction) -> bool:
        return any(e["set"] for e in self.actions[label(cand)]["effects"])

    def action_cost(self, cand: CandidateAction) -> float:
        return float(self.actions[label(cand)]["cost"]) if cand.kind == DecisionKind.REQUEST_DEVICE else 0.0

    def make_belief(self, priors: Optional[Mapping[str, float]] = None, cost: Optional[float] = None) -> Belief:
        pri = priors or {h["id"]: h["prior"] for h in self.spec["hypotheses"]}
        parts = tuple(Particle(hyp_key(h["id"]), _snap(h["init"]), float(pri[h["id"]])) for h in self.spec["hypotheses"]
                      if pri[h["id"]] > 0)
        return Belief(parts, cost=self.spec.get("start_cost", 0.0) if cost is None else cost,
                      intent_probs=_frozen_items({"toy": 1.0}))

    @staticmethod
    def _merge(items: Sequence[tuple]) -> tuple[Particle, ...]:
        acc: dict = defaultdict(float)
        rep: dict = {}
        for hk, snap, w in items:
            k = (hk, snap.key)
            acc[k] += w
            rep.setdefault(k, (hk, snap))
        tot = sum(acc.values())
        return tuple(Particle(rep[k][0], rep[k][1], w / tot) for k, w in acc.items())

    def predict(self, belief: Belief, cand: CandidateAction) -> dict:
        if cand.kind == DecisionKind.CLAIM_RESOLVED:
            return {("claim",): (1.0, belief)}
        if cand.kind == DecisionKind.TRANSFER:
            fix = any(not self.grader.requires_transfer(p.hyp_key) for p in belief.particles)
            return {("transfer",): (1.0, Belief(belief.particles, belief.v, belief.kappa or fix, belief.cost, True,
                                                belief.intent_probs, belief.step + 1))}
        a = self.actions[label(cand)]
        mut = self.is_mutating(cand)
        mp = float(a["obs"]["mention_p"])
        field = a["obs"]["field"]
        groups: dict = defaultdict(list)
        for p in belief.particles:
            unfix = self.hyps[p.hyp_key].get("unfixable", False)
            for e in a["effects"]:
                st = dict(p.state.agent_db)
                if not unfix:
                    st.update(e["set"])
                snap = _snap(st)
                w = p.weight * float(e["prob"])
                if w == 0:
                    continue
                val = self.grader.goal_holds(snap) if field == "repaired" else st.get(field)
                if mp > 0:
                    groups[(field, val)].append((p.hyp_key, snap, w * mp))
                if mp < 1:
                    groups[NULL].append((p.hyp_key, snap, w * (1 - mp)))
        out: dict = {}
        for key, items in groups.items():
            prob = sum(w for _, _, w in items)
            v = False if mut else belief.v
            if key == ("repaired", True):
                v = True
            elif key == ("repaired", False):
                v = False
            out[key] = (prob, Belief(self._merge(items), v, belief.kappa, belief.cost + float(a["cost"]), belief.transferred,
                                     belief.intent_probs, belief.step + 1))
        return out


def toy_cfg(params: Mapping[str, Any]) -> dict:
    p = params
    return {
        "cost": {"lambda": p["lambda"], "turn_cost": p["turn_cost"], "tool_cost": 0.25, "rho": p["rho"], "budget_B": p["budget_B"]},
        "utility": {"c_FA": p["c_FA"], "c_esc": p["c_esc"]},
        "reach_avoid": {"gamma": p["gamma"], "delta": p["delta"], "eta": p["eta"], "p_star": 0.9, "v_lr": 10.0},
        "lookahead": {"depth_k": p["depth_k"], "inner_m": 6, "memo_round": 1.0e-9},
    }


def load_toy(case_name: str) -> tuple:
    spec = load_spec()[case_name]
    req = {hyp_key(h["id"]) for h in spec["hypotheses"] if h.get("requires_transfer")}
    grader = ToyGrader(spec["goal"], req)
    pred = ToyPredictor(spec, grader)
    cands = [CandidateAction(DecisionKind.REQUEST_DEVICE, (DeviceAction("user", n),)) for n in pred.actions]
    cands += [CandidateAction(DecisionKind.CLAIM_RESOLVED), CandidateAction(DecisionKind.TRANSFER)]
    return pred, grader, pred.make_belief(), cands, toy_cfg(spec["params"])


def parse_path_obs(v: Any) -> tuple:
    return NULL if v == "null" else (v[0], v[1])


def run_path(predictor: ToyPredictor, belief: Belief, candidates: Sequence[CandidateAction], decide: Callable,
             path: Optional[Mapping[str, Any]] = None, max_steps: int = 10) -> list[str]:
    """Run decide -> predict -> feed an observation until a terminal. Observation for action name n: path[n] if listed
    (applied every time n is chosen), else the most likely one (ties: lexicographically smallest repr(key), deterministic)."""
    seq: list[str] = []
    for _ in range(max_steps):
        cand = candidates[decide(belief, candidates)]
        seq.append(label(cand))
        if cand.kind in (DecisionKind.CLAIM_RESOLVED, DecisionKind.TRANSFER):
            break
        outs = predictor.predict(belief, cand)
        if path and label(cand) in path:
            key = parse_path_obs(path[label(cand)])
        else:
            key = min(outs, key=lambda k: (-round(outs[k][0], 12), repr(k)))
        belief = outs[key][1]
    return seq
