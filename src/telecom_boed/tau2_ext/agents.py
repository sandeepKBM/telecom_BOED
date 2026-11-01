"""[telecom_boed] tau2 agents `boed-plain` and `boed-controller` (owner: W3-S11). DEPLOYABLE module.

Both agents see only what tau2 shows an agent: AssistantMessage, UserMessage (text), ToolMessage(requestor="assistant").
They never read the episode's task or the user DB: the factory takes `task=` and discards it.

boed-plain       Exactly tau2's LLMAgent (same generate() call, same emitted messages). In SHADOW mode it also parses the message
                 it just emitted into a CandidateAction, scores it + the two terminals with the configured scorer (override
                 "none"), updates the deployable Bayes filter and appends one SidecarRecord per turn. Nothing it does changes
                 the emitted message; any exception in the BOED path is logged in the sidecar (`parse["error"]`) and ignored.
boed-controller  Proposal §4. Per turn: sample N candidates with one litellm.completion(n=N) call (candidates.py), parse each into
                 a CandidateAction, `Controller.decide(belief, cands)`, emit the chosen message (or the verify TEMPLATE text when
                 filter_commit replaced an unverified claim). Failure ladder: BOED path raises -> emit the first sampled
                 candidate; no usable sample -> one greedy generate() call, i.e. LLMAgent behaviour.

Turn flow (both): incoming message -> [first UserMessage: RuleIntentParser -> filter.init] [later UserMessage: observation parser over
the label space of the actions we requested last turn (source "user_report")] [ToolMessage(assistant): Observation("agent_tool",
featurize(tool, content))] -> filter.update(belief, <candidate we emitted last turn>, obs) -> score/decide -> emit.

Settings side channel: the launcher (register.py, W3-S10) puts `_boed` = {run, agent_kwargs, boed_config, boed_config_sha256} into
llm_args; `register.split_boed_args` strips it. agent_kwargs keys = BoedSettings fields (scorer, override, n_candidates,
candidate_temperature, time_budget_ms, persona, sidecar, ...). Also: `configure(**kw)` process-wide defaults (lowest precedence) and
`boed_<field>` llm_args keys (highest; used by tests, also stripped before litellm). Fields e.g. boed_scorer, boed_override,
boed_n_candidates, boed_candidate_temperature, boed_persona, boed_config (yaml path), boed_time_budget_ms, boed_sidecar (jsonl path),
boed_run_id, boed_sim_id, boed_parser, boed_model_rev, boed_lora_rev, boed_dedupe, boed_filter_factory (tests).
The filter is built per episode (cold caches), so the anytime budget sees the same cache state in every episode.
W3-F1 kwargs: `repair_invalid_user_tools` (default False; True: a call to a phone tool the agent lacks, e.g. check_network_status, is replaced
by the text "Could you please <action> and tell me exactly what you see?" + a REQUEST_DEVICE candidate; boed-plain emits the repaired message,
so boed-plain+repair is the control for boed-controller+repair; sidecar `parse.repaired` = [{idx, from_tool, to_request}]) and
`intent_prune_below` (default 0.05; intents the first user message gives less than this are not in the filter support when the message
carried an intent cue; an uninformative first message keeps the uniform prior and a later informative one reweights/prunes it).

Sidecar line = dataclasses.asdict(contracts.SidecarRecord). Extra information lives inside `parse` (belief summary, incoming
observation, emitted/candidate parses, sampling counters, `error`, `fallback`), because the dataclass is frozen by the contract.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml
from pydantic import Field

import tau2.agent.llm_agent as _la  # `generate` is looked up on this module at call time (tests patch tau2.agent.llm_agent.generate)
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage

from telecom_boed import paths
from telecom_boed.belief.filter import ParticleFilter
from telecom_boed.belief.parsers import build_parsers
from telecom_boed.contracts import (Belief, CandidateAction, Decision, DecisionKind, Observation, Particle, SidecarRecord, _frozen_items)
from telecom_boed.decide.controller import Controller
from telecom_boed.decide.utility import p_repaired
from telecom_boed.env.featurize import featurize
from telecom_boed.tau2_ext.register import split_boed_args
from telecom_boed.tau2_ext.candidates import (TRANSFER_TOOL, Samples, repair_user_tool_call, sample_candidates, to_candidate_info)

PLAIN, CONTROLLER = "boed-plain", "boed-controller"
_SIDECAR_LOCK = threading.Lock()


# --------------------------------------------------------------------------- settings (side channel)
@dataclass
class BoedSettings:
    scorer: Optional[str] = None  # None -> cfg controller.scorer
    override: Optional[str] = None  # None -> cfg controller.override (boed-plain always runs "none")
    n_candidates: Optional[int] = None  # None -> cfg controller.n_candidates
    candidate_temperature: Optional[float] = None  # None -> cfg controller.candidate_temperature
    candidate_max_tokens: Optional[int] = None  # None -> cfg controller.candidate_max_tokens if present, else uncapped
    persona: str = "default"  # user-model persona of the filter (configs/user_model/<persona>.yaml)
    config: Optional[str] = None  # path of the BOED yaml; None -> configs/boed_default.yaml
    time_budget_ms: Any = "cfg"  # "cfg" -> cfg latency.p95_scoring_ms; None -> unlimited
    sidecar: Optional[str] = None  # JSONL path; None -> records are only kept in memory (agent.records)
    run_id: str = ""
    sim_id: Optional[str] = None  # None -> random per agent instance
    parser: str = "rule"
    model_rev: str = ""
    lora_rev: str = ""
    dedupe: bool = True  # controller: candidates with the same (kind, actions) are scored once (first sample wins)
    top_k: int = 5  # posterior_topk length
    config_sha256: str = ""  # set from the launcher payload; logged in the sidecar
    filter_factory: Optional[Callable[..., Any]] = None  # (cfg, persona) -> ParticleFilter-like; tests inject small supports
    repair_invalid_user_tools: bool = False  # W3-F1: an agent call to a PHONE tool becomes a request to the user (both agents); off = LLMAgent identity
    pick: str = "boed"  # controller: "boed" = Controller.decide; "first_new" = NO scoring: first candidate whose device/tool actions were not
    # emitted earlier this episode (else the first candidate). Baseline that isolates loop-avoidance from BOED scoring.
    intent_prune_below: float = 0.05  # W3-F1: an INFORMATIVE first user message drops intents with prob below this from the filter support (0 = keep all)


_PRUNE_DEFAULT = 0.05
_DEFAULTS: dict[str, Any] = {}
_FIELDS = {f.name for f in dataclasses.fields(BoedSettings)}


def configure(**kw: Any) -> None:
    """Process-wide defaults for every boed agent built afterwards (llm_args `boed_*` keys still win)."""
    bad = set(kw) - _FIELDS
    if bad:
        raise ValueError(f"unknown boed setting(s) {sorted(bad)}; valid: {sorted(_FIELDS)}")
    _DEFAULTS.update(kw)


def reset_configuration() -> None:
    _DEFAULTS.clear()


def resolve_settings(llm_args: Optional[Mapping[str, Any]]) -> tuple[BoedSettings, dict]:
    """(settings, llm_args without `_boed` / boed_ keys). Precedence (low -> high): configure() defaults, the launcher's `_boed`
    payload (register.split_boed_args: agent_kwargs, boed_config, run, boed_config_sha256), explicit `boed_<field>` llm_args."""
    raw, payload = split_boed_args(dict(llm_args or {}))
    own = {k[5:]: raw.pop(k) for k in list(raw) if k.startswith("boed_")}
    viap: dict = dict(payload.get("agent_kwargs") or {})
    if payload.get("boed_config"):
        viap.setdefault("config", payload["boed_config"])
    if payload.get("run"):
        viap.setdefault("run_id", payload["run"])
    if payload.get("boed_config_sha256"):
        viap["config_sha256"] = payload["boed_config_sha256"]
    merged = {**_DEFAULTS, **viap, **own}
    bad = set(merged) - _FIELDS
    if bad:
        raise ValueError(f"unknown boed setting(s) {sorted(bad)}; valid: {sorted(_FIELDS)}")
    return BoedSettings(**merged), raw


@lru_cache(maxsize=8)
def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- json helpers
def _action_dict(a: Any) -> dict:
    return {"requestor": a.requestor, "tool": a.tool, "args": a.args_dict(), "ensure": list(a.ensure) if a.ensure else None}


def cand_dict(c: CandidateAction) -> dict:
    return {"kind": c.kind.value, "actions": [_action_dict(a) for a in c.actions], "candidate_idx": c.candidate_idx, "text": c.text}


def decision_dict(d: Decision, position: Optional[int] = None) -> dict:
    return {"chosen": cand_dict(d.chosen), "chosen_position": position, "scores": {str(k): v for k, v in d.scores},
            "scorer": d.scorer, "overridden": d.overridden, "reason": d.reason}


def _jsonable(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=repr)
    if isinstance(o, DecisionKind):
        return o.value
    return str(o)


def _default_sidecar() -> Optional[str]:
    """$TELECOM_RUN_DIR/sidecar.jsonl when the launcher set a run dir (one file per run; writes are locked)."""
    d = os.environ.get("TELECOM_RUN_DIR")
    return str(Path(d) / "sidecar.jsonl") if d else None


def _write_record(path: Optional[str], rec: SidecarRecord) -> None:
    if not path:
        return
    line = json.dumps(dataclasses.asdict(rec), default=_jsonable) + "\n"
    with _SIDECAR_LOCK:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(line)


# --------------------------------------------------------------------------- intent from the first user message (W3-F1)
# Fallback cues used ONLY when the rule parser (belief/parsers.py RuleIntentParser) found none: its mobile_data cue needs "data ... slow" inside one
# sentence, but Qwen-user openers read "trouble with my mobile data. It's either not working or very slow." (all 5 smoke-train tasks scored uniform).
_FALLBACK_CUES: dict[str, re.Pattern] = {
    "mobile_data_issue": re.compile(r"\b(?:mobile|cellular)\s+(?:data|internet)\b|\binternet\s+speeds?\b|\bdata\s+speeds?\b|\bslow\s+(?:internet|data)\b|"
                                    r"\b(?:internet|data)\b.{0,60}\b(?:slow|not working|stops? working)\b"),
}


def parse_intent(parsers: Any, text: str) -> tuple[dict, bool]:
    """(intent_probs, informative). Informative = at least one intent cue fired (the rule parser's, else the fallback cues above)."""
    probs = dict(parsers.intent.parse(text))
    vals = list(probs.values())
    if vals and max(vals) - min(vals) > 1e-9:
        return probs, True
    t = (text or "").lower().replace("\u2019", "'")
    hits = {i: len(rx.findall(t)) for i, rx in _FALLBACK_CUES.items()}
    hits = {i: min(float(hits.get(i, 0)), 3.0) for i in probs}
    tot = sum(hits.values())
    if tot == 0:
        return probs, False
    a, fl = 0.1, 0.02  # same smoothing as RuleIntentParser
    raw = {i: max((h + a) / (tot + a * len(hits)), fl) for i, h in hits.items()}
    z = sum(raw.values())
    return {i: v / z for i, v in raw.items()}, True


# --------------------------------------------------------------------------- per-episode BOED state
class _Episode:
    """Filter + belief + parsers + bookkeeping for ONE episode. Everything that can fail is wrapped by the callers."""

    def __init__(self, st: BoedSettings, llm: str, llm_args: Mapping[str, Any], policy: str, tool_names: tuple):
        self.st, self.llm, self.llm_args, self.tool_names = st, llm, llm_args, tool_names
        self.sim_id = st.sim_id or uuid.uuid4().hex[:12]
        self.prompt_hash = hashlib.sha1(policy.encode()).hexdigest()[:16]
        self.turn = 0
        self.belief: Optional[Belief] = None
        self.last: Optional[CandidateAction] = None  # what we emitted last turn
        self.emitted_keys: set = set()  # (kind, actions) of emitted candidates with actions (pick="first_new")
        self.pending: dict[str, str] = {}  # tool_call id -> backend tool name
        self.post_transfer = False
        self.intent_informative = False  # False while the belief still carries the uniform-intent prior
        self.records: list[SidecarRecord] = []
        self.init_error: Optional[str] = None
        self.flt = self.parsers = self.ctrl = None
        self.cfg: dict = {}
        try:
            self.cfg = _load_cfg(str(st.config or paths.CONFIG_DIR / "boed_default.yaml"))
            self.parsers = build_parsers(st.parser)
            fac = st.filter_factory or (lambda cfg, persona: ParticleFilter("deployable", cfg, persona))
            self.flt = fac(self.cfg, st.persona)
        except Exception as e:  # BOED path disabled for this episode; the agent behaves like LLMAgent
            self.init_error = f"episode_init: {type(e).__name__}: {e}"

    # ---- config
    @property
    def n_candidates(self) -> int:
        return int(self.st.n_candidates if self.st.n_candidates is not None else self.cfg["controller"]["n_candidates"])

    @property
    def max_tokens(self) -> Optional[int]:
        m = self.st.candidate_max_tokens
        return m if m is not None else self.cfg.get("controller", {}).get("candidate_max_tokens")

    @property
    def temperature(self) -> float:
        t = self.st.candidate_temperature
        return float(t if t is not None else self.cfg["controller"]["candidate_temperature"])

    def controller(self, override: str) -> Controller:
        if self.ctrl is None:
            c = self.cfg["controller"]
            self.ctrl = Controller(self.st.scorer or c["scorer"], self.flt, self.flt.grader, self.cfg, override=override,
                                   time_budget_ms=self.st.time_budget_ms)
        return self.ctrl

    # ---- incoming message -> belief
    def ingest(self, message: Any, info: dict, errors: list) -> None:
        if message is None:
            return
        for m in (message.tool_messages if isinstance(message, MultiToolMessage) else [message]):
            try:
                if isinstance(m, UserMessage):
                    self._ingest_user(m, info)
                elif isinstance(m, ToolMessage) and m.requestor == "assistant":
                    self._ingest_tool(m, info)
            except Exception as e:
                errors.append(f"ingest {type(m).__name__}: {type(e).__name__}: {e}")
                self.last = None  # that update is lost; keep the previous belief
                if self.belief is None:  # filter.init failed: BOED stays off for this episode (no retries on later user texts)
                    self.init_error = f"init: {type(e).__name__}: {e}"

    def label_space(self, cand: CandidateAction) -> dict:
        """Observation label space = the values the CURRENT PARTICLES can show for the requested actions (filter's expansion; the
        live env is never consulted). Reuses the expansion `update` needs next, so it is not extra work."""
        fn = getattr(self.flt, "label_space", None)
        if fn is not None:
            return fn(self.belief, cand)
        exp = self.flt._expand(self.belief, cand)
        out: dict[str, list] = {}
        for grp in exp.groups.values():
            for i, fname in enumerate(grp.fields):
                vals = out.setdefault(fname, [])
                for tup in grp.tuples:
                    if tup[i] is not None and tup[i] not in vals:
                        vals.append(tup[i])
        return out

    def _ingest_user(self, m: UserMessage, info: dict) -> None:
        text = m.content or ""
        if self.belief is None:
            probs, self.intent_informative = parse_intent(self.parsers, text)
            info["intent_probs"] = {k: round(v, 4) for k, v in probs.items()}
            info["intent_informative"] = self.intent_informative
            self.belief = self.flt.init(self._pruned(probs) if self.intent_informative else probs)
            self.last = None
            return
        if not self.intent_informative:  # first message carried no cue (e.g. a greeting): the first informative one reweights the prior
            probs, self.intent_informative = parse_intent(self.parsers, text)
            if self.intent_informative:
                info["intent_probs"] = {k: round(v, 4) for k, v in probs.items()}
                info["intent_reweighted"] = True
                self.belief = self._reweight(self.belief, self._pruned(probs))
        last, self.last = self.last, None
        if last is None or last.kind == DecisionKind.AGENT_TOOL:
            return  # unsolicited text or a backend call that never got its result: no observation
        if last.kind == DecisionKind.REQUEST_DEVICE:
            obs = self.parsers.observation.parse(text, last.actions, self.label_space(last))
            info["obs"] = {"source": obs.source, "fields": dict(obs.fields), "mentioned": sorted(obs.mentioned),
                           "conflicts": [list(c) for c in getattr(obs, "conflicts", ())],
                           "out_of_space": [list(c) for c in getattr(obs, "out_of_space", ())]}
        else:
            obs = Observation(last.actions, "user_report", raw=text)
        self.belief = self.flt.update(self.belief, last, obs)

    @property
    def prune_below(self) -> float:
        """BoedSettings.intent_prune_below, unless the BOED yaml sets agent.intent_prune_below (configs/boed_v2.yaml: 0.1) and the setting is
        still the dataclass default (explicit agent_kwargs / boed_ llm_args win). Configs without an `agent:` block behave as before."""
        y = (self.cfg.get("agent") or {}).get("intent_prune_below")
        return float(y) if y is not None and self.st.intent_prune_below == _PRUNE_DEFAULT else self.st.intent_prune_below

    def _pruned(self, probs: Mapping[str, float]) -> dict:
        keep = {k: v for k, v in probs.items() if v >= self.prune_below}
        return keep or dict(probs)

    @staticmethod
    def _reweight(b: Belief, probs: Mapping[str, float]) -> Belief:
        """Bayes: scale each particle by new_intent_prob / old_intent_prob (intents absent from `probs` are dropped), renormalise.
        The user's message is a likelihood factor on the intent only; the filter state (evidence absorbed so far) is kept."""
        old = dict(b.intent_probs)
        z = sum(probs.values())
        new = {k: v / z for k, v in probs.items()}
        ps = [Particle(p.hyp_key, p.state, p.weight * new[p.hyp_key[0]] / max(old.get(p.hyp_key[0], 0.0), 1e-12), p.exec_history)
              for p in b.particles if p.hyp_key[0] in new]
        tot = sum(p.weight for p in ps)
        if not ps or tot <= 0.0:
            return b
        ps = [Particle(p.hyp_key, p.state, p.weight / tot, p.exec_history) for p in ps]
        return dataclasses.replace(b, particles=tuple(ps), intent_probs=_frozen_items(new))

    def _ingest_tool(self, m: ToolMessage, info: dict) -> None:
        name = self.pending.pop(m.id, None)
        last, self.last = self.last, None
        if self.belief is None or last is None or name is None:
            return
        if last.kind == DecisionKind.TRANSFER:
            self.belief = self.flt.update(self.belief, last, Observation(last.actions, "agent_tool", raw=m.content))
            self.post_transfer = True
            info["obs"] = {"source": "agent_tool", "tool": name, "transfer": True}
        elif last.kind == DecisionKind.AGENT_TOOL:
            fields = {} if m.error else featurize(name, m.content or "")
            obs = Observation(last.actions, "agent_tool", _frozen_items(fields), frozenset(fields), raw=m.content)
            info["obs"] = {"source": "agent_tool", "tool": name, "error": bool(m.error), "n_fields": len(fields)}
            self.belief = self.flt.update(self.belief, last, obs)

    # ---- invalid phone-tool repair (W3-F1)
    def repair(self, msg: AssistantMessage, idx: Optional[int], parse: dict) -> tuple[AssistantMessage, Optional[CandidateAction]]:
        """(msg, None) unless `repair_invalid_user_tools` is on and msg calls a phone tool we do not own; then (text request, its candidate)."""
        if not self.st.repair_invalid_user_tools:
            return msg, None
        r = repair_user_tool_call(msg, idx, self.tool_names)
        if r is None:
            return msg, None
        parse.setdefault("repaired", []).append(r[2])
        return r[0], r[1]

    # ---- emitted message bookkeeping
    def note_emitted(self, msg: AssistantMessage, cand: Optional[CandidateAction]) -> None:
        for tc in msg.tool_calls or []:
            self.pending[tc.id] = tc.name
        self.last = None if (self.post_transfer and not msg.tool_calls) else cand
        if cand is not None and cand.actions:
            self.emitted_keys.add((cand.kind, cand.actions))
        if cand is not None and self.ctrl is not None and hasattr(self.ctrl.scorer, "mark_done"):
            self.ctrl.scorer.mark_done(cand)

    # ---- sidecar
    def belief_info(self) -> dict:
        b = self.belief
        if b is None:
            return {}
        out = {"n_particles": len(b.particles), "v": b.v, "kappa": b.kappa, "cost": b.cost, "transferred": b.transferred,
               "degenerate": b.degenerate, "step": b.step, "intent_probs": dict(b.intent_probs)}
        try:
            out["p_repaired"] = p_repaired(b, self.flt.grader)
        except Exception as e:
            out["p_repaired_error"] = f"{type(e).__name__}: {e}"
        return out

    def topk(self) -> list:
        if self.belief is None:
            return []
        agg: dict = {}
        for k, w in self.flt.marginal(self.belief).items():
            fs = tuple(sorted(k[1]))
            agg[fs] = agg.get(fs, 0.0) + w
        return [[list(k), v] for k, v in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))[: self.st.top_k]]

    def record(self, *, scores: dict, decision: dict, parse: dict, timings: dict) -> SidecarRecord:
        self.turn += 1
        try:
            parse = {**parse, "belief": self.belief_info(), "cfg": self.cfg.get("name", ""), "cfg_sha256": self.st.config_sha256}
            top = self.topk()
        except Exception as e:
            parse = {**parse, "error": [*parse.get("error", []), f"sidecar: {type(e).__name__}: {e}"]}
            top = []
        rec = SidecarRecord(run_id=self.st.run_id, sim_id=self.sim_id, turn=self.turn, posterior_topk=top, scores=scores,
                            decision=decision, parse=parse, timings={k: round(v, 6) for k, v in timings.items()},
                            seeds={"llm_seed": self.llm_args.get("seed"), "candidate_temperature": self.st.candidate_temperature},
                            model_rev=self.st.model_rev or self.llm, lora_rev=self.st.lora_rev, prompt_hash=self.prompt_hash)
        self.records.append(rec)
        try:
            _write_record(self.st.sidecar or _default_sidecar(), rec)
        except Exception:  # a full disk must not kill the episode
            pass
        return rec


# --------------------------------------------------------------------------- state + agents
class BoedAgentState(LLMAgentState):
    """LLMAgentState + the per-episode BOED state (not serialised)."""
    boed: Any = Field(default=None, exclude=True)


class _BoedBase(LLMAgent):
    """Common construction. `boed_*` llm_args are consumed here and never reach litellm."""

    def __init__(self, tools: list, domain_policy: str, llm: str, llm_args: Optional[dict] = None):
        self.boed_settings, clean = resolve_settings(llm_args)
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=clean)
        self._last_ep: Optional[_Episode] = None

    def get_init_state(self, message_history: Optional[list] = None) -> BoedAgentState:
        base = super().get_init_state(message_history)
        st = BoedAgentState.model_construct(system_messages=base.system_messages, messages=base.messages)
        st.boed = _Episode(self.boed_settings, self.llm, self.llm_args, self.domain_policy, tuple(t.name for t in self.tools))
        return st

    def _episode(self, state: Any) -> _Episode:
        ep = getattr(state, "boed", None)
        if ep is None:  # state built by someone else (e.g. resumed run): create lazily
            ep = _Episode(self.boed_settings, self.llm, self.llm_args, self.domain_policy, tuple(t.name for t in self.tools))
            try:
                state.boed = ep
            except Exception:
                pass
        return ep

    @property
    def records(self) -> list:
        """SidecarRecords of the most recent episode started by this agent instance."""
        return self._last_ep.records if self._last_ep is not None else []


class BoedPlainAgent(_BoedBase):
    """tau2 LLMAgent + shadow BOED sidecar. Emitted messages are exactly LLMAgent's."""

    def get_init_state(self, message_history: Optional[list] = None) -> BoedAgentState:
        st = super().get_init_state(message_history)
        self._last_ep = st.boed
        return st

    def generate_next_message(self, message: Any, state: Any) -> tuple[AssistantMessage, Any]:
        ep = self._episode(state)
        self._last_ep = ep
        t0 = time.perf_counter()
        msg, state = super().generate_next_message(message, state)  # identical to LLMAgent; may raise like LLMAgent
        llm_s = time.perf_counter() - t0
        rep: dict = {}
        if self.boed_settings.repair_invalid_user_tools:  # control condition of boed-controller+repair: same rewrite, no scoring
            try:
                new, cand = ep.repair(msg, 0, rep)
                if cand is not None:
                    state.messages[-1] = msg = new  # LLMAgent already appended the original to the history
                    rep["_cand"] = cand
            except Exception:
                rep.pop("repaired", None)
        try:
            self._shadow(ep, message, msg, llm_s, rep)
        except Exception as e:  # last line of defence: the shadow can never change the episode
            try:
                ep.record(scores={}, decision={}, parse={"error": [f"shadow: {type(e).__name__}: {e}"]}, timings={"llm": llm_s})
            except Exception:
                pass
        return msg, state

    def _shadow(self, ep: _Episode, message: Any, msg: AssistantMessage, llm_s: float, rep: Optional[dict] = None) -> None:
        parse: dict = {"error": []}
        rep = dict(rep or {})
        rcand = rep.pop("_cand", None)
        if rep:
            parse.update(rep)
        tm = {"llm": llm_s, "filter": 0.0, "parse": 0.0, "score": 0.0}
        scores: dict = {}
        decision: dict = {}
        if ep.init_error:
            parse["error"].append(ep.init_error)
        else:
            t = time.perf_counter()
            ep.ingest(message, parse, parse["error"])
            tm["filter"] = time.perf_counter() - t
            emitted: Optional[CandidateAction] = None
            try:
                t = time.perf_counter()
                emitted, einfo = to_candidate_info(msg, ep.parsers, 0, ep.tool_names)
                if rcand is not None:  # the repaired request carries the exact DeviceAction (the text parser could re-read it differently)
                    emitted = rcand
                tm["parse"] = time.perf_counter() - t
                parse["emitted"] = einfo
            except Exception as e:
                parse["error"].append(f"parse_emitted: {type(e).__name__}: {e}")
            if ep.belief is not None and emitted is not None and not ep.post_transfer:
                try:
                    t = time.perf_counter()
                    cands = [emitted] if emitted.kind not in (DecisionKind.CLAIM_RESOLVED, DecisionKind.TRANSFER) else []
                    pos = 0 if cands else None
                    for term in (DecisionKind.CLAIM_RESOLVED, DecisionKind.TRANSFER):
                        if emitted.kind == term:
                            pos = len(cands)
                        cands.append(emitted if emitted.kind == term else CandidateAction(term))
                    ctrl = ep.controller("none")
                    dec = ctrl.decide(ep.belief, cands)
                    tm["score"] = time.perf_counter() - t
                    scores = {dec.scorer: {str(k): v for k, v in dec.scores}}
                    cpos = next((i for i, c in enumerate(cands) if c is dec.chosen), None)
                    decision = {**decision_dict(dec, cpos), "mode": "shadow", "emitted_position": pos,
                                "emitted": cand_dict(emitted), "stats": dict(ctrl.stats)}
                except Exception as e:
                    parse["error"].append(f"score: {type(e).__name__}: {e}")
            ep.note_emitted(msg, emitted)
        if not parse["error"]:
            parse.pop("error")
        ep.record(scores=scores, decision=decision, parse=parse, timings=tm)


class BoedControllerAgent(_BoedBase):
    """Samples N candidates, lets the BOED controller choose."""

    def get_init_state(self, message_history: Optional[list] = None) -> BoedAgentState:
        st = super().get_init_state(message_history)
        self._last_ep = st.boed
        return st

    def generate_next_message(self, message: Any, state: Any) -> tuple[AssistantMessage, Any]:
        ep = self._episode(state)
        self._last_ep = ep
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)
        msgs = state.system_messages + state.messages
        parse: dict = {"error": []}
        tm = {"llm": 0.0, "filter": 0.0, "parse": 0.0, "score": 0.0}
        scores: dict = {}
        decision: dict = {}
        turn: dict = {"samples": None}
        out: Optional[AssistantMessage] = None
        emitted: Optional[CandidateAction] = None
        try:
            out, emitted = self._turn(ep, message, msgs, parse, tm, scores, decision, turn)
        except Exception as e:
            parse["error"].append(f"boed: {type(e).__name__}: {e}")
        if out is None:  # failure ladder: first sampled candidate, else one greedy generate() call (= LLMAgent)
            samples = turn["samples"]
            if samples:
                out = samples[0]
                parse["fallback"] = parse.get("fallback") or "first_candidate"
            else:
                parse["fallback"] = "greedy_generate"
                t = time.perf_counter()
                out = _la.generate(model=self.llm, tools=self.tools, messages=msgs, call_name="agent_response", **self.llm_args)
                tm["llm"] += time.perf_counter() - t
            out, rcand = ep.repair(out, 0 if samples else None, parse)
            emitted = None
            if not ep.init_error and ep.belief is not None:
                try:
                    emitted = rcand or to_candidate_info(out, ep.parsers, 0 if samples else None, ep.tool_names)[0]
                except Exception as e:
                    parse["error"].append(f"parse_fallback: {type(e).__name__}: {e}")
        ep.note_emitted(out, emitted)
        if not parse["error"]:
            parse.pop("error")
        ep.record(scores=scores, decision=decision, parse=parse, timings=tm)
        state.messages.append(out)
        return out, state

    def _turn(self, ep: _Episode, message: Any, msgs: list, parse: dict, tm: dict, scores: dict, decision: dict, turn: dict):
        """-> (message to emit or None, its candidate). Stores the samples in `turn` before anything can fail."""
        boed_on = not ep.init_error
        t = time.perf_counter()
        samples = sample_candidates(self.llm, msgs, self.tools, ep.n_candidates if boed_on else 1,
                                    ep.temperature if boed_on else float(self.llm_args.get("temperature") or 0.0), self.llm_args,
                                    max_tokens=ep.max_tokens if boed_on else None)
        tm["llm"] = time.perf_counter() - t
        turn["samples"] = samples
        parse["sampling"] = samples.stats()
        if not boed_on:
            parse["error"].append(ep.init_error)
            return None, None
        t = time.perf_counter()
        ep.ingest(message, parse, parse["error"])
        tm["filter"] = time.perf_counter() - t
        if ep.init_error or ep.belief is None or ep.post_transfer or not samples:
            return None, None
        t = time.perf_counter()
        cands: list[CandidateAction] = []
        infos: list[dict] = []
        seen: set = set()
        dropped = {"invalid_tool": 0, "duplicate": 0}
        for i, m in enumerate(samples):
            m, rcand = ep.repair(m, i, parse)
            samples[i] = m  # _emit / the first-candidate fallback must see the repaired message
            cand, info = to_candidate_info(m, ep.parsers, i, ep.tool_names)
            if rcand is not None:
                cand, info["repaired"] = rcand, True
            infos.append(info)
            if info.get("invalid_tool"):
                dropped["invalid_tool"] += 1
            elif ep.st.dedupe and (cand.kind, cand.actions) in seen:
                dropped["duplicate"] += 1
                info["duplicate"] = True
            else:
                seen.add((cand.kind, cand.actions))
                cands.append(cand)
        tm["parse"] = time.perf_counter() - t
        parse["candidates"], parse["dropped"] = infos, dropped
        if not cands:
            parse["fallback"] = "no_scorable_candidate"
            return None, None
        if ep.st.pick == "first_new":  # baseline: same samples/repair/filter, no BOED scoring
            ch = next((c for c in cands if not (c.actions and (c.kind, c.actions) in ep.emitted_keys)), cands[0])
            decision.update(mode="first_new", chosen=cand_dict(ch), position=cands.index(ch),
                            repeat=bool(ch.actions and (ch.kind, ch.actions) in ep.emitted_keys),
                            scored_candidate_idx=[x.candidate_idx for x in cands])
            return self._emit(ch, samples), ch
        if ep.st.pick != "boed":
            raise ValueError(f"pick must be 'boed' or 'first_new', got {ep.st.pick!r}")
        ctrl = ep.controller(ep.st.override or ep.cfg["controller"]["override"])
        t = time.perf_counter()
        dec = ctrl.decide(ep.belief, cands)
        tm["score"] = time.perf_counter() - t
        scores[dec.scorer] = {str(k): v for k, v in dec.scores}
        decision.update(decision_dict(dec, next((i for i, x in enumerate(cands) if x is dec.chosen), None)))
        decision.update(mode="controller", override=ctrl.override, stats=dict(ctrl.stats),
                        scored_candidate_idx=[x.candidate_idx for x in cands], template=dec.chosen.candidate_idx is None)
        return self._emit(dec.chosen, samples), dec.chosen

    @staticmethod
    def _emit(ch: CandidateAction, samples: Samples) -> AssistantMessage:
        """The tau2 message for the chosen candidate; the whole completion's cost/usage ride on it."""
        meta = {"cost": samples.cost, "usage": samples.usage, "generation_time_seconds": samples.seconds}
        if ch.candidate_idx is not None:
            return samples[ch.candidate_idx].model_copy(update=meta)
        if ch.kind == DecisionKind.AGENT_TOOL:  # template backend call (override "full")
            a = ch.actions[0]
            call = ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=a.tool, arguments=a.args_dict(), requestor="assistant")
            return AssistantMessage(role="assistant", content=None, tool_calls=[call], **meta)
        return AssistantMessage(role="assistant", content=ch.text or "", **meta)  # template text (e.g. the verify request)


# --------------------------------------------------------------------------- factories + registration
def _factory(cls: type) -> Callable[..., Any]:
    def make(tools: list, domain_policy: str, llm: Optional[str] = None, llm_args: Optional[dict] = None, task: Any = None,
             **kwargs: Any) -> Any:
        del task, kwargs  # never read the episode's task; audio_native_config / audio_taps_dir are irrelevant to text agents
        return cls(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)

    make.__name__ = f"make_{cls.__name__}"
    return make


FACTORIES: dict[str, Callable[..., Any]] = {PLAIN: _factory(BoedPlainAgent), CONTROLLER: _factory(BoedControllerAgent)}


def register(registry: Any = None) -> list[str]:
    """Register `boed-plain` and `boed-controller` in the tau2 registry (default: tau2.registry.registry). Idempotent."""
    if registry is None:
        from tau2.registry import registry
    names: list[str] = []
    for name, fac in FACTORIES.items():
        try:
            registry.register_agent_factory(fac, name)
        except ValueError:
            if registry.get_agent_factory(name) is not fac:
                raise
        names.append(name)
    return names
