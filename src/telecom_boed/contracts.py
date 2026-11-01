"""[telecom_boed] Frozen shared types and interfaces for all work packages (owner: Opus / W0-O).

Agents never edit this file. If a WP needs a change it stops and reports; Opus bumps CONTRACT_VERSION.
Nothing here imports tau2 (see telecom_boed/__init__.py for the import rule).

Notation follows docs/proposal.md §3:
- h: hidden fault hypothesis.
- particle: (h, current env state).
- β = (b, v, κ, C) plus the `transferred` flag, which tau2's ACTION check needs.
- Utility U(r, h′) = grade(h′'s state, h, transferred). This is an exact copy of tau2's final reward
  (`evaluator/evaluator.py:223-256`): the product of ENV_ASSERTION and, when it is in reward_basis, ACTION.
- In telecom, ACTION means "transfer_to_human_agents was called at some point". It is a GENERIC tool,
  so the env state never shows it.

Verified flag v (eq. 4), as fixed here:
- v = 1 iff, since the last state-mutating action, the agent received an observation whose likelihood ratio
  P(o | repaired) / P(o | not repaired) >= cfg.v_lr.
- An intervention's own output counts only if the agent actually saw the confirming field. Agent-side tool
  results are always seen. For user-side tools, the user's reply must mention the field.
- An observation with LR <= 1 / v_lr clears v. This is strong evidence of "not repaired", e.g. a check showing
  the fault is still there. An observation with LR strictly between 1 / v_lr and v_lr leaves v unchanged.
- P(o | class) is belief-weighted within each class. If the post-action belief has zero mass on a class, v = 1
  iff the observation is non-null AND consistent only with "repaired" under the observation model. A null or
  unmentioned report never sets v.

Utility semantics (fixed here so that W2 scorers agree; c_FA, c_esc, λ, ρ, γ, δ, η, B come from the config):
- repaired(h′) means grade(state, h, transferred) == 1, with transferred = the belief's flag.
- Myopic / k-step U (eq. 2′):
  - U(claim_resolved, h′) = grade(h′, transferred) − c_FA · [grade == 0]
  - U(transfer, h′) = grade(h′, True) if h.requires_transfer, else −c_esc
- Reach-avoid (eq. 4, probability semantics):
  - Q(claim_resolved) = P_β(repaired ∧ v = 1)
  - Q(transfer) = P_β(h.requires_transfer ∧ grade(h′, True) == 1)
  - Transferring a fixable fault violates the τ² policy, so it sets κ = 1, which is failure (value 0).
  - This closes the loophole where tau2 still scores a transfer as success after a fix on ENV-only tasks.
  - Failure F: κ = 1, or C > B, or a wrong claim. All have value 0.
- Every request_device / agent_tool / ask costs λ · C(a): C = turn_cost for a message to the user, tool_cost for a
  backend call. R(a) comes from decide/risk.py, weighted by ρ.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, runtime_checkable

CONTRACT_VERSION = 1

Intent = Literal["service_issue", "mobile_data_issue", "mms_issue"]
INTENTS: tuple[str, ...] = ("service_issue", "mobile_data_issue", "mms_issue")
Requestor = Literal["user", "assistant"]
HypKey = tuple[str, frozenset]  # (intent, faults); faults = frozenset of subtask names from the task id
StateKey = tuple[str, str, str]  # (agent_db_hash, user_db_hash, vpn_default_hash)
ObsKey = tuple  # hashable observation key; see Observation.key()
ExecBranch = Literal["not_done", "done", "done_twice", "extra"]  # how the user executed a device request


def _frozen_items(d: Optional[Mapping[str, Any]]) -> tuple:
    """Canonical hashable form of a JSON-like mapping (sorted keys, nested dicts and lists frozen)."""
    if not d:
        return ()

    def fz(v: Any) -> Any:
        if isinstance(v, Mapping):
            return tuple(sorted((k, fz(x)) for k, x in v.items()))
        if isinstance(v, (list, tuple)):
            return tuple(fz(x) for x in v)
        return v

    return fz(d)


# --------------------------------------------------------------------------- catalog (prior / support)
@dataclass(frozen=True)
class EnvCall:
    """Plain-data mirror of a tau2 EnvFunctionCall (fault injection) or EnvAssertion (goal check)."""
    env_type: Requestor
    func_name: str
    arguments: tuple = ()  # _frozen_items(arguments dict)
    assert_value: Optional[bool] = None  # set for assertions only

    def args_dict(self) -> dict:
        return {k: v for k, v in self.arguments}


@dataclass(frozen=True, eq=False)
class Hypothesis:
    """One shipped fault combination. Identity is `key`; other fields are catalog data read from tasks.json.
    Reading the catalog is allowed in deployable code (it is the prior). Reading the *current episode's* task is not."""
    key: HypKey
    task_ids: tuple[str, ...]
    init_actions: tuple[EnvCall, ...]
    goal_assertions: tuple[EnvCall, ...]
    reward_basis: tuple[str, ...]  # e.g. ("ENV_ASSERTION",) or ("ENV_ASSERTION", "ACTION")
    requires_transfer: bool  # True iff "ACTION" in reward_basis (gold action = transfer_to_human_agents)
    goal_id: str  # stable hash of (goal_assertions, reward_basis); particles merge on it
    is_test40: bool

    @property
    def intent(self) -> str:
        return self.key[0]

    @property
    def faults(self) -> frozenset:
        return self.key[1]

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Hypothesis) and other.key == self.key


# --------------------------------------------------------------------------- env state
@dataclass(frozen=True, eq=False)
class Snapshot:
    """Full restorable env state. Restore = fresh TelecomEnvironment, then direct DB assignment, then sync_tools()
    (never set_state(initialization_data=...), which cross-wires the two DBs). The VPN default object is included
    because tau2 keeps it in a mutable class attribute (user_tools.py:765,789)."""
    agent_db: Mapping[str, Any]  # TelecomDB.model_dump()
    user_db: Mapping[str, Any]  # TelecomUserDB.model_dump()
    vpn_default: Optional[Mapping[str, Any]]  # VpnDetails.model_dump() of the per-instance default
    key: StateKey
    snapshot_id: str = ""
    parent_id: Optional[str] = None
    step: int = 0
    messages: tuple = ()  # tau2 Message.model_dump() dicts, agent+user+tool, in order


@dataclass(frozen=True)
class DeviceAction:
    """One concrete tool call by the user (phone, 30 tools) or the agent (backend, 13 tools).
    `ensure` = optional (field, value) intent, e.g. ("airplane_mode", False). Toggles undo themselves, so planners
    reason in ensure-terms and compile them to the tool call that reaches the value from each particle's state."""
    requestor: Requestor
    tool: str
    args: tuple = ()  # _frozen_items(args dict)
    ensure: Optional[tuple[str, Any]] = None

    def args_dict(self) -> dict:
        return {k: v for k, v in self.args}


@dataclass(frozen=True, eq=False)
class Particle:
    hyp_key: HypKey
    state: Snapshot
    weight: float
    exec_history: tuple = ()  # ExecBranch per past device request (diagnostics)

    def merge_key(self, goal_id: str) -> tuple:
        return (self.state.key, goal_id)


# --------------------------------------------------------------------------- observations
@dataclass(frozen=True)
class Observation:
    """What the agent learned after one turn.
    - source="tool": oracle mode, the exact tool content (labels only; never at deployment).
    - source="agent_tool": a backend tool result, which the agent sees exactly (deployable).
    - source="user_report": the user's text reply as parsed. fields = values extracted for the fields in `mentioned`.
    `fields` uses featurize() names (W1-S2), e.g. "status_bar.airplane_mode", "speed_test.desc"."""
    actions: tuple[DeviceAction, ...]
    source: Literal["tool", "user_report", "agent_tool"]
    fields: tuple = ()  # _frozen_items({field: value})
    mentioned: frozenset = frozenset()
    raw: Optional[str] = None

    def key(self) -> ObsKey:
        return (self.source, self.actions, self.fields, self.mentioned)


# --------------------------------------------------------------------------- belief
@dataclass(frozen=True, eq=False)
class Belief:
    """β of eq. 4 over merged particles (weights sum to 1 unless degenerate)."""
    particles: tuple[Particle, ...]
    v: bool = False  # verified since the last mutating action (definition in module docstring)
    kappa: bool = False  # an irreversible or policy-violating action was taken
    cost: float = 0.0  # accumulated λC(a) units
    transferred: bool = False  # transfer_to_human_agents already called (feeds ACTION in grade)
    intent_probs: tuple = ()  # _frozen_items({intent: prob})
    step: int = 0
    degenerate: bool = False  # all weight vanished: flagged, never NaN


# --------------------------------------------------------------------------- decisions
class DecisionKind(str, enum.Enum):
    REQUEST_DEVICE = "request_device"  # ask the user to run/check something on the phone
    AGENT_TOOL = "agent_tool"  # call a backend tool
    ASK = "ask"  # free-text question (no device action)
    CLAIM_RESOLVED = "claim_resolved"  # tell the user the issue is fixed (the "commit" r)
    TRANSFER = "transfer"  # transfer_to_human_agents (escalate)


@dataclass(frozen=True)
class CandidateAction:
    """A pre-parsed candidate. Source: one of Qwen's N sampled messages (candidate_idx), or a template (None)."""
    kind: DecisionKind
    actions: tuple[DeviceAction, ...] = ()
    candidate_idx: Optional[int] = None
    text: Optional[str] = None


@dataclass(frozen=True)
class Decision:
    chosen: CandidateAction
    scores: tuple = ()  # _frozen_items({candidate position: score})
    scorer: str = ""
    overridden: bool = False  # True when filter_commit replaced an unverified claim
    reason: str = ""


# --------------------------------------------------------------------------- interfaces
@runtime_checkable
class Predictor(Protocol):
    """Simulator-as-likelihood one-step model (W2-S5): p(o, β′ | β, do(a)) over the particle support."""

    def predict(self, belief: Belief, action: CandidateAction) -> dict[ObsKey, tuple[float, Belief]]: ...


@runtime_checkable
class Grader(Protocol):
    """Lets scorers stay independent of tau2. The real implementation is env.particles (W1-S2); toy POMDPs in tests
    implement it too. A Predictor must return β′ with cost, v, kappa and transferred already updated."""

    def grade(self, state: Snapshot, hyp_key: HypKey, transferred: bool) -> float: ...

    def requires_transfer(self, hyp_key: HypKey) -> bool: ...


@runtime_checkable
class Filter(Protocol):
    mode: Literal["oracle", "deployable"]

    def init(self, intent_probs: Mapping[str, float]) -> Belief: ...

    def update(self, belief: Belief, action: CandidateAction, obs: Observation) -> Belief: ...


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, belief: Belief, candidates: Sequence[CandidateAction]) -> list[float]: ...


@runtime_checkable
class ActionParser(Protocol):
    def parse(self, agent_text: str) -> tuple[DeviceAction, ...]: ...


@runtime_checkable
class ObservationParser(Protocol):
    def parse(self, user_text: str, requested: tuple[DeviceAction, ...], label_space: Mapping[str, Sequence[Any]]) -> Observation: ...


@runtime_checkable
class ClaimClassifier(Protocol):
    def prob_claim(self, agent_text: str) -> float: ...


@runtime_checkable
class IntentParser(Protocol):
    def parse(self, first_user_text: str) -> dict[str, float]: ...


# Function signatures implemented by WPs (documented here, defined in their modules):
#   telecom_boed.env.featurize.featurize(tool: str, content: str) -> dict[str, Any]             (W1-S2)
#   telecom_boed.env.particles.grade(state: Snapshot, hyp: Hypothesis, transferred: bool) -> float  (W1-S2)
#   telecom_boed.env.snapshot.snapshot(env, ...) -> Snapshot ; restore(snap) -> env              (W1-S1)


# --------------------------------------------------------------------------- logging
@dataclass
class SidecarRecord:
    """One JSONL line per agent turn, written next to tau2's results.json (schema owner: contracts)."""
    run_id: str
    sim_id: str
    turn: int
    posterior_topk: list = field(default_factory=list)  # [(fault set as sorted list, prob), ...]
    scores: dict = field(default_factory=dict)  # scorer -> {candidate position: score}
    decision: dict = field(default_factory=dict)  # Decision as dict
    parse: dict = field(default_factory=dict)  # parser outputs for this turn
    timings: dict = field(default_factory=dict)  # seconds: filter, score, parse, llm
    seeds: dict = field(default_factory=dict)
    model_rev: str = ""
    lora_rev: str = ""
    prompt_hash: str = ""
    snapshot_parent: Optional[str] = None
    contract_version: int = CONTRACT_VERSION
