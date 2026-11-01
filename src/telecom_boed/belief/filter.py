"""[telecom_boed] Exact Bayes filter over the real tau2 particle support + the one-step Predictor (owner: W2-S5).

DEPLOYABLE module (testing/leakage_scan.py DEPLOYABLE_MODULES): imports no `oracle_*` module, never touches a task or a user DB.
Particle states are reached only through env.particles (apply_calls, grade) and env.featurize. Oracle mode receives its
exact tool contents from the CALLER as `Observation(source="tool")`.

Model (simulator-as-likelihood).  One REQUEST_DEVICE candidate with user-side actions a_1..a_n:
- Execution branches (belief/user_model.py): each a_j is not_done / done / done_twice with (1-q-eps_double, q, eps_double),
  and with prob eps_extra the user also runs one unrequested `reboot_device` after the requested actions (it resets the APN,
  user_tools.py:955-958). A branch's call list is executed on the particle's state with env.particles.apply_calls.
  `DeviceAction.ensure=(field, value)`: if the particle's current value (read with a cheap read tool, user_model.ENSURE_READ)
  already equals `value`, a done / done_twice branch compiles to NO state change; the user just looks at the screen (the
  report then comes from the primary group's read tool, user_model.PRIMARY_READ). Ensure is evaluated on the state at the
  START of the request. AGENT_TOOL candidates run exactly once, no branching. ASK changes no state.
- Posterior weight of item (particle i, branch b): w_i * pi_b * L(o | i, b); items merge on (post-state key, goal_id), the
  contracts merge rule. Weights below `filter.prune_eps` (default 1e-12) are dropped; zero total weight -> `degenerate=True`
  carrying the PRIOR particles (the input belief's, unchanged) -- never NaN.

Likelihoods L(o | i, b):
- oracle (source "tool") and agent_tool: the observation `fields` must equal the featurize() of the branch's executed outputs
  (merged over the requested actions, later wins; the unrequested extra is not part of it): L in {0, 1}. A not-done branch has
  no output (fields == ()). The agent sees backend output exactly.
- user_report (deployable): report model per output GROUP (group = prefix of the featurize() field name). Within group g with
  label space T_g = the distinct value tuples the group takes over all (particle, branch) outputs in the current support:
    P(report tuple r | true tuple t) = (1 - eps_conf) * [r == t] + eps_conf / |T_g|
  If the tool did not run (not_done) the true tuple does not exist: `report_when_not_done = uniform` gives 1/|T_g|;
  `claims_done` uses the tuple the particle's all-done branch would show. A report mentioning only a subset M of the group's
  fields uses the projection: P(r_M | t) = (1 - eps_conf) * [r_M == t_M] + eps_conf * |{s in T_g : s_M = r_M}| / |T_g|.
  A reported value outside the label space is treated as uninformative (L = 1) instead of zero. The P(mentioned) factors
  (mention_p) are common to every item, so they cancel in the posterior; they matter only for outcome probabilities in predict.
  Groups the user did not mention contribute nothing.

predict(belief, cand) -> {ObsKey: (p, beta')} (contracts.Predictor).  The lookahead observation space is abstract:
- REQUEST_DEVICE, deployable: for each PRIMARY group g of the request (user_model.PRIMARY_GROUPS): either not mentioned
  (prob 1 - m_g) or mentioned in full with a tuple r in T_g (prob m_g * P(r)). Several groups multiply (independent mentions).
  The ObsKey is the Observation.key() of exactly the Observation `update` would receive for that outcome, so
  update(b, c, obs) == predict(b, c)[obs.key()][1] (tests/test_filter.py::test_predict_update_consistency). Both call the
  same posterior routine.
- REQUEST_DEVICE, oracle: the full featurized output (exact, as for backend tools).
- AGENT_TOOL: one outcome per distinct full featurized output.
- ASK: one outcome ("ask",), cost only. CLAIM_RESOLVED: ("claim",), belief unchanged (as in the toy POMDP).
- TRANSFER: ("transfer",), transferred=True; kappa is set iff some particle does not require transfer, i.e.
  "kappa if transfer and P(requires_transfer) < 1": transferring a fixable fault violates the policy (contracts.py).
beta' fields: cost += decide.utility.action_cost (REQUEST_DEVICE / ASK -> turn_cost, AGENT_TOOL -> tool_cost); kappa |= risk()[1]
(evaluated on the pre-action belief); transferred; step + 1; v per below.

v rule (contracts.py): a mutating action (any requested tool that is not ToolType.READ) resets v before the observation is
applied. Then v becomes True iff the observation is non-null and LR = P(o | repaired) / P(o | not repaired) >= reach_avoid.v_lr,
with P(o | class) = sum_{k in class} W_k L_k(o) / sum_{k in class} W_k over the post-action classes (W_k = weight before the
observation) and repaired = grade(state, hyp, belief.transferred) == 1 and not requires_transfer. If a class has zero mass: no
repaired mass -> v unchanged; no not-repaired mass -> v iff the observation is non-null and has L > 0 on repaired. A null
report (nothing mentioned) never sets v; a verified v is never cleared by a later non-mutating observation (literal contract).

Performance (W2-P2, no semantic change; originals in tests/fixtures/W2-P2/orig/): `_expand_raw` is array-based. States get small
integer ids; per slot-tuple ("variant") a transition table maps state id -> (final state id, output-structure id, final Snapshot), filled
lazily through the stepper and reused across beliefs, so expanding a belief costs O(#particles) dict lookups plus numpy (item order,
class order, hence every float sum, is that of the former per-item loops). Posterior `Particle`s are built without the frozen-dataclass
`__init__` (same fields, same values). Output-structure / label-space bookkeeping is cached per distinct set of structures.

Caching: single-call results are cached by (state key, call) in a StepCache-like dict (cap STEP_CACHE_CAP entries, cleared when
exceeded). Read-only calls store only the contents (their post-state IS the pre-state, so no new Snapshot lineage is created);
write calls store the Snapshot returned by env.particles.apply_calls. The parent's parent_id / step on cached Snapshots follow the
first state that computed them. Lineage fidelity is not needed by the filter.
"""
from __future__ import annotations

import contextlib
import gc
import itertools
from collections import OrderedDict
from operator import itemgetter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from telecom_boed.belief.stepper import ParallelStepper
from telecom_boed.belief.user_model import (
    ENSURE_READ, EXTRA_ACTION, PRIMARY_GROUPS, PRIMARY_READ, UserModel, group_of, is_mutating_tool, reportable)
from telecom_boed.contracts import (
    Belief, CandidateAction, DecisionKind, DeviceAction, Observation, ObsKey, Particle, Snapshot, _frozen_items)
from telecom_boed.decide.risk import risk
from telecom_boed.decide.utility import action_cost
from telecom_boed.env import particles as P
from telecom_boed.env.featurize import featurize
from telecom_boed.env.hypotheses import hypothesis_by_key, load_support

STEP_CACHE_CAP = 250_000
MAX_COMBOS = 20_000  # per-request branch-combination bound (2*3^n for n user actions: n <= 8 passes); larger raises ValueError
_EXP_CACHE_SIZE = 8
_PRED_CACHE_SIZE = 64
CLAIM_KEY: ObsKey = ("claim",)
TRANSFER_KEY: ObsKey = ("transfer",)
ASK_KEY: ObsKey = ("ask",)


# --------------------------------------------------------------------------- expansion records
@dataclass
class _Group:
    """Label space of one output group over the current support."""
    name: str
    fields: tuple  # ordered field names of the tuple
    tuples: list  # distinct value tuples, deterministic order
    item_tid: np.ndarray  # per item: index into `tuples` of the tuple the user would report from, or -1 (uniform)


@dataclass
class _Exp:
    """One belief x one candidate, expanded to items (particle, branch) with their merge class."""
    n_cls: int
    cls_state: list
    cls_hyp: list
    cls_hist: list
    cls: np.ndarray  # item -> class
    wt: np.ndarray  # item prior weight w_i * pi_b
    fid: np.ndarray  # item -> exact-feature id (index into ftab)
    ftab: list  # distinct frozen exact feature tuples
    groups: dict  # group name -> _Group
    req_groups: tuple  # primary groups of the request that have a label space
    mutating: bool
    W: np.ndarray = field(default=None)  # per-class weight before the observation
    _rep: Optional[np.ndarray] = None
    _kappa: Optional[bool] = None  # risk(cand, belief)[1], computed once per (belief, candidate)

    def __post_init__(self) -> None:
        self.W = np.bincount(self.cls, weights=self.wt, minlength=self.n_cls)


class _Trans:
    """Transition table of one call variant (slot tuple): per start-state id -> final state id, output-structure id, final Snapshot.
    fin < 0 = not computed yet; snap None = the variant only reads (the final state is the start state object itself)."""
    __slots__ = ("fin", "gid", "snap")

    def __init__(self) -> None:
        self.fin = np.zeros(0, dtype=np.int64) - 1
        self.gid = np.zeros(0, dtype=np.int64)
        self.snap = np.empty(0, dtype=object)

    def grow(self, n: int) -> None:
        m = len(self.fin)
        if n > m:
            c = max(n, 2 * m, 256)
            self.fin = np.concatenate([self.fin, np.full(c - m, -1, dtype=np.int64)])
            self.gid = np.concatenate([self.gid, np.zeros(c - m, dtype=np.int64)])
            self.snap = np.concatenate([self.snap, np.empty(c - m, dtype=object)])


_new, _setd = object.__new__, object.__setattr__


@contextlib.contextmanager
def _gc_paused():
    """Pause the cyclic GC while a predict / update builds its posterior Particles (tens of thousands of acyclic objects per call, on
    top of a heap of millions of Snapshot dict nodes): per-posterior young-generation passes cascade into repeated full passes, which
    cost more than the construction itself. The state is restored on exit (no forced collection); nested use is a no-op."""
    on = gc.isenabled()
    if on:
        gc.disable()
    try:
        yield
    finally:
        if on:
            gc.enable()


def _particles(hyps: list, states: list, ws: list, hists: list) -> tuple:
    """tuple(Particle(h, s, w, hh) ...) without the dataclass __init__ call chain (same fields and values, same object layout)."""
    out = []
    ap = out.append
    for h, s, w, hh in zip(hyps, states, ws, hists):
        p = _new(Particle)
        _setd(p, "hyp_key", h); _setd(p, "state", s); _setd(p, "weight", w); _setd(p, "exec_history", hh)
        ap(p)
    return tuple(out)


def _gather(lst: list, ix: list) -> list:
    return list(itemgetter(*ix)(lst)) if len(ix) > 1 else [lst[i] for i in ix]


class ParticleFilter:
    """contracts.Filter + contracts.Predictor over the real tau2 support. See module docstring for the model."""

    def __init__(self, mode: str, cfg: Mapping[str, Any], persona: str = "default",
                 stepper: Optional[ParallelStepper] = None, grader: Optional[P.ParticleGrader] = None):
        if mode not in ("oracle", "deployable"):
            raise ValueError(f"mode must be 'oracle' or 'deployable', got {mode!r}")
        self.mode = mode
        self.cfg = cfg
        self.persona = persona
        self.um = UserModel.from_cfg(cfg, persona)
        self.v_lr = float(cfg["reach_avoid"]["v_lr"])
        self.prune_eps = float(cfg.get("filter", {}).get("prune_eps", 1e-12))
        self.stepper = stepper if stepper is not None else ParallelStepper(1)
        # P0-2 (boed_v2): cfg["perf"] switches; absent -> defaults (everything off, old behaviour). abort_check: optional callable the
        # controller installs; it raises when the anytime deadline has passed and is polled between stepper waves (never mid-wave).
        P.configure_perf(**{**P.PERF_DEFAULT, **{k: v for k, v in (cfg.get("perf") or {}).items() if k in P.PERF_DEFAULT}})
        self.abort_check = None
        self.grader = grader if grader is not None else P.ParticleGrader()
        self._step_cache: dict = {}
        self._cids: dict = {}
        self._vids: dict = {}
        self._feats: dict = {}
        self._ecache: OrderedDict = OrderedDict()
        self._pcache: OrderedDict = OrderedDict()
        self._repcache: dict = {}  # (state key, hyp_key, transferred) -> repaired
        self._skid: dict = {}  # state key -> small int id
        self._goal_ix: dict = {}  # goal_id -> small int
        self._trans: dict = {}  # variant id -> _Trans
        self._sreg: dict = {}  # (variant id, outputs) -> structure id
        self._structs: list = []  # structure id -> (act dict, noop dict, frozen exact features, act groups, noop groups)
        self._proto: dict = {}  # tuple of structure ids -> (ftab, s_fid, {group: (fields, distinct tuples, s_tid)})

    # ------------------------------------------------------------------ Filter protocol
    def init(self, intent_probs: Mapping[str, float]) -> Belief:
        """Support = union of the load_support(intent) of every intent with prob > 0; prior weight =
        intent_prob / |support_intent| (uniform within an intent). Probabilities are renormalised to sum to 1."""
        probs = {k: float(v) for k, v in intent_probs.items() if float(v) > 0.0}
        if not probs:
            raise ValueError("intent_probs has no positive entry")
        z = sum(probs.values())
        probs = {k: v / z for k, v in probs.items()}
        parts: list[Particle] = []
        for intent, pr in probs.items():
            hyps = load_support(intent)
            if not hyps:
                raise KeyError(f"no hypotheses for intent {intent!r}")
            parts.extend(P.initial_particles(hyps, [pr / len(hyps)] * len(hyps)))
        merged: dict = {}
        for p in parts:
            k = p.merge_key(hypothesis_by_key(p.hyp_key).goal_id)
            merged[k] = p if k not in merged else Particle(merged[k].hyp_key, merged[k].state, merged[k].weight + p.weight,
                                                           merged[k].exec_history)
        return Belief(tuple(merged.values()), intent_probs=_frozen_items(probs))

    def update(self, belief: Belief, action: CandidateAction, obs: Observation) -> Belief:
        kind = action.kind
        if kind in (DecisionKind.CLAIM_RESOLVED, DecisionKind.TRANSFER, DecisionKind.ASK):
            return self._terminal_or_ask(belief, action)[1]
        self._check_source(action, obs)
        with _gc_paused():
            exp = self._expand(belief, action)
            if obs.source == "user_report":
                lik, informative = self._lik_report(exp, obs)
            else:
                lik, informative = self._lik_exact(exp, obs), bool(obs.fields)
            return self._posterior(exp, belief, action, lik, informative)

    # ------------------------------------------------------------------ Predictor protocol
    def predict(self, belief: Belief, action: CandidateAction) -> dict:
        kind = action.kind
        if kind in (DecisionKind.CLAIM_RESOLVED, DecisionKind.TRANSFER, DecisionKind.ASK):
            key, b2 = self._terminal_or_ask(belief, action)
            return {key: (1.0, b2)}
        ck = (id(belief), action)
        hit = self._pcache.get(ck)
        if hit is not None and hit[0] is belief:
            return hit[1]
        with _gc_paused():
            exp = self._expand(belief, action)
            if kind == DecisionKind.AGENT_TOOL or self.mode == "oracle":
                out = self._predict_exact(exp, belief, action)
            else:
                out = self._predict_report(exp, belief, action)
        self._pcache[ck] = (belief, out)
        while len(self._pcache) > _PRED_CACHE_SIZE:
            self._pcache.popitem(last=False)
        return out

    # ------------------------------------------------------------------ helpers
    def marginal(self, belief: Belief) -> dict:
        """hyp_key -> weight. Particles merged on (state, goal_id) keep only their first member's hyp_key."""
        out: dict = {}
        for p in belief.particles:
            out[p.hyp_key] = out.get(p.hyp_key, 0.0) + p.weight
        return out

    def clear_caches(self) -> None:
        self._step_cache.clear(); self._feats.clear(); self._ecache.clear(); self._pcache.clear(); self._repcache.clear(); self._clear_tables()

    def _clear_tables(self) -> None:
        self._skid.clear(); self._trans.clear(); self._sreg.clear(); self._structs.clear(); self._proto.clear()  # tables hold Snapshots: they go with the step cache

    def _check_source(self, cand: CandidateAction, obs: Observation) -> None:
        if cand.kind == DecisionKind.AGENT_TOOL:
            if obs.source not in ("agent_tool", "tool"):
                raise ValueError(f"AGENT_TOOL expects source agent_tool, got {obs.source!r}")
            if obs.source == "tool" and self.mode != "oracle":
                raise ValueError("source='tool' is oracle-only")
        elif self.mode == "oracle":
            if obs.source != "tool":
                raise ValueError(f"oracle mode expects source='tool' for a device request, got {obs.source!r}")
        elif obs.source != "user_report":
            raise ValueError(f"deployable mode expects source='user_report' for a device request, got {obs.source!r}")

    def _terminal_or_ask(self, belief: Belief, cand: CandidateAction) -> tuple:
        if cand.kind == DecisionKind.CLAIM_RESOLVED:
            return CLAIM_KEY, belief
        if cand.kind == DecisionKind.TRANSFER:
            fixable = any(not hypothesis_by_key(p.hyp_key).requires_transfer for p in belief.particles)
            return TRANSFER_KEY, Belief(belief.particles, belief.v, belief.kappa or fixable, belief.cost, True,
                                        belief.intent_probs, belief.step + 1)
        return ASK_KEY, Belief(belief.particles, belief.v, belief.kappa, belief.cost + action_cost(cand, self.cfg),
                               belief.transferred, belief.intent_probs, belief.step + 1)

    # ------------------------------------------------------------------ cached single-call stepping
    def _feat(self, tool: str, content: str) -> dict:
        k = (tool, content)
        f = self._feats.get(k)
        if f is None:
            f = self._feats[k] = featurize(tool, content)
        return f

    def _cid(self, call: DeviceAction) -> int:
        i = self._cids.get(call)
        if i is None:
            i = self._cids[call] = len(self._cids)
        return i

    def _advance(self, states: Sequence[Snapshot], call: DeviceAction) -> list:
        """[(post-state Snapshot or None if the call is read-only, content)] aligned with `states`. Misses go to the stepper
        in ONE batch (unique state keys only)."""
        if self.abort_check is not None:
            self.abort_check()
        cid = self._cid(call)
        cache = self._step_cache
        read_only = not is_mutating_tool(call.tool)
        missing: dict = {}
        for s in states:
            if (s.key, cid) not in cache and s.key not in missing:
                missing[s.key] = s
        if missing:
            calls = (call,)
            res = self.stepper.step_many(list(missing.values()), calls, outputs_only=read_only)
            for s, (snap, outs) in zip(missing.values(), res):
                cache[(s.key, cid)] = (None if read_only else snap, outs[0])
                P._APPLY_MEMO.pop((s.key, calls), None)  # keep ONE copy of each result (the cache above)
        return [cache[(s.key, cid)] for s in states]

    def _prefetch(self, work: list) -> None:
        """Compute every (state, call) pair the work items will need, one stepper dispatch per call depth (all variants together), so
        a parallel stepper sees whole waves instead of one call at a time. work = [(slots, states)]. Pure optimisation: results equal
        lazy `_advance`."""
        cache = self._step_cache
        cur = [list(sl) for _, sl in work]
        depth = max((len(slots) for slots, _ in work), default=0)
        for d in range(depth):
            if self.abort_check is not None:
                self.abort_check()
            need: dict = {}
            for k, (slots, _) in enumerate(work):
                if d >= len(slots):
                    continue
                call = slots[d][0]
                cid = self._cid(call)
                for s in cur[k]:
                    ck = (s.key, cid)
                    if ck not in cache and ck not in need:
                        need[ck] = (s, call)
            if need:
                calls_obj: dict = {}
                items = []
                for (_, _), (s, call) in need.items():
                    cc = calls_obj.setdefault(call, (call,))
                    items.append((s, cc, not is_mutating_tool(call.tool)))
                res = self.stepper.step_pairs(items)
                for ck, (s, call), (snap, outs), it in zip(need, need.values(), res, items):
                    cache[ck] = (None if it[2] else snap, outs[0])
                    P._APPLY_MEMO.pop((s.key, it[1]), None)
            for k, (slots, _) in enumerate(work):
                if d < len(slots):
                    cid = self._cid(slots[d][0])
                    cur[k] = [cache[(s.key, cid)][0] or s for s in cur[k]]

    # ------------------------------------------------------------------ expansion
    def _expand(self, belief: Belief, cand: CandidateAction) -> _Exp:
        ck = (id(belief), cand)
        hit = self._ecache.get(ck)
        if hit is not None and hit[0] is belief:
            return hit[1]
        if len(self._step_cache) > STEP_CACHE_CAP:
            self._step_cache.clear(); self._clear_tables()
        exp = self._expand_raw(belief, cand)
        self._ecache[ck] = (belief, exp)
        while len(self._ecache) > _EXP_CACHE_SIZE:
            self._ecache.popitem(last=False)
        return exp

    def _combos(self, cand: CandidateAction) -> tuple:
        """([(branches, extra, pi, emit)], proxy index or None). branches[j] in nd | d | dd."""
        um, acts = self.um, cand.actions
        user_branch = cand.kind == DecisionKind.REQUEST_DEVICE
        opts = []
        for a in acts:
            if user_branch and a.requestor == "user":
                opts.append([(b, pr) for b, pr in (("nd", um.p_not_done), ("d", um.q), ("dd", um.eps_double)) if pr > 0.0])
            else:
                opts.append([("d", 1.0)])
        has_user = user_branch and any(a.requestor == "user" for a in acts)
        extra_opts = [(False, 1.0 - um.eps_extra), (True, um.eps_extra)] if has_user else [(False, 1.0)]
        extra_opts = [(e, pr) for e, pr in extra_opts if pr > 0.0]
        n_combos = len(extra_opts)
        for o in opts:
            n_combos *= len(o)
        if n_combos > MAX_COMBOS:  # a 31-action bundle (model echoing its prompt) = 2*3^31 combos: the list below filled 50+ GB
            raise ValueError(f"request bundle of {len(acts)} actions expands to {n_combos} branch combos > MAX_COMBOS={MAX_COMBOS}")
        combos = []
        for choice in itertools.product(*opts, extra_opts):
            *brs, (extra, pe) = choice
            pi = pe
            for _, pr in brs:
                pi *= pr
            combos.append((tuple(b for b, _ in brs), extra, pi, True))
        all_done = (tuple("d" for _ in acts), False)
        proxy = next((i for i, c in enumerate(combos) if (c[0], c[1]) == all_done), None)
        if proxy is None and self.um.report_when_not_done == "claims_done" and user_branch:
            combos.append((all_done[0], False, 0.0, False))
            proxy = len(combos) - 1
        return combos, proxy

    def _sid(self, key: tuple) -> int:
        i = self._skid.get(key)
        if i is None:
            i = self._skid[key] = len(self._skid)
        return i

    def _tr(self, vid: int) -> _Trans:
        tr = self._trans.get(vid)
        if tr is None:
            tr = self._trans[vid] = _Trans()
        return tr

    def _struct_id(self, vid: int, slots_t: tuple, outs: tuple) -> int:
        """Output structure of one executed variant: (exact features of the requested calls, features of the 'just look' reads)."""
        sk = (vid, outs)
        g = self._sreg.get(sk)
        if g is None:
            act: dict = {}
            noop: dict = {}
            for (call, role, _j), o in zip(slots_t, outs):
                if role == "act":
                    act.update(self._feat(call.tool, o))
                elif role == "noop":
                    noop.update(self._feat(call.tool, o))
            g = self._sreg[sk] = len(self._structs)
            self._structs.append((act, noop, _frozen_items(act), {group_of(f) for f in act if reportable(f)},
                                  {group_of(f) for f in noop if reportable(f)}))
        return g

    def _fill(self, tr: _Trans, vid: int, slots_t: tuple, uids: np.ndarray, starts: list) -> None:
        """Run the variant's call chain on `starts` (one state per start-state id in `uids`) and record the results in `tr`."""
        cur = list(starts)
        outs_acc: list = [[] for _ in cur]
        wrote = [False] * len(cur)
        for call, _role, _j in slots_t:
            for t, (sn, o) in enumerate(self._advance(cur, call)):
                outs_acc[t].append(o)
                if sn is not None:
                    cur[t] = sn
                    wrote[t] = True
        for t, uid in enumerate(uids.tolist()):
            tr.fin[uid] = self._sid(cur[t].key)
            tr.gid[uid] = self._struct_id(vid, slots_t, tuple(outs_acc[t]))
            tr.snap[uid] = cur[t] if wrote[t] else None

    def _expand_raw(self, belief: Belief, cand: CandidateAction) -> _Exp:
        parts = belief.particles
        n = len(parts)
        acts = cand.actions
        states = [p.state for p in parts]
        wvec = np.array([p.weight for p in parts], dtype=float)
        goal_of: dict = {}
        for p in parts:
            if p.hyp_key not in goal_of:
                goal_of[p.hyp_key] = self._goal_ix.setdefault(hypothesis_by_key(p.hyp_key).goal_id, len(self._goal_ix))
        gix = np.array([goal_of[p.hyp_key] for p in parts], dtype=np.int64)
        uid = np.array([self._sid(s.key) for s in states], dtype=np.int64)  # start-state ids
        combos, proxy_ix = self._combos(cand)
        user_branch = cand.kind == DecisionKind.REQUEST_DEVICE  # exec_history records device requests only

        # ensure satisfaction (state at the start of the request) and the "just look at the screen" read per action
        sat: dict = {}
        noop_call: dict = {}
        for j, a in enumerate(acts):
            if a.ensure is not None and a.ensure[0] in ENSURE_READ:
                rtool, ffield = ENSURE_READ[a.ensure[0]]
                res = self._advance(states, DeviceAction("user", rtool))
                sat[j] = np.array([self._feat(rtool, o).get(ffield) == a.ensure[1] for _, o in res], dtype=bool)
                g0 = PRIMARY_GROUPS.get(a.tool, (None,))[0]
                rt = PRIMARY_READ.get(g0)
                noop_call[j] = DeviceAction("user", rt) if rt else None

        plans: list = []  # (combo index, particle index array, slots tuple, exec label)
        for ci, (branches, extra, pi, emit) in enumerate(combos):
            rel = [j for j, b in enumerate(branches) if b in ("d", "dd") and j in sat]
            if rel:
                codes = np.zeros(n, dtype=np.int64)
                for bit, j in enumerate(rel):
                    codes |= sat[j].astype(np.int64) << bit
                splits = [(int(c), np.nonzero(codes == c)[0]) for c in np.unique(codes)]
            else:
                splits = [(0, np.arange(n))]
            if all(b == "nd" for b in branches) and not extra:
                label = "not_done"
            elif extra:
                label = "extra"
            elif any(b == "dd" for b in branches):
                label = "done_twice"
            else:
                label = "done"
            for code, ix in splits:
                slots: list = []
                for j, (a, b) in enumerate(zip(acts, branches)):
                    if b == "nd":
                        continue
                    if j in rel and (code >> rel.index(j)) & 1:
                        if noop_call.get(j) is not None:
                            slots.append((noop_call[j], "noop", j))
                        continue
                    slots.extend([(a, "act", j)] * (2 if b == "dd" else 1))
                if extra:
                    slots.append((EXTRA_ACTION, "extra", -1))
                plans.append((ci, ix, tuple(slots), label))

        # transition tables: compute only the (start state, variant) pairs not seen before
        trs: list = []
        todo: list = []
        n_ids = len(self._skid)
        for k, (_ci, ix, slots_t, _label) in enumerate(plans):
            vid = self._vids.setdefault(slots_t, len(self._vids))
            tr = self._tr(vid)
            tr.grow(n_ids)
            trs.append((vid, tr))
            ui = uid[ix]
            miss = tr.fin[ui] < 0
            if miss.any():
                mu, first = np.unique(ui[miss], return_index=True)
                todo.append((k, mu, [states[i] for i in ix[miss][first].tolist()]))
        if todo:
            self._prefetch([(plans[k][2], st) for k, _mu, st in todo])
            for k, mu, st in todo:
                self._fill(trs[k][1], trs[k][0], plans[k][2], mu, st)

        # per-plan arrays, concatenated in plan order (= the former item order: plan-major, particle ascending)
        fin_p, gid_p = [], []
        for (_ci, ix, _s, _l), (_vid, tr) in zip(plans, trs):
            ui = uid[ix]
            fin_p.append(tr.fin[ui]); gid_p.append(tr.gid[ui])
        gid_cat = np.concatenate(gid_p) if gid_p else np.zeros(0, dtype=np.int64)
        uq, first = np.unique(gid_cat, return_index=True)
        order = uq[np.argsort(first, kind="stable")]  # global structure ids in order of first appearance = local structure ids
        loc = np.zeros(int(order.max()) + 1 if len(order) else 1, dtype=np.int64)
        loc[order] = np.arange(len(order))
        proxy_sid = np.full(n, -1, dtype=np.int64)
        em = [k for k, pl in enumerate(plans) if combos[pl[0]][3]]
        for k, pl in enumerate(plans):
            if pl[0] == proxy_ix:
                proxy_sid[pl[1]] = loc[gid_p[k]]
        pix = np.concatenate([plans[k][1] for k in em]) if em else np.zeros(0, dtype=np.int64)
        fin_all = np.concatenate([fin_p[k] for k in em]) if em else np.zeros(0, dtype=np.int64)
        sid_arr = loc[np.concatenate([gid_p[k] for k in em])] if em else np.zeros(0, dtype=np.int64)
        wt = np.concatenate([wvec[plans[k][1]] * combos[plans[k][0]][2] for k in em]) if em else np.zeros(0)
        lens = np.array([len(plans[k][1]) for k in em], dtype=np.int64)

        # classes: items merge on (final state key, goal_id); class ids in order of first appearance; the class takes the hyp_key /
        # exec_history / Snapshot of its first item
        code = fin_all * max(len(self._goal_ix), 1) + gix[pix]
        uqc, firstc, inv = np.unique(code, return_index=True, return_inverse=True)
        o = np.argsort(firstc, kind="stable")
        rank = np.empty(len(uqc), dtype=np.int64)
        rank[o] = np.arange(len(uqc))
        cls = rank[inv.reshape(-1)]
        reps = firstc[o]
        snap_all = np.concatenate([trs[k][1].snap[uid[plans[k][1]]] for k in em]) if em else np.empty(0, dtype=object)
        plan_of = np.repeat(np.arange(len(em)), lens)[reps].tolist() if len(reps) else []
        rp = pix[reps].tolist()
        cls_state = [states[i] if sn is None else sn for i, sn in zip(rp, snap_all[reps].tolist())]
        cls_hyp = [parts[i].hyp_key for i in rp]
        lab = [(plans[k][3],) if user_branch else () for k in em]
        cls_hist = [parts[i].exec_history + lab[e] for i, e in zip(rp, plan_of)]

        # exact feature ids, report label spaces: cached per distinct set of output structures
        pk = tuple(order.tolist())
        proto = self._proto.get(pk)
        if proto is None:
            proto = self._proto[pk] = self._make_proto(order.tolist())
            while len(self._proto) > 256:
                self._proto.pop(next(iter(self._proto)))
        ftab, s_fid, gprotos = proto
        claims = self.um.report_when_not_done == "claims_done"
        groups: dict = {}
        for g, (fs, distinct, s_tid) in gprotos.items():
            item_tid = s_tid[sid_arr] if len(sid_arr) else np.zeros(0, dtype=np.int64)
            if claims and len(sid_arr):
                prox = proxy_sid[pix]
                fallback = np.where(prox >= 0, s_tid[np.maximum(prox, 0)], -1)
                item_tid = np.where(item_tid >= 0, item_tid, fallback)
            groups[g] = _Group(g, fs, distinct, item_tid)

        req: list = []
        for a in acts:
            pg = PRIMARY_GROUPS.get(a.tool)
            for g in (pg if pg is not None else tuple(groups)):
                if g in groups and g not in req:
                    req.append(g)
        return _Exp(len(cls_state), cls_state, cls_hyp, cls_hist, cls, wt, s_fid[sid_arr] if len(sid_arr) else sid_arr,
                    ftab, groups, tuple(req), any(is_mutating_tool(a.tool) for a in acts))

    def _make_proto(self, gids: list) -> tuple:
        """(ftab, structure -> exact-feature id, {group: (fields, distinct label tuples, structure -> label id)}) for the structures `gids`."""
        structs = [self._structs[g] for g in gids]
        ftab: list = []
        findex: dict = {}
        s_fid = np.empty(len(structs), dtype=np.int64)
        for s, st in enumerate(structs):
            f = st[2]
            j = findex.get(f)
            if j is None:
                j = findex[f] = len(ftab)
                ftab.append(f)
            s_fid[s] = j
        by_group: dict = {}
        for act, noop, *_ in structs:
            for d in (act, noop):
                for f in d:
                    if reportable(f):
                        by_group.setdefault(group_of(f), set()).add(f)
        gprotos: dict = {}
        for g in sorted(by_group):
            fs = tuple(sorted(by_group[g]))
            raw: list = []
            for act, noop, _f, ga, gn in structs:
                src = act if g in ga else noop if g in gn else None
                raw.append(None if src is None else tuple(src.get(f) for f in fs))
            distinct = sorted({t for t in raw if t is not None}, key=repr)
            idx = {t: i for i, t in enumerate(distinct)}
            gprotos[g] = (fs, distinct, np.array([-1 if t is None else idx[t] for t in raw], dtype=np.int64))
        return ftab, s_fid, gprotos

    # ------------------------------------------------------------------ likelihoods
    def _group_lik(self, grp: _Group, match: np.ndarray) -> Optional[np.ndarray]:
        """Per-item likelihood of a report that matches the label tuples flagged in `match` (bool over grp.tuples).
        None = uninformative (no label tuple matches: out-of-space value)."""
        t = len(grp.tuples)
        if t == 0 or not match.any():
            return None
        eps = self.um.eps_conf
        u = float(match.sum()) / t
        tid = grp.item_tid
        hit = match[np.maximum(tid, 0)].astype(float)
        return np.where(tid >= 0, (1.0 - eps) * hit + eps * u, u)

    def _lik_report(self, exp: _Exp, obs: Observation) -> tuple:
        n = len(exp.wt)
        lik = np.ones(n)
        fields = dict(obs.fields)
        informative = False
        for g, grp in exp.groups.items():
            idxs = [i for i, f in enumerate(grp.fields) if f in obs.mentioned and f in fields]
            if not idxs:
                continue
            informative = True
            r = tuple(fields[grp.fields[i]] for i in idxs)
            match = np.array([tuple(tup[i] for i in idxs) == r for tup in grp.tuples], dtype=bool)
            gl = self._group_lik(grp, match)
            if gl is not None:
                lik = lik * gl
        return lik, informative

    def _lik_exact(self, exp: _Exp, obs: Observation) -> np.ndarray:
        target = obs.fields
        for j, f in enumerate(exp.ftab):
            if f == target:
                return (exp.fid == j).astype(float)
        return np.zeros(len(exp.wt))

    # ------------------------------------------------------------------ posterior + v
    def _repaired(self, exp: _Exp, belief: Belief) -> np.ndarray:
        """Per class: grade(state, hyp, transferred) == 1 and not requires_transfer(hyp). Pure in (state key, hyp, transferred): cached."""
        if exp._rep is None:
            g, tr, rc = self.grader, belief.transferred, self._repcache
            if len(rc) > STEP_CACHE_CAP:
                rc.clear()
            keys = [(s.key, h, tr) for s, h in zip(exp.cls_state, exp.cls_hyp)]
            miss = [i for i, k in enumerate(keys) if k not in rc]
            if miss:
                self._prefetch_grades([exp.cls_state[i] for i in miss], [exp.cls_hyp[i] for i in miss], tr)
                for i in miss:
                    rc[keys[i]] = g.grade(exp.cls_state[i], exp.cls_hyp[i], tr) == 1 and not g.requires_transfer(exp.cls_hyp[i])
            exp._rep = np.array([rc[k] for k in keys], dtype=bool)
        return exp._rep

    def _prefetch_grades(self, states: list, hyps: list, transferred: bool) -> None:
        """Grade the not-yet-memoised classes through the stepper's workers (only if it has any) and install the values in
        env.particles' grade memo, so every later grader.grade() call (here and in the scorers) is a dict hit."""
        if self.stepper.n_workers <= 1:
            return
        todo = []
        for s, hk in zip(states, hyps):
            h = hypothesis_by_key(hk)
            if ("ACTION" in h.reward_basis and not transferred) or (s.key, h.goal_id) in P._GRADE_MEMO:
                continue
            todo.append((s, hk, transferred, h.goal_id))
        if todo:
            vals = self.stepper.grade_many([(s, hk, tr) for s, hk, tr, _ in todo])
            for (s, _hk, _tr, gid), v in zip(todo, vals):
                P._GRADE_MEMO[(s.key, gid)] = v

    def _v_after(self, exp: _Exp, belief: Belief, num: np.ndarray, informative: bool) -> bool:
        # contract v rule (v1, clarified 2026-10-16): LR >= v_lr sets v; LR <= 1/v_lr clears it (strong contrary
        # evidence, same as the toy POMDP's ("repaired", False)); otherwise v is unchanged; mutating actions reset it.
        v = False if exp.mutating else belief.v
        if not informative:
            return v
        rep = self._repaired(exp, belief)
        mass_rep, mass_nr = float(exp.W[rep].sum()), float(exp.W[~rep].sum())
        if mass_rep <= 0.0:
            return False
        p_rep = float(num[rep].sum()) / mass_rep
        p_nr = float(num[~rep].sum()) / mass_nr if mass_nr > 0.0 else 0.0
        if p_nr <= 0.0:
            return p_rep > 0.0
        lr = p_rep / p_nr
        if lr >= self.v_lr:
            return True
        if lr <= 1.0 / self.v_lr:
            return False
        return v

    def _posterior(self, exp: _Exp, belief: Belief, cand: CandidateAction, lik: np.ndarray, informative: bool) -> Belief:
        num = np.bincount(exp.cls, weights=exp.wt * lik, minlength=exp.n_cls)
        if belief.kappa:
            kappa = True
        else:
            if exp._kappa is None:
                exp._kappa = bool(risk(cand, belief)[1])
            kappa = exp._kappa
        cost = belief.cost + action_cost(cand, self.cfg)
        total = float(num.sum())
        if not total > 0.0:
            return Belief(belief.particles, False if exp.mutating else belief.v, kappa, cost, belief.transferred,
                          belief.intent_probs, belief.step + 1, degenerate=True)
        v = self._v_after(exp, belief, num, informative)
        pw = num / total
        keep = np.nonzero(pw >= self.prune_eps)[0]
        w = pw[keep]
        w = w / w.sum()
        ks = keep.tolist()
        ps = _particles(_gather(exp.cls_hyp, ks), _gather(exp.cls_state, ks), w.tolist(), _gather(exp.cls_hist, ks))
        return Belief(ps, v, kappa, cost, belief.transferred, belief.intent_probs, belief.step + 1)

    # ------------------------------------------------------------------ predict paths
    def _predict_exact(self, exp: _Exp, belief: Belief, cand: CandidateAction) -> dict:
        source = "agent_tool" if cand.kind == DecisionKind.AGENT_TOOL else "tool"
        tot = float(exp.wt.sum())
        out: dict = {}
        for j, f in enumerate(exp.ftab):
            lik = (exp.fid == j).astype(float)
            p = float((exp.wt * lik).sum()) / tot
            if not p > 0.0:
                continue
            obs = Observation(cand.actions, source, f, frozenset(k for k, _ in f))
            out[obs.key()] = (p, self._posterior(exp, belief, cand, lik, bool(f)))
        return out

    def _predict_report(self, exp: _Exp, belief: Belief, cand: CandidateAction) -> dict:
        tot = float(exp.wt.sum())
        gs = [exp.groups[g] for g in exp.req_groups]
        mp = [self.um.mention_p(g.name) for g in gs]
        per_group = []  # per group: [(prob factor, match vector or None, tuple or None)]
        for grp, m in zip(gs, mp):
            opts = [(1.0 - m, None, None)] if m < 1.0 else []
            if m > 0.0:
                for r, tup in enumerate(grp.tuples):
                    match = np.arange(len(grp.tuples)) == r
                    opts.append((m, match, tup))
            per_group.append(opts)
        out: dict = {}
        for choice in itertools.product(*per_group):
            lik = np.ones(len(exp.wt))
            factor, fields, mentioned = 1.0, {}, set()
            for grp, (pf, match, tup) in zip(gs, choice):
                factor *= pf
                if match is None:
                    continue
                gl = self._group_lik(grp, match)
                if gl is not None:
                    lik = lik * gl
                fields.update(zip(grp.fields, tup))
                mentioned.update(grp.fields)
            p = factor * float((exp.wt * lik).sum()) / tot
            if not p > 0.0:
                continue
            obs = Observation(cand.actions, "user_report", _frozen_items(fields), frozenset(mentioned))
            out[obs.key()] = (p, self._posterior(exp, belief, cand, lik, bool(mentioned)))
        return out
