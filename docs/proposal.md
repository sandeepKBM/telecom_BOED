# Repair-relevant Bayesian experimental design and distributed counterfactual learning for telecom agents

Telecom agents under partial observability. Distributed Deep Learning project design. Sandeep S, October 2026.
(Source: the user's proposal document, pasted 2026-10-11, reformatted to markdown. Content unchanged except revisions
marked *Revision <date>*. The feasibility review and implementation plan are in `../PLAN.md`.)

## Executive recommendation
We will build a telecom troubleshooting agent that learns when to ask, diagnose, intervene, verify, retry, or escalate.
The project uses partial observability, state-changing tool actions, programmatic final-state grading, and
counterfactual simulator branches to study repair-relevant Bayesian experimental design. A100 GPUs support model
inference and learning, while multiple actors generate verified interactive rollouts for a learner. The course
implementation uses the class A100 resources; approximately 250 Perlmutter GPU-node hours are a separate resource for
larger experiments outside the course allocation.

| Layer | Primary contribution | Why it matters |
|---|---|---|
| Learning | Repair-relevant BOED objective over actions and observations | Optimizes the final repair, not generic uncertainty reduction. |
| Systems | Asynchronous multi-GPU branch generation and learning | Measures the throughput and compute efficiency of verified interactive rollouts. |
| Evaluation | Counterfactual branch, calibration, and deployment metrics | Tests repair quality, calibration, and interaction efficiency at matched budgets. |

**Project thesis.** A dual-control telecom agent should spend a turn only when the expected improvement in the best
repair decision exceeds the cost and risk of that turn. We learn this policy and evaluate how distributed rollout
generation and training affect time to quality.

**Contribution scope.**
- Learn decisions about asking, diagnosis, intervention, verification, and stopping using repair-relevant utility.
- Build counterfactual training data from simulator snapshots with programmatic final-state labels.
- Study distributed rollout generation and learning with explicit compute accounting. Use existing model-serving and
  tensor libraries for inference and decision scoring.

## 1 Dual control under hidden telecom state
The τ²-Bench telecom environment is naturally a partially observable, dual-control problem. The customer and the agent
possess different tools and information; their actions can change the shared world; and success is determined by the
final environment state rather than by a textual similarity score. The runtime exposes conversation history,
observations, and tool results, but not the hidden grader state.

Hidden state h → Belief bₜ(h) → Action do(a) → Observation o → Repair / verify r

| Element | Telecom instantiation | Research implication |
|---|---|---|
| Hidden state h | Fault, device configuration, account condition, previous action success, user compliance | A trace can look plausible while the true state remains wrong. |
| Observation o | User answer, diagnostic result, tool output, confirmation or contradiction | Observations differ in cost, noise, and decision relevance. |
| Design/action a | Ask, inspect, diagnose, intervene, reobserve, retry, verify, escalate | Some experiments also alter the system: p(o \| h, do(a)), not only p(o \| h, a). |
| Final decision r | Chosen repair, final configuration, or escalation | The goal is a correct repair under budget, not exact latent-state identification. |
| Utility U(r,h) | Task success minus turns, tool cost, unsafe actions, and false acceptance | Information is valuable only insofar as it improves the decision. |

**The failure mode the project targets.** Current agents often commit too early, keep asking after the repair is
already identifiable, or accept a superficially successful tool call without verifying the resulting state. A
trace-level recovery model can repair some failures after the fact, but the stronger formulation is prospective:
maintain a calibrated belief, estimate the value of the next intervention or observation, and stop when the posterior
already supports a robust repair.

**Controlled experimental substrate.** The telecom domain contains 2,285 controlled combinations across 15 subtask
groups and three intents, with 114 standard tasks in the current experimental plan. This finite support is unusually
useful: it permits exact or high-quality oracle computations for acquisition labels, counterfactual branching from
common prefixes, and direct calibration checks against known simulator state, while still preventing hidden state from
entering the deployed policy.

**Central research question.** Can repair-relevant Bayesian experimental design improve final-state success and
calibration at a fixed interaction budget, and can asynchronous rollout generation make counterfactual supervision
practical for Qwen-based SFT and optional GRPO?

## 2 Related work and research hypotheses
The proposal builds on related work in conversational dual control, Bayesian question selection, amortized information
gathering, decision utility gain, and branching policy optimization. We will evaluate the combination of
repair-relevant intervention value, active verification, counterfactual supervision, and distributed learning. The
research hypotheses require experimental validation; novelty should be assessed against the selected primary references.

| Existing capability | Already established | Gap used here |
|---|---|---|
| Dual-control benchmark | Agent and user act in a shared partially observable environment | Measure intervention value, verification, and posterior calibration, not benchmark success alone. |
| BED-LLM | Bayesian selection of informative questions at inference time | Include state-changing tool actions and downstream repair utility. |
| ASIG | Multi-turn GRPO with EIG reward | Train on repair-relevant utility gain, action risk, false acceptance, and robust user-model shifts. |
| Decision-aware BED | DUG optimizes downstream decisions | Evaluate repair utility across asks, diagnostics, interventions, and verification, with a standard reference implementation. |
| Branching RL | Counterfactual branches improve credit assignment | Exploit simulator snapshots and common-prefix branches with verified final-state labels. |
| Prefix caching | Hash-based reuse avoids repeated prompt prefill | Treat as infrastructure; test interaction with branching, but do not claim it as the core novelty. |

**Hypotheses.**
1. A repair-relevant objective will outperform entropy-only EIG at the same turn and tool budget because many latent
   distinctions do not change the optimal repair.
2. Explicit intervention models p(o | h, do(a)) will reduce harmful or redundant tool use relative to observational
   question selection.
3. Active verification will reduce false acceptance: episodes that appear complete in text but fail the programmatic
   final-state grader.
4. Counterfactual branch supervision will improve long-horizon credit assignment and sample efficiency for GRPO.
5. Asynchronous rollout generation with bounded policy staleness will increase verified trajectories per node-hour
   without degrading policy quality.

**Claim discipline.** Report policy quality and distributed-system efficiency separately. A throughput improvement
supports the systems claim only when it improves end-to-end verified rollout production or time to a fixed
policy-quality target. Use standard CPU and PyTorch implementations for decision scoring, and validate their outputs
against finite-hypothesis reference cases.

## 3 Repair-relevant BOED objective
Let h denote the hidden telecom state and bₜ(h) the belief after history Hₜ. A candidate action a may be observational
(ask or diagnose), interventional (change a setting or apply a repair), or terminal (commit or escalate). Because
interventions change the world, an action produces both an observation o and a post-action state h′ under the joint
interventional model p(o, h′ | h, do(a)) = p(h′ | h, do(a)) · p(o | h, h′, do(a)). Observational actions leave the
state unchanged (h′ = h). In the τ²-Bench telecom tools the transition is deterministic, h′ = T(h, a). The downstream
decision r is evaluated on the state it acts on, via U(r, h′).

The belief update therefore includes the transition:

    bₜ₊₁(h′ | o) ∝ Σₕ bₜ(h) p(o, h′ | h, do(a))

    a* = argmaxₐ { Eₒ[V(bₜ₊₁)] − V(bₜ) − λ C(a) − ρ R(a) }    (1)

Here V(b) = maxᵣ Σₕ b(h) U(r,h) is evaluated over the belief's own state variable: bₜ₊₁ is a belief over post-action
states h′, while V(bₜ) uses the pre-action states. C(a) charges turns and tools, and R(a) penalizes unsafe, disruptive,
or irreversible interventions. A stop action is selected when no nonterminal action has positive net value. Note that
this is a one-step (myopic) rule; the non-myopic extension is in "Myopia and multi-step value" below.

**Finite hypothesis calculation.** For the controlled telecom support, expected decision utility can be evaluated
without materializing a normalized posterior for every candidate outcome. The predictive probability and posterior
normalizer cancel inside the expectation, and a repair action earns value by moving the state into one where the
committed decision succeeds:

    DUG(a) = Σₒ maxᵣ Σₕ Σₕ′ bₜ(h) p(o, h′ | h, do(a)) U(r, h′) − maxᵣ Σₕ bₜ(h) U(r, h) − λC(a) − ρR(a)    (2)

With deterministic transitions h′ = T(h, a) this reduces to

    DUG(a) = Σₒ maxᵣ Σₕ bₜ(h) p(o | h, do(a)) U(r, T(h, a)) − maxᵣ Σₕ bₜ(h) U(r, h) − λC(a) − ρR(a)    (2′)

which for observational actions (T(h, a) = h) is the original decision-utility gain. Equation (2)/(2′) provides a reference calculation
for candidate ranking and supervision. Implement it with ordinary CPU or PyTorch operations, using exact enumeration
where the hypothesis support is tractable and a controlled sampling approximation otherwise. Log scoring cost as part
of end-to-end profiling.

*Revision 2026-10-11: eqs. (1)-(2) now include the intervention transition p(h′ | h, do(a)). The original version scored
U(r, h) on the pre-action state, which gives a state-changing repair no value of its own.*

| Signal | Construction | Used by |
|---|---|---|
| Oracle acquisition score | Exact or high-sample DUG on simulator hypotheses | Candidate ranking and SFT labels |
| Belief calibration | Cross-entropy/Brier score against hidden simulator state | Belief head and confidence control |
| Branch advantage | Final-state utility difference between sibling actions from one snapshot | GRPO group-relative reward |
| Verification reward | Reward catching a silent failure; penalize unnecessary checks | Stop/verify policy |
| Robustness reward | Worst-case or mixture utility across user-response models | Distribution-shift training |

**Information gain remains a baseline.** Entropy reduction and expected information gain are retained as ablations. The
claim is not that uncertainty is irrelevant; it is that the useful uncertainty is the uncertainty that can change the
best repair, verification decision, or escalation choice.

### Myopia and multi-step value: a reach-avoid formulation
*Revision 2026-10-11 (new subsection).*

Equation (2) is a one-step rule. Huang et al. (NeurIPS 2024) ablate exactly this case (α = 0) of decision-aware BOED.
The rule undervalues actions whose payoff only appears two or more steps later. The canonical case is repair → verify:
- A repair raises P(repaired), but committing is only safe after a confirming observation.
- A one-step rule therefore either commits unverified (false acceptance), or skips a repair whose immediate gain is
  smaller than its cost and risk.

We replace the selection and stop rule with a **reach-avoid value over beliefs**. It is inspired by Hamilton–Jacobi
reachability RL (Fisac et al., ICRA 2019; Hsu et al., RSS 2021) and its probabilistic forms (Summers & Lygeros,
Automatica 2010; Ganai et al., NeurIPS 2023).

**Augmented belief.** βₜ = (bₜ(h), vₜ, κₜ, Cₜ), where:
- bₜ is the posterior over the fault combination, optionally with a user-compliance latent
- vₜ = 1 if a confirming observation followed the last intervention
- κₜ = 1 if an irreversible or policy-violating action was taken
- Cₜ is the cost spent so far

**Sets.**
- Target T: committed, the final-state grader passes, and vₜ = 1.
- Failure F: κₜ = 1, or a commit while not repaired (false acceptance), or Cₜ > B.

**Value.** The value is the cost-adjusted probability of reaching T before F:

    V*(β) = max{ 1_T(β),  1_¬F(β) · maxₐ [ γ·E_{o,h′} V*(β′_{a,o}) − λC(a) − ρR(a) ] }    (4)
    Q(β, commit) = P_β(repaired ∧ v = 1)        (a wrong commit is absorbed into F)
    Q(β, a)      = γ·E_{o,h′} V*(β′_{a,o}) − λC(a) − ρR(a)    for non-terminal a

**Decision rule.**
- Commit iff Q(β, commit) ≥ max_{a≠commit} Q(β, a) and P_β(repaired) ≥ 1 − δ.
- Escalate if maxₐ Q(β, a) < η.
- Otherwise take the argmax.

A repair now inherits the value of the verification that follows it, through V*(β′), which eq. (2) cannot see.

**Margin variant.** A smoother version uses the margin functions ℓ(β) = min{P_β(repaired) − p*, vₜ − ½} and
g(β) = min{B − Cₜ, ½ − κₜ} in the discounted reach-avoid backup V = (1−γ)·min{ℓ, g} + γ·min{g, max{ℓ, maxₐ V(β′)}}
(Hsu et al. 2021). With γ < 1 and cost terms, V is a reach-avoid value rather than an exact probability. The
Lipschitz formulation of Li et al. (2024) keeps the reach-avoid set exact for all γ.

**How V is obtained.**
1. **Exact expectimax** over the finite hypothesis support at small depth. This is the oracle, and it also yields the
   k-step-lookahead DUG baselines.
2. **Fitted reach-avoid value iteration** on counterfactual branches from the τ² simulator, with targets
   y = max{1_T, 1_¬F·maxₐ(γ·mean_o V̄(β′) − λc)}. Validated against the exact DP on small supports.
3. **In GRPO**, used only as:
   - potential-based shaping γV(β′) − V(β), which is policy-invariant (Ng et al. 1999), and/or
   - a least-restrictive runtime filter that replaces an unverified commit with a verify action.

   It never replaces the binary final-state reward.

**Baselines kept:** myopic DUG (2), k-step lookahead DUG, discounted amortized DUG (TNDP-style, α > 0), and EIG.

### Model internals of the tuned policy as a monitor
*Revision 2026-10-11 (new subsection).*

The Qwen3-8B policy that we SFT/GRPO-tune is also read as a monitor through its hidden states. Features are taken at
the last token before each agent action, with a sweep over layers. Only these vectors are cached.

- **Belief probes.** One probe per fault factor, supervised by the simulator's hidden state at training time only, and
  temperature-scaled. Compared against, and fused with, the explicit Bayes filter over the finite support. The probe
  has to add something the filter lacks, such as the user-compliance latent. Precedent: belief states are linearly
  decodable from the residual stream (Shai et al., NeurIPS 2024; Othello-GPT, Li et al. 2023).
- **Reach head V_φ(zₜ).** A probe on hidden states that estimates the reach-avoid value (4), trained first on Monte
  Carlo targets and then on Bellman targets. BRT-Align (Karnik & Bansal 2025) does this for avoid-only LLM safety;
  this project adds the liveness side (reaching a verified repair).
- **False-acceptance monitor.** Runs at commit turns. With a conformal threshold, it triggers a verify action instead
  of a commit.
- **Interpretability audits.**
  - SelfIE readouts (Chen, Vondrick & Mao, ICML 2024) of the commit-turn hidden state in failure cases.
  - Probes at every checkpoint and every layer, because alignment and fine-tuning can suppress or shift concept
    representations (Sun, Mao et al., ACL 2025; Prakash et al., ICLR 2024; Lugoloobi & Russell 2025).
- **Guardrail.** The policy is never optimized against the monitor; RL learns to evade latent monitors (Gupta & Jenner
  2025; Baker et al. 2025). V enters training only through potential-based shaping, and monitors are always evaluated
  on held-out personas and fault combinations.
- **Metrics.**
  - Probe quality: NLL, Brier score and ECE against the exact posterior.
  - Monitor quality: AUROC/AUPRC, false-acceptance rate at fixed recall, and lead time before failure.
  - Transfer: from base to SFT to GRPO checkpoints, and across personas.

**Where should the reach value live? A planned comparison.** The same reach-avoid targets are fitted on four state
representations, ordered from most structured to least:
1. The exact augmented belief βₜ: the oracle DP, which is exact on the finite support.
2. A probe-derived concept bottleneck β̂ₜ: belief, verified and cost estimates read from the hidden states. It is
   low-dimensional, interpretable, and can be checked against the exact DP.
3. The raw hidden state zₜ: an MLP or low-rank head, as in BRT-Align.
4. A SelfIE-verbalized state: the SelfIE text of zₜ, re-embedded.

Each is reported on calibration of V, false acceptance at matched turns, and the cost of computing the state. By the
data-processing inequality, option 4 cannot carry more information than option 3. It is included because it is
interpretable and may act as a bottleneck that strips out nuisance features. SelfIE is otherwise used to audit states
near the decision threshold and in failure cases.

**Added hypotheses.**
- H6. The reach-avoid rule (4) lowers false acceptance at matched turns, compared with myopic DUG.
- H7. Hidden-state probes on the tuned policy give calibrated belief and reach estimates that add information beyond
  the explicit Bayes filter, and they stay informative after GRPO when the policy is not optimized against them.

## 4 Qwen SFT, GRPO and the decision layer
Qwen3-8B is not asked to perform exact nested Bayesian arithmetic in text. It provides semantic compression and
candidate generation; the decision layer performs calibrated scoring. This division makes the model easier to train
and the policy easier to inspect.

History + tools → Qwen belief / candidates → Decision utility scoring → Selected action → Environment + verifier

| Stage | Model / system behavior | Supervision |
|---|---|---|
| Belief extraction | Predict distribution or structured logits over relevant latent factors | Simulator state during training only; never passed at deployment |
| Candidate proposal | Generate a compact valid set of asks, tests, repairs, and verifications | Teacher traces, tool schema, and successful trajectories |
| Acquisition scoring | Evaluate repair-relevant utility, cost, and risk | Exact or Monte Carlo reference scores computed with standard libraries |
| SFT | Imitate high-value actions and concise rationales | Filtered counterfactual branches and teacher repairs |
| GRPO | Optimize sibling branches from common prefixes | Final-state success, DUG, efficiency, calibration, and safety penalties |
| Controller | Choose continue, ask, intervene, verify, retry, stop, or escalate | Policy plus score thresholds and valid-action masks |

**Proposed group-relative reward.**

    R = α·FinalSuccess + β·ΔDecisionUtility + γ·CaughtFailure − η·Turns − κ·ToolCost − μ·UnsafeAction − ν·FalseAccept    (3)

Groups are formed from sibling continuations at a saved environment snapshot, which controls prefix difficulty and
improves credit assignment. Reward terms are logged separately; a single scalar is used for optimization but never
hides the tradeoff between success, cost, verification, and safety.

**GRPO as an optional experiment.** The minimum credible ladder is: base Qwen → teacher-filtered LoRA SFT → SFT plus
decision layer → GRPO with final-state reward → GRPO with branch-relative repair utility. If GRPO does not outperform
the SFT/controller system at matched node-hours, that negative result is still interpretable. The project should not
depend on reinforcement learning winning.

**Robustness protocol.** Train and test across different simulated customer response models: cooperative, terse,
confused, contradictory, noncompliant, and delayed confirmation. Optimize expected performance on the training
mixture, then report worst-group and shifted-mixture performance to detect policies that merely learn the simulator's
conversational style.

## 5 Distributed rollout generation and learning
The course implementation targets the available A100 nodes. For an 8B LoRA policy, generation and environment
interaction are expected to be substantial costs, which profiling will verify. Actor replicas generate counterfactual
branches and feed a learner through a trajectory queue. Larger Perlmutter experiments can use its four-A100 GPU nodes
outside the course allocation; a candidate layout assigns three GPUs to actors and one to the learner, subject to
measured utilization and memory requirements.

CPU env + snapshots → A100 actors 0–2 → Trajectory queue → A100 learner 3 → LoRA refresh (back to actors)

| Component | Responsibility | Key design choice |
|---|---|---|
| Environment workers | Run τ²-Bench tools, save/restore state, grade final state | Pin CPU workers by NUMA locality; batch metadata transfers |
| Actor replicas | Generate candidate actions and sibling continuations | Continuous batching; prefix caching as infrastructure |
| Decision scoring | Score candidate actions and produce oracle labels | Standard CPU or PyTorch implementation; profile its share of end-to-end runtime |
| Trajectory store | Hold token IDs, tool events, masks, rewards, and snapshot IDs | Compact schema; no repeated prompt text |
| Learner | LoRA SFT/GRPO updates | Bounded policy staleness; periodic adapter broadcast |
| Evaluator | Frozen seeds and held-out state combinations | Separate queue so training throughput cannot hide regressions |

**Scaling experiments.**
- Compare 1 and 4 GPUs on the available class A100 nodes. Extend rollout scaling to 8 and 16 GPUs only when additional
  allocations permit it; report strong scaling and verified episodes per node-hour.
- Synchronous versus bounded-staleness collection with adapter refresh intervals measured in policy updates and
  generated tokens.
- One shared actor/learner node versus disaggregated actor and learner nodes; account for communication and queue wait time.
- Branch factor 1/2/4/8 from identical prefixes; measure quality gained per additional generated token.

**Prefix reuse.** Counterfactual siblings share a long conversation prefix, so vLLM automatic prefix caching should be
enabled and its hit rate reported. The research question is whether branch scheduling and acquisition work coexist
efficiently, not whether prefix caching itself is novel. If tool-result serialization breaks token-identical prefixes,
canonicalize only semantically equivalent metadata and verify that model-visible content is unchanged.

**Failure isolation and reproducibility.** Every trajectory carries the model revision, LoRA revision, simulator seed,
hidden-state identifier for offline evaluation, prompt hash, tool-event log, snapshot parent, and decision-scoring
configuration. Actors can fail independently; committed trajectories remain immutable, and evaluation always uses
frozen adapters and deterministic seed sets.

## 6 Experimental matrix and metrics
Evaluation measures policy quality and distributed learning efficiency. Improvements must survive held-out tasks and
fixed compute or interaction budgets. End-to-end profiling identifies costs in environment execution, model
generation, decision scoring, communication, and learning.

| Axis | Conditions |
|---|---|
| Policy | Base Qwen; LoRA SFT; recovery controller; EIG; repair-relevant DUG; DUG + GRPO |
| Action semantics | Question-only; observational diagnostics; interventions; interventions + active verification |
| Training data | Linear successful traces; teacher repairs; sibling counterfactual branches; branch filtering |
| User model | In-distribution mixture; held-out personas; contradiction/noise; compliance shift |
| Scale | 1 and 4 GPUs for the course study; optional 8 and 16 GPU extensions; synchronous/asynchronous collection; branch factors and refresh intervals |

**Primary policy metrics.**

| Metric | Definition / reporting |
|---|---|
| Final-state success | Programmatic grader success; macro-average across subtask group and intent |
| Verified success | Successful episodes in which terminal confidence is supported by an observation or state check |
| False acceptance | Agent stops as successful while final state is incorrect |
| Interaction efficiency | Turns, tool calls, disruptive actions, and tokens per verified success |
| Decision regret | Utility gap from oracle action/repair under the simulator belief support |
| Calibration | Brier score, NLL, reliability curves, and selective risk versus coverage |
| Robustness | Mean, worst-group, and shifted-mixture success; safety violations by group |

**System metrics.**

| Level | Metrics |
|---|---|
| Online scoring | Candidate actions scored/s, p50/p95 acquisition latency, fraction of episode time |
| Rollout | Tokens/s, episodes/s, verified episodes/node-hour, actor utilization, queue delay |
| Learning | Updates/hour, freshness lag, time-to-quality, node-hours to target success |
| End-to-end | Wall time and node-hours per statistically supported improvement over SFT baseline |

## 7 Execution plan and compute allocation
Run the course experiments on the class A100 resources. The separate larger-experiment budget assumes approximately
250 Perlmutter GPU node-hours, corresponding to up to roughly 1,000 A100 GPU-hours if all four GPUs on each node are
occupied. The allocation below preserves the total budget while directing effort to reference scoring, data
generation, learning, and evaluation. Allocate the additional 65 node-hour reserve after baseline profiling.

| Work package | Node-hours | Exit criterion |
|---|---|---|
| Profile + reproduce baselines | 20 | Stable τ²-Bench runs; trace-level CPU/GPU timeline |
| Reference decision scoring and validation | 30 | Reliable acquisition labels and representative decision cases |
| Counterfactual rollout generation | 70 | Filtered branch dataset with snapshot lineage |
| LoRA SFT and GRPO | 35 | Matched-budget checkpoints and training curves |
| Scaling and ablations | 20 | 1/4/8/16-GPU efficiency and objective ablations |
| Reserve / reruns | 10 | Failed jobs, variance checks, or final confirmation |
| Additional learning and evaluation reserve | 65 | Allocate to branch generation, training, or evaluation after profiling |

**Stage gates.**
1. After baseline profiling, address environment batching, state serialization, model generation, and queue
   bottlenecks using standard tooling.
2. After SFT, run GRPO only if the reward decomposition is stable and counterfactual siblings produce meaningful return variation.
3. Before scaling, confirm the learner is not starved and actor queues are not dominated by CPU simulation or serialization.

**Risks and fallbacks.**

| Risk | Evidence to monitor | Fallback |
|---|---|---|
| Belief head is miscalibrated | Poor reliability and action regret | Use factorized hypotheses, temperature calibration, and conformal stop thresholds |
| GRPO destabilizes | KL spikes, reward hacking, false acceptance | Use SFT/controller finalist; constrain actions and simplify reward |
| Branch explosion | Low marginal quality per token | Adaptive branch factor based on acquisition margin |
| Simulator overfitting | Sharp user-model shift drop | Hold out personas and latent combinations; optimize robust mixture utility |

## 8 Deliverables and success criteria

| Deliverable | Concrete artifact | Minimum success criterion |
|---|---|---|
| D1. Decision formulation | Belief schema, interventional likelihoods, utility/cost/risk definition | Oracle actions reproduce known cases and expose stop/verify tradeoffs |
| D2. Counterfactual dataset | Snapshot-linked sibling branches with final-state grades | Coverage across 15 subtask groups and held-out combinations |
| D3. Qwen policy | LoRA SFT/controller and optional GRPO checkpoints | Improved verified success or lower cost at matched success |
| D4. Distributed runtime | Asynchronous actors, learner, compact trajectory queue | Scaling study with utilization and node-hour accounting |
| D5. Research report | Ablations, failure taxonomy, reproducible configurations | Claims match profiler, held-out evaluation, and confidence intervals |

**Course contribution.** The project combines a learning question about repair decisions with an end-to-end
distributed experiment. It evaluates verified success, interaction cost, calibration, and time to quality. The telecom
simulator supplies controlled branches and final-state grading, so both positive and negative results can be
interpreted at matched budgets.

**Immediate implementation order.**
1. Freeze the task split, grader, trace schema, and hidden-state leakage checks.
2. Implement the exact CPU/PyTorch DUG oracle and profile it inside a real episode.
3. Build counterfactual branching and compare EIG, DUG, and myopic heuristics before training.
4. Train Qwen LoRA SFT, add the controller, and treat GRPO as the final controlled experiment.
5. Run fixed-budget distributed scaling and report verified success per node-hour.

**Statistical protocol.** Use fixed held-out seeds, paired episode comparisons where possible, bootstrap confidence
intervals over tasks, and multiple training seeds for the finalists. Report both test-set means and the distribution
across the 15 subtask groups; aggregate success alone can conceal a policy that collapses on rare but costly cases.

## 9 Selected primary references
*Revision 2026-10-11: identifiers added, verified 2026-10-11. Items 8-21 are new.*
1. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. Barres et al., arXiv:2506.07982, ICML 2026.
2. BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design. Choudhury et al., arXiv:2508.21184, ICLR 2026.
3. Amortising Bayesian Experimental Design for Sequential Information Gathering in LLMs (ASIG). Hartmann et al., arXiv:2607.03426, ICML 2026 workshop (non-archival).
4. Amortized Bayesian Experimental Design for Decision-Making (TNDP; DUG, Def. 3.1). Huang et al., arXiv:2411.02064, NeurIPS 2024.
5. Branching Policy Optimization: Sandbox-Native Language Agent Reinforcement Learning. He et al., arXiv:2607.14171, WAIC 2026.
6. NERSC Perlmutter Architecture (docs.nersc.gov).
7. vLLM Automatic Prefix Caching (docs.vllm.ai/en/latest/features/automatic_prefix_caching/).
8. Bridging Hamilton-Jacobi Safety Analysis and Reinforcement Learning. Fisac et al., ICRA 2019.
9. Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning. Hsu et al., arXiv:2112.12288, RSS 2021.
10. Certifiable Reachability Learning Using a New Lipschitz Continuous Value Function. Li et al., arXiv:2408.07866.
11. Iterative Reachability Estimation for Safe Reinforcement Learning (RESPO). Ganai et al., arXiv:2309.13528, NeurIPS 2023.
12. DeepReach. Bansal & Tomlin, arXiv:2011.02082, ICRA 2021.
13. Preemptive Detection and Steering of LLM Misalignment via Latent Reachability (BRT-Align). Karnik & Bansal, arXiv:2509.21528.
14. From Refusal to Recovery. Pandya et al., arXiv:2510.13727.
15. Deep Adaptive Design. Foster et al., arXiv:2103.02438, ICML 2021. Optimizing Sequential Experimental Design with Deep RL. Blau et al., arXiv:2202.00821, ICML 2022.
16. SelfIE: Self-Interpretation of Large Language Model Embeddings. Chen, Vondrick & Mao, arXiv:2403.10949, ICML 2024.
17. Aligned but Blind. Sun, Mao et al., arXiv:2506.00253, ACL 2025. Raidar. Mao et al., arXiv:2401.12970, ICLR 2024.
18. Transformers represent belief state geometry in their residual stream. Shai et al., arXiv:2405.15943, NeurIPS 2024.
19. Agent-failure probes. Ruan et al., arXiv:2607.06503. Unreliable mid-task progress reports on τ²-bench: Wang et al., arXiv:2609.08589.
20. RL-Obfuscation. Gupta & Jenner, arXiv:2506.14261. Monitoring reasoning models for misbehavior. Baker et al., arXiv:2503.11926.
21. AReaL (async RL with τ² example). arXiv:2601.22607; github.com/areal-project/AReaL.
