# telecom_boed: feasibility review and implementation plan (2026-10-11)

Proposal: [`docs/proposal.md`](docs/proposal.md). That copy already includes:
- the eq. (1)-(2) transition fix
- the reach-avoid section
- the model-internals section
- the reach-value state comparison
- verified references

Evidence for this review:
- τ²-bench code read at commit `4ce7c03` (2026-10-06, v1.0.1, MIT)
- web checks of every cited paper and RL framework, done 2026-10-11

**Constraints (user, 2026-10-11):**
- No GPT-4.1 or any API model. Every LLM (policy, user simulator, teacher) is self-hosted on westeros, ilab or rlab
  (or Perlmutter later).
- Perlmutter is not used for now.
- Scoping decisions belong to the user. This file lists facts, options and costs, not cuts.

## 1. Proposal claims checked

| Claim | Status | Evidence / correction |
|---|---|---|
| 2,285 controlled combinations | ✅ | `data/tau2/domains/telecom/tasks.json` (= `tasks_full.json`); paper Table 1 |
| 114 standard tasks | ✅ | `split_tasks.json`: base 114 = **train 74 + test 40**; small 20 |
| 3 intents | ✅ | service 47 / mobile_data 254 / mms 1,984 (heavily mms-skewed) |
| 15 subtask groups | ⚠️ | The paper says 15. The code has **14** SelectionSets (service 5, mobile 6, mms 3) and 20 base faults (19 in the shipped data). |
| Finite support / exact oracle | ✅ | Cartesian product of SelectionSets (`tasks/utils.py:61-92`). The current code would regenerate about 3,789 tasks, not 2,285. **Pin the shipped `tasks.json`; never run `create_tasks.py`** (it overwrites the file). |
| Personas crossed with faults | ❌ | Personas are round-robin (None/Easy/Hard), so each combination appears with one persona. The persona study needs our own user-model variants. |
| Final-state grading | ✅ | Binary reward = product over `reward_basis`. 2,253 tasks use ENV_ASSERTION; **32 also use ACTION** (transfer_to_human for SIM-PIN or contract-end cases; 20 of them are in base). Docs claim ACTION is unused in telecom, which is wrong. |
| Simulator snapshots | ⚠️ | No fork API. Restore works by (a) dumping both pydantic DBs, rebuilding with `get_environment(db, user_db)`, then calling `sync_tools()`, or (b) replay via `set_state` / `Orchestrator.initialize(message_history)`. **Avoid the `initialization_data` path**: it cross-wires the agent and user DBs (`environment.py:352-363`). |
| Deterministic branching | ⚠️ | Tools are deterministic. The user is an LLM. With a self-hosted user, sampling seeds are under our control, but vLLM outputs are not guaranteed bitwise reproducible across batch compositions. Branch outcomes therefore include user variance. BPO assumes determinism. |
| Prefix caching for branches | ⚠️ | On by default in vLLM. **The cache key includes the LoRA adapter**, so each new adapter starts cold. Reuse happens only within one policy version (system prompt, GRPO siblings, branches). Qwen3's thinking mode drops earlier thinking blocks from the template, which breaks prefixes. |
| 3 actor GPUs + 1 learner GPU | ⚠️ | Written for a 4×A100 node. The local GPUs differ in memory (see §3), and the self-hosted user simulator needs GPUs too. |
| SFT/RL gains on τ²-telecom | context | AReaL reports Qwen3-30B-A3B 27.1 → 80.7 (SFT) → 89.5 (RL) with a GPT-4.1 user, on 24+8 H200-class GPUs. Our numbers come from a self-hosted user simulator, so they are **not comparable to that or to the τ² leaderboard**. The leaderboard records the user model; report ours explicitly. |
| Eq. (2) | fixed | Now includes the transition p(o, h′ \| h, do(a)) and U(r, T(h,a)) (proposal §3). |
| One-step DUG stop rule | fixed | Reach-avoid rule added (proposal §3, eq. 4). One-step DUG is kept as the α=0 baseline. |

## 2. Design decisions taken so far
1. **Contamination-safe split.**
   - Test = test-40.
   - Train = train-74, plus `full − test` combinations for the branch pool.
   - Before any use of AReaL's released data (9,294 SFT, 271 RL telecom rows), check it for overlap with test-40.
2. **The simulator is the likelihood model.**
   - For each hypothesis h, build the environment from h with the generator's fault-injection functions and apply
     the real tools. The tool part of p(o, h′ | h, do(a)) is exact.
   - Observations that go through the user get a per-persona noise model.
   - At deployment the Bayes filter conditions only on observations. Hidden-state labels are used only for training
     probes and as oracle labels.
3. **All LLMs are self-hosted with vLLM**, OpenAI-compatible, passed to tau2 via litellm as
   `--user-llm hosted_vllm/<model> --user-llm-args '{"api_base": ...}'`. This is inferred from litellm pass-through
   and is to be verified in P2. The user simulator is a frozen open model with persona prompts.
4. **Qwen3-8B, non-thinking mode**, with Hermes tool-call parser.
5. Reach-avoid value and probes follow proposal §3.
   - Exact expectimax on small support serves as oracle and validation.
   - Fitted reach-avoid value iteration on branches.
   - The policy is never optimized against a monitor.

## 3. Local hardware facts (2026-10-11)

| Host | GPUs | Memory/GPU | Notes |
|---|---|---|---|
| westeros | 8× RTX 2080 Ti | 11 GB | Turing sm_75: **no bf16**, fp16 only. Qwen3-8B bf16 (16.4 GB) does not fit on one card; needs TP=2 (fp16) or a 4-bit checkpoint. Per the cluster rule, **GPUs 0-3 belong to jenga and 4-7 to bimanual**. Which westeros GPUs this project may use is your call. |
| ilab1-2 | 8× A4500 each | 20 GB | Slurm `--gres=gpu:a4500:N` |
| ilab3 | 8× RTX 4500 Ada | 24 GB | |
| ilab4 | 8× RTX PRO 5000 Blackwell | 48 GB (to confirm) | needs CUDA ≥ 12.8 builds |
| rlab1 | 3× RTX 4500 Ada | 24 GB | login/gateway node |
| rlab2 | 4× A100 | 40 or 80 GB (to confirm) | |
| rlab3-4 | 8× A4000 each | 16 GB | |
| rlab6 | 4× A5000 | 24 GB | |
| rlab7 | 7× A6000 | 48 GB | |

**Constraints that apply to every local job:**
- 12-GPU cap (GrpTRES) shared across all ss5772 projects.
- ilab `ulimit -u 2000`. vLLM and torch.compile spawn processes, so set `OMP_NUM_THREADS=1` and compile threads
  to 1.
- Node `/tmp` quota is about 2 GB.
- Never mmap data from NFS (`/common/users`); stage to node-local disk.
- `/common/users` holds about 216 GB across all projects. Model weights (~16 GB per Qwen3-8B copy) should live on
  `/data/local` (westeros) or node-local disk.
- Today's queue was nearly full: 1 of 66 GPUs free.

**What fits where (bf16 Qwen3-8B = 16.4 GB weights, 144 KiB/token KV):**
- 48 GB cards (A6000, Blackwell 5000) or A100: one policy plus ample KV; LoRA SFT at 8-16k tokens per GPU.
- 24 GB cards: serving with about 5 GB of KV (~35k tokens). LoRA SFT needs FSDP across cards or QLoRA.
- 16-20 GB cards: serving needs TP=2 or 4-bit weights.
- Quantized weights change the policy model, so train and eval must use the same weights.
- A larger user simulator than Qwen3-8B (to reduce simulator-driven failures) fits only as 4-bit on 48 GB cards or
  with TP. **User-simulator quality affects measured success** (τ² attributes some failures to the user simulator),
  so it is an experimental variable to report.

### Single-GPU fit, from HF metadata (2026-10-11)
Assumptions: vLLM at `gpu_memory_utilization` 0.90, minus about 2 GiB of non-KV overhead. These are estimates; P2
measures the real numbers. Weights are the on-disk safetensors; KV/token = layers × kv_heads × head_dim × 2 × 2 B.

| Model | Weights | KV/token | A4500 20 GB: KV tokens | 4500 Ada 24 GB: KV tokens | Max context |
|---|---|---|---|---|---|
| Qwen3-8B (bf16) | 15.3 GiB | 144 KiB | ≈5k (not usable) | ≈31k | 40,960 |
| Qwen3-8B-AWQ (int4) | 5.7 GiB | 144 KiB | ≈75k | ≈101k | 40,960 |
| Qwen3-14B-AWQ (int4) | 9.3 GiB | 160 KiB | ≈44k | ≈67k | 40,960 |
| Qwen3-4B-Instruct-2507 (bf16, non-thinking only) | 7.5 GiB | 144 KiB | ≈62k | ≈88k | 262,144 |

**Fit differs by job type:**
- Serving fits as in the table.
- LoRA SFT of an 8B bf16 base does not fit on a single 20-24 GB card at τ² lengths. It fits with QLoRA (nf4 base),
  or with FSDP across cards.
- Colocated GRPO (sampler and learner on the same GPU) does not fit in bf16.
- Hidden-state capture for probes/SelfIE needs a forward pass that returns activations (HF or a vLLM hook). That is a
  second model copy or a separate phase on the same GPU.

## 4. Phases with gates
"Where" uses local resources only, as requested. Perlmutter can be added later.

| # | Phase | Where | Exit criterion (gate) |
|---|---|---|---|
| P0 | Pin tau2 commit + `ddl-telecom` env (Python 3.12); snapshot/branch API (dump DBs, rebuild, sync, hash-equality test vs replay); hypothesis enumerator from SelectionSets; leakage checks | westeros CPU | branch-restore hash equals replay hash on 100 random prefixes; no hidden state reaches policy inputs |
| P1 | Exact Bayes filter + DUG (2′) + k-step expectimax + reach-avoid V on the finite support (CPU/PyTorch); unit tests (repair→verify, unfixable→escalate) | westeros CPU | oracle reproduces the hand cases; myopic vs reach-avoid differ exactly where expected |
| P2 | Self-hosted Qwen3-8B policy + self-hosted user simulator on vLLM; 5-10 telecom tasks end-to-end; verify the litellm `hosted_vllm` path and tool-call parsing | ilab/rlab Slurm | episodes complete; results graded by tau2's evaluator |
| P3 | Baseline profiling on test-40 × ≥4 trials; CPU/GPU timeline; prefix-cache hit rate | ilab/rlab | stable pass^k; time breakdown (env / prefill / decode / user sim / scoring) |
| P4 | Controller study (no training): {myopic DUG, k-step, reach-avoid, EIG, heuristics} on base Qwen candidates | ilab/rlab | false acceptance and turns at matched budget (H6) |
| P5 | Counterfactual branch dataset with snapshot lineage + hidden-state caching (for probes and the reach value) | ilab/rlab | branch set covering all 14 SelectionSets and 3 intents |
| P6 | LoRA SFT (FSDP) + probes (belief / reach / false acceptance) on base and SFT checkpoints; reach-value state comparison (exact β, probe concepts, raw z, SelfIE text) | ilab/rlab | probe calibration; SFT vs base on test-40 (H7 part 1) |
| P7 | Async GRPO (AReaL or verl), branch-relative reward + potential shaping γV(β′)−V(β) | ilab/rlab (later Perlmutter) | reward decomposition stable; siblings vary; matched-compute comparison vs SFT |
| P8 | Systems study: actor GPUs 1→4, sync vs bounded staleness, branch factor 1/2/4/8, 1-node vs 2-node | local first | verified episodes per GPU-hour; time-to-quality; learner starvation check |

Compute is to be measured in P3, as GPU-hours per GPU type. Perlmutter node-hour estimates are not used while
Perlmutter is off.

## 5. Folder layout (as it grows)
```
telecom_boed/
  PLAN.md, README.md, docs/proposal.md
  src/telecom_boed/   env/ (snapshot, branch, hypotheses)  belief/ (Bayes filter)  decide/ (dug, expectimax, reach)
                      probes/ (hidden-state heads, SelfIE readout)  rollout/ (actors, queue)  learn/ (sft, grpo glue)
  configs/  slurm/  scripts/  tests/  results/   (logs/, outputs/ gitignored)
```
tau2-bench stays an external, pinned dependency (not vendored, not edited); any patches live as wrappers in `src/`.

## 6. Decisions needed from you
1. ~~Westeros GPUs~~ **Decided (user, 2026-10-11): use ilab/rlab GPUs, not westeros.** Use the best model that fits a
   single GPU, given what is free.
2. **RL framework.**
   - **AReaL's τ² example:** ready-made τ² workflow and async staleness bound, but it depends on the
     `dhh1995/tau2-bench` fork and assumes 8-GPU nodes.
   - **Our own harness on upstream tau2 + verl:** we write the agent loop, upstream stays clean, and verl has a
     Qwen3-8B LoRA example.
3. **User-simulator model:** Qwen3-8B (same as the policy) vs a larger self-hosted model (4-bit or TP). This trades
   fidelity against GPUs.

## 7. Next steps (no GPU needed)
1. Create the `ddl-telecom` env on westeros (Python 3.12); install tau2-bench at `4ce7c03`; smoke-test that it imports
   and loads telecom tasks.
2. P0: snapshot/branch API, hypothesis enumerator, replay-equality test.
3. P1: exact filter, DUG (2′), expectimax and reach-avoid oracle, with unit tests.
