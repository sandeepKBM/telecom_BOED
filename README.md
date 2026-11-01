# telecom_boed - repair-relevant BOED + distributed counterfactual learning for telecom agents

DDL course project: a τ²-Bench telecom troubleshooting agent (Qwen3-8B, LoRA SFT, optional async GRPO) that decides when to
ask, diagnose, intervene, verify, or escalate. It has three parts:
- a repair-relevant Bayesian experimental-design layer
- a reach-avoid (non-myopic) value
- hidden-state probes as monitors

Distributed rollout generation and training are measured with node-hour accounting on Perlmutter.

- Proposal (with revisions): [`docs/proposal.md`](docs/proposal.md)
- Feasibility review + phased plan + open questions: [`PLAN.md`](PLAN.md)
- Status log: [`../docs/status/telecom_boed.md`](../docs/status/telecom_boed.md)

Separate from task_1/task_2. Upstream τ²-bench is an external pinned dependency (commit `4ce7c03`), never edited in place.
