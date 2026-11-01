# verl per-request LoRA switch (ddl patch)

- Base: verl v0.9.1, commit `1876b06d0a3e4e71e06230be10af14492ca8a75b` (read-only reference clone `/common/users/ss5772/external/verl`).
- Patch: `per_request_lora.patch`, md5 `6340ea734b738d702598169c8f6f2843` (re-check with `md5sum`).
- What it changes (default-preserving):
  - `verl/workers/rollout/vllm_rollout/vllm_async_server.py`: `vLLMHttpServer.generate(..., use_lora: bool = True)`; with `use_lora=False` the request is sent with `lora_request=None`, so the base model answers.
  - `verl/workers/rollout/llm_server.py`: `LLMServerClient.generate(..., use_lora: bool = True)`; the flag is forwarded to the server only when it is False, so the default call is identical to upstream.
- Not covered: the prefill/decode `_pd_dispatch` path does not forward the flag.
- Apply to a copy (never the reference clone): `cp -r <verl at 1876b06d> <copy> && cd <copy> && git apply /common/users/ss5772/ddl/telecom_boed/patches/verl/per_request_lora.patch`
- Perlmutter copy used by env `ddl-verl`: `/global/cfs/cdirs/m4431/kbmkbm/src/verl-ddl` (editable install). Build: `scripts/build_ddl_verl_pm.sh`.
- CPU test: `tests/test_verl_lora_patch.py`. Still to do (first P5/P6 GPU smoke): check that adapter-off logprobs equal the base model's.
