#!/bin/bash
# [telecom_boed] τ²-bench telecom episodes against self-hosted vLLM, on ONE Perlmutter GPU (shared QOS, charged per GPU).
# Agent and user simulator are both served by the same vLLM server (base model, separate prompts). Non-thinking mode.
# No API models. Run outputs: $SCRATCH/ddl/telecom_boed/runs/<RUN_TAG>_<jobid>/ (tau2 results.json, vLLM metrics, logs).
# The pinned tau2 clone is never written: tau2's data dir (where it also saves results) is copied per run.
# Knobs (via --export): RUN_TAG, MODEL (Qwen/Qwen3-8B), SPLIT (train), NUM_TASKS (3), TASK_IDS (optional, space-separated),
#   NUM_TRIALS (1), MAX_CONC (3), MAX_STEPS (200), MAX_MODEL_LEN (32768), GPU_UTIL (0.90), SEED (300).
#SBATCH -A m4431_g
#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH -t 00:45:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH -J t2tel
#SBATCH -o telecom_boed/logs/%x-%j.out
#SBATCH -e telecom_boed/logs/%x-%j.err
set -euo pipefail
ENVS=/global/cfs/cdirs/m4431/kbmkbm/envs
TAU2=/global/cfs/cdirs/m4431/kbmkbm/ext/tau2-bench
TAU2_SHA=4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699
MODEL="${MODEL:-Qwen/Qwen3-8B}"
RUN="${SCRATCH}/ddl/telecom_boed/runs/${RUN_TAG:-tau2}_${SLURM_JOB_ID}"
mkdir -p "${RUN}"
export HF_HOME="${SCRATCH}/ddl/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Caches on node-local /tmp: FlashInfer JIT / filelock need flock, which Lustre $SCRATCH lacks (FC-011, job 59841127).
# hbm40g pins 40 GB A100s so T1 matches the local rlab2 A100-40GB twin (job 59841127 landed on an 80 GB node).
NODE_CACHE="/tmp/kbmkbm_tel_${SLURM_JOB_ID}"; mkdir -p "${NODE_CACHE}"
export VLLM_CACHE_ROOT="${NODE_CACHE}/vllm" XDG_CACHE_HOME="${NODE_CACHE}/xdg" FLASHINFER_WORKSPACE_BASE="${NODE_CACHE}/flashinfer" \
       TRITON_CACHE_DIR="${NODE_CACHE}/triton" TORCHINDUCTOR_CACHE_DIR="${NODE_CACHE}/inductor" TMPDIR="${NODE_CACHE}/tmp"
mkdir -p "${TMPDIR}"
# Nothing may land in $HOME (job 59841127 wrote ~/.config/vllm, ~/.cache/flashinfer, ~/.triton, ~/.humming): point HOME at the
# node-local cache for this job (no ssh/credentials are needed inside it) and switch off vLLM usage stats.
export HOME="${NODE_CACHE}/home" VLLM_CONFIG_ROOT="${NODE_CACHE}/vllm_config" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_DO_NOT_TRACK=1
mkdir -p "${HOME}" "${VLLM_CONFIG_ROOT}"
# FlashInfer sampler JIT-builds ~2 min on first use per node (/tmp is fresh each job); the torch sampler gives the same
# greedy (temperature 0) outputs. Set VLLM_USE_FLASHINFER_SAMPLER=1 via --export to restore.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
[[ "$(git -C "${TAU2}" rev-parse HEAD)" == "${TAU2_SHA}" ]] || { echo "tau2 clone not at ${TAU2_SHA}" >&2; exit 1; }
cp -r "${TAU2}/data" "${RUN}/tau2_data"
export TAU2_DATA_DIR="${RUN}/tau2_data"

{ # provenance
  echo "job=${SLURM_JOB_ID} node=$(hostname) model=${MODEL} split=${SPLIT:-train} num_tasks=${NUM_TASKS:-3} task_ids=${TASK_IDS:-} trials=${NUM_TRIALS:-1} conc=${MAX_CONC:-3}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  "${ENVS}/ddl-vllm/bin/python" -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
  "${ENVS}/ddl-telecom/bin/python" -c "import importlib.metadata as m;print('tau2',m.version('tau2'),'litellm',m.version('litellm'))"
  echo "tau2_sha=${TAU2_SHA}"; md5sum "${TAU2_DATA_DIR}/tau2/domains/telecom/tasks.json" "${TAU2_DATA_DIR}/tau2/domains/telecom/split_tasks.json"
} | tee "${RUN}/job_info.txt"

# --- vLLM server (background, this node only) ---
PORT=$((8000 + SLURM_JOB_ID % 1000))
t0=${SECONDS}
setsid "${ENVS}/ddl-vllm/bin/vllm" serve "${MODEL}" --host 127.0.0.1 --port "${PORT}" --served-model-name "${MODEL}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching --seed 0 \
  > "${RUN}/vllm.log" 2>&1 &
VPID=$!
trap 'kill -TERM -- -${VPID} 2>/dev/null || kill ${VPID} 2>/dev/null || true; sleep 5; kill -KILL -- -${VPID} 2>/dev/null || true' EXIT   # whole process group (EngineCore children too; job 59842860 idled to TIMEOUT)
for i in $(seq 1 $(( ${HEALTH_TIMEOUT_S:-1500} / 5 ))); do   # CFS env import ~3 min + engine init (FC-005); job 59842860 hit the old 10 min
  curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null && break
  kill -0 ${VPID} 2>/dev/null || { echo "vLLM died during startup" >&2; tail -n 40 "${RUN}/vllm.log" >&2; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null || { echo "vLLM not healthy after ${HEALTH_TIMEOUT_S:-1500}s" >&2; exit 1; }
echo "vllm_ready_s=$((SECONDS - t0))" | tee -a "${RUN}/job_info.txt"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/gpu_mem_after_load=/' | tee -a "${RUN}/job_info.txt"

# --- tau2 run: agent + user both on the local server, non-thinking (Qwen3 chat template switch) ---
LLM_ARGS="{\"api_base\": \"http://127.0.0.1:${PORT}/v1\", \"api_key\": \"local\", \"temperature\": 0.0, \"extra_body\": {\"chat_template_kwargs\": {\"enable_thinking\": false}}}"
args=(run --domain telecom --task-split-name "${SPLIT:-train}" --num-trials "${NUM_TRIALS:-1}"
      --agent-llm "hosted_vllm/${MODEL}" --agent-llm-args "${LLM_ARGS}"
      --user-llm "hosted_vllm/${MODEL}" --user-llm-args "${LLM_ARGS}"
      --max-concurrency "${MAX_CONC:-3}" --max-steps "${MAX_STEPS:-200}" --seed "${SEED:-300}"
      --save-to "${RUN_TAG:-tau2}_${SLURM_JOB_ID}")
if [[ -n "${TASK_IDS:-}" ]]; then args+=(--task-ids ${TASK_IDS}); else args+=(--num-tasks "${NUM_TASKS:-3}"); fi
echo "tau2 ${args[*]}" | tee -a "${RUN}/job_info.txt"
t1=${SECONDS}
set +e
"${ENVS}/ddl-telecom/bin/tau2" "${args[@]}" > "${RUN}/tau2.log" 2>&1
rc=$?
set -e
echo "tau2_rc=${rc} tau2_wall_s=$((SECONDS - t1))" | tee -a "${RUN}/job_info.txt"
curl -sf "http://127.0.0.1:${PORT}/metrics" > "${RUN}/vllm_metrics.prom" || true
grep -E "^vllm:(prefix_cache_(hits|queries)|prompt_tokens|generation_tokens)_total" "${RUN}/vllm_metrics.prom" | tee -a "${RUN}/job_info.txt" || true
cp "${TAU2_DATA_DIR}/simulations/${RUN_TAG:-tau2}_${SLURM_JOB_ID}/results.json" "${RUN}/results.json" 2>/dev/null || true
tail -n 25 "${RUN}/tau2.log"
exit ${rc}
