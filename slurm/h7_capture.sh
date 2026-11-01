#!/bin/bash
# [telecom_boed] H7 capture STEP SCRIPT (not an sbatch file): hidden states of Qwen3-8B at every agent decision point of a
# logged tau2 run. Runs inside the shared rlab2 hold under the GPU lock protocol (one GPU, ~17 GB weights + activations):
#   cdx-agent --gpu-lock 526576 --n 1 --eta 30 --mem 48G -- env RUN=f1plrep_526576.35 bash slurm/h7_capture.sh
# Knobs (env): RUN (run dir name under results/runs, required), RUN_DIR (override), OUT (default /freespace/local/ss5772/h7/$RUN),
#   BUILD (1 = build prompts.jsonl on the node first, CPU, ddl-telecom env; default 1 if missing), LIMIT_EPS, VERIFY (K per episode, default 0).
# Outputs (node-local, never /common/users): $OUT/{prompts.jsonl,index.jsonl,summary.json,shards/*.safetensors}.
# Then from westeros: scp -r rlab2:$OUT /data/local/ss5772/telecom_h7/   (exclude prompts.jsonl if not wanted).
set -euo pipefail
HERE="${TELECOM_HERE:-$(cd "$(dirname "${CDX_GPU_LOCK_SCRIPT_ORIG:-${BASH_SOURCE[0]}}")/.." && pwd)}"   # .../telecom_boed
LOCAL_ROOT="${LOCAL_ROOT:-/freespace/local/ss5772}"
ENVS="${ENVS:-/common/users/ss5772/conda_envs}"
VLLM_ENV="${VLLM_ENV:-${LOCAL_ROOT}/conda_envs/ddl-vllm}"
: "${RUN:?set RUN=<run dir name under results/runs>}"
RUN_DIR="${RUN_DIR:-${HERE}/results/runs/${RUN}}"
OUT="${OUT:-${LOCAL_ROOT}/h7/${RUN}}"
JOB="${SLURM_JOB_ID:-local$$}.${SLURM_STEP_ID:-0}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="${LOCAL_ROOT}/hf_cache" DO_NOT_TRACK=1
export HOME="${LOCAL_ROOT}/z/home_h7_${JOB}"; export TMPDIR="${LOCAL_ROOT}/z/h7_${JOB}"
export XDG_CACHE_HOME="${HOME}/.cache" TRITON_CACHE_DIR="${TMPDIR}/triton"
mkdir -p "${HOME}" "${TMPDIR}" "${OUT}"
trap 'rm -rf "${TMPDIR}" "${HOME}"' EXIT
[[ -f "${RUN_DIR}/results.json" ]] || { echo "missing ${RUN_DIR}/results.json" >&2; exit 1; }
if [[ "${BUILD:-1}" == "1" && ! -s "${OUT}/prompts.jsonl" ]] || [[ "${BUILD:-}" == "force" ]]; then
  H7_ROOT="${LOCAL_ROOT}/h7" TELECOM_SCRATCH="${LOCAL_ROOT}/tmp/telecom_boed" \
    "${ENVS}/ddl-telecom/bin/python" "${HERE}/scripts/h7_build_prompts.py" --run-dir "${RUN_DIR}" --out "${OUT}/prompts.jsonl" | tail -1
fi
ARGS=(--prompts "${OUT}/prompts.jsonl" --out "${OUT}")
[[ -n "${LIMIT_EPS:-}" ]] && ARGS+=(--limit-eps "${LIMIT_EPS}")
[[ -n "${VERIFY:-}" ]] && ARGS+=(--verify-alone "${VERIFY}")
[[ -n "${CAPTURE_MODE:-}" ]] && ARGS+=(--mode "${CAPTURE_MODE}")
echo "[h7_capture] run=${RUN} gpu=${CUDA_VISIBLE_DEVICES:-?} out=${OUT} host=$(hostname)"
"${VLLM_ENV}/bin/python" -I "${HERE}/scripts/h7_capture.py" "${ARGS[@]}"
