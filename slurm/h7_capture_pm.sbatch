#!/bin/bash
# [telecom_boed] H7 capture on Perlmutter (new named variant of slurm/h7_capture.sh, which is the rlab2 step script and stays as is).
# Qwen3-8B (bf16, HF transformers, no vLLM) re-encodes the logged prompts and stores the last-prompt-token hidden state of all
# 37 layers + next-token entropy/top1 per agent decision point (telecom_boed/internals/capture.py). Teacher-forced: no generation.
# Inputs : $SCRATCH/ddl/telecom_boed/h7_prompts/<run>/prompts.jsonl (built on westeros by scripts/h7_build_prompts.py, rsynced).
# Outputs: $SCRATCH/ddl/telecom_boed/h7/<run>/{index.jsonl,summary.json,shards/*.safetensors}  (one dir per run: shard names
#          are task_id__trial only). Resumable per episode (capture_run skips episodes already in index.jsonl).
# Knobs (--export): RUNS (space-separated run names; default = every dir under h7_prompts), LIMIT_EPS (per run), VERIFY (K per episode,
#          recompute K prefix-served turns with their own forward and report the diff), CAPTURE_MODE (prefix|full).
# Smoke:  sbatch -t 00:20:00 --export=ALL,RUNS=l2pm_boedplain_native_train5_60432276,LIMIT_EPS=2,VERIFY=2 telecom_boed/slurm/h7_capture_pm.sbatch
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=55G
#SBATCH --gpus=1
#SBATCH -J h7cap
#SBATCH -o telecom_boed/logs/%x-%j.out
#SBATCH -e telecom_boed/logs/%x-%j.err
set -euo pipefail
CFS=/global/cfs/cdirs/m4431/kbmkbm
PY=${CFS}/envs/ddl-vllm/bin/python
PKG=${CFS}/ddl/telecom_boed
PROMPTS_ROOT="${SCRATCH}/ddl/telecom_boed/h7_prompts"
OUT_ROOT="${SCRATCH}/ddl/telecom_boed/h7"
export HF_HOME="${SCRATCH}/ddl/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NODE_CACHE="/tmp/kbmkbm_h7_${SLURM_JOB_ID}"; mkdir -p "${NODE_CACHE}/home" "${NODE_CACHE}/tmp"
export HOME="${NODE_CACHE}/home" TMPDIR="${NODE_CACHE}/tmp" XDG_CACHE_HOME="${NODE_CACHE}/xdg" TRITON_CACHE_DIR="${NODE_CACHE}/triton"
trap 'rm -rf "${NODE_CACHE}"' EXIT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 DO_NOT_TRACK=1
RUNS="${RUNS:-$(ls "${PROMPTS_ROOT}")}"
mkdir -p "${OUT_ROOT}"
{ # provenance
  echo "job=${SLURM_JOB_ID} node=$(hostname) runs=$(echo ${RUNS} | wc -w) limit_eps=${LIMIT_EPS:-} verify=${VERIFY:-0} mode=${CAPTURE_MODE:-prefix}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  "${PY}" -c "import torch,transformers,safetensors;print('torch',torch.__version__,'transformers',transformers.__version__,'safetensors',safetensors.__version__)"
  md5sum "${PKG}/src/telecom_boed/internals/capture.py" "${PKG}/src/telecom_boed/internals/prompts.py" "${PKG}/scripts/h7_capture.py"
} 2>&1 | tee "${OUT_ROOT}/job_info_${SLURM_JOB_ID}.txt"
JOBS=()
for r in ${RUNS}; do
  P="${PROMPTS_ROOT}/${r}/prompts.jsonl"
  [[ -s "${P}" ]] || { echo "[h7cap] skip ${r}: no prompts"; continue; }
  JOBS+=(--job "${P}" "${OUT_ROOT}/${r}")
done
(( ${#JOBS[@]} )) || { echo "[h7cap] no runs with prompts under ${PROMPTS_ROOT}" >&2; exit 1; }
ARGS=()
[[ -n "${LIMIT_EPS:-}" ]] && ARGS+=(--limit-eps "${LIMIT_EPS}")
[[ -n "${VERIFY:-}" ]] && ARGS+=(--verify-alone "${VERIFY}")
[[ -n "${CAPTURE_MODE:-}" ]] && ARGS+=(--mode "${CAPTURE_MODE}")
echo "[h7cap] start=$(date -Is) jobs=$(( ${#JOBS[@]} / 3 ))"
set +e
"${PY}" -I "${PKG}/scripts/h7_capture.py" "${JOBS[@]}" "${ARGS[@]}"; rc=$?
set -e
echo "[h7cap] end=$(date -Is) rc=${rc}"
exit ${rc}
