#!/bin/bash
# [telecom_boed] H7-B steering on Perlmutter (scripts/h7_steer.py). Qwen3-8B bf16, HF transformers, 1 A100, no vLLM.
# Inputs : $SCRATCH/ddl/telecom_boed/h7_steer_prep/{directions.npz,eval_turns.jsonl} (scripts/h7_steer_prep.py on westeros, rsynced)
#          $SCRATCH/ddl/telecom_boed/h7_prompts/<run>/prompts.jsonl (same prompts the H7 capture used)
# Outputs: $SCRATCH/ddl/telecom_boed/h7_steer/<OUT_NAME>/{b1.jsonl,b2.jsonl,summary.json}; resumable (done turns skipped).
# Knobs (--export): OUT_NAME (default full), LIMIT (turns), GEN_N (turns with generation), MAX_MIN (clean stop, minutes),
#          SHARD=I/N (turns with index % N == I; give each shard its own OUT_NAME, e.g. full_s0 .. full_s3).
# Full (4 shards): for i in 0 1 2 3; do sbatch -t 01:00:00 --export=ALL,OUT_NAME=full_s$i,SHARD=$i/4,MAX_MIN=50 telecom_boed/slurm/h7_steer_pm.sbatch; done
# Smoke:  sbatch -q debug -t 00:15:00 --export=ALL,OUT_NAME=smoke,LIMIT=2,GEN_N=1,MAX_MIN=6 telecom_boed/slurm/h7_steer_pm.sbatch
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:45:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=55G
#SBATCH --gpus=1
#SBATCH -J h7steer
#SBATCH -o telecom_boed/logs/%x-%j.out
#SBATCH -e telecom_boed/logs/%x-%j.err
set -euo pipefail
CFS=/global/cfs/cdirs/m4431/kbmkbm
PY=${CFS}/envs/ddl-vllm/bin/python
PKG=${CFS}/ddl/telecom_boed
PREP="${SCRATCH}/ddl/telecom_boed/h7_steer_prep"
PROMPTS_ROOT="${SCRATCH}/ddl/telecom_boed/h7_prompts"
OUT="${SCRATCH}/ddl/telecom_boed/h7_steer/${OUT_NAME:-full}"
export HF_HOME="${SCRATCH}/ddl/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NODE_CACHE="/tmp/kbmkbm_h7s_${SLURM_JOB_ID}"; mkdir -p "${NODE_CACHE}/home" "${NODE_CACHE}/tmp"
export HOME="${NODE_CACHE}/home" TMPDIR="${NODE_CACHE}/tmp" XDG_CACHE_HOME="${NODE_CACHE}/xdg" TRITON_CACHE_DIR="${NODE_CACHE}/triton"
trap 'rm -rf "${NODE_CACHE}"' EXIT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 DO_NOT_TRACK=1
mkdir -p "${OUT}"
{ # provenance
  echo "job=${SLURM_JOB_ID} node=$(hostname) out=${OUT} limit=${LIMIT:-} gen_n=${GEN_N:-48} max_min=${MAX_MIN:-75} shard=${SHARD:-0/1}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  "${PY}" -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"
  md5sum "${PKG}/scripts/h7_steer.py" "${PKG}/src/telecom_boed/internals/capture.py" "${PREP}/directions.npz" "${PREP}/eval_turns.jsonl"
} 2>&1 | tee "${OUT}/job_info_${SLURM_JOB_ID}.txt"
ARGS=(--prep "${PREP}" --prompts-root "${PROMPTS_ROOT}" --out "${OUT}" --gen-n "${GEN_N:-48}" --max-minutes "${MAX_MIN:-75}" --shard "${SHARD:-0/1}")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")
echo "[h7steer] start=$(date -Is)"
set +e
"${PY}" -X faulthandler -I "${PKG}/scripts/h7_steer.py" "${ARGS[@]}"; rc=$?
set -e
echo "[h7steer] end=$(date -Is) rc=${rc}"
exit ${rc}
