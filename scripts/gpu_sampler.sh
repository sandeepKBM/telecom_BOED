#!/usr/bin/env bash
# [telecom_boed] Sample GPU utilisation/memory/power to a CSV for the duration of a job (read by eval/systems.py).
#   scripts/gpu_sampler.sh OUT.csv [INTERVAL_S=5] [GPU_IDS=all] &   ; SAMPLER=$! ; ... ; kill $SAMPLER
# Columns (no header): timestamp,index,util_gpu_%,util_mem_%,mem_used_MiB,power_W. Exits when its parent dies.
set -u
OUT="${1:?usage: gpu_sampler.sh OUT.csv [INTERVAL_S] [GPU_IDS]}"
INT="${2:-5}"
IDS="${3:-}"
PPID0="${PPID}"
ARGS=(--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits)
[ -n "${IDS}" ] && ARGS+=(-i "${IDS}")
command -v nvidia-smi >/dev/null 2>&1 || { echo "gpu_sampler: nvidia-smi not found" >&2; exit 0; }
mkdir -p "$(dirname "${OUT}")"
trap 'exit 0' TERM INT
while kill -0 "${PPID0}" 2>/dev/null; do
  nvidia-smi "${ARGS[@]}" 2>/dev/null | sed 's/, */,/g' >> "${OUT}"
  sleep "${INT}"
done
