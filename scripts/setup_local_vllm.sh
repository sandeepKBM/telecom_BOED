#!/bin/bash
# [telecom_boed] One-time LOCAL (ilab/rlab/westeros) setup of the vLLM server env + model weights. Idempotent. CPU only.
#   - env ddl-vllm (py3.12) at /common/users/ss5772/conda_envs/ddl-vllm, installed under the Perlmutter pins
#     (constraints/perlmutter_freeze_ddl-vllm.txt: vllm==0.31.0, torch==2.13.0, transformers==5.17.0). Never unpin: if pip
#     cannot satisfy the constraints this script fails and the pins must be revisited by a human.
#   - Qwen/Qwen3-8B at the pinned revision (same snapshot as Perlmutter) -> HF_HOME on NFS (~16 GB). NFS copy is only a
#     source: GPU jobs must stage it to node-local disk (scripts/stage_weights_local.sh) - never mmap weights from NFS.
#   - writes results/env/versions_local_vllm.txt
# Run on westeros (detached):  setsid nohup bash telecom_boed/scripts/setup_local_vllm.sh > <log> 2>&1 < /dev/null &
# Knobs: ENV_PREFIX, HF_HOME, MODEL, REVISION.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-/common/users/ss5772/miniforge3/bin/conda}"
ENV_PREFIX="${ENV_PREFIX:-/common/users/ss5772/conda_envs/ddl-vllm}"
export HF_HOME="${HF_HOME:-/filer/tmp1/ss5772/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"; unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE   # keep all three consistent
MODEL="${MODEL:-Qwen/Qwen3-8B}"
REVISION="${REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
CONSTRAINTS="${HERE}/constraints/perlmutter_freeze_ddl-vllm.txt"
OUT="${HERE}/results/env/versions_local_vllm.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1
mkdir -p "${HF_HOME}" "$(dirname "${OUT}")"
echo "== $(date -Is) env ${ENV_PREFIX}"
[[ -x "${ENV_PREFIX}/bin/python" ]] || "${CONDA}" create -y -q -p "${ENV_PREFIX}" python=3.12
echo "== $(date -Is) pip install (constrained)"
"${ENV_PREFIX}/bin/pip" install -q -c "${CONSTRAINTS}" vllm==0.31.0 "huggingface_hub[cli]"
echo "== $(date -Is) weights ${MODEL}@${REVISION} -> ${HF_HOME}"
"${ENV_PREFIX}/bin/hf" download "${MODEL}" --revision "${REVISION}" > /dev/null
SNAP="${HF_HOME}/hub/models--${MODEL//\//--}/snapshots/${REVISION}"
[[ -d "${SNAP}" ]] || { echo "snapshot dir missing: ${SNAP}" >&2; exit 1; }
{
  echo "date=$(date -Is) host=$(hostname) constraints=$(basename "${CONSTRAINTS}") (vllm==0.31.0 torch==2.13.0 transformers==5.17.0)"
  echo "python $("${ENV_PREFIX}/bin/python" -c 'import sys;print(sys.version.split()[0])')"
  "${ENV_PREFIX}/bin/python" -I -c "import importlib.metadata as m;print('vllm',m.version('vllm'),'torch',m.version('torch'),'transformers',m.version('transformers'),'huggingface_hub',m.version('huggingface_hub'))"
  "${ENV_PREFIX}/bin/python" -I -c "import torch;print('torch.version.cuda',torch.version.cuda,'cuda_available',torch.cuda.is_available())" 2>&1 | tail -n 1
  if "${ENV_PREFIX}/bin/python" -I -c "import vllm" > /dev/null 2>&1; then echo "import vllm: ok"; else echo "import vllm: FAILED on $(hostname)"; fi
  echo "model=${MODEL} revision=${REVISION}"
  echo "snapshot=${SNAP}"
  ls -lL "${SNAP}" | tail -n +2
  echo "total_size_bytes=$(du -scbL "${SNAP}" | tail -n 1 | cut -f1) ($(du -shL "${SNAP}" | cut -f1))"
} | tee "${OUT}"
echo "== $(date -Is) setup done"
