#!/bin/bash
# [telecom_boed] One-time Perlmutter setup, run on a LOGIN node (no allocation charge). Idempotent.
#   - tau2-bench pinned clone (read-only dependency; never edit): $EXT/tau2-bench @ TAU2_SHA
#   - env ddl-telecom (py3.12): tau2 client + our code (tau2 needs python >=3.12; the `ddl` env is 3.11)
#   - env ddl-vllm    (py3.12): vLLM OpenAI-compatible server (own torch/fastapi; kept apart from tau2's litellm pins)
#   - model weights into HF_HOME on $SCRATCH (Qwen/Qwen3-8B)
# Launch detached (FC-002):  setsid nohup bash telecom_boed/scripts/setup_perlmutter.sh > <log> 2>&1 < /dev/null &
set -euo pipefail
TAU2_SHA="${TAU2_SHA:-4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699}"
ENVS="${ENVS:-/global/cfs/cdirs/m4431/kbmkbm/envs}"
EXT="${EXT:-/global/cfs/cdirs/m4431/kbmkbm/ext}"
export HF_HOME="${HF_HOME:-${SCRATCH}/ddl/hf_cache}"
MODELS="${MODELS:-Qwen/Qwen3-8B}"
OUT="${SCRATCH}/ddl/telecom_boed/setup"
mkdir -p "${EXT}" "${HF_HOME}" "${OUT}"
module load conda
echo "== $(date -Is) tau2-bench @ ${TAU2_SHA}"
if [[ ! -d "${EXT}/tau2-bench/.git" ]]; then
  git init -q "${EXT}/tau2-bench"
  git -C "${EXT}/tau2-bench" remote add origin https://github.com/sierra-research/tau2-bench
fi
git -C "${EXT}/tau2-bench" fetch -q --depth 1 origin "${TAU2_SHA}"
git -C "${EXT}/tau2-bench" checkout -q "${TAU2_SHA}"
[[ "$(git -C "${EXT}/tau2-bench" rev-parse HEAD)" == "${TAU2_SHA}" ]] || { echo "tau2 commit mismatch" >&2; exit 1; }
for e in ddl-telecom ddl-vllm; do
  [[ -x "${ENVS}/${e}/bin/python" ]] || conda create -y -q -p "${ENVS}/${e}" python=3.12
done
echo "== $(date -Is) pip: ddl-telecom"
"${ENVS}/ddl-telecom/bin/pip" install -q -e "${EXT}/tau2-bench" 'websockets>=13'   # tau2 imports voice code at CLI import (W0-H1, 2026-10-16)
echo "== $(date -Is) pip: ddl-vllm"
"${ENVS}/ddl-vllm/bin/pip" install -q vllm "huggingface_hub[cli]"
echo "== $(date -Is) weights -> ${HF_HOME}"
for m in ${MODELS}; do "${ENVS}/ddl-vllm/bin/hf" download "${m}" > /dev/null; done
{
  echo "date=$(date -Is) tau2=$(git -C "${EXT}/tau2-bench" rev-parse HEAD)"
  "${ENVS}/ddl-telecom/bin/python" -c "import sys,tau2,litellm,importlib.metadata as m;print('ddl-telecom python',sys.version.split()[0],'tau2',m.version('tau2'),'litellm',m.version('litellm'))"
  "${ENVS}/ddl-vllm/bin/python" -c "import sys,vllm,torch;print('ddl-vllm python',sys.version.split()[0],'vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"
  du -sh "${HF_HOME}" 2>/dev/null
} | tee "${OUT}/versions.txt"
echo "== $(date -Is) setup done"
