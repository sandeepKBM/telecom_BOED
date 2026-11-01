#!/bin/bash
# [telecom_boed] One-time LOCAL (westeros, CPU-only) setup. Idempotent. Mirrors setup_perlmutter.sh for the ddl-telecom env only.
#   - tau2-bench pinned clone (read-only dependency; never edit): $EXT/tau2-bench @ TAU2_SHA
#   - env ddl-telecom (py3.12): tau2 client + our code, pinned via constraints/perlmutter_freeze_ddl-telecom.txt
#   - results/env/versions_local.txt: versions + data md5s for comparison with the Perlmutter env
# Never runs create_tasks.py and never writes inside the clone. No GPU, no Slurm.
set -euo pipefail
TAU2_SHA="${TAU2_SHA:-4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699}"
REPO="${REPO:-/common/users/ss5772/ddl}"
PKG="${REPO}/telecom_boed"
CONSTRAINTS="${PKG}/constraints/perlmutter_freeze_ddl-telecom.txt"
ENVS="${ENVS:-/common/users/ss5772/conda_envs}"
EXT="${EXT:-/common/users/ss5772/external}"
CONDA="${CONDA:-/common/users/ss5772/miniforge3/bin/conda}"
ENV="${ENVS}/ddl-telecom"
OUT="${PKG}/results/env"
CLONE="${EXT}/tau2-bench"
mkdir -p "${EXT}" "${ENVS}" "${OUT}"
echo "== $(date -Is) tau2-bench @ ${TAU2_SHA}"
if [[ ! -d "${CLONE}/.git" ]]; then
  git init -q "${CLONE}"
  git -C "${CLONE}" remote add origin https://github.com/sierra-research/tau2-bench
fi
if [[ "$(git -C "${CLONE}" rev-parse HEAD 2>/dev/null || true)" != "${TAU2_SHA}" ]]; then
  git -C "${CLONE}" fetch -q --depth 1 origin "${TAU2_SHA}"
  git -C "${CLONE}" checkout -q "${TAU2_SHA}"
fi
[[ "$(git -C "${CLONE}" rev-parse HEAD)" == "${TAU2_SHA}" ]] || { echo "tau2 commit mismatch" >&2; exit 1; }
[[ -x "${ENV}/bin/python" ]] || "${CONDA}" create -y -q -p "${ENV}" python=3.12
echo "== $(date -Is) pip: ddl-telecom (constrained)"
"${ENV}/bin/pip" install -q -c "${CONSTRAINTS}" -e "${CLONE}" websockets pytest pyyaml   # websockets: tau2 CLI imports voice code; missing from tau2 base install
echo "== $(date -Is) pip: telecom_boed"
"${ENV}/bin/pip" install -q -e "${PKG}"
{
  echo "date=$(date -Is)"
  echo "tau2_clone=${CLONE}"
  echo "tau2_head=$(git -C "${CLONE}" rev-parse HEAD)"
  echo "tau2_porcelain_lines=$(git -C "${CLONE}" status --porcelain | wc -l)"
  echo "python=$("${ENV}/bin/python" -c 'import sys;print(sys.version.split()[0])')"
  "${ENV}/bin/pip" list --format=freeze | grep -E '^(tau2|litellm|pydantic|websockets|pytest|PyYAML|numpy)==' || true
  echo "md5 tasks.json=$(md5sum "${CLONE}/data/tau2/domains/telecom/tasks.json" | cut -d' ' -f1)"
  echo "md5 split_tasks.json=$(md5sum "${CLONE}/data/tau2/domains/telecom/split_tasks.json" | cut -d' ' -f1)"
} | tee "${OUT}/versions_local.txt"
echo "== $(date -Is) setup done"
