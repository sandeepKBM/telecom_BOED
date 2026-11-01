#!/bin/bash
# [telecom_boed] W3-L2 on Perlmutter: three run configs through telecom_boed.tau2_ext.launch --config, ONE shared vLLM server,
# arms run sequentially (one startup). New named variant of the T1-proven slurm/tau2_vllm.sbatch (that file is untouched);
# serving block is identical to it: node-local caches, VLLM_USE_FLASHINFER_SAMPLER=0, HEALTH_TIMEOUT_S, process-group kill, -C gpu&hbm40g.
# Serving config = T1 DEFAULT (cuda graphs + prefix cache, non-batch-invariant). DETERMINISTIC=1 (via --export) switches to
# VLLM_BATCH_INVARIANT=1 + --enforce-eager + no prefix cache (w3d1 finding; ~3x slower decode); recorded in serving_config.json per arm.
# Knobs (--export): RUN_TAG (l2pm), CONFIGS (space-separated run configs), HEALTH_TIMEOUT_S (1500), TAU2_TIMEOUT_S (per arm, 900),
#   DETERMINISTIC (0), MODEL (Qwen/Qwen3-8B), RSS_INTERVAL_S (60).
# Outputs: $SCRATCH/ddl/telecom_boed/runs/<RUN_TAG>_<jobid>/<config>/{results.json,sidecar.jsonl,run_meta.json,events.jsonl,serving_config.json,tau2.log}
#          and <run>/rss.log (RSS of vLLM + tau2 client every RSS_INTERVAL_S), vllm.log, job_info.txt.
#SBATCH -A m4431_g
#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH -t 01:15:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=55G
#SBATCH --gpus=1
#SBATCH -J l2pm
#SBATCH -o telecom_boed/logs/%x-%j.out
#SBATCH -e telecom_boed/logs/%x-%j.err
set -euo pipefail
CFS=/global/cfs/cdirs/m4431/kbmkbm
ENVS=${CFS}/envs
TAU2=${CFS}/ext/tau2-bench
PKG=${CFS}/ddl/telecom_boed
TAU2_SHA=4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699
MODEL="${MODEL:-Qwen/Qwen3-8B}"
CONFIGS="${CONFIGS:-l2pm_boedplain_native_train5 l2pm_boedctrl_reach_native_train5 l2pm_boedplain_noncompliant_train5}"
RUN="${SCRATCH}/ddl/telecom_boed/runs/${RUN_TAG:-l2pm}_${SLURM_JOB_ID}"
mkdir -p "${RUN}"
export HF_HOME="${SCRATCH}/ddl/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NODE_CACHE="/tmp/kbmkbm_tel_${SLURM_JOB_ID}"; mkdir -p "${NODE_CACHE}"
export VLLM_CACHE_ROOT="${NODE_CACHE}/vllm" XDG_CACHE_HOME="${NODE_CACHE}/xdg" FLASHINFER_WORKSPACE_BASE="${NODE_CACHE}/flashinfer" \
       TRITON_CACHE_DIR="${NODE_CACHE}/triton" TORCHINDUCTOR_CACHE_DIR="${NODE_CACHE}/inductor" TMPDIR="${NODE_CACHE}/tmp"
mkdir -p "${TMPDIR}"
export HOME="${NODE_CACHE}/home" VLLM_CONFIG_ROOT="${NODE_CACHE}/vllm_config" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_DO_NOT_TRACK=1
mkdir -p "${HOME}" "${VLLM_CONFIG_ROOT}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
PREFIX_FLAG=--enable-prefix-caching; EXTRA=""
if [[ "${DETERMINISTIC:-0}" == "1" ]]; then export VLLM_BATCH_INVARIANT=1; EXTRA="--enforce-eager"; PREFIX_FLAG=--no-enable-prefix-caching; fi
[[ "$(git -C "${TAU2}" rev-parse HEAD)" == "${TAU2_SHA}" ]] || { echo "tau2 clone not at ${TAU2_SHA}" >&2; exit 1; }
cp -r "${TAU2}/data" "${RUN}/tau2_data"
export TAU2_DATA_DIR="${RUN}/tau2_data"
export PYTHONPATH="${PKG}/src${PYTHONPATH:+:${PYTHONPATH}}"
EV() { python3 -I "${PKG}/scripts/run_events.py" emit --out "$1" --run "$2" "${@:3}" 2>/dev/null || true; }

{ # provenance
  echo "job=${SLURM_JOB_ID} node=$(hostname) model=${MODEL} configs=${CONFIGS} deterministic=${DETERMINISTIC:-0} mem_req=55G"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  "${ENVS}/ddl-vllm/bin/python" -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
  "${ENVS}/ddl-telecom/bin/python" -c "import importlib.metadata as m;print('tau2',m.version('tau2'),'litellm',m.version('litellm'))"
  echo "tau2_sha=${TAU2_SHA}"; md5sum "${TAU2_DATA_DIR}/tau2/domains/telecom/tasks.json" "${TAU2_DATA_DIR}/tau2/domains/telecom/split_tasks.json"
  echo "serving: prefix=${PREFIX_FLAG} extra='${EXTRA}' batch_invariant=${VLLM_BATCH_INVARIANT:-0} flashinfer_sampler=${VLLM_USE_FLASHINFER_SAMPLER} max_model_len=32768 gpu_util=0.90 seed=0"
} 2>&1 | tee "${RUN}/job_info.txt"

PORT=$((8000 + SLURM_JOB_ID % 1000))
t0=${SECONDS}
setsid "${ENVS}/ddl-vllm/bin/vllm" serve "${MODEL}" --host 127.0.0.1 --port "${PORT}" --served-model-name "${MODEL}" \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes ${PREFIX_FLAG} --seed 0 ${EXTRA} \
  > "${RUN}/vllm.log" 2>&1 &
VPID=$!
RPID=
trap 'kill ${RPID} 2>/dev/null || true; kill -TERM -- -${VPID} 2>/dev/null || kill ${VPID} 2>/dev/null || true; sleep 5; kill -KILL -- -${VPID} 2>/dev/null || true' EXIT
# RSS logger (host memory growth source: h7nat OOM). KB; vllm tree = all processes in the vLLM process group; client = launch.py.
( while true; do
    v=$(ps -o rss= -g "${VPID}" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    c=$(ps -eo rss=,args= 2>/dev/null | awk '/telecom_boed.tau2_ext.launch/ && !/awk/ {s+=$1} END{print s+0}')
    echo "$(date -Is) vllm_group_rss_kb=${v} tau2_client_rss_kb=${c} job_cgroup_kb=$(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0) / 1024 ))" >> "${RUN}/rss.log"
    sleep "${RSS_INTERVAL_S:-60}"
  done ) &
RPID=$!
for i in $(seq 1 $(( ${HEALTH_TIMEOUT_S:-1500} / 5 ))); do
  curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null && break
  kill -0 ${VPID} 2>/dev/null || { echo "vLLM died during startup" >&2; tail -n 40 "${RUN}/vllm.log" >&2; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null || { echo "vLLM not healthy after ${HEALTH_TIMEOUT_S:-1500}s" >&2; exit 1; }
echo "vllm_ready_s=$((SECONDS - t0))" | tee -a "${RUN}/job_info.txt"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/gpu_mem_after_load=/' | tee -a "${RUN}/job_info.txt"
export VLLM_API_BASE="http://127.0.0.1:${PORT}/v1"

worst=0
for CFG in ${CONFIGS}; do
  A="${RUN}/${CFG}"; mkdir -p "${A}"
  export TELECOM_RUN_DIR="${A}" TAU2_SAVE_TO="${CFG}_${SLURM_JOB_ID}"
  printf '{"job":"%s","config":"%s","prefix_cache":"%s","extra_args":"%s","batch_invariant":"%s","flashinfer_sampler":"%s","deterministic":"%s","vllm_seed":0,"max_model_len":32768,"gpu":"A100-40GB","mem_req":"55G"}\n' \
    "${SLURM_JOB_ID}" "${CFG}" "${PREFIX_FLAG}" "${EXTRA}" "${VLLM_BATCH_INVARIANT:-0}" "${VLLM_USE_FLASHINFER_SAMPLER}" "${DETERMINISTIC:-0}" > "${A}/serving_config.json"
  EV "${A}/events.jsonl" "${CFG}" --phase step_start host="$(hostname)" job="${SLURM_JOB_ID}" mode=launch config="${CFG}"
  python3 -I "${PKG}/scripts/run_events.py" watch --out "${A}/events.jsonl" --run "${CFG}" --results "${TAU2_DATA_DIR}/simulations/${TAU2_SAVE_TO}/results.json" \
    --metrics "${VLLM_API_BASE%/v1}/metrics" --gpu 0 --ppid $$ --interval 60 --heartbeat 300 &
  WPID=$!
  EV "${A}/events.jsonl" "${CFG}" --phase tau2_start
  t1=${SECONDS}
  set +e
  timeout -k 30 "${TAU2_TIMEOUT_S:-900}" "${ENVS}/ddl-telecom/bin/python" -m telecom_boed.tau2_ext.launch --config "${CFG}.yaml" > "${A}/tau2.log" 2>&1
  rc=$?
  set -e
  echo "arm=${CFG} tau2_rc=${rc} tau2_wall_s=$((SECONDS - t1))" | tee -a "${RUN}/job_info.txt"
  kill -TERM "${WPID}" 2>/dev/null || true; wait "${WPID}" 2>/dev/null || true
  EV "${A}/events.jsonl" "${CFG}" --phase tau2_end rc="${rc}" wall_s=$((SECONDS - t1))
  curl -sf "${VLLM_API_BASE%/v1}/metrics" > "${A}/vllm_metrics.prom" || true
  cp "${TAU2_DATA_DIR}/simulations/${TAU2_SAVE_TO}/results.json" "${A}/results.json" 2>/dev/null || true
  tail -n 8 "${A}/tau2.log" || true
  [[ ${rc} -gt ${worst} ]] && worst=${rc}
done
echo "all_arms_done worst_rc=${worst} job_wall_s=${SECONDS}" | tee -a "${RUN}/job_info.txt"
exit "${worst}"
