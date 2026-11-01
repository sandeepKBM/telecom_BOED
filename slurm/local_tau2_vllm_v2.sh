#!/bin/bash
# [telecom_boed] v2 (P0-4, 2026-10-31): HARDENED COPY of slurm/local_tau2_vllm.sh (original untouched). Differences:
#   - python -X faulthandler for the client, PYTHONFAULTHANDLER=1 for vLLM (a native crash prints its stack);
#   - STARTUP_RETRIES (default 2): the vLLM server is restarted only if it died with a segfault BEFORE weight load (FC-013:
#     rc 139 / "Segmentation fault" in vllm.log and no "Loading model"/"Loading safetensors" line); any other failure is final;
#   - FC-011 guard: refuses cache dirs on Lustre/NFS/CFS paths (flock unsupported there); adds FLASHINFER_WORKSPACE_BASE node-local;
#   - worker recycling (MODE=launch): run_client below; rss.log (vLLM group + client RSS every RSS_INTERVAL_S, default 60);
#   - optional GPU_SAMPLER=1 -> gpu_samples.csv via scripts/gpu_sampler.sh; DRY_RUN=1 prints the plan and exits (no GPU, no job).
#   Extra knobs: STARTUP_RETRIES (2), RSS_CAP_KB (16777216), MAX_RECYCLES (5), RECYCLE_EVERY_S (0=off), RSS_INTERVAL_S (60).
#   Resubmit/launch commands in the header below apply with this file name. The self-snapshot to node-local disk is kept.
# [telecom_boed] LOCAL (ilab/rlab Slurm GPU) twin of slurm/tau2_vllm.sbatch: tau2 telecom episodes against a self-hosted vLLM
# (Qwen/Qwen3-8B, non-thinking; agent and user simulator share one server) on ONE node-local GPU. No API models.
# This is a STEP SCRIPT (not an sbatch file): it runs inside an existing allocation, picks a free GPU itself via flock,
# works on node-local disk, and copies only small artifacts back to NFS. Same vLLM flags / tau2 knobs / defaults as Perlmutter.
#
# Submit INSIDE the GPU hold (job 526576 on rlab2, 4x A100-PCIE-40GB, shared with other projects; every --overlap step sees GPU 0,
# so this script chooses the GPU by taking /freespace/local/ss5772/gpu_locks/gpu<N>.lock with flock -n):
#   ssh rlab1 "srun --jobid=526576 --overlap --gres=gpu:a100:4 --mem=48G -t 00:30:00 \
#     --export=ALL,RUN_TAG=smoke,NUM_TASKS=3 bash /common/users/ss5772/ddl/telecom_boed/slurm/local_tau2_vllm.sh"
#   (--gres=gpu:a100:4 is requested so the step may lock any of GPU 0-3; GPU_INDEX=2 forces a specific one - still locked.)
# Standalone, once the hold is gone (one GPU; #SBATCH lines cannot expand vars, so the type goes on the command line):
#   ssh rlab1 "cd /common/users/ss5772/ddl && sbatch --gres=gpu:a100:1 --export=ALL,RUN_TAG=smoke telecom_boed/slurm/local_tau2_vllm.sbatch"
#
# Determinism knobs (env): VLLM_BATCH_INVARIANT=1, VLLM_EXTRA_ARGS="--enforce-eager", VLLM_PREFIX_CACHE_FLAG=--no-enable-prefix-caching.
# Knobs (env): RUN_TAG (tau2), MODEL (Qwen/Qwen3-8B), SPLIT (train), NUM_TASKS (3), TASK_IDS (space-separated, overrides NUM_TASKS),
#   NUM_TRIALS (1), MAX_CONC (3), MAX_STEPS (200), SEED (300), MAX_MODEL_LEN (32768), GPU_UTIL (0.90)  [identical to Perlmutter];
#   MODE (raw = tau2 CLI | launch = python -m telecom_boed.tau2_ext.launch --config $RUN_CONFIG), RUN_CONFIG (launch mode),
#   GPU_INDEX (force GPU), ETA_MIN (30, written into the GPU owner file), HEALTH_TIMEOUT_S (600), TAU2_TIMEOUT_S (1500),
#   RESULT_DIR (NFS, default <repo>/telecom_boed/results/runs/<RUN_TAG>_<job>.<step>), LOCAL_ROOT (/freespace/local/ss5772),
#   FORCE_BUSY=1 (skip the "GPU must be idle" check), KEEP_WORK=1 (do not delete the node-local run dir).
# Node-local run dir: $LOCAL_ROOT/telecom_boed/runs/<RUN_TAG>_<job>.<step>/ (vLLM/XDG/triton caches, tau2 data copy, logs); deleted
# at exit. Weights: $LOCAL_ROOT/hf_cache (staged + sha256-verified by scripts/stage_weights_local.sh; kept between runs).
# Exit codes: 3 no free GPU lock, 4 GPU not idle, 5 launch.py missing (MODE=launch), 1 other setup/vLLM failure, else tau2's rc.
set -euo pipefail
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN local_tau2_vllm_v2: MODE=${MODE:-raw} RUN_TAG=${RUN_TAG:-tau2} STARTUP_RETRIES=${STARTUP_RETRIES:-2} RSS_CAP_KB=${RSS_CAP_KB:-16777216} MAX_RECYCLES=${MAX_RECYCLES:-5} RECYCLE_EVERY_S=${RECYCLE_EVERY_S:-0} RSS_INTERVAL_S=${RSS_INTERVAL_S:-60} GPU_SAMPLER=${GPU_SAMPLER:-0}"
  echo "DRY_RUN client: python -X faulthandler -m telecom_boed.tau2_ext.launch ... (MODE=launch recycles via run_client; patched/raw do not)"
  echo "DRY_RUN vllm: PYTHONFAULTHANDLER=1 vllm serve ${MODEL:-Qwen/Qwen3-8B} (retry only on pre-weight-load segfault, max ${STARTUP_RETRIES:-2})"
  exit 0
fi
T_START=${SECONDS}
HERE="${TELECOM_HERE:-$(cd "$(dirname "${CDX_GPU_LOCK_SCRIPT_ORIG:-${BASH_SOURCE[0]}}")/.." && pwd)}"   # under cdx-agent --gpu-lock: original path   # .../telecom_boed
# Run from a node-local snapshot of this script: bash reads scripts lazily, and when the NFS copy is replaced while a
# step runs, the reader on another client gets "Stale file handle" (P3 baseline step 526576.16 died that way).
if [[ -z "${TELECOM_LAUNCHER_COPY:-}" && -z "${CDX_GPU_LOCK_SCRIPT_ORIG:-}" ]]; then   # cdx-agent --gpu-lock already snapshots
  _snap="/freespace/local/ss5772/z/launcher_v2_${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-0}.$$.sh"
  mkdir -p "$(dirname "${_snap}")" && cp "${BASH_SOURCE[0]}" "${_snap}"
  TELECOM_HERE="${HERE}" TELECOM_LAUNCHER_COPY="${_snap}" exec bash "${_snap}" "$@"
fi
REPO="$(dirname "${HERE}")"                                       # .../ddl
ENVS="${ENVS:-/common/users/ss5772/conda_envs}"   # NFS: ddl-telecom (tau2 client, small, shared by westeros tests + nodes)
# ddl-vllm lives on NODE-LOCAL disk (5+ GB; /common/users quota is shared and was full on 2026-10-16). Build per node with
# scripts/rlab2_local_build.sh (or setup_local_vllm.sh ENV_PREFIX=...).
VLLM_ENV="${VLLM_ENV:-/freespace/local/ss5772/conda_envs/ddl-vllm}"
TAU2="${TAU2:-/common/users/ss5772/external/tau2-bench}"
TAU2_SHA=4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODE="${MODE:-raw}"
LOCAL_ROOT="${LOCAL_ROOT:-/freespace/local/ss5772}"
JOB="${SLURM_JOB_ID:-local$$}"; STEP="${SLURM_STEP_ID:-0}"
TAG="${RUN_TAG:-tau2}"; NAME="${TAG}_${JOB}.${STEP}"; SAVE_TO="${TAG}_${JOB}_${STEP}"
RUN="${LOCAL_ROOT}/telecom_boed/runs/${NAME}"
RESULT_DIR="${RESULT_DIR:-${HERE}/results/runs/${NAME}}"
# --- live machine-written progress: <RESULT_DIR>/events.jsonl (scripts/run_events.py; read by scripts/telecom_status.py) ---
mkdir -p "${RESULT_DIR}"; EVENTS="${RESULT_DIR}/events.jsonl"; WPID=
ev() { python3 -I "${HERE}/scripts/run_events.py" emit --out "${EVENTS}" --run "${NAME}" "$@" 2>/dev/null || true; }

# --- ilab/rlab hygiene (process limit ulimit -u 2000; node /tmp quota ~2 GB: nothing goes to /tmp) ---
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# DETERMINISTIC=1: batch-invariant kernels + eager + no prefix cache -> identical transcripts across launches/GPUs
# (W3-D1, results/determinism/w3d1_2026-10-16.md: 5/5 identical; ~3x slower decode). Use for every compared arm.
if [[ "${DETERMINISTIC:-0}" == "1" ]]; then
  export VLLM_BATCH_INVARIANT=1 VLLM_EXTRA_ARGS="--enforce-eager ${VLLM_EXTRA_ARGS:-}" VLLM_PREFIX_CACHE_FLAG=--no-enable-prefix-caching
fi
# keep every ~/ write off /common/home (quota 50 GB; vLLM usage_stats, flashinfer/triton caches landed there): per-step HOME
export HOME="${LOCAL_ROOT}/z/home_${JOB}.${STEP}" VLLM_CONFIG_ROOT="${LOCAL_ROOT}/z/home_${JOB}.${STEP}/.config/vllm" VLLM_DO_NOT_TRACK=1
mkdir -p "${VLLM_CONFIG_ROOT}"
[[ -x "${VLLM_ENV}/bin/vllm" ]] || { echo "missing ${VLLM_ENV} (run scripts/setup_local_vllm.sh)" >&2; exit 1; }
[[ -x "${ENVS}/ddl-telecom/bin/tau2" ]] || { echo "missing ${ENVS}/ddl-telecom/bin/tau2" >&2; exit 1; }
[[ "$(git -C "${TAU2}" rev-parse HEAD)" == "${TAU2_SHA}" ]] || { echo "tau2 clone not at ${TAU2_SHA}" >&2; exit 1; }

# --- GPU selection: flock one of /freespace/local/ss5772/gpu_locks/gpu<N>.lock; fd 9 stays open for the whole run ---
LOCKS="${LOCAL_ROOT}/gpu_locks"; mkdir -p "${LOCKS}"
G=
if [[ -n "${CDX_GPU_LOCK_WRAPPER:-}" ]]; then   # cdx-agent --gpu-lock holds the lock(s) and set CUDA_VISIBLE_DEVICES
  G="${CUDA_VISIBLE_DEVICES%%,*}"
fi
for c in $([[ -n "${G}" ]] || echo "${GPU_INDEX:-0 1 2 3}"); do
  exec 9>"${LOCKS}/gpu${c}.lock"
  if flock -n 9; then G=${c}; break; fi
  exec 9>&-
done
[[ -n "${G}" ]] || { echo "no free GPU lock under ${LOCKS} (GPU_INDEX='${GPU_INDEX:-}')" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=${G}
[[ -n "${CDX_GPU_LOCK_WRAPPER:-}" ]] || echo "$(date -Is) telecom_boed pid=$$ until=$(date -d "+${ETA_MIN:-30} min" -Is)" > "${LOCKS}/gpu${G}.owner"
T_EV=${SECONDS}
ev --phase step_start host="$(hostname)" job="${JOB}" step="${STEP}" gpu="${G}" mode="${MODE}" split="${SPLIT:-train}" trials="${NUM_TRIALS:-1}" conc="${MAX_CONC:-3}" seed="${SEED:-300}" max_steps="${MAX_STEPS:-200}" num_tasks="${NUM_TASKS:-}"

VPID=
cleanup() {
  rc=$?; set +e; trap - EXIT
  [[ -n "${WPID}" ]] && { kill -TERM "${WPID}" 2>/dev/null; wait "${WPID}" 2>/dev/null; }
  kill ${RSSPID:-} ${GPUSPID:-} 2>/dev/null
  ev --phase step_end exit_rc="${rc}" wall_s=$((SECONDS - ${T_EV:-0})) gpu_hours_est="$(awk -v w=$((SECONDS - ${T_EV:-0})) 'BEGIN{printf "%.3f", w/3600}')"
  if [[ -n "${VPID}" ]]; then
    kill -TERM -- "-${VPID}" 2>/dev/null || kill -TERM "${VPID}" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "${VPID}" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-${VPID}" 2>/dev/null; wait "${VPID}" 2>/dev/null
  fi
  if [[ -d "${RUN}" ]]; then
    wall=$((SECONDS - T_START))
    echo "exit_rc=${rc} step_wall_s=${wall} gpu_hours_est=$(awk -v w=${wall} 'BEGIN{printf "%.3f", w/3600}') (1 GPU x wall; gpu=${G})" >> "${RUN}/job_info.txt"
    mkdir -p "${RESULT_DIR}"
    for _f in "${RUN}"/vllm.attempt*.log; do [[ -e "${_f}" ]] && tail -n 80 "${_f}" > "${RESULT_DIR}/$(basename "${_f}").tail"; done 2>/dev/null
    cp "${RUN}/job_info.txt" "${RUN}/rss.log" "${RUN}/gpu_samples.csv" "${RUN}/results.json" "${RUN}/vllm_metrics.prom" "${RUN}/run_meta.json" "${RUN}/sidecar.jsonl" "${RESULT_DIR}/" 2>/dev/null
    tail -n 200 "${RUN}/tau2.log" > "${RESULT_DIR}/tau2.log.tail" 2>/dev/null
    tail -n 200 "${RUN}/vllm.log" > "${RESULT_DIR}/vllm.log.tail" 2>/dev/null
    head -n 400 "${RUN}/vllm.log" > "${RESULT_DIR}/vllm.log.head" 2>/dev/null   # startup: KV cache size, max concurrency
    echo "artifacts -> ${RESULT_DIR}"
    [[ -n "${KEEP_WORK:-}" ]] || rm -r -f "${RUN}"       # weights cache lives in ${LOCAL_ROOT}/hf_cache, untouched
    [[ -n "${TMPDIR:-}" && "${TMPDIR}" == "${LOCAL_ROOT}/z/"* ]] && rm -r -f "${TMPDIR}"   # short ipc tmp dir (outside RUN)
    [[ "${HOME}" == "${LOCAL_ROOT}/z/home_"* ]] && rm -r -f "${HOME}"            # per-step HOME
    [[ -n "${TELECOM_LAUNCHER_COPY:-}" && "${TELECOM_LAUNCHER_COPY}" == "${LOCAL_ROOT}/z/launcher_"* ]] && rm -f "${TELECOM_LAUNCHER_COPY}"
  fi
  grep -q "pid=$$ " "${LOCKS}/gpu${G}.owner" 2>/dev/null && rm -f "${LOCKS}/gpu${G}.owner"
  exec 9>&-                                   # releases the flock
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP

# >>> client_recycle (extracted and tested by tests/test_launchers_v2.py)
# Worker recycling. tau2 runs its pool as THREADS in one process (tau2/runner/worker.py ThreadPoolExecutor), so multiprocessing
# `maxtasksperchild` does not apply; the equivalent is recycling the whole client process. run_client LOG BUDGET_S CMD... runs the
# client in its own session; if the session RSS exceeds RSS_CAP_KB (default 16 GiB) or the process ran RECYCLE_EVERY_S (0 = off),
# it is SIGTERMed and restarted with TAU2_RESUME=1 (launch.py --config resumes: tau2 skips finished simulations; in-flight ones
# are redone). At most MAX_RECYCLES restarts (default 5; 0 disables). Sets globals RC (last client rc; 124 = budget spent) and RECYCLES.
# Call it with `set +e` active (it uses (( )) tests). RSS_LOG (optional) gets recycle events.
run_client() {
  local log=$1 budget=$2; shift 2
  local end=$((SECONDS + budget)) cpid left t_start rss recycled tick base_resume="${TAU2_RESUME:-}"
  RECYCLES=0; RC=0
  while :; do
    left=$((end - SECONDS))
    if (( left <= 0 )); then RC=124; break; fi
    if (( RECYCLES > 0 )); then export TAU2_RESUME=1; fi
    setsid timeout -k 30 "${left}" "$@" >> "${log}" 2>&1 &
    cpid=$!; t_start=${SECONDS}; recycled=0; tick=0
    while kill -0 "${cpid}" 2>/dev/null; do
      sleep 1; tick=$((tick + 1))
      (( tick % ${RECYCLE_POLL_S:-10} == 0 )) || continue
      rss=$(ps -o rss= -s "${cpid}" 2>/dev/null | awk '{s+=$1} END{print s+0}')
      if (( RECYCLES < ${MAX_RECYCLES:-5} )) && { (( rss > ${RSS_CAP_KB:-16777216} )) || { (( ${RECYCLE_EVERY_S:-0} > 0 )) && (( SECONDS - t_start > ${RECYCLE_EVERY_S:-0} )); }; }; then
        echo "$(date -Is) recycle client: rss_kb=${rss} ran_s=$((SECONDS - t_start)) restart=$((RECYCLES + 1))/${MAX_RECYCLES:-5}" | tee -a "${RSS_LOG:-/dev/null}"
        pkill -TERM -s "${cpid}" 2>/dev/null; sleep "${RECYCLE_GRACE_S:-15}"; pkill -KILL -s "${cpid}" 2>/dev/null
        recycled=1; break
      fi
    done
    RC=0; wait "${cpid}" 2>/dev/null || RC=$?
    if (( recycled )); then RECYCLES=$((RECYCLES + 1)); continue; fi
    break
  done
  if [[ -z "${base_resume}" ]]; then unset TAU2_RESUME; fi
  return 0
}
# <<< client_recycle

# --- GPU must be idle (we hold the lock, but other projects may not honour it) ---
used=$(nvidia-smi -i "${G}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [[ "${used}" -gt 1500 && -z "${FORCE_BUSY:-}" ]]; then
  echo "GPU ${G} holds ${used} MiB although its lock was free - not idle; refusing (FORCE_BUSY=1 to override)" >&2; exit 4
fi

# --- node-local run dir, caches, weights, tau2 data copy (tau2 writes results into TAU2_DATA_DIR; never the clone) ---
mkdir -p "${RUN}/tmp" "${RUN}/cache"
# TMPDIR must stay short: vLLM binds ZMQ ipc sockets under it and AF_UNIX paths max out at 107 chars (failed 2026-10-16, step 526576.15)
TMPDIR="${LOCAL_ROOT}/z/${JOB}.${STEP}"; mkdir -p "${TMPDIR}"
export TMPDIR VLLM_CACHE_ROOT="${RUN}/cache/vllm" XDG_CACHE_HOME="${RUN}/cache/xdg"
export TRITON_CACHE_DIR="${RUN}/cache/triton" TORCHINDUCTOR_CACHE_DIR="${RUN}/cache/inductor" FLASHINFER_WORKSPACE_BASE="${RUN}/cache/flashinfer"
export PYTHONFAULTHANDLER=1   # vllm / EngineCore native crashes print a stack (python -I would ignore it; the vllm CLI is not run with -I)
# FC-011 guard: JIT caches need flock; refuse network/Lustre/CFS paths
for _v in VLLM_CACHE_ROOT XDG_CACHE_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR FLASHINFER_WORKSPACE_BASE TMPDIR HOME; do
  case "${!_v}" in /pscratch/*|/global/*|/common/*|/home/*) echo "FC-011: ${_v}=${!_v} is on a network filesystem (no flock)" >&2; exit 1;; esac
done
export HF_HOME; HF_HOME="$(bash "${HERE}/scripts/stage_weights_local.sh")"
export HF_HUB_CACHE="${HF_HOME}/hub"; unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE   # keep consistent with HF_HOME
export PATH="${VLLM_ENV}/bin:${PATH}"
cp -r "${TAU2}/data" "${RUN}/tau2_data"
export TAU2_DATA_DIR="${RUN}/tau2_data"

{ # provenance
  echo "run=${NAME} job=${JOB} step=${STEP} host=$(hostname) date=$(date -Is) mode=${MODE} model=${MODEL} hf_home=${HF_HOME}"
  echo "split=${SPLIT:-train} num_tasks=${NUM_TASKS:-<raw:3,patched:all>} task_ids=${TASK_IDS:-} trials=${NUM_TRIALS:-1} conc=${MAX_CONC:-3} max_steps=${MAX_STEPS:-200} seed=${SEED:-300} max_model_len=${MAX_MODEL_LEN:-32768} gpu_util=${GPU_UTIL:-0.90}"
  echo "gpu_index=${G} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} mem_used_before_MiB=${used}"
  echo "vllm_env: $(env | grep -E '^VLLM_(BATCH_INVARIANT|ATTENTION_BACKEND|USE_|EXTRA_ARGS|PREFIX_CACHE_FLAG)' | sort | tr '\n' ' ')"
  nvidia-smi -i "${G}" --query-gpu=name,memory.total,driver_version --format=csv,noheader
  "${VLLM_ENV}/bin/python" -I -c "import importlib.metadata as m,torch;print('vllm',m.version('vllm'),'torch',torch.__version__,'cuda',torch.version.cuda,'transformers',m.version('transformers'))"
  "${ENVS}/ddl-telecom/bin/python" -I -c "import importlib.metadata as m;print('tau2',m.version('tau2'),'litellm',m.version('litellm'))"
  echo "tau2_sha=$(git -C "${TAU2}" rev-parse HEAD)"; md5sum "${TAU2_DATA_DIR}/tau2/domains/telecom/tasks.json" "${TAU2_DATA_DIR}/tau2/domains/telecom/split_tasks.json"
} 2>&1 | tee "${RUN}/job_info.txt"

RSS_LOG="${RUN}/rss.log"; VPID_FILE="${RUN}/vllm.pid"; export RSS_LOG
( while true; do   # RSS sampler (KB): vLLM process group + tau2 client; ends with this script (cleanup kills the group via VPID; also checks parent)
    kill -0 $$ 2>/dev/null || break
    vp=$(cat "${VPID_FILE}" 2>/dev/null); v=$([[ -n "${vp}" ]] && ps -o rss= -g "${vp}" 2>/dev/null | awk '{s+=$1} END{print s+0}' || echo 0)
    c=$(ps -eo rss=,args= 2>/dev/null | awk '/telecom_boed.tau2_ext.launch/ && !/awk/ {s+=$1} END{print s+0}')
    echo "$(date -Is) vllm_group_rss_kb=${v} tau2_client_rss_kb=${c}" >> "${RSS_LOG}"
    sleep "${RSS_INTERVAL_S:-60}"
  done ) &
RSSPID=$!
if [[ "${GPU_SAMPLER:-0}" == "1" ]]; then bash "${HERE}/scripts/gpu_sampler.sh" "${RUN}/gpu_samples.csv" "${GPU_SAMPLER_INTERVAL_S:-5}" "${G}" & GPUSPID=$!; fi

# --- unique free port per step: 8000 + job%500 + 10*gpu + step%10 (bumped while in use) ---
JOBN="${JOB//[!0-9]/}"
PORT=$((8000 + ${JOBN:-0} % 500 + 10 * G + STEP % 10))
while (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; do PORT=$((PORT + 1)); done
echo "port=${PORT}" | tee -a "${RUN}/job_info.txt"

# --- vLLM server (own process group; same flags as Perlmutter). FC-013: startup retried only for a pre-weight-load segfault ---
t0=${SECONDS}
attempt=0
while :; do
  setsid "${VLLM_ENV}/bin/vllm" serve "${MODEL}" --host 127.0.0.1 --port "${PORT}" --served-model-name "${MODEL}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
    --enable-auto-tool-choice --tool-call-parser hermes ${VLLM_PREFIX_CACHE_FLAG:---enable-prefix-caching} --seed 0 ${VLLM_EXTRA_ARGS:-} \
    > "${RUN}/vllm.log" 2>&1 &
  VPID=$!; echo "${VPID}" > "${VPID_FILE}"
  healthy=0
  for _ in $(seq 1 $(( ${HEALTH_TIMEOUT_S:-600} / 5 ))); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null; then healthy=1; break; fi
    kill -0 ${VPID} 2>/dev/null || break
    sleep 5
  done
  [[ ${healthy} -eq 1 ]] && break
  if ! kill -0 ${VPID} 2>/dev/null; then
    vrc=0; wait ${VPID} 2>/dev/null || vrc=$?
    # pre-weight-load segfault = rc 139 or a segfault line, and no weight-load marker in the log
    if { [[ ${vrc} -eq 139 ]] || grep -q -E "Segmentation fault|Fatal Python error: Segmentation" "${RUN}/vllm.log"; } \
       && ! grep -q -E "Loading model from scratch|Loading safetensors checkpoint|Starting to load model|Loading weights" "${RUN}/vllm.log" \
       && [[ ${attempt} -lt ${STARTUP_RETRIES:-2} ]]; then
      attempt=$((attempt + 1))
      echo "FC-013: pre-weight-load segfault (rc=${vrc}); startup retry ${attempt}/${STARTUP_RETRIES:-2}" | tee -a "${RUN}/job_info.txt" >&2
      cp "${RUN}/vllm.log" "${RUN}/vllm.attempt${attempt}.log"; ev --phase startup_retry attempt="${attempt}" rc="${vrc}"
      kill -KILL -- "-${VPID}" 2>/dev/null || true; sleep 10
      continue
    fi
    echo "vLLM died during startup (rc=${vrc}, retries used ${attempt})" >&2; tail -n 40 "${RUN}/vllm.log" >&2; exit 1
  fi
  echo "vLLM not healthy after ${HEALTH_TIMEOUT_S:-600}s" >&2; tail -n 40 "${RUN}/vllm.log" >&2; exit 1
done
echo "startup_retries=${attempt}" | tee -a "${RUN}/job_info.txt"
echo "vllm_ready_s=$((SECONDS - t0))" | tee -a "${RUN}/job_info.txt"
ev --phase vllm_ready ready_s=$((SECONDS - t0)) port="${PORT}" gpu_mem_MiB="$(nvidia-smi -i "${G}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
nvidia-smi -i "${G}" --query-gpu=memory.used --format=csv,noheader | sed 's/^/gpu_mem_after_load=/' | tee -a "${RUN}/job_info.txt"

# --- episodes: agent + user both on the local server, non-thinking (Qwen3 chat template switch) ---
LLM_ARGS="{\"api_base\": \"http://127.0.0.1:${PORT}/v1\", \"api_key\": \"local\", \"temperature\": 0.0, \"extra_body\": {\"chat_template_kwargs\": {\"enable_thinking\": false}}}"
t1=${SECONDS}
python3 -I "${HERE}/scripts/run_events.py" watch --out "${EVENTS}" --run "${NAME}" --results "${TAU2_DATA_DIR}/simulations/${SAVE_TO}/results.json" \
  --metrics "http://127.0.0.1:${PORT}/metrics" --gpu "${G}" --ppid $$ --interval "${EVENT_INTERVAL_S:-60}" --heartbeat "${EVENT_HEARTBEAT_S:-300}" &
WPID=$!
ev --phase tau2_start
set +e
if [[ "${MODE}" == "raw" ]]; then
  args=(run --domain telecom --task-split-name "${SPLIT:-train}" --num-trials "${NUM_TRIALS:-1}"
        --agent-llm "hosted_vllm/${MODEL}" --agent-llm-args "${LLM_ARGS}"
        --user-llm "hosted_vllm/${MODEL}" --user-llm-args "${LLM_ARGS}"
        --max-concurrency "${MAX_CONC:-3}" --max-steps "${MAX_STEPS:-200}" --seed "${SEED:-300}"
        --save-to "${SAVE_TO}")
  if [[ -n "${TASK_IDS:-}" ]]; then args+=(--task-ids ${TASK_IDS}); else args+=(--num-tasks "${NUM_TASKS:-3}"); fi
  echo "tau2 ${args[*]}" | tee -a "${RUN}/job_info.txt"
  MAX_RECYCLES=0 run_client "${RUN}/tau2.log" "${TAU2_TIMEOUT_S:-1500}" "${ENVS}/ddl-telecom/bin/tau2" "${args[@]}"
  rc=${RC}
elif [[ "${MODE}" == "patched" ]]; then
  # ---- MODE=patched: same tau2 CLI args as raw, through telecom_boed.tau2_ext.launch (patches + guards + run_meta) ----
  args=(run --domain telecom --task-split-name "${SPLIT:-train}" --num-trials "${NUM_TRIALS:-1}"
        --agent-llm "hosted_vllm/${MODEL}" --agent-llm-args "${LLM_ARGS}"
        --user-llm "hosted_vllm/${MODEL}" --user-llm-args "${LLM_ARGS}"
        --max-concurrency "${MAX_CONC:-3}" --max-steps "${MAX_STEPS:-200}" --seed "${SEED:-300}"
        --save-to "${SAVE_TO}")
  if [[ -n "${TASK_IDS:-}" ]]; then set -f; args+=(--task-ids ${TASK_IDS}); set +f; elif [[ -n "${NUM_TASKS:-}" ]]; then args+=(--num-tasks "${NUM_TASKS}"); fi
  lflags=(); [[ "${EVAL_FINAL:-0}" == "1" ]] && lflags+=(--eval-final)
  export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" TELECOM_RUN_DIR="${RUN}"
  echo "launch ${lflags[*]} --tau2-cli -- ${args[*]}" | tee -a "${RUN}/job_info.txt"
  MAX_RECYCLES=0 run_client "${RUN}/tau2.log" "${TAU2_TIMEOUT_S:-1500}" "${ENVS}/ddl-telecom/bin/python" -X faulthandler -m telecom_boed.tau2_ext.launch "${lflags[@]}" --tau2-cli -- "${args[@]}"
  rc=${RC}
elif [[ "${MODE}" == "launch" ]]; then
  # ---- MODE=launch (W3-S10): python -m telecom_boed.tau2_ext.launch --config $RUN_CONFIG; it reads server info from env ----
  : "${RUN_CONFIG:?MODE=launch needs RUN_CONFIG}"
  export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" VLLM_API_BASE="http://127.0.0.1:${PORT}/v1" TAU2_SAVE_TO="${SAVE_TO}" TELECOM_RUN_DIR="${RUN}"
  if ! "${ENVS}/ddl-telecom/bin/python" -c "import importlib.util,sys;sys.exit(importlib.util.find_spec('telecom_boed.tau2_ext.launch') is None)" 2>/dev/null; then
    echo "launch.py not built yet (W3-S10)" | tee "${RUN}/tau2.log" >&2; rc=5
  else
    echo "launch --config ${RUN_CONFIG}" | tee -a "${RUN}/job_info.txt"
    run_client "${RUN}/tau2.log" "${TAU2_TIMEOUT_S:-1500}" "${ENVS}/ddl-telecom/bin/python" -X faulthandler -m telecom_boed.tau2_ext.launch --config "${RUN_CONFIG}"
    rc=${RC}; echo "client_recycles=${RECYCLES}" | tee -a "${RUN}/job_info.txt"
  fi
else
  echo "unknown MODE=${MODE} (raw|patched|launch)" >&2; rc=1
fi
set -e
echo "tau2_rc=${rc} tau2_wall_s=$((SECONDS - t1))" | tee -a "${RUN}/job_info.txt"
kill -TERM "${WPID}" 2>/dev/null; wait "${WPID}" 2>/dev/null; WPID=
ev --phase tau2_end rc="${rc}" wall_s=$((SECONDS - t1))
curl -sf "http://127.0.0.1:${PORT}/metrics" > "${RUN}/vllm_metrics.prom" || true
grep -E "^vllm:(prefix_cache_(hits|queries)|prompt_tokens|generation_tokens)_total" "${RUN}/vllm_metrics.prom" | tee -a "${RUN}/job_info.txt" || true
cp "${TAU2_DATA_DIR}/simulations/${SAVE_TO}/results.json" "${RUN}/results.json" 2>/dev/null \
  || cp "$(ls -t "${TAU2_DATA_DIR}"/simulations/*/results.json 2>/dev/null | head -n 1)" "${RUN}/results.json" 2>/dev/null || true
tail -n 25 "${RUN}/tau2.log" || true
exit "${rc}"
