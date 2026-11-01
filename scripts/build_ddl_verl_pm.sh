#!/bin/bash
# [telecom_boed] Build ddl-verl (Option B: ddl-vllm pins + patched verl v0.9.1 --no-deps) on a Perlmutter LOGIN node. CPU only, no Slurm.
# Prereqs (rsync/scp from westeros to Perlmutter first):
#   $K/src/verl-ref   <- copy of /common/users/ss5772/external/verl (commit 1876b06d)
#   $K/src/perlmutter_freeze_ddl-vllm.txt <- telecom_boed/constraints/perlmutter_freeze_ddl-vllm.txt
#   $K/src/per_request_lora.patch <- telecom_boed/patches/verl/per_request_lora.patch
# Run detached:  nohup nice -n 10 bash build_ddl_verl_pm.sh > $SCRATCH/cache/build_ddl_verl.log 2>&1 &
set -euo pipefail
K=/global/cfs/cdirs/m4431/kbmkbm
E=$K/envs/ddl-verl; PY=$E/bin/python; PIP="$E/bin/pip"
export PIP_CACHE_DIR=/pscratch/sd/k/kbmkbm/cache/pip UV_CACHE_DIR=/pscratch/sd/k/kbmkbm/cache/uv CONDA_PKGS_DIRS=/pscratch/sd/k/kbmkbm/cache/conda_pkgs
export MAX_JOBS=8 OMP_NUM_THREADS=8 TMPDIR=/pscratch/sd/k/kbmkbm/cache/tmp
mkdir -p $PIP_CACHE_DIR $UV_CACHE_DIR $CONDA_PKGS_DIRS $TMPDIR
FREEZE=$K/src/perlmutter_freeze_ddl-vllm.txt
# 1. clone ddl-vllm pins (same recipe as ddl-vllm: conda create python=3.12 + pip -c freeze vllm==0.31.0)
[[ -x $PY ]] || /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/bin/conda create -y -q -p $E python=3.12
$PIP install -q -c $FREEZE vllm==0.31.0 pytest
# 2. patched verl copy at the same commit
V=$K/src/verl-ddl
if [[ ! -d $V ]]; then rsync -a --chmod=u+w $K/src/verl-ref/ $V/ && (cd $V && git apply $K/src/per_request_lora.patch); fi
md5sum $K/src/per_request_lora.patch > $E/verl_patch.md5
$PIP install -q --no-deps -e $V
# 3. verl runtime deps without moving torch/vllm/transformers/numpy
grep -E '^(torch|torchvision|torchaudio|vllm|transformers|numpy)==' $FREEZE > $TMPDIR/pins.txt
$PIP install -q "ray[default]>=2.41.0" peft datasets accelerate hydra-core tensordict==0.10.0 torchdata codetiming \
  "pyarrow>=19" pandas dill pybind11 pylatexenc wandb tensorboard uvicorn fastapi --constraint $TMPDIR/pins.txt
# 4. checks (no GPU)
$PY -c "import verl, tensordict, ray, peft; print(verl.__version__, tensordict.__version__)"
$PY -c "import verl.workers.rollout.vllm_rollout.vllm_async_server"
$PY -c "import verl.trainer.sft_trainer, verl.experimental.agent_loop.tool_agent_loop"
$PY -m pytest $V/tests/experimental/agent_loop/test_tool_call_id_on_cpu.py -q
echo BUILD_DONE
