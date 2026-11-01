#!/bin/bash
# [telecom_boed] Build ddl-vllm on rlab2 node-local disk + stage Qwen3-8B there (CPU only; runs as a step of hold 526576).
set -euo pipefail
L=/freespace/local/ss5772
export CONDA_PKGS_DIRS=$L/conda_pkgs PIP_NO_CACHE_DIR=1 TMPDIR=$L/tmp XDG_CACHE_HOME=$L/xdg OMP_NUM_THREADS=1
mkdir -p $L/conda_envs $L/conda_pkgs $L/tmp $L/xdg
P=$L/conda_envs/ddl-vllm
[[ -x $P/bin/python ]] || /common/users/ss5772/miniforge3/bin/conda create -y -q -p $P python=3.12 >/dev/null
$P/bin/pip install -q -c /common/users/ss5772/ddl/telecom_boed/constraints/perlmutter_freeze_ddl-vllm.txt vllm==0.31.0 "huggingface_hub[cli]" 2>&1 | tail -3
$P/bin/python -c "import vllm,torch,transformers;print('ENV_OK vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda,'tf',transformers.__version__)"
du -sh $P
SRC_HF_HOME=/filer/tmp1/ss5772/huggingface DST_HF_HOME=$L/hf_cache bash /common/users/ss5772/ddl/telecom_boed/scripts/stage_weights_local.sh 2>&1 | tail -4
du -sh $L/hf_cache; df -h /freespace/local | tail -1
echo BUILD_DONE
