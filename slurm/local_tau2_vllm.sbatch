#!/bin/bash
# [telecom_boed] Thin sbatch wrapper around slurm/local_tau2_vllm.sh for the ilab/rlab Slurm cluster, for use when the GPU hold
# (job 526576) is gone. #SBATCH lines cannot expand variables, so the GPU type is given on the command line (the wrapper must
# NOT pass --cpus-per-task / -c / -n: the cluster rejects CPU requests). Partition: default. Knobs: see local_tau2_vllm.sh header.
# Submit (from westeros, repo on NFS; 1 GPU of the named type):
#   ssh rlab1 "cd /common/users/ss5772/ddl && sbatch --gres=gpu:a100:1 \
#     --export=ALL,RUN_TAG=smoke,NUM_TASKS=3 telecom_boed/slurm/local_tau2_vllm.sbatch"
# Inside the hold instead, call the .sh directly:
#   ssh rlab1 "srun --jobid=526576 --overlap --gres=gpu:a100:4 --mem=48G -t 00:30:00 bash /common/users/ss5772/ddl/telecom_boed/slurm/local_tau2_vllm.sh"
#SBATCH -J tel_local
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --mem=48G
#SBATCH -o telecom_boed/logs/%x-%j.out
set -euo pipefail
DDL_REPO="${DDL_REPO:-${SLURM_SUBMIT_DIR}}"
exec bash "${DDL_REPO}/telecom_boed/slurm/local_tau2_vllm.sh"
