#!/bin/bash
# Yggdrax distributed statistics on JUWELS Booster (4x A100-40GB per node, NVLink, 4x HDR200 IB):
# the same rows as stats_scaling_horeka.sbatch (see there and bench/slurm/README.md), one process
# per GPU, resumable via $OUT/rows/<tag>.json.
#
#   STAGE=weak N_PER_DEVICE=16777216 sbatch --nodes=16 bench/slurm/stats_scaling_juwels.sbatch
#
#SBATCH --job-name=yggdrax-stats-scaling
#SBATCH --account=__JUWELS_PROJECT__
#SBATCH --partition=__booster_or_develbooster__
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: the venv from bench/slurm/env/build_env.sh juwels (on $PROJECT, not $HOME), the
# results on $SCRATCH
# export VENV=$PROJECT_<project>/envs/yggdrax-jax0112
# export OUT=$SCRATCH/yggdrax-stats-scaling-juwels
CLUSTER=juwels
YGGDRAX=${YGGDRAX:-${SLURM_SUBMIT_DIR:-$PWD}}

# JSC: compute nodes reach each other on the InfiniBand hostnames (node name + "i"); the
# jax.distributed coordinator must listen there (scaling.py passes it on)
_head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export JAX_COORDINATOR_ADDRESS="${_head}i:$((20000 + SLURM_JOB_ID % 10000))"
# JSC's recommendations for NCCL over their IB (check the current user docs)
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-50}
export UCX_RC_TIMEOUT=${UCX_RC_TIMEOUT:-4s}

source "$YGGDRAX/bench/slurm/stats_scaling_common.sh"
ygg_main "$@"
