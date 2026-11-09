#!/bin/bash
# Yggdrax distributed statistics (xi pair counts, kNN-CDFs; forward and value+grad) on HoreKa
# accelerated nodes (4x A100-40GB, NVLink, HDR IB): the C3 scaling campaign and the C4 flagship,
# bench/distributed/scaling.py, ONE PROCESS PER GPU over as many exclusive nodes as the submit
# line asks for. Resumable, like differentiability_scaling.sbatch: a row whose JSON exists in
# $OUT/rows is skipped, so a resubmission continues where the last one stopped (a row that failed
# left no JSON and runs again). The stages and their order are in bench/slurm/README.md;
# submit_stats_scaling.sh submits a whole stage.
#
#   STAGE=test sbatch --ntasks-per-node=1 bench/slurm/stats_scaling_horeka.sbatch
#   STAGE=verify sbatch --nodes=2 bench/slurm/stats_scaling_horeka.sbatch
#   STAGE=ceiling sbatch --ntasks-per-node=1 bench/slurm/stats_scaling_horeka.sbatch   # 1 GPU
#   STAGE=ceiling sbatch bench/slurm/stats_scaling_horeka.sbatch                       # 4 GPUs
#   STAGE=weak N_PER_DEVICE=16777216 sbatch --nodes=4 bench/slurm/stats_scaling_horeka.sbatch
#
#SBATCH --job-name=yggdrax-stats-scaling
#SBATCH --account=__HOREKA_PROJECT__
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=19
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12) and this yggdrax checkout installed (editable);
# bench/slurm/env/build_env.sh builds it at ~/envs/yggdrax-jax0112 (also the one
# differentiability_scaling.sbatch expects)
# source ~/envs/yggdrax-jax0112/bin/activate
CLUSTER=horeka
YGGDRAX=${YGGDRAX:-${SLURM_SUBMIT_DIR:-$PWD}} # this yggdrax checkout; submit from its root
# HoreKa's IB HCAs, if NCCL_DEBUG=INFO shows NET/Socket:
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3

source "$YGGDRAX/bench/slurm/stats_scaling_common.sh"
ygg_main "$@"
