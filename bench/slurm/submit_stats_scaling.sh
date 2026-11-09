#!/usr/bin/env bash
# Submit one stage of the distributed-statistics campaign, one job per GPU count. Run from the
# repository root. All jobs of a cluster share OUT, so the rows of a curve land side by side and
# a resubmission only reruns what is missing (stats_scaling_common.sh).
#
#   bench/slurm/submit_stats_scaling.sh CLUSTER STAGE [extra scaling.py args...]
#
#   CLUSTER  horeka | juwels
#   STAGE    test      the GPU test suite, one node, single process (do this first)
#            verify    native vs all-gather exchange, bit for bit, 2 nodes (8 GPUs)
#            ceiling   memory ceiling on 1 GPU and on 1 node (4 GPUs), each payload and mode
#            weak      GPUS (default "1 4 16 64 128") at N_PER_DEVICE per GPU
#            strong    GPUS (default "4 16 64 128") at N_TOTAL particles
#            flagship  GPUS (default 128) at N_PER_DEVICE per GPU, clustered
#
# Knobs (exported to the jobs): GPUS, N_PER_DEVICE, N_TOTAL, PAYLOADS, ICS, OUT, ROWS, RUNS,
# WARMUP, VENV, PY; TIME (sbatch --time); DRY=1 prints the sbatch lines only.
set -euo pipefail

CLUSTER=${1:?cluster: horeka | juwels}
STAGE=${2:?stage: test | verify | ceiling | weak | strong | flagship}
shift 2
SCRIPT=bench/slurm/stats_scaling_${CLUSTER}.sbatch
[[ -f $SCRIPT ]] || { echo "no $SCRIPT (run from the repository root)" >&2; exit 2; }
TIME=${TIME:-02:00:00}
GPN=4 # GPUs per node, both machines
export STAGE

sub() {
  # sub GPUS ARGS... -> one job; < 4 GPUs is one exclusive node with that many tasks
  local gpus=$1 shape
  shift
  if (( gpus < GPN )); then
    shape=(--nodes=1 --ntasks-per-node="$gpus")
  else
    shape=(--nodes=$(( gpus / GPN )) --ntasks-per-node="$GPN")
  fi
  local cmd=(sbatch --time="$TIME" --export=ALL "${shape[@]}" "$SCRIPT" "$@")
  echo "STAGE=$STAGE ${cmd[*]}"
  [[ ${DRY:-0} == 1 ]] || "${cmd[@]}"
}

case $STAGE in
  test) sub 1 "$@" ;; # one task that sees all 4 GPUs of the exclusive node
  verify) sub 8 "$@" ;;
  ceiling) for g in 1 4; do sub "$g" "$@"; done ;;
  weak)
    : "${N_PER_DEVICE:?set N_PER_DEVICE (about half the 4-GPU value_and_grad ceiling)}"
    export N_PER_DEVICE
    for g in ${GPUS:-1 4 16 64 128}; do sub "$g" "$@"; done ;;
  strong)
    : "${N_TOTAL:?set N_TOTAL (fits on the smallest GPU count of the series)}"
    export N_TOTAL
    for g in ${GPUS:-4 16 64 128}; do sub "$g" "$@"; done ;;
  flagship)
    : "${N_PER_DEVICE:?set N_PER_DEVICE (2048^3 / GPUs)}"
    export N_PER_DEVICE
    for g in ${GPUS:-128}; do sub "$g" "$@"; done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
