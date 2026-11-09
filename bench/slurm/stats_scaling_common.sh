# Shared body of stats_scaling_{horeka,juwels}.sbatch (sourced, not run): the distributed
# statistics campaign (bench/distributed/scaling.py) as RESUMABLE ROWS, the convention of
# bench/slurm/differentiability_scaling.sbatch. A row is one command; its marker is
# $OUT/rows/<tag>.json, written only when the row succeeded, so a resubmission skips every
# finished row and reruns the ones that failed or never ran. Row tags carry the GPU count,
# so one OUT directory collects a whole curve across jobs of different sizes.
#
# The job script sets CLUSTER and the "adapt" variables, then calls ygg_main. Knobs (env):
#   STAGE     test | verify | ceiling | weak | strong | flagship   (or the first argument)
#   PY        python with jax 0.11.2 (CUDA 12) and this yggdrax installed (bench/slurm/env)
#   VENV      activated when it exists (default ~/envs/yggdrax-jax0112)
#   YGGDRAX   this checkout (default: the submit directory)
#   OUT       results root (default ${WORKSPACE:-$HOME}/yggdrax-stats-scaling-$CLUSTER)
#   ROWS      regex over row tags (default: all)
#   RUNS, WARMUP   timed repeats / warm-ups per case (default 3 / 1)
#   PAYLOADS ("xi knn"), ICS ("uniform"), N_PER_DEVICE (weak), N_TOTAL (strong)
#   ARGS      extra scaling.py arguments for every row
#   YGG_CEILING_START / _STOP / _FACTOR   the ceiling sweep (2^20 .. 2^28, x2)

ygg_setup() {
  : "${CLUSTER:?the job script sets CLUSTER}"
  YGGDRAX=${YGGDRAX:-${SLURM_SUBMIT_DIR:-$PWD}}
  OUT=${OUT:-${WORKSPACE:-$HOME}/yggdrax-stats-scaling-${CLUSTER}}
  ROWS=${ROWS:-.}
  PY=${PY:-python}
  RUNS=${RUNS:-3}
  WARMUP=${WARMUP:-1}
  VENV=${VENV:-$HOME/envs/yggdrax-jax0112}
  if [[ -n "${YGGDRAX_SITE_MODULES:-}" ]]; then eval "${YGGDRAX_SITE_MODULES}"; fi
  # shellcheck disable=SC1091
  if [[ -f "$VENV/bin/activate" ]]; then source "$VENV/bin/activate"; fi
  export PYTHONPATH="${YGGDRAX}${PYTHONPATH:+:${PYTHONPATH}}"
  G=${SLURM_NTASKS:-1} # one process per GPU: the row tags carry this

  # --- JAX / XLA
  export JAX_ENABLE_X64=1                    # the statistics need it (ids, Morton codes)
  export XLA_PYTHON_CLIENT_PREALLOCATE=false # peak_bytes_in_use = the working set
  # Across nodes the native ragged all-to-all runs NCCL send/recv. On jax >= 0.11 that path
  # delivers no cross-device rows and then hangs while use_barrier_with_nccl is on, its new
  # default (yggdrax.distributed.comm.ragged_native_flag_hazard). Harmless on one node. The
  # flag exists from jax 0.10 on; an XLA that does not know a flag aborts, hence the gate.
  if $PY -c "import jax, sys; sys.exit(tuple(map(int, jax.__version__.split('.')[:2])) < (0, 10))"; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false"
  else
    echo "WARNING: jax < 0.10: the native exchange is not available (auto -> buf); see bench/slurm/env"
  fi
  export JAX_COMPILATION_CACHE_DIR="$OUT/jax-cache" # 2048^3 programs compile for minutes
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10

  # --- NCCL. First multi-node job: NCCL_DEBUG=INFO, and the log must say "NET/IB" (not
  # "NET/Socket") and "GDRDMA". If not, set NCCL_IB_HCA to the HCAs `ibv_devinfo` lists (4 HDR
  # HCAs per node on both machines, one per GPU) and NCCL_SOCKET_IFNAME=ib0.
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
  export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-10}

  mkdir -p "$OUT/rows" "$OUT/logs" "$OUT/cases"
  cd "$YGGDRAX"
  local job=${SLURM_JOB_ID:-local}
  git rev-parse HEAD > "$OUT/yggdrax_commit-$job.txt" 2>/dev/null || true
  git status --porcelain --untracked-files=no > "$OUT/yggdrax_dirty-$job.txt" 2>/dev/null || true
  $PY -c "import jax, jaxlib, yggdrax; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__); print('yggdrax', yggdrax.__file__)" \
    | tee "$OUT/env-$job.txt"
  echo "node(s) ${SLURM_JOB_NODELIST:-$(hostname)}: $G task(s), stage ${STAGE:-?} -> $OUT"
  echo "XLA_FLAGS=${XLA_FLAGS:-}"
}

# ygg_row TAG ENVS CMD... : run CMD (ENVS: "K=V K=V" or empty) unless the row is done.
# Returns the command's status; a success leaves $OUT/rows/TAG.json.
ygg_row() {
  local tag=$1 envs=$2
  shift 2
  [[ $tag =~ $ROWS ]] || return 0
  local json=$OUT/rows/$tag.json
  if [ -s "$json" ]; then echo "skip $tag (done)"; return 0; fi
  echo "=== $tag  [$envs]  $(date '+%Y-%m-%dT%H:%M:%S%z')"
  local start rc=0
  start=$(date +%s)
  # a subshell, not `env`: the command may be a shell function (ygg_scaling)
  # shellcheck disable=SC2086
  ( if [ -n "$envs" ]; then export $envs; fi; "$@" ) > "$OUT/logs/$tag.log" 2>&1 || rc=$?
  if [ $rc -eq 0 ] && [ ! -s "$json" ]; then # a command without its own marker (pytest)
    echo "{\"tag\": \"$tag\", \"status\": \"ok\", \"job\": \"${SLURM_JOB_ID:-}\"}" > "$json"
  fi
  if [ $rc -eq 0 ]; then echo "    ok in $(( $(date +%s) - start )) s"; else
    echo "    FAILED rc=$rc in $(( $(date +%s) - start )) s (log $OUT/logs/$tag.log)"; fi
  grep -E '^(xi|knn) |exchange check|passed|failed|Error' "$OUT/logs/$tag.log" | tail -8 || true
  return $rc
}

# one scaling.py run over all of the job's GPUs, one process per GPU
ygg_scaling() {
  local tag=$1
  shift
  # --kill-on-bad-exit: one rank's OOM must not leave the others hanging in a collective
  # shellcheck disable=SC2086
  srun --kill-on-bad-exit=1 --cpu-bind=none \
    $PY bench/distributed/scaling.py --platform gpu --gpu-select none \
    --repeats "$RUNS" --warmup "$WARMUP" --tag "$tag" \
    --out-dir "$OUT/cases/$tag" --output "$OUT/rows/$tag.json" "$@" ${ARGS:-}
}

ygg_main() {
  ygg_setup
  local stage=${STAGE:-${1:-}}
  [[ $# -gt 0 && -z "${STAGE:-}" ]] && shift
  if [[ $G -eq 1 && $stage != test && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # one card of the exclusive node; the others stay idle (no contention)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  fi
  local payloads=${PAYLOADS:-xi knn} ics_list=${ICS:-uniform}
  local pl ics mode
  case "$stage" in
    test)
      # the C1/C2/C3 tests on the node's GPUs (single process, 1/2/4 devices, brute force,
      # gradients, both summary exchanges) + the opt-in native-ragged side-effect check
      ygg_row "test_g4" "YGGDRAX_CHECK_UPSTREAM_RAGGED_FIX=1" \
        srun --ntasks=1 --cpu-bind=none $PY -m pytest -q --no-cov -p no:cacheprovider \
        tests/distributed/test_distributed_stats.py \
        tests/distributed/test_distributed_stats_grad.py \
        tests/distributed/test_distributed_stats_exchange.py
      ;;
    verify)
      # native exchange vs the all-gather one, bit for bit, across nodes; small N (buf
      # all-gathers every send buffer)
      for pl in $payloads; do
        ygg_row "verify_${pl}_g$G" "NCCL_DEBUG=${NCCL_DEBUG_VERIFY:-INFO}" \
          ygg_scaling "verify_${pl}_g$G" --scaling weak --n-per-device 262144 \
          --payload "$pl" --modes forward --verify-exchange "$@"
      done
      ;;
    ceiling)
      # double n per device until a row fails; one mode per sweep (different ceilings); a
      # failure leaves rows/<tag>.failed, so a resubmission does not redo the OOM
      for pl in $payloads; do
        for mode in forward value_and_grad; do
          local n=${YGG_CEILING_START:-1048576}
          while (( n <= ${YGG_CEILING_STOP:-268435456} )); do
            local tag="ceiling_${pl}_${mode}_g${G}_n$n"
            if [ -e "$OUT/rows/$tag.failed" ]; then echo "skip $tag (failed before)"; break; fi
            # near-tight first guess (uniform margins >= 1/0.84), then refit every capacity
            # to 1.25x the probe's counters: memory follows the capacities
            if ! ygg_row "$tag" "" ygg_scaling "$tag" --scaling weak --n-per-device "$n" \
              --payload "$pl" --modes "$mode" --clustering 1.0 --tighten 1.25 "$@"; then
              [[ $tag =~ $ROWS ]] && touch "$OUT/rows/$tag.failed"
              break
            fi
            n=$(( n * ${YGG_CEILING_FACTOR:-2} ))
          done
        done
      done
      ;;
    weak)
      : "${N_PER_DEVICE:?set N_PER_DEVICE (about half the 4-GPU value_and_grad ceiling)}"
      for ics in $ics_list; do
        local tag="weak_${ics}_g${G}_n${N_PER_DEVICE}"
        ygg_row "$tag" "" ygg_scaling "$tag" --scaling weak --n-per-device "$N_PER_DEVICE" \
          --payload "${payloads// /,}" --ics "$ics" --tighten 1.25 "$@" || true
      done
      ;;
    strong)
      : "${N_TOTAL:?set N_TOTAL (fits on the smallest GPU count of the series)}"
      for ics in $ics_list; do
        local tag="strong_${ics}_g${G}_N${N_TOTAL}"
        ygg_row "$tag" "" ygg_scaling "$tag" --scaling strong --n-total "$N_TOTAL" \
          --payload "${payloads// /,}" --ics "$ics" --tighten 1.25 "$@" || true
      done
      ;;
    flagship)
      : "${N_PER_DEVICE:?set N_PER_DEVICE (2048^3 / GPUs)}"
      for ics in ${ICS:-clustered}; do
        local tag="flagship_${ics}_g${G}_n${N_PER_DEVICE}"
        ygg_row "$tag" "" ygg_scaling "$tag" --scaling weak --n-per-device "$N_PER_DEVICE" \
          --payload "${payloads// /,}" --ics "$ics" --tighten 1.25 "$@" || true
      done
      ;;
    *)
      echo "STAGE must be test | verify | ceiling | weak | strong | flagship (got '$stage')" >&2
      return 2
      ;;
  esac
  echo "rows in $OUT/rows; table: $PY bench/slurm/summarize_stats_scaling.py $OUT/cases"
}
