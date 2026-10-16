#!/usr/bin/env bash
# Yggdrax I, Phase 2: the GPU run list.
#
# Reruns every JSON a figure or a sentence of the paper quotes a *time* from,
# keeps every JSON that carries only accuracy, and adds the two new
# measurements: the accuracy-matched Stein-update sweep and the pair-count
# theta sweep at N = 1e5. One machine, one environment, one device for all of
# it. Why each line is here: edda-programme, prompts/PROMPT_yggdrax_i_submission.md
# section 2.
#
# Usage, from the repository root, on the paper branch, worktree clean, in the
# environment the whole run uses (e.g. `micromamba activate <env>` first):
#
#   bash bench/paper_runlist.sh                          # the GPU run
#   bash bench/paper_runlist.sh --smoke --gpu-select none  # the CPU check
#
# Options:
#   --gpu-select free|least-used|none   default free: autocvd picks one card,
#                                       once, and every command runs on it
#   --smoke                             every bench at its --smoke size, outputs
#                                       to a scratch directory, gate verdicts
#                                       printed but not enforced
#   --python CMD                        interpreter (default: python)
#
# It prints the JAX version, device, yggdrax commit and jax_enable_x64, and
# exits unless x64 is on, exactly one device is visible (a GPU, unless
# --gpu-select none) and the worktree is clean. It stops on the first failure.
# After each rerun, bench/paper_gate.py compares the accuracy and tree-count
# fields with the file as committed at the starting commit; a mismatch stops
# the list ("if accuracy moved, stop": a change is a bug, not a result).
#
# A size that runs out of memory is recorded in its JSON as
# sweep.stopped_early and the smaller sizes are kept; the bench exits 0 and
# the list carries on.

set -euo pipefail

GPU_SELECT=free
SMOKE=0
PY=python
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-select) GPU_SELECT="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --python) PY="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$GPU_SELECT" in free|least-used|none) ;; *) echo "bad --gpu-select" >&2; exit 2 ;; esac

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# The accuracy-matched theta grid, read off the CPU pilot
# (results/svgd/accuracy_matched/pilot_cpu.json): it brackets a relative
# error of 1e-3 at every N of the sweep that can be bracketed with theta <= 1.
THETA_GRID="0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.60 0.70 0.80 0.90 1.00"
AM_SIZES="1000 2000 5000 10000 20000 50000 100000"
PILOT="results/svgd/accuracy_matched/pilot_cpu.json"

# -- the GPU, picked once ---------------------------------------------------------
if [[ "$GPU_SELECT" != none && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  LEAST=False; [[ "$GPU_SELECT" == least-used ]] && LEAST=True
  CVD="$("$PY" -c "
import os
from autocvd import autocvd
autocvd(num_gpus=1, least_used=$LEAST)
print('CVD=' + os.environ['CUDA_VISIBLE_DEVICES'])
" | sed -n 's/^CVD=//p' | tail -n 1)"
  [[ -n "$CVD" ]] || { echo "[runlist] autocvd picked no GPU" >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="$CVD"
fi

# -- preflight --------------------------------------------------------------------
BASE="$(git rev-parse HEAD)"
DIRTY="$(git status --porcelain --untracked-files=no)"
"$PY" - "$GPU_SELECT" "$ROOT" <<'PY'
import os, sys
gpu_select, root = sys.argv[1], os.path.realpath(sys.argv[2])
import jax
jax.config.update("jax_enable_x64", True)  # as every bench does
import jax.numpy as jnp
sys.path.insert(0, root)
import yggdrax
devices = jax.devices()
x64 = bool(jax.config.read("jax_enable_x64")) and jnp.zeros(1).dtype == jnp.float64
print(f"[runlist] jax {jax.__version__}  backend {jax.default_backend()}")
print(f"[runlist] devices {devices}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[runlist] yggdrax from {os.path.dirname(yggdrax.__file__)}")
print(f"[runlist] jax_enable_x64 {x64}")
problems = []
if not x64:
    problems.append("x64 is not on")
if len(devices) != 1:
    problems.append(f"{len(devices)} devices visible, need exactly 1")
if gpu_select != "none" and devices[0].platform != "gpu":
    problems.append(f"--gpu-select {gpu_select} but the device is {devices[0].platform}")
if not os.path.realpath(yggdrax.__file__).startswith(root + os.sep):
    problems.append("yggdrax is not imported from this checkout")
if problems:
    sys.exit("[runlist] refusing to start: " + "; ".join(problems))
PY
echo "[runlist] yggdrax commit $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
if [[ -n "$DIRTY" ]]; then
  echo "[runlist] refusing to start: the worktree is not clean" >&2
  echo "$DIRTY" >&2
  exit 1
fi

# -- where the outputs go ------------------------------------------------------------
if [[ $SMOKE == 1 ]]; then
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/paper_runlist.XXXXXX")"
  echo "[runlist] smoke: outputs under $OUT; gate verdicts are not enforced"
else
  OUT=results
fi
mkdir -p "$OUT/svgd/accuracy_matched" "$OUT/differentiability" "$OUT/corrfunc"
WRITTEN=()

step() { echo; echo "================ $* ================"; }

# bench SMOKE_FLAG OUTPUT -- CMD...: run one bench, adding SMOKE_FLAG in smoke mode.
bench() {
  local smoke_flag="$1" output="$2"; shift 3
  local cmd=("$@" --gpu-select "$GPU_SELECT" --output "$output")
  [[ $SMOKE == 1 ]] && cmd+=("$smoke_flag")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
  WRITTEN+=("$output")
}

# gate ARGS...: compare with the committed file; enforced unless --smoke.
gate() {
  local status=0
  "$PY" bench/paper_gate.py "$@" || status=$?
  if [[ $status == 0 ]]; then return 0; fi
  if [[ $SMOKE == 1 && $status == 3 ]]; then
    echo "[runlist] smoke: gate mismatch expected at smoke sizes; not enforced"
    return 0
  fi
  echo "[runlist] STOP: the gate failed (exit $status). Report it; do not rerun around it." >&2
  exit "$status"
}

# -- 1. Stein-update scaling, d = 3 and d = 5 ----------------------------------------------
step "svgd/scaling.json (d=3)"
bench --smoke "$OUT/svgd/scaling.json" -- "$PY" bench/svgd/scaling.py --dim 3 \
  --sizes 1000 2000 5000 10000 20000 50000 100000 200000 1000000 --exact-max-n 100000
gate rerun svgd_scaling "$OUT/svgd/scaling.json" --base "$BASE" --path results/svgd/scaling.json

step "svgd/scaling_d5.json (d=5)"
bench --smoke "$OUT/svgd/scaling_d5.json" -- "$PY" bench/svgd/scaling.py --dim 5 \
  --sizes 1000 2000 5000 10000 20000 50000 100000 200000 1000000 --exact-max-n 100000
gate rerun svgd_scaling "$OUT/svgd/scaling_d5.json" --base "$BASE" --path results/svgd/scaling_d5.json

# -- 2. the accuracy-matched sweep: one theta per run, then the schedule ---------------------
THETA_FILES=()
for theta in $THETA_GRID; do
  f="$OUT/svgd/accuracy_matched/theta_${theta}.json"
  step "svgd/accuracy_matched/theta_${theta}.json"
  bench --smoke "$f" -- "$PY" bench/svgd/scaling.py --dim 3 --density fixed-cloud \
    --dtype float64 --theta "$theta" --sizes $AM_SIZES --exact-max-n 100000
  gate pilot "$f" --pilot "$PILOT"
  THETA_FILES+=("$f")
done
step "svgd/accuracy_matched.json"
"$PY" bench/svgd/accuracy_matched.py --inputs "${THETA_FILES[@]}" \
  --output "$OUT/svgd/accuracy_matched.json"
WRITTEN+=("$OUT/svgd/accuracy_matched.json")

# -- 3. tree builds and traversals, and the tree-level autodiff overhead --------------------
step "differentiability/scaling.json"
bench --smoke "$OUT/differentiability/scaling.json" -- "$PY" bench/differentiability/scaling.py \
  --sizes 1000 10000 100000 1000000
gate rerun diff_scaling "$OUT/differentiability/scaling.json" --base "$BASE" \
  --path results/differentiability/scaling.json

step "differentiability/autodiff_overhead.json"
bench --smoke "$OUT/differentiability/autodiff_overhead.json" -- "$PY" \
  bench/differentiability/autodiff_overhead.py --sizes 1000 10000 100000 1000000
# timings only: no accuracy field to gate

# -- 4. pair counts ---------------------------------------------------------------------
step "corrfunc/scaling.json"
bench --smoke "$OUT/corrfunc/scaling.json" -- "$PY" bench/corrfunc/scaling.py \
  --sizes 1000 10000 20000 100000 1000000 --brute-force-max-n 20000
gate rerun corrfunc_scaling "$OUT/corrfunc/scaling.json" --base "$BASE" \
  --path results/corrfunc/scaling.json

# -- 5. series C of the rebuild scale-up, exactly as its JSON records it ----------------------
# The flags are the file's own fields (fixed_budget, target/lr modes and
# reference N, sweep.num_particles_requested, sweep.seeds); the allocator
# settings are its metadata.xla_memory_env, and the fraction sets where the
# sweep runs out of memory.
step "differentiability/nn_rebuild_scaling_step_scaled.json (series C)"
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 bench --smoke-sweep \
  "$OUT/differentiability/nn_rebuild_scaling_step_scaled.json" -- \
  "$PY" bench/differentiability/nn_rebuild.py \
  --sweep-n 256 1024 4096 16384 65536 131072 262144 524288 1048576 2097152 \
  4194304 8388608 16777216 33554432 67108864 134217728 \
  --sweep-seeds 0 1 2 --target-mode density-matched --target-reference-n 256 \
  --lr-mode n-scaled --lr-reference-n 256 --dim 3 --steps 120 --learning-rate 2.0 \
  --clip 1.5 --k-neighbors 8 --leaf-size 16 --target-distance 0.12 \
  --related-series results/differentiability/nn_rebuild_scaling.json \
  results/differentiability/nn_rebuild_scaling_density_matched.json
gate rerun series_c "$OUT/differentiability/nn_rebuild_scaling_step_scaled.json" \
  --base "$BASE" --path results/differentiability/nn_rebuild_scaling_step_scaled.json

# -- 6. bandwidth learning, seed 0 only, as its params block records ------------------------
# If it moved beyond rounding the gate stops here; seeds 1-4 are not rerun.
step "svgd/bandwidth_learning.json (seed 0)"
bench --smoke "$OUT/svgd/bandwidth_learning.json" -- "$PY" \
  bench/svgd/bandwidth_learning_experiment.py --n 1000 --cov-diag 1 9 \
  --num-svgd-steps 30 --num-outer-steps 60 --step-size 0.3 --theta 0.5 \
  --leaf-size 64 --backend radix --learning-rate 0.15 --eval-steps 60 --seed 0
gate rerun bandwidth "$OUT/svgd/bandwidth_learning.json" --base "$BASE" \
  --path results/svgd/bandwidth_learning.json

# -- 7. pair-count error vs cost at N = 1e5 (new data; validation.json stays) ----------------
# --num-bins, --r-min, --r-max, --sharpness, --leaf-size, --backend and --seed
# are validation.json's params, so its N = 1e4 points and these are comparable.
# Above --brute-force-max-n the reference is the theta = 0 estimator (near
# field exact), not brute force; the record's "baseline" field says so.
step "corrfunc/validation_n1e5.json"
bench --smoke "$OUT/corrfunc/validation_n1e5.json" -- "$PY" \
  bench/corrfunc/validate_vs_baseline.py --sizes 100000 \
  --thetas 0 0.2 0.3 0.4 0.5 0.6 0.9 --num-bins 10 --r-min 0.01 --r-max 0.5 \
  --sharpness 150 --leaf-size 32 --backend radix --seed 0

# -- Gate 2's machine half: one jax, one device, one commit, x64 on ------------------------
step "Gate 2: every JSON from this run on one machine"
gate machine "${WRITTEN[@]}"

echo
echo "[runlist] done. Outputs:"
printf '  %s\n' "${WRITTEN[@]}"
