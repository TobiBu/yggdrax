#!/bin/bash
# Yggdrax I tree scaling (bench/differentiability/scaling.py) on ONE card of a dedicated node: the
# timings fig_scaling plots, both walk arms of the fast path (flat XLA loop, Pallas kernel).
# Written for a HoreKa accelerated node (4x A100-40GB); adapt the partition, the environment and the
# paths below. One process at a time on the allocation's first card (the others idle: no contention,
# no shared host link). Resumable: a row whose JSON exists is skipped, so a resubmission continues
# where the last one stopped (a row that failed left no JSON and runs again).
#
#   sbatch bench/slurm/differentiability_scaling.sbatch                 # all rows (~1-2 h, rough)
#   ROWS='^fast' sbatch bench/slurm/differentiability_scaling.sbatch    # the rows whose tag matches a regex
#
# Rows:
#   scaling        results/differentiability/scaling.json: N = 1e3..1e6, the legacy per-backend
#                  adaptive build + eager dual walk (radix/octree/kdtree), the fast path (flat and
#                  Pallas arms) and the cKDTree CPU anchor. The legacy walk is host-bound at small N,
#                  so this row is the one that needs the idle node most.
#   scaling_fast   results/differentiability/scaling_fast.json: the fast path alone, N = 2^11..2^26,
#                  every power of two. The cell leaves are the coarsest Morton cells holding <= 64
#                  particles, so the mean leaf occupancy is a sawtooth in N (15..42 per leaf for a
#                  uniform cube in the 5 %-padded box, one cell level per factor 8); the sizes
#                  N = 2^(3k+2) (2^11, 2^14, ..., 2^26) all sit at ~42 per leaf and give the clean
#                  scaling line, the others show the sawtooth.
# Both rows stop if the Pallas arm's counts, rounds, peak or sorted far/near pair sets differ from the
# flat arm's at any size (the JSON's fast_pallas.matches_reference records the check).
#
#SBATCH --job-name=yggdrax-diff-scaling
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12) and this yggdrax checkout installed (editable)
# source ~/envs/yggdrax-jax0112/bin/activate
YGGDRAX=${YGGDRAX:-$PWD}                                  # this yggdrax checkout (paper branch)
OUT=${OUT:-${WORKSPACE:-$HOME}/yggdrax-diff-scaling-$(date +%Y%m%d)}
ROWS=${ROWS:-.}                                           # a regex over the row tags
PY=${PY:-python}
RUNS=${RUNS:-5}
WARMUP=${WARMUP:-2}

# one card of the exclusive node; the others stay idle
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
echo "node $(hostname) card ${CUDA_VISIBLE_DEVICES}"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || true
mkdir -p "$OUT/rows" "$OUT/logs"
cd "$YGGDRAX"
git rev-parse HEAD > "$OUT/yggdrax_commit.txt" 2>/dev/null || true
git status --porcelain --untracked-files=no > "$OUT/yggdrax_dirty.txt" 2>/dev/null || true
$PY -c "import jax, yggdrax; print('jax', jax.__version__, jax.devices()); print('yggdrax', yggdrax.__file__)" \
  | tee "$OUT/env.txt"

S=bench/differentiability/scaling.py
COMMON="--gpu-select none --runs $RUNS --warmup $WARMUP --walk-backends flat pallas"
POW2=""
for e in $(seq 11 26); do POW2="$POW2 $((1 << e))"; done

# tag | allocator env | script and arguments (the row's JSON is $OUT/rows/<tag>.json)
ROW_LIST=(
  "scaling||$S $COMMON --sizes 1000 10000 100000 1000000"
  "scaling_fast||$S $COMMON --no-legacy --no-scipy --sizes$POW2"
)

for row in "${ROW_LIST[@]}"; do
  IFS='|' read -r tag envs cmd <<< "$row"
  [[ $tag =~ $ROWS ]] || continue
  json=$OUT/rows/$tag.json
  if [ -s "$json" ]; then echo "skip $tag (done)"; continue; fi
  echo "=== $tag  [$envs]  $(date -Is)"
  start=$(date +%s)
  # shellcheck disable=SC2086
  if env $envs $PY $cmd --output "$json" > "$OUT/logs/$tag.log" 2>&1; then status=ok; else status="FAILED rc=$?"; fi
  echo "    $status in $(( $(date +%s) - start )) s (log $OUT/logs/$tag.log)"
  grep -E '^n=|stopped|wrote' "$OUT/logs/$tag.log" || true
done
echo "copy $OUT/rows/scaling.json and scaling_fast.json to results/differentiability/ to update fig_scaling"
