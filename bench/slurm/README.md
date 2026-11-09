# SLURM scripts

`differentiability_scaling.sbatch` (paper branch) times the single-GPU tree on
one card. The `stats_scaling_*` files below are the multi-GPU campaign for the
distributed clustering statistics (C3, C4, C5): `yggdrax.distributed.stats`,
which computes ξ pair counts and kNN-CDFs in a periodic box, with gradients.
Both use the same environment (`~/envs/yggdrax-jax0112`, built by
`env/build_env.sh`) and the same resumable-row convention. The measurement is
`bench/distributed/scaling.py`; `docs/distributed_statistics.md` describes the
pipeline, its capacities and the sizing model.

| file | what |
|---|---|
| `env/requirements-jax0112-cuda12.lock.txt` | pinned wheels: jax 0.11.2 + CUDA 12 (nvidia-* wheels), numpy, jaxtyping, beartype, pytest |
| `env/build_env.sh` | builds `~/envs/yggdrax-jax0112` from the lock file on a login node |
| `stats_scaling_common.sh` | the shared job body: environment, NCCL/XLA settings, the rows of every stage |
| `stats_scaling_horeka.sbatch` | HoreKa `accelerated` (4× A100-40GB per node) |
| `stats_scaling_juwels.sbatch` | JUWELS Booster (4× A100-40GB per node) |
| `submit_stats_scaling.sh` | submits one stage: one job per GPU count |
| `summarize_stats_scaling.py` | table of the results: time, stages, memory, parallel efficiency |

## 0. Once: environment and placeholders

On a **login node** (compute nodes may have no internet), from the repository
root:

```bash
bash bench/slurm/env/build_env.sh horeka      # -> ~/envs/yggdrax-jax0112
```

Use `juwels` on JUWELS, with a path on `$PROJECT` as the second argument, then
`export VENV=<that path>`. The script:

- loads a Python 3.12 module (check the module names it uses against
  `module avail`);
- makes the venv and installs the lock file;
- installs this checkout editable, so a `git pull` updates the code without a
  rebuild;
- checks that jax ≥ 0.9.1. Below that the native ragged all-to-all corrupts its
  output (`yggdrax.distributed.comm.RAGGED_NATIVE_FIXED_JAX`), so the driver
  falls back to an all-gather exchange that does not scale.

Fill in the job scripts' placeholders:

- `__HOREKA_PROJECT__` / `__JUWELS_PROJECT__`: the compute project;
- `__booster_or_develbooster__`: the JUWELS partition.

Submit from the repository root (`YGGDRAX` defaults to the submit directory).

## How a job runs

- **Rows.** A stage is a list of rows: one command each, tagged with its GPU
  count (`weak_uniform_g16_n16777216`). Every row writes:
  - its cases to `$OUT/cases/<tag>/`, one JSON per (payload, mode);
  - its log to `$OUT/logs/<tag>.log`;
  - on success only, `$OUT/rows/<tag>.json`.

  A resubmission skips every row whose JSON exists. `ROWS='^weak_.*_g64'` runs a
  subset. `OUT` defaults to `${WORKSPACE:-$HOME}/yggdrax-stats-scaling-<cluster>`,
  without a date, so a campaign spanning days stays in one place. Each job also
  records `yggdrax_commit-<job>.txt`, `yggdrax_dirty-<job>.txt` and
  `env-<job>.txt` there.
- **One process per GPU.**
  - `srun` starts 4 tasks per node. Each calls `jax.distributed.initialize()`
    from the SLURM environment and uses the GPU of its `SLURM_LOCALID`.
  - A 1-GPU job is one task on an exclusive node, restricted to the first card,
    as in `differentiability_scaling.sbatch`.
  - On JUWELS the coordinator is the head node's InfiniBand name
    (`<node>i:<port>`).
  - GPUs are exclusive under SLURM, so `autocvd` is not used. On the group's
    shared A100 server, run `scaling.py` directly with `--platform gpu --ndev 4
    --gpu-select free`: it waits for free cards with `autocvd`, the org rule.
- **XLA/NCCL** (`stats_scaling_common.sh`):
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so `peak_bytes_in_use` is the
    working set.
  - `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`.
    Across nodes the native ragged all-to-all runs NCCL send/recv. With that
    flag on, which is the default from jax 0.11, that path delivers no
    cross-device rows and then hangs (`comm.ragged_native_flag_hazard`).
  - A persistent compilation cache in `$OUT/jax-cache`.
  - Patient IB timeouts, plus JSC's recommended settings on JUWELS.
  - The `verify` stage sets `NCCL_DEBUG=INFO`. Check that its log says `NET/IB`
    and `GDRDMA`, not `NET/Socket`.

## The order of runs (HoreKa first)

Each stage is one `submit_stats_scaling.sh` call from the repository root.
`DRY=1` prints the `sbatch` lines without submitting.

1. **GPU test suite.** One node, a single process on 4 GPUs (~15 min). It runs:
   - the C1/C2/C3 tests against brute force on 1, 2 and 4 real GPUs, with both
     summary exchanges;
   - the opt-in check that a forward after a gradient is bit-identical through
     the native ragged exchange (`YGGDRAX_CHECK_UPSTREAM_RAGGED_FIX=1`).

   Go on only if it passes.

   ```bash
   bench/slurm/submit_stats_scaling.sh horeka test
   ```

2. **Exchange check across nodes.** 2 nodes, 8 GPUs, small N. The forward runs
   once with the native exchange and once with `buf` (all-gather), and the
   check records whether they are bit-identical (`exchange_check.identical` in
   the case JSON, `exchange check` in the log). This is the first time the NCCL
   path of the ragged all-to-all carries data.

   ```bash
   bench/slurm/submit_stats_scaling.sh horeka verify
   ```

3. **Memory ceiling** on 1 GPU and on 1 node (4 GPUs). Each payload and each
   mode (`forward`, `value_and_grad`) gets its own sweep: `--n-per-device`
   doubles from 2^20 until a row fails (OOM, or an overflow that 3 retries
   cannot fix).
   - The failure leaves `rows/<tag>.failed`, so a resubmission does not repeat
     it.
   - Every size is its own `srun` step, so its peak is its own.
   - The rows run with `--clustering 1.0 --tighten 1.25`, because memory
     follows the capacities.

   ```bash
   TIME=04:00:00 bench/slurm/submit_stats_scaling.sh horeka ceiling
   python bench/slurm/summarize_stats_scaling.py $OUT/cases --tag ceiling
   ```

   The 4-GPU ceiling is the one that counts: it has a halo, and 1 GPU has
   none. `YGG_CEILING_START`, `_STOP` and `_FACTOR` change the sweep.

4. **Weak scaling** 1 → 4 → 16 → 64 → 128 GPUs. Use `N_PER_DEVICE` at about
   half the 4-GPU `value_and_grad` ceiling.

   ```bash
   N_PER_DEVICE=16777216 ICS="uniform clustered" bench/slurm/submit_stats_scaling.sh horeka weak
   ```

   Lengths are in units of the mean interparticle spacing (`--length-unit
   spacing`, the default), so the work per particle stays fixed along the
   curve: ξ bins span 0.5 to 2.5 spacings (r_hi ≈ 3.4), and kNN `r_max` is 3
   spacings. Extra `scaling.py` arguments go after the stage name, or in
   `ARGS`.

5. **Strong scaling** at a fixed N that fits on the smallest count of the
   series. With 4 GPUs as the smallest, `N_TOTAL ≈ 4 × N_PER_DEVICE`.

   ```bash
   N_TOTAL=67108864 bench/slurm/submit_stats_scaling.sh horeka strong
   ```

6. **Flagship** (C4): 2048³ = 8.6e9 particles. First check the GPU count
   against the measured ceiling. 128 A100s means 6.7e7 per GPU, which is
   expected not to fit (see *Expected ceilings*).

   ```bash
   GPUS=512 N_PER_DEVICE=16777216 TIME=06:00:00 \
     bench/slurm/submit_stats_scaling.sh horeka flagship
   ```

   The JSONs of steps 4–6 are the scaling evidence for the JUWELS proposal.
   JUWELS Booster runs the same stages with `juwels` instead of `horeka`.

## Reading a result

- `overflow` must be `false`. The driver sizes every capacity with
  `DistributedStatsConfig.for_box`. On an overflow it grows the stage that
  overflowed and reruns (up to `--retry-overflow`), and `overflow_retries` logs
  what it did. A run that still overflows measured a truncated problem.
- `counters_summary` against `config` shows how much slack the sizing left: for
  example `halo.max` / `halo_capacity`, `near_pairs.max` / `near_cap`,
  `summary_recv.max` / `summary_recv_capacity`. **Memory follows the
  capacities, not the live counts.** The walk's `max_pair_queue` and `near_cap`
  cost about 156 B per slot each in XLA's plan, and they dominate. Ways to cut
  the slack:
  - `--tighten 1.25`: after a clean probe, refit every capacity to 1.25× its
    measured counter and probe again. `tightened` records the outcome. The
    ceiling runs use it, with `--clustering 1.0`.
  - Lower `--clustering` (default 1.5 for uniform ICs, 4 for clustered).
  - Pass explicit fields with `--config-json '{"near_cap": ...}'`.
- `timings.stages`: `partition_s` is the sample sort and its exchange;
  `halo_s` adds the local tree, the summary exchange and the halo import, with
  the payload's maximum reach (kNN's adaptive first pass is not in it);
  `walk_and_statistic_s` is the rest of the forward.
- `memory.compiled.temp` is XLA's plan for one device. `memory.runtime.peak_bytes`
  is what the allocator actually reached, per GPU. In a job that runs
  `forward,value_and_grad`, the second peak includes the first. The ceiling
  rows use one mode per row.

## Expected ceilings (from XLA's CPU plan; to be replaced by step 3)

These numbers come from compiling the forward pass on CPU (`scaling.py
--compile-only`), which allocates nothing. The setup: 2048³ split over 1024
devices (8.4e6 per device), ξ, float32, uniform. Costs per device:

- The pipeline without the walk's lists (partition, local tree, summary
  exchange, halo import, combined tree, and the statistic with 1M-slot walk
  capacities) plans at **2.55 GB**, about 300 B per particle.
- Every slot of `max_pair_queue` and every slot of `near_cap` adds about
  156 B. Both must cover the near leaf pairs.
- The near leaf pairs are measured on CPU (kNN uses its adaptive first pass):

  | payload, reach | near leaf pairs per particle |
  |---|---|
  | ξ, 2.8 mean spacings | ~12 |
  | ξ, 3.4 mean spacings | ~17 (model) |
  | kNN, k = 8, `r_max` = 2.5 spacings | ~4 |

So with 1.3× slack on the walk lists:

| payload | bytes per particle | particles per A100-40GB | 2048³ needs |
|---|---|---|---|
| ξ, 3.4 spacings | ~7 KB | ~5e6 | ~1500 GPUs |
| kNN, k = 8 | ~1.9 KB | ~2e7 | ~400 GPUs |

The ξ cost scales as `(reach + leaf radius)³`, so at 2 spacings it is about a
third. The CPU plan is not the GPU plan, and the CPU plan cannot include the
native ragged exchange, so read these as the order of magnitude to plan the
first allocation with. The ceiling runs (step 3) give the real numbers.

## Not yet run on a GPU

Everything here is tested on fake CPU devices only: the drivers, the neighbour
summary exchange, the sizing, `scaling.py` and its stage probes. Not exercised
yet:

- the native ragged all-to-all inside a node and across nodes;
- `jax.distributed` under SLURM;
- the GPU memory numbers;
- the sizing model at 10⁷–10⁸ particles per device.

Steps 1–3 above are where each of these is first checked.
