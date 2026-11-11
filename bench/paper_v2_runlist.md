# Paper v2 single-GPU run list

One command on the group's shared A100 server produces every single-GPU number
that Yggdrax I v2 quotes:

- at **one jax version**, 0.11.2, the version CI pins;
- on **one hardware type**;
- from **one commit**;
- each JSON stamped with the **same provenance record**.

This is the referee's "single re-run instead of mixed jax versions and
hardware". The driver is `bench/paper_v2_runlist.py`. Multi-node scaling (C3,
HoreKa/JUWELS) is separate: see `bench/slurm/README.md`. The v1 list
`bench/paper_runlist.sh` is superseded.

## Commands, in order

Run them on the A100 server, from a checkout of the paper branch
(`paper/differentiable-applications`), at its repository root.

**0. The code.** It is best to start after B2 (`claude/b2-xi`) is merged into the
paper branch: then its two rows run at the same commit as the rest (see
"B2 slot" below).

```bash
git fetch origin && git checkout paper/differentiable-applications && git pull
```

**1. The environment, once (~10 min).** This is the C3 environment, jax 0.11.2
with the CUDA 12 pip wheels, so only the NVIDIA driver comes from the system
(≥ 525; ≥ 575 avoids PTX-JIT fallbacks). It needs a Python 3.12.

```bash
bash bench/slurm/env/build_env.sh generic ~/envs/yggdrax-jax0112
```

If the server has no `python3.12`, use one from micromamba:

```bash
micromamba create -y -n py312 python=3.12 && PYTHON=$(micromamba run -n py312 which python) bash bench/slurm/env/build_env.sh generic ~/envs/yggdrax-jax0112
```

**2. The run list's extras and the baselines, once (~5 min).**

- autocvd and matplotlib are required.
- jaxkd, JZ-Tree, KeOps and TreeCorr are optional. Each installs on its own.
- The script ends by printing which baselines import.

```bash
bash bench/slurm/env/install_paper_v2_extras.sh ~/envs/yggdrax-jax0112
```

**3. Activate it** in every shell that runs the list.

```bash
source ~/envs/yggdrax-jax0112/bin/activate
```

**4. Look at the rows (instant).** This prints every row's command, walk and
estimated time.

```bash
python bench/paper_v2_runlist.py --list
```

**5. GPU smoke (~10 min, then ~3 min once KeOps has cached its kernels).** Every
row runs at tiny N on the GPU, with the same GPU picks, provenance and
exactness checks as the real run. Outputs go to
`~/yggdrax-paper-v2-runlist-smoke-gpu`. Go on only if it ends with `all
finished rows share jax 0.11.2, ...`.

```bash
python bench/paper_v2_runlist.py --smoke
```

**6. The run (~9 h on one A100; rough, see the table).** Run it inside tmux or
screen. Outputs go to `~/yggdrax-paper-v2-runlist`, or `$WORKSPACE/...` if
`WORKSPACE` is set.

```bash
python bench/paper_v2_runlist.py 2>&1 | tee -a ~/yggdrax-paper-v2-runlist.log
```

**7. Resume, and the optional rows.** Run the same command again to continue:
rows that finished are skipped, and failed or skipped rows run again. The C2
rows on 2 and 4 GPUs, and the native-ragged check, wait at most 10 min for
free cards and are skipped otherwise. Rerun them when the cards are idle:

```bash
python bench/paper_v2_runlist.py --rows '^c2_' --optional-gpu-wait 3600
```

**8. Check the summary.** A finished campaign prints and writes
`$OUT/summary.json`, with status, wall time, jax, device and commit per row.
Exit codes:

| code | meaning |
|---|---|
| 0 | all rows finished, one jax, one device, one commit |
| 1 | a row failed: its log is `$OUT/logs/<tag>.log` |
| 2 | the preflight refused to start |
| 4 | the rows disagree on jax, device or commit |

Copy `$OUT` back as one directory. "Where the paper reads it" below says which
file feeds which section.

## What a run does

- **Preflight.** Checks that jax is 0.11.2 (`--expect-jax`), that `yggdrax` is
  imported from this checkout, that the backend is the GPU and that the
  worktree is clean. `--smoke` and `--allow-dirty` only warn about a dirty
  tree.
- **Campaign record.** The first run writes `$OUT/campaign.json` (commit, jax,
  device, host), `pip-freeze.txt` and `nvidia-smi.json`. A later run from
  another commit or jax refuses to continue that `$OUT`. Use a new `--out`, or
  pass `--allow-new-commit`, which records each row's commit and turns the
  check into a warning.
- **GPU per row (org rule).** Right before each GPU row, the driver calls
  `autocvd(num_gpus=<row>, least_used=False)`, which waits until that many
  cards are idle. It then passes the bench `--gpu-select free` as well. The
  driver never imports JAX, so it holds no card between rows. If you export
  `CUDA_VISIBLE_DEVICES` yourself, autocvd is skipped and the rows use those
  cards.
- **Rows** follow the `bench/slurm/` convention:
  - outputs go to `$OUT/cases/<tag>/` and the log to `$OUT/logs/<tag>.log`;
  - on success, the driver writes `$OUT/rows/<tag>.json` (command, wall time,
    provenance);
  - on failure, it writes `$OUT/rows/<tag>.failed`, and the next run retries
    the row.

  `--rows REGEX` and `--skip-rows REGEX` select rows. `--stop-on-failure` stops
  at the first failure; by default the list carries on.
- **Provenance in every JSON.** Each JSON object a row writes gets a top-level
  `"runlist"` block with:
  - tag, arm and walk/backend;
  - jax and jaxlib versions, backend and device kind;
  - the cards' index, name, UUID, driver, memory and utilisation at the start;
  - `CUDA_VISIBLE_DEVICES`, `XLA_FLAGS` and the row's extra environment;
  - commit, branch and dirty flag;
  - the exact command, start time and wall time.

  The benches' own metadata blocks stay as they were.
- **Environment of every row.** `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so
  `peak_bytes_in_use` is the working set and other users keep their memory.
  Multi-GPU rows also get
  `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`, as in
  `bench/slurm/stats_scaling_common.sh`.

## The rows

The times are rough per-row estimates for one A100-40GB. They are guesses from
the CPU runs and the 2026-11-07 A1 run, not measurements, so treat the total
as "overnight". "Smoke" is the measured CPU smoke time on an M-series laptop.

| tag | arm | what | est. GPU | smoke (CPU) |
|---|---|---|---|---|
| `a1_scaling` | A1 | build + traversal, N = 1e3–1e6: legacy backends, fast path (flat + Pallas), cKDTree anchor | 25 min | 29 s |
| `a1_scaling_fast` | A1 | fast path alone, N = 2^11–2^26 (every power of two) | 45 min | 41 s |
| `a1_autodiff_overhead` | A1 | kNN value vs value+grad, N = 1e3–1e6 | 5 min | 3 s |
| `a5_gradient_accuracy` | A5 | gradient vs exact O(N²) autodiff, N = 4000, 16000, 64000 | 40 min | 57 s |
| `a5_switch_rates` | A5 | MAC switch jumps along GD/HMC paths, N = 16000 | 30 min | 103 s |
| `a5_hmc_energy` | A5 | leapfrog \|ΔH\| per rule, N = 8000 | 30 min | 50 s |
| `b1_knn_cdf_open` | B1 | kNN-CDF fwd + value+grad, N = 1e3–1e7, brute force to 1e5 | 20 min | 9 s |
| `b1_knn_cdf_periodic` | B1 | the same, periodic unit box | 30 min | 21 s |
| `b2_xi_box` | B2 | **slot**: periodic ξ, traced vs distributed vs TreeCorr vs Corrfunc | 30 min | skipped |
| `b2_acceptance_rules` | B2 | **slot**: window ε vs geometric θ at matched accuracy | 15 min | skipped |
| `b3_forecast` | B3 | point-process Fisher forecast + recovery, N = 32768 (GPU command of `docs/point_process.md`) | 60 min | 117 s |
| `b4_positions_1e6` | B4 | **slot**: move N = 1e6 particles until ξ and kNN-CDF match a Thomas target, 600 Adam steps, + gradient check | 40 min (guess) | – |
| `b4_positions_1e7` | B4 | **slot**: the same at N = 1e7, 100 steps (cost per step) | 40 min (guess) | – |
| `c2_grad_memory_g1` | C2 | distributed ξ/kNN forward vs value+grad memory and time, N = 2^20, 1 GPU | 10 min | 56 s |
| `c2_grad_memory_g2` | C2 | the same on 2 GPUs (optional) | 10 min | 76 s |
| `c2_grad_memory_g4` | C2 | the same on 4 GPUs (optional) | 10 min | 88 s |
| `c2_native_ragged_check` | C2 | `YGGDRAX_CHECK_UPSTREAM_RAGGED_FIX=1` test, ≥ 2 GPUs (optional, GPU only) | 3 min | skipped |
| `base_knn_open` | baseline | kNN forward: yggdrax vs jaxkd, jaxkd-cuda, JZ-Tree, cKDTree; N = 1e4–3e7, k = 8, 32 | 30 min | 3 s |
| `base_knn_periodic` | baseline | the same in a periodic box (yggdrax, JZ-Tree, cKDTree) | 30 min | 8 s |
| `base_kernel_sums_uniform` | baseline | soft pair counts: yggdrax traced vs KeOps vs dense JAX, N = 1e4–1e6 | 25 min | 26 s |
| `base_kernel_sums_clustered` | baseline | the same on a clustered catalogue | 25 min | 32 s |

The whole CPU smoke takes about 12 min on the laptop (compile-bound; most rows spend their time in XLA compilation, not in the sizes).

### Where the paper reads it

| tag | file under `$OUT/cases/<tag>/` | paper |
|---|---|---|
| `a1_scaling`, `a1_scaling_fast` | `scaling.json`, `scaling_fast.json` | §2/§3 scaling, fig_scaling (`results/differentiability/`) |
| `a1_autodiff_overhead` | `autodiff_overhead.json` | §3 autodiff ratios |
| `a5_*` | `<name>.json` + `.png` | §3 gradient accuracy, switch rates, HMC (`bench/differentiability/results/`) |
| `b1_*` | `knn_cdf_{open,periodic}.json` | §5 kNN-CDF showcase |
| `b2_*` | `xi_box.json`, `acceptance_rules.json` | §5 ξ showcase |
| `b3_forecast` | `results.json` + 4 figures | §5 end-to-end forecast |
| `b4_*` | `positions_1e{6,7}.json` + `.npz` + `.png` | §5 matching positions to ξ and kNN-CDF (B4) |
| `c2_*` | `grad_memory_g{1,2,4}.json`, `junit.xml` | §4 gradient memory, the native-exchange check |
| `base_*` | `knn_forward_*.json`, `kernel_sums_*.json` | §4 "the cost of pure JAX" |

## Competitor baselines

Both benches check exactness **before** timing, at every size. An arm that
disagrees with the exact reference is recorded with its error and **not
timed** at that size. Each arm stops at the size where it runs out of memory,
or where one call exceeds `--time-budget-s` (120 s by default); the other arms
go on. A missing package records `"available": false` with the reason, and
the row still succeeds.

**kNN forward** (`bench/baselines/knn_forward.py`). The time is one search from
positions, tree build included, at matched N, Q = N (independent uniform
queries, so no self-match conventions), k and dtype (float32).

- **yggdrax**: `build_kdtree` + `query_neighbors(backend="tree")`, the search
  B1's kNN-CDF uses.
- **jaxkd** 0.1.2: pure JAX. It breaks under `jax_enable_x64`: its `lax.cond`
  branches mix float32 and float64. The bench therefore traces it inside
  `jax.enable_x64(False)`, which is harmless for float32 inputs. jaxkd has no
  periodic boundaries.
- **jaxkd-cuda**: its CUDA extension. It is not pinned (CMake + nvcc build),
  and the arm skips without it.
- **JZ-Tree** 1.1.0 + `jztree-cu12`: `jztree.knn.knn(part, k, boxsize=,
  part_query=)`. CUDA only, so it skips on CPU.
- **cKDTree** on the host, all cores: context only.

The reference is a float64 cKDTree on the same rounded positions, for 2048 of
the queries. An arm must match the sorted distances to 1e-4 relative (float32)
and the neighbour sets on ≥ 99 % of the checked queries. The paper states the
gap as the cost of pure JAX, forward only: none of the baselines
differentiates.

**Kernel sums against KeOps** (`bench/baselines/kernel_sums.py`). SVGD left the
paper (D1), but KeOps still fits the v2 story. The soft pair counts behind ξ,
`C_k = Σ_{i<j} w_k(r_ij)`, *are* kernel sums. "Why not brute-force them exactly
on the GPU with KeOps?" is the obvious referee question for B2 (it was M8 for
SVGD).

The bench times forward and value+grad of `L = Σ c_k C_k`:

- **yggdrax**: `traced_soft_pair_counts`, window rule ε = 1e-6, with the tree
  rebuilt inside the differentiated function.
- **KeOps**: `pykeops.numpy`. The forward is one (N, K+1) reduction of the
  logistic gates. The gradient is the closed form `Σ_j (s/r²) Σ_m a_m g_m(1-g_m)
  (x_i - x_j)`: the numpy binding has no autodiff, and its torch binding would
  bring a second CUDA stack.
- **dense**: the pure-JAX blocked brute force of the A5 benches, with autodiff.

The radial bins span 0.25–4 mean spacings, so the tree's work per particle is
fixed while the brute forces grow as N². On CPU (N = 1000, 2000), KeOps agrees
with dense JAX autodiff to 1e-11 in the counts and 3e-15 in the gradient, and
the tree to 2e-10 and 7e-9.

## B2 slot

The B2 benches (`bench/corrfunc/xi_box.py`, `acceptance_rules.py`) are on
`claude/b2-xi`, which is not merged yet. Their rows `b2_xi_box` and
`b2_acceptance_rules` are already in the list, with the command line taken
from that branch (06c953a), and run as soon as the files exist. Until then
they print `skip b2_...: bench/corrfunc/xi_box.py not on this branch`.

- **If B2 merges before the run:** nothing to do.
- **If it merges after:** pull, then run only those rows:

  ```bash
  python bench/paper_v2_runlist.py --rows '^b2_' --allow-new-commit
  ```

  Each B2 JSON then records its own commit, and the summary warns about the
  second commit instead of failing.
- **B2 owner:** check the GPU sizes in the two rows (`_rows()` in the driver).

## B4 slot

The B4 bench (`bench/positions/optimise.py`) is on `claude/b4-positions`, for a
PR to main; it reaches this branch when main is merged in. Until then the rows
`b4_positions_1e6` and `b4_positions_1e7` print `skip b4_...: ... not on this
branch`. If it arrives after the run, run only those rows:

```bash
python bench/paper_v2_runlist.py --rows '^b4_' --allow-new-commit
```

The box grows with N at fixed density (L = 3.2 at 1e6, 6.9 at 1e7), so the
cost per particle per step is fixed. The 1e7 row runs 100 steps only: it
measures the cost per step and memory, not convergence.

## Locally (CPU)

```bash
python bench/paper_v2_runlist.py --smoke --platform cpu --out /tmp/rl-smoke
```

On CPU, the GPU-only rows (`c2_native_ragged_check`) and arms (JZ-Tree,
jaxkd-cuda) skip themselves. The multi-device C2 rows run on fake CPU devices.
KeOps on macOS needs OpenMP linked by hand:

```bash
CXXFLAGS="-L$CONDA_PREFIX/lib -lomp -Wl,-rpath,$CONDA_PREFIX/lib" python bench/paper_v2_runlist.py --smoke --platform cpu --out /tmp/rl-smoke
```

## Troubleshooting

- **A row seems to hang at `autocvd free xN ...`.** It is waiting for idle
  cards, which is the rule. `nvidia-smi` shows who is using them.
- **KeOps: "CUDA libraries not found ... Switching to CPU only".** Point it at
  the toolkit (`export CUDA_PATH=/usr/local/cuda`) and remove
  `~/.cache/keops*`.
- **JZ-Tree import error about the CUDA major version.** jztree checks that
  `jax-cuda12-plugin` matches `jztree-cu12`; both are CUDA 12 in the lock.
  `JZTREE_SKIP_JAX_CUDA_CHECK=1` bypasses the check, and the bench's exactness
  check still guards the result.
- **A size runs out of memory.** This is recorded in that JSON as
  `sweep.stopped_early`; the smaller sizes are kept and the row succeeds.
