# `results/differentiability/` — what each file is

Results for the differentiability sections (Yggdrax paper Sect. 2; cited as a precedent by
Jaccpot I §7). Produced by `bench/differentiability/*.py`; every payload carries a
`metadata` block with the jax version, backend, device, x64 flag and git SHA.

| File | Bench | What it holds |
|---|---|---|
| `nn_rebuild.json` / `.npz` | `nn_rebuild.py` | The original single-run existence proof: N = 256, 768 free coordinates, CPU. Read by `examples/differentiable_paper/fig_nn_rebuild.ipynb`. |
| `nn_rebuild_scaling.json` | `nn_rebuild.py --sweep-n` | **Series A** of the scale-up sweep — fixed r\*, fixed lr. |
| `nn_rebuild_scaling_density_matched.json` | `nn_rebuild.py --sweep-n` | **Series B** — density-matched r\*, fixed lr. |
| `nn_rebuild_scaling_step_scaled.json` | `nn_rebuild.py --sweep-n` | **Series C** — density-matched r\*, N-scaled lr. **The curve to cite.** |
| `scaling.json` | `scaling.py` | Build/traverse timing vs N. |
| `autodiff_overhead.json` | `autodiff_overhead.py` | Reverse-mode cost relative to forward. |
| `mac_accuracy.json` | `mac_accuracy.py` | MAC accuracy sweep. |

---

## The tree-rebuild scale-up sweep (2026-09-02)

**Largest N reached: 16 777 216 particles = 50 331 648 free coordinates**, at
`187d582`, on one NVIDIA A100-PCIE-40GB, jax 0.10.2, x64 enabled, 3 seeds per N,
120 gradient-descent steps per run.

That is a factor **65 536 in N** and **65 536 in free-parameter count** above the committed
N = 256 / 768 point.

### The N = 256 point reproduces across the backend and version change

The committed `nn_rebuild.json` was taken on **CPU at jax 0.8.3**. Re-measured here on
**GPU at jax 0.10.2** at the same seed, it agrees to float32 rounding and the discrete
switch sequence is *bit-identical*:

| | committed (CPU, jax 0.8.3) | re-measured (GPU, jax 0.10.2) | rel. diff |
|---|---|---|---|
| initial loss | 1.6939882e-2 | 1.6939880e-2 | 1.1e-7 |
| final loss | 5.8819209e-5 | 5.8819223e-5 | 2.5e-7 |
| initial mean NN | 0.20854491 | 0.20854490 | 7.2e-8 |
| final mean NN | 0.12436551 | 0.12436549 | 1.2e-7 |
| topology changes | 55 / 120 | 55 / 120 | — |

Max relative deviation over the whole 120-step trajectory is 4.5e-7, i.e. ~4 ULP of float32
(eps = 1.19e-7), and all 120 per-step switch booleans match. So the curve below contains an
N effect only — no backend or version effect is hiding inside it. The committed CPU point is
kept in each payload as `historical_reference`, labelled as a reference rather than as the
curve's first point.

### Three series, because a fixed learning rate is not an N-invariant budget

The prompted design was one fixed budget (same lr, clip, r\*, step count) at every N. Run
that way it does not measure what it looks like it measures, for two separate reasons — so
the sweep is reported as an ablation. All three series are plain gradient descent; no
adaptive optimizer was substituted.

| Series | r\* | lr | File |
|---|---|---|---|
| **A** | fixed, 0.12 | fixed, 2.0 | `nn_rebuild_scaling.json` |
| **B** | density-matched, ∝ N^(-1/3) | fixed, 2.0 | `nn_rebuild_scaling_density_matched.json` |
| **C** | density-matched, ∝ N^(-1/3) | N-scaled, ∝ N | `nn_rebuild_scaling_step_scaled.json` |

**Reason 1 — a fixed r\* becomes geometrically unreachable.** `--clip` bounds positions to
[-1.5, 1.5]³, a box of volume 27. Holding r\* = 0.12 while N grows demands a volume
N·r\*³ that passes the box at N ≈ 1.6e4 and overshoots it by **1074×** at N = 1.7e7
(`geometry.target_packing_ratio` in each record). Above N ≈ 4096 the initial spacing is
already *below* r\*, so the task silently flips from compression to expansion. Series B fixes
this by holding r\* at a constant multiple of the mean interparticle spacing; its packing
ratio is 0.016 and its initial-spacing-to-target ratio 1.71 at *every* N.

**Reason 2 — the objective is a mean, so the gradient shrinks as 1/N.** Measured on this
problem, the rms per-particle gradient scales as **N^(-4/3)** (1/N from the mean reduction,
N^(-1/3) from r\*), while the distance each particle must travel scales as N^(-1/3). Progress
per step under a fixed lr therefore falls as 1/N. Series B shows this cleanly: with the
geometry made scale-invariant, convergence *still* dies (loss ratio → 1.00 by N ≈ 2.6e5) and
the switch rate then *collapses* to 0.017 at N = 1.7e7 — not because rebuilds became stable,
but because the state was nearly frozen. Series C scales lr ∝ N to hold the descent dynamics
N-invariant.

### The result: at matched optimizer progress the switch rate saturates at 1.0

Series C converges to the *same relative degree* at every N across five decades:

| N | free params | r\* | lr | loss ratio (final/initial) | switch fraction |
|---:|---:|---:|---:|---:|---:|
| 256 | 768 | 0.12000 | 2 | 5.60e-3 | 0.517 ± 0.108 |
| 1 024 | 3 072 | 0.07560 | 8 | 4.53e-3 | 0.781 ± 0.068 |
| 4 096 | 12 288 | 0.04762 | 32 | 4.37e-3 | 0.992 ± 0.014 |
| 16 384 | 49 152 | 0.03000 | 128 | 5.06e-3 | 1.000 ± 0.000 |
| 65 536 | 196 608 | 0.01890 | 512 | 4.96e-3 | 1.000 ± 0.000 |
| 131 072 | 393 216 | 0.01500 | 1 024 | 4.90e-3 | 1.000 ± 0.000 |
| 262 144 | 786 432 | 0.01191 | 2 048 | 5.13e-3 | 1.000 ± 0.000 |
| 524 288 | 1 572 864 | 0.00945 | 4 096 | 5.15e-3 | 1.000 ± 0.000 |
| 1 048 576 | 3 145 728 | 0.00750 | 8 192 | 4.79e-3 | 1.000 ± 0.000 |
| 2 097 152 | 6 291 456 | 0.00595 | 16 384 | 5.82e-3 | 1.000 ± 0.000 |
| 4 194 304 | 12 582 912 | 0.00472 | 32 768 | 5.61e-3 | 1.000 ± 0.000 |
| 8 388 608 | 25 165 824 | 0.00375 | 65 536 | 4.84e-3 | 1.000 ± 0.000 |
| 16 777 216 | 50 331 648 | 0.00298 | 131 072 | 5.50e-3 | 1.000 ± 0.000 |

So the switch-rate trend is **monotone increasing and then saturated**: 0.52 at N = 256,
0.78 at N = 1024, 0.99 at N = 4096, and pinned at **1.000 for every N ≥ 16 384** — the
Morton ordering changes on *every single step* of a converging descent, with zero seed
scatter. Gradient descent through a per-step-rebuilt tree still converges there; what does
not survive scaling is any expectation that the ordering is stable between steps.

### Where the wall is, and why: memory, measured

The sweep was stopped by a **memory** wall, not by wall clock and not by convergence.

- N = 33 554 432 fails with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate
  12.00GiB` inside `jit_loss_only`, recorded as `sweep.stopped_early` in all three files.
- Peak device memory grows linearly at **~524 bytes/particle**: 461 MB at N = 1.0e6, 1099 MB
  at 2.1e6, 2231 MB at 4.2e6, 4463 MB at 8.4e6, 8791 MB at 1.7e7.
- The ceiling was **12.29 GB**, a deliberate `XLA_PYTHON_CLIENT_MEM_FRACTION=0.30` share of
  the 40 GB device (recorded as `metadata.xla_memory_env`), chosen so the sweep could not
  starve a co-tenant job. It is a budget we picked, not the card's limit — the same
  524 B/particle coefficient puts a full 40 GB A100's wall near **N ≈ 8e7**, which is an
  extrapolation, not a measurement. The wall sits at the same N in all three series because
  the cap is a fraction of total memory, independent of what else is resident.
- Wall clock was never binding: 4–17 s per 120-step run, 230–291 s for a whole 39-run series.
- **Read the timings per series, not across them** (`metadata.gpu_contention_at_start` /
  `_at_end`). Series A and B shared GPU 3 with a job holding 16.4 GB at 72–83% utilisation,
  so their wall clocks are **upper bounds**. Series C happened to be placed by `autocvd` on
  an idle GPU 7 (0 MiB resident, 0% utilisation, no co-tenant), so its 4.2 s → 10.2 s
  per-run figures are clean. The numerical results are unaffected either way; only the
  timings are.

### Limitations — carried in each payload as `limitations`, not just here

1. **The objective is geometric**, not an inference loss. Reaching N = 1.7e7 does not change
   that. The claim these runs support is "gradient descent through a per-step-rebuilt tree
   converges", **not** "high-dimensional inference through rebuilds converges".
2. **This is a permutation-switch rate**, recorded as `switch_metric: "morton_ordering"` in
   every payload and every record. It asks "did the Morton *ordering* change?", which is one
   component of a topology. A reordering can occur with no change in which pairs interact, so
   this rate is **≥** the rate of interaction-list change. It must **not** be plotted beside
   Jaccpot I's fig19 as the same quantity. This matters more now than before the sweep: a
   quantity saturated at its 1.0 ceiling is an upper bound that has gone uninformative, so
   "switch rate = 1.0" here implies **nothing quantitative** about fig19's interaction-switch
   rate. Jaccpot's `jaccpot/mutual/identity.py` separates `permutation` / `tree_shape` /
   `leaf_partition` / `node_ranges` / `far_pairs` / `near_pairs` and reports
   `interaction_switches` alongside `switches` for exactly this reason; it is deliberately
   not used here, because the dependency edge runs jaccpot → yggdrax and yggdrax has no
   `MutualTopology` to fingerprint.
3. **No measure was estimated.** What is measured is the frequency of switch events *along an
   optimization path*. `docs/differentiability_model.md`'s measure-zero statement is a
   reasoned argument and stays one.
4. **A fixed lr is not an N-invariant budget** for a mean-reduction objective (see above), so
   series A's and B's switch-rate curves mix the switch rate with how far each run got. Only
   series C compares runs at matched optimizer progress.
5. **Plain gradient descent throughout**, by design — it was chosen as an existence proof.
   Scaling lr with N changes the step size, not the optimizer. No Adam run was performed; one
   would be a separate series.

### Reproducing

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
L="256 1024 4096 16384 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432"

# Series C — the citable curve
python bench/differentiability/nn_rebuild.py --gpu-select least-used \
    --sweep-n $L --sweep-seeds 0 1 2 --steps 120 \
    --target-mode density-matched --lr-mode n-scaled \
    --output results/differentiability/nn_rebuild_scaling_step_scaled.json
```

Swap `--target-mode fixed --lr-mode fixed` for series A and `--target-mode density-matched
--lr-mode fixed` for series B. `--smoke-sweep` runs the same code path at N = 64/128 in a few
seconds; the original single-run invocation is unchanged and still reproduces
`nn_rebuild.json` bit-for-bit.
