# `results/differentiability/` — what each file is

Results for the differentiability sections (Yggdrax paper Sect. 2; cited as a precedent by
Jaccpot I §7). Produced by `bench/differentiability/*.py`; every payload carries a
`metadata` block with the jax version, backend, device, x64 flag and git SHA.

| File | Bench | What it holds |
|---|---|---|
| `nn_rebuild.json` / `.npz` | `nn_rebuild.py` | The original single-run existence proof: N = 256, 768 free coordinates, CPU. Read by `examples/differentiable_paper/fig_nn_rebuild.ipynb`. **Untouched by the scale-up**, so that figure is unchanged and still current. |
| `nn_rebuild_gradient_check.json` | `nn_rebuild.py --gradient-check-n` | **Gradient correctness vs N.** The certificate, not a convergence run. |
| `nn_rebuild_scaling.json` | `nn_rebuild.py --sweep-n` | **Series A** — fixed r\*, fixed lr. |
| `nn_rebuild_scaling_density_matched.json` | `nn_rebuild.py --sweep-n` | **Series B** — density-matched r\*, fixed lr. |
| `nn_rebuild_scaling_step_scaled.json` | `nn_rebuild.py --sweep-n` | **Series C** — density-matched r\*, N-scaled lr. **The citable curve.** |
| `scaling.json` | `scaling.py` | Build/traverse timing vs N. |
| `autodiff_overhead.json` | `autodiff_overhead.py` | Reverse-mode cost relative to forward. |
| `mac_accuracy.json` | `mac_accuracy.py` | MAC accuracy sweep. |

---

## The tree-rebuild scale-up (2026-09-02)

One NVIDIA A100-PCIE-40GB with **no co-tenant** (`metadata.gpu_contention_at_start` shows
0 MiB resident, 0 other processes), jax 0.10.2, x64 enabled, 3 seeds per N, 120 gradient
descent steps per run.

| | N | free coordinates |
|---|---:|---:|
| Committed 2026-07-17 point | 256 | 768 |
| Descent sweeps reach | **33 554 432** | **100 663 296** |
| Gradient check reaches | **67 108 864** | **201 326 592** |

### 1. The gradient certificate: the rebuild is differentiably transparent, exactly

A converging loss is not evidence of a correct gradient, so the gradient is certified
separately, at one configuration per N. Three comparisons, in
`nn_rebuild_gradient_check.json`:

**(a) Autodiff through the rebuild == autodiff at a frozen ordering, bit-for-bit.**
`build_tree` returns an integer ordering, so it should take zero cotangent. Measured
`max|∇L_rebuilt − ∇L_pinned| = 0.0` — **exactly zero, at every N from 256 to 67 108 864**,
in both float32 and float64. This is the strongest form the claim can take: differentiating
*through* the rebuild is not an approximation of the frozen-topology gradient, it *is* the
frozen-topology gradient.

**(b) That gradient matches central differences — once both discrete choices are pinned.**
The objective carries **two** discrete choices, not one: the tree ordering, and the `argmin`
inside its `min` over 2k neighbour candidates. Pinning the tree does not pin the argmin, and
the argmin's switching boundaries get dense at large N. Isolating them (float64, relative
error against the autodiff directional derivative):

| N | tree pinned only | tree **and** argmin pinned | improvement |
|---:|---:|---:|---:|
| 256 | 1.60e-10 | 7.50e-11 | 2× |
| 65 536 | 2.50e-11 | 2.50e-11 | 1× |
| 262 144 | 1.34e-3 | **1.70e-9** | 787 531× |
| 1 048 576 | 1.73e-3 | **4.28e-9** | 405 541× |
| 4 194 304 | 1.62e-2 | **4.02e-8** | 404 110× |
| 16 777 216 | 1.58e-2 | **8.01e-8** | 197 521× |

So with both discrete choices frozen, autodiff agrees with central differences to
**≤ 8.0e-8 relative at every N up to 16 777 216** (50 331 648 free coordinates). The apparent
large-N degradation was never the tree — it was the objective's own `min`. Anyone re-running
this and pinning only the tree will see 1e-2 and mistake it for a gradient bug.

**(c) Unpinned central differences are legitimately wrong.** Relative error 1e0 … 6e2 once
the ±ε stencil straddles a rebuild. That is the piecewise structure of the map, not a defect;
it is why the check must pin.

A useful control falls out of the same data: at N = 65 536 in float64, 0.01% of slots moved
across the stencil and the **unpinned** difference still agreed to 2.50e-11. A reordering
occurred with no effect on the loss or its derivative — direct local evidence that a
permutation-switch count over-counts what matters.

### 2. Why the old switch metric saturated, and what replaced it

`topology_changed` asks *did the length-N permutation change at all?* That is an **extensive**
indicator. If each particle changes slot with probability p, then P(no change) ≈ (1−p)^N, so
the statistic is driven to 1 by N alone. **Its saturation is a property of the estimator, not
of the tree** — it carries one bit per step, and at large N that bit is pinned. It is retained
only because the committed N = 256 point and the figure notebook use it.

The replacements are **intensive** — per-particle rates with a well-defined large-N limit:

| Field | Question |
|---|---|
| `slot_change_fraction` | what fraction of particles changed Morton slot? |
| `leaf_change_fraction` | what fraction changed *leaf*? (leaves are `rank // leaf_size` blocks) |
| `mean_abs_rank_shift_normalized` | how far did a particle move in the ordering, as a fraction of N? |

### 3. The result: a resolved curve where the indicator had none

Series C (converging at every N), 3 seeds, mean ± sd over seeds:

| N | free coords | loss ratio | any-change | slot frac | leaf frac | \|Δrank\|/N |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 768 | 5.60e-3 | 0.517 | 0.016 ± 0.011 | 0.0008 ± 0.0004 | 1.02e-4 |
| 1 024 | 3 072 | 4.53e-3 | 0.781 | 0.016 ± 0.003 | 0.0013 ± 0.0002 | 2.63e-5 |
| 4 096 | 12 288 | 4.37e-3 | 0.992 | 0.023 ± 0.009 | 0.0019 ± 0.0007 | 1.01e-5 |
| 16 384 | 49 152 | 5.06e-3 | **1.000** | 0.069 ± 0.029 | 0.0050 ± 0.0019 | 8.76e-6 |
| 65 536 | 196 608 | 4.96e-3 | **1.000** | 0.142 ± 0.031 | 0.0100 ± 0.0019 | 4.80e-6 |
| 262 144 | 786 432 | 5.13e-3 | **1.000** | 0.221 ± 0.014 | 0.0173 ± 0.0015 | 2.34e-6 |
| 1 048 576 | 3 145 728 | 4.79e-3 | **1.000** | 0.382 ± 0.044 | 0.0379 ± 0.0013 | 1.51e-6 |
| 4 194 304 | 12 582 912 | 5.61e-3 | **1.000** | 0.640 ± 0.081 | 0.0991 ± 0.0125 | 1.09e-6 |
| 16 777 216 | 50 331 648 | 5.50e-3 | **1.000** | 0.761 ± 0.053 | 0.1816 ± 0.0514 | 6.63e-7 |
| 33 554 432 | 100 663 296 | 5.23e-3 | **1.000** | 0.785 ± 0.023 | 0.2190 ± 0.0143 | 4.67e-7 |

Read the columns against each other. `any-change` is pinned at 1.000 from N = 16 384 and says
nothing thereafter. Over that same range the intensive rates resolve a clean monotone trend
with seed error bars:

- **`slot_change_fraction` rises 0.016 → 0.785** and is still rising at N = 3.4e7 — a real
  curve, not a ceiling.
- **`leaf_change_fraction` rises 0.0008 → 0.219.** This is the interaction-relevant one, and
  it is *four times smaller* than the slot rate: most particles that shuffle within the
  ordering stay in the same leaf.
- **`mean_abs_rank_shift_normalized` *falls* monotonically, 1.02e-4 → 4.67e-7.** Churn becomes
  more **local** as N grows. Its per-step displacement in absolute slots stays roughly
  proportional to N in the worst case (`max_abs_rank_shift` ≈ 0.44 N) while the mean shrinks.

So the honest summary of rebuild stability at scale is not "everything changes every step". It
is: **the ordering churns pervasively but locally — ~79% of particles shift slot, only ~22%
cross a leaf boundary, and the mean shift falls to under 1e-6 of N — while convergence stays
N-invariant (loss ratio 4.4e-3 … 5.8e-3 across six decades).**

### 4. Three series, because a fixed budget does not hold the experiment fixed

| Series | r\* | lr | Converges at large N? |
|---|---|---|---|
| A | fixed 0.12 | fixed 2.0 | no — loss ratio → 1.00 |
| B | ∝ N^(−1/3) | fixed 2.0 | no — loss ratio → 1.00 |
| **C** | ∝ N^(−1/3) | ∝ N | **yes — 5.2e-3 at N = 3.4e7** |

**A** fails two ways: holding r\* = 0.12 needs volume N·r\*³, which passes the `--clip` box
(volume 27) at N = 15 625 and overshoots it 2147× at N = 3.4e7; and above N ≈ 4096 the initial
spacing is already *below* r\*, flipping the task from compression to expansion.

**B** removes the geometry problem (packing ratio 0.016, initial-spacing/target 1.71 at every
N) and still does not converge, which isolates the real cause: the objective is a **mean**, so
the measured per-particle gradient scales as **N^(−4/3)** against a travel distance of
N^(−1/3), and progress per step under a fixed lr falls as 1/N.

B is also the cleanest illustration of the extensive/intensive distinction. At N = 3.4e7 its
`any-change` is 0.006 while `slot_change_fraction` is 0.0000 and `max_abs_rank_shift` is 6 —
on 0.6% of steps, about six particles shifted a few slots. The indicator reports "the topology
changed"; the intensive rates report, correctly, that essentially nothing moved. The state is
frozen, not stable.

**C** scales lr ∝ N to hold the descent dynamics N-invariant. Still plain gradient descent —
this changes the step size, not the optimizer. No Adam run was performed.

### 5. The N = 256 anchor reproduces across the backend change

Committed: CPU, jax 0.8.3. Re-measured on GPU at jax 0.10.2, same seed — agreement to
≤2.5e-7 relative (4.5e-7 max over the whole 120-step trajectory, ~4 ULP of float32) with
**all 120 per-step switch booleans identical** and the same 55/120 count. The curves contain
an N effect only. The committed point rides along in each payload as
`historical_reference`, labelled as a reference rather than as a curve's first point.

### 6. Where the wall is: memory, measured

- **N = 67 108 864 fails** with `RESOURCE_EXHAUSTED` on a single **27.75 GiB** allocation
  inside `jit_loss_only`, in all three series (`sweep.stopped_early`). float64 in the gradient
  check fails one rung earlier, as expected from doubling the element size.
- Peak device memory is linear at **528 bytes/particle** (470 MB at N = 1.0e6 → 17 717 MB at
  N = 3.4e7).
- Ceiling was `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` ≈ 34.8 GB of an otherwise idle 40 GB card,
  so this is close to a device limit rather than a self-imposed share; the same coefficient
  puts a fully-available A100-40GB near N ≈ 7.7e7.
- **Wall clock never bound:** 4.0–17.4 s per 120-step run; each 42-run series finished in
  292–299 s. Timings are clean — no co-tenant on any of these runs.

### 7. Limitations — carried in each payload as `limitations`, not just here

1. **The objective is geometric**, not an inference loss, and the operator is a radix tree's
   Morton ordering, **not the FMM with a MAC**. Reaching 1.0e8 free coordinates converts
   neither. The claim supported is "gradient descent through a per-step-rebuilt tree
   converges, and its gradient is exactly the frozen-topology gradient" — not
   "high-dimensional inference through rebuilds converges".
2. **These are permutation-derived rates** (`switch_metric: "morton_ordering"` on every
   payload and record), including `leaf_change_fraction`. A leaf-membership change is much
   closer to an interaction-list change than a slot change is, but it is still **not** one:
   yggdrax has no `MutualTopology`, and the dependency edge runs jaccpot → yggdrax, so
   `jaccpot/mutual/identity.py`'s `interaction_switches` is deliberately not used here. §7
   must measure its own rate with that facility rather than borrowing these.
3. **No measure was estimated.** These are switch events along an optimization path.
   `docs/differentiability_model.md`'s measure-zero statement stays a reasoned argument.
4. **A fixed lr is not an N-invariant budget** for a mean-reduction objective, so series A and
   B mix churn rates with how far each run got. Only C compares at matched progress.
5. **Plain gradient descent throughout**, by design.

### 8. The figure

`examples/differentiable_paper/fig_nn_rebuild_scaling.ipynb` →
`examples/differentiable_paper/figures/fig_nn_rebuild_scaling.{pdf,png}`. Two panels, built
from the committed JSON only — it never recomputes, so it needs no GPU:

- **(a)** the extensive indicator flat at 1.000 against the intensive rates resolving a trend
  over the same range, with seed error bars, plus mean rank shift on a right log axis.
- **(b)** the gradient certificate: unpinned / tree-pinned / tree-and-argmin-pinned FD error
  vs N, with the exactly-zero rebuild gap annotated. The notebook **asserts** that gap is
  `0.0` across every run rather than only claiming it in the caption, so the figure fails
  loudly if a future re-run breaks the exactness result.

`fig_nn_rebuild.{pdf,png}` (the N = 256 existence proof) is deliberately **not** regenerated —
its inputs were preserved byte-for-byte.

Note that sweep mode does not write a position history (at N = 3.4e7 over 121 steps it would be
~49 GB), so a positions-style panel like `fig_nn_rebuild`'s (a)/(b) is not available at scale
and would need a purpose-built run.

### 9. Reproducing

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
L="256 1024 4096 16384 65536 131072 262144 524288 1048576 2097152 4194304 8388608 \
   16777216 33554432 67108864 134217728"

# Series C -- the citable curve
python bench/differentiability/nn_rebuild.py --gpu-select free \
    --sweep-n $L --sweep-seeds 0 1 2 --steps 120 \
    --target-mode density-matched --lr-mode n-scaled \
    --output results/differentiability/nn_rebuild_scaling_step_scaled.json

# The gradient certificate
python bench/differentiability/nn_rebuild.py --gpu-select free \
    --gradient-check-n 256 1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 \
    --target-mode density-matched \
    --output results/differentiability/nn_rebuild_gradient_check.json
```

Swap `--target-mode fixed --lr-mode fixed` for series A, `--target-mode density-matched
--lr-mode fixed` for series B. `--smoke-sweep` and `--smoke-gradient-check` run the same code
paths at N = 64/128 in seconds. The original single-run invocation is unchanged and still
reproduces `nn_rebuild.json` byte-for-byte.
