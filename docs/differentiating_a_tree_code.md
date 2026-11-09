# Differentiating a tree code

This note backs the section of the same name in *Yggdrax I*. It answers three
questions about the gradient of the traced soft pair counts
(`yggdrax.applications.corrfunc.traced`):

1. what gradient you get;
2. how accurate it is;
3. where it breaks, and how the smooth acceptance rule fixes it.

The benches behind every number are in `bench/differentiability/`. The JSON and
figures of the reference runs are kept on the paper branch, under
`bench/differentiability/results/`. All numbers come
from CPU runs in float64 (Apple M-series, jax 0.8.3).

## Setup

The estimator is `DD_k = Σ_{i<j} w_k(r_ij)`, with the soft log-bin window
`w_k(u) = σ(S(u − a_k)) − σ(S(u − b_k))`, `u = ln r`. The tree is a cell tree
(`build_static_cells_tree`) and the walk is the symmetric mutual walk. Every
unordered particle pair falls in exactly one of four groups:

- a leaf's own block (exact);
- a near leaf pair (exact);
- a far node pair (the monopole `n_a n_b w_k(|com_a − com_b|)`);
- a pruned node pair (`0`).

The bench payload is `ξ_k = DD_k / RR_k − 1`, against a fixed uniform random
catalogue (RR carries no gradient), with the loss
`L = Σ_k ln²((1 + ξ_k) / (1 + ξ*_k))` against a clustered target. All benches use
10 log bins over [0.02, 0.3], `S = 40`, leaf size 16, and N points in the unit
cube.

## 1. What gradient you get

The topology is built from `stop_gradient(positions)`. It has three parts:

- the Morton order;
- the leaf partition;
- the walk's far, near and pruned lists.

Given the topology, the counts are a smooth function of the positions. Gradients
flow through the exact near-field separations and through the centres of mass
of the far pairs. `jax.grad` therefore returns the **fixed-topology gradient**:
the exact gradient of the estimator on the open set where no tree decision
changes. The topology is piecewise constant, so this is the gradient of the
estimator almost everywhere, but not across a switch (Section 3).

For a far pair `(A, B)` and a particle `i ∈ A`, the tree gradient is
`n_B f'(D) D̂`, where `f = Σ_k c_k w_k`, `c_k = ∂L/∂DD_k`, and `D` is the
centre-of-mass separation. The exact gradient is `Σ_{j∈B} f'(r_ij) r̂_ij`.
Every particle of `A` gets the same far-field force from `B`. That is the
monopole approximation of the force, one order below that of the value.

**kNN-CDFs.** The tree is used only for the neighbour *search*, under
`stop_gradient`. The distances are then recomputed from the live positions, so
given the neighbour set the gradient is exact: no far field is approximated.
`bench/differentiability/gradient_accuracy.py` checks this against a dense
`top_k` reference that is differentiated end to end (N = 2000, k = 1, 2, 4, 8):

- the neighbour sets are identical;
- the losses agree to the last bit;
- the gradients agree to 1.2e-15 (max relative).

A kNN-CDF is still only piecewise smooth: the k-th distance is a continuous
order statistic, with a kink where two neighbours swap rank. The tree adds no
error to it.

## 2. How accurate it is

### Bounds

Every particle separation of a node pair lies in `I = [d − s, d + s]`, where `d`
is the distance of the bounding-box centres and `s = r_a + r_b` is the sum of
the bounding-sphere radii. The centres of mass lie in the spheres too, so
`D ∈ I` and `|r_ij − D| ≤ 2s`. Hence:

- **Value**, per bin:
  `|ΔDD_k| ≤ Σ_far n_a n_b (max_I w_k − min_I w_k) + Σ_pruned n_a n_b max_I w_k`.
  - The window rule accepts only pairs whose variation is at most `ε`, and
    prunes only pairs whose window is at most `ε`. This is the rule's guarantee:
    at most `ε` per particle pair.
- **Gradient**, per particle:
  `|Δg_i| ≤ Σ_{far P ∋ i} n_other (Var_I f' + |f'(D)| · 4s / (d − s))`.
  - The second term is the direction error: `|x̂ − ŷ| ≤ 2|x − y| / max(|x|, |y|)`.
  - Unlike the value, the gradient error is not controlled by the window
    variation of `w` alone. It needs the variation of `w'`.

The bench evaluates both bounds a posteriori, from the walk's own far list
(`Var` and `max` on a 65-point grid of `I`).

### Measurements

`bench/differentiability/gradient_accuracy.py`, clustered catalogue (half the
points in 12 Gaussian blobs of width 0.03), N = 4000. "value" is the max per-bin
relative error of `DD`; "grad" is `|g − g_ex| / |g_ex|` (L2 over particles) of
the ξ loss. Each cell is hard / smooth (β = 2).

| rule | value | grad |
|---|---|---|
| geometric θ = 0.2 | 5.3e-3 / 2.2e-3 | 5.7e-2 / 2.7e-2 |
| geometric θ = 0.3 | 1.6e-2 / 5.9e-3 | 1.3e-1 / 7.1e-2 |
| geometric θ = 0.4 | 3.3e-2 / 1.5e-2 | 2.2e-1 / 1.3e-1 |
| geometric θ = 0.5 | 5.4e-2 / 2.6e-2 | 3.7e-1 / 2.0e-1 |
| geometric θ = 0.6 | 7.4e-2 / 4.6e-2 | 5.8e-1 / 3.2e-1 |
| geometric θ = 0.8 | 1.9e-1 / 7.4e-2 | 1.5 / 6.2e-1 |
| window ε = 1e-1 | 2.7e-4 / 2.1e-4 | 7.3e-4 / 8.1e-4 |
| window ε = 1e-2 | 3.4e-5 / 2.6e-5 | 2.7e-5 / 2.6e-5 |
| window ε = 1e-3 | 4.3e-6 / 3.2e-6 | 3.5e-6 / 3.3e-6 |
| window ε = 1e-4 | 5.2e-7 / 3.9e-7 | 3.8e-7 / 3.4e-7 |
| window ε = 1e-6 | 7.7e-9 / 5.9e-9 | 5.1e-9 / 5.0e-9 |
| window ε = 1e-8 | 1.0e-10 / 7.9e-11 | 6.6e-11 / 6.6e-11 |

N = 1000 and 2000 are in the JSON. The window rows barely move with N. The
geometric gradient error grows with N: at θ = 0.5 it is 0.18, 0.42 and 0.37 at
N = 1000, 2000 and 4000.

Reading:

- **Window rule.** Value and gradient errors both scale linearly with `ε`, at
  `ε / 300` to `ε / 100` relative. The gradient is as accurate as the value: the tree
  is used to prune the binned range's outside and to sum the rest exactly, and
  the few far pairs sit where the windows are flat, so `w'` is flat too. At the
  default `ε = 1e-6` both errors are ≈ 5e-9.
- **Geometric rule.** The gradient is 5–10× worse than the value: 13–58 % at
  θ = 0.3–0.6, against 2–7 % in value. The monopole moves every particle of a
  far node by the same force, and a sharp window's `w'` changes sign across a
  bin edge, inside one pair's separation range. A geometric θ that is fine for
  values is not fine for gradients of sharply binned statistics. The smooth rule
  roughly halves both errors, because its band refines what the hard rule
  accepts at `θ/β < (r_a + r_b)/d < θ`, at about 4.5× the cost (Cost, below).
- **Bounds.** The value bound holds bin by bin in every run. The gradient bound
  holds per particle for the geometric rule (all particles), checked against
  the counts' VJP with the exact cotangent. It is conservative by 7× (θ = 0.2)
  to 180× (θ = 0.8) in L2, because it adds the worst case of every far pair. It does not cover the window
  rule's pruned pairs, which the walk does not emit. That part is the dominant
  window gradient error at small ε, so only the far part is plotted there. See
  `results/gradient_accuracy_cpu.png` (right panel).
- The loss gradient also moves through `∂L/∂DD` evaluated at the tree's counts.
  That is a value-error effect on every particle, and the bound leaves it out.

## 3. Where it breaks

Four tree decisions are piecewise constant in the positions:

- **prune** (window rule): a pair whose windows are all ≤ ε is dropped;
- **accept / refine**: the far monopole is swapped for the sum over the pair's
  children;
- **split order**: a refined pair splits the node with the larger bounding
  radius, so when the radii cross, the pair's descendants change;
- **leaf membership**: a particle crossing a cell face moves to another leaf,
  which changes every node on the paths to the two leaves' common ancestor, and
  possibly splits or merges cells.

At each switch the value jumps by the approximation error of the pairs
involved, and the gradient jumps with it. This is a jump in the estimator, not
in the gradient code. The consequences:

- **Finite differences** across a switch measure the jump divided by `h`.
- **Line searches** see a discontinuous objective.
- **HMC** pays the jump in full in the energy error: leapfrog integrates the
  force, and the force carries no delta function at the switch. The jumps add a
  floor to `|ΔH|` that does not shrink with the step size.

## 4. The smooth acceptance rule

`TracedPairCountConfig(smooth_band=β > 1)`, for both the window and the
geometric rule. It is a policy on the walk's hooks, plus one small engine hook
(`pair_band`, below).

### Definition

Each visited node pair `P` gets two fractions, both C² ramps in log space:

- a far fraction `σ_P ∈ [0, 1]`:
  - window rule: of the window variation `v_P = max_k (max_I w_k − min_I w_k)`,
    1 at `v ≤ ε/β`, 0 at `v ≥ ε`;
  - geometric rule: of `(r_a + r_b) / d`, 1 at `≤ θ/β`, 0 at `≥ θ`;
- a keep fraction `κ_P` (window rule only): of `m_P = max_k max_I w_k`, 0 at
  `m ≤ ε/β`, 1 at `m ≥ ε`.

The pair contributes

```
C(P) = κ_P · ( σ_P · far_P + (1 − σ_P) · Σ_children C(Q) ),
```

where a leaf pair's children sum is its exact sum. Unrolled, every emitted term
carries a weight that is a product along its ancestry:

- a far term: `σ_Q κ_Q Π_{A ⊃ Q} κ_A (1 − σ_A)`;
- a near term: `Π_{A ⊇ Q} κ_A (1 − σ_A)`.

Across the band edges:

- where `σ → 1`, the descendants' weight goes to 0 before the walk stops
  visiting them;
- where `κ → 0`, everything below goes to 0 before the pair is pruned.

So at a fixed leaf partition the sum is continuous. The ramps are C², so it is
C¹ wherever the min / max inside `v`, `m` and the radii does not switch, and
those kinks only matter inside a band. The error bound of the hard rule at the
band's outer edge (`ε`, `θ`) still holds.

### Implementation

1. **Engine hook: `dual_tree_walk_mutual(pair_band=..., band_cap=...)`.** This
   is the one engine change, about 100 lines in the flat walk, with a trace-time
   branch so walks without the hook are bit-identical. It was needed because a
   band pair must be *both* emitted far and refined, and the existing hooks can
   express either but not both.
   - A band pair (`0 < σ < 1` or `κ < 1`) is emitted far if accepted, and also
     refined.
   - It is logged in a band list, and every queued pair carries the band-list
     slot of its nearest band ancestor.
   - The walk returns that slot for every emitted pair (`far_parent`,
     `near_parent`), plus the band list's own parent links.
   - `tests/unit/test_mutual_walk_pair_band.py` checks that the weighted lists
     are a partition of unity over the cross-leaf particle pairs, to 1e-12.
2. **Weights recomputed with gradient.** The walk sees only `stop_gradient`
   geometry. The accumulation recomputes `σ` and `κ` for every band pair and
   far pair from *live* node geometry: the exact range min / max of the
   positions, from a sparse table, so the values equal the walk's bit for bit.
   It then forms the ancestry products by pointer doubling over the parent
   links. `grad` therefore includes the `dσ/dx` terms.
   - They matter. With frozen weights, the missing force is
     `Σ_band (M − R) ∇σ`, which integrates along a path to exactly the hard
     rule's jump. That would leave the HMC energy error unchanged.
3. **Split by count.** The smooth walk splits the node with more particles.
   That order is constant at a fixed leaf partition, so the decomposition under
   a band pair never flips. The hooks read the true radii from their data.
4. **Planner.** `plan_traced_pair_counts(smooth_band=β)` sizes `band_cap` the
   same way it sizes the other lists. A band overflow raises `band_overflow`
   and `overflow`.

What is left is the **leaf membership** switch. Making it continuous would need
a tree whose topology does not depend on the positions. Instead,
`traced_soft_pair_counts(membership=x0)` builds the leaf partition from `x0`
and everything else from the live positions. The geometry stays exact, so the
error bounds still hold.

- Pass the start of an HMC trajectory or of a line search. The smooth objective
  is then continuous along all of it.
- Rebuilding between trajectories makes the potential depend on the
  trajectory's start, a bias at the level of the far error. It is unmeasurable
  for the window rule.

### Continuity, measured

These are trapezoid residuals `|ΔF − ½(∇F + ∇F')·Δx|`: `O(Δx³)` for a C¹
function, and the jump size at a jump.

| scan | hard | smooth |
|---|---|---|
| one particle, 0.008 path, 801 points (no restructuring), geometric θ = 0.5 | 3.6e-7 jump (per-step change 2.6e-7) | 4.8e-12 everywhere (round-off floor) |
| same, window ε = 1e-3 | no switch on this path | 4.8e-12 |
| all 400 particles, 0.01 RMS path, 81 points (leaves change), geometric smooth | rebuilt: 1.8e-3 max | `membership=x0`: 1.47e-4 max, which is the exact estimator's own residual on this path (1.47e-4) |


The test `tests/applications/test_corrfunc_smooth.py::test_a_scan_across_acceptance_switches_jumps_only_under_the_hard_rule`
holds the single-particle case.

### Cost

The blend costs whatever the band refines, on top of the hard rule at the outer
edge. N = 4000, value + gradient, CPU (timings taken with other benches
running, so they are indicative only):

| rule | hard: far / near pairs | smooth: far / near / band pairs | time ratio |
|---|---|---|---|
| geometric θ = 0.5 | 37k / 14k | 341k / 67k / 200k | ≈ 4.5× |
| window ε = 1e-3 | 2.9k / 108k | 10k / 111k / 7.4k | ≈ 1.1–1.2× |
| window ε = 1e-6 | 1.5k / 150k | 11k / 155k / 9.4k | ≈ 1.2–1.9× |

The window band is thin, because the sigmoid tails and flat interiors cross
`ε/β … ε` over a short range of separations. The geometric band
`θ/2 … θ` is a factor-of-two shell in opening angle. A smaller β is cheaper but
makes `σ` steeper, which matters for HMC through the curvature it adds.

## 5. Along trajectories (A5b, A5c)

### A5b: switches along trajectories

`bench/differentiability/switch_rates.py`, N = 1000. Two paths are generated
once with the exact gradient and replayed under every rule:

- **descent:** 200 steps of normalised gradient descent toward the target ξ,
  RMS step 1e-4;
- **leapfrog:** 200 leapfrog steps of the A5c potential, ε = 5e-4, which moves
  each particle ≈ 9e-4 per step.

"Jump" is `|F(x') − F_x(x')|`: the step's end value minus the value of the
estimator continued from `x` without a switch. For a hard rule that is the
whole topology of `x`; for a smooth rule, the leaf partition of `x`. Values are
median / max, against a median per-step change in `F` of 4.9e-2 (descent) and
2.2e-2 (leapfrog).

| path | rule | steps with a restructuring | jump, steps without | jump, steps with |
|---|---|---|---|---|
| descent | geometric θ=0.5 | 38 % (1.7 particles) | 8.6e-5 / 4.3e-3 | 7.4e-4 / 1.3e-2 |
| descent | geometric, smooth | 38 % | **0 / 3.6e-15** | 1.4e-4 / 5.4e-3 |
| descent | window ε=1e-3 | 38 % | 1.9e-10 / 1.8e-8 | 5.2e-9 / 4.9e-7 |
| descent | window, smooth | 38 % | **0 / 3.6e-15** | 3.5e-9 / 4.7e-7 |
| leapfrog | geometric θ=0.5 | 100 % (25 particles) | – | 1.6e-3 / 1.9e-2 |
| leapfrog | geometric, smooth | 100 % | – | 5.2e-4 / 5.5e-3 |
| leapfrog | window ε=1e-3 | 100 % | – | 2.4e-8 / 3.2e-7 |
| leapfrog | window, smooth | 100 % | – | 1.5e-8 / 2.5e-7 |

- **Every step switches something.** On the descent path, all steps without a
  restructuring switch 11–17 node pairs under the geometric and smooth rules,
  and 81 % of them switch about 2 pairs under the hard window rule.
- **Hard geometric jumps are comparable to a whole step's change:** up to 1e-2
  against 5e-2. The gradient jumps by up to 7 % (max of `|Δ∇F| / |∇F|`).
- **The smooth rule removes the MAC-switch jumps exactly**, to round-off
  (gradient jumps ≤ 1e-13). The restructuring jumps remain, about 3–5× smaller
  than the hard rule's. They are set by the far error of the pairs that change.
- **Under the window rule every jump is below 5e-7, i.e. 1e-5 of a step.** The
  far error is ≤ ε per particle pair, and the far list is small.
- **HMC-sized steps restructure the tree every step.** At N = 1000, about 25
  particles change leaf per leapfrog step, so in HMC the MAC switches cannot be
  separated from restructuring.

### A5c: leapfrog energy error and HMC acceptance

`bench/differentiability/hmc_energy.py`, N = 500. The potential is
`U = 30 L + wall`, with unit masses and 20 leapfrog steps. There are 24
trajectories per point, from clustered starting states and shared momenta. A
20-step trajectory at ε = 5e-4 moves each particle about 0.017 RMS (about the
smallest bin edge), with dozens of leaf changes. `|ΔH|` is the mean over the 24
trajectories; acceptance is the mean of `min(1, e^{−ΔH})`.

| potential | ε = 1.25e-4 | 2.5e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|
| exact (brute force) | 1.1e-3 | 5.3e-3 | 4.0e-2 | 0.34 |
| geometric θ=0.5 | **0.12** | **0.16** | 0.16 | 0.48 |
| geometric, smooth β=2 | 3.6e-2 | 6.4e-2 | 8.1e-2 | 0.40 |
| geometric, smooth, `membership=x0` | 1.1e-3 | 5.2e-3 | 3.9e-2 | 0.33 |
| window ε=1e-3 (hard, smooth, frozen) | 1.1e-3 | 5.3e-3 | 4.0e-2 | 0.34 |

Acceptance at ε = 1.25e-4 is 1.00 for exact, window and frozen smooth, 0.98
for smooth, and 0.94 for hard geometric. Figure: `results/hmc_energy_cpu.png`.

- **The hard geometric rule has an ε-independent floor**, `|ΔH| ≈ 0.12–0.16`:
  the switch jumps enter `ΔH` whole. Shrinking the step does not help. At
  ε = 1.25e-4 the energy error is 100× the exact potential's.
- **The smooth rule removes the MAC part.** The rest of the floor comes from
  restructuring. With the leaf partition frozen per trajectory, the tree
  potential's energy error equals the exact potential's at every step size, to
  2 digits.
- **The window rule at ε = 1e-3 is already indistinguishable from exact in
  every variant.** Its jumps (≤ 3e-7) are far below the integration error.

## Recommendations

- **For gradients of binned statistics, use the window rule.** Its value and
  gradient errors are both ε/300 to ε/100, and its switches are below any integrator's
  error, so it needs neither the smooth band nor frozen membership. The default
  ε = 1e-6 gives ≈ 5e-9.
- **Do not use the geometric MAC at θ ≳ 0.2 for gradients with sharp windows.**
  Its gradient error is several times its value error (13–58 % at θ =
  0.3–0.6).
- **If a geometric (or loose-ε) rule is needed** in HMC, line searches or
  finite-difference checks: use `smooth_band=2` and pass
  `membership=<trajectory start>`. The objective is then continuous along the
  trajectory, and the energy error is the integrator's alone. Re-planning the
  capacities over the positions a run visits is unchanged
  (`plan_traced_pair_counts(..., smooth_band=β)`).
- **kNN-CDFs need none of this.** The tree only finds neighbours, and the
  gradient is exact given them.

## Reproducing, and what is untested

```bash
python bench/differentiability/gradient_accuracy.py      # A5a, CPU
python bench/differentiability/switch_rates.py           # A5b
python bench/differentiability/hmc_energy.py             # A5c
```

On the GPU server, pick a free card with autocvd (the org rule):

```bash
python bench/differentiability/gradient_accuracy.py --platform gpu --gpu-select free --ns 4000,16000,64000
python bench/differentiability/switch_rates.py --platform gpu --gpu-select free --n 16000
python bench/differentiability/hmc_energy.py --platform gpu --gpu-select free --n 8000
```

Untested here:

- **GPU.** The engine change, the sparse-table geometry (`O(N log N)` memory,
  about 1 GB at N = 1e6 in float64) and the benches at large N.
- **Periodic boxes.** `traced.py` has no `box_size` yet. The band hook
  respects the walk's periodic guard (a guarded pair is not emitted far, and
  `band_accepted` records it), but the smooth rule is not wired to a box.
- **The distributed driver.** `yggdrax/distributed` does not use the smooth
  rule.
- **Float32.** All runs are float64.
- **A full HMC chain.** A5c measures energy errors and acceptance from fixed
  starts; it does not run a chain to convergence.
- **The bias of freezing membership per trajectory.** The potential then
  depends on the trajectory's start, at the level of the far error. For the
  window rule that is below 1e-8; it was not measured separately.
