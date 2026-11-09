# A differentiable point-process forward model (B3)

`yggdrax.applications.point_process` composes a point-process forward model with
the clustering statistics of `yggdrax.distributed.stats`:

```
theta = (sigma, beta) ──► positions (fixed draws) ──► xi(r), kNN-CDF ──► loss
                         wrap into [0, L)^3            tree, walk, halo
                                                      rebuilt every call
```

Every link is JAX, so `jax.jacfwd`, `jax.grad`, `jax.vmap` and `jax.jit` apply to
the whole chain. It is the paper's demonstration of what "everything is JAX"
buys:

- parameter gradients checked against finite differences of the full chain;
- a Fisher forecast from autodiff Jacobians and an ensemble covariance;
- parameter recovery on *independent* realisations.

There is no cosmological simulator in it (decision D2). The point process is a
stand-in with an analytic two-point function, which validates the chain end to
end. N-body snapshots enter the paper as data only.

## The model

A Thomas (Neyman–Scott) process in a periodic box. `P` parents are uniform in
`[0, L)^3`. Each parent has exactly `m` children, offset from it by an isotropic
random vector. The children are wrapped into the box (`yggdrax.periodic.wrap_positions`).

**Reparameterisation.** Every random number is drawn once from a seed: uniform
parents `p_a` and standard normals `z_ai`. The parameters enter only through the
offset map

```
x_ai = wrap( p_a + sigma c(beta) |z_ai|^(beta - 1) z_ai ),
c(beta) = sqrt(3 / E[|z|^(2 beta)]),   E[|z|^(2 beta)] = 2^beta Γ(3/2 + beta) / Γ(3/2).
```

So the positions are smooth functions of `theta` at fixed draws, and
`d/dtheta E[S] = E[dS/dtheta]` (the reparameterisation, or pathwise, gradient).

**Discrete parts are conditioned on.** The number of parents and the number of
children per parent are fixed per realisation, not Poisson. They have no
gradient; a realisation is a draw conditional on them. This changes ξ by a known
factor (below).

**The parameters.**

- `sigma` is the cluster's RMS size per axis: `E|offset|^2 = 3 sigma^2` for every
  `beta`. `c(beta)` exists to make that true.
- `beta` is the radial-profile shape at fixed RMS size. The offset radius is
  `r = sigma c(beta) rho^beta` with `rho = |z| ~ chi_3`. That is a monotone map
  of a fixed draw, i.e. a smooth inverse-CDF reparameterisation of the radial
  distribution, with a closed form. The cluster's density profile is
  `n(r) ∝ r^(3/beta - 3) exp(-(r / c sigma)^(2/beta) / 2)`:
  - `beta = 1` is the Gaussian, the Thomas process proper;
  - `beta > 1` has a central cusp of slope `3 - 3/beta` (`beta = 1.5` is the NFW
    inner slope `r^-1`) and a stretched-exponential tail;
  - `beta < 1` is cored and sharper-edged.

**Why `beta`.** It is the concentration of the halo model in one smooth number.
At fixed size it moves pairs between small and intermediate separations, so it
changes the shape of ξ rather than its amplitude. It also moves the local
densities that kNN-CDFs measure. Two other candidates were rejected:

- A clustered/unclustered mixture fraction is discrete per point. Making it
  smooth needs weights, which the kNN statistic does not take.
- A line-of-sight stretch (a Finger-of-God-like anisotropy) is smooth, but the
  monopole statistics see it mainly through the cluster volume `sigma^3 q`, so it
  is nearly degenerate with `sigma`.

## Validation against the analytic ξ

For the Poisson Thomas process, `xi(r) = exp(-r^2/4 sigma^2) / (n_p (4 pi sigma^2)^(3/2))`.

**Fixed counts.** The model fixes the counts. Of the `N(N-1)/2` pairs,
`P m(m-1)/2` are siblings with separation density `p_Delta` (Gaussian of
variance `2 sigma^2` per axis). Every other pair is uniform in the box. The
natural estimator's mean is therefore exactly

```
xi(r) = (m - 1)/(N - 1) (V p_Delta(r) - 1).
```

That is a factor `(1 - 1/m) N/(N-1)` on the amplitude, -6 % at `m = 16`, plus
the integral constraint `-(m-1)/(N-1)`. It matters: at the paper's sizes the
Poisson formula is rejected at many sigma, and the fixed-count one is not.
`window_average` integrates the analytic ξ against each soft window, which is
what the soft estimator measures. Without that step the comparison would carry
a binning bias.

For `beta != 1`, `p_Delta` has no closed form. `profile_xi_fixed_counts_mc`
draws sibling pairs and averages the windows of their separation, which is exact
in expectation.

## Statistics (pluggable)

A `Statistic` is `fn: positions -> (values, overflow)` plus labels. Jacobians,
ensembles, Fisher forecasts and fits see nothing else. Two are provided:

- **ξ** (`make_xi_statistic`): soft log-binned `DD` from
  `distributed_pair_counts` (periodic via `box_size`, exact acceptance), and the
  analytic soft `RR = N(N-1)/2 V_b / V` of the periodic box. Here
  `V_b = ∫ W_b 4 pi r^2 dr` over the same hard cut `[r_lo, r_hi)` the pair count
  uses. Natural estimator `DD/RR - 1`.
- **Soft kNN-CDF** (`make_knn_cdf_statistic`): the fraction of fixed query
  points whose `k`-th nearest data point lies within `r`, with step
  `sigmoid(s ln(r / r_k))` (`distributed_knn(..., r_grid, sharpness)`). The
  queries are a regular grid (`query_grid`), a volume-filling quadrature with no
  shot noise of its own. A query with fewer than `k` neighbours within `r_max`
  contributes 0. That is a jump of `sigmoid(s ln(r / r_max))` in the
  statistic, so `r_max` defaults to where the jump is below `1e-6` on the whole
  grid.

`concat_statistics` stacks data vectors. The traced single-device ξ with the
bin-aware acceptance (A2/A4a) and the `knn_stats` application (B1) are not on
main yet. Each drops in as one more `Statistic` builder.

The `overflow` flag must be read. Every routine returns it, and the bench raises
on it.

## Gradients through the full chain

`statistic_jacobian` is `jax.jacfwd` of `theta -> positions -> statistic`, with
2 tangents for 2 parameters. Inside, every call rebuilds the partition, the
cell tree, the walk and (on several devices) the halo from the current
positions. Topology carries no cotangent (`docs/distributed_statistics.md`). With
exact acceptance the statistic does not depend on the topology, so this is the
derivative of the statistic itself.

`central_difference_jacobian` re-runs the *whole* chain, tree rebuild included,
at `theta ± h e_i`. A finite difference therefore tests everything the gradient
skips: that rebuilding the topology at a moved point gives the same function.

Reverse mode (`jax.grad` of a scalar loss, `jax.jacrev`) equals forward mode to
round-off.

## Fisher forecast

`F = h J^T C^-1 J`, with these ingredients:

- **`J`** is the mean of the per-realisation autodiff Jacobians over `n_jac`
  realisations. That is an unbiased estimate of `d<S>/dtheta`, by the
  reparameterisation. It is cross-checked against finite differences of the
  same realisations.
- **`C`** is the sample covariance of `n_cov` independent realisations at the
  fiducial point.
- **`h`** is the Hartlap, Simon & Schneider (2007) factor `(n - p - 2)/(n - 1)`,
  which makes `h C^-1` an unbiased precision matrix for Gaussian data.
  Sellentin & Heavens (2016) is a different correction: it replaces the
  Gaussian likelihood by a t-distribution for inference with an estimated
  covariance, and has no Fisher-matrix counterpart. The forecast uses Hartlap.
- The covariance's own parameter dependence (`dC/dtheta`) is neglected, the
  usual approximation.

Ensembles run as `vmap` over the batched draws (`ensemble_statistics`,
`ensemble_jacobians`), or as `lax.map(..., batch_size=b)`, a vmap over batches of
`b`, when memory is the limit. On CPU, `b = 1` is fastest: a vmapped `while_loop`
runs every lane to the slowest lane's trip count.

## Recovery on independent realisations

The old `corrfunc/inference_demo.py` generated its target from the same draws it
fitted with. Its recovery was a fixed point by construction. Here each trial
draws a target realisation `B` at `theta_true` and model draws `A_1..A_m` from
disjoint seeds, then fits `theta` by minimising
`chi^2 = (S_B - mean_i S_{A_i}(theta))^T h C^-1 (...)`.

`fit_parameters` is Levenberg–Marquardt on the autodiff Jacobian. Each step
solves `(J^T W J + lambda diag(J^T W J)) delta = J^T W r`. A step that lowers
`chi^2` is accepted and `lambda` shrinks; otherwise `lambda` grows. The whole
fit is one jitted `lax.scan`.

Both target and model are noisy, so the scatter of the recovered `theta` should
be `sqrt(1 + 1/m)` times the Fisher errors. The bench checks four things:

- the ratio of the measured to the expected standard deviation, with a
  bootstrap error;
- the mean bias in units of the expected sigma;
- the mean of the per-trial `Delta theta^T [(1+1/m) F^-1]^-1 Delta theta`
  (expected 2);
- that every fit ends with `chi^2(theta_hat) <= chi^2(theta_true)`, i.e. it
  reached the minimum.

The σ–β valley is curved, so plain Gauss–Newton zig-zags. The adaptive damping
is what makes `chi^2` monotone.

## Results

`bench/point_process/forecast.py`, CPU (Apple M-series), float64, 1 device. The
paper-default configuration:

- `P = 128` parents × `m = 16` children (`N = 2048`) in `L = 1`;
- fiducial `theta = (sigma, beta) = (0.02, 1.0)`;
- ξ: 8 log bins over `[0.2, 6] sigma`, sharpness 20, `eps = 1e-6`;
- kNN-CDF: `k = 1, 2, 4` on 5 radii over `[0.5, 1.6]` × the mean separation,
  sharpness 15, 12³ grid queries;
- ensembles: `n_cov = 150`, `n_jac = 24`, and 30 recovery trials with `m = 1`.

The command was `python bench/point_process/forecast.py --n-fit 30`. It writes
`results.json` plus four figures; this reference run's output is kept on the paper
branch, under `bench/point_process/results/cpu_n2048/`.

**Validation** (150 realisations, `beta = 1`):

| prediction | `chi^2` / 8 bins (Hartlap-corrected) |
|---|---|
| analytic, fixed counts | 6.1 |
| analytic, Poisson counts | 851 (pulls down to −27σ) |

At `beta = 1.5`, 16 realisations against the Monte Carlo prediction give every
pull within 1.5σ.

**Gradient vs finite differences of the full chain** (one realisation; error is
`max_b |FD − AD| / max_b |AD|`):

| `h / theta` | ξ, ∂σ | ξ, ∂β | kNN, ∂σ | kNN, ∂β |
|---|---|---|---|---|
| 1e-2 | 9.3e-3 | 4.6e-3 | 3.8e-3 | 2.1e-2 |
| 1e-3 | 9.5e-5 | 4.7e-5 | 1.6e-3 | 6.2e-3 |
| 1e-4 | 9.5e-7 | 4.7e-7 | 3.9e-4 | 3.2e-8 |
| 1e-5 | 9.6e-9 | 4.4e-9 | 2.6e-10 | 5.2e-10 |
| 1e-6 | 4.8e-10 | 2.4e-9 | 3.3e-10 | 4.1e-9 |
| 1e-7 | 9.7e-9 | 3.8e-8 | 4.1e-9 | 5.6e-8 |

- **ξ** converges as `h^2`, a factor of 100 per decade, down to 5e-10. Below
  that, round-off (`∝ 1/h`) takes over.
- **kNN-CDF** converges irregularly at large steps. `r_k` is the k-th smallest
  distance: continuous, but with a kink wherever two neighbours swap rank. A step
  that crosses kinks picks up an `O(h)` error from the fraction of queries
  involved. At `h ≤ 1e-5` no swap is crossed and the agreement is 3e-10.
- **Reverse mode** (`jacrev`) equals forward mode to 5e-15.
- **The ensemble-mean Jacobian** (4 realisations) matches finite differences to
  1.6e-8 (ξ) and 1.4e-10 (kNN).

**Fisher forecast** (`n_cov = 150`, `n_jac = 24`, Hartlap):

| data | `p` | Hartlap | `sigma(σ)/σ` | `sigma(β)` | corr(σ, β) |
|---|---|---|---|---|---|
| ξ | 8 | 0.940 | 1.55 % | 0.049 | +0.62 |
| kNN-CDF | 15 | 0.893 | 2.00 % | 0.058 | −0.44 |
| ξ + kNN-CDF | 23 | 0.839 | 1.21 % | 0.032 | +0.28 |

The two statistics degenerate σ and β in opposite directions. In ξ, a larger
cluster (fewer close pairs) can be offset by a cuspier one (more close pairs),
hence the positive correlation; the kNN-CDF's degeneracy runs the other way. So
the combination shrinks the β error by 1.5–1.8× relative to either statistic
alone.

**Recovery on independent realisations** (30 trials, `m = 1`, so the expected
scatter is `sqrt(2)` × the Fisher error; starting point `(1.25 σ, 0.8 β)`):

| data | std / expected, σ | std / expected, β | bias / expected σ (σ, β) | mean pull `chi^2` (exp. 2) | recovered corr (Fisher) |
|---|---|---|---|---|---|
| ξ | 1.09 ± 0.12 | 0.94 ± 0.13 | +0.27, −0.03 | 2.13 | +0.63 (+0.62) |
| kNN-CDF | 0.88 ± 0.10 | 1.01 ± 0.11 | +0.01, +0.06 | 1.70 | −0.50 (−0.44) |
| ξ + kNN-CDF | 1.15 ± 0.12 | 1.00 ± 0.16 | +0.20, −0.08 | 2.42 | +0.13 (+0.28) |

- Each `±` is a bootstrap error.
- The standard error of a mean bias over 30 trials is 0.18σ, so every bias is
  within 1.5 of its error.
- Every fit (90/90) ended with `chi^2(theta_hat) <= chi^2(theta_true)`.
- The mean final `chi^2` is 10.8 / 22.9 / 36.6, against the expected
  `2(p − 2)` = 12 / 26 / 42.

The scatter of the recovered parameters, and its correlation, are what the
Fisher matrix predicts.

**Cost** (CPU, run alongside the test suite, so read the times as upper
bounds):

| step | time |
|---|---|
| forward, ξ + kNN | 1.4 s |
| `jacfwd`, 2 parameters | 1.6 s |
| compile | 20–30 s |
| one 12-step fit | about 10–45 s, depending on the statistic |


## Not covered

- No GPU run yet. The bench takes `--platform gpu --gpu-select free` (autocvd)
  and `--batch-size` for the vmap batches.
- Gradients only with respect to `theta`. The discrete counts (`P`, `m`) are
  conditioned on.
- The covariance is taken at the fiducial point only; `dC/dtheta` is neglected
  in the Fisher matrix.
- Multi-device runs (`--ndev > 1`) go through the same distributed drivers.
  Those are tested in C1/C2, but the point-process tests run on 1 device.
