# B4 optimisers: Adam vs L-BFGS (CPU, N = 3e4)

These are the runs behind the B4 default (L-BFGS with a 0.25 l step cap). The
target, start, randoms and seed are those of `../optimise_cpu.json`, the Adam
run: N = 30000, a Thomas target (16 children, σ = 0.3 l), and a uniform start.
The code is `claude/b4-positions` 2dd8de9. The JSONs were written by the same
code before it was committed. M-series CPU, jax 0.8.3, x64.

Up to five runs shared the CPU, so their own wall times are inflated (2–5×).
Cost is therefore counted in **loss-and-gradient evaluations**, which every
step records, line-search trials included. `compare.py` also converts that
count to modelled CPU minutes, using the clean timings of the Adam run: 2.59 s
per evaluation and 2.75 s per kNN search (one per refresh). The one run that
had the CPU alone for most of its course, C25, measured 4.1 s per step at 1.31
evaluations per step.

| run | optimiser | flags |
|---|---|---|
| adam | Adam, 0.05 l, cosine to 0.1× over 800 steps | (`../optimise_cpu.json`) |
| P10, P5, P1 | `optax.lbfgs()` as is, rebuild every 10/5/1 | `--lbfgs-plain` |
| C25, C10 | L-BFGS, step cap 0.25 l / 0.1 l | `--lbfgs-max-step` |
| C10R | C10, memory cleared at every refresh | `--lbfgs-reset` |
| H, HR | Adam to step 450 (where the stale fraction reaches 0), then C10 / C10R | `--optimizer hybrid` |

`NAME~` in `table.txt` and `compare.png` counts a plain run as if it had
reused the line search's value and gradient, as the capped runs do. The
trajectory is the same (`test_reusing_the_line_search_value_only_saves_evaluations`);
only the count drops by one on every step except an epoch's first. `at800`
is the run's last value when it stopped short of 800 evaluations.

Findings:

- **All L-BFGS variants beat Adam per evaluation.** Up to about 400
  evaluations, while the kNN candidate lists still go stale, every method
  follows roughly the same 1e-2 → 1e-3 path. The L-BFGS runs that keep their
  memory are smoother and 2–2.5× lower at 200 evaluations. After that L-BFGS keeps descending at a constant log-rate. Adam's
  cosine-decayed step levels off at 7e-4.
- **Final loss against the independent realisation (1.6e-4):** C25 1.1e-5
  (653 evaluations), H 5.3e-6 (872), C10 8.9e-6 (1102), P10~ 3.2e-5 (514),
  Adam 7.1e-4 (800).
- **ξ outermost bin (target +0.007):** Adam −0.10; C25 +0.010; H +0.008.
  The mismatch closes.
- **Max |hard kNN-CDF − target|:** Adam 0.034; C25 0.014; H 0.011. The
  independent realisation gives 0.009.
- **Cap.** Uncapped L-BFGS is the fastest per evaluation early: 1.1e-4 at 400
  evaluations, against 2.2e-4 for C25. But it moves particles up to 4.9 l per
  epoch, and up to 31% of candidate lists go stale. A 0.25 l cap bounds the
  moves to 1.2 l at nearly the same rate. A 0.1 l cap is slower, and stalls in
  the line search for about 150 evaluations near 3.6e-5. Scaling the whole
  update by its largest row, instead of clipping each particle, stalls at once.
- **Memory reset at every refresh** is worse: 3–9× higher loss at a matched
  count. Curvature pairs that cross a refresh are harmless.
- **Rebuild cadence:** refreshing every step (P1) or every 5 steps (P5)
  converges no better per evaluation than every 10.
- **Overflow.** No run overflowed its traced capacities: 0 re-plans.
