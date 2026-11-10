"""Soft pair counts as kernel sums: yggdrax against KeOps and dense JAX (paper v2).

The soft pair counts behind the paper's ξ are kernel sums,
``C_k = Σ_{i<j} w_k(r_ij)`` with the logistic log-r bin window ``w_k`` of
:func:`yggdrax.applications.corrfunc.binning.soft_bin_weights`. A referee's
first question is why not evaluate them exactly by brute force on the GPU with
KeOps (Charlier et al. 2021), the standard tool for O(N²) kernel sums. This
bench answers it, forward and value+grad of ``L = Σ_k c_k C_k``:

* ``yggdrax``: :func:`~yggdrax.applications.corrfunc.traced.traced_soft_pair_counts`
  (static cell tree + mutual walk, bin-aware window rule at ``--eps``; the tree
  is rebuilt inside the jitted, differentiated function);
* ``keops``: ``pykeops.numpy`` LazyTensor reductions, exact O(N²). The forward
  is one ``(N, K+1)`` reduction of the logistic gates; the gradient is the
  hand-written closed form ``∂L/∂x_i = Σ_j (s / r_ij²) Σ_m a_m g_m (1 - g_m)
  (x_i - x_j)``, since the numpy binding has no autodiff (the torch one would
  add a second CUDA stack to the environment). Host arrays in and out, so its
  times include two small transfers;
* ``dense``: the row-blocked, checkpointed pure-JAX brute force of the A5
  benches (``_a5_common.brute_counts``), autodiff for the gradient.

The radial range is in units of the mean interparticle spacing
``N^(-1/3) L`` by default (``--length-unit spacing``), so the tree's work per
particle is fixed along the sweep while the brute forces grow as N².

Exactness is checked before timing, at every size that has an exact reference
(``dense`` up to ``--dense-max-n``, else ``keops``): counts and gradient of each
arm against the reference, with ``--exact-rtol`` for the exact arms and
``--tree-rtol`` for the tree (the window rule's error is ≤ eps per pair). An
arm that fails is not timed at that size. An arm runs up to its ``--*-max-n``,
or until it runs out of memory or its median call exceeds ``--time-budget-s``.

Examples::

    python bench/baselines/kernel_sums.py --smoke --gpu-select none

    python bench/baselines/kernel_sums.py --gpu-select free \\
        --sizes 10000 30000 100000 300000 1000000 \\
        --output results/baselines/kernel_sums.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.differentiability._common import (  # noqa: E402
    dump_json,
    is_out_of_memory,
    run_metadata,
    select_free_gpu,
    time_callable,
)

ARMS = ("yggdrax", "keops", "dense")
MODES = ("forward", "value_and_grad")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 30_000])
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--ic", choices=("uniform", "clustered"), default="uniform")
    p.add_argument("--length-unit", choices=("spacing", "box"), default="spacing")
    p.add_argument("--r-min", type=float, default=0.25)
    p.add_argument("--r-max", type=float, default=4.0)
    p.add_argument("--nbins", type=int, default=10)
    p.add_argument("--sharpness", type=float, default=40.0, help="in ln r")
    p.add_argument("--eps", type=float, default=1e-6, help="window_eps (yggdrax)")
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument("--dense-max-n", type=int, default=65_536)
    p.add_argument("--keops-max-n", type=int, default=2_000_000)
    p.add_argument("--exact-rtol", type=float, default=1e-9)
    p.add_argument("--tree-rtol", type=float, default=1e-4)
    p.add_argument("--time-budget-s", type=float, default=120.0)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--output", default="results/baselines/kernel_sums.json")
    p.add_argument("--smoke", action="store_true", help="tiny sizes, 1 timed run")
    args = p.parse_args(argv)
    if args.smoke:
        args.sizes, args.runs, args.warmup = [1000, 3000], 1, 1
    return args


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _catalogue(n: int, ic: str, seed: int):
    """Positions in the unit cube (float64 numpy)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    if ic == "uniform":
        return rng.uniform(0.0, 1.0, (n, 3))
    centers = rng.uniform(0.15, 0.85, (max(4, n // 500), 3))
    blob = centers[rng.integers(0, len(centers), n)] + 0.02 * rng.normal(size=(n, 3))
    back = rng.uniform(0.0, 1.0, (n, 3))
    take = (np.arange(n) % 2 == 0)[:, None]
    return np.clip(np.where(take, blob, back), 0.0, 1.0)


def _keops_fns(
    log_edges: Any, sharpness: float, coeff: Any
) -> tuple[Callable, Callable]:
    """KeOps forward ``x -> C`` and value+grad ``x -> (L, dL/dx)`` (host numpy).

    Parameters
    ----------
    log_edges
        ``ln`` of the ``K + 1`` bin edges.
    sharpness
        Logistic steepness ``s`` in ``ln r``.
    coeff
        Loss weights ``c_k``, length ``K``.

    Returns
    -------
    tuple
        ``(counts_fn, value_and_grad_fn)``.
    """
    import numpy as np
    from pykeops.numpy import LazyTensor

    e = np.ascontiguousarray(log_edges, dtype=np.float64)
    # Σ_k c_k (g'(z_k) - g'(z_{k+1})) = Σ_m a_m g'(z_m)
    a = np.zeros(len(e))
    a[:-1] += coeff
    a[1:] -= coeff
    s = float(sharpness)

    def gates(x):
        x_i = LazyTensor(x[:, None, :])
        x_j = LazyTensor(x[None, :, :])
        d = x_i - x_j
        r2 = (d**2).sum(-1) + 1e-30
        z = s * (0.5 * r2.log() - LazyTensor(e))  # (.., K+1) by broadcasting
        return d, r2, 1.0 / (1.0 + (-z).exp())

    def counts(x):
        _, _, g = gates(x)
        tot = g.sum(axis=1).sum(axis=0)  # ordered pairs, self ≈ 0 (r = 1e-15)
        return 0.5 * (tot[:-1] - tot[1:])

    def value_and_grad(x):
        d, r2, g = gates(x)
        tot = g.sum(axis=1).sum(axis=0)
        value = float(np.dot(coeff, 0.5 * (tot[:-1] - tot[1:])))
        dl_dr_over_r = (LazyTensor(a) * g * (1.0 - g)).sum(-1) * s / r2
        grad = (dl_dr_over_r * d).sum(axis=1)
        return value, grad

    return counts, value_and_grad


def main(argv=None) -> int:
    args = _parse_args(argv)
    select_free_gpu(args.gpu_select, tag="kernel-sums")

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    from bench.differentiability import _a5_common as A5
    from yggdrax.applications.corrfunc.binning import make_log_edges
    from yggdrax.applications.corrfunc.traced import (
        plan_traced_pair_counts,
        traced_soft_pair_counts,
    )

    coeff = 1.0 / np.arange(1, args.nbins + 1, dtype=np.float64)
    arm_info: dict[str, Any] = {}
    keops_ok = "keops" in args.arms
    if keops_ok:
        try:
            import pykeops  # noqa: F401

            arm_info["keops"] = {
                "available": True,
                "version": _version("pykeops"),
                "note": "pykeops.numpy; closed-form gradient; host arrays",
            }
        except Exception as exc:  # pragma: no cover - env-dependent
            keops_ok = False
            arm_info["keops"] = {"available": False, "reason": f"{exc}"}
            print(f"[kernel-sums] skip keops: {exc}")
    if "yggdrax" in args.arms:
        arm_info["yggdrax"] = {"available": True, "version": _version("yggdrax")}
    if "dense" in args.arms:
        arm_info["dense"] = {"available": True, "note": "_a5_common.brute_counts"}

    payload: dict[str, Any] = {
        "metadata": run_metadata(
            {
                "bench": "baselines/kernel_sums",
                "walk": "traced_soft_pair_counts: static cell tree + flat mutual "
                "walk, window acceptance",
            }
        ),
        "params": vars(args),
        "loss_coefficients": coeff.tolist(),
        "arms": arm_info,
        "results": [],
        "sweep": {"stopped_early": {}},
    }
    stopped: dict[str, Any] = payload["sweep"]["stopped_early"]
    cj = jnp.asarray(coeff)

    for n in args.sizes:
        unit = n ** (-1.0 / 3.0) if args.length_unit == "spacing" else 1.0
        r_min, r_max = args.r_min * unit, args.r_max * unit
        edges = make_log_edges(r_min, r_max, args.nbins)
        bounds = (jnp.full(3, -0.05), jnp.full(3, 1.05))
        x_np = _catalogue(n, args.ic, args.seed)
        x = jnp.asarray(x_np)

        fns: dict[str, dict[str, Callable]] = {}
        if "yggdrax" in args.arms and "yggdrax" not in stopped:
            cfg = plan_traced_pair_counts(
                x,
                edges=edges,
                sharpness=args.sharpness,
                acceptance="window",
                window_eps=args.eps,
                leaf_size=args.leaf_size,
                bounds=bounds,
            )

            def ygg_counts(p, _cfg=cfg):
                out = traced_soft_pair_counts(
                    p, edges, _cfg, sharpness=args.sharpness, bounds=bounds
                )
                return out.counts, out.overflow

            def ygg_loss(p):
                c, of = ygg_counts(p)
                return jnp.dot(cj, c), of

            fwd = jax.jit(ygg_counts)
            vg = jax.jit(jax.value_and_grad(ygg_loss, has_aux=True))
            fns["yggdrax"] = {
                "forward": lambda: fwd(x),
                "value_and_grad": lambda: vg(x),
                "counts": lambda: np.asarray(fwd(x)[0]),
                "grad": lambda: np.asarray(vg(x)[1]),
                "overflow": lambda: bool(fwd(x)[1]),
            }
        if "dense" in args.arms and n <= args.dense_max_n and "dense" not in stopped:

            def d_counts(p):
                return A5.brute_counts(p, edges, args.sharpness)

            dfwd = jax.jit(d_counts)
            dvg = jax.jit(jax.value_and_grad(lambda p: jnp.dot(cj, d_counts(p))))
            fns["dense"] = {
                "forward": lambda: dfwd(x),
                "value_and_grad": lambda: dvg(x),
                "counts": lambda: np.asarray(dfwd(x)),
                "grad": lambda: np.asarray(dvg(x)[1]),
            }
        if keops_ok and n <= args.keops_max_n and "keops" not in stopped:
            k_counts, k_vg = _keops_fns(
                np.log(np.asarray(edges)), args.sharpness, coeff
            )
            fns["keops"] = {
                "forward": lambda: k_counts(x_np),
                "value_and_grad": lambda: k_vg(x_np),
                "counts": lambda: np.asarray(k_counts(x_np)),
                "grad": lambda: np.asarray(k_vg(x_np)[1]),
            }

        # -- exactness before timing ------------------------------------------------
        ref_name = "dense" if "dense" in fns else ("keops" if "keops" in fns else None)
        ref: dict[str, Any] = {}
        checks: dict[str, Any] = {}
        for name, f in list(fns.items()):
            try:
                c = f["counts"]()
                g = f["grad"]() if "value_and_grad" in args.modes else None
            except Exception as exc:
                if name == "keops" and not is_out_of_memory(exc):
                    # first KeOps call compiles; a missing compiler/CUDA lands here
                    arm_info["keops"] = {"available": False, "reason": repr(exc)[:400]}
                    keops_ok = False
                    print(f"[kernel-sums] keops unusable: {exc!r}"[:400])
                    fns.pop(name)
                    continue
                if not is_out_of_memory(exc):
                    raise
                stopped[name] = {"reason": "out_of_memory", "at_n": n}
                fns.pop(name)
                continue
            if name == ref_name:
                ref = {"counts": c, "grad": g}
            checks[name] = {"counts": c, "grad": g}
        if ref_name is not None and ref_name not in checks:
            ref_name = next(
                (a for a in ("dense", "keops") if a in checks), None
            )  # the preferred reference failed: fall back
            ref = dict(checks[ref_name]) if ref_name else {}

        for name, f in fns.items():
            tol = args.tree_rtol if name == "yggdrax" else args.exact_rtol
            agree: dict[str, Any] = {"reference": ref_name, "rtol": tol}
            if ref_name is None or name not in checks:
                agree["ok"] = None
                agree["note"] = "no exact reference at this N"
            else:
                c, g = checks[name]["counts"], checks[name]["grad"]
                rc = float(np.max(np.abs(c - ref["counts"]) / np.abs(ref["counts"])))
                agree["counts_max_rel_error"] = rc
                ok = rc <= tol
                if g is not None and ref["grad"] is not None:
                    rg = float(
                        np.linalg.norm(g - ref["grad"]) / np.linalg.norm(ref["grad"])
                    )
                    agree["grad_rel_l2_error"] = rg
                    ok = ok and rg <= tol
                if name == "yggdrax":
                    agree["overflow"] = f["overflow"]()
                    ok = ok and not agree["overflow"]
                agree["ok"] = bool(ok)
            for mode in args.modes:
                row: dict[str, Any] = {
                    "n": n,
                    "arm": name,
                    "mode": mode,
                    "r_range": [r_min, r_max],
                    "agreement": agree,
                }
                if agree["ok"] is False:
                    row["timing"] = None
                    row["not_timed"] = "disagrees with the exact reference"
                else:
                    try:
                        t0 = time.perf_counter()
                        jax.block_until_ready(f[mode]())
                        row["first_call_s"] = time.perf_counter() - t0
                        if row["first_call_s"] > 2 * args.time_budget_s:
                            # far over budget in one call: an upper bound, no repeats
                            row["timing"] = None
                            row["not_timed"] = "first call over 2x --time-budget-s"
                            stopped[name] = {"reason": "time_budget", "at_n": n}
                            payload["results"].append(row)
                            continue
                        t = time_callable(f[mode], warmup=args.warmup, runs=args.runs)
                        row["timing"] = t.as_dict()
                        print(
                            f"[kernel-sums] n={n} {name} {mode}: "
                            f"median {t.median_s * 1e3:.1f} ms  ok={agree['ok']}"
                        )
                        if t.median_s > args.time_budget_s:
                            stopped[name] = {"reason": "time_budget", "after_n": n}
                    except Exception as exc:
                        if not is_out_of_memory(exc):
                            raise
                        stopped[name] = {"reason": "out_of_memory", "at_n": n}
                        row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                payload["results"].append(row)
        dump_json(payload, args.output)

    bad = [r for r in payload["results"] if r["agreement"].get("ok") is False]
    if bad:
        print(f"[kernel-sums] {len(bad)} row(s) disagree with the exact reference")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
