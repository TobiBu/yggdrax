"""Scaling of the differentiable kNN-CDF vs N.

Times, over a sweep of catalogue size ``N`` (with ``Q = --queries-per-point * N``
volume-filling random queries):

* ``indices`` -- exact ``k_max``-neighbour search, build included (no gradient);
  ``--backend fast`` (default) is :func:`yggdrax.fast_knn.fast_knn`, the GPU
  kNN, with its plan made once per size outside the timed calls;
* ``forward`` -- the full :func:`~yggdrax.applications.knn_stats.knn_cdf`
  (search + distance recomputation + smooth step);
* ``value_and_grad`` -- forward + reverse of a scalar of the CDFs with respect
  to every data coordinate.

All three are jitted and warmed (``--warmup`` calls, then ``--warmup-s``
seconds of repeated calls so an idle GPU reaches its boost clock). The dense ``top_k`` reference
(``brute_force_knn_cdf``, ``O(Q N)`` memory) is timed up to
``--brute-force-max-n``, and each such record carries the search path's error
against it.

Results -> ``results/knn_stats/scaling.json``.

GPU server (``autocvd`` picks a free GPU before JAX starts):

    python bench/knn_stats/scaling.py --sizes 100000 1000000 10000000 \
        --periodic --gpu-select free

Local smoke (CPU):

    JAX_PLATFORMS=cpu conda run -n jaccpot python bench/knn_stats/scaling.py --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.knn_stats._common import (
    device_memory_stats,
    dump_json,
    is_out_of_memory,
    out_of_memory_stop,
    run_metadata,
    select_free_gpu,
    time_callable,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    p.add_argument("--queries-per-point", type=float, default=1.0)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-r", type=int, default=24)
    p.add_argument(
        "--r-scale",
        type=float,
        nargs=2,
        default=[0.1, 3.0],
        help="r grid spans [lo, hi] times the mean separation N^(-1/3)",
    )
    p.add_argument("--sharpness", type=float, default=10.0)
    p.add_argument(
        "--periodic",
        action="store_true",
        help="periodic unit box (minimum-image search and distances)",
    )
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument(
        "--backend", default="fast", choices=("fast", "tree", "tiled", "dense")
    )
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--warmup-s",
        type=float,
        default=1.0,
        help="seconds of untimed calls before each timing (GPU idle clocks)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--brute-force-max-n", type=int, default=10000)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--output", type=str, default="results/knn_stats/scaling.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.sizes = [1000, 4000]
        args.runs = 2
        args.warmup = 1
        args.brute_force_max_n = 4000
        args.warmup_s = 0.0
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="knn-cdf-scaling")

    import jax
    import jax.numpy as jnp

    # yggdrax needs x64 (#97); float32 inputs keep the computation in float32.
    jax.config.update("jax_enable_x64", True)

    from yggdrax.applications.knn_stats import knn_cdf, knn_indices
    from yggdrax.applications.knn_stats.baselines import brute_force_knn_cdf
    from yggdrax.fast_knn import plan_fast_knn

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    ks = tuple(args.ks)
    box = 1.0 if args.periodic else None
    kw = dict(backend=args.backend, leaf_size=args.leaf_size, box_size=box)
    sharp = args.sharpness

    def _timed(fn):
        """``time_callable`` after ``--warmup-s`` seconds of untimed calls."""
        jax.block_until_ready(fn())  # compile outside the clock warm-up
        t_end = time.perf_counter() + args.warmup_s
        while time.perf_counter() < t_end:
            jax.block_until_ready(fn())
        return time_callable(fn, warmup=args.warmup, runs=args.runs)

    def _measure(n: int) -> dict:
        """Measure one size; raises if it does not fit in memory."""
        q = max(1, int(round(args.queries_per_point * n)))
        k_d, k_q = jax.random.split(jax.random.PRNGKey(args.seed))
        data = jax.random.uniform(k_d, (n, 3), dtype=dtype)
        queries = jax.random.uniform(k_q, (q, 3), dtype=dtype)
        sep = n ** (-1.0 / 3.0)
        r_grid = jnp.geomspace(
            args.r_scale[0] * sep, args.r_scale[1] * sep, args.num_r
        ).astype(dtype)
        weights = jnp.arange(1, len(ks) * args.num_r + 1, dtype=dtype).reshape(
            len(ks), args.num_r
        )

        kw_n = dict(kw)
        plan_info = None
        if args.backend == "fast":
            # static bucket counts, sized once from these points (eager, untimed)
            t0 = time.perf_counter()
            plan = plan_fast_knn(
                data.astype(jnp.float32),
                queries.astype(jnp.float32),
                k=max(ks),
                leaf_size=args.leaf_size,
                box_size=box,
            )
            plan_info = {
                "plan_s": time.perf_counter() - t0,
                "bucket_capacity": plan.bucket_capacity,
                "query_capacity": plan.query_capacity,
            }
            kw_n["plan"] = plan
        indices = jax.jit(lambda d, qq: knn_indices(d, qq, max(ks), **kw_n))
        forward = jax.jit(
            lambda d, qq: knn_cdf(d, qq, ks, r_grid, sharpness=sharp, **kw_n)
        )
        vgrad = jax.jit(
            jax.value_and_grad(
                lambda d, qq: jnp.sum(
                    weights * knn_cdf(d, qq, ks, r_grid, sharpness=sharp, **kw_n)
                )
            )
        )
        # a fast-search overflow poisons the CDFs with NaN: refuse to time it
        if not bool(jnp.all(jnp.isfinite(forward(data, queries)))):
            raise RuntimeError(f"n={n}: non-finite kNN-CDF (search overflow?)")
        idx_t = _timed(lambda: indices(data, queries))
        fwd_t = _timed(lambda: forward(data, queries))
        vg_t = _timed(lambda: vgrad(data, queries))
        entry = {
            "n": n,
            "num_queries": q,
            "fast_plan": plan_info,
            "indices": idx_t.as_dict(),
            "forward": fwd_t.as_dict(),
            "value_and_grad": vg_t.as_dict(),
            "grad_ratio": vg_t.min_s / fwd_t.min_s,
            "device_memory": device_memory_stats(),
        }
        if n <= args.brute_force_max_n:
            brute = jax.jit(
                lambda d, qq: brute_force_knn_cdf(
                    d, qq, ks, r_grid, sharpness=sharp, box_size=box
                )
            )
            brute_vg = jax.jit(
                jax.value_and_grad(
                    lambda d, qq: jnp.sum(
                        weights
                        * brute_force_knn_cdf(
                            d, qq, ks, r_grid, sharpness=sharp, box_size=box
                        )
                    )
                )
            )
            entry["brute_force_forward"] = _timed(
                lambda: brute(data, queries)
            ).as_dict()
            entry["brute_force_value_and_grad"] = _timed(
                lambda: brute_vg(data, queries)
            ).as_dict()
            _, g_ref = brute_vg(data, queries)
            _, g = vgrad(data, queries)
            entry["max_abs_cdf_error_vs_brute_force"] = float(
                jnp.max(jnp.abs(forward(data, queries) - brute(data, queries)))
            )
            entry["max_rel_grad_error_vs_brute_force"] = float(
                jnp.max(jnp.abs(g - g_ref)) / jnp.max(jnp.abs(g_ref))
            )
        print(
            f"n={n:>9d} q={q:>9d} indices={idx_t.min_s * 1e3:9.2f} ms "
            f"fwd={fwd_t.min_s * 1e3:9.2f} ms vgrad={vg_t.min_s * 1e3:9.2f} ms"
        )
        return entry

    records = []
    stopped_early = None
    for n in args.sizes:
        try:
            records.append(_measure(n))
        except Exception as exc:
            if not is_out_of_memory(exc):
                raise
            stopped_early = out_of_memory_stop(n, exc)
            break

    payload = {
        "benchmark": "knn_cdf_scaling",
        "params": {
            "sizes": args.sizes,
            "queries_per_point": args.queries_per_point,
            "ks": list(ks),
            "num_r": args.num_r,
            "r_scale": args.r_scale,
            "sharpness": sharp,
            "leaf_size": args.leaf_size,
            "box_size": box,
            "backend": args.backend,
            "warmup_s": args.warmup_s,
            "dtype": args.dtype,
            "brute_force_max_n": args.brute_force_max_n,
            "brute_force": "brute_force_knn_cdf (dense top_k), jitted and warmed",
            "runs": args.runs,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "metadata": run_metadata(),
        "sweep": {"stopped_early": stopped_early},
        "records": records,
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
