"""Autodiff overhead: forward-only vs forward+backward wall-clock ratio.

Measures the cost of reverse-mode differentiation through a representative
kNN pipeline (build -> k-NN query -> smooth reduction), which is the
neighbour search both paper case studies use. The default query backend is
``fast``: :func:`yggdrax.fast_knn.fast_knn` with ``differentiable=True``
(indices from the frozen positions, distances recomputed from the live ones;
its plan is made once per size, outside the timed calls). Reports the ratio of forward+backward to
forward-only time over a sweep of N; a well-behaved differentiable primitive
keeps this ratio modest (typically 2-4x).

Results -> ``results/differentiability/autodiff_overhead.json``.

Production (GPU server):

    micromamba run -n odisseo python bench/differentiability/autodiff_overhead.py \
        --sizes 1000 10000 100000 1000000 --gpu-select free

Local smoke (CPU):

    conda run -n jaccpot python bench/differentiability/autodiff_overhead.py --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.differentiability._common import (
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
    p.add_argument("--k", type=int, default=8, help="Neighbours per query point.")
    p.add_argument(
        "--query-backend",
        choices=("fast", "dense", "tiled", "tree", "auto"),
        default="fast",
        help=(
            "kNN query backend. Default 'fast' (yggdrax.fast_knn, the GPU kNN); "
            "'tree' is the KD-tree walk of query_neighbors; 'tiled' materialises "
            "a dense pairwise reverse-mode residual and OOMs on GPU for N>=1e5."
        ),
    )
    p.add_argument(
        "--leaf-size",
        type=int,
        default=None,
        help="bucket size (default: 32 for fast, 64 for the KD-tree backends)",
    )
    p.add_argument(
        "--warmup-s",
        type=float,
        default=1.0,
        help="seconds of untimed calls before each timing (GPU idle clocks)",
    )
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument(
        "--output", type=str, default="results/differentiability/autodiff_overhead.json"
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.sizes = [2000]
        args.runs = 2
        args.warmup = 1
        args.gpu_select = "none"
        args.warmup_s = 0.0

    select_free_gpu(args.gpu_select, tag="autodiff")

    import jax
    import jax.numpy as jnp

    # yggdrax needs x64 and no longer switches it on at import (#97). The
    # points below stay float32, as they were when the import switched it on.
    jax.config.update("jax_enable_x64", True)

    from yggdrax import build_kdtree, query_neighbors
    from yggdrax.fast_knn import fast_knn, plan_fast_knn

    k = args.k
    query_backend = args.query_backend
    leaf_size = args.leaf_size or (32 if query_backend == "fast" else 64)

    def make_loss(plan):
        def loss(points):
            # Rebuild the tree inside the traced function (the SVGD-style
            # pattern), query k nearest neighbours, reduce their distances.
            if query_backend == "fast":
                res = fast_knn(points, None, plan=plan, differentiable=True)
                return jnp.where(res.overflow, jnp.nan, jnp.mean(res.distances))
            tree = build_kdtree(points, leaf_size=leaf_size)
            _, distances = query_neighbors(
                tree, points, k=k, exclude_self=True, backend=query_backend
            )
            return jnp.mean(distances)

        return loss

    def _timed(fn):
        """``time_callable`` after ``--warmup-s`` seconds of untimed calls."""
        jax.block_until_ready(fn())  # compile outside the clock warm-up
        t_end = time.perf_counter() + args.warmup_s
        while time.perf_counter() < t_end:
            jax.block_until_ready(fn())
        return time_callable(fn, warmup=args.warmup, runs=args.runs)

    def _measure(n: int) -> dict:
        """Measure one size; raises if it does not fit in memory."""
        key = jax.random.PRNGKey(args.seed)
        points = jax.random.uniform(
            key, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
        )
        plan = None
        if query_backend == "fast":
            plan = plan_fast_knn(points, None, k=k, leaf_size=leaf_size)
        loss = make_loss(plan)
        fwd = jax.jit(loss)
        fwd_bwd = jax.jit(jax.value_and_grad(loss))
        if not bool(jnp.isfinite(fwd(points))):
            raise RuntimeError(f"n={n}: non-finite loss (search overflow?)")

        fwd_timing = _timed(lambda: fwd(points))
        bwd_timing = _timed(lambda: fwd_bwd(points))
        ratio = bwd_timing.min_s / fwd_timing.min_s
        print(
            f"n={n:>9d}  fwd={fwd_timing.min_s * 1e3:8.2f} ms  "
            f"fwd+bwd={bwd_timing.min_s * 1e3:8.2f} ms  ratio={ratio:5.2f}x"
        )
        return {
            "n": n,
            "forward": fwd_timing.as_dict(),
            "forward_backward": bwd_timing.as_dict(),
            "overhead_ratio": ratio,
        }

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
        "benchmark": "autodiff_overhead",
        "params": {
            "sizes": args.sizes,
            "k": args.k,
            "query_backend": args.query_backend,
            "leaf_size": leaf_size,
            "warmup_s": args.warmup_s,
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
