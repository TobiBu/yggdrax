"""Autodiff overhead: forward-only vs forward+backward wall-clock ratio.

Measures the cost of reverse-mode differentiation through a representative
KD-tree pipeline (build tree -> k-NN query -> smooth reduction), which is the
backend both paper case studies use. Reports the ratio of forward+backward to
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
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.differentiability._common import (
    dump_json,
    run_metadata,
    select_free_gpu,
    time_callable,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    p.add_argument("--k", type=int, default=8, help="Neighbours per query point.")
    p.add_argument("--leaf-size", type=int, default=64)
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

    select_free_gpu(args.gpu_select, tag="autodiff")

    import jax
    import jax.numpy as jnp

    from yggdrax import build_kdtree, query_neighbors

    k = args.k
    leaf_size = args.leaf_size

    def loss(points):
        # Rebuild the tree inside the traced function (the SVGD-style pattern),
        # query k nearest neighbours, and reduce their distances smoothly.
        tree = build_kdtree(points, leaf_size=leaf_size)
        _, distances = query_neighbors(
            tree, points, k=k, exclude_self=True, backend="tiled"
        )
        return jnp.mean(distances)

    fwd = jax.jit(loss)
    fwd_bwd = jax.jit(jax.value_and_grad(loss))

    records = []
    for n in args.sizes:
        key = jax.random.PRNGKey(args.seed)
        points = jax.random.uniform(
            key, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
        )

        fwd_timing = time_callable(
            lambda: fwd(points), warmup=args.warmup, runs=args.runs
        )
        bwd_timing = time_callable(
            lambda: fwd_bwd(points), warmup=args.warmup, runs=args.runs
        )
        ratio = bwd_timing.min_s / fwd_timing.min_s

        records.append(
            {
                "n": n,
                "forward": fwd_timing.as_dict(),
                "forward_backward": bwd_timing.as_dict(),
                "overhead_ratio": ratio,
            }
        )
        print(
            f"n={n:>9d}  fwd={fwd_timing.min_s * 1e3:8.2f} ms  "
            f"fwd+bwd={bwd_timing.min_s * 1e3:8.2f} ms  ratio={ratio:5.2f}x"
        )

    payload = {
        "benchmark": "autodiff_overhead",
        "params": {
            "sizes": args.sizes,
            "k": args.k,
            "leaf_size": args.leaf_size,
            "runs": args.runs,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "metadata": run_metadata(),
        "records": records,
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
