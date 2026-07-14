"""Scaling benchmark: tree build + dual-tree traversal time vs N.

Measures wall-clock build time and interaction/neighbor traversal time for the
radix, octree, and KD-tree backends over a sweep of particle counts. Includes a
CPU ``scipy.spatial.cKDTree`` build+query baseline for context (not a fair
full comparison -- it does neither the dual-tree far-field nor the GPU work --
but it anchors "why GPU").

Results are written to ``results/differentiability/scaling.json``; no plotting
happens here (see ``examples/differentiable_paper/fig_scaling.ipynb``).

Production (GPU server, paper branch):

    micromamba run -n odisseo python bench/differentiability/scaling.py \
        --sizes 1000 10000 100000 1000000 --gpu-select free

Local smoke (CPU):

    conda run -n jaccpot python bench/differentiability/scaling.py --smoke
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

_BACKENDS = ("radix", "octree", "kdtree")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    p.add_argument("--backends", type=str, nargs="+", default=list(_BACKENDS))
    p.add_argument("--leaf-size", type=int, default=64)
    p.add_argument("--theta", type=float, default=0.6)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--no-scipy", action="store_true", help="Skip cKDTree baseline.")
    p.add_argument(
        "--output", type=str, default="results/differentiability/scaling.json"
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny single-size CPU run for end-to-end validation.",
    )
    return p.parse_args()


def _scipy_baseline(positions_np, leaf_size: int, runs: int):
    """Build+query time for scipy.spatial.cKDTree (CPU reference)."""
    import time

    import numpy as np
    from scipy.spatial import cKDTree

    build_times, query_times = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        tree = cKDTree(positions_np, leafsize=leaf_size)
        build_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        tree.query(positions_np, k=8)
        query_times.append(time.perf_counter() - t0)
    return {
        "build_min_s": float(np.min(build_times)),
        "query_min_s": float(np.min(query_times)),
    }


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.sizes = [2000]
        args.runs = 2
        args.warmup = 1
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="scaling")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from yggdrax import (
        Tree,
        build_interactions_and_neighbors,
        compute_tree_geometry,
    )

    def make_problem(n: int, seed: int):
        key = jax.random.PRNGKey(seed)
        kp, km = jax.random.split(key)
        pos = jax.random.uniform(kp, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32)
        mass = jax.random.uniform(km, (n,), minval=0.5, maxval=1.5, dtype=jnp.float32)
        return pos, mass

    records = []
    for n in args.sizes:
        pos, mass = make_problem(n, args.seed)
        pos_np = np.asarray(pos)
        entry: dict = {"n": n, "backends": {}}

        for backend in args.backends:

            def build():
                return Tree.from_particles(
                    pos,
                    mass,
                    tree_type=backend,
                    build_mode="adaptive",
                    leaf_size=args.leaf_size,
                    return_reordered=True,
                )

            build_timing = time_callable(build, warmup=args.warmup, runs=args.runs)
            tree = build()
            geometry = compute_tree_geometry(tree, tree.positions_sorted)

            def traverse():
                return build_interactions_and_neighbors(
                    tree,
                    geometry,
                    theta=args.theta,
                    mac_type="dehnen",
                )

            trav_timing = time_callable(traverse, warmup=args.warmup, runs=args.runs)
            interactions, neighbors = traverse()

            entry["backends"][backend] = {
                "build": build_timing.as_dict(),
                "traverse": trav_timing.as_dict(),
                "num_nodes": int(tree.num_nodes),
                "num_far_interactions": int(interactions.sources.shape[0]),
                "num_near_neighbors": int(neighbors.offsets[-1]),
            }
            print(
                f"n={n:>9d} {backend:7s} "
                f"build={build_timing.min_s * 1e3:8.2f} ms  "
                f"traverse={trav_timing.min_s * 1e3:8.2f} ms"
            )

        if not args.no_scipy:
            entry["scipy_ckdtree"] = _scipy_baseline(pos_np, args.leaf_size, args.runs)

        records.append(entry)

    payload = {
        "benchmark": "scaling",
        "params": {
            "sizes": args.sizes,
            "backends": args.backends,
            "leaf_size": args.leaf_size,
            "theta": args.theta,
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
