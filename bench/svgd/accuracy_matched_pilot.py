"""Pilot for the accuracy-matched sweep: Stein-update error vs (N, theta), no timing.

``accuracy_matched.py`` needs a theta grid that brackets a relative error of
1e-3 at every N of the timed sweep. Error is hardware-independent here, so this
pilot measures it directly -- on CPU, at the sweep's own N -- instead of
extrapolating from small N. It reproduces ``scaling.py``'s problem exactly
(``--density fixed-cloud``, the same seed, sigma, bandwidth, leaf size, backend
and traversal capacities) and times nothing: the exact update is evaluated once
per N, since it does not depend on theta, and the tree update once per
(N, theta).

Each record also carries the partition's size counters, so the cost of a small
theta is visible without a timing.

Results -> ``results/svgd/accuracy_matched/pilot_cpu.json``, rewritten after
every N so an interrupted run keeps what it measured.

    conda run -n edda-tri python bench/svgd/accuracy_matched_pilot.py \
        --sizes 1000 2000 5000 10000 20000 50000 100000 \
        --thetas 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 --gpu-select none
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.differentiability._common import dump_json, run_metadata, select_free_gpu


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    p.add_argument("--thetas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument("--backend", type=str, default="auto")
    p.add_argument("--sigma", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exact-block", type=int, default=2048)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="none"
    )
    p.add_argument(
        "--output", type=str, default="results/svgd/accuracy_matched/pilot_cpu.json"
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.sizes = [2000]
        args.thetas = [0.3, 0.5]
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="svgd-accuracy-pilot")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from yggdrax import DualTreeTraversalConfig
    from yggdrax.applications.svgd.exact import exact_phi
    from yggdrax.applications.svgd.sampler import (
        assemble_svgd_topology,
        build_svgd_traversal,
        svgd_phi_from_topology,
    )

    # Identical to scaling.py, so a pilot record and a timed record describe the
    # same partition of the same cloud.
    cfg = DualTreeTraversalConfig(
        max_pair_queue=1 << 22,
        process_block=64,
        max_interactions_per_node=1 << 15,
        max_neighbors_per_leaf=1 << 15,
    )
    dim = args.dim
    backend = args.backend
    if backend == "auto":
        backend = "radix" if dim <= 3 else "leaf_kdtree"
    h = 0.5

    phi = jax.jit(
        lambda pp, sc, hh, t: svgd_phi_from_topology(pp, sc, hh, t, accumulate="auto")
    )
    exact = jax.jit(
        lambda pp, sc, hh: exact_phi(pp, sc, hh, block_size=args.exact_block)
    )

    params = {
        "sizes": args.sizes,
        "thetas": args.thetas,
        "dim": dim,
        "leaf_size": args.leaf_size,
        "backend": backend,
        "dtype": "float64",
        "density": "fixed-cloud",
        "sigma": args.sigma,
        "h": h,
        "seed": args.seed,
        "exact_block": args.exact_block,
    }
    records = []
    for n in args.sizes:
        key = jax.random.PRNGKey(args.seed)
        p = (jax.random.normal(key, (n, dim), dtype=jnp.float64) * args.sigma).astype(
            jnp.float64
        )
        scores = -p
        t0 = time.perf_counter()
        ref = jax.block_until_ready(exact(p, scores, h))
        exact_wall = time.perf_counter() - t0
        ref_norm = jnp.linalg.norm(ref)
        for theta in args.thetas:
            t0 = time.perf_counter()
            walk = build_svgd_traversal(
                p,
                theta=theta,
                leaf_size=args.leaf_size,
                backend=backend,
                traversal_config=cfg,
            )
            topo = assemble_svgd_topology(walk)
            tree = jax.block_until_ready(phi(p, scores, h, topo))
            wall = time.perf_counter() - t0
            err = float(jnp.linalg.norm(tree - ref) / ref_norm)
            records.append(
                {
                    "n": n,
                    "theta": theta,
                    "rel_error_vs_exact": err,
                    "num_far_pairs": int(walk.far_src.shape[0]),
                    "num_near_leaf_pairs": int(topo.num_near_leaf_pairs),
                    "num_far_contribs": int(topo.num_far_contribs),
                    "max_leaf": int(topo.leaf_slots.shape[1]),
                    "exact_wall_s_cold": exact_wall,
                    "tree_wall_s_cold": wall,
                }
            )
            print(
                f"n={n:>7d} theta={theta:4.2f} err={err:9.3e} "
                f"far_pairs={records[-1]['num_far_pairs']:>8d} "
                f"near_leaf_pairs={records[-1]['num_near_leaf_pairs']:>9d} "
                f"({wall:6.1f} s)",
                flush=True,
            )
        dump_json(
            {
                "benchmark": "svgd_accuracy_matched_pilot",
                "params": params,
                "metadata": run_metadata(),
                "records": records,
            },
            args.output,
        )


if __name__ == "__main__":
    main()
