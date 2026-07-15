"""Validate the tree soft pair-count estimator against the brute-force baseline.

For a few catalog sizes and opening angles, compare the tree-accelerated soft
counts against the exact :math:`O(N^2)` soft counts (and, if available, against
Corrfunc). Reports per-bin and total relative error. At ``theta`` tight enough
that no far pairs are accepted the near field is exact and the error is at the
float-precision floor; as ``theta`` opens, the monopole far approximation adds
a bounded error.

Results -> ``results/corrfunc/validation.json``.

Local smoke (CPU):

    conda run -n jaccpot python bench/corrfunc/validate_vs_baseline.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.differentiability._common import dump_json, run_metadata, select_free_gpu


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 3000, 10000])
    p.add_argument("--thetas", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9])
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--r-min", type=float, default=0.01)
    p.add_argument("--r-max", type=float, default=0.5)
    p.add_argument("--sharpness", type=float, default=150.0)
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument("--backend", type=str, default="radix")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--output", type=str, default="results/corrfunc/validation.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.sizes = [800]
        args.thetas = [0.0, 0.6]
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="corrfunc-validate")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from yggdrax import DualTreeTraversalConfig
    from yggdrax.applications.corrfunc.baselines import brute_force_soft_pair_counts
    from yggdrax.applications.corrfunc.binning import make_log_edges
    from yggdrax.applications.corrfunc.estimator import (
        build_pair_topology,
        soft_pair_counts_from_topology,
    )

    cfg = DualTreeTraversalConfig(
        max_pair_queue=1 << 20,
        process_block=64,
        max_interactions_per_node=1 << 15,
        max_neighbors_per_leaf=1 << 15,
    )
    edges = make_log_edges(args.r_min, args.r_max, args.num_bins)

    records = []
    for n in args.sizes:
        key = jax.random.PRNGKey(args.seed)
        pos = jax.random.uniform(key, (n, 3), dtype=jnp.float64)
        exact = brute_force_soft_pair_counts(pos, edges, args.sharpness)
        for theta in args.thetas:
            topo = build_pair_topology(
                pos,
                theta=theta,
                leaf_size=args.leaf_size,
                backend=args.backend,
                traversal_config=cfg,
            )
            est = soft_pair_counts_from_topology(pos, topo, edges, args.sharpness)
            per_bin = jnp.abs(est - exact) / jnp.maximum(exact, 1.0)
            rec = {
                "n": n,
                "theta": theta,
                "num_far_pairs": int(topo.far_src_start.shape[0]),
                "max_per_bin_rel_error": float(jnp.max(per_bin)),
                "total_rel_error": float(
                    jnp.abs(est.sum() - exact.sum()) / exact.sum()
                ),
                "estimator_counts": [float(x) for x in est],
                "baseline_counts": [float(x) for x in exact],
            }
            records.append(rec)
            print(
                f"n={n:>7d} theta={theta:4.2f} far={rec['num_far_pairs']:6d} "
                f"max_relerr={rec['max_per_bin_rel_error']:.3e}"
            )

    payload = {
        "benchmark": "corrfunc_validation",
        "params": {
            "sizes": args.sizes,
            "thetas": args.thetas,
            "num_bins": args.num_bins,
            "r_min": args.r_min,
            "r_max": args.r_max,
            "sharpness": args.sharpness,
            "leaf_size": args.leaf_size,
            "backend": args.backend,
            "seed": args.seed,
        },
        "bin_edges": [float(x) for x in edges],
        "metadata": run_metadata(),
        "records": records,
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
