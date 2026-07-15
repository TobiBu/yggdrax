"""Effect of soft-bin sharpness on accuracy and gradient quality.

The soft-window sharpness trades bias against gradient conditioning: too soft
(low sharpness) biases the counts away from the hard-bin histogram; too sharp
(high sharpness) recovers the hard bins but concentrates the gradient into a
thin shell at each bin edge, making it stiff and finite-difference-fragile.
This sweep quantifies both effects against sharpness:

* ``bias`` -- total relative error of the soft counts vs. the hard-bin counts;
* ``grad_fd_error`` -- max relative disagreement between the reverse-mode
  gradient (w.r.t. positions) and central finite differences, which grows as
  the window sharpens and the finite-difference step straddles the transition.

Results -> ``results/corrfunc/bin_sensitivity.json``.

Local smoke (CPU):

    conda run -n jaccpot python bench/corrfunc/bin_width_sensitivity.py --smoke
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
    p.add_argument("--n", type=int, default=3000)
    p.add_argument(
        "--sharpness",
        type=float,
        nargs="+",
        default=[10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
    )
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--r-min", type=float, default=0.01)
    p.add_argument("--r-max", type=float, default=0.5)
    p.add_argument("--theta", type=float, default=0.0)
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument(
        "--output", type=str, default="results/corrfunc/bin_sensitivity.json"
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.n = 800
        args.sharpness = [30.0, 300.0]
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="corrfunc-binsens")

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from yggdrax import DualTreeTraversalConfig
    from yggdrax.applications.corrfunc.baselines import brute_force_pair_counts
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

    key = jax.random.PRNGKey(args.seed)
    pos = jax.random.uniform(key, (args.n, 3), dtype=jnp.float64)
    topo = build_pair_topology(
        pos, theta=args.theta, leaf_size=args.leaf_size, traversal_config=cfg
    )
    hard = brute_force_pair_counts(pos, edges)

    x = pos.reshape(-1)
    rng = np.random.default_rng(0)
    idxs = rng.choice(x.shape[0], size=12, replace=False)

    records = []
    for sharpness in args.sharpness:

        def scalar(flat, s=sharpness):
            counts = soft_pair_counts_from_topology(
                flat.reshape(pos.shape), topo, edges, s
            )
            weights = jnp.arange(1, counts.shape[0] + 1, dtype=counts.dtype)
            return jnp.sum(counts * weights)

        soft = soft_pair_counts_from_topology(pos, topo, edges, sharpness)
        bias = float(jnp.abs(soft.sum() - hard.sum()) / hard.sum())

        analytic = np.asarray(jax.grad(scalar)(x))[idxs]
        h = 1e-6
        fd = np.array(
            [
                float((scalar(x.at[i].add(h)) - scalar(x.at[i].add(-h))) / (2 * h))
                for i in idxs
            ]
        )
        grad_fd_error = float(
            np.max(np.abs(analytic - fd) / np.maximum(np.abs(fd), 1e-8))
        )

        records.append(
            {
                "sharpness": sharpness,
                "bias": bias,
                "grad_fd_error": grad_fd_error,
            }
        )
        print(
            f"sharpness={sharpness:8.1f}  bias={bias:.3e}  grad_fd_error={grad_fd_error:.3e}"
        )

    payload = {
        "benchmark": "corrfunc_bin_sensitivity",
        "params": {
            "n": args.n,
            "sharpness": args.sharpness,
            "num_bins": args.num_bins,
            "theta": args.theta,
            "leaf_size": args.leaf_size,
            "seed": args.seed,
        },
        "metadata": run_metadata(),
        "records": records,
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
