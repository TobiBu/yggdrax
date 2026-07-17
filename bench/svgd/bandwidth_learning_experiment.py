"""Learned vs. median-heuristic SVGD bandwidth on a mismatched-scale target.

The differentiability payoff experiment. On an anisotropic Gaussian (where the
median heuristic under-disperses the wide axis), learn the RBF bandwidth by
backpropagating an MMD validation loss through several tree-accelerated SVGD
steps, then compare the sampler's dispersion under the learned bandwidth
against the median-heuristic default.

Results -> ``results/svgd/bandwidth_learning.json``.

Local smoke (CPU):

    conda run -n jaccpot python bench/svgd/bandwidth_learning_experiment.py --smoke
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
    p.add_argument("--n", type=int, default=300)
    p.add_argument(
        "--cov-diag",
        type=float,
        nargs="+",
        default=[1.0, 9.0],
        help="Diagonal covariance of the anisotropic Gaussian target.",
    )
    p.add_argument("--num-svgd-steps", type=int, default=20)
    p.add_argument("--num-outer-steps", type=int, default=40)
    p.add_argument("--step-size", type=float, default=0.3)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=0.15)
    p.add_argument("--eval-steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--output", type=str, default="results/svgd/bandwidth_learning.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.n = 120
        args.num_svgd_steps = 6
        args.num_outer_steps = 5
        args.eval_steps = 15
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="svgd-bandwidth")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from yggdrax import DualTreeTraversalConfig
    from yggdrax.applications.svgd import targets as T
    from yggdrax.applications.svgd.bandwidth_learning import learn_bandwidth
    from yggdrax.applications.svgd.kernel import median_heuristic
    from yggdrax.applications.svgd.sampler import run_tree_svgd

    cfg = DualTreeTraversalConfig(
        max_pair_queue=1 << 18,
        process_block=32,
        max_interactions_per_node=1 << 14,
        max_neighbors_per_leaf=1 << 14,
    )
    cov_diag = jnp.asarray(args.cov_diag)
    dim = int(cov_diag.shape[0])
    tgt = T.gaussian(jnp.zeros(dim), cov_diag)

    key = jax.random.PRNGKey(args.seed)
    p0 = jax.random.normal(key, (args.n, dim)) * 0.5
    target_samples = tgt.sample(jax.random.PRNGKey(args.seed + 100), 2 * args.n)
    mmd_bw = float(jnp.sqrt(jnp.mean(cov_diag)))

    h_median = float(median_heuristic(p0))
    result = learn_bandwidth(
        p0,
        tgt.score,
        target_samples,
        h_init=h_median,
        mmd_bandwidth=mmd_bw,
        step_size=args.step_size,
        num_svgd_steps=args.num_svgd_steps,
        theta=args.theta,
        leaf_size=args.leaf_size,
        learning_rate=args.learning_rate,
        num_outer_steps=args.num_outer_steps,
        traversal_config=cfg,
    )

    def dispersion(h):
        pf = run_tree_svgd(
            p0,
            tgt.score,
            h,
            args.step_size,
            args.eval_steps,
            theta=args.theta,
            leaf_size=args.leaf_size,
            traversal_config=cfg,
        )
        return [float(v) for v in pf.std(0)]

    target_std = [float(v) for v in jnp.sqrt(cov_diag)]
    std_median = dispersion(h_median)
    std_learned = dispersion(result.h_final)
    print(
        f"h_median={h_median:.3f} -> h_learned={result.h_final:.3f}\n"
        f"target std {target_std}\n"
        f"median-h std {std_median}\nlearned-h std {std_learned}"
    )

    payload = {
        "benchmark": "svgd_bandwidth_learning",
        "params": {
            "n": args.n,
            "cov_diag": args.cov_diag,
            "num_svgd_steps": args.num_svgd_steps,
            "num_outer_steps": args.num_outer_steps,
            "step_size": args.step_size,
            "theta": args.theta,
            "leaf_size": args.leaf_size,
            "learning_rate": args.learning_rate,
            "eval_steps": args.eval_steps,
            "seed": args.seed,
        },
        "target_std": target_std,
        "h_median": h_median,
        "h_learned": result.h_final,
        "h_history": result.h_history,
        "loss_history": result.loss_history,
        "std_median_heuristic": std_median,
        "std_learned": std_learned,
        "metadata": run_metadata(),
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
