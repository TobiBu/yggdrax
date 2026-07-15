"""Gradient-based clustering-parameter recovery through the estimator.

Runs the differentiability-payoff demo: generate a catalogue at a known
clustering strength ``a_true``, then recover it by gradient descent on the soft
pair counts (see :mod:`yggdrax.applications.corrfunc.inference_demo`). Records
the optimization trajectory for one or more seeds/initial guesses.

Results -> ``results/corrfunc/recovery.json``.

Local smoke (CPU):

    conda run -n jaccpot python bench/corrfunc/parameter_recovery_demo.py --smoke
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
    p.add_argument("--num-centers", type=int, default=8)
    p.add_argument("--a-true", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--a-init", type=float, default=0.1)
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--r-min", type=float, default=0.01)
    p.add_argument("--r-max", type=float, default=0.5)
    p.add_argument("--sharpness", type=float, default=100.0)
    p.add_argument("--theta", type=float, default=0.4)
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--output", type=str, default="results/corrfunc/recovery.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.n = 400
        args.a_true = [0.5]
        args.num_steps = 4
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="corrfunc-recovery")

    import jax

    jax.config.update("jax_enable_x64", True)

    from yggdrax import DualTreeTraversalConfig
    from yggdrax.applications.corrfunc.binning import make_log_edges
    from yggdrax.applications.corrfunc.inference_demo import (
        make_cluster_model,
        recover_parameter,
    )

    cfg = DualTreeTraversalConfig(
        max_pair_queue=1 << 20,
        process_block=64,
        max_interactions_per_node=1 << 15,
        max_neighbors_per_leaf=1 << 15,
    )
    edges = make_log_edges(args.r_min, args.r_max, args.num_bins)
    model = make_cluster_model(
        jax.random.PRNGKey(args.seed), n=args.n, num_centers=args.num_centers
    )

    records = []
    for a_true in args.a_true:
        res = recover_parameter(
            model,
            edges,
            a_true=a_true,
            a_init=args.a_init,
            sharpness=args.sharpness,
            theta=args.theta,
            leaf_size=args.leaf_size,
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            traversal_config=cfg,
        )
        records.append(
            {
                "a_true": res.a_true,
                "a_init": res.a_init,
                "a_final": res.a_final,
                "abs_error": abs(res.a_final - res.a_true),
                "a_history": res.a_history,
                "loss_history": res.loss_history,
            }
        )
        print(
            f"a_true={a_true:.2f} -> a_final={res.a_final:.4f} "
            f"err={abs(res.a_final - a_true):.4f}"
        )

    payload = {
        "benchmark": "corrfunc_recovery",
        "params": {
            "n": args.n,
            "num_centers": args.num_centers,
            "a_true": args.a_true,
            "a_init": args.a_init,
            "num_bins": args.num_bins,
            "sharpness": args.sharpness,
            "theta": args.theta,
            "leaf_size": args.leaf_size,
            "learning_rate": args.learning_rate,
            "num_steps": args.num_steps,
            "seed": args.seed,
        },
        "metadata": run_metadata(),
        "records": records,
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
