"""Tree-accelerated SVGD vs. exact O(N^2) SVGD on toy targets, across theta.

For Gaussian / GMM / banana targets, run both the exact and the tree sampler
from the same initial particles and compare the resulting empirical
distributions (moment errors and MMD) as a function of the opening angle
``theta``. Tight ``theta`` recovers the exact result; larger ``theta`` trades
accuracy for a lighter far field.

Results -> ``results/svgd/convergence.json``.

Local smoke (CPU):

    conda run -n jaccpot python bench/svgd/convergence_vs_exact.py --smoke
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
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--thetas", type=float, nargs="+", default=[0.2, 0.4, 0.7, 1.0])
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--step-size", type=float, default=0.2)
    # Convergence is the ACCURACY figure (MMD-to-exact vs theta). A finer tree
    # (small leaf) keeps the far field non-trivial across the theta sweep so the
    # accuracy/theta tradeoff shows a clean diagonal; a coarse leaf (e.g. 64)
    # makes tight-theta MMD collapse to ~0 (all-near, exact) and flattens the
    # curve. Speed here comes from the radix backend + jitted geometry/phi, not
    # from enlarging the leaf. (Scaling/bandwidth use leaf_size=64 for speed.)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        help="Tree backend: 'auto' -> radix for d<=3 else leaf_kdtree.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--output", type=str, default="results/svgd/convergence.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.n = 150
        args.thetas = [0.3, 0.8]
        args.num_steps = 30
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="svgd-convergence")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from yggdrax import DualTreeTraversalConfig
    from yggdrax.applications.svgd import targets as T
    from yggdrax.applications.svgd.bandwidth_learning import squared_mmd
    from yggdrax.applications.svgd.exact import run_svgd
    from yggdrax.applications.svgd.kernel import median_heuristic
    from yggdrax.applications.svgd.sampler import run_tree_svgd

    cfg = DualTreeTraversalConfig(
        max_pair_queue=1 << 20,
        process_block=64,
        max_interactions_per_node=1 << 15,
        max_neighbors_per_leaf=1 << 15,
    )

    def make_targets():
        return {
            "gaussian": T.gaussian(jnp.array([1.0, 0.0]), jnp.array([1.0, 1.0])),
            "gmm": T.gaussian_mixture(
                jnp.array([[-2.5, 0.0], [2.5, 0.0]]), jnp.array([0.5, 0.5])
            ),
            "banana": T.banana(curvature=0.3, scale=2.0),
        }

    records = []
    for name, tgt in make_targets().items():
        backend = args.backend
        if backend == "auto":
            backend = "radix" if tgt.dim <= 3 else "leaf_kdtree"
        key = jax.random.PRNGKey(args.seed)
        p0 = jax.random.normal(key, (args.n, tgt.dim)) * 0.6
        h = float(median_heuristic(p0))
        pe = run_svgd(p0, tgt.score, h, args.step_size, args.num_steps)
        mmd_bw = float(jnp.sqrt(jnp.mean(pe.var(0))) + 1e-6)
        for theta in args.thetas:
            pt = run_tree_svgd(
                p0,
                tgt.score,
                h,
                args.step_size,
                args.num_steps,
                theta=theta,
                leaf_size=args.leaf_size,
                backend=backend,
                traversal_config=cfg,
            )
            rec = {
                "target": name,
                "theta": theta,
                "mean_abs_err": float(jnp.max(jnp.abs(pe.mean(0) - pt.mean(0)))),
                "std_abs_err": float(jnp.max(jnp.abs(pe.std(0) - pt.std(0)))),
                "mmd_to_exact": float(squared_mmd(pt, pe, mmd_bw)),
            }
            records.append(rec)
            print(
                f"{name:9s} theta={theta:4.2f} "
                f"mean_err={rec['mean_abs_err']:.3e} std_err={rec['std_abs_err']:.3e} "
                f"mmd={rec['mmd_to_exact']:.3e}"
            )

    payload = {
        "benchmark": "svgd_convergence",
        "params": {
            "n": args.n,
            "thetas": args.thetas,
            "num_steps": args.num_steps,
            "step_size": args.step_size,
            "leaf_size": args.leaf_size,
            "backend": args.backend,
            "seed": args.seed,
        },
        "metadata": run_metadata(),
        "records": records,
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
