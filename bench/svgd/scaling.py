"""Scaling of the tree-accelerated Stein update vs N at fixed dimension.

Times the two stages of one SVGD update over a sweep of particle count:

* ``build`` -- construct the (non-differentiable) near/far partition;
* ``phi`` -- the differentiable Stein-update accumulation over that partition;
* ``value_and_grad`` -- forward + reverse of the accumulation w.r.t. positions.

The exact O(N^2) Stein update is timed for the small sizes as context. For the
small sizes the accuracy of the tree update vs. exact is also recorded.

Results -> ``results/svgd/scaling.json``.

Production (GPU server):

    micromamba run -n odisseo python bench/svgd/scaling.py \
        --sizes 10000 100000 1000000 --gpu-select free

Local smoke (CPU):

    conda run -n jaccpot python bench/svgd/scaling.py --smoke
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
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--theta", type=float, default=0.5)
    # leaf_size=32 keeps the far field non-trivial at these N so the timing and
    # far-pair sweep actually exercise the far-field (monopole) path; a coarse
    # leaf (e.g. 64) collapses the far field to ~0 (all-near, near-exact) at
    # these N. The build is timed once per size, so the finer leaf costs little.
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        help="Tree backend: 'auto' -> radix for d<=3 else leaf_kdtree.",
    )
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exact-max-n", type=int, default=5000)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--output", type=str, default="results/svgd/scaling.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.sizes = [2000]
        args.runs = 2
        args.warmup = 1
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="svgd-scaling")

    import time

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from yggdrax import DualTreeTraversalConfig
    from yggdrax.applications.svgd.exact import exact_phi
    from yggdrax.applications.svgd.sampler import (
        build_svgd_topology,
        svgd_phi_from_topology,
    )

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

    records = []
    for n in args.sizes:
        key = jax.random.PRNGKey(args.seed)
        kp, ks = jax.random.split(key)
        p = jax.random.normal(kp, (n, dim)) * 1.2
        scores = -p  # standard-normal-like score for timing purposes
        h = 0.5

        t0 = time.perf_counter()
        topo = build_svgd_topology(
            p,
            theta=args.theta,
            leaf_size=args.leaf_size,
            backend=backend,
            traversal_config=cfg,
        )
        build_s = time.perf_counter() - t0

        phi = jax.jit(lambda pp, t=topo: svgd_phi_from_topology(pp, scores, h, t))
        vg = jax.jit(
            lambda pp, t=topo: jax.value_and_grad(
                lambda q: jnp.sum(svgd_phi_from_topology(q, scores, h, t) ** 2)
            )(pp)
        )
        phi_t = time_callable(lambda: phi(p), warmup=args.warmup, runs=args.runs)
        vg_t = time_callable(lambda: vg(p), warmup=args.warmup, runs=args.runs)

        entry = {
            "n": n,
            "dim": dim,
            "build_s": build_s,
            "phi": phi_t.as_dict(),
            "value_and_grad": vg_t.as_dict(),
            "num_far_contribs": int(topo.far_tgt_slot.shape[0]),
        }
        if n <= args.exact_max_n:
            ref = exact_phi(p, scores, h)
            tree = phi(p)
            entry["rel_error_vs_exact"] = float(
                jnp.linalg.norm(tree - ref) / jnp.linalg.norm(ref)
            )
            t0 = time.perf_counter()
            jax.block_until_ready(exact_phi(p, scores, h))
            entry["exact_s"] = time.perf_counter() - t0
        records.append(entry)
        print(
            f"n={n:>8d} build={build_s * 1e3:8.1f} ms "
            f"phi={phi_t.min_s * 1e3:8.2f} ms vgrad={vg_t.min_s * 1e3:8.2f} ms"
        )

    payload = {
        "benchmark": "svgd_scaling",
        "params": {
            "sizes": args.sizes,
            "dim": dim,
            "theta": args.theta,
            "leaf_size": args.leaf_size,
            "backend": backend,
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
