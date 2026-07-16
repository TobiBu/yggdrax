"""Scaling of the differentiable pair-count estimator vs N.

Times the two stages of the estimator separately over a sweep of catalog size:

* ``build`` -- construct the (non-differentiable) pair topology;
* ``accumulate`` -- the differentiable soft-count reduction over that topology;
* ``value_and_grad`` -- forward + reverse of the accumulation.

A brute-force :math:`O(N^2)` soft count is timed for the small sizes as a
context baseline (it is quadratic and only feasible for small N).

Results -> ``results/corrfunc/scaling.json``.

Production (GPU server):

    micromamba run -n odisseo python bench/corrfunc/scaling.py \
        --sizes 10000 100000 1000000 --gpu-select free

Local smoke (CPU):

    conda run -n jaccpot python bench/corrfunc/scaling.py --smoke
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
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--num-bins", type=int, default=12)
    p.add_argument("--r-min", type=float, default=0.005)
    p.add_argument("--r-max", type=float, default=0.5)
    p.add_argument("--sharpness", type=float, default=150.0)
    p.add_argument("--leaf-size", type=int, default=64)
    p.add_argument("--backend", type=str, default="radix")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--brute-force-max-n", type=int, default=20000)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--output", type=str, default="results/corrfunc/scaling.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.sizes = [2000]
        args.runs = 2
        args.warmup = 1
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="corrfunc-scaling")

    import time

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

    # The traversal preallocates its far-interaction and near-neighbour lists as
    # (num_nodes x max_interactions_per_node) and (num_leaves x
    # max_neighbors_per_leaf). At theta=0.5 the actual per-node/per-leaf counts
    # are O(10-100), so the previous 1<<15 caps preallocated ~8.6 GiB at N=1e6
    # and OOM'd the *build* before the estimator ran. Right-size them (still
    # >>40x the observed peaks) so the build fits; the near/far list contents --
    # and hence the result -- are unchanged. The pair queue is only a work
    # buffer (~a few hundred MB even at 1<<24), so keep it generous.
    cfg = DualTreeTraversalConfig(
        max_pair_queue=1 << 24,
        process_block=64,
        max_interactions_per_node=1 << 13,
        max_neighbors_per_leaf=1 << 13,
    )
    edges = make_log_edges(args.r_min, args.r_max, args.num_bins)
    sharp = args.sharpness

    records = []
    for n in args.sizes:
        key = jax.random.PRNGKey(args.seed)
        pos = jax.random.uniform(key, (n, 3), dtype=jnp.float64)

        # Topology build. The first call pays the capacity-retry compile ladder,
        # so time it warmed (like accumulate/value_and_grad) rather than as a
        # single cold call; block on the returned arrays so async dispatch does
        # not corrupt the measurement.
        build_fn = lambda: build_pair_topology(
            pos,
            theta=args.theta,
            leaf_size=args.leaf_size,
            backend=args.backend,
            traversal_config=cfg,
        )
        # Fewer repeats than the JAX kernels: each build also does host-side
        # numpy work that scales with N, so keep the warmed count bounded.
        build_t = time_callable(build_fn, warmup=1, runs=min(args.runs, 3))
        build_s = build_t.min_s
        topo = build_fn()

        accumulate = jax.jit(
            lambda p, t=topo: soft_pair_counts_from_topology(p, t, edges, sharp)
        )
        vgrad = jax.jit(
            lambda p, t=topo: jax.value_and_grad(
                lambda q: jnp.sum(soft_pair_counts_from_topology(q, t, edges, sharp))
            )(p)
        )
        acc_t = time_callable(
            lambda: accumulate(pos), warmup=args.warmup, runs=args.runs
        )
        vg_t = time_callable(lambda: vgrad(pos), warmup=args.warmup, runs=args.runs)

        entry = {
            "n": n,
            "build_s": build_s,
            "build": build_t.as_dict(),
            "accumulate": acc_t.as_dict(),
            "value_and_grad": vg_t.as_dict(),
            "num_far_pairs": int(topo.far_src_start.shape[0]),
            "num_near_leaf_pairs": int(topo.near_target_row.shape[0]),
        }
        if n <= args.brute_force_max_n:
            t0 = time.perf_counter()
            brute_force_soft_pair_counts(pos, edges, sharp)
            entry["brute_force_s"] = time.perf_counter() - t0
        records.append(entry)
        print(
            f"n={n:>8d} build={build_s * 1e3:8.1f} ms "
            f"acc={acc_t.min_s * 1e3:8.2f} ms vgrad={vg_t.min_s * 1e3:8.2f} ms"
        )

    payload = {
        "benchmark": "corrfunc_scaling",
        "params": {
            "sizes": args.sizes,
            "theta": args.theta,
            "num_bins": args.num_bins,
            "sharpness": sharp,
            "leaf_size": args.leaf_size,
            "backend": args.backend,
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
