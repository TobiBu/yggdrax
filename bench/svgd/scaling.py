"""Scaling of the tree-accelerated Stein update vs N at fixed dimension.

Times the stages of one SVGD update over a sweep of particle count:

* ``build`` -- construct the (non-differentiable) near/far partition, split into
  its device half (tree build, geometry, dual-tree walk) and its host half (the
  numpy assembly of the partition);
* ``phi`` -- the differentiable Stein-update accumulation over that partition;
* ``value_and_grad`` -- forward + reverse of the accumulation w.r.t. positions.

The exact O(N^2) Stein update is timed for the small sizes as the baseline the
tree update has to beat. It is **jitted and warmed** like everything else here:
timing an eager op-dispatch loop against a compiled one measures the dispatch
overhead, not the algorithm.

Per-record counters (``num_far_pairs``, ``num_near_leaf_pairs``,
``num_far_contribs``, peak device memory) make the effect of a far-field or
near-field change visible in the JSON rather than only in the wall clock.

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
    device_memory_stats,
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
    p.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help=(
            "Working precision. 'float64' enables jax_enable_x64 (this bench's "
            "historical default); 'float32' leaves it off."
        ),
    )
    p.add_argument(
        "--cutoff-bandwidths",
        type=float,
        default=None,
        help=(
            "Kernel-aware far-field cutoff c, in bandwidths: node pairs whose "
            "closest possible separation exceeds c * h contribute nothing and "
            "are dropped by the pair policy. Omit for the monopole-everything "
            "far field."
        ),
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

    import jax
    import jax.numpy as jnp

    use_x64 = args.dtype == "float64"
    jax.config.update("jax_enable_x64", use_x64)

    from yggdrax import DualTreeTraversalConfig
    from yggdrax.applications.svgd.exact import exact_phi
    from yggdrax.applications.svgd.sampler import (
        assemble_svgd_topology,
        build_svgd_traversal,
        svgd_phi_from_topology,
    )

    dtype = jnp.float64 if use_x64 else jnp.float32

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
        p = (jax.random.normal(key, (n, dim), dtype=dtype) * 1.2).astype(dtype)
        scores = -p  # standard-normal-like score for timing purposes
        h = 0.5

        # The build is timed in its two halves. The first call pays the
        # traversal's capacity-retry compile ladder, so both halves are warmed
        # first; a cold ``build_s`` measures compilation, not the partition.
        cutoff = None if args.cutoff_bandwidths is None else args.cutoff_bandwidths * h

        def walk_fn(pp=p, rc=cutoff):
            return build_svgd_traversal(
                pp,
                theta=args.theta,
                leaf_size=args.leaf_size,
                backend=backend,
                traversal_config=cfg,
                kernel_cutoff=rc,
            )

        # Fewer repeats than the JAX kernels: the host half does numpy work that
        # scales with N, so keep the warmed count bounded.
        build_runs = min(args.runs, 3)
        walk_t = time_callable(walk_fn, warmup=1, runs=build_runs)
        walk = walk_fn()
        assemble_t = time_callable(
            lambda w=walk: assemble_svgd_topology(w), warmup=1, runs=build_runs
        )
        topo = assemble_svgd_topology(walk)
        build_s = walk_t.min_s + assemble_t.min_s

        # The partition and the scores are jit *arguments*, not closure
        # constants. Closed over, XLA constant-folds the whole
        # sco[leaf_slots] / pos[leaf_slots] gather tree into a literal (it says
        # so, at N=1e4: "%gather.7 = f32[250704,1,32,3] gather(%constant.56,
        # %constant.58)") and the timing then omits work every real per-step
        # rebuild has to do -- flattering the tree update against a baseline
        # that has no gathers to fold.
        phi = jax.jit(svgd_phi_from_topology)
        vg = jax.jit(
            lambda pp, sc, hh, t: jax.value_and_grad(
                lambda q: jnp.sum(svgd_phi_from_topology(q, sc, hh, t) ** 2)
            )(pp)
        )
        phi_t = time_callable(
            lambda: phi(p, scores, h, topo), warmup=args.warmup, runs=args.runs
        )
        vg_t = time_callable(
            lambda: vg(p, scores, h, topo), warmup=args.warmup, runs=args.runs
        )

        entry = {
            "n": n,
            "dim": dim,
            "build_s": build_s,
            "build_device": walk_t.as_dict(),
            "build_host": assemble_t.as_dict(),
            "phi": phi_t.as_dict(),
            "value_and_grad": vg_t.as_dict(),
            "grad_ratio": vg_t.min_s / phi_t.min_s,
            "num_far_pairs": int(walk.far_src.shape[0]),
            # Directed count, so it stays comparable across work packages: WP2
            # stores one row per *unordered* pair and reports the doubled count.
            "num_near_leaf_pairs": int(topo.num_near_leaf_pairs),
            "num_near_pair_rows": int(topo.near_target_row.shape[0]),
            "num_far_contribs": int(topo.far_tgt_slot.shape[0]),
            "max_leaf": int(topo.leaf_slots.shape[1]),
            "device_memory": device_memory_stats(),
        }
        if n <= args.exact_max_n:
            exact = jax.jit(exact_phi)
            ref = exact(p, scores, h)
            tree = phi(p, scores, h, topo)
            entry["rel_error_vs_exact"] = float(
                jnp.linalg.norm(tree - ref) / jnp.linalg.norm(ref)
            )
            exact_t = time_callable(
                lambda: exact(p, scores, h), warmup=args.warmup, runs=args.runs
            )
            entry["exact"] = exact_t.as_dict()
            entry["exact_s"] = exact_t.min_s
            entry["speedup_vs_exact"] = exact_t.min_s / phi_t.min_s
        records.append(entry)
        msg = (
            f"n={n:>8d} build={build_s * 1e3:8.1f} ms "
            f"(dev={walk_t.min_s * 1e3:7.1f} host={assemble_t.min_s * 1e3:7.1f}) "
            f"phi={phi_t.min_s * 1e3:8.2f} ms vgrad={vg_t.min_s * 1e3:8.2f} ms "
            f"ratio={entry['grad_ratio']:4.2f} "
            f"far_pairs={entry['num_far_pairs']:>8d} "
            f"M={entry['num_far_contribs']:>9d}"
        )
        if "exact_s" in entry:
            msg += f" exact={entry['exact_s'] * 1e3:8.2f} ms"
        print(msg)

    payload = {
        "benchmark": "svgd_scaling",
        "params": {
            "sizes": args.sizes,
            "dim": dim,
            "theta": args.theta,
            "leaf_size": args.leaf_size,
            "backend": backend,
            "dtype": args.dtype,
            "cutoff_bandwidths": args.cutoff_bandwidths,
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
