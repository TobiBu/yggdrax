"""Scaling benchmark: tree build + dual-tree traversal time vs N.

Measures wall-clock build time and interaction/neighbor traversal time for the
radix, octree, and KD-tree backends over a sweep of particle counts.

Two paths are timed per size:

* ``legacy`` (per backend): adaptive ``Tree.from_particles`` build, then the
  eager ``build_interactions_and_neighbors`` dual walk with its per-node CSR
  output. This is what the first paper draft reported.
* ``fast`` (radix only): the jitted static-shape cell-leaf radix build
  (``build_static_cells_tree``), the geometry and Dehnen extents, and the flat
  ``dual_tree_walk_mutual`` emitting each unordered far/near node pair once.
  Each stage is timed under ``jax.jit``, and so is the fused build+walk step
  (the number that matters for a rebuild inside a differentiated function).
  Capacities (leaf capacity, pair queue, far/near list caps) are sized from an
  untimed probe run and then fixed, so the timed calls never overflow; the
  sizing and the overflow flags are recorded.

Includes a
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
    is_out_of_memory,
    out_of_memory_stop,
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
        "--no-legacy",
        action="store_true",
        help="Skip the per-backend adaptive build + eager dual walk "
        "(build_interactions_and_neighbors) timings.",
    )
    p.add_argument(
        "--no-fast",
        action="store_true",
        help="Skip the fast path: jitted static cell-leaf radix build + "
        "dual_tree_walk_mutual, timed separately and as one fused jit.",
    )
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

    # yggdrax needs x64 and no longer switches it on at import (#97). The
    # positions below stay float32, as they were when the import switched it on.
    jax.config.update("jax_enable_x64", True)

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

    from yggdrax._interactions_impl import _build_mac_extents
    from yggdrax._tree_impl import build_static_cells_tree
    from yggdrax.bounds import infer_bounds
    from yggdrax.interactions import dual_tree_walk_mutual

    def _pow2(x: int) -> int:
        return 1 << max(int(x) - 1, 1).bit_length()

    def _fast_path(pos, mass) -> dict:
        """Jitted cell-leaf radix build + mutual walk, capacities sized first."""
        n = int(pos.shape[0])
        leaf = args.leaf_size
        bounds = infer_bounds(pos)

        # --- size the leaf capacity from the partition's own live-leaf count.
        cap0 = max(64, 4 * (-(-n // leaf)))
        _, _, _, _, ovf, nl = build_static_cells_tree(
            pos,
            mass,
            bounds,
            leaf_size=leaf,
            leaf_capacity=cap0,
            return_reordered=True,
            return_overflow=True,
            return_num_leaves=True,
        )
        live_leaves = int(nl)
        leaf_capacity = int(1.1 * live_leaves) + 8

        def build(p, m):
            topo, ps, _ms, _inv, overflow = build_static_cells_tree(
                p,
                m,
                bounds,
                leaf_size=leaf,
                leaf_capacity=leaf_capacity,
                return_reordered=True,
                return_overflow=True,
            )
            return topo, ps, overflow

        def walk_inputs(topo, ps):
            geometry = compute_tree_geometry(topo, ps)
            num_internal = topo.left_child.shape[0]
            total = topo.parent.shape[0]
            idx = topo.parent.dtype
            pad = jnp.full((total - num_internal,), -1, idx)
            lf = jnp.concatenate([jnp.asarray(topo.left_child, idx), pad])
            rf = jnp.concatenate([jnp.asarray(topo.right_child, idx), pad])
            root = jnp.argmin(topo.parent).astype(idx)
            centers = jnp.asarray(geometry.center)
            extents = jnp.asarray(
                _build_mac_extents(topo.parent, geometry, num_internal, "dehnen", 1.0)[
                    0
                ],
                centers.dtype,
            )
            active = topo.node_ranges[:, 1] >= topo.node_ranges[:, 0]
            return lf, rf, centers, extents, root, active

        def make_walk(q, far_cap, near_cap):
            def walk(lf, rf, centers, extents, root, active):
                return dual_tree_walk_mutual(
                    lf,
                    rf,
                    centers,
                    extents,
                    args.theta,
                    root,
                    max_pair_queue=q,
                    far_cap=far_cap,
                    near_cap=near_cap,
                    mac_type="dehnen",
                    node_active=active,
                )

            return walk

        build_jit = jax.jit(build)
        inputs_jit = jax.jit(walk_inputs)
        topo, ps, overflow = build_jit(pos, mass)
        if bool(overflow):
            raise RuntimeError(f"cell partition overflowed at n={n}")
        winputs = inputs_jit(topo, ps)

        # --- size queue and list caps: double on overflow, then tighten.
        q, far_cap, near_cap = 1 << 14, 1 << 16, 1 << 16
        for _ in range(16):
            res = jax.jit(make_walk(q, far_cap, near_cap))(*winputs)
            qo, fo, no = (
                bool(res.queue_overflow),
                bool(res.far_overflow),
                bool(res.near_overflow),
            )
            if not (qo or fo or no):
                break
            q, far_cap, near_cap = (
                q * (4 if qo else 1),
                far_cap * (4 if fo else 1),
                near_cap * (4 if no else 1),
            )
        else:
            raise RuntimeError(f"mutual walk capacities did not converge at n={n}")
        peak = int(res.peak_wavefront)
        far_count, near_count = int(res.far_count), int(res.near_count)
        q = _pow2(int(1.5 * peak) + 1)
        far_cap = _pow2(int(1.25 * far_count) + 1)
        near_cap = _pow2(int(1.25 * near_count) + 1)
        walk = make_walk(q, far_cap, near_cap)
        walk_jit = jax.jit(walk)

        @jax.jit
        def step(p, m):
            t, s, o = build(p, m)
            r = walk(*walk_inputs(t, s))
            return (
                r.far_count,
                r.near_count,
                o,
                r.queue_overflow,
                r.far_overflow,
                r.near_overflow,
            )

        build_timing = time_callable(
            lambda: build_jit(pos, mass), warmup=args.warmup, runs=args.runs
        )
        inputs_timing = time_callable(
            lambda: inputs_jit(topo, ps), warmup=args.warmup, runs=args.runs
        )
        walk_timing = time_callable(
            lambda: walk_jit(*winputs), warmup=args.warmup, runs=args.runs
        )
        step_timing = time_callable(
            lambda: step(pos, mass), warmup=args.warmup, runs=args.runs
        )
        res = walk_jit(*winputs)
        flags = step(pos, mass)
        if any(bool(x) for x in flags[2:]) or bool(res.queue_overflow):
            raise RuntimeError(
                f"fast path overflowed in the timed configuration at n={n}"
            )
        out = {
            "tree": "radix_static_cells",
            "walk": "dual_tree_walk_mutual",
            "mac_type": "dehnen",
            "build": build_timing.as_dict(),
            "geometry_and_extents": inputs_timing.as_dict(),
            "traverse": walk_timing.as_dict(),
            "fused_build_and_traverse": step_timing.as_dict(),
            "live_leaves": live_leaves,
            "leaf_capacity": leaf_capacity,
            "max_pair_queue": q,
            "far_cap": far_cap,
            "near_cap": near_cap,
            "rounds": int(res.rounds),
            "peak_wavefront": int(res.peak_wavefront),
            # canonical (unordered) pairs; directed = 2x, comparable to the
            # legacy walk's per-node lists.
            "far_pairs_canonical": int(res.far_count),
            "near_pairs_canonical": int(res.near_count),
        }
        print(
            f"n={n:>9d} fast    "
            f"build={build_timing.min_s * 1e3:8.2f} ms  "
            f"traverse={walk_timing.min_s * 1e3:8.2f} ms  "
            f"fused={step_timing.min_s * 1e3:8.2f} ms  "
            f"(leaves {live_leaves}, rounds {out['rounds']}, peak {out['peak_wavefront']})"
        )
        return out

    def _measure(n: int) -> dict:
        """Measure one size; raises if it does not fit in memory."""
        pos, mass = make_problem(n, args.seed)
        pos_np = np.asarray(pos)
        entry: dict = {"n": n, "backends": {}}

        if not args.no_fast:
            entry["fast"] = _fast_path(pos, mass)

        for backend in [] if args.no_legacy else args.backends:

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
        return entry

    records = []
    stopped_early = None
    for n in args.sizes:
        try:
            records.append(_measure(n))
        except Exception as exc:
            if not is_out_of_memory(exc):
                raise
            stopped_early = out_of_memory_stop(n, exc)
            break

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
            "legacy": not args.no_legacy,
            "fast": not args.no_fast,
        },
        "metadata": run_metadata(),
        "sweep": {"stopped_early": stopped_early},
        "records": records,
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
