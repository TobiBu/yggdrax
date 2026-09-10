"""Time the traced dual-tree walk in isolation (plan "tree walk", Phase 0.1).

The fused jaccpot lane refreshes its interaction lists every step through
``_dual_tree_walk_impl`` at a fixed ``max_pair_queue``; at N=200k / leaf 64 that
walk plus its list emission cost ~100 ms of a 177 ms step. This script times the
walk alone, on one tree, so a change to the loop body can be read without the
rest of an FMM step around it:

* ``dual``   -- ``build_interactions_and_neighbors`` under ``jax.jit`` with a fixed
  ``DualTreeTraversalConfig`` (traced inputs -> one wavefront walk at the given
  queue, no retry ladder), i.e. exactly what the fused lane runs per step;
* ``mutual`` -- ``dual_tree_walk_mutual`` on the same tree with the dual walk's own
  ``mac_extents`` (so the two produce the same far/near SETS), the flat-emission
  candidate for that lane;
* ``scatter`` -- a length-``Q`` ``.at[idx].set(..., mode="drop")`` with and without
  ``unique_indices=True``, the lowering claim behind the scatter promises.

Reports ms per walk (min and median of ``--repeats``), far/near pair counts, and
-- when the walk exposes them -- ``rounds`` and ``peak_wavefront``. Index precision
is read at import, so ``--index int32`` sets ``YGGDRAX_INDEX_PRECISION`` before
``import yggdrax``.

    python bench/traversal_walk_bench.py --n 200000 --leaf-size 64 --theta 0.6 \
        --max-pair-queue 1048576 --out bench/results/traversal_walk.json
    python bench/traversal_walk_bench.py --smoke          # tiny CPU run, seconds

A timing is only worth recording from an unoccupied card: the GPU is chosen with
``autocvd`` (site tool) and the card's utilisation before and after the run is
written into the result so a contended row can be recognised.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _plummer(n: int, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    return pos.astype(np.float32), np.full(n, 1.0 / n, np.float32)


def _gpu_util(index: int | None) -> int | None:
    if index is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             "-i", str(index)],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        return int(out)
    except Exception:  # pragma: no cover - diagnostics only
        return None


def _timed(fn, repeats: int):
    import jax

    out = jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn())
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return out, 1e3 * samples[0], 1e3 * samples[len(samples) // 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--leaf-size", type=int, default=64)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--mac-type", default="dehnen", choices=["bh", "dehnen"])
    ap.add_argument("--max-pair-queue", type=int, default=1 << 20)
    ap.add_argument("--max-interactions-per-node", type=int, default=16384)
    ap.add_argument("--max-neighbors-per-leaf", type=int, default=8192)
    ap.add_argument("--far-cap", type=int, default=1 << 21,
                    help="mutual walk: canonical far pairs capacity")
    ap.add_argument("--near-cap", type=int, default=1 << 21,
                    help="mutual walk: canonical near pairs capacity")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--index", default=None, choices=[None, "int32", "int64"],
                    help="YGGDRAX_INDEX_PRECISION (read at import)")
    ap.add_argument("--walks", default="dual,mutual,scatter")
    ap.add_argument("--cpu", action="store_true", help="force the CPU backend")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.n, args.leaf_size, args.max_pair_queue = 2000, 16, 1 << 12
        args.far_cap = args.near_cap = 1 << 14
        args.max_interactions_per_node = args.max_neighbors_per_leaf = 512
        args.repeats, args.cpu = 1, True
    if args.index:
        os.environ["YGGDRAX_INDEX_PRECISION"] = args.index
    gpu = None
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        from autocvd import autocvd  # site tool: picks a free card, sets CUDA_VISIBLE_DEVICES

        gpu = int(autocvd(num_gpus=1, least_used=False, timeout=600, progress=False)[0])
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    util_before = _gpu_util(gpu)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from yggdrax import (
        INDEX_DTYPE,
        DualTreeTraversalConfig,
        Tree,
        build_interactions_and_neighbors,
        compute_tree_geometry,
    )
    from yggdrax._interactions_impl import _build_mac_extents
    from yggdrax.interactions import dual_tree_walk_mutual

    pos, mass = _plummer(args.n)
    tree = Tree.from_particles(
        jnp.asarray(pos), jnp.asarray(mass), leaf_size=args.leaf_size, tree_type="radix"
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted, max_leaf_size=args.leaf_size)
    topo = tree.topology
    num_internal = int(topo.left_child.shape[0])
    total_nodes = int(topo.parent.shape[0])
    num_leaves = total_nodes - num_internal
    result = dict(
        n=args.n, leaf_size=args.leaf_size, theta=args.theta, mac_type=args.mac_type,
        num_leaves=num_leaves, total_nodes=total_nodes, index_dtype=str(jnp.dtype(INDEX_DTYPE)),
        max_pair_queue=args.max_pair_queue, device=str(jax.devices()[0]), gpu=gpu,
        gpu_util_before=util_before, walks={},
    )
    print(f"{result['device']} N={args.n} leaf {args.leaf_size} -> {num_leaves} leaves, "
          f"{total_nodes} nodes, index {result['index_dtype']}, Q={args.max_pair_queue}", flush=True)
    walks = set(args.walks.split(","))

    if "dual" in walks:
        config = DualTreeTraversalConfig(
            max_pair_queue=args.max_pair_queue, process_block=256,
            max_interactions_per_node=args.max_interactions_per_node,
            max_neighbors_per_leaf=args.max_neighbors_per_leaf,
        )

        @jax.jit
        def dual(topology, geom):
            _far, _near, res = build_interactions_and_neighbors(
                topology, geom, theta=args.theta, traversal_config=config,
                mac_type=args.mac_type, return_result=True,
            )
            return (res.far_pair_count, res.near_pair_count, res.queue_overflow,
                    res.far_overflow, res.near_overflow)

        (far, near, qo, fo, no), tmin, tmed = _timed(lambda: dual(topo, geometry), args.repeats)
        row = dict(ms_min=tmin, ms_median=tmed, far_pairs=int(far), near_pairs=int(near),
                   queue_overflow=bool(qo), far_overflow=bool(fo), near_overflow=bool(no))
        result["walks"]["dual"] = row
        print(f"dual   : {tmin:8.2f} ms (median {tmed:8.2f})  far {int(far)}  near {int(near)}  "
              f"overflow q/f/n {bool(qo)}/{bool(fo)}/{bool(no)}", flush=True)

    if "mutual" in walks:
        idx = topo.parent.dtype
        left_full = jnp.concatenate([jnp.asarray(topo.left_child, idx), jnp.full((num_leaves,), -1, idx)])
        right_full = jnp.concatenate([jnp.asarray(topo.right_child, idx), jnp.full((num_leaves,), -1, idx)])
        root = jnp.argmin(topo.parent).astype(idx)
        centers = jnp.asarray(geometry.center)
        extents = _build_mac_extents(topo.parent, geometry, num_internal, args.mac_type, 1.0)[0]
        extents = jnp.asarray(extents, dtype=centers.dtype)

        @jax.jit
        def mutual(lf, rf, c, r, rt):
            return dual_tree_walk_mutual(
                lf, rf, c, r, args.theta, rt, max_pair_queue=args.max_pair_queue,
                far_cap=args.far_cap, near_cap=args.near_cap,
            )

        res, tmin, tmed = _timed(lambda: mutual(left_full, right_full, centers, extents, root), args.repeats)
        row = dict(ms_min=tmin, ms_median=tmed, far_pairs_canonical=int(res.far_count),
                   near_pairs_canonical=int(res.near_count), far_pairs_directed=2 * int(res.far_count),
                   near_pairs_directed=2 * int(res.near_count),
                   queue_overflow=bool(res.queue_overflow), far_overflow=bool(res.far_overflow),
                   near_overflow=bool(res.near_overflow))
        for extra in ("peak_wavefront", "rounds"):
            if hasattr(res, extra):
                row[extra] = int(getattr(res, extra))
        result["walks"]["mutual"] = row
        print(f"mutual : {tmin:8.2f} ms (median {tmed:8.2f})  far {2*int(res.far_count)} (directed)  "
              f"near {2*int(res.near_count)}  overflow q/f/n {bool(res.queue_overflow)}/"
              f"{bool(res.far_overflow)}/{bool(res.near_overflow)}"
              + (f"  peak_wf {row['peak_wavefront']} rounds {row['rounds']}" if "rounds" in row else ""),
              flush=True)

    if "scatter" in walks:
        Q = int(args.max_pair_queue)
        rng = np.random.default_rng(1)
        live = rng.random(Q) < 0.3
        prefix = np.cumsum(live) - live
        slot = np.where(live, prefix, Q).astype(np.dtype(INDEX_DTYPE))
        vals = jnp.asarray(rng.integers(0, 1000, Q).astype(np.dtype(INDEX_DTYPE)))
        slot_j = jnp.asarray(slot)

        @jax.jit
        def plain(s, v):
            return jnp.full((Q,), -1, dtype=v.dtype).at[s].set(v, mode="drop")

        @jax.jit
        def promised(s, v):
            return jnp.full((Q,), -1, dtype=v.dtype).at[s].set(v, mode="drop", unique_indices=True)

        a, t_plain, _ = _timed(lambda: plain(slot_j, vals), args.repeats)
        b, t_prom, _ = _timed(lambda: promised(slot_j, vals), args.repeats)
        same = bool(jnp.array_equal(a, b))
        result["walks"]["scatter"] = dict(Q=Q, ms_plain=t_plain, ms_unique=t_prom, identical=same)
        print(f"scatter: Q={Q} plain {t_plain:7.3f} ms  unique_indices {t_prom:7.3f} ms  identical {same}",
              flush=True)

    result["gpu_util_after"] = _gpu_util(gpu)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
