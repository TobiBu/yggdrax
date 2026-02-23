"""Smoke test for KD-tree vs Radix-tree compatibility with jaccpot call paths.

Run in the `expanse` conda environment:
    python examples/kdtree_jaccpot_smoke.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp

from yggdrax import (
    DualTreeTraversalConfig,
    Tree,
    build_interactions_and_neighbors,
    compute_tree_geometry,
    has_fmm_core_topology,
    missing_fmm_core_topology_fields,
)


def _add_jaccpot_repo_to_path() -> None:
    """Allow imports from a sibling local jaccpot checkout."""
    here = pathlib.Path(__file__).resolve()
    candidate = here.parents[2] / "jaccpot"
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def _make_particles(n: int, seed: int) -> tuple[jax.Array, jax.Array]:
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    positions = jax.random.uniform(k1, (n, 3), minval=0.0, maxval=1.0)
    masses = 0.1 + jax.random.uniform(k2, (n,), minval=0.0, maxval=1.0)
    return positions, masses


def _run_tree_pipeline(tree: Tree, label: str) -> dict[str, int]:
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    config = DualTreeTraversalConfig(
        max_interactions_per_node=256,
        max_neighbors_per_leaf=2048,
        max_pair_queue=max(1024, int(8 * tree.num_nodes)),
        process_block=64,
    )
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        traversal_config=config,
    )
    stats = {
        "num_nodes": int(tree.num_nodes),
        "num_leaves": int(tree.num_leaves),
        "num_interaction_edges": int(interactions.sources.shape[0]),
        "num_neighbor_edges": int(neighbors.neighbors.shape[0]),
    }
    print(f"[{label}] stats:", stats)
    return stats


def _run_jaccpot_nearfield_probe(tree: Tree, neighbors) -> None:
    _add_jaccpot_repo_to_path()
    try:
        from jaccpot.nearfield.near_field import (
            compute_leaf_p2p_accelerations,
            prepare_leaf_neighbor_pairs,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print("[jaccpot] import skipped:", repr(exc))
        return

    target_ids, source_ids, valid = prepare_leaf_neighbor_pairs(
        tree.node_ranges,
        neighbors.leaf_indices,
        neighbors.offsets,
        neighbors.neighbors,
        sort_by_source=True,
    )
    print(
        "[jaccpot] nearfield probe:",
        {
            "pairs": int(target_ids.shape[0]),
            "valid_pairs": int(jnp.sum(valid)),
            "source_ids_shape": tuple(source_ids.shape),
        },
    )
    leaf_ranges = tree.node_ranges[neighbors.leaf_indices]
    leaf_counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(jnp.max(leaf_counts))
    acc = compute_leaf_p2p_accelerations(
        tree,
        neighbors,
        tree.positions_sorted,
        tree.masses_sorted,
        max_leaf_size=max_leaf_size,
        nearfield_mode="baseline",
    )
    acc_norm = jnp.linalg.norm(acc)
    print(
        "[jaccpot] nearfield consume:",
        {
            "acc_shape": tuple(acc.shape),
            "acc_norm": float(acc_norm),
            "has_valid_pairs": bool(jnp.any(valid)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=2_000)
    parser.add_argument("--leaf-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("jax:", jax.__version__)
    print("device:", jax.devices()[0])
    print("config:", vars(args))

    positions, masses = _make_particles(args.n_points, args.seed)

    radix = Tree.from_particles(
        positions,
        masses,
        tree_type="radix",
        return_reordered=True,
        leaf_size=args.leaf_size,
    )
    kd = Tree.from_particles(
        positions,
        masses,
        tree_type="kdtree",
        return_reordered=True,
        leaf_size=args.leaf_size,
    )

    for label, tree in (("radix", radix), ("kdtree", kd)):
        missing = missing_fmm_core_topology_fields(tree)
        print(
            f"[{label}] fmm-core:",
            {
                "ok": bool(has_fmm_core_topology(tree)),
                "missing": missing,
            },
        )
        _run_tree_pipeline(tree, label)

    # Re-run under JIT per backend and report tracing blockers without aborting.
    def _jit_dual_tree_run(tree: Tree, label: str):
        try:
            jitted = jax.jit(
                lambda t: build_interactions_and_neighbors(
                    t,
                    compute_tree_geometry(t, t.positions_sorted),
                    theta=0.6,
                    traversal_config=DualTreeTraversalConfig(
                        max_interactions_per_node=256,
                        max_neighbors_per_leaf=2048,
                        max_pair_queue=max(1024, int(8 * t.num_nodes)),
                        process_block=64,
                    ),
                ),
            )
            _, neighbors = jitted(tree)
            print(f"[jit][{label}] dual-tree walk: ok")
            return neighbors
        except Exception as exc:  # pragma: no cover - dependent on backend state
            print(f"[jit][{label}] dual-tree walk: FAILED -> {type(exc).__name__}: {exc}")
            return None

    radix_neighbors = _jit_dual_tree_run(radix, "radix")
    kd_neighbors = _jit_dual_tree_run(kd, "kdtree")

    if radix_neighbors is not None:
        _run_jaccpot_nearfield_probe(radix, radix_neighbors)
    if kd_neighbors is not None:
        _run_jaccpot_nearfield_probe(kd, kd_neighbors)


if __name__ == "__main__":
    main()
