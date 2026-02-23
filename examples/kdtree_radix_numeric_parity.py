"""Numeric parity check for Radix vs KD downstream nearfield outputs.

This script compares backend outputs for one representative downstream FMM
component (jaccpot nearfield P2P) and reports consistency/error metrics.
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
)


def _add_jaccpot_repo_to_path() -> None:
    here = pathlib.Path(__file__).resolve()
    candidate = here.parents[2] / "jaccpot"
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def _make_problem(n: int, seed: int) -> tuple[jax.Array, jax.Array]:
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    positions = jax.random.uniform(k1, (n, 3), minval=-1.0, maxval=1.0)
    masses = 0.2 + jax.random.uniform(k2, (n,), minval=0.0, maxval=1.0)
    return positions, masses


def _dense_reference_acc(
    positions: jax.Array,
    masses: jax.Array,
    *,
    softening: float,
    G: float,
) -> jax.Array:
    delta = positions[:, None, :] - positions[None, :, :]
    r2 = jnp.sum(delta * delta, axis=-1) + (softening * softening)
    n = positions.shape[0]
    mask = ~jnp.eye(n, dtype=bool)
    inv_r3 = jnp.where(mask, jax.lax.rsqrt(r2) / r2, 0.0)
    weights = G * masses[None, :] * inv_r3
    return -jnp.sum(weights[:, :, None] * delta, axis=1)


def _restore_original_order(
    values_sorted: jax.Array,
    inverse_permutation: jax.Array,
    *,
    positions_sorted: jax.Array,
    positions_original: jax.Array,
) -> jax.Array:
    # yggdrax currently exposes inverse permutation with backend-dependent
    # conventions in a few paths. Pick the mapping that reconstructs positions.
    cand_scatter = (
        jnp.zeros_like(values_sorted).at[inverse_permutation].set(values_sorted)
    )
    pos_scatter = (
        jnp.zeros_like(positions_sorted).at[inverse_permutation].set(positions_sorted)
    )
    err_scatter = jnp.linalg.norm(pos_scatter - positions_original)

    cand_gather = values_sorted[inverse_permutation]
    pos_gather = positions_sorted[inverse_permutation]
    err_gather = jnp.linalg.norm(pos_gather - positions_original)

    use_scatter = err_scatter <= err_gather
    return jnp.where(use_scatter, cand_scatter, cand_gather)


def _build_artifacts(
    positions: jax.Array,
    masses: jax.Array,
    *,
    tree_type: str,
    leaf_size: int,
    theta: float,
    mac_type: str,
) -> tuple[Tree, object]:
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type=tree_type,
        return_reordered=True,
        leaf_size=leaf_size,
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    cfg = DualTreeTraversalConfig(
        max_interactions_per_node=512,
        max_neighbors_per_leaf=4096,
        max_pair_queue=max(4096, int(16 * tree.num_nodes)),
        process_block=64,
    )
    _, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=theta,
        mac_type=mac_type,  # type: ignore[arg-type]
        traversal_config=cfg,
    )
    return tree, neighbors


def _nearfield_acc_original(
    tree: Tree,
    neighbors,
    *,
    positions_original: jax.Array,
    softening: float,
    G: float,
) -> jax.Array:
    from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations

    leaf_ranges = tree.node_ranges[neighbors.leaf_indices]
    leaf_counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(jnp.max(leaf_counts)) if int(leaf_counts.shape[0]) > 0 else 1

    acc_sorted = compute_leaf_p2p_accelerations(
        tree,
        neighbors,
        tree.positions_sorted,
        tree.masses_sorted,
        softening=softening,
        G=G,
        max_leaf_size=max_leaf_size,
        nearfield_mode="baseline",
    )
    return _restore_original_order(
        acc_sorted,
        tree.inverse_permutation,
        positions_sorted=tree.positions_sorted,
        positions_original=positions_original,
    )


def _rel_l2(err: jax.Array, ref: jax.Array) -> float:
    num = float(jnp.linalg.norm(err))
    den = float(jnp.linalg.norm(ref)) + 1e-12
    return num / den


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--leaf-size", type=int, default=32)
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--mac-type", type=str, default="dehnen")
    parser.add_argument("--softening", type=float, default=1e-3)
    parser.add_argument("--G", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _add_jaccpot_repo_to_path()
    from jaccpot.nearfield.near_field import prepare_leaf_neighbor_pairs

    positions, masses = _make_problem(args.n_points, args.seed)
    a_dense = _dense_reference_acc(
        positions,
        masses,
        softening=float(args.softening),
        G=float(args.G),
    )

    radix_tree, radix_neighbors = _build_artifacts(
        positions,
        masses,
        tree_type="radix",
        leaf_size=args.leaf_size,
        theta=args.theta,
        mac_type=args.mac_type,
    )
    kd_tree, kd_neighbors = _build_artifacts(
        positions,
        masses,
        tree_type="kdtree",
        leaf_size=args.leaf_size,
        theta=args.theta,
        mac_type=args.mac_type,
    )

    # downstream consume step
    a_radix = _nearfield_acc_original(
        radix_tree,
        radix_neighbors,
        positions_original=positions,
        softening=float(args.softening),
        G=float(args.G),
    )
    a_kd = _nearfield_acc_original(
        kd_tree,
        kd_neighbors,
        positions_original=positions,
        softening=float(args.softening),
        G=float(args.G),
    )

    rt, rs, rv = prepare_leaf_neighbor_pairs(
        radix_tree.node_ranges,
        radix_neighbors.leaf_indices,
        radix_neighbors.offsets,
        radix_neighbors.neighbors,
    )
    kt, ks, kv = prepare_leaf_neighbor_pairs(
        kd_tree.node_ranges,
        kd_neighbors.leaf_indices,
        kd_neighbors.offsets,
        kd_neighbors.neighbors,
    )

    print("jax:", jax.__version__)
    print("device:", jax.devices()[0])
    print("config:", vars(args))
    print(
        "radix_nearfield_edges:",
        {
            "pairs": int(rt.shape[0]),
            "valid_pairs": int(jnp.sum(rv)),
            "nodes": int(radix_tree.num_nodes),
        },
    )
    print(
        "kdtree_nearfield_edges:",
        {
            "pairs": int(kt.shape[0]),
            "valid_pairs": int(jnp.sum(kv)),
            "nodes": int(kd_tree.num_nodes),
        },
    )
    print(
        "nearfield_error_vs_dense:",
        {
            "radix_rel_l2": _rel_l2(a_radix - a_dense, a_dense),
            "kdtree_rel_l2": _rel_l2(a_kd - a_dense, a_dense),
        },
    )
    print(
        "radix_vs_kdtree:",
        {
            "rel_l2_gap_wrt_dense_norm": _rel_l2(a_kd - a_radix, a_dense),
            "max_abs_diff": float(jnp.max(jnp.abs(a_kd - a_radix))),
            "cosine_similarity": float(
                jnp.vdot(a_radix.reshape(-1), a_kd.reshape(-1))
                / ((jnp.linalg.norm(a_radix) * jnp.linalg.norm(a_kd)) + 1e-12)
            ),
        },
    )


if __name__ == "__main__":
    main()
