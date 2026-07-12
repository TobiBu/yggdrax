"""Prepared tree artifact construction for Yggdrax."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped

from .bounds import infer_bounds
from .dtypes import INDEX_DTYPE
from .geometry import compute_tree_geometry
from .interactions import (
    DualTreeTraversalConfig,
    MACType,
    build_interactions_and_neighbors,
)
from .tree import Tree, TreeType
from .types import PreparedTreeArtifacts, traversal_result_from_expanse


@jaxtyped(typechecker=beartype)
def build_prepared_tree_artifacts(
    positions: Array,
    masses: Array,
    bounds: Optional[tuple[Array, Array]] = None,
    *,
    tree_type: TreeType = "radix",
    leaf_size: int = 16,
    theta: float = 0.6,
    mac_type: MACType = "bh",
    traversal_config: Optional[DualTreeTraversalConfig] = None,
    dehnen_radius_scale: float = 1.0,
) -> PreparedTreeArtifacts:
    """Build a tree, geometry, and interactions in one call.

    End-to-end convenience: reorders particles into a tree, computes per-node
    geometry, and runs the dual-tree walk, returning everything a downstream
    solver needs in a single bundle with stable public field names. A
    KD-tree-tuned traversal configuration is chosen automatically when
    ``traversal_config`` is omitted and ``tree_type == "kdtree"``.

    Parameters
    ----------
    positions
        Particle positions of shape ``(n, 3)``.
    masses
        Particle masses of shape ``(n,)``.
    bounds
        Optional ``(min_corner, max_corner)`` box; inferred from ``positions``
        when omitted.
    tree_type
        Backend to build: ``"radix"``, ``"octree"``, or ``"kdtree"``.
    leaf_size
        Maximum particles per leaf.
    theta
        Opening-angle parameter of the multipole acceptance criterion.
    mac_type
        MAC variant: ``"bh"``, ``"dehnen"``, or ``"engblom"``.
    traversal_config
        Optional traversal capacity/queue settings; a KD-tree-tuned default is
        used when omitted for KD-trees.
    dehnen_radius_scale
        Effective-radius scale applied for the Dehnen MAC.

    Returns
    -------
    PreparedTreeArtifacts
        Bundle of the tree, reordered particle buffers, inverse permutation,
        geometry, far-field interactions, near-field neighbors, and the raw
        traversal result.

    Raises
    ------
    RuntimeError
        If the tree build did not return reordered particle buffers.
    """

    bounds_resolved = infer_bounds(positions) if bounds is None else bounds
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type=tree_type,
        bounds=bounds_resolved,
        return_reordered=True,
        leaf_size=int(leaf_size),
    )
    if (
        tree.positions_sorted is None
        or tree.masses_sorted is None
        or tree.inverse_permutation is None
    ):
        raise RuntimeError("tree build did not return reordered particle buffers")

    traversal_cfg = traversal_config
    if traversal_cfg is None and getattr(tree, "tree_type", "radix") == "kdtree":
        traversal_cfg = DualTreeTraversalConfig(
            max_interactions_per_node=512,
            max_neighbors_per_leaf=2048,
            max_pair_queue=max(4096, int(16 * tree.num_nodes)),
            process_block=64,
        )

    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    interactions, neighbors, traversal_result_raw = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=float(theta),
        traversal_config=traversal_cfg,
        mac_type=mac_type,
        dehnen_radius_scale=float(dehnen_radius_scale),
        return_result=True,
    )
    traversal_result = (
        traversal_result_from_expanse(traversal_result_raw)
        if traversal_result_raw is not None
        else None
    )
    return PreparedTreeArtifacts(
        tree=tree,
        positions_sorted=tree.positions_sorted,
        masses_sorted=tree.masses_sorted,
        inverse_permutation=jnp.asarray(tree.inverse_permutation, dtype=INDEX_DTYPE),
        geometry=geometry,
        interactions=interactions,
        neighbors=neighbors,
        traversal_result=traversal_result,
    )


__all__ = ["build_prepared_tree_artifacts"]
