"""Prepared tree artifact construction for Yggdrasil."""

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
from .tree import build_tree
from .types import PreparedTreeArtifacts, traversal_result_from_expanse


@jaxtyped(typechecker=beartype)
def build_prepared_tree_artifacts(
    positions: Array,
    masses: Array,
    bounds: Optional[tuple[Array, Array]] = None,
    *,
    leaf_size: int = 16,
    theta: float = 0.6,
    mac_type: MACType = "bh",
    traversal_config: Optional[DualTreeTraversalConfig] = None,
    dehnen_radius_scale: float = 1.0,
) -> PreparedTreeArtifacts:
    """Build a prepared tree + traversal bundle with stable public fields."""

    bounds_resolved = infer_bounds(positions) if bounds is None else bounds
    tree, pos_sorted, mass_sorted, inverse = build_tree(
        positions,
        masses,
        bounds_resolved,
        leaf_size=int(leaf_size),
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    interactions, neighbors, traversal_result_raw = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=float(theta),
        traversal_config=traversal_config,
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
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        inverse_permutation=jnp.asarray(inverse, dtype=INDEX_DTYPE),
        geometry=geometry,
        interactions=interactions,
        neighbors=neighbors,
        traversal_result=traversal_result,
    )


__all__ = ["build_prepared_tree_artifacts"]
