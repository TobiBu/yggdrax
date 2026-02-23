"""Public interaction/traversal API for Yggdrax."""

from __future__ import annotations

from . import _interactions_impl
from ._interactions_impl import (
    DEFAULT_PAIR_QUEUE_MULTIPLIER,
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    _compute_effective_extents,
    _compute_leaf_effective_extents,
    _interaction_capacity_candidates,
    interactions_for_node,
    neighbors_for_leaf,
)
from .geometry import TreeGeometry
from .tree import resolve_tree_topology


def build_well_separated_interactions(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_interactions_per_node: int | None = None,
    mac_type: MACType = "bh",
    *,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger=None,
    dehnen_radius_scale: float = 1.0,
) -> NodeInteractionList:
    """Construct far-field interaction lists from a dual-tree walk."""

    topology = resolve_tree_topology(tree)
    return _interactions_impl.build_well_separated_interactions(
        topology,
        geometry,
        theta=theta,
        max_interactions_per_node=max_interactions_per_node,
        mac_type=mac_type,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        dehnen_radius_scale=dehnen_radius_scale,
    )


def build_leaf_neighbor_lists(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_neighbors_per_leaf: int = _interactions_impl._DEFAULT_MAX_NEIGHBORS,
    max_interactions_per_node: int | None = None,
    mac_type: MACType = "bh",
    *,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger=None,
    dehnen_radius_scale: float = 1.0,
) -> NodeNeighborList:
    """Construct near-field neighbor lists from a dual-tree walk."""

    topology = resolve_tree_topology(tree)
    return _interactions_impl.build_leaf_neighbor_lists(
        topology,
        geometry,
        theta=theta,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        max_interactions_per_node=max_interactions_per_node,
        mac_type=mac_type,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        dehnen_radius_scale=dehnen_radius_scale,
    )


def build_interactions_and_neighbors(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_interactions_per_node: int | None = None,
    max_neighbors_per_leaf: int = _interactions_impl._DEFAULT_MAX_NEIGHBORS,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger=None,
    mac_type: MACType = "bh",
    dehnen_radius_scale: float = 1.0,
    *,
    return_result: bool = False,
    return_grouped: bool = False,
):
    """Construct both far-field interactions and near-field neighbors."""

    topology = resolve_tree_topology(tree)
    return _interactions_impl.build_interactions_and_neighbors(
        topology,
        geometry,
        theta=theta,
        max_interactions_per_node=max_interactions_per_node,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        mac_type=mac_type,
        dehnen_radius_scale=dehnen_radius_scale,
        return_result=return_result,
        return_grouped=return_grouped,
    )


def build_grouped_interactions_from_pairs(
    tree: object,
    geometry: TreeGeometry,
    interaction_sources,
    interaction_targets,
    *,
    level_offsets=None,
):
    """Group interaction pairs by tree level for level-major processing."""

    topology = resolve_tree_topology(tree)
    return _interactions_impl.build_grouped_interactions_from_pairs(
        topology,
        geometry,
        interaction_sources,
        interaction_targets,
        level_offsets=level_offsets,
    )


def diagnose_leaf_neighbor_growth(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    *,
    max_neighbors_per_leaf: int = _interactions_impl._DEFAULT_MAX_NEIGHBORS,
    top_k: int = 10,
    sample_neighbors: int = 20,
) -> dict:
    """Return a compact report of highest per-leaf neighbor counts."""

    topology = resolve_tree_topology(tree)
    return _interactions_impl.diagnose_leaf_neighbor_growth(
        topology,
        geometry,
        theta=theta,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        top_k=top_k,
        sample_neighbors=sample_neighbors,
    )


__all__ = [
    "DEFAULT_PAIR_QUEUE_MULTIPLIER",
    "MACType",
    "DualTreeRetryEvent",
    "DualTreeTraversalConfig",
    "DualTreeWalkResult",
    "NodeInteractionList",
    "NodeNeighborList",
    "build_interactions_and_neighbors",
    "build_leaf_neighbor_lists",
    "build_grouped_interactions_from_pairs",
    "build_well_separated_interactions",
    "diagnose_leaf_neighbor_growth",
    "interactions_for_node",
    "neighbors_for_leaf",
    "_interaction_capacity_candidates",
    "_compute_effective_extents",
    "_compute_leaf_effective_extents",
]
