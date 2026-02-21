"""Shared prepared artifact contracts for Yggdrasil consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from jaxtyping import Array

from .geometry import TreeGeometry
from .interactions import DualTreeWalkResult, NodeInteractionList, NodeNeighborList
from .tree import RadixTree


@dataclass(frozen=True)
class TraversalResult:
    """Local yggdrasil view of dual-tree traversal outputs."""

    interaction_offsets: Array
    interaction_sources: Array
    interaction_targets: Array
    interaction_counts: Array
    neighbor_offsets: Array
    neighbor_indices: Array
    neighbor_counts: Array
    leaf_indices: Array
    far_pair_count: Array
    near_pair_count: Array
    queue_overflow: Array
    far_overflow: Array
    near_overflow: Array
    accept_decisions: Array
    near_decisions: Array
    refine_decisions: Array


def traversal_result_from_expanse(result: DualTreeWalkResult) -> TraversalResult:
    """Convert expanse traversal result into yggdrasil local contract."""

    return TraversalResult(
        interaction_offsets=result.interaction_offsets,
        interaction_sources=result.interaction_sources,
        interaction_targets=result.interaction_targets,
        interaction_counts=result.interaction_counts,
        neighbor_offsets=result.neighbor_offsets,
        neighbor_indices=result.neighbor_indices,
        neighbor_counts=result.neighbor_counts,
        leaf_indices=result.leaf_indices,
        far_pair_count=result.far_pair_count,
        near_pair_count=result.near_pair_count,
        queue_overflow=result.queue_overflow,
        far_overflow=result.far_overflow,
        near_overflow=result.near_overflow,
        accept_decisions=result.accept_decisions,
        near_decisions=result.near_decisions,
        refine_decisions=result.refine_decisions,
    )


@dataclass(frozen=True)
class PreparedTreeArtifacts:
    """Canonical prepared tree/traversal bundle."""

    tree: RadixTree
    positions_sorted: Array
    masses_sorted: Array
    inverse_permutation: Array
    geometry: TreeGeometry
    interactions: NodeInteractionList
    neighbors: NodeNeighborList
    traversal_result: Optional[TraversalResult] = None


# Alias used by external planners/docs.
TraversalArtifacts = PreparedTreeArtifacts

__all__ = [
    "PreparedTreeArtifacts",
    "TraversalResult",
    "TraversalArtifacts",
    "traversal_result_from_expanse",
]
