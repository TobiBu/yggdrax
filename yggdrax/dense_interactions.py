"""Dense level-major interaction buffers for the downward sweep."""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jax import lax
from jaxtyping import Array, jaxtyped

from .dtypes import INDEX_DTYPE, as_index
from .geometry import LevelMajorTreeGeometry, TreeGeometry, geometry_to_level_major
from .interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    NodeInteractionList,
    build_interactions_and_neighbors,
)
from .tree import RadixTree


class DenseInteractionBuffers(NamedTuple):
    """Dense level-major representation of far-field interaction pairs."""

    geometry: LevelMajorTreeGeometry
    m2l_sources: Array
    m2l_displacements: Array
    m2l_mask: Array
    m2l_counts: Array
    sparse_interactions: NodeInteractionList


def _dense_m2l_buffers(
    level_nodes: Array,
    level_centers: Array,
    geometry_centers: Array,
    interactions: NodeInteractionList,
) -> tuple[Array, Array, Array, Array]:
    num_levels, max_nodes = level_nodes.shape
    total_slots = int(num_levels * max_nodes)
    interaction_counts = jnp.asarray(interactions.counts, dtype=INDEX_DTYPE)
    interaction_offsets = jnp.asarray(interactions.offsets, dtype=INDEX_DTYPE)
    source_pool = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
    if source_pool.size == 0:
        source_pool = jnp.zeros((1,), dtype=INDEX_DTYPE)
    max_interactions = (
        int(jnp.max(interaction_counts)) if interaction_counts.size else 0
    )
    max_interactions = max(1, max_interactions)
    interaction_range = jnp.arange(max_interactions, dtype=INDEX_DTYPE)

    sources_tensor = jnp.full(
        (num_levels, max_nodes, max_interactions),
        as_index(-1),
        dtype=INDEX_DTYPE,
    )
    mask_tensor = jnp.zeros(
        (num_levels, max_nodes, max_interactions),
        dtype=jnp.bool_,
    )
    counts_tensor = jnp.zeros((num_levels, max_nodes), dtype=INDEX_DTYPE)
    disp_tensor = jnp.zeros(
        (num_levels, max_nodes, max_interactions, geometry_centers.shape[1]),
        dtype=geometry_centers.dtype,
    )

    flat_nodes = level_nodes.reshape((total_slots,))
    level_grid = jnp.broadcast_to(
        jnp.arange(num_levels, dtype=INDEX_DTYPE)[:, None],
        (num_levels, max_nodes),
    ).reshape((total_slots,))
    slot_grid = jnp.broadcast_to(
        jnp.arange(max_nodes, dtype=INDEX_DTYPE)[None, :],
        (num_levels, max_nodes),
    ).reshape((total_slots,))

    def body(idx, state):
        sources_arr, mask_arr, counts_arr, disp_arr = state
        node = lax.dynamic_index_in_dim(
            flat_nodes,
            idx,
            axis=0,
            keepdims=False,
        )
        level_idx = lax.dynamic_index_in_dim(
            level_grid,
            idx,
            axis=0,
            keepdims=False,
        )
        slot_idx = lax.dynamic_index_in_dim(
            slot_grid,
            idx,
            axis=0,
            keepdims=False,
        )

        def update(values):
            src_arr, m_arr, c_arr, d_arr = values
            count = lax.dynamic_index_in_dim(
                interaction_counts, node, axis=0, keepdims=False
            )
            start = lax.dynamic_index_in_dim(
                interaction_offsets, node, axis=0, keepdims=False
            )
            row_indices = start + interaction_range
            valid = interaction_range < count
            safe_idx = jnp.where(valid, row_indices, as_index(0))
            source_nodes = jnp.where(
                valid,
                source_pool[safe_idx],
                as_index(-1),
            )

            src_arr = src_arr.at[level_idx, slot_idx].set(source_nodes)
            m_arr = m_arr.at[level_idx, slot_idx].set(valid)
            c_arr = c_arr.at[level_idx, slot_idx].set(count)

            target_center = level_centers[level_idx, slot_idx]
            total_nodes = geometry_centers.shape[0]
            safe_sources = jnp.clip(
                source_nodes,
                min=0,
                max=total_nodes - 1,
            )
            source_centers = geometry_centers[safe_sources]
            displacement = target_center - source_centers
            displacement = jnp.where(valid[..., None], displacement, 0.0)
            d_arr = d_arr.at[level_idx, slot_idx].set(displacement)
            return src_arr, m_arr, c_arr, d_arr

        return lax.cond(
            node >= 0,
            update,
            lambda vals: vals,
            (sources_arr, mask_arr, counts_arr, disp_arr),
        )

    result = lax.fori_loop(
        0,
        total_slots,
        body,
        (sources_tensor, mask_tensor, counts_tensor, disp_tensor),
    )
    return result


@jaxtyped(typechecker=beartype)
def densify_interactions(
    tree: RadixTree,
    geometry: TreeGeometry,
    interactions: NodeInteractionList,
) -> DenseInteractionBuffers:
    """Convert sparse far-field interactions into dense level-major tensors."""

    level_geometry = geometry_to_level_major(tree, geometry)
    level_nodes = level_geometry.node_indices
    level_centers = level_geometry.centers
    geometry_centers = jnp.asarray(geometry.center)

    m2l_sources, m2l_mask, m2l_counts, m2l_disp = _dense_m2l_buffers(
        level_nodes,
        level_centers,
        geometry_centers,
        interactions,
    )

    return DenseInteractionBuffers(
        geometry=level_geometry,
        m2l_sources=m2l_sources,
        m2l_displacements=m2l_disp,
        m2l_mask=m2l_mask,
        m2l_counts=m2l_counts,
        sparse_interactions=interactions,
    )


@jaxtyped(typechecker=beartype)
def build_dense_interactions(
    tree: RadixTree,
    geometry: TreeGeometry,
    theta: float = 0.5,
    *,
    max_pair_queue: Optional[int] = None,
    process_block: Optional[int] = None,
    traversal_config: Optional[DualTreeTraversalConfig] = None,
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
) -> DenseInteractionBuffers:
    """Build sparse far-field interactions and emit their dense view."""

    interactions, _neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=theta,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
    )
    return densify_interactions(tree, geometry, interactions)


__all__ = [
    "DenseInteractionBuffers",
    "build_dense_interactions",
    "densify_interactions",
]
