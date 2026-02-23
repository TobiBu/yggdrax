"""Grouped interaction utilities for class-major far-field execution."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .dtypes import INDEX_DTYPE
from .geometry import TreeGeometry
from .tree import get_node_levels, get_nodes_by_level


class GroupedInteractionBuffers(NamedTuple):
    """Class-major grouped representation of far-field interactions."""

    class_keys: jnp.ndarray  # (num_classes, 5) int32
    class_displacements: jnp.ndarray  # (num_classes, 3)
    class_offsets: jnp.ndarray  # (num_classes + 1,)
    class_sources: jnp.ndarray  # (num_pairs,)
    class_targets: jnp.ndarray  # (num_pairs,)
    class_ids: jnp.ndarray  # (num_pairs,)
    level_offsets: jnp.ndarray  # pass-through for diagnostics
    level_nodes: jnp.ndarray  # tree.nodes_by_level


def _safe_cell_size(root_extent: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Return per-axis cell size for each level."""
    denom = np.power(2.0, level.astype(np.float64))[:, None]
    extent = np.asarray(root_extent, dtype=np.float64)[None, :]
    return np.maximum(extent / np.maximum(denom, 1.0), 1e-12)


def build_grouped_interactions(
    tree: object,
    geometry: TreeGeometry,
    interactions,
) -> GroupedInteractionBuffers:
    """Group sparse far-field pairs into displacement classes."""
    return build_grouped_interactions_from_pairs(
        tree,
        geometry,
        interactions.sources,
        interactions.targets,
        level_offsets=getattr(interactions, "level_offsets", None),
    )


def build_grouped_interactions_from_pairs(
    tree: object,
    geometry: TreeGeometry,
    sources: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    level_offsets: jnp.ndarray | None = None,
) -> GroupedInteractionBuffers:
    """Group far-field source/target arrays into displacement classes."""

    src = np.asarray(jax.device_get(sources), dtype=np.int64)
    tgt = np.asarray(jax.device_get(targets), dtype=np.int64)
    if src.size == 0:
        empty_i32 = jnp.zeros((0, 5), dtype=jnp.int32)
        empty_f = jnp.zeros((0, 3), dtype=geometry.center.dtype)
        empty_i = jnp.zeros((0,), dtype=INDEX_DTYPE)
        if level_offsets is None:
            level_offsets_out = jnp.zeros((1,), dtype=INDEX_DTYPE)
        else:
            level_offsets_out = jnp.asarray(level_offsets, dtype=INDEX_DTYPE)
        return GroupedInteractionBuffers(
            class_keys=empty_i32,
            class_displacements=empty_f,
            class_offsets=jnp.zeros((1,), dtype=INDEX_DTYPE),
            class_sources=empty_i,
            class_targets=empty_i,
            class_ids=empty_i,
            level_offsets=level_offsets_out,
            level_nodes=get_nodes_by_level(tree),
        )

    centers = np.asarray(jax.device_get(geometry.center), dtype=np.float64)
    levels = np.asarray(jax.device_get(get_node_levels(tree)), dtype=np.int32)

    delta = centers[tgt] - centers[src]
    tgt_level = levels[tgt]
    src_level = levels[src]

    bounds_min = np.asarray(jax.device_get(tree.bounds_min), dtype=np.float64)
    bounds_max = np.asarray(jax.device_get(tree.bounds_max), dtype=np.float64)
    if bounds_min.ndim == 1:
        root_extent = bounds_max - bounds_min
    else:
        root_extent = bounds_max[0] - bounds_min[0]
    cell_size = _safe_cell_size(root_extent, tgt_level)
    disp_idx = np.rint(delta / cell_size).astype(np.int32)

    keys = np.stack(
        [
            tgt_level.astype(np.int32),
            src_level.astype(np.int32),
            disp_idx[:, 0],
            disp_idx[:, 1],
            disp_idx[:, 2],
        ],
        axis=1,
    )

    order = np.lexsort((keys[:, 4], keys[:, 3], keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    src_sorted = src[order]
    tgt_sorted = tgt[order]

    boundary = np.empty((keys_sorted.shape[0],), dtype=np.bool_)
    boundary[0] = True
    if keys_sorted.shape[0] > 1:
        boundary[1:] = np.any(keys_sorted[1:] != keys_sorted[:-1], axis=1)
    class_starts = np.nonzero(boundary)[0].astype(np.int64)
    class_offsets = np.concatenate(
        [class_starts, np.array([keys_sorted.shape[0]], dtype=np.int64)],
        axis=0,
    )
    class_ids_sorted = np.searchsorted(class_starts, np.arange(keys_sorted.shape[0]))

    class_keys = keys_sorted[class_starts]
    class_disp = np.zeros((class_keys.shape[0], 3), dtype=np.float64)
    for class_i, key in enumerate(class_keys):
        tgt_lev = int(key[0])
        disp = key[2:5].astype(np.float64)
        class_cell = _safe_cell_size(
            root_extent, np.asarray([tgt_lev], dtype=np.int32)
        )[0]
        class_disp[class_i] = disp * class_cell

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.shape[0], dtype=order.dtype)
    class_ids_original = class_ids_sorted[inv_order]

    return GroupedInteractionBuffers(
        class_keys=jnp.asarray(class_keys, dtype=jnp.int32),
        class_displacements=jnp.asarray(class_disp, dtype=geometry.center.dtype),
        class_offsets=jnp.asarray(class_offsets, dtype=INDEX_DTYPE),
        class_sources=jnp.asarray(src_sorted, dtype=INDEX_DTYPE),
        class_targets=jnp.asarray(tgt_sorted, dtype=INDEX_DTYPE),
        class_ids=jnp.asarray(class_ids_original, dtype=INDEX_DTYPE),
        level_offsets=(
            jnp.asarray(level_offsets, dtype=INDEX_DTYPE)
            if level_offsets is not None
            else jnp.zeros((1,), dtype=INDEX_DTYPE)
        ),
        level_nodes=get_nodes_by_level(tree),
    )


__all__ = [
    "GroupedInteractionBuffers",
    "build_grouped_interactions",
    "build_grouped_interactions_from_pairs",
]
