"""Per-GPU local tree helpers for Yggdrax multi-GPU (Phase 2).

After :mod:`yggdrax.distributed.partition` gives each GPU a contiguous Morton
domain, a local tree is built over it inside ``jax.shard_map``.
:func:`sanitize_padding` neutralises the padding rows first, and
``_build_local_tree`` dispatches the build to the radix, octree or kdtree
backend through un-jitted impls, so it traces inside a manual ``shard_map`` body.

The shared top tree that used to be built here (``distributed_tree_moments``,
``build_local_moments`` and their ``DistributedTreeMoments`` result: every
domain's root moment, all-gathered) had no caller and was removed in the 2026-10
cleanup (Y3).
"""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
from jaxtyping import Array

from .._tree_impl import RadixTree, build_tree

# Backends whose topology satisfies the distributed contract (binary parent/
# child links, compact contiguous ``node_ranges`` tiling, and ``nodes_by_level``)
# that :mod:`~yggdrax.distributed.cross_walk` and the moment stages consume.
# ``octree`` rides the radix binary topology (augmented with explicit cell
# buffers the cross-walk ignores); ``kdtree`` uses the leaf-only bucket build
# (:func:`yggdrax.kdtree.build_leaf_kdtree`), which stores all points in leaves
# and exposes the same contract (the heap KD-tree used by the KNN kernels does
# not, and is not used here).
DISTRIBUTED_TREE_TYPES: tuple[str, ...] = ("radix", "octree", "kdtree")

# What the radix builders return with ``return_reordered=True`` and
# ``return_workspace=False``: (tree, positions_sorted, masses_sorted, inverse).
_ReorderedRadixBuild = tuple[RadixTree, Array, Array, Array]


def _validate_distributed_tree_type(tree_type: str) -> None:
    """Raise a clear error for tree types the distributed path cannot handle."""

    if tree_type not in DISTRIBUTED_TREE_TYPES:
        supported = ", ".join(f"'{t}'" for t in DISTRIBUTED_TREE_TYPES)
        raise ValueError(
            f"Unsupported tree_type '{tree_type}'. Supported: ({supported})."
        )


def _build_local_tree(
    positions: Array,
    masses: Array,
    bounds: tuple[Array, Array],
    *,
    tree_type: str,
    leaf_size: int,
):
    """Backend-dispatched local tree build, safe to call inside ``shard_map``.

    Returns ``(topology, positions_sorted, masses_sorted)``. Only *un-jitted*
    impls are used so the build traces inside a manual ``shard_map`` body (a
    nested top-level ``@jax.jit`` would re-enter with a mesh mismatch). All
    supported backends expose the parent/child/``node_ranges``/``nodes_by_level``
    contract downstream stages rely on: ``radix`` directly, ``octree`` by
    augmenting the radix topology with explicit cell buffers, and ``kdtree`` via
    the leaf-only bucket build (all points in leaves).
    """

    _validate_distributed_tree_type(tree_type)
    if tree_type == "octree":
        # Local imports: avoid importing the public tree module (which pulls in
        # the kdtree/octree backends) at package import time.
        from ..octree import augment_radix_topology_with_octree
        from ..tree import _build_octree_result

        topo, pos_sorted, mass_sorted, _inv = cast(
            _ReorderedRadixBuild,
            _build_octree_result(
                positions,
                masses,
                build_mode="adaptive",
                bounds=bounds,
                return_reordered=True,
                workspace=None,
                return_workspace=False,
                leaf_size=leaf_size,
            ),
        )
        return augment_radix_topology_with_octree(topo), pos_sorted, mass_sorted

    if tree_type == "kdtree":
        from ..dtypes import INDEX_DTYPE
        from ..kdtree import build_leaf_kdtree

        topo = build_leaf_kdtree(positions, leaf_size=leaf_size)
        pidx = jnp.asarray(topo.particle_indices, dtype=INDEX_DTYPE)
        return topo, positions[pidx], masses[pidx]

    tree, pos_sorted, mass_sorted, _inv = cast(
        _ReorderedRadixBuild,
        build_tree(
            positions, masses, bounds, return_reordered=True, leaf_size=leaf_size
        ),
    )
    return tree, pos_sorted, mass_sorted


def sanitize_padding(positions: Array, masses: Array, count: Array):
    """Neutralise padding rows for the tree build.

    Padding masses are zeroed (so they contribute nothing to mass/COM moments)
    and padding positions are replaced by the first valid position (so they do
    not enlarge any node's geometric bounds -- ``compute_tree_geometry`` is not
    mass-aware, so a far-away sentinel would inflate node radii and corrupt MAC
    decisions).
    """

    cap = positions.shape[0]
    valid = jnp.arange(cap) < count
    positions = jnp.where(valid[:, None], positions, positions[0])
    masses = jnp.where(valid, masses, jnp.zeros_like(masses))
    return positions, masses


__all__ = [
    "DISTRIBUTED_TREE_TYPES",
    "sanitize_padding",
]
