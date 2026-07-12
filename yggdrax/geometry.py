"""Public geometry API for Yggdrax."""

from __future__ import annotations

from beartype import beartype
from jax import core as jax_core
from jaxtyping import Array, jaxtyped

from . import _geometry_impl
from .tree import resolve_tree_topology

TreeGeometry = _geometry_impl.TreeGeometry
LevelMajorTreeGeometry = _geometry_impl.LevelMajorTreeGeometry
_MAX_MORTON_LEVEL = _geometry_impl._MAX_MORTON_LEVEL


@jaxtyped(typechecker=beartype)
def compute_tree_geometry(
    tree: object,
    positions_sorted: Array,
    *,
    max_leaf_size: int | None = None,
) -> TreeGeometry:
    """Compute per-node geometric bounds and helper radii.

    Produces, for every node, its bounding-box center and half-extent, the
    box max half-extent (L-infinity radius), and the bounding-sphere radius,
    from Morton-sorted particle positions.

    Parameters
    ----------
    tree
        Tree container or topology exposing the FMM-core contract.
    positions_sorted
        Particle positions in the tree's Morton order (i.e. indexed by
        ``tree.particle_indices``), shape ``(n_particles, 3)``.
    max_leaf_size
        Optional cap on the temporary leaf-gather buffer. Optional for
        correctness, but important for large JIT-compiled radix trees: it
        bounds the staging shape used during construction and avoids falling
        back to a ``num_particles``-sized buffer under tracing. Defaults to the
        tree's ``leaf_size`` when available.

    Returns
    -------
    TreeGeometry
        Per-node ``center``, ``half_extent``, ``max_extent``, and ``radius``.
    """

    topology = resolve_tree_topology(tree)
    resolved_leaf_cap = max_leaf_size
    if (
        resolved_leaf_cap is None
        and hasattr(topology, "leaf_size")
        and topology.leaf_size is not None
        and not isinstance(topology.leaf_size, jax_core.Tracer)
    ):
        resolved_leaf_cap = int(topology.leaf_size)

    return _geometry_impl.compute_tree_geometry(
        topology,
        positions_sorted,
        max_leaf_size=resolved_leaf_cap,
    )


@jaxtyped(typechecker=beartype)
def geometry_to_level_major(
    tree: object,
    geometry: TreeGeometry,
) -> LevelMajorTreeGeometry:
    """Convert per-node geometry to padded level-major buffers.

    Regroups the flat per-node geometry into ``(num_levels, max_nodes_per_level)``
    padded arrays, so batched kernels can process one tree level at a time.

    Parameters
    ----------
    tree
        Tree container or topology exposing level-order metadata.
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.

    Returns
    -------
    LevelMajorTreeGeometry
        Level-major padded geometry buffers and per-level node counts.
    """

    topology = resolve_tree_topology(tree)
    return _geometry_impl.geometry_to_level_major(topology, geometry)


__all__ = [
    "LevelMajorTreeGeometry",
    "TreeGeometry",
    "_MAX_MORTON_LEVEL",
    "compute_tree_geometry",
    "geometry_to_level_major",
]
