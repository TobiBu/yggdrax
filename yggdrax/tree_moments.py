"""Mass moments for radix tree nodes.

This module accumulates per-node total masses and centres of mass from
particles that were reordered into Morton order by :mod:`yggdrax.tree`.
These summaries are useful when building multipole expansions during the
upward FMM sweep.

The Cartesian multipole moments that used to live here
(``compute_tree_multipole_moments``, ``TreeMultipoleMoments``, the packing and
M2M translation helpers) and ``compute_tree_mass_moments_jit`` were removed in
the 2026-10 cleanup (Y2): no consumer used them once jaccpot dropped its
Cartesian basis.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .dtypes import INDEX_DTYPE, as_index
from .tree import require_fmm_core_topology, resolve_tree_topology


class TreeMassMoments(NamedTuple):
    """Aggregate mass information for every node in a radix tree."""

    mass: Array
    center_of_mass: Array


def _validate_inputs(
    tree: object,
    positions_sorted: Array,
    masses_sorted: Array,
) -> None:
    topology = resolve_tree_topology(tree)
    require_fmm_core_topology(topology)

    total_nodes = topology.parent.shape[0]
    if topology.node_ranges.shape[0] != total_nodes:
        raise ValueError("tree.node_ranges must align with tree.parent shape")
    try:
        num_particles = int(jnp.asarray(topology.num_particles))
    except jax.errors.ConcretizationTypeError:
        # Under outer jit, tree.num_particles can be traced. Fall back to
        # shape-based validation, which remains static and trace-safe.
        num_particles = int(positions_sorted.shape[0])
    if positions_sorted.shape[0] != num_particles:
        raise ValueError("positions_sorted must match tree.num_particles")
    if masses_sorted.shape[0] != num_particles:
        raise ValueError("masses_sorted must match tree.num_particles")
    if positions_sorted.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")


def _accumulation_dtype(dtype: Any) -> Any:
    """Wide dtype for prefix sums whose differences are range queries.

    A node's sum is the DIFFERENCE of two prefixes that grow to the total, while a
    small leaf holds ~leaf/N of it. In float32 that cancellation is fatal at large N:
    at N = 3.2e7 with masses 1/N a 20-particle leaf's mass is 6e-7 against a
    prefix's absolute error of ~6e-8, so its centre of mass came out ~0.1 units off
    -- twenty times the width of a core leaf. The error grows with N. Measured on the
    fused jaccpot lane (clipped Plummer, 3.2e7, one A100): the MAC radii about those
    centres inflated the near list from 35.8M to 515.9M directed pairs and the force
    from 200 to 774 ms. With x64 on (yggdrax enables it at import) the prefixes are
    accumulated in float64.

    Parameters
    ----------
    dtype
        The working dtype.

    Returns
    -------
    Any
        ``float64`` when x64 is enabled and ``dtype`` is floating, else ``dtype``.
    """
    wide = jax.dtypes.canonicalize_dtype(jnp.float64)
    if wide == jnp.float64 and jnp.issubdtype(dtype, jnp.floating):
        return jnp.float64
    return dtype


def _prefix_range_sums(values: Array, starts: Array, ends: Array) -> Array:
    """Per-node sums of ``values[start:end]`` from one prefix sum (wide accumulation).

    Parameters
    ----------
    values
        ``(n, ...)`` values in particle order, already in the accumulation dtype.
    starts
        ``(nodes,)`` inclusive range starts.
    ends
        ``(nodes,)`` exclusive range ends.

    Returns
    -------
    Array
        ``(nodes, ...)`` range sums, in ``values``' dtype.
    """
    pad = jnp.zeros((1,) + tuple(values.shape[1:]), dtype=values.dtype)
    prefix = jnp.concatenate([pad, jnp.cumsum(values, axis=0)], axis=0)
    return prefix[ends] - prefix[starts]


def compute_tree_mass_moments(
    tree: object,
    positions_sorted: Array,
    masses_sorted: Array,
) -> TreeMassMoments:
    """Compute total mass and center of mass for every tree node.

    Uses prefix sums over the Morton-sorted masses and mass-weighted positions
    for O(1)-per-node range queries.

    Parameters
    ----------
    tree
        Tree container or topology exposing the FMM-core contract.
    positions_sorted
        Particle positions in Morton order, shape ``(n_particles, 3)``.
    masses_sorted
        Particle masses reordered identically to ``positions_sorted``.

    Returns
    -------
    TreeMassMoments
        Per-node total mass and center of mass.
    """

    topology = resolve_tree_topology(tree)
    require_fmm_core_topology(topology)
    _validate_inputs(topology, positions_sorted, masses_sorted)

    ranges = topology.node_ranges.astype(INDEX_DTYPE)
    starts = ranges[:, 0]
    ends = ranges[:, 1] + as_index(1)  # make end exclusive for prefix sums

    # Prefix sums over masses and mass-weighted positions give O(1) range queries
    # per node, accumulated wide (see `_accumulation_dtype`).
    acc = _accumulation_dtype(positions_sorted.dtype)
    masses_acc = masses_sorted.astype(acc)
    total_mass = _prefix_range_sums(masses_acc, starts, ends)
    moment_sum = _prefix_range_sums(
        masses_acc[:, None] * positions_sorted.astype(acc), starts, ends
    )

    safe_mass = jnp.where(
        total_mass == 0,
        jnp.ones_like(total_mass),
        total_mass,
    )
    center = moment_sum / safe_mass[:, None]
    center = jnp.where(total_mass[:, None] == 0, 0.0, center)

    return TreeMassMoments(
        mass=total_mass.astype(masses_sorted.dtype),
        center_of_mass=center.astype(positions_sorted.dtype),
    )


__all__ = [
    "TreeMassMoments",
    "compute_tree_mass_moments",
]
