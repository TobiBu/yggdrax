"""Mesh drivers that run the LET stages end to end for the tests.

``classify_against_remote`` and ``distributed_let_import`` lived in
:mod:`yggdrax.distributed.let` until the 2026-10 cleanup (Y3). Nothing outside
the tests called them, but they are the only end-to-end checks of the live
stages jaccpot's mesh lane chains together: :func:`build_coarse_frontier`,
:func:`build_remote_coarse_tree`, the cross walk and :func:`import_near_halo`.
So they moved here unchanged, except for two things. The exact-rebalance pass
(``equalize``) left the library with them; nothing the tests assert depends on
it. And the local tree is always radix, the backend jaccpot builds.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from yggdrax.distributed import AXIS_NAME
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl
from yggdrax.distributed.let import (
    _GID_DTYPE,
    _GID_STRIDE,
    build_coarse_frontier,
    build_remote_coarse_tree,
    import_near_halo,
)
from yggdrax.distributed.local_tree import _build_local_tree, sanitize_padding
from yggdrax.distributed.partition import global_bounds, sfc_partition
from yggdrax.dtypes import INDEX_DTYPE, as_index
from yggdrax.geometry import compute_tree_geometry
from yggdrax.tree_moments import compute_tree_mass_moments

try:  # stable across recent JAX versions
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map


@dataclass
class ClassifyMetrics:
    """Global-view diagnostics for the remote classification driver."""

    remote_root_mass: Array  # [ndev] mass of the remote coarse tree
    own_domain_mass: Array  # [ndev] own domain total mass
    total_mass: Array  # [ndev] global total mass (replicated)
    far_count: Array  # [ndev] remote far (M2L) interactions
    near_count: Array  # [ndev] remote near (import) interactions
    overflow: Array  # [ndev] bool: any walk capacity overflow


@dataclass
class ImportMetrics:
    """Global-view diagnostics for the halo-import driver."""

    needed_mass: Array  # [ndev] mass of requested remote leaves
    imported_mass: Array  # [ndev] mass actually received (== needed_mass)
    request_overflow: Array  # [ndev] bool
    wrong_domain: Array  # [ndev] bool: any halo particle sourced from self
    n_halo_valid: Array  # [ndev] imported particle count
    n_mapped: Array  # [ndev] coarse positions mapped to a halo block


def _local_stage(pos, mass, ndev, *, leaf_size, output_capacity, num_samples):
    """Decompose, build the local tree and its frontier, and the remote coarse tree."""
    bounds = global_bounds(pos, axis_name=AXIS_NAME)
    p, m, _c, cnt, _ = sfc_partition(
        pos,
        mass,
        ndev,
        output_capacity=output_capacity,
        bounds=bounds,
        num_samples=num_samples,
        axis_name=AXIS_NAME,
    )
    p, m = sanitize_padding(p, m, cnt)
    tree, pos_sorted, mass_sorted = _build_local_tree(
        p, m, bounds, tree_type="radix", leaf_size=leaf_size
    )
    geom = compute_tree_geometry(tree, pos_sorted, max_leaf_size=leaf_size)
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)
    fr = build_coarse_frontier(
        tree,
        moments.mass,
        moments.center_of_mass,
        positions_sorted=pos_sorted,
        max_leaf_size=leaf_size,
    )
    rct = build_remote_coarse_tree(fr, ndev, bounds=bounds, axis_name=AXIS_NAME)
    return tree, geom, moments, pos_sorted, mass_sorted, rct


def classify_against_remote(
    mesh,
    positions,
    masses,
    *,
    leaf_size,
    output_capacity,
    theta=0.5,
    mac_type="bh",
    max_interactions_per_node=256,
    max_neighbors_per_leaf=256,
    max_pair_queue=8192,
    num_samples=8,
):
    """Cross-walk each GPU's local tree against the remote-only coarse tree."""
    ndev = mesh.size

    def fn(pos, mass):
        tree, geom, moments, _, _, rct = _local_stage(
            pos,
            mass,
            ndev,
            leaf_size=leaf_size,
            output_capacity=output_capacity,
            num_samples=num_samples,
        )
        root = as_index(jnp.argmin(jnp.asarray(tree.parent)))
        own_mass = moments.mass[root]
        rroot = as_index(jnp.argmin(jnp.asarray(rct.tree.parent)))
        res = dual_tree_walk_cross_impl(
            tree,
            geom,
            rct.tree,
            rct.geometry,
            theta,
            mac_type=mac_type,
            max_interactions_per_node=max_interactions_per_node,
            max_neighbors_per_leaf=max_neighbors_per_leaf,
            max_pair_queue=max_pair_queue,
        )
        total = jax.lax.psum(own_mass, AXIS_NAME)
        overflow = res.queue_overflow | res.far_overflow | res.near_overflow
        return (
            rct.moments.mass[rroot][None],
            own_mass[None],
            total[None],
            res.far_pair_count[None],
            res.near_pair_count[None],
            overflow[None],
        )

    outs = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME),) * 6,
        check_vma=False,
    )(positions, masses)
    return ClassifyMetrics(*outs)


def distributed_let_import(
    mesh,
    positions,
    masses,
    *,
    leaf_size,
    output_capacity,
    theta=0.5,
    mac_type="bh",
    max_interactions_per_node=256,
    max_neighbors_per_leaf=256,
    max_pair_queue=8192,
    num_samples=8,
):
    """Full LET pipeline through the halo import (decompose -> classify -> import)."""
    ndev = mesh.size
    n_leaf = (output_capacity + leaf_size - 1) // leaf_size
    max_req = (ndev - 1) * n_leaf
    max_recv = (ndev - 1) * n_leaf

    def fn(pos, mass):
        tree, geom, _, pos_sorted, mass_sorted, rct = _local_stage(
            pos,
            mass,
            ndev,
            leaf_size=leaf_size,
            output_capacity=output_capacity,
            num_samples=num_samples,
        )
        res = dual_tree_walk_cross_impl(
            tree,
            geom,
            rct.tree,
            rct.geometry,
            theta,
            mac_type=mac_type,
            max_interactions_per_node=max_interactions_per_node,
            max_neighbors_per_leaf=max_neighbors_per_leaf,
            max_pair_queue=max_pair_queue,
        )
        halo = import_near_halo(
            rct,
            res,
            pos_sorted,
            mass_sorted,
            ndev,
            leaf_size=leaf_size,
            max_req_leaves=max_req,
            max_recv_leaves=max_recv,
            axis_name=AXIS_NAME,
        )
        me = jax.lax.axis_index(AXIS_NAME)
        halo_domain = jnp.where(
            halo.valid,
            jnp.asarray(halo.gid, _GID_DTYPE) // jnp.asarray(_GID_STRIDE, _GID_DTYPE),
            jnp.asarray(-1, _GID_DTYPE),
        )
        wrong = jnp.any(halo.valid & (halo_domain == me))
        return (
            halo.needed_mass[None],
            halo.imported_mass[None],
            halo.request_overflow[None],
            wrong[None],
            jnp.sum(halo.valid.astype(INDEX_DTYPE))[None],
            jnp.sum((halo.coarse_to_halo >= 0).astype(INDEX_DTYPE))[None],
        )

    outs = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME),) * 6,
        check_vma=False,
    )(positions, masses)
    return ImportMetrics(*outs)
