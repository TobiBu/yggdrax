"""Locally Essential Tree construction for Yggdrax multi-GPU (Phase 3).

The enriched shared top tree (this module) is the first stage of the
"top-tree cross-walk request" LET protocol:

1. each GPU extracts a **disjoint, mass-conserving frontier** of its local tree
   (truncate at ``level``: nodes at that level, plus shallower leaves) -- an
   antichain that partitions the domain's particles;
2. every GPU ``all_gather``s all domains' frontiers and builds an identical
   **global coarse tree** over the frontier centres-of-mass (reusing the
   single-device ``build_tree``), tagging each coarse leaf with its origin
   ``(domain, local_node, particle_range)``.

A local target tree can then be cross-walked (``cross_walk.dual_tree_walk_cross``)
against this global coarse tree: far coarse nodes drive M2L; near coarse leaves
name the remote subtrees whose particles must be imported (stage 3b/3c, later).

Coarse resolution is tunable via ``level``/``max_frontier`` -- finer frontiers
shrink what "near" pulls in, trading communication for the coarse M2L radius.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._tree_impl import build_tree
from ..dtypes import INDEX_DTYPE, as_index
from ..geometry import compute_tree_geometry
from ..tree_moments import compute_tree_mass_moments
from .sharding import AXIS_NAME


@dataclass
class CoarseFrontier:
    """One domain's disjoint frontier: its local tree's leaf nodes.

    The leaves of a tree always partition its particles exactly (every particle
    is in exactly one leaf), so leaf masses sum to the domain total -- robust
    even when equalize padding duplicates real positions into degenerate
    Morton-code clusters (which breaks a level-truncation antichain). Leaf
    granularity is the natural coarse resolution; it can be coarsened later by a
    leaf-up level cut without changing the force path.
    """

    mass: Array          # [num_leaves]
    com: Array           # [num_leaves, 3]
    node_range: Array    # [num_leaves, 2] particle range in the local tree
    node_id: Array       # [num_leaves] local node id (-1 = empty/padding leaf)


def build_coarse_frontier(
    tree: object,
    node_mass: Array,
    node_com: Array,
) -> CoarseFrontier:
    """Frontier = all leaf nodes of ``tree`` (a guaranteed mass-conserving cut)."""

    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    total = node_mass.shape[0]
    num_internal = tree.left_child.shape[0]
    root = as_index(jnp.argmin(jnp.asarray(tree.parent)))

    leaf_ids = jnp.arange(num_internal, total, dtype=INDEX_DTYPE)
    f_mass = node_mass[leaf_ids]
    # Empty (mass-0) leaves get COM 0 from the moment code; move them to the
    # root COM (a real in-box point) so they don't stretch the coarse tree box.
    nonempty = f_mass > 0
    f_com = jnp.where(nonempty[:, None], node_com[leaf_ids], node_com[root])
    f_range = node_ranges[leaf_ids]
    f_nodeid = jnp.where(nonempty, leaf_ids, as_index(-1))
    return CoarseFrontier(
        mass=f_mass, com=f_com, node_range=f_range, node_id=f_nodeid
    )


@dataclass
class GlobalCoarseTree:
    """Global coarse tree (identical on every device) + per-leaf origin tags."""

    tree: object
    geometry: object
    moments: object
    tag_domain: Array     # [ncoarse] origin GPU of each coarse particle
    tag_node_id: Array    # [ncoarse] origin local node id (-1 = padding)
    tag_range: Array      # [ncoarse, 2] origin particle range
    positions_sorted: Array
    masses_sorted: Array


def gather_global_coarse_tree(
    frontier: CoarseFrontier,
    *,
    bounds: tuple[Array, Array],
    axis_name: str = AXIS_NAME,
    coarse_leaf_size: int = 1,
) -> GlobalCoarseTree:
    """All_gather every domain's frontier and build the identical coarse tree.

    Each frontier node becomes a "coarse particle" (its COM, weighted by node
    mass). Building over identical gathered input yields the same tree on every
    device -- no agreement round needed.
    """

    n_top = frontier.mass.shape[0]
    me = jax.lax.axis_index(axis_name)

    coms = jax.lax.all_gather(frontier.com, axis_name, tiled=True)          # [ncoarse,3]
    masses = jax.lax.all_gather(frontier.mass, axis_name, tiled=True)       # [ncoarse]
    domain = jax.lax.all_gather(
        jnp.broadcast_to(me, (n_top,)).astype(INDEX_DTYPE), axis_name, tiled=True
    )
    node_id = jax.lax.all_gather(frontier.node_id, axis_name, tiled=True)
    node_range = jax.lax.all_gather(frontier.node_range, axis_name, tiled=True)  # [ncoarse,2]

    tree, pos_sorted, mass_sorted, _inv = build_tree(
        coms, masses, bounds, return_reordered=True, leaf_size=int(coarse_leaf_size)
    )
    geometry = compute_tree_geometry(tree, pos_sorted, max_leaf_size=int(coarse_leaf_size))
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    # Reorder origin tags into the coarse tree's Morton-sorted particle order.
    pidx = jnp.asarray(tree.particle_indices, dtype=INDEX_DTYPE)
    return GlobalCoarseTree(
        tree=tree,
        geometry=geometry,
        moments=moments,
        tag_domain=domain[pidx],
        tag_node_id=node_id[pidx],
        tag_range=node_range[pidx],
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
    )


@dataclass
class CoarseTreeMetrics:
    """Global-view diagnostics for the coarse-tree driver (testing)."""

    frontier_mass_sum: Array   # [ndev] per-domain frontier mass total
    domain_mass: Array         # [ndev] per-domain root mass
    coarse_root_mass: Array    # [ndev] coarse-tree total mass (should be global)
    coarse_root_com: Array     # [ndev, 3] coarse-tree COM (should be replicated)
    n_coarse_valid: Array      # [ndev] non-empty coarse leaves (== global count)


def build_distributed_coarse_tree(
    mesh,
    positions: Array,
    masses: Array,
    *,
    leaf_size: int,
    output_capacity: int,
    num_samples: int = 8,
    equalize: bool = True,
    axis_name: str = AXIS_NAME,
) -> CoarseTreeMetrics:
    """Decompose, build per-GPU trees, and gather the global coarse tree.

    Returns global-view :class:`CoarseTreeMetrics`. The full per-device
    :class:`GlobalCoarseTree` is built inside the ``shard_map`` and consumed by
    later LET stages; this driver surfaces the invariants worth testing.
    """

    try:  # stable across recent JAX versions
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    from .local_tree import sanitize_padding
    from .partition import equalize_domain, global_bounds, sfc_partition

    ndev = mesh.size

    def fn(pos, mass):
        bounds = global_bounds(pos, axis_name=axis_name)
        p, m, c, cnt = sfc_partition(
            pos, mass, ndev, output_capacity=output_capacity, bounds=bounds,
            num_samples=num_samples, axis_name=axis_name,
        )
        if equalize:
            p, m, c, cnt = equalize_domain(
                p, m, c, cnt, ndev, output_capacity=output_capacity, axis_name=axis_name
            )
        p, m = sanitize_padding(p, m, cnt)
        tree, pos_sorted, mass_sorted, _ = build_tree(
            p, m, bounds, return_reordered=True, leaf_size=leaf_size
        )
        moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)
        node_mass = moments.mass
        node_com = moments.center_of_mass
        root = as_index(jnp.argmin(jnp.asarray(tree.parent)))

        fr = build_coarse_frontier(tree, node_mass, node_com)
        gct = gather_global_coarse_tree(fr, bounds=bounds, axis_name=axis_name)

        croot = as_index(jnp.argmin(jnp.asarray(gct.tree.parent)))
        return (
            jnp.sum(fr.mass)[None],
            node_mass[root][None],
            gct.moments.mass[croot][None],
            gct.moments.center_of_mass[croot][None],
            jnp.sum((gct.tag_node_id >= 0).astype(INDEX_DTYPE))[None],
        )

    outs = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(axis_name), P(axis_name)),
        out_specs=(P(axis_name),) * 5,
        check_vma=False,
    )(positions, masses)
    return CoarseTreeMetrics(*outs)


__all__ = [
    "CoarseFrontier",
    "CoarseTreeMetrics",
    "GlobalCoarseTree",
    "build_coarse_frontier",
    "build_distributed_coarse_tree",
    "gather_global_coarse_tree",
]
