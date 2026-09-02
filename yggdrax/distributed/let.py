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
from typing import Optional

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jaxtyping import Array

from ..dtypes import INDEX_DTYPE, as_index
from ..geometry import compute_tree_geometry
from ..tree import Tree
from ..tree_moments import compute_tree_mass_moments
from .comm import _COUNT_DTYPE, ragged_all_to_all_exchange
from .local_tree import _build_local_tree, _validate_distributed_tree_type
from .sharding import AXIS_NAME

# Global particle id = source_domain * _GID_STRIDE + local_sorted_index. Lets an
# importer verify provenance (which domain/index a halo particle came from).
_GID_STRIDE = 1 << 40


@dataclass
class CoarseFrontier:
    """One domain's disjoint frontier: its local tree's leaf nodes.

    The leaves of a tree always partition its particles exactly (every particle
    is in exactly one leaf), so leaf masses sum to the domain total -- robust
    even when equalize padding duplicates real positions into degenerate
    Morton-code clusters (which breaks a level-truncation antichain). Leaf
    granularity is the natural coarse resolution; it can be coarsened later by a
    leaf-up level cut without changing the force path.

    ``payload`` is an OPAQUE per-leaf row block that rides along unread. It exists
    because a cross-domain far field needs the remote leaf's MULTIPOLE, not just its
    mass and centre of mass -- and a multipole is expressed in the caller's basis
    (jaccpot's real spherical harmonics, ``(num_nodes, sh_size(order))``), which is
    not something a tree library should know. So the caller computes whatever it
    needs per node and this carries it across the boundary, permuted into the coarse
    tree's order alongside the origin tags, which is the part a caller cannot do for
    itself: only :func:`build_remote_coarse_tree` knows that permutation.

    Deliberately untyped beyond "some rows per leaf": monopole-only callers pass
    nothing, and a caller wanting expansions to order p pays ``(p+1)**2`` floats per
    leaf on the frontier ``all_gather`` -- still ``O(N / leaf_size)`` and negligible
    against the particles, but the caller's choice to make rather than this
    module's.
    """

    mass: Array  # [num_leaves]
    com: Array  # [num_leaves, 3]
    node_range: Array  # [num_leaves, 2] particle range in the local tree
    node_id: Array  # [num_leaves] local node id (-1 = empty/padding leaf)
    radius: Array  # [num_leaves] max |x - com| over the leaf's own particles
    payload: Optional[Array] = None  # [num_leaves, k] caller's own per-leaf data


def _leaf_radii_about(
    positions_sorted: Array,
    leaf_ranges: Array,
    centres: Array,
    nonempty: Array,
    *,
    max_leaf_size: Optional[int] = None,
) -> Array:
    """Distance from each leaf's ``centre`` to its farthest own particle.

    Mirrors ``_geometry_impl._compute_leaf_bounds``'s padded gather -- a
    ``(num_leaves, width)`` index block with a validity mask -- so the staging
    buffer is bounded by leaf occupancy and not by the particle count.

    Parameters
    ----------
    positions_sorted
        Particle positions in the tree's own order.
    leaf_ranges
        ``(num_leaves, 2)`` inclusive particle range per leaf.
    centres
        ``(num_leaves, 3)`` point each leaf is reduced to (its centre of mass).
    nonempty
        ``(num_leaves,)`` mask. Empty leaves get radius 0 rather than a distance to
        the root centre of mass their COM was substituted with.
    max_leaf_size
        Cap on the gather width. Falls back to the particle count, which is correct
        but stages a whole-problem-sized buffer.

    Returns
    -------
    Array
        ``(num_leaves,)`` radius, zero on empty leaves.
    """

    num_particles = positions_sorted.shape[0]
    starts = leaf_ranges[:, 0]
    counts = leaf_ranges[:, 1] - starts + as_index(1)
    width = max(int(max_leaf_size) if max_leaf_size is not None else num_particles, 1)

    offsets = jnp.arange(width, dtype=INDEX_DTYPE)[None, :]
    valid = (offsets < counts[:, None]) & nonempty[:, None]
    indices = jnp.clip(starts[:, None] + offsets, 0, num_particles - 1)
    deltas = positions_sorted[indices] - centres[:, None, :]
    # sqrt(sum(d^2)) rather than jnp.linalg.norm: for a real array reduced along one
    # axis the two are the same computation, but norm is typed as returning
    # ``Array | tuple[Array, ...]`` (its multi-axis overloads), which does not satisfy
    # the ``ArrayLike`` that jnp.max takes.
    distances = jnp.sqrt(jnp.sum(deltas * deltas, axis=-1))
    return jnp.max(jnp.where(valid, distances, 0.0), axis=1)


def build_coarse_frontier(
    tree: object,
    node_mass: Array,
    node_com: Array,
    *,
    positions_sorted: Optional[Array] = None,
    max_leaf_size: Optional[int] = None,
    node_payload: Optional[Array] = None,
) -> CoarseFrontier:
    """Frontier = all leaf nodes of ``tree`` (a guaranteed mass-conserving cut).

    Each entry also carries ``radius``: the distance from the leaf's centre of mass
    to its farthest own particle. Reducing a leaf to a point discards that, and a
    coarse tree built from the points alone reports a MAC extent bounding the
    centres of mass rather than the particles -- so a cross-domain walk accepts
    pairs whose true separation is smaller than the source's own radius, and the M2L
    is evaluated inside the region it is expanding. One float per leaf is what lets
    :func:`build_remote_coarse_tree` hand honest extents to that walk.

    Parameters
    ----------
    tree
        Local tree whose leaves form the frontier.
    node_mass
        Per-node mass, from ``compute_tree_mass_moments``.
    node_com
        Per-node centre of mass, from the same source.
    positions_sorted
        Particle positions in the tree's own order. Defaults to
        ``tree.positions_sorted``; one of the two must be available, because the
        radius cannot be recovered from the mass moments.
    node_payload
        ``(num_nodes, k)`` opaque per-node rows; the leaf rows are carried on the
        frontier as ``payload``. ``None`` carries nothing. Never read here -- see
        :class:`CoarseFrontier`.
    max_leaf_size
        Cap on the per-leaf gather, as for ``compute_tree_geometry``. Defaults to the
        tree's ``leaf_size`` when it exposes a concrete one.

    Returns
    -------
    CoarseFrontier
        The leaf cut: mass, centre of mass, particle range, node id and radius.

    Raises
    ------
    ValueError
        If no particle positions are available from either source. Returning a zero
        radius instead would silently reinstate the understated extents this field
        exists to prevent.
    """

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

    if positions_sorted is None:
        positions_sorted = getattr(tree, "positions_sorted", None)
    if positions_sorted is None:
        raise ValueError(
            "build_coarse_frontier needs particle positions to bound each leaf: "
            "pass positions_sorted=, or build the tree with return_reordered=True. "
            "The leaf radius cannot be derived from the mass moments, and "
            "defaulting it to zero would understate every coarse MAC extent."
        )
    if max_leaf_size is None:
        leaf_size = getattr(tree, "leaf_size", None)
        if leaf_size is not None and not isinstance(leaf_size, jax_core.Tracer):
            max_leaf_size = int(leaf_size)
    f_radius = _leaf_radii_about(
        jnp.asarray(positions_sorted),
        f_range,
        f_com,
        nonempty,
        max_leaf_size=max_leaf_size,
    )

    f_payload = None if node_payload is None else jnp.asarray(node_payload)[leaf_ids]
    return CoarseFrontier(
        mass=f_mass,
        com=f_com,
        node_range=f_range,
        node_id=f_nodeid,
        radius=f_radius,
        payload=f_payload,
    )


@dataclass
class GlobalCoarseTree:
    """Global coarse tree (identical on every device) + per-leaf origin tags."""

    tree: object
    geometry: object
    moments: object
    tag_domain: Array  # [ncoarse] origin GPU of each coarse particle
    tag_node_id: Array  # [ncoarse] origin local node id (-1 = padding)
    tag_range: Array  # [ncoarse, 2] origin particle range
    positions_sorted: Array
    masses_sorted: Array
    # [ncoarse, k] the frontier's opaque per-leaf rows, in THIS tree's order --
    # which is the only reason they travel through here rather than the caller
    # doing its own all_gather. None when the frontier carried none.
    payload: Optional[Array] = None


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
    device -- no agreement round needed. The coarse tree is always radix (an
    internal representation over remote COMs; the backend is immaterial and
    radix is robust at ``coarse_leaf_size=1``).
    """

    n_top = frontier.mass.shape[0]
    me = jax.lax.axis_index(axis_name)

    coms = jax.lax.all_gather(frontier.com, axis_name, tiled=True)  # [ncoarse,3]
    masses = jax.lax.all_gather(frontier.mass, axis_name, tiled=True)  # [ncoarse]
    radii = jax.lax.all_gather(frontier.radius, axis_name, tiled=True)  # [ncoarse]
    domain = jax.lax.all_gather(
        jnp.broadcast_to(me, (n_top,)).astype(INDEX_DTYPE), axis_name, tiled=True
    )
    node_id = jax.lax.all_gather(frontier.node_id, axis_name, tiled=True)
    node_range = jax.lax.all_gather(
        frontier.node_range, axis_name, tiled=True
    )  # [ncoarse,2]

    tree, pos_sorted, mass_sorted = _build_local_tree(
        coms, masses, bounds, tree_type="radix", leaf_size=int(coarse_leaf_size)
    )
    # Origin tags and the per-particle radii both live in the gathered order, so
    # reorder them into the coarse tree's Morton-sorted particle order.
    pidx = jnp.asarray(tree.particle_indices, dtype=INDEX_DTYPE)
    # Bound each coarse particle as the ball it stands for, not as the point it was
    # reduced to; see build_coarse_frontier on why the point bound is not safe.
    geometry = compute_tree_geometry(
        tree,
        pos_sorted,
        max_leaf_size=int(coarse_leaf_size),
        particle_radius=radii[pidx],
    )
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

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

    frontier_mass_sum: Array  # [ndev] per-domain frontier mass total
    domain_mass: Array  # [ndev] per-domain root mass
    coarse_root_mass: Array  # [ndev] coarse-tree total mass (should be global)
    coarse_root_com: Array  # [ndev, 3] coarse-tree COM (should be replicated)
    n_coarse_valid: Array  # [ndev] non-empty coarse leaves (== global count)


def build_distributed_coarse_tree(
    mesh,
    positions: Array,
    masses: Array,
    *,
    leaf_size: int,
    output_capacity: int,
    num_samples: int = 8,
    equalize: bool = True,
    tree_type: str = "radix",
    axis_name: str = AXIS_NAME,
) -> CoarseTreeMetrics:
    """Decompose, build per-GPU trees, and gather the global coarse tree.

    Returns global-view :class:`CoarseTreeMetrics`. The full per-device
    :class:`GlobalCoarseTree` is built inside the ``shard_map`` and consumed by
    later LET stages; this driver surfaces the invariants worth testing.
    ``tree_type`` selects the local per-device backend (``"radix"``,
    ``"octree"``, or ``"kdtree"``); the coarse tree is always radix.
    """

    _validate_distributed_tree_type(tree_type)

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
            pos,
            mass,
            ndev,
            output_capacity=output_capacity,
            bounds=bounds,
            num_samples=num_samples,
            axis_name=axis_name,
        )
        if equalize:
            p, m, c, cnt = equalize_domain(
                p, m, c, cnt, ndev, output_capacity=output_capacity, axis_name=axis_name
            )
        p, m = sanitize_padding(p, m, cnt)
        tree, pos_sorted, mass_sorted = _build_local_tree(
            p, m, bounds, tree_type=tree_type, leaf_size=leaf_size
        )
        moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)
        node_mass = moments.mass
        node_com = moments.center_of_mass
        root = as_index(jnp.argmin(jnp.asarray(tree.parent)))

        fr = build_coarse_frontier(
            tree,
            node_mass,
            node_com,
            positions_sorted=pos_sorted,
            max_leaf_size=leaf_size,
        )
        # Coarse tree is always radix: it is an internal representation over
        # remote frontier COMs (backend-immaterial), and radix is robust at the
        # coarse leaf_size=1 that KD/octree bucket builds cannot honour exactly.
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


def build_remote_coarse_tree(
    frontier: CoarseFrontier,
    ndev: int,
    *,
    bounds: tuple[Array, Array],
    axis_name: str = AXIS_NAME,
    coarse_leaf_size: int = 1,
) -> GlobalCoarseTree:
    """Coarse tree over *other* domains' frontiers only (own domain excluded).

    Cross-walking the local tree against this yields purely remote far (M2L) and
    near (import) interactions -- the local self-walk already covers own-domain
    pairs at full resolution, so excluding own domain here prevents any
    double-counting. Own-domain frontier nodes are dropped structurally (not
    just mass-zeroed), so they never clog the near list.

    The geometry handed back bounds each coarse particle as the **ball** its remote
    leaf occupies, using ``frontier.radius``, not as the centre of mass the leaf was
    reduced to. That is what makes the returned ``geometry`` safe to use as a MAC
    extent: bounding the centres of mass understates the source region, so the walk
    accepts pairs whose true separation is smaller than the source's own radius and
    the M2L is then evaluated inside the region it is expanding -- an error that
    exceeds the term it is approximating.
    """

    n_top = frontier.mass.shape[0]
    me = jax.lax.axis_index(axis_name)

    coms = jax.lax.all_gather(frontier.com, axis_name, tiled=True)
    masses = jax.lax.all_gather(frontier.mass, axis_name, tiled=True)
    radii = jax.lax.all_gather(frontier.radius, axis_name, tiled=True)
    domain = jax.lax.all_gather(
        jnp.broadcast_to(me, (n_top,)).astype(INDEX_DTYPE), axis_name, tiled=True
    )
    node_id = jax.lax.all_gather(frontier.node_id, axis_name, tiled=True)
    node_range = jax.lax.all_gather(frontier.node_range, axis_name, tiled=True)
    payload = (
        None
        if frontier.payload is None
        else jax.lax.all_gather(frontier.payload, axis_name, tiled=True)
    )

    # Compact to the (ndev-1)*n_top remote coarse particles (static size).
    keep = domain != me
    n_remote = (ndev - 1) * n_top
    idx = jnp.nonzero(keep, size=n_remote, fill_value=0)[0]
    r_coms = coms[idx]
    r_mass = masses[idx]
    r_radius = radii[idx]
    r_domain = domain[idx]
    r_node_id = node_id[idx]
    r_range = node_range[idx]
    r_payload = None if payload is None else payload[idx]

    # Build a Tree wrapper (identical adaptive-radix topology) rather than the
    # raw RadixTree, so jaccpot's Tree-typed stages (e.g. compute_node_multipoles
    # for the remote M2L source) accept it directly.
    tree = Tree.from_particles(
        r_coms,
        r_mass,
        tree_type="radix",
        bounds=bounds,
        return_reordered=True,
        leaf_size=int(coarse_leaf_size),
    )
    pos_sorted = tree.positions_sorted
    mass_sorted = tree.masses_sorted
    pidx = jnp.asarray(tree.particle_indices, dtype=INDEX_DTYPE)
    geometry = compute_tree_geometry(
        tree,
        pos_sorted,
        max_leaf_size=int(coarse_leaf_size),
        particle_radius=r_radius[pidx],
    )
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)
    return GlobalCoarseTree(
        tree=tree,
        geometry=geometry,
        moments=moments,
        tag_domain=r_domain[pidx],
        tag_node_id=r_node_id[pidx],
        tag_range=r_range[pidx],
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        # Same `pidx` as the tags, for the same reason: a row has to stay
        # attached to the coarse particle it describes once the build reorders
        # them.
        payload=None if r_payload is None else r_payload[pidx],
    )


@dataclass
class ClassifyMetrics:
    """Global-view diagnostics for the remote classification driver (testing)."""

    remote_root_mass: Array  # [ndev] mass of the remote coarse tree
    own_domain_mass: Array  # [ndev] own domain total mass
    total_mass: Array  # [ndev] global total mass (replicated)
    far_count: Array  # [ndev] remote far (M2L) interactions
    near_count: Array  # [ndev] remote near (import) interactions
    overflow: Array  # [ndev] bool: any walk capacity overflow


def classify_against_remote(
    mesh,
    positions: Array,
    masses: Array,
    *,
    leaf_size: int,
    output_capacity: int,
    theta: float = 0.5,
    mac_type: str = "bh",
    max_interactions_per_node: int = 256,
    max_neighbors_per_leaf: int = 256,
    max_pair_queue: int = 8192,
    num_samples: int = 8,
    equalize: bool = True,
    tree_type: str = "radix",
    axis_name: str = AXIS_NAME,
) -> ClassifyMetrics:
    """Cross-walk each GPU's local tree against the remote-only coarse tree.

    Produces the far (remote M2L) and near (remote import) classifications that
    Phase-3 stage 3 turns into an actual particle import. This driver surfaces
    the invariants; the far/near *lists* are consumed by later stages.
    ``tree_type`` selects the local per-device backend (``"radix"``,
    ``"octree"``, or ``"kdtree"``); the coarse tree is always radix.
    """

    _validate_distributed_tree_type(tree_type)

    try:  # stable across recent JAX versions
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    from .cross_walk import dual_tree_walk_cross_impl
    from .local_tree import sanitize_padding
    from .partition import equalize_domain, global_bounds, sfc_partition

    ndev = mesh.size

    def fn(pos, mass):
        bounds = global_bounds(pos, axis_name=axis_name)
        p, m, c, cnt = sfc_partition(
            pos,
            mass,
            ndev,
            output_capacity=output_capacity,
            bounds=bounds,
            num_samples=num_samples,
            axis_name=axis_name,
        )
        if equalize:
            p, m, c, cnt = equalize_domain(
                p, m, c, cnt, ndev, output_capacity=output_capacity, axis_name=axis_name
            )
        p, m = sanitize_padding(p, m, cnt)
        tree, pos_sorted, mass_sorted = _build_local_tree(
            p, m, bounds, tree_type=tree_type, leaf_size=leaf_size
        )
        geom = compute_tree_geometry(tree, pos_sorted, max_leaf_size=leaf_size)
        moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)
        root = as_index(jnp.argmin(jnp.asarray(tree.parent)))
        own_mass = moments.mass[root]

        fr = build_coarse_frontier(
            tree,
            moments.mass,
            moments.center_of_mass,
            positions_sorted=pos_sorted,
            max_leaf_size=leaf_size,
        )
        # Coarse tree is always radix (internal remote-COM representation).
        rct = build_remote_coarse_tree(fr, ndev, bounds=bounds, axis_name=axis_name)
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
        total = jax.lax.psum(own_mass, axis_name)
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
        in_specs=(P(axis_name), P(axis_name)),
        out_specs=(P(axis_name),) * 6,
        check_vma=False,
    )(positions, masses)
    return ClassifyMetrics(*outs)


@dataclass
class HaloImport:
    """Remote particles imported for this GPU's near field (padded).

    ``positions``/``masses``/``gid`` are laid out as ``max_req_leaves`` blocks of
    ``leaf_size`` rows each (block ``h`` = imported remote leaf ``h``).
    ``coarse_to_halo[p]`` maps a remote coarse-tree sorted position ``p`` (a near
    source names one via its node range) to its halo block index ``h`` (or -1) --
    the link the combined near-field P2P uses to point a local target leaf at its
    imported halo particles.

    ``payload`` is the caller's own per-PARTICLE data for the same rows, or ``None``
    when none was passed. It is the per-particle counterpart of
    :attr:`CoarseFrontier.payload` and rides the same round-B exchange as the
    positions, so it is sized by the halo rather than by the system -- which is why
    it is here and not on the frontier. The frontier is ``all_gather``-ed, so
    publishing ``leaf_size`` columns there would ship every remote particle's value
    to every device, i.e. ``O(N_total)``, defeating the demand-driven import.
    """

    positions: Array  # [max_req_leaves * leaf_size, 3]
    masses: Array  # [max_req_leaves * leaf_size]
    gid: Array  # [max_req_leaves * leaf_size] source global id (-1 = pad)
    valid: Array  # [max_req_leaves * leaf_size] bool
    coarse_to_halo: Array  # [n_remote] coarse sorted position -> halo block (-1)
    needed_mass: Array  # scalar: total mass of the requested remote leaves
    imported_mass: Array  # scalar: total mass actually received (== needed_mass)
    request_overflow: Array  # scalar bool
    payload: Optional[Array] = None  # [max_req_leaves * leaf_size, k] caller's own


def import_near_halo(
    rct: GlobalCoarseTree,
    near_result,
    positions_sorted: Array,
    masses_sorted: Array,
    ndev: int,
    *,
    leaf_size: int,
    max_req_leaves: int,
    max_recv_leaves: int,
    payload_sorted: Optional[Array] = None,
    axis_name: str = AXIS_NAME,
) -> HaloImport:
    """Two-round ragged import of the remote near-field particles.

    Round A sends, to each owning domain, requests ``(start, count)`` naming the
    remote leaves this GPU classified *near* (deduped). Round B returns those
    leaves' actual particles. The importer never over-imports (each leaf sends
    exactly its ``count`` particles), so a halo particle is never also covered
    by a far M2L node -- no double counting.

    ``payload_sorted`` is an optional ``(n_local, k)`` block of the caller's own
    per-particle data, in the same tree order as ``positions_sorted``. It rides
    round B in the same buffer as the positions and masses -- no extra collective --
    and comes back on :attr:`HaloImport.payload`, row for row with
    ``positions``/``gid``. ``None`` carries nothing and costs nothing.

    A per-particle quantity has to travel this way rather than on the frontier: the
    frontier is ``all_gather``-ed, so ``leaf_size`` columns published there would
    ship every remote particle's value to every device. Anything defined per
    *leaf* -- a multipole, an extent -- belongs on the frontier instead.
    """

    me = jax.lax.axis_index(axis_name)
    n_remote = rct.tag_domain.shape[0]
    cnr = jnp.asarray(rct.tree.node_ranges, dtype=INDEX_DTYPE)

    # 1. Mark the remote coarse-leaf positions that appear in the near list.
    nbr = near_result.neighbor_indices
    valid_n = nbr >= 0
    safe_nbr = jnp.where(valid_n, nbr, as_index(0))
    posn = cnr[safe_nbr, 0]  # coarse sorted position (leaf_size=1 -> range start)
    needed = (
        jnp.zeros((n_remote,), dtype=jnp.bool_)
        .at[jnp.where(valid_n, posn, as_index(n_remote))]
        .set(True, mode="drop")
    )

    # 2. Compact needed leaves -> request records (dest, start, count).
    req_pos = jnp.nonzero(needed, size=max_req_leaves, fill_value=n_remote)[0]
    valid_req = req_pos < n_remote
    safe_pos = jnp.where(valid_req, req_pos, as_index(0))
    dest = jnp.where(valid_req, rct.tag_domain[safe_pos], as_index(ndev))
    start = jnp.where(valid_req, rct.tag_range[safe_pos, 0], as_index(0))
    count = jnp.where(
        valid_req,
        rct.tag_range[safe_pos, 1] - rct.tag_range[safe_pos, 0] + 1,
        as_index(0),
    )
    request_overflow = jnp.sum(needed.astype(INDEX_DTYPE)) > as_index(max_req_leaves)

    # Group requests by destination (padding dest=ndev sorts to the tail).
    order = jnp.argsort(dest)
    dest, start, count = dest[order], start[order], count[order]
    send_sizes_a = jnp.bincount(dest, length=ndev).astype(_COUNT_DTYPE)
    reqbuf = jnp.stack([start, count], axis=1)

    # Halo blocks come back in this sorted-request order, so request slot h holds
    # coarse position sorted_req_pos[h]. Invert to map coarse position -> block.
    sorted_req_pos = req_pos[order]
    coarse_to_halo = (
        jnp.full((n_remote,), -1, dtype=INDEX_DTYPE)
        .at[sorted_req_pos]
        .set(jnp.arange(max_req_leaves, dtype=INDEX_DTYPE), mode="drop")
    )

    # 3. Round A: exchange requests.
    recv_req, recv_sizes_a, _ = ragged_all_to_all_exchange(
        reqbuf, send_sizes_a, output_capacity=max_recv_leaves, axis_name=axis_name
    )
    n_recv = jnp.sum(recv_sizes_a)
    recv_start = recv_req[:, 0]
    recv_count = recv_req[:, 1]
    valid_recv = jnp.arange(max_recv_leaves, dtype=INDEX_DTYPE) < n_recv

    # 4. Fulfill: expand each received request to leaf_size rows of our particles.
    n_local = positions_sorted.shape[0]
    k = jnp.arange(leaf_size, dtype=INDEX_DTYPE)
    idx = recv_start[:, None] + k[None, :]  # [max_recv_leaves, leaf_size]
    in_leaf = (k[None, :] < recv_count[:, None]) & valid_recv[:, None]
    safe_idx = jnp.clip(idx, 0, n_local - 1)
    resp_pos = jnp.where(in_leaf[..., None], positions_sorted[safe_idx], 0.0)
    resp_mass = jnp.where(in_leaf, masses_sorted[safe_idx], 0.0)
    resp_gid = jnp.where(in_leaf, me * as_index(_GID_STRIDE) + safe_idx, as_index(-1))
    R = max_recv_leaves
    # The payload joins the position/mass buffer rather than getting an exchange of
    # its own: same rows, same sizes, same ordering, so it cannot drift out of step
    # with the particles it describes -- and one fewer collective.
    resp_cols = [resp_pos, resp_mass[..., None]]
    n_payload = 0
    if payload_sorted is not None:
        pay = jnp.asarray(payload_sorted, dtype=resp_pos.dtype)
        if pay.ndim != 2 or pay.shape[0] != positions_sorted.shape[0]:
            raise ValueError(
                "payload_sorted must be (n_local, k) in the same order as "
                f"positions_sorted ({positions_sorted.shape[0]} rows); got "
                f"{tuple(pay.shape)}"
            )
        n_payload = int(pay.shape[1])
        resp_cols.append(jnp.where(in_leaf[..., None], pay[safe_idx], 0.0))
    resp_posm = jnp.concatenate(resp_cols, axis=-1).reshape(
        (R * leaf_size, 4 + n_payload)
    )
    resp_gid = resp_gid.reshape((R * leaf_size, 1))
    send_sizes_b = (recv_sizes_a * as_index(leaf_size)).astype(_COUNT_DTYPE)

    # 5. Round B: return the requested particles to each requester.
    cap_b = max_req_leaves * leaf_size
    halo_posm, _, _ = ragged_all_to_all_exchange(
        resp_posm, send_sizes_b, output_capacity=cap_b, axis_name=axis_name
    )
    halo_gid, _, _ = ragged_all_to_all_exchange(
        resp_gid,
        send_sizes_b,
        output_capacity=cap_b,
        axis_name=axis_name,
        fill_value=-1.0,
    )
    halo_pos = halo_posm[:, :3]
    halo_mass = halo_posm[:, 3]
    halo_payload = None if payload_sorted is None else halo_posm[:, 4:]
    halo_gid = halo_gid[:, 0]
    halo_valid = halo_gid >= 0

    needed_mass = jnp.sum(jnp.where(needed, rct.masses_sorted, 0.0))
    imported_mass = jnp.sum(jnp.where(halo_valid, halo_mass, 0.0))
    return HaloImport(
        positions=halo_pos,
        masses=halo_mass,
        gid=halo_gid,
        valid=halo_valid,
        coarse_to_halo=coarse_to_halo,
        needed_mass=needed_mass,
        imported_mass=imported_mass,
        request_overflow=request_overflow,
        payload=halo_payload,
    )


@dataclass
class ImportMetrics:
    """Global-view diagnostics for the halo-import driver (testing)."""

    needed_mass: Array  # [ndev] mass of requested remote leaves
    imported_mass: Array  # [ndev] mass actually received (== needed_mass)
    request_overflow: Array  # [ndev] bool
    wrong_domain: Array  # [ndev] bool: any halo particle sourced from self
    n_halo_valid: Array  # [ndev] imported particle count
    n_mapped: Array  # [ndev] coarse positions mapped to a halo block


def distributed_let_import(
    mesh,
    positions: Array,
    masses: Array,
    *,
    leaf_size: int,
    output_capacity: int,
    theta: float = 0.5,
    mac_type: str = "bh",
    max_interactions_per_node: int = 256,
    max_neighbors_per_leaf: int = 256,
    max_pair_queue: int = 8192,
    num_samples: int = 8,
    equalize: bool = True,
    tree_type: str = "radix",
    axis_name: str = AXIS_NAME,
) -> ImportMetrics:
    """Full LET pipeline through the halo import (decompose -> classify -> import).

    ``tree_type`` selects the local per-device backend (``"radix"``,
    ``"octree"``, or ``"kdtree"``); the coarse tree is always radix.
    """

    _validate_distributed_tree_type(tree_type)

    try:  # stable across recent JAX versions
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    from .cross_walk import dual_tree_walk_cross_impl
    from .local_tree import sanitize_padding
    from .partition import equalize_domain, global_bounds, sfc_partition

    ndev = mesh.size
    n_leaf = (output_capacity + leaf_size - 1) // leaf_size
    max_req = (ndev - 1) * n_leaf
    max_recv = (ndev - 1) * n_leaf

    def fn(pos, mass):
        bounds = global_bounds(pos, axis_name=axis_name)
        p, m, c, cnt = sfc_partition(
            pos,
            mass,
            ndev,
            output_capacity=output_capacity,
            bounds=bounds,
            num_samples=num_samples,
            axis_name=axis_name,
        )
        if equalize:
            p, m, c, cnt = equalize_domain(
                p, m, c, cnt, ndev, output_capacity=output_capacity, axis_name=axis_name
            )
        p, m = sanitize_padding(p, m, cnt)
        tree, pos_sorted, mass_sorted = _build_local_tree(
            p, m, bounds, tree_type=tree_type, leaf_size=leaf_size
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
        # Coarse tree is always radix (internal remote-COM representation).
        rct = build_remote_coarse_tree(fr, ndev, bounds=bounds, axis_name=axis_name)
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
            axis_name=axis_name,
        )
        me = jax.lax.axis_index(axis_name)
        halo_domain = jnp.where(
            halo.valid, halo.gid // as_index(_GID_STRIDE), as_index(-1)
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
        in_specs=(P(axis_name), P(axis_name)),
        out_specs=(P(axis_name),) * 6,
        check_vma=False,
    )(positions, masses)
    return ImportMetrics(*outs)


__all__ = [
    "ClassifyMetrics",
    "CoarseFrontier",
    "CoarseTreeMetrics",
    "GlobalCoarseTree",
    "HaloImport",
    "ImportMetrics",
    "build_coarse_frontier",
    "build_distributed_coarse_tree",
    "build_remote_coarse_tree",
    "classify_against_remote",
    "distributed_let_import",
    "gather_global_coarse_tree",
    "import_near_halo",
]
