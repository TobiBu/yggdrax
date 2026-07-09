"""Per-GPU local tree + shared top tree for Yggdrax multi-GPU (Phase 2).

After :mod:`yggdrax.distributed.partition` gives each GPU a contiguous Morton
domain, this module builds a *local* radix tree over that domain (reusing the
existing single-device builder unchanged) and computes per-node mass/COM
moments (the P2M input). It then ``all_gather``s each domain's root moment into
a **shared top tree** -- the coarse multipole representation every GPU needs for
the far field, following the distributed-FMM Locally-Essential-Tree design.

Everything runs inside a single ``jax.shard_map`` body so tree build and moment
computation stay on-device with no host round-trips. The heavy lifting is the
existing pure/static-shape functions ``build_tree`` (adaptive LBVH),
``compute_tree_geometry`` and ``compute_tree_mass_moments`` -- this module only
adds padding hygiene and the coarse-moment gather.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._tree_impl import build_tree
from ..geometry import compute_tree_geometry
from ..tree_moments import compute_tree_mass_moments
from .partition import equalize_domain, global_bounds, sfc_partition
from .sharding import AXIS_NAME


@dataclass
class DistributedTreeMoments:
    """Global (host-side) view of per-GPU local trees + the shared top tree.

    Per-node leaves have leading dim ``ndev * total_nodes``; per-domain leaves
    have leading dim ``ndev``; ``top_mass``/``top_com`` are the gathered coarse
    multipoles, identical across devices (leading dim ``ndev`` of ``[ndev, ...]``
    rows).
    """

    node_mass: Array  # [ndev * total_nodes]
    node_com: Array  # [ndev * total_nodes, 3]
    domain_mass: Array  # [ndev]
    domain_com: Array  # [ndev, 3]
    top_mass: Array  # [ndev, ndev]  (each row = all domains' masses)
    top_com: Array  # [ndev, ndev, 3]
    counts: Array  # [ndev]


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


def build_local_moments(
    positions: Array,
    masses: Array,
    count: Array,
    ndev: int,
    *,
    bounds: tuple[Array, Array],
    leaf_size: int,
    axis_name: str = AXIS_NAME,
):
    """Build this device's local tree and gather the shared top tree.

    Returns ``(node_mass, node_com, domain_mass, domain_com, top_mass,
    top_com)`` for this device. ``bounds`` must be the *global* box (shared
    across devices) so geometry/Morton codes are consistent.
    """

    positions, masses = sanitize_padding(positions, masses, count)
    tree, pos_sorted, mass_sorted, _inv = build_tree(
        positions, masses, bounds, return_reordered=True, leaf_size=leaf_size
    )
    # Geometry is computed for completeness / downstream MAC use; keep the leaf
    # cap bounded so no num_particles-sized staging buffer is allocated.
    _geometry = compute_tree_geometry(tree, pos_sorted, max_leaf_size=leaf_size)
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    node_mass = moments.mass  # [total_nodes]
    node_com = moments.center_of_mass  # [total_nodes, 3]
    # Root (node 0) of the local LBVH spans the whole domain -> its moment is
    # the domain's aggregate (coarsest) multipole.
    domain_mass = node_mass[0]
    domain_com = node_com[0]

    # Shared top tree: every device learns every domain's coarse moment.
    top_mass = jax.lax.all_gather(domain_mass[None], axis_name, tiled=True)  # [ndev]
    top_com = jax.lax.all_gather(domain_com[None], axis_name, tiled=True)  # [ndev,3]
    return node_mass, node_com, domain_mass, domain_com, top_mass, top_com


def distributed_tree_moments(
    mesh,
    positions: Array,
    masses: Array,
    *,
    leaf_size: int,
    output_capacity: int,
    num_samples: int = 8,
    equalize: bool = True,
    axis_name: str = AXIS_NAME,
) -> DistributedTreeMoments:
    """Decompose, build per-GPU local trees, and gather the shared top tree.

    Fuses SFC decomposition (:func:`sfc_partition`/:func:`equalize_domain`) with
    the local tree build into one ``shard_map``. ``positions`` (``[N, 3]``) and
    ``masses`` (``[N]``) are sharded evenly over the mesh (``N`` divisible by
    ``mesh.size``). Uses the *global* bounding box for consistent geometry.
    """

    try:  # stable across recent JAX versions
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

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
        nm, nc, dm, dc, tm, tc = build_local_moments(
            p, m, cnt, ndev, bounds=bounds, leaf_size=leaf_size, axis_name=axis_name
        )
        return nm, nc, dm[None], dc[None], tm, tc, cnt[None]

    # check_vma=False: the reused single-device tree builder has a while_loop
    # whose carry starts uniform but becomes varying across the manual axis;
    # disabling varying-manual-axis checking lets it trace unchanged (all values
    # are treated as varying). Collectives (all_gather) still work.
    nm, nc, dm, dc, tm, tc, cnt = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(axis_name), P(axis_name)),
        out_specs=(P(axis_name),) * 7,
        check_vma=False,
    )(positions, masses)

    return DistributedTreeMoments(
        node_mass=nm,
        node_com=nc,
        domain_mass=dm,
        domain_com=dc,
        top_mass=tm,
        top_com=tc,
        counts=cnt,
    )


__all__ = [
    "DistributedTreeMoments",
    "build_local_moments",
    "distributed_tree_moments",
    "sanitize_padding",
]
