"""The coarse tree's MAC extents must bound remote particles, not their COMs.

A coarse "particle" is a whole remote leaf reduced to its centre of mass. If the
coarse tree's geometry bounds only those points, the extent a cross walk divides by
understates the source region it stands for -- so the MAC accepts pairs whose true
separation is smaller than the source's own radius, and the M2L is evaluated inside
the region it is expanding, where the series does not converge. The resulting error
exceeds the term being approximated, which is how it was found: computing the
cross-domain far field was measurably *worse* than dropping it.

The IC below is what makes the difference visible: leaves that are large relative to
the distances between them. Every domain gets particles from both of two widely
separated clusters, so a single leaf can span the gap and its centre of mass sits in
empty space with particles far away on either side.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_coarse_tree_extents.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.let import build_coarse_frontier, build_remote_coarse_tree
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.tree import Tree
from yggdrax.tree_moments import compute_tree_mass_moments

try:  # pragma: no cover - import shape only
    from jax import shard_map
except ImportError:  # pragma: no cover - older jax
    from jax.experimental.shard_map import shard_map

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="remote coarse tree needs >= 2 devices"
)

_NDEV = 2
_LEAF = 8
_PER = 32


def _interleaved_two_clusters():
    """Two clusters 6 apart, split so each domain holds part of both.

    Deliberately *not* one cluster per domain: interpenetrating domains are what put
    a large remote leaf near a local target, which is the configuration an understated
    extent gets wrong. Interleaving by index is a stand-in for what a space-filling
    curve does at any real domain boundary.
    """
    rng = np.random.default_rng(4)
    centres = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float32)
    blocks = [centres[c] + rng.uniform(-0.5, 0.5, (_PER, 3)) for c in range(2)]
    per_device = []
    for d in range(_NDEV):
        half = _PER // 2
        lo = d * half
        take = [blocks[c][lo : lo + half] for c in range(2)]
        per_device.append(np.concatenate(take).astype(np.float32))
    positions = np.stack(per_device)  # [ndev, _PER, 3]
    mass = rng.uniform(0.5, 2.0, size=(_NDEV, _PER)).astype(np.float32)
    return positions, mass


@pytest.fixture(scope="module")
def coarse():
    """Per-device coarse-tree geometry, plus what it should have bounded.

    Returns the coarse tree's node boxes and, for every coarse particle, the true
    extent of the remote leaf behind it -- gathered out of ``shard_map`` so the
    assertions can be written in plain NumPy.
    """
    positions, mass = _interleaved_two_clusters()
    lo = positions.reshape(-1, 3).min(0)
    hi = positions.reshape(-1, 3).max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    bounds = (jnp.asarray(lo - span * 1e-6), jnp.asarray(hi + span * 1e-6))
    mesh = make_mesh(_NDEV)

    def fn(pos, mss):
        tree = Tree.from_particles(
            pos,
            mss,
            tree_type="radix",
            bounds=bounds,
            return_reordered=True,
            leaf_size=_LEAF,
        )
        moments = compute_tree_mass_moments(
            tree, tree.positions_sorted, tree.masses_sorted
        )
        frontier = build_coarse_frontier(
            tree,
            moments.mass,
            moments.center_of_mass,
            positions_sorted=tree.positions_sorted,
            max_leaf_size=_LEAF,
        )
        rct = build_remote_coarse_tree(frontier, _NDEV, bounds=bounds)
        return (
            rct.geometry.center,
            rct.geometry.half_extent,
            jnp.asarray(rct.tree.node_ranges, INDEX_DTYPE),
            rct.positions_sorted,
            rct.tag_range,
            rct.tag_domain,
            frontier.radius[None, :],
            tree.positions_sorted[None, ...],
        )

    out = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P("gpus"), P("gpus")),
        out_specs=(P("gpus"),) * 8,
        check_vma=False,
    )(
        jnp.asarray(positions.reshape(-1, 3)),
        jnp.asarray(mass.reshape(-1)),
    )
    return [np.asarray(o) for o in out], positions


def test_frontier_radius_covers_each_leaf(coarse):
    """``frontier.radius`` is the true distance from the COM to the farthest particle."""
    (_, _, _, _, _, _, radius, local_pos), _ = coarse
    radius = radius.reshape(_NDEV, -1)
    local_pos = local_pos.reshape(_NDEV, -1, 3)

    # Non-degenerate: an interleaved domain must produce at least one leaf wide
    # enough to matter, or this IC is not exercising what it claims to.
    assert radius.max() > 0.5, (
        "no leaf is larger than a cluster radius, so this IC no longer creates the "
        f"gap-spanning leaves the test needs (max radius {radius.max():.4f})"
    )
    assert np.all(radius >= 0.0)
    # And it must never exceed the domain's own diameter.
    for d in range(_NDEV):
        diameter = np.linalg.norm(local_pos[d].max(0) - local_pos[d].min(0))
        assert radius[d].max() <= diameter + 1e-4


def test_coarse_node_boxes_bound_the_remote_particles(coarse):
    """Every coarse node's box must contain every remote particle beneath it.

    The property the cross-domain MAC rests on. Before the frontier carried a radius
    this failed on the very first coarse leaf: a leaf spanning the cluster gap was
    represented as a zero-extent point, so its box excluded its own particles.
    """
    (centers, half, ranges, coarse_pos, tag_range, tag_domain, _, local_pos), _ = coarse
    n_nodes = centers.shape[0] // _NDEV
    n_coarse = coarse_pos.shape[0] // _NDEV
    local_pos = local_pos.reshape(_NDEV, -1, 3)

    checked = 0
    for d in range(_NDEV):
        c = centers[d * n_nodes : (d + 1) * n_nodes]
        h = half[d * n_nodes : (d + 1) * n_nodes]
        r = ranges[d * n_nodes : (d + 1) * n_nodes]
        tr = tag_range[d * n_coarse : (d + 1) * n_coarse]
        td = tag_domain[d * n_coarse : (d + 1) * n_coarse]
        for node in range(n_nodes):
            lo, hi = r[node]
            if hi < lo:
                continue
            box_lo = c[node] - h[node]
            box_hi = c[node] + h[node]
            for particle in range(lo, hi + 1):
                start, end = tr[particle]
                if end < start:
                    continue
                owner = int(td[particle])
                pts = local_pos[owner][start : end + 1]
                assert np.all(pts >= box_lo - 1e-4), (
                    f"device {d} coarse node {node} excludes remote particles of "
                    f"coarse particle {particle} (low side)"
                )
                assert np.all(pts <= box_hi + 1e-4), (
                    f"device {d} coarse node {node} excludes remote particles of "
                    f"coarse particle {particle} (high side)"
                )
                checked += 1
    assert checked > 0, "no populated coarse nodes were checked"


def test_coarse_leaf_extent_is_not_a_point(coarse):
    """A coarse leaf standing for a wide remote leaf must not report ~zero extent.

    Specifically the regression: with COM-only bounds every coarse leaf had a
    zero half-extent (floored at 1e-6), so the MAC's leaf extent came from an
    unrelated depth heuristic instead of from the geometry.
    """
    (_, half, ranges, coarse_pos, _, _, _, _), _ = coarse
    n_nodes = half.shape[0] // _NDEV
    for d in range(_NDEV):
        h = half[d * n_nodes : (d + 1) * n_nodes]
        r = ranges[d * n_nodes : (d + 1) * n_nodes]
        leaf_extents = [
            float(np.max(h[node]))
            for node in range(n_nodes)
            if r[node, 1] >= r[node, 0] and r[node, 1] == r[node, 0]
        ]
        assert leaf_extents, f"device {d} has no single-particle coarse leaves"
        assert max(leaf_extents) > 1e-3, (
            f"device {d}'s coarse leaves are all point-like (max half-extent "
            f"{max(leaf_extents):.2e}): the frontier radius is not reaching the "
            "coarse geometry"
        )
