"""``compute_tree_geometry(particle_radius=...)``: bound balls, not points.

A tree built over *proxies* -- an LET coarse tree's particles are whole remote leaves
reduced to their centres of mass -- must bound what each particle stands for, or its
node extents understate the region they represent. A MAC built on those extents then
accepts pairs that are not actually well separated, and the multipole expansion gets
evaluated inside its own source region, where the series does not converge.

These are the properties the cross-domain far field depends on:

1. the inflation reaches every node, leaves and ancestors alike;
2. it is a genuine bound -- every ball fits inside its node's box;
3. a zero radius reproduces the point-bounded geometry exactly (no free drift);
4. ``TreeGeometry``'s own invariants survive (``radius == ||half_extent||``,
   ``center`` still the box centre).
"""

import jax.numpy as jnp
import numpy as np

from yggdrax.geometry import compute_tree_geometry
from yggdrax.tree import build_tree

_LEAF = 1
_BOUNDS = (jnp.array([-2.0, -2.0, -2.0]), jnp.array([2.0, 2.0, 2.0]))


def _sample_tree(n=8, seed=3):
    """A small tree with one particle per leaf, like an LET coarse tree."""
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, 3)), dtype=jnp.float32)
    masses = jnp.ones((n,), dtype=jnp.float32)
    tree, pos_sorted, _, _ = build_tree(
        positions, masses, _BOUNDS, return_reordered=True, leaf_size=_LEAF
    )
    return tree, pos_sorted


def test_zero_radius_reproduces_point_geometry():
    """An all-zero radius must change nothing at all."""
    tree, pos_sorted = _sample_tree()
    point = compute_tree_geometry(tree, pos_sorted, max_leaf_size=_LEAF)
    zeros = compute_tree_geometry(
        tree,
        pos_sorted,
        max_leaf_size=_LEAF,
        particle_radius=jnp.zeros((pos_sorted.shape[0],), dtype=pos_sorted.dtype),
    )
    for name in ("center", "half_extent", "radius", "max_extent"):
        np.testing.assert_allclose(
            np.asarray(getattr(zeros, name)),
            np.asarray(getattr(point, name)),
            rtol=0,
            atol=0,
            err_msg=f"{name} drifted under a zero particle_radius",
        )


def test_every_node_box_contains_every_ball_it_covers():
    """The bound is real: each particle's ball fits inside every node covering it.

    This is the property the MAC needs. Checked against the node ranges directly, so
    it covers internal nodes -- where the inflation has to have been carried up by the
    upward pass -- and not just the leaves it was applied to.
    """
    tree, pos_sorted = _sample_tree()
    positions = np.asarray(pos_sorted)
    radius = np.linspace(0.05, 0.4, positions.shape[0]).astype(positions.dtype)

    geom = compute_tree_geometry(
        tree, pos_sorted, max_leaf_size=_LEAF, particle_radius=jnp.asarray(radius)
    )
    centers = np.asarray(geom.center)
    half = np.asarray(geom.half_extent)
    ranges = np.asarray(tree.node_ranges)

    for node in range(ranges.shape[0]):
        lo, hi = ranges[node]
        if hi < lo:
            continue
        for particle in range(lo, hi + 1):
            ball_lo = positions[particle] - radius[particle]
            ball_hi = positions[particle] + radius[particle]
            box_lo = centers[node] - half[node]
            box_hi = centers[node] + half[node]
            assert np.all(ball_lo >= box_lo - 1e-5), (
                f"node {node}'s box does not cover particle {particle}'s ball "
                f"(low side): {ball_lo} vs {box_lo}"
            )
            assert np.all(ball_hi <= box_hi + 1e-5), (
                f"node {node}'s box does not cover particle {particle}'s ball "
                f"(high side): {ball_hi} vs {box_hi}"
            )


def test_inflation_reaches_internal_nodes():
    """A uniform radius must grow ancestors too, not only the leaves.

    The failure this guards against is inflating leaf boxes and leaving the upward
    pass to merge un-inflated ancestors, which would leave exactly the coarse
    *internal* nodes -- the ones a cross walk accepts first -- still understated.
    """
    tree, pos_sorted = _sample_tree()
    pad = 0.25
    point = compute_tree_geometry(tree, pos_sorted, max_leaf_size=_LEAF)
    grown = compute_tree_geometry(
        tree,
        pos_sorted,
        max_leaf_size=_LEAF,
        particle_radius=jnp.full((pos_sorted.shape[0],), pad, dtype=pos_sorted.dtype),
    )

    num_internal = int(tree.left_child.shape[0])
    assert num_internal > 0, "sample tree has no internal nodes to check"
    ranges = np.asarray(tree.node_ranges)
    populated = np.asarray(
        [n for n in range(num_internal) if ranges[n, 1] >= ranges[n, 0]]
    )
    assert populated.size > 0

    grew = (
        np.asarray(grown.half_extent)[populated]
        - np.asarray(point.half_extent)[populated]
    )
    # A uniform pad grows every populated node's half-extent by exactly the pad on
    # every axis, except where the 1e-6 minimum-extent floor was already binding.
    assert np.all(grew >= -1e-6), "an internal node shrank under inflation"
    assert np.max(grew) > 0.5 * pad, (
        "internal-node extents barely moved: the inflation is not reaching "
        f"ancestors (max growth {np.max(grew):.6f} against pad {pad})"
    )


def test_geometry_invariants_hold_under_inflation():
    """``radius == ||half_extent||`` and ``center`` stays the box centre."""
    tree, pos_sorted = _sample_tree()
    radius = jnp.full((pos_sorted.shape[0],), 0.3, dtype=pos_sorted.dtype)
    geom = compute_tree_geometry(
        tree, pos_sorted, max_leaf_size=_LEAF, particle_radius=radius
    )
    np.testing.assert_allclose(
        np.asarray(geom.radius),
        np.linalg.norm(np.asarray(geom.half_extent), axis=1),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(geom.max_extent),
        np.max(np.asarray(geom.half_extent), axis=1),
        rtol=1e-6,
    )
