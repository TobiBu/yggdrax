"""Tests for tree mass moments."""

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.tree import Tree, build_tree
from yggdrax.tree_moments import compute_tree_mass_moments

DEFAULT_TEST_LEAF_SIZE = 1


def _direct_node_mass_and_com(tree, positions_sorted, masses_sorted):
    """Per-node mass and centre of mass, summed directly in float64 numpy.

    Each node owns the inclusive Morton-order range ``node_ranges[node]``; an
    empty or massless node gets centre 0, the convention the prefix sums follow.
    """
    ranges = np.asarray(tree.node_ranges)
    pos = np.asarray(positions_sorted, np.float64)
    mass = np.asarray(masses_sorted, np.float64)
    node_mass = np.zeros(ranges.shape[0])
    node_com = np.zeros((ranges.shape[0], 3))
    for node, (start, end) in enumerate(ranges):
        start, end = int(start), int(end) + 1
        node_mass[node] = mass[start:end].sum()
        if node_mass[node] != 0.0:
            node_com[node] = (mass[start:end, None] * pos[start:end]).sum(0) / (
                node_mass[node]
            )
    return node_mass, node_com


def _assert_matches_direct_sums(tree, positions_sorted, masses_sorted, moments):
    expected_mass, expected_com = _direct_node_mass_and_com(
        tree, positions_sorted, masses_sorted
    )
    np.testing.assert_allclose(
        np.asarray(moments.mass), expected_mass, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(moments.center_of_mass), expected_com, rtol=1e-12, atol=1e-12
    )


def _build_sample_tree():
    positions = jnp.array(
        [
            [-0.5, -0.5, -0.5],
            [0.1, 0.0, 0.3],
            [0.6, 0.4, -0.2],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0])
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    return tree, pos_sorted, mass_sorted


def test_mass_moment_shapes():
    tree, pos_sorted, mass_sorted = _build_sample_tree()
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    total_nodes = tree.parent.shape[0]
    assert moments.mass.shape == (total_nodes,)
    assert moments.center_of_mass.shape == (total_nodes, 3)


def test_leaf_masses_and_centers_match_particles():
    tree, pos_sorted, mass_sorted = _build_sample_tree()
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    num_internal = tree.num_internal_nodes
    assert jnp.allclose(moments.mass[num_internal:], mass_sorted)
    assert jnp.allclose(moments.center_of_mass[num_internal:], pos_sorted)
    _assert_matches_direct_sums(tree, pos_sorted, mass_sorted, moments)


@pytest.mark.parametrize("tree_type", ["radix", "kdtree"])
def test_tree_wrapper_supports_mass_moments(tree_type: str):
    positions = jnp.array(
        [
            [-0.6, -0.3, -0.2],
            [-0.2, 0.4, -0.1],
            [0.1, -0.5, 0.3],
            [0.5, 0.2, -0.4],
            [0.8, 0.7, 0.6],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 2.0, 1.5, 0.5, 3.0], dtype=jnp.float64)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type=tree_type,
        return_reordered=True,
        leaf_size=2,
    )

    moments = compute_tree_mass_moments(
        tree,
        tree.positions_sorted,
        tree.masses_sorted,
    )

    assert moments.mass.shape[0] == int(tree.num_nodes)
    assert jnp.all(jnp.isfinite(moments.mass))
    assert jnp.all(jnp.isfinite(moments.center_of_mass))
    assert jnp.isclose(jnp.max(moments.mass), jnp.sum(tree.masses_sorted))
    _assert_matches_direct_sums(
        tree, tree.positions_sorted, tree.masses_sorted, moments
    )


def test_root_mass_and_com():
    tree, pos_sorted, mass_sorted = _build_sample_tree()
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    num_internal = tree.num_internal_nodes
    parents = tree.parent[:num_internal]
    root_mask = parents == -1
    assert int(root_mask.sum()) == 1
    root_index = int(jnp.argmax(root_mask.astype(INDEX_DTYPE)))

    expected_mass = jnp.sum(mass_sorted)
    expected_com = (
        jnp.sum(
            mass_sorted[:, None] * pos_sorted,
            axis=0,
        )
        / expected_mass
    )

    assert jnp.isclose(moments.mass[root_index], expected_mass)
    assert jnp.allclose(moments.center_of_mass[root_index], expected_com)
    _assert_matches_direct_sums(tree, pos_sorted, mass_sorted, moments)


def test_zero_mass_nodes_return_zero_center():
    positions = jnp.array(
        [
            [-0.2, 0.3, 0.4],
            [0.1, -0.5, 0.2],
        ]
    )
    masses = jnp.zeros((positions.shape[0],))
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    assert jnp.allclose(moments.mass, 0.0)
    assert jnp.allclose(moments.center_of_mass, 0.0)


def test_prefix_range_sums_do_not_cancel_at_large_n():
    """A small range at the END of a long prefix keeps its own precision.

    In float32 a node's mass is the difference of two prefixes that grow to the total
    mass; at N = 4e6 with masses 1/N that difference for 20 particles carries a
    relative error of order 1e-2. The wide accumulation must not.
    """
    from yggdrax.tree_moments import _accumulation_dtype, _prefix_range_sums

    # not a power of two (1/2^k masses sum exactly in float32) and positive
    # positions, so both prefixes grow to O(1) as they do in Morton order
    n = 4_000_000
    masses = jnp.full((n,), 1.0 / n, dtype=jnp.float32)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.uniform(0.5, 1.5, n), dtype=jnp.float32)
    starts = jnp.asarray([n - 20], dtype=INDEX_DTYPE)
    ends = jnp.asarray([n], dtype=INDEX_DTYPE)
    exact_m = float(np.sum(np.asarray(masses[-20:], np.float64)))
    exact_c = float(
        np.sum(np.asarray(masses[-20:], np.float64) * np.asarray(x[-20:], np.float64))
        / exact_m
    )

    acc = _accumulation_dtype(jnp.float32)
    m = _prefix_range_sums(masses.astype(acc), starts, ends)
    c = _prefix_range_sums((masses * x).astype(acc), starts, ends) / m
    assert abs(float(c[0]) - exact_c) < 1e-6

    # the control: the same range query accumulated in float32 is visibly wrong
    m32 = _prefix_range_sums(masses, starts, ends)
    c32 = _prefix_range_sums(masses * x, starts, ends) / m32
    assert abs(float(c32[0]) - exact_c) > 1e-4


def test_leaf_centres_of_mass_are_exact_at_moderate_n():
    """Leaf centres from the prefix sums equal per-leaf float64 means (250k particles)."""
    n = 250_000
    rng = np.random.default_rng(1)
    positions = jnp.asarray(rng.uniform(0.0, 2.0, (n, 3)), dtype=jnp.float32)
    masses = jnp.full((n,), 1.0 / n, dtype=jnp.float32)
    bounds = (
        jnp.array([0.0, 0.0, 0.0], jnp.float32),
        jnp.array([2.0, 2.0, 2.0], jnp.float32),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions, masses, bounds, return_reordered=True, leaf_size=16
    )
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)
    assert moments.center_of_mass.dtype == pos_sorted.dtype

    ranges = np.asarray(tree.node_ranges)
    num_internal = int(np.asarray(tree.left_child).shape[0])
    p64 = np.asarray(pos_sorted, np.float64)
    m64 = np.asarray(mass_sorted, np.float64)
    com = np.asarray(moments.center_of_mass, np.float64)
    worst = 0.0
    for node in range(num_internal, ranges.shape[0]):
        s, e = int(ranges[node, 0]), int(ranges[node, 1]) + 1
        if e <= s:
            continue
        exact = (m64[s:e, None] * p64[s:e]).sum(0) / m64[s:e].sum()
        worst = max(worst, float(np.abs(com[node] - exact).max()))
    # float32 rounding of the result only (the float32 prefixes were ~1e-4 off here)
    assert worst < 1e-6
