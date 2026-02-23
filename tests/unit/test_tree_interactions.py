"""Tests for tree interaction list construction."""

import jax.numpy as jnp

from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import (
    _compute_effective_extents,
    _compute_leaf_effective_extents,
    _interaction_capacity_candidates,
    build_leaf_neighbor_lists,
    build_well_separated_interactions,
    interactions_for_node,
    neighbors_for_leaf,
)
from yggdrax.tree import build_tree

DEFAULT_TEST_LEAF_SIZE = 1


def _build_sample_tree():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.9, -0.1, 0.05],
            [0.7, 0.0, -0.05],
            [0.9, -0.1, 0.1],
        ]
    )
    masses = jnp.ones((positions.shape[0],))
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, _mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    return tree, geometry


def _mac_accept_fn(tree, geometry, theta):
    parent = tree.parent
    # bh uses box max half-extent.
    extents = geometry.max_extent
    far_extents = _compute_effective_extents(parent, extents)
    num_internal = int(tree.num_internal_nodes)
    leaf_extents = _compute_leaf_effective_extents(
        parent,
        extents,
        num_internal,
    )
    from yggdrax.dtypes import INDEX_DTYPE

    indices = jnp.arange(parent.shape[0], dtype=INDEX_DTYPE)
    mac_extents = jnp.where(
        indices >= num_internal,
        leaf_extents,
        far_extents,
    )
    centers = geometry.center
    theta_sq = float(theta) ** 2

    def accept(target, source):
        if target == source:
            return False
        delta = centers[target] - centers[source]
        dist_sq = float(jnp.dot(delta, delta))
        if dist_sq <= 0.0:
            return False
        extent_sum = float(mac_extents[target] + mac_extents[source])
        return extent_sum * extent_sum <= theta_sq * dist_sq

    return accept


def _mac_accept_fn_engblom(tree, geometry, theta):
    parent = tree.parent
    # engblom uses bounding-sphere radii.
    extents = geometry.radius
    far_extents = _compute_effective_extents(parent, extents)
    num_internal = int(tree.num_internal_nodes)
    leaf_extents = _compute_leaf_effective_extents(
        parent,
        extents,
        num_internal,
    )
    from yggdrax.dtypes import INDEX_DTYPE

    indices = jnp.arange(parent.shape[0], dtype=INDEX_DTYPE)
    mac_extents = jnp.where(
        indices >= num_internal,
        leaf_extents,
        far_extents,
    )
    centers = geometry.center

    theta_val = float(theta)
    theta_sq = theta_val * theta_val

    def accept(target, source):
        if target == source:
            return False
        delta = centers[target] - centers[source]
        dist_sq = float(jnp.dot(delta, delta))
        if dist_sq <= 0.0:
            return False
        rt = float(mac_extents[target])
        rs = float(mac_extents[source])
        R = max(rt, rs)
        r = min(rt, rs)
        lhs = R + theta_val * r
        return lhs * lhs <= theta_sq * dist_sq

    return accept


def _mac_accept_fn_dehnen(tree, geometry, theta):
    """Dehnen (2014) Eq. (6) style MAC.

    In the implementation, the symmetric opening-angle check is expressed as:
        (rt + rs)^2 <= theta^2 * d^2
    where rt/rs are conservative per-node extents.
    """

    # Dehnen uses bounding-sphere radii.
    parent = tree.parent
    extents = geometry.radius
    far_extents = _compute_effective_extents(parent, extents)
    num_internal = int(tree.num_internal_nodes)
    leaf_extents = _compute_leaf_effective_extents(
        parent,
        extents,
        num_internal,
    )
    from yggdrax.dtypes import INDEX_DTYPE

    indices = jnp.arange(parent.shape[0], dtype=INDEX_DTYPE)
    mac_extents = jnp.where(
        indices >= num_internal,
        leaf_extents,
        far_extents,
    )
    centers = geometry.center
    theta_sq = float(theta) ** 2

    def accept(target, source):
        if target == source:
            return False
        delta = centers[target] - centers[source]
        dist_sq = float(jnp.dot(delta, delta))
        if dist_sq <= 0.0:
            return False
        extent_sum = float(mac_extents[target] + mac_extents[source])
        return extent_sum * extent_sum <= theta_sq * dist_sq

    return accept


def test_mac_bh_and_dehnen_can_differ_for_elongated_nodes():
    # Construct extents directly to isolate MAC geometry behavior.
    # Elongated box: max_extent = 1, sphere radius = sqrt(3).
    rt_box = 1.0
    rt_sphere = float(jnp.sqrt(3.0))
    rs_box = 1.0
    rs_sphere = float(jnp.sqrt(3.0))
    theta = 0.8

    # Choose separation so BH accepts but Dehnen rejects.
    # BH accept: (2)^2 <= theta^2 d^2  -> d >= 2/theta = 2.5
    # Dehnen reject: (2*sqrt3)^2 > theta^2 d^2 -> d < 2*sqrt3/theta ~ 4.33
    d = 3.0

    bh_ok = (rt_box + rs_box) ** 2 <= (theta * theta) * (d * d)
    dehnen_ok = (rt_sphere + rs_sphere) ** 2 <= (theta * theta) * (d * d)

    assert bh_ok is True
    assert dehnen_ok is False


def _ancestors(tree, node):
    parent = tree.parent
    chain = set()
    current = int(parent[node])
    while current >= 0:
        chain.add(current)
        current = int(parent[current])
    return chain


def test_interaction_list_excludes_ancestors():
    tree, geometry = _build_sample_tree()
    interactions = build_well_separated_interactions(tree, geometry, theta=2.0)

    total = tree.parent.shape[0]
    for node in range(total):
        indices = interactions_for_node(interactions, node)
        ancestors = _ancestors(tree, node)
        for src in indices:
            assert int(src) not in ancestors
            # interactions should be symmetric
            reverse = interactions_for_node(interactions, int(src))
            assert node in set(int(x) for x in reverse)


def test_far_leaves_present_and_near_leaves_skipped():
    tree, geometry = _build_sample_tree()
    interactions = build_well_separated_interactions(tree, geometry, theta=2.0)
    mac_accept = _mac_accept_fn(tree, geometry, theta=2.0)

    num_internal = int(tree.num_internal_nodes)
    total = tree.parent.shape[0]
    leaves = range(num_internal, total)

    centers = geometry.center
    left_leaves = [idx for idx in leaves if float(centers[idx, 0]) < 0.0]
    right_leaves = [idx for idx in leaves if float(centers[idx, 0]) > 0.0]

    assert left_leaves and right_leaves

    # Every recorded interaction must satisfy the MAC accept condition.
    for left in left_leaves:
        actual = {
            int(x)
            for x in interactions_for_node(
                interactions,
                left,
            )
        }
        assert actual, "each leaf should have at least one far interaction"
        for src in actual:
            assert mac_accept(int(left), src)

    for right in right_leaves:
        actual = {
            int(x)
            for x in interactions_for_node(
                interactions,
                right,
            )
        }
        assert actual, "each leaf should have at least one far interaction"
        for src in actual:
            assert mac_accept(int(right), src)


def test_interaction_list_is_level_ordered():
    tree, geometry = _build_sample_tree()
    interactions = build_well_separated_interactions(tree, geometry, theta=2.0)

    levels = interactions.target_levels
    assert levels.shape[0] == interactions.sources.shape[0]
    if levels.shape[0] > 1:
        diffs = levels[1:] - levels[:-1]
        assert jnp.all(diffs >= 0)

    num_levels = int(tree.num_levels)
    level_offsets = interactions.level_offsets
    assert level_offsets.shape[0] == num_levels + 1
    assert int(level_offsets[0]) == 0
    assert int(level_offsets[-1]) == int(levels.shape[0])

    for level in range(num_levels):
        start = int(level_offsets[level])
        end = int(level_offsets[level + 1])
        if end == start:
            continue
        first_level = int(levels[start])
        last_level = int(levels[end - 1])
        assert first_level == level
        assert last_level == level


def test_leaf_neighbor_list_contains_only_near_leaves():
    tree, geometry = _build_sample_tree()
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.4)
    mac_accept = _mac_accept_fn(tree, geometry, theta=0.4)

    # Offsets must align with number of leaf indices
    assert neighbor_list.offsets.shape[0] == (neighbor_list.leaf_indices.shape[0] + 1)

    parent = tree.parent
    num_internal = int(tree.num_internal_nodes)
    total = parent.shape[0]
    leaves = range(num_internal, total)

    leaves_list = list(leaves)
    for leaf in leaves_list:
        neighbor_indices = {
            int(x)
            for x in neighbors_for_leaf(
                neighbor_list,
                leaf,
            )
        }
        expected_neighbors = {
            idx
            for idx in leaves_list
            if idx != leaf and not mac_accept(int(leaf), int(idx))
        }
        assert neighbor_indices == expected_neighbors

    # Symmetry check
    for leaf in leaves:
        row = {int(x) for x in neighbors_for_leaf(neighbor_list, leaf)}
        for neighbor in row:
            round_trip = {
                int(x)
                for x in neighbors_for_leaf(
                    neighbor_list,
                    int(neighbor),
                )
            }
            assert leaf in round_trip


def test_leaf_neighbor_list_contains_only_near_leaves_engblom():
    tree, geometry = _build_sample_tree()
    neighbor_list = build_leaf_neighbor_lists(
        tree,
        geometry,
        theta=0.4,
        mac_type="engblom",
    )
    mac_accept = _mac_accept_fn_engblom(tree, geometry, theta=0.4)

    assert neighbor_list.offsets.shape[0] == (neighbor_list.leaf_indices.shape[0] + 1)

    parent = tree.parent
    num_internal = int(tree.num_internal_nodes)
    total = parent.shape[0]
    leaves = range(num_internal, total)

    leaves_list = list(leaves)
    for leaf in leaves_list:
        neighbor_indices = {int(x) for x in neighbors_for_leaf(neighbor_list, leaf)}
        expected_neighbors = {
            idx
            for idx in leaves_list
            if idx != leaf and not mac_accept(int(leaf), int(idx))
        }
        assert neighbor_indices == expected_neighbors


def test_well_separated_interactions_accept_only_mac_engblom():
    tree, geometry = _build_sample_tree()
    interactions = build_well_separated_interactions(
        tree,
        geometry,
        theta=2.0,
        mac_type="engblom",
    )
    mac_accept = _mac_accept_fn_engblom(tree, geometry, theta=2.0)

    num_internal = int(tree.num_internal_nodes)
    total = tree.parent.shape[0]
    leaves = range(num_internal, total)

    for leaf in leaves:
        actual = {int(x) for x in interactions_for_node(interactions, leaf)}
        assert actual
        for src in actual:
            assert mac_accept(int(leaf), int(src))


def test_leaf_neighbor_list_contains_only_near_leaves_dehnen():
    tree, geometry = _build_sample_tree()
    neighbor_list = build_leaf_neighbor_lists(
        tree,
        geometry,
        theta=0.4,
        mac_type="dehnen",
    )
    mac_accept = _mac_accept_fn_dehnen(tree, geometry, theta=0.4)

    assert neighbor_list.offsets.shape[0] == (neighbor_list.leaf_indices.shape[0] + 1)

    parent = tree.parent
    num_internal = int(tree.num_internal_nodes)
    total = parent.shape[0]
    leaves = range(num_internal, total)
    leaves_list = list(leaves)

    for leaf in leaves_list:
        neighbor_indices = {int(x) for x in neighbors_for_leaf(neighbor_list, leaf)}
        expected_neighbors = {
            idx
            for idx in leaves_list
            if idx != leaf and not mac_accept(int(leaf), int(idx))
        }
        assert neighbor_indices == expected_neighbors


def test_well_separated_interactions_accept_only_mac_dehnen():
    tree, geometry = _build_sample_tree()
    interactions = build_well_separated_interactions(
        tree,
        geometry,
        theta=2.0,
        mac_type="dehnen",
    )
    mac_accept = _mac_accept_fn_dehnen(tree, geometry, theta=2.0)

    num_internal = int(tree.num_internal_nodes)
    total = tree.parent.shape[0]
    leaves = range(num_internal, total)

    for leaf in leaves:
        actual = {int(x) for x in interactions_for_node(interactions, leaf)}
        assert actual
        for src in actual:
            assert mac_accept(int(leaf), int(src))


def test_leaf_neighbor_list_handles_single_leaf():
    positions = jnp.array([[0.0, 0.0, 0.0]])
    masses = jnp.array([1.0])
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, _mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)

    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.4)

    assert neighbor_list.leaf_indices.shape[0] == 1
    assert neighbor_list.neighbors.shape[0] == 0
    assert neighbor_list.offsets.tolist() == [0, 0]


def test_interaction_capacity_auto_scales_with_tree():
    candidates_small, allow_small = _interaction_capacity_candidates(None, 128)
    candidates_large, allow_large = _interaction_capacity_candidates(
        None,
        200_000,
    )

    assert not allow_small
    assert not allow_large
    assert len(candidates_small) == 1
    assert len(candidates_large) == 1
    assert candidates_small[0] >= 2048
    assert candidates_large[0] > candidates_small[0]
