"""Tests for radix tree construction."""

import jax.numpy as jnp
import numpy as np

from yggdrasil.geometry import compute_tree_geometry
from yggdrasil.tree import (
    RadixTreeWorkspace,
    build_fixed_depth_tree,
    build_fixed_depth_tree_jit,
    build_tree,
    build_tree_jit,
    reorder_particles_by_indices,
)

DEFAULT_TEST_LEAF_SIZE = 1


def _max_leaf_count(tree) -> int:
    leaf_ranges = tree.node_ranges[tree.num_internal_nodes :]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    return int(jnp.max(counts))


def test_tree_construction():
    """Test basic tree construction."""
    # Create simple particle distribution
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [-0.5, -0.5, -0.5],
        ]
    )
    masses = jnp.array([1.0, 1.0, 1.0, 1.0])

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    # Check basic properties
    assert tree.num_particles == 4
    assert tree.num_internal_nodes == 3
    assert len(tree.particle_indices) == 4
    assert len(tree.morton_codes) == 4


def test_tree_sorted_particles():
    """Test that particles are sorted by Morton code."""
    positions = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    masses = jnp.array([1.0, 1.0, 1.0])

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    # Morton codes should be sorted
    assert jnp.all(tree.morton_codes[:-1] <= tree.morton_codes[1:])


def test_single_particle_tree():
    """Test tree with single particle."""
    positions = jnp.array([[0.0, 0.0, 0.0]])
    masses = jnp.array([1.0])

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    assert tree.num_particles == 1
    assert tree.num_internal_nodes == 0


def test_tree_node_structure():
    """Test tree node parent-child relationships."""
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    masses = jnp.array([1.0, 1.0])

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    # With 2 particles, should have 1 internal node
    assert tree.num_particles == 2
    assert tree.num_internal_nodes == 1

    # Find root (internal node with no parent)
    parents = tree.parent[: tree.num_internal_nodes]
    roots = [i for i, p in enumerate(parents) if int(p) == -1]
    assert len(roots) == 1
    root_idx = roots[0]

    left = int(tree.left_child[root_idx])
    right = int(tree.right_child[root_idx])

    # Both children should exist
    assert left != -1
    assert right != -1

    # Children should point back to parent
    assert int(tree.parent[left]) == root_idx
    assert int(tree.parent[right]) == root_idx


def test_child_leaf_flags():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.9, 0.9, 0.9],
        ]
    )
    masses = jnp.ones((3,))

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    assert tree.left_is_leaf.shape == (tree.num_internal_nodes,)
    assert tree.right_is_leaf.shape == (tree.num_internal_nodes,)
    assert tree.left_is_leaf.dtype == jnp.bool_
    assert tree.right_is_leaf.dtype == jnp.bool_

    num_internal = tree.num_internal_nodes
    expected_left = tree.left_child >= num_internal
    expected_right = tree.right_child >= num_internal

    assert jnp.array_equal(tree.left_is_leaf, expected_left)
    assert jnp.array_equal(tree.right_is_leaf, expected_right)


def test_duplicate_codes_stability():
    """When all codes are equal, sort should be stable by index."""
    n = 8
    positions = jnp.zeros((n, 3))  # identical positions
    masses = jnp.ones((n,))
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    # lexsort by (idx, codes) should return [0,1,...,n-1]
    from yggdrasil.dtypes import INDEX_DTYPE

    expected = jnp.arange(n, dtype=INDEX_DTYPE)
    assert jnp.array_equal(tree.particle_indices, expected)


def _dfs_leaf_order(tree):
    """Return list of leaf indices (0..num_leaves-1) from DFS (left-first)."""
    num_internal = tree.num_internal_nodes
    total = int(tree.parent.shape[0])

    # Find root internal node (parent == -1)
    parents = tree.parent[:num_internal]
    roots = [i for i, p in enumerate(parents) if int(p) == -1]
    assert len(roots) == 1
    root = roots[0]

    stack = [root]
    leaves = []
    while stack:
        node = stack.pop()
        if node >= num_internal:
            # leaf
            leaves.append(node - num_internal)
        else:
            # internal: push right then left (so left is visited first)
            r = int(tree.right_child[node])
            left = int(tree.left_child[node])
            assert 0 <= r < total
            assert 0 <= left < total
            stack.append(r)
            stack.append(left)
    return leaves


def test_dfs_leaf_order_matches_sorted():
    """DFS leaf order should match sorted particle order 0..n-1."""
    positions = jnp.array(
        [
            [0.2, 0.1, -0.3],
            [-0.7, 0.4, 0.0],
            [0.5, -0.2, 0.8],
            [0.0, 0.0, 0.0],
            [0.9, 0.9, -0.9],
            [-0.1, -0.2, 0.3],
        ]
    )
    masses = jnp.ones((positions.shape[0],))

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    leaves = _dfs_leaf_order(tree)
    expected = list(range(tree.num_particles))
    assert leaves == expected


def test_reorder_particles_and_inverse():
    """Reorder by Morton order and invert mapping."""
    positions = jnp.array(
        [
            [0.2, 0.1, -0.3],
            [-0.7, 0.4, 0.0],
            [0.5, -0.2, 0.8],
            [0.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0, 4.0])

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    pos_s, mass_s, inv = reorder_particles_by_indices(
        positions, masses, tree.particle_indices
    )

    # Inverse should undo the permutation
    pos_back = pos_s[inv]
    mass_back = mass_s[inv]

    assert jnp.allclose(pos_back, positions)
    assert jnp.allclose(mass_back, masses)


def test_build_tree_return_reordered():
    positions = jnp.array(
        [
            [0.2, 0.1, -0.3],
            [-0.7, 0.4, 0.0],
            [0.5, -0.2, 0.8],
            [0.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0, 4.0])

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_s, mass_s, inv = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    # Reordered equals gather by permutation
    assert jnp.allclose(pos_s, positions[tree.particle_indices])
    assert jnp.allclose(mass_s, masses[tree.particle_indices])

    # Inverse recovers original
    assert jnp.allclose(pos_s[inv], positions)
    assert jnp.allclose(mass_s[inv], masses)


def test_leaf_size_greater_than_one():
    positions = jnp.array(
        [
            [0.2, 0.1, -0.3],
            [-0.7, 0.4, 0.0],
            [0.5, -0.2, 0.8],
            [0.0, 0.0, 0.0],
            [0.9, 0.9, -0.9],
            [-0.1, -0.2, 0.3],
        ]
    )
    masses = jnp.ones((positions.shape[0],))

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree = build_tree(positions, masses, bounds, leaf_size=2)

    assert tree.num_particles == positions.shape[0]
    assert tree.num_internal_nodes == 2  # ceil(6/2) - 1

    leaf_ranges = tree.node_ranges[tree.num_internal_nodes :].tolist()
    covered = []
    for start, end in leaf_ranges:
        assert 0 <= start <= end < tree.num_particles
        covered.extend(range(start, end + 1))
        assert (end - start + 1) <= 2

    assert covered == list(range(tree.num_particles))


def test_leaf_size_larger_than_particle_count():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ]
    )
    masses = jnp.ones((3,))

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree = build_tree(positions, masses, bounds, leaf_size=8)

    assert tree.num_internal_nodes == 0
    assert tree.node_ranges.shape[0] == 1

    start, end = tree.node_ranges[0]
    assert int(start) == 0
    assert int(end) == positions.shape[0] - 1


def test_fixed_depth_tree_uniform_leaf_levels():
    rng_positions = jnp.linspace(-1.0, 1.0, 64)
    grid = jnp.meshgrid(rng_positions, rng_positions, rng_positions)
    positions = jnp.stack(grid, axis=-1).reshape(-1, 3)
    masses = jnp.ones((positions.shape[0],))

    bounds = (
        jnp.array([-1.5, -1.5, -1.5]),
        jnp.array([1.5, 1.5, 1.5]),
    )

    target = 32
    tree = build_fixed_depth_tree(
        positions,
        masses,
        bounds,
        target_leaf_particles=target,
        refine_local=False,
    )

    num_internal = tree.num_internal_nodes
    total_nodes = tree.parent.shape[0]
    num_leaves = total_nodes - num_internal

    depth_vals = tree.leaf_depths
    assert int(jnp.min(depth_vals)) == int(jnp.max(depth_vals))
    depth = int(depth_vals[0])
    expected_leaves = 1 << (3 * depth)

    assert num_leaves > 0
    assert num_internal == num_leaves - 1
    assert num_leaves == expected_leaves


def test_fixed_depth_tree_depth_scales_with_target():
    positions = jnp.array(
        [[float(i), 0.0, 0.0] for i in range(128)],
        dtype=jnp.float32,
    )
    masses = jnp.ones((positions.shape[0],))
    bounds = (
        jnp.array([-10.0, -1.0, -1.0]),
        jnp.array([10.0, 1.0, 1.0]),
    )

    tree_loose = build_fixed_depth_tree(
        positions,
        masses,
        bounds,
        target_leaf_particles=64,
    )
    tree_tight = build_fixed_depth_tree(
        positions,
        masses,
        bounds,
        target_leaf_particles=8,
    )

    max_loose = _max_leaf_count(tree_loose)
    max_tight = _max_leaf_count(tree_tight)

    assert tree_tight.num_internal_nodes >= tree_loose.num_internal_nodes
    assert max_tight <= max_loose


def test_fixed_depth_morton_geometry_contains_particles():
    coords = jnp.linspace(-0.75, 0.75, 4)
    grid = jnp.meshgrid(coords, coords, coords)
    positions = jnp.stack(grid, axis=-1).reshape(-1, 3)
    masses = jnp.ones((positions.shape[0],))

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, positions_sorted, *_ = build_fixed_depth_tree(
        positions,
        masses,
        bounds,
        target_leaf_particles=8,
        return_reordered=True,
        refine_local=False,
    )

    depth_vals = tree.leaf_depths
    assert int(jnp.min(depth_vals)) == int(jnp.max(depth_vals))
    depth = int(depth_vals[0])
    expected_leaves = 1 << (3 * depth)
    num_internal = int(tree.num_internal_nodes)
    num_leaves = int(tree.parent.shape[0]) - num_internal
    assert num_leaves == expected_leaves

    geometry = compute_tree_geometry(tree, positions_sorted)
    leaf_ranges = np.asarray(tree.node_ranges[num_internal:])
    centers = np.asarray(geometry.center[num_internal:])
    half_extent = np.asarray(geometry.half_extent[num_internal:])
    positions_np = np.asarray(positions_sorted)
    eps = 1e-9

    for idx, (start, end) in enumerate(leaf_ranges):
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            continue
        pts = positions_np[start_i : end_i + 1]
        if pts.size == 0:
            continue
        cell_min = centers[idx] - half_extent[idx]
        cell_max = centers[idx] + half_extent[idx]
        assert np.all(pts >= cell_min - eps)
        assert np.all(pts <= cell_max + eps)

    domain_half_extent = 0.5 * (bounds[1] - bounds[0])
    root_extent = geometry.max_extent[0]
    assert jnp.allclose(
        root_extent,
        jnp.max(domain_half_extent),
        rtol=1e-5,
        atol=1e-5,
    )


def test_workspace_reuse_roundtrip():
    positions = jnp.array(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.5, -0.6],
            [-0.7, 0.8, 0.9],
            [-0.4, -0.3, 0.2],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0, 4.0])

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree1, workspace = build_tree(
        positions,
        masses,
        bounds,
        return_workspace=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    assert isinstance(workspace, RadixTreeWorkspace)
    tree2, workspace2 = build_tree(
        positions,
        masses,
        bounds,
        workspace=workspace,
        return_workspace=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    assert jnp.array_equal(tree1.parent, tree2.parent)
    assert workspace2.parent.shape == workspace.parent.shape


def test_return_reordered_with_workspace():
    positions = jnp.array(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.3, 0.4],
            [-0.5, -0.6, -0.7],
        ]
    )
    masses = jnp.array([1.0, 1.5, 2.0])

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    result = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        return_workspace=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    assert len(result) == 5
    tree, pos_s, mass_s, inv, workspace = result
    assert tree.num_particles == positions.shape[0]
    assert isinstance(workspace, RadixTreeWorkspace)
    assert jnp.allclose(pos_s[inv], positions)


def test_build_tree_jit_matches_eager():
    positions = jnp.array(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.5, -0.6],
            [-0.7, 0.8, 0.9],
            [-0.4, -0.3, 0.2],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0, 4.0])

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    eager = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    jitted = build_tree_jit(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    assert isinstance(jitted, tuple) and isinstance(eager, tuple)
    assert len(eager) == len(jitted) == 4

    tree_eager, pos_eager, mass_eager, inv_eager = eager
    tree_jit, pos_jit, mass_jit, inv_jit = jitted

    assert jnp.array_equal(tree_eager.parent, tree_jit.parent)
    assert jnp.array_equal(tree_eager.left_child, tree_jit.left_child)
    assert jnp.array_equal(tree_eager.right_child, tree_jit.right_child)
    assert jnp.array_equal(tree_eager.node_ranges, tree_jit.node_ranges)
    assert jnp.array_equal(pos_eager, pos_jit)
    assert jnp.array_equal(mass_eager, mass_jit)
    assert jnp.array_equal(inv_eager, inv_jit)


def test_fixed_depth_tree_jit_matches_eager():
    positions = jnp.array(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.5, -0.6],
            [-0.7, 0.8, 0.9],
            [-0.4, -0.3, 0.2],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0, 4.0])

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    eager = build_fixed_depth_tree(
        positions,
        masses,
        bounds,
        target_leaf_particles=2,
        return_reordered=True,
        refine_local=False,
    )
    jitted = build_fixed_depth_tree_jit(
        positions,
        masses,
        bounds,
        target_leaf_particles=2,
        return_reordered=True,
        refine_local=False,
    )

    assert isinstance(eager, tuple) and isinstance(jitted, tuple)
    assert len(eager) == len(jitted) == 4

    tree_eager, pos_eager, mass_eager, inv_eager = eager
    tree_jit, pos_jit, mass_jit, inv_jit = jitted

    assert jnp.array_equal(tree_eager.parent, tree_jit.parent)
    assert jnp.array_equal(tree_eager.left_child, tree_jit.left_child)
    assert jnp.array_equal(tree_eager.right_child, tree_jit.right_child)
    assert jnp.array_equal(tree_eager.node_ranges, tree_jit.node_ranges)
    assert jnp.array_equal(pos_eager, pos_jit)
    assert jnp.array_equal(mass_eager, mass_jit)
    assert jnp.array_equal(inv_eager, inv_jit)
