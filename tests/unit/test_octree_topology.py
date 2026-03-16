"""Tests for explicit octree metadata on top of radix-compatible trees."""

import jax
import jax.numpy as jnp

from yggdrax import (
    DualTreeTraversalConfig,
    Tree,
    _interactions_impl,
    build_interactions_and_neighbors,
    build_octree_native_far_pairs,
    build_octree_native_neighbor_lists,
    compute_tree_geometry,
)

_TEST_TRAVERSAL_CFG = DualTreeTraversalConfig(
    max_pair_queue=1024,
    process_block=64,
    max_interactions_per_node=256,
    max_neighbors_per_leaf=256,
)


def test_fixed_depth_octree_populates_eight_root_children():
    positions = jnp.array(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.75, 0.75, 0.25],
            [0.25, 0.25, 0.75],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.75, 0.75],
        ],
        dtype=jnp.float32,
    )
    masses = jnp.ones((8,), dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type="octree",
        build_mode="fixed_depth",
        bounds=(
            jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
        ),
        return_reordered=True,
        target_leaf_particles=1,
        max_depth=1,
        refine_local=False,
    )

    assert tree.tree_type == "octree"
    assert tree.oct_num_nodes == 9
    assert int(tree.oct_child_counts[0]) == 8
    assert jnp.all(tree.oct_children[0] >= 0)
    assert jnp.all(tree.oct_node_depths[tree.oct_children[0]] == 1)
    assert jnp.array_equal(
        tree.oct_leaf_nodes[: tree.oct_num_leaf_nodes],
        tree.oct_children[0],
    )


def test_octree_radix_node_mapping_covers_all_nodes():
    positions = jnp.array(
        [
            [0.05, 0.10, 0.15],
            [0.18, 0.21, 0.22],
            [0.31, 0.30, 0.35],
            [0.44, 0.41, 0.48],
            [0.55, 0.62, 0.58],
            [0.68, 0.70, 0.66],
            [0.82, 0.84, 0.79],
            [0.93, 0.92, 0.95],
        ],
        dtype=jnp.float32,
    )
    masses = jnp.ones((8,), dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type="octree",
        build_mode="adaptive",
        return_reordered=True,
        leaf_size=2,
    )

    assert tree.radix_node_to_oct.shape == (tree.num_nodes,)
    assert tree.radix_leaf_to_oct.shape == (tree.num_leaves,)
    assert jnp.all(tree.radix_node_to_oct >= 0)
    assert jnp.all(tree.radix_node_to_oct < tree.oct_num_nodes)
    assert jnp.all(tree.radix_leaf_to_oct < tree.oct_num_nodes)


def test_octree_matches_radix_interaction_counts():
    positions = jnp.array(
        [
            [0.10, 0.10, 0.10],
            [0.15, 0.12, 0.14],
            [0.20, 0.24, 0.28],
            [0.35, 0.32, 0.30],
            [0.48, 0.46, 0.50],
            [0.70, 0.72, 0.74],
        ],
        dtype=jnp.float32,
    )
    masses = jnp.linspace(1.0, 2.0, positions.shape[0], dtype=jnp.float32)
    radix = Tree.from_particles(
        positions,
        masses,
        tree_type="radix",
        return_reordered=True,
        leaf_size=2,
    )
    octree = Tree.from_particles(
        positions,
        masses,
        tree_type="octree",
        return_reordered=True,
        leaf_size=2,
    )

    radix_geom = compute_tree_geometry(radix, radix.positions_sorted)
    oct_geom = compute_tree_geometry(octree, octree.positions_sorted)
    radix_interactions, radix_neighbors = build_interactions_and_neighbors(
        radix,
        radix_geom,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )
    oct_interactions, oct_neighbors = build_interactions_and_neighbors(
        octree,
        oct_geom,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )

    assert jnp.array_equal(radix.positions_sorted, octree.positions_sorted)
    assert jnp.array_equal(radix.masses_sorted, octree.masses_sorted)
    assert jnp.array_equal(radix_interactions.counts, oct_interactions.counts)
    assert jnp.array_equal(radix_neighbors.counts, oct_neighbors.counts)


def test_octree_native_far_pairs_stay_in_octree_node_space():
    key = jax.random.PRNGKey(7)
    positions = jax.random.uniform(
        key,
        (64, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.linspace(1.0, 2.0, positions.shape[0], dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type="octree",
        return_reordered=True,
        leaf_size=8,
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    far_pairs = build_octree_native_far_pairs(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )

    assert far_pairs.sources.ndim == 1
    assert far_pairs.targets.ndim == 1
    assert far_pairs.tags.ndim == 1
    assert far_pairs.sources.shape == far_pairs.targets.shape == far_pairs.tags.shape
    if far_pairs.sources.shape[0] > 0:
        assert int(jnp.min(far_pairs.sources)) >= 0
        assert int(jnp.min(far_pairs.targets)) >= 0
        assert int(jnp.max(far_pairs.sources)) < int(tree.oct_num_nodes)
        assert int(jnp.max(far_pairs.targets)) < int(tree.oct_num_nodes)


def test_octree_native_neighbor_lists_stay_in_octree_leaf_space():
    key = jax.random.PRNGKey(11)
    positions = jax.random.uniform(
        key,
        (64, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.linspace(1.0, 2.0, positions.shape[0], dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type="octree",
        return_reordered=True,
        leaf_size=8,
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    native_neighbors = build_octree_native_neighbor_lists(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )

    leaf_indices = native_neighbors.leaf_indices
    assert leaf_indices.ndim == 1
    assert native_neighbors.counts.shape == leaf_indices.shape
    assert native_neighbors.offsets.shape == (leaf_indices.shape[0] + 1,)
    assert jnp.array_equal(
        native_neighbors.offsets[1:], jnp.cumsum(native_neighbors.counts)
    )
    if leaf_indices.shape[0] > 0:
        assert int(jnp.min(leaf_indices)) >= 0
        assert int(jnp.max(leaf_indices)) < int(tree.oct_num_nodes)
        carrier_nodes = jnp.unique(tree.radix_leaf_to_oct)
        assert jnp.array_equal(jnp.sort(leaf_indices), jnp.sort(carrier_nodes))
    if native_neighbors.neighbors.shape[0] > 0:
        assert int(jnp.min(native_neighbors.neighbors)) >= 0
        assert int(jnp.max(native_neighbors.neighbors)) < int(tree.oct_num_nodes)
        carrier_nodes = jnp.unique(tree.radix_leaf_to_oct)
        assert jnp.all(jnp.isin(native_neighbors.neighbors, carrier_nodes))
    native_map = native_neighbors.particle_order_to_native_leaf
    assert native_map.shape == leaf_indices.shape
    assert jnp.array_equal(jnp.sort(native_map), jnp.arange(leaf_indices.shape[0]))


def test_octree_tree_is_jittable_pytree():
    positions = jnp.array(
        [
            [0.10, 0.10, 0.10],
            [0.15, 0.12, 0.14],
            [0.20, 0.24, 0.28],
            [0.35, 0.32, 0.30],
        ],
        dtype=jnp.float32,
    )
    masses = jnp.ones((4,), dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type="octree",
        return_reordered=True,
        leaf_size=2,
    )

    jitted = jax.jit(
        lambda t: t.positions_sorted.shape[0]
        + jnp.sum(t.oct_valid_mask.astype(jnp.int32))
    )
    result = jitted(tree)

    assert int(result) >= 4


def test_build_explicit_octree_metadata_is_jit_compatible():
    """``build_explicit_octree_metadata`` must execute inside ``jax.jit``."""
    from yggdrax.octree import build_explicit_octree_metadata

    positions = jnp.array(
        [
            [0.10, 0.10, 0.10],
            [0.15, 0.12, 0.14],
            [0.20, 0.24, 0.28],
            [0.35, 0.32, 0.30],
            [0.55, 0.58, 0.53],
            [0.70, 0.72, 0.74],
            [0.82, 0.84, 0.78],
            [0.95, 0.94, 0.96],
        ],
        dtype=jnp.float32,
    )
    masses = jnp.ones((8,), dtype=jnp.float32)
    radix_tree = Tree.from_particles(
        positions,
        masses,
        tree_type="radix",
        return_reordered=True,
        leaf_size=2,
    )
    topology = radix_tree.topology

    # Eager reference run.
    eager = build_explicit_octree_metadata(topology)

    # JIT run: must not raise ConcretizationTypeError or any trace-time error.
    jitted = jax.jit(build_explicit_octree_metadata)
    jit_result = jitted(topology)

    assert jnp.array_equal(jit_result.oct_valid_mask, eager.oct_valid_mask)
    assert jnp.array_equal(jit_result.oct_node_depths, eager.oct_node_depths)
    assert jnp.array_equal(jit_result.oct_parent, eager.oct_parent)


def test_build_explicit_octree_metadata_from_leaf_partitions_is_jit_compatible():
    from yggdrax.octree import (
        _resolved_leaf_partitions,
        build_explicit_octree_metadata_from_leaf_partitions,
    )

    positions = jnp.array(
        [
            [0.10, 0.10, 0.10],
            [0.15, 0.12, 0.14],
            [0.20, 0.24, 0.28],
            [0.35, 0.32, 0.30],
            [0.55, 0.58, 0.53],
            [0.70, 0.72, 0.74],
            [0.82, 0.84, 0.78],
            [0.95, 0.94, 0.96],
        ],
        dtype=jnp.float32,
    )
    masses = jnp.ones((8,), dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type="octree",
        return_reordered=True,
        leaf_size=2,
    )
    topology = tree.topology
    leaf_starts, leaf_ends_exclusive, leaf_codes, leaf_depths = (
        _resolved_leaf_partitions(topology)
    )

    eager = build_explicit_octree_metadata_from_leaf_partitions(
        num_particles=int(topology.num_particles),
        leaf_starts=leaf_starts,
        leaf_ends_exclusive=leaf_ends_exclusive,
        leaf_codes=leaf_codes,
        leaf_depths=leaf_depths,
    )
    jitted = jax.jit(
        build_explicit_octree_metadata_from_leaf_partitions,
        static_argnames=("num_particles",),
    )
    jit_result = jitted(
        num_particles=int(topology.num_particles),
        leaf_starts=leaf_starts,
        leaf_ends_exclusive=leaf_ends_exclusive,
        leaf_codes=leaf_codes,
        leaf_depths=leaf_depths,
    )

    assert jnp.array_equal(jit_result.oct_valid_mask, eager.oct_valid_mask)
    assert jnp.array_equal(jit_result.oct_node_depths, eager.oct_node_depths)
    assert jnp.array_equal(jit_result.oct_leaf_mask, eager.oct_leaf_mask)


def test_build_explicit_octree_traversal_view_exposes_native_geometry_and_mappings():
    from yggdrax.octree import build_explicit_octree_traversal_view

    positions = jnp.array(
        [
            [0.10, 0.10, 0.10],
            [0.15, 0.12, 0.14],
            [0.20, 0.24, 0.28],
            [0.35, 0.32, 0.30],
            [0.55, 0.58, 0.53],
            [0.70, 0.72, 0.74],
            [0.82, 0.84, 0.78],
            [0.95, 0.94, 0.96],
        ],
        dtype=jnp.float32,
    )
    masses = jnp.ones((8,), dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type="octree",
        return_reordered=True,
        leaf_size=2,
    )

    view = build_explicit_octree_traversal_view(tree.topology)

    assert view.valid_mask.shape == view.parent.shape
    assert view.children.shape[1] == 8
    assert view.box_centers.shape == (view.valid_mask.shape[0], 3)
    assert view.box_half_extents.shape == (view.valid_mask.shape[0], 3)
    assert view.box_radii.shape == view.valid_mask.shape
    assert view.box_max_extents.shape == view.valid_mask.shape
    assert view.radix_node_to_oct.shape[0] == tree.num_nodes
    assert view.radix_leaf_to_oct.shape[0] == tree.num_leaves


def test_octree_refine_pairs_expand_same_node_children():
    pairs = _interactions_impl._octree_refine_pairs_single(
        jnp.asarray(9, dtype=jnp.int32),
        jnp.asarray(9, dtype=jnp.int32),
        jnp.bool_(True),
        jnp.bool_(True),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.asarray([11, 13, 17, -1, -1, -1, -1, -1], dtype=jnp.int32),
        jnp.asarray(3, dtype=jnp.int32),
        jnp.asarray([11, 13, 17, -1, -1, -1, -1, -1], dtype=jnp.int32),
        jnp.asarray(3, dtype=jnp.int32),
    )

    valid = pairs[pairs[:, 0] >= 0]
    expected = jnp.asarray(
        [
            [11, 11],
            [11, 13],
            [11, 17],
            [13, 13],
            [13, 17],
            [17, 17],
        ],
        dtype=jnp.int32,
    )
    assert jnp.array_equal(valid, expected)


def test_octree_refine_pairs_expand_cross_children():
    pairs = _interactions_impl._octree_refine_pairs_single(
        jnp.asarray(20, dtype=jnp.int32),
        jnp.asarray(30, dtype=jnp.int32),
        jnp.bool_(False),
        jnp.bool_(True),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.asarray([22, 24, -1, -1, -1, -1, -1, -1], dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray([31, 33, 35, -1, -1, -1, -1, -1], dtype=jnp.int32),
        jnp.asarray(3, dtype=jnp.int32),
    )

    valid = pairs[pairs[:, 0] >= 0]
    expected = jnp.asarray(
        [
            [22, 31],
            [22, 33],
            [22, 35],
            [24, 31],
            [24, 33],
            [24, 35],
        ],
        dtype=jnp.int32,
    )
    assert jnp.array_equal(valid, expected)


def test_octree_refine_pairs_support_jit_for_one_sided_split():
    jitted = jax.jit(_interactions_impl._octree_refine_pairs_single)
    pairs = jitted(
        jnp.asarray(40, dtype=jnp.int32),
        jnp.asarray(55, dtype=jnp.int32),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.bool_(True),
        jnp.bool_(False),
        jnp.asarray([41, 42, 43, -1, -1, -1, -1, -1], dtype=jnp.int32),
        jnp.asarray(3, dtype=jnp.int32),
        jnp.asarray([56, 57, -1, -1, -1, -1, -1, -1], dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
    )

    valid = pairs[pairs[:, 0] >= 0]
    expected = jnp.asarray(
        [
            [41, 55],
            [42, 55],
            [43, 55],
        ],
        dtype=jnp.int32,
    )
    assert jnp.array_equal(valid, expected)


def test_octree_refine_pairs_expand_full_8x8_cross_children():
    """8x8 cross-child split must not drop any of the 64 child pairs."""
    tgt_children = jnp.arange(10, 18, dtype=jnp.int32)
    src_children = jnp.arange(20, 28, dtype=jnp.int32)

    pairs = _interactions_impl._octree_refine_pairs_single(
        jnp.asarray(5, dtype=jnp.int32),
        jnp.asarray(6, dtype=jnp.int32),
        jnp.bool_(False),
        jnp.bool_(True),
        jnp.bool_(False),
        jnp.bool_(False),
        tgt_children,
        jnp.asarray(8, dtype=jnp.int32),
        src_children,
        jnp.asarray(8, dtype=jnp.int32),
    )

    valid = pairs[pairs[:, 0] >= 0]
    assert valid.shape[0] == 64, f"expected 64 pairs, got {valid.shape[0]}"

    expected = jnp.asarray(
        [[t, s] for t in range(10, 18) for s in range(20, 28)],
        dtype=jnp.int32,
    )
    assert jnp.array_equal(jnp.sort(valid, axis=0), jnp.sort(expected, axis=0))
