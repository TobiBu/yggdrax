"""Tests for explicit octree metadata on top of radix-compatible trees."""

import jax
import jax.numpy as jnp

from yggdrax import (
    DualTreeTraversalConfig,
    Tree,
    build_interactions_and_neighbors,
    compute_tree_geometry,
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
            [0.55, 0.58, 0.53],
            [0.70, 0.72, 0.74],
            [0.82, 0.84, 0.78],
            [0.88, 0.90, 0.92],
            [0.95, 0.94, 0.96],
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

    cfg = DualTreeTraversalConfig(
        max_pair_queue=1024,
        process_block=64,
        max_interactions_per_node=256,
        max_neighbors_per_leaf=256,
    )
    radix_geom = compute_tree_geometry(radix, radix.positions_sorted)
    oct_geom = compute_tree_geometry(octree, octree.positions_sorted)
    radix_interactions, radix_neighbors = build_interactions_and_neighbors(
        radix,
        radix_geom,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=cfg,
    )
    oct_interactions, oct_neighbors = build_interactions_and_neighbors(
        octree,
        oct_geom,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=cfg,
    )

    assert jnp.array_equal(radix.positions_sorted, octree.positions_sorted)
    assert jnp.array_equal(radix.masses_sorted, octree.masses_sorted)
    assert jnp.array_equal(radix_interactions.counts, oct_interactions.counts)
    assert jnp.array_equal(radix_neighbors.counts, oct_neighbors.counts)


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
    leaf_starts, leaf_ends_exclusive, leaf_codes, leaf_depths = _resolved_leaf_partitions(
        topology
    )

    eager = build_explicit_octree_metadata_from_leaf_partitions(
        num_particles=int(topology.num_particles),
        leaf_starts=leaf_starts,
        leaf_ends_exclusive=leaf_ends_exclusive,
        leaf_codes=leaf_codes,
        leaf_depths=leaf_depths,
    )
    jitted = jax.jit(build_explicit_octree_metadata_from_leaf_partitions, static_argnames=("num_particles",))
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
