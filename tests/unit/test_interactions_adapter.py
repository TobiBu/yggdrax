"""Coverage for yggdrax interaction config adapters."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from yggdrax import (
    CompactTaggedFarPairs,
    DualTreeTraversalConfig,
    NodeInteractionList,
    NodeNeighborList,
    Tree,
    _interactions_impl,
    build_compact_far_pairs_and_leaf_neighbor_lists,
    build_interactions_and_neighbors,
    build_leaf_neighbor_lists,
    build_tree,
    compute_tree_geometry,
    infer_bounds,
)
from yggdrax.interactions import DualTreeTraversalConfig as ExpanseTraversalConfig
from yggdrax.interactions import (
    build_compact_far_pairs,
    build_interactions_and_neighbors_split,
    build_well_separated_interactions,
)

_TEST_N = 16
_TEST_TRAVERSAL_CFG = DualTreeTraversalConfig(
    max_pair_queue=512,
    process_block=32,
    max_interactions_per_node=128,
    max_neighbors_per_leaf=128,
)
_TEST_EXPANSE_TRAVERSAL_CFG = ExpanseTraversalConfig(
    max_pair_queue=512,
    process_block=32,
    max_interactions_per_node=128,
    max_neighbors_per_leaf=128,
)


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(19)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    return positions, masses


@dataclass(frozen=True)
class _TopologyCarrier:
    topology: object


class _MinimalTopology(NamedTuple):
    parent: object
    left_child: object
    right_child: object
    node_ranges: object
    num_particles: object
    bounds_min: object
    bounds_max: object
    leaf_codes: object
    leaf_depths: object
    use_morton_geometry: object


def test_interactions_accepts_yggdrax_traversal_config():
    positions, masses = _sample_problem(n=_TEST_N)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        infer_bounds(positions),
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )
    assert isinstance(interactions, NodeInteractionList)
    assert isinstance(neighbors, NodeNeighborList)
    assert interactions.sources.ndim == 1
    assert neighbors.neighbors.ndim == 1


def test_interactions_accept_topology_carrier():
    positions, masses = _sample_problem(n=_TEST_N)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        infer_bounds(positions),
        leaf_size=16,
        return_reordered=True,
    )
    carrier = _TopologyCarrier(topology=tree.topology)
    geometry = compute_tree_geometry(carrier, pos_sorted)
    interactions, neighbors = build_interactions_and_neighbors(
        carrier,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )
    assert isinstance(interactions, NodeInteractionList)
    assert isinstance(neighbors, NodeNeighborList)


def test_interactions_accepts_expanse_traversal_config():
    positions, masses = _sample_problem(n=_TEST_N)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        infer_bounds(positions),
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_EXPANSE_TRAVERSAL_CFG,
    )
    assert isinstance(interactions, NodeInteractionList)
    assert isinstance(neighbors, NodeNeighborList)
    assert interactions.sources.ndim == 1
    assert neighbors.neighbors.ndim == 1


def test_shared_count_far_pairs_and_neighbors_match_split_builders():
    positions, masses = _sample_problem(n=48)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        infer_bounds(positions),
        leaf_size=8,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)

    shared_far, shared_neighbors = build_compact_far_pairs_and_leaf_neighbor_lists(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )
    split_far = build_compact_far_pairs(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )
    split_neighbors = build_leaf_neighbor_lists(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )

    assert isinstance(shared_far, CompactTaggedFarPairs)
    assert isinstance(shared_neighbors, NodeNeighborList)
    assert jnp.array_equal(shared_far.sources, split_far.sources)
    assert jnp.array_equal(shared_far.targets, split_far.targets)
    assert jnp.array_equal(shared_far.tags, split_far.tags)
    assert jnp.array_equal(shared_neighbors.offsets, split_neighbors.offsets)
    assert jnp.array_equal(shared_neighbors.neighbors, split_neighbors.neighbors)
    assert jnp.array_equal(shared_neighbors.leaf_indices, split_neighbors.leaf_indices)
    assert jnp.array_equal(shared_neighbors.counts, split_neighbors.counts)


def test_interactions_can_derive_missing_level_fields():
    positions, masses = _sample_problem(n=_TEST_N)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        infer_bounds(positions),
        leaf_size=16,
        return_reordered=True,
    )
    topo = tree.topology
    minimal = _MinimalTopology(
        parent=topo.parent,
        left_child=topo.left_child,
        right_child=topo.right_child,
        node_ranges=topo.node_ranges,
        num_particles=topo.num_particles,
        bounds_min=topo.bounds_min,
        bounds_max=topo.bounds_max,
        leaf_codes=topo.leaf_codes,
        leaf_depths=topo.leaf_depths,
        use_morton_geometry=topo.use_morton_geometry,
    )
    geometry = compute_tree_geometry(minimal, pos_sorted)
    interactions, neighbors = build_interactions_and_neighbors(
        minimal,
        geometry,
        theta=0.6,
        mac_type="dehnen",
    )
    assert isinstance(interactions, NodeInteractionList)
    assert isinstance(neighbors, NodeNeighborList)


def test_dual_tree_queue_cache_is_bounded():
    _interactions_impl._DUAL_TREE_QUEUE_CACHE.clear()
    for idx in range(_interactions_impl._MAX_DUAL_TREE_QUEUE_CACHE_ENTRIES + 5):
        _interactions_impl._store_cached_queue_capacity((idx,), idx + 1)

    assert (
        len(_interactions_impl._DUAL_TREE_QUEUE_CACHE)
        == _interactions_impl._MAX_DUAL_TREE_QUEUE_CACHE_ENTRIES
    )
    assert (0,) not in _interactions_impl._DUAL_TREE_QUEUE_CACHE
    assert (
        _interactions_impl._get_cached_queue_capacity(
            (_interactions_impl._MAX_DUAL_TREE_QUEUE_CACHE_ENTRIES + 4,)
        )
        == _interactions_impl._MAX_DUAL_TREE_QUEUE_CACHE_ENTRIES + 5
    )


def _near_pairs(neighbors: NodeNeighborList) -> set[tuple[int, int]]:
    """Return the near list as a set of ``(leaf, neighbour)`` node pairs."""

    leaf_rows = jnp.repeat(neighbors.leaf_indices, neighbors.counts)
    total = int(neighbors.offsets[-1])
    return set(
        zip(
            leaf_rows[:total].tolist(),
            neighbors.neighbors[:total].tolist(),
        )
    )


def _walk_problem(tree_type: str):
    # 64 leaves: enough for the walk to accept far pairs as well as near ones.
    positions, masses = _sample_problem(n=512)
    tree = Tree.from_particles(
        positions,
        masses,
        bounds=infer_bounds(positions),
        leaf_size=8,
        return_reordered=True,
        tree_type=tree_type,
    )
    return tree, compute_tree_geometry(tree, tree.positions_sorted)


_WALK_CFG = DualTreeTraversalConfig(
    max_pair_queue=4096,
    process_block=32,
    max_interactions_per_node=256,
    max_neighbors_per_leaf=256,
)


def test_dual_tree_walk_serves_every_tree_type():
    """The joint far + near walk serves radix, octree and kd-tree topologies."""

    for tree_type in ("radix", "octree", "kdtree"):
        tree, geometry = _walk_problem(tree_type)
        interactions, neighbors = build_interactions_and_neighbors(
            tree,
            geometry,
            theta=0.6,
            mac_type="dehnen",
            traversal_config=_WALK_CFG,
        )

        assert isinstance(interactions, NodeInteractionList)
        assert isinstance(neighbors, NodeNeighborList)
        assert int(interactions.sources.shape[0]) > 0, tree_type
        # At this density every leaf has near neighbours.
        assert int(jnp.min(neighbors.counts)) >= 1, tree_type


def _far_pairs(interactions) -> set[tuple[int, int]]:
    """Return a far list (either container) as a set of ``(target, source)`` pairs."""

    if isinstance(interactions, CompactTaggedFarPairs):
        total = int(interactions.far_pair_count)
    else:
        total = int(jnp.sum(interactions.counts))
    return set(
        zip(
            interactions.targets[:total].tolist(),
            interactions.sources[:total].tolist(),
        )
    )


@pytest.mark.parametrize("tree_type", ["octree", "kdtree"])
def test_walk_builders_agree_with_the_joint_walk(tree_type):
    """Every builder serves the tree with the joint far + near walk's pairs.

    Two ways they used to differ, both under explicit capacities:

    * octree: the near-only builders routed to a native octree walk that left most
      near rows empty; removed in the 2026-10 cleanup (Y1), every builder now runs
      the binary walk;
    * kd-tree with the Dehnen MAC: the kd default ``dehnen_radius_scale`` (1.2) was
      applied by the joint walk only, so the bounded count-pass routes walked at
      1.0 -- 1,710 near pairs against 2,408 at this size (fixed 2026-10).
    """

    tree, geometry = _walk_problem(tree_type)
    joint_far, joint = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_WALK_CFG,
    )
    far_only = build_well_separated_interactions(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_WALK_CFG,
    )
    near_only = build_leaf_neighbor_lists(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_WALK_CFG,
    )
    _, split = build_interactions_and_neighbors_split(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_WALK_CFG,
    )
    compact_far, compact = build_compact_far_pairs_and_leaf_neighbor_lists(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_WALK_CFG,
    )

    assert isinstance(near_only, NodeNeighborList)
    assert int(jnp.min(joint.counts)) >= 1
    joint_pairs = _near_pairs(joint)
    assert _near_pairs(near_only) == joint_pairs
    assert _near_pairs(split) == joint_pairs
    assert _near_pairs(compact) == joint_pairs
    joint_far_pairs = _far_pairs(joint_far)
    assert joint_far_pairs, "the walk accepted no far pair; the far check is vacuous"
    assert _far_pairs(far_only) == joint_far_pairs
    assert _far_pairs(compact_far) == joint_far_pairs
