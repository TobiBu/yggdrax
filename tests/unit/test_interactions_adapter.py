"""Coverage for yggdrax interaction config adapters."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from yggdrax import (
    DualTreeTraversalConfig,
    NodeInteractionList,
    NodeNeighborList,
    Tree,
    _interactions_impl,
    build_interactions_and_neighbors,
    build_leaf_neighbor_lists,
    build_tree,
    compute_tree_geometry,
    infer_bounds,
)
from yggdrax.interactions import DualTreeTraversalConfig as ExpanseTraversalConfig

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


def test_dual_tree_dispatch_is_octree_specific(monkeypatch):
    positions, masses = _sample_problem(n=_TEST_N)

    dispatch_calls: list[str] = []
    original_octree = _interactions_impl._run_octree_walk_raw
    original_legacy = _interactions_impl._run_legacy_dual_tree_walk_raw

    def _record_octree(*args, **kwargs):
        dispatch_calls.append("octree")
        return original_octree(*args, **kwargs)

    def _record_legacy(*args, **kwargs):
        dispatch_calls.append("legacy")
        return original_legacy(*args, **kwargs)

    monkeypatch.setattr(_interactions_impl, "_run_octree_walk_raw", _record_octree)
    monkeypatch.setattr(
        _interactions_impl,
        "_run_legacy_dual_tree_walk_raw",
        _record_legacy,
    )

    for tree_type, expected_calls in (
        ("radix", ["legacy"]),
        ("octree", ["octree", "legacy"]),
        ("kdtree", ["legacy"]),
    ):
        dispatch_calls.clear()
        tree = Tree.from_particles(
            positions,
            masses,
            bounds=infer_bounds(positions),
            leaf_size=16,
            return_reordered=True,
            tree_type=tree_type,
        )
        geometry = compute_tree_geometry(tree, tree.positions_sorted)
        interactions, neighbors = build_interactions_and_neighbors(
            tree,
            geometry,
            theta=0.6,
            mac_type="dehnen",
            traversal_config=_TEST_TRAVERSAL_CFG,
        )

        assert isinstance(interactions, NodeInteractionList)
        assert isinstance(neighbors, NodeNeighborList)
        assert dispatch_calls == expected_calls


def test_octree_leaf_neighbor_lists_use_native_dispatch(monkeypatch):
    positions, masses = _sample_problem(n=_TEST_N)
    dispatch_calls: list[str] = []
    original_octree = _interactions_impl._run_octree_walk_raw
    original_legacy = _interactions_impl._run_legacy_dual_tree_walk_raw

    def _record_octree(*args, **kwargs):
        dispatch_calls.append("octree")
        return original_octree(*args, **kwargs)

    def _record_legacy(*args, **kwargs):
        dispatch_calls.append("legacy")
        return original_legacy(*args, **kwargs)

    monkeypatch.setattr(_interactions_impl, "_run_octree_walk_raw", _record_octree)
    monkeypatch.setattr(
        _interactions_impl,
        "_run_legacy_dual_tree_walk_raw",
        _record_legacy,
    )

    tree = Tree.from_particles(
        positions,
        masses,
        bounds=infer_bounds(positions),
        leaf_size=16,
        return_reordered=True,
        tree_type="octree",
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    neighbors = build_leaf_neighbor_lists(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )

    assert isinstance(neighbors, NodeNeighborList)
    assert dispatch_calls == ["octree"]
