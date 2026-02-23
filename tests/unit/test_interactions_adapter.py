"""Coverage for yggdrax interaction config adapters."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from yggdrax import (
    DualTreeTraversalConfig,
    NodeInteractionList,
    NodeNeighborList,
    build_interactions_and_neighbors,
    build_tree,
    compute_tree_geometry,
    infer_bounds,
)
from yggdrax.interactions import DualTreeTraversalConfig as ExpanseTraversalConfig


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
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        infer_bounds(positions),
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    traversal_cfg = DualTreeTraversalConfig(
        max_pair_queue=8192,
        process_block=256,
        max_interactions_per_node=2048,
        max_neighbors_per_leaf=2048,
    )
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=traversal_cfg,
    )
    assert isinstance(interactions, NodeInteractionList)
    assert isinstance(neighbors, NodeNeighborList)
    assert interactions.sources.ndim == 1
    assert neighbors.neighbors.ndim == 1


def test_interactions_accept_topology_carrier():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        infer_bounds(positions),
        leaf_size=16,
        return_reordered=True,
    )
    carrier = _TopologyCarrier(topology=tree.topology)
    geometry = compute_tree_geometry(carrier, pos_sorted)
    traversal_cfg = DualTreeTraversalConfig(
        max_pair_queue=8192,
        process_block=256,
        max_interactions_per_node=2048,
        max_neighbors_per_leaf=2048,
    )
    interactions, neighbors = build_interactions_and_neighbors(
        carrier,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=traversal_cfg,
    )
    assert isinstance(interactions, NodeInteractionList)
    assert isinstance(neighbors, NodeNeighborList)


def test_interactions_accepts_expanse_traversal_config():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        infer_bounds(positions),
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    traversal_cfg = ExpanseTraversalConfig(
        max_pair_queue=8192,
        process_block=256,
        max_interactions_per_node=2048,
        max_neighbors_per_leaf=2048,
    )
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=traversal_cfg,
    )
    assert isinstance(interactions, NodeInteractionList)
    assert isinstance(neighbors, NodeNeighborList)
    assert interactions.sources.ndim == 1
    assert neighbors.neighbors.ndim == 1


def test_interactions_can_derive_missing_level_fields():
    positions, masses = _sample_problem(n=64)
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
