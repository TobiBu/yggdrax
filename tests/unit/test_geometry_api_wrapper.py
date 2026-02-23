"""Coverage for yggdrax geometry API wrappers."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from yggdrax import build_tree, compute_tree_geometry, geometry_to_level_major


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(29)
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


def test_compute_tree_geometry_wrapper_shapes():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    total_nodes = int(tree.parent.shape[0])
    assert geometry.center.shape == (total_nodes, 3)
    assert geometry.half_extent.shape == (total_nodes, 3)
    assert geometry.radius.shape == (total_nodes,)
    assert geometry.max_extent.shape == (total_nodes,)


def test_compute_tree_geometry_accepts_topology_carrier():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    carrier = _TopologyCarrier(topology=tree.topology)
    geometry = compute_tree_geometry(carrier, pos_sorted)
    assert geometry.center.shape[0] == int(tree.parent.shape[0])


def test_geometry_to_level_major_wrapper_shapes():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    level_major = geometry_to_level_major(tree, geometry)
    assert level_major.centers.ndim == 3
    assert level_major.half_extents.ndim == 3
    assert level_major.radii.ndim == 2
    assert level_major.max_extents.ndim == 2
    assert level_major.level_counts.ndim == 1
    assert level_major.node_indices.ndim == 2


def test_geometry_to_level_major_accepts_topology_carrier():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    carrier = _TopologyCarrier(topology=tree.topology)
    geometry = compute_tree_geometry(carrier, pos_sorted)
    level_major = geometry_to_level_major(carrier, geometry)
    assert level_major.centers.ndim == 3


def test_compute_tree_geometry_supports_outer_jit():
    jitted = jax.jit(lambda t, ps: compute_tree_geometry(t, ps))
    geometry = jitted(tree, pos_sorted)
    total_nodes = int(tree.parent.shape[0])
    assert geometry.center.shape == (total_nodes, 3)


def test_geometry_level_views_can_derive_missing_level_fields():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
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
    level_major = geometry_to_level_major(minimal, geometry)
    assert level_major.node_indices.ndim == 2
