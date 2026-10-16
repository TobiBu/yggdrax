"""Coverage for yggdrax geometry API wrappers."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from yggdrax import (
    build_tree,
    compute_tree_geometry,
    get_level_offsets,
    get_node_levels,
    get_nodes_by_level,
    get_num_levels,
)


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


def test_compute_tree_geometry_supports_outer_jit():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )

    @jax.jit
    def call(tree_arg, pos_arg):
        geom = compute_tree_geometry(tree_arg, pos_arg)
        return jnp.sum(geom.radius)

    value = call(tree, pos_sorted)
    assert jnp.isfinite(value)


def test_compute_tree_geometry_supports_outer_jit_wrapper():
    jitted = jax.jit(lambda t, ps: compute_tree_geometry(t, ps))
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    geometry = jitted(tree, pos_sorted)
    total_nodes = int(tree.parent.shape[0])
    assert geometry.center.shape == (total_nodes, 3)


def test_compute_tree_geometry_supports_explicit_leaf_cap_under_outer_jit():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    jitted = jax.jit(
        lambda t, ps: compute_tree_geometry(t, ps, max_leaf_size=16).max_extent.sum()
    )
    value = jitted(tree, pos_sorted)
    assert jnp.isfinite(value)


def test_compute_tree_geometry_uses_tree_leaf_size_under_outer_jit():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    jitted = jax.jit(lambda t, ps: compute_tree_geometry(t, ps).max_extent.sum())
    value = jitted(tree, pos_sorted)
    assert jnp.isfinite(value)
    assert int(tree.leaf_size) == 16


def test_geometry_and_level_views_derive_missing_level_fields():
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
    # The minimal topology carries no level fields; geometry and the level views
    # must derive them from `parent` and agree with the build's own.
    geometry = compute_tree_geometry(minimal, pos_sorted)
    reference = compute_tree_geometry(tree, pos_sorted)
    for derived, built in zip(geometry, reference, strict=True):
        assert jnp.array_equal(derived, built)

    num_levels = int(topo.num_levels)
    node_levels = get_node_levels(minimal)
    assert jnp.array_equal(node_levels, topo.node_level)
    assert get_num_levels(minimal, node_levels=node_levels) == num_levels
    assert jnp.array_equal(
        get_level_offsets(minimal, node_levels=node_levels),
        topo.level_offsets[: num_levels + 1],
    )
    assert jnp.array_equal(
        get_nodes_by_level(minimal, node_levels=node_levels), topo.nodes_by_level
    )
