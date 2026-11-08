"""``mutual_walk_inputs``: the mutual walk on its output is the dual walk, per backend.

``dual_tree_walk_mutual`` applies no per-backend defaults, so a caller that
builds its node arrays by hand walks a kd-tree at Dehnen radius scale 1.0 while
the dual walk silently uses 1.2 there. The helper resolves the scale the way the
dual walk does. Pinned here: the far and near lists agree as sets on the radix,
octree and kd backends; the kd radii differ from the unscaled hand-built ones
(and the radix radii do not); the helper traces under ``jit`` with the
capacity-padded static cells tree and masks its empty nodes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import DualTreeTraversalConfig, Tree, compute_tree_geometry
from yggdrax._interactions_impl import _build_mac_extents
from yggdrax._tree_impl import build_static_cells_tree
from yggdrax.bounds import infer_bounds
from yggdrax.interactions import (
    build_interactions_and_neighbors,
    dual_tree_walk_mutual,
    mutual_walk_inputs,
)

_CAPS = {"max_pair_queue": 1 << 16, "far_cap": 1 << 17, "near_cap": 1 << 17}


def _points(n: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n, 3))
    mass = rng.uniform(0.8, 1.2, size=n)
    return jnp.asarray(pos), jnp.asarray(mass)


def _tree(tree_type: str, n: int = 3000, leaf: int = 16):
    pos, mass = _points(n)
    tree = Tree.from_particles(
        pos, mass, tree_type=tree_type, leaf_size=leaf, return_reordered=True
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    return tree, geometry


def _dual_sets(tree, geometry, theta, mac_type):
    config = DualTreeTraversalConfig(
        max_pair_queue=1 << 16,
        process_block=64,
        max_interactions_per_node=4096,
        max_neighbors_per_leaf=4096,
    )
    _, neighbors, result = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=theta,
        traversal_config=config,
        mac_type=mac_type,
        return_result=True,
    )
    assert not (
        bool(result.queue_overflow)
        or bool(result.far_overflow)
        or bool(result.near_overflow)
    )
    src = np.asarray(result.interaction_sources)
    tgt = np.asarray(result.interaction_targets)
    live = (src >= 0) & (tgt >= 0)
    far = {
        (min(a, b), max(a, b)) for a, b in zip(tgt[live].tolist(), src[live].tolist())
    }
    offsets = np.asarray(neighbors.offsets)
    counts = np.asarray(neighbors.counts)
    nbrs = np.asarray(neighbors.neighbors)
    near = set()
    for row, leaf_node in enumerate(np.asarray(neighbors.leaf_indices).tolist()):
        for k in range(int(counts[row])):
            other = int(nbrs[int(offsets[row]) + k])
            near.add((min(leaf_node, other), max(leaf_node, other)))
    return far, near


def _walk(w, theta, mac_type):
    res = dual_tree_walk_mutual(
        w.left_child_full,
        w.right_child_full,
        w.centers,
        w.radii,
        theta,
        w.root,
        mac_type=mac_type,
        node_active=w.node_active,
        **_CAPS,
    )
    assert not (
        bool(res.queue_overflow) or bool(res.far_overflow) or bool(res.near_overflow)
    )
    return res


def _mutual_sets(res):
    nf, nn = int(res.far_count), int(res.near_count)
    fa, fb = np.asarray(res.far_a)[:nf], np.asarray(res.far_b)[:nf]
    na, nb = np.asarray(res.near_a)[:nn], np.asarray(res.near_b)[:nn]
    return set(zip(fa.tolist(), fb.tolist())), set(zip(na.tolist(), nb.tolist()))


@pytest.mark.parametrize("tree_type", ["radix", "octree", "kdtree"])
@pytest.mark.parametrize("mac_type", ["bh", "dehnen"])
def test_mutual_walk_on_helper_inputs_is_the_dual_walk(tree_type, mac_type):
    theta = 0.6
    tree, geometry = _tree(tree_type)
    far_ref, near_ref = _dual_sets(tree, geometry, theta, mac_type)
    far, near = _mutual_sets(
        _walk(mutual_walk_inputs(tree, geometry, mac_type=mac_type), theta, mac_type)
    )
    assert far == far_ref
    assert near == near_ref
    assert len(far_ref) > 0 and len(near_ref) > 0


@pytest.mark.parametrize("tree_type,scaled", [("radix", False), ("kdtree", True)])
def test_dehnen_radii_carry_the_backend_scale(tree_type, scaled):
    tree, geometry = _tree(tree_type)
    topo = tree.topology
    num_internal = int(topo.left_child.shape[0])
    unscaled = _build_mac_extents(topo.parent, geometry, num_internal, "dehnen", 1.0)[0]
    radii = mutual_walk_inputs(tree, geometry, mac_type="dehnen").radii
    expected = 1.2 * np.asarray(unscaled) if scaled else np.asarray(unscaled)
    np.testing.assert_allclose(np.asarray(radii), expected, rtol=1e-12)
    # an explicit scale is the caller's, on every backend
    explicit = mutual_walk_inputs(
        tree, geometry, mac_type="dehnen", dehnen_radius_scale=1.5
    ).radii
    np.testing.assert_allclose(
        np.asarray(explicit), 1.5 * np.asarray(unscaled), rtol=1e-12
    )


def test_helper_traces_with_the_static_cells_tree_and_masks_empty_nodes():
    pos, mass = _points(4000, seed=5)
    bounds = infer_bounds(pos)
    leaf = 32

    @jax.jit
    def step(p, m):
        topo, ps, _ms, _inv = build_static_cells_tree(
            p, m, bounds, leaf_size=leaf, leaf_capacity=512, return_reordered=True
        )
        geometry = compute_tree_geometry(topo, ps)
        w = mutual_walk_inputs(topo, geometry, mask_empty_nodes=True)
        return w, dual_tree_walk_mutual(
            w.left_child_full,
            w.right_child_full,
            w.centers,
            w.radii,
            0.6,
            w.root,
            mac_type="dehnen",
            node_active=w.node_active,
            **_CAPS,
        )

    w, res = step(pos, mass)
    active = np.asarray(w.node_active)
    assert active.dtype == bool and active.any() and not active.all()
    assert not (
        bool(res.queue_overflow) or bool(res.far_overflow) or bool(res.near_overflow)
    )
    far, near = _mutual_sets(res)
    touched = {i for pair in far | near for i in pair}
    assert touched and all(active[i] for i in touched)


def test_unmasked_by_default_and_rejects_non_geometric_mac():
    tree, geometry = _tree("radix", n=500)
    assert mutual_walk_inputs(tree, geometry).node_active is None
    with pytest.raises(ValueError, match="geometric mac_type"):
        mutual_walk_inputs(tree, geometry, mac_type="nope")  # type: ignore[arg-type]
