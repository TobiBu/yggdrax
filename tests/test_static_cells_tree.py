"""``build_static_cells_tree`` / ``build_static_radix_tree(leaf_partition="cells")``: static shapes, cell leaves, padding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy
from yggdrax._tree_impl import (
    build_static_cells_tree,
    build_static_radix_tree,
    rebuild_static_radix_tree_from_template,
)
from yggdrax.bounds import infer_bounds
from yggdrax.morton import morton_encode


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _check_tree(tree, n, cap, leaf_size):
    node_ranges = np.asarray(tree.node_ranges)
    num_internal = int(tree.left_child.shape[0])
    total = node_ranges.shape[0]
    assert total == 2 * cap - 1 and num_internal == cap - 1
    leaves = node_ranges[num_internal:]
    occ = np.where(leaves[:, 1] >= leaves[:, 0], leaves[:, 1] - leaves[:, 0] + 1, 0)
    live = occ > 0
    k = int(live.sum())
    assert np.all(live[:k]) and not np.any(live[k:]), "live leaves are a prefix"
    assert occ.max() <= leaf_size
    assert occ.sum() == n, "every particle is in exactly one leaf"
    assert leaves[0, 0] == 0 and np.all(leaves[1:k, 0] == leaves[: k - 1, 1] + 1)
    # internal ranges cover exactly their children's ranges
    lc, rc = np.asarray(tree.left_child), np.asarray(tree.right_child)
    for i in range(num_internal):
        a, b = node_ranges[lc[i]], node_ranges[rc[i]]
        lo = (
            min(x[0] for x in (a, b) if x[1] >= x[0])
            if (a[1] >= a[0] or b[1] >= b[0])
            else n
        )
        hi = (
            max(x[1] for x in (a, b) if x[1] >= x[0])
            if (a[1] >= a[0] or b[1] >= b[0])
            else n - 1
        )
        assert node_ranges[i, 0] == lo and node_ranges[i, 1] == hi
    # padding leaves are dead: empty ranges and depth -1
    assert np.all(np.asarray(tree.leaf_depths)[k:] == -1)
    assert np.all(np.asarray(tree.leaf_depths)[:k] >= 0)
    return k


@pytest.mark.parametrize("leaf_size", [16, 64])
def test_cells_tree_has_cell_leaves_padded_to_capacity(leaf_size):
    n = 5000
    P = jnp.asarray(_plummer(n), jnp.float32)
    M = jnp.ones((n,), jnp.float32)
    bounds = infer_bounds(P)
    codes = np.asarray(jnp.sort(morton_encode(P, bounds))).astype(np.uint64)
    s_ref, _, _ = adaptive_cell_leaf_partition_numpy(codes, leaf_size=leaf_size)
    cap = int(s_ref.size) + 50
    tree, ps, ms, inv, overflow = build_static_cells_tree(
        P,
        M,
        bounds,
        leaf_size=leaf_size,
        leaf_capacity=cap,
        return_reordered=True,
        return_overflow=True,
    )
    assert not bool(overflow)
    k = _check_tree(tree, n, cap, leaf_size)
    assert k == s_ref.size
    assert tree.leaf_size == leaf_size and int(tree.leaf_codes.shape[0]) == cap
    # sorted order really is Morton order of the original particles
    assert np.array_equal(
        np.asarray(ps), np.asarray(P)[np.asarray(tree.particle_indices)]
    )
    assert np.array_equal(
        np.asarray(inv)[np.asarray(tree.particle_indices)], np.arange(n)
    )


def test_overflow_flag_when_capacity_is_too_small():
    n = 3000
    P = jnp.asarray(_plummer(n, 1), jnp.float32)
    M = jnp.ones((n,), jnp.float32)
    tree, overflow = build_static_cells_tree(
        P, M, infer_bounds(P), leaf_size=8, leaf_capacity=64, return_overflow=True
    )
    assert bool(overflow)
    assert int(tree.leaf_codes.shape[0]) == 64


def test_partition_knob_and_template_rebuild_trace():
    n = 4000
    P = jnp.asarray(_plummer(n, 2), jnp.float32)
    M = jnp.ones((n,), jnp.float32)
    bounds = infer_bounds(P)
    with pytest.raises(ValueError):
        build_static_radix_tree(P, M, bounds, leaf_size=32, leaf_partition="cells")
    with pytest.raises(ValueError):
        build_static_radix_tree(
            P, M, bounds, leaf_size=32, leaf_partition="hex", leaf_capacity=8
        )
    tree = build_static_radix_tree(
        P, M, bounds, leaf_size=32, leaf_partition="cells", leaf_capacity=1024
    )
    cap = int(tree.leaf_codes.shape[0])
    assert cap == 1024
    # buckets are untouched
    bt = build_static_radix_tree(P, M, bounds, leaf_size=32)
    assert int(bt.leaf_codes.shape[0]) == -(-n // 32)
    # the template rebuild with moved particles is jittable and keeps the shape
    P2 = P * 1.01 + 0.02

    @jax.jit
    def refresh(pos):
        t, ps, ms, inv, over = rebuild_static_radix_tree_from_template(
            pos,
            M,
            tree,
            return_reordered=True,
            leaf_partition="cells",
            return_overflow=True,
        )
        return t.node_ranges, t.parent, over

    ranges2, parent2, over2 = refresh(P2)
    assert not bool(over2)
    assert (
        ranges2.shape == tree.node_ranges.shape and parent2.shape == tree.parent.shape
    )
    with pytest.raises(ValueError):
        rebuild_static_radix_tree_from_template(P2, M, bt, return_overflow=True)
