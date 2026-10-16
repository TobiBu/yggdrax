"""The radix tree's per-level tables, against their definitions.

``_build_tree_from_leaf_partitions`` derives each node's depth by pointer doubling
(eight unrolled rounds: depths up to 256, past ``MAX_TREE_LEVELS``) and the level
tables from ONE stable sort of the depths: ``nodes_by_level`` is the sort's
permutation and ``level_offsets`` the starts of its runs. They used to come from a
22-25-round loop and a scatter-add histogram. Pinned here to the definitions --
distance to the root along the parent chain, a stable argsort, a cumulative count
-- on radix and static-cells trees, including a deep one (a dense core).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax._tree_impl import MAX_TREE_LEVELS, build_static_cells_tree, build_tree


def _depths(parent: np.ndarray) -> np.ndarray:
    depth = np.zeros(parent.shape[0], np.int64)
    for i in range(parent.shape[0]):
        j, d = i, 0
        while parent[j] >= 0:
            j, d = parent[j], d + 1
        depth[i] = d
    return depth


def _positions(n: int, seed: int, core: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3)) * np.array([1.0, 0.3, 0.1])
    x[: n // 4] *= core  # a dense core: a deep tree
    return x


@pytest.mark.parametrize(
    "n, leaf, kind, core",
    [
        (3, 1, "radix", 1.0),
        (5000, 16, "radix", 1e-3),
        (20000, 32, "cells", 1e-3),
        (20000, 64, "cells", 1e-5),
    ],
)
def test_level_tables_match_their_definitions(n, leaf, kind, core):
    x = _positions(n, 7, core)
    m = np.ones(n)
    bounds = (jnp.asarray(x.min(0) - 1e-3), jnp.asarray(x.max(0) + 1e-3))
    if kind == "radix":
        tree = build_tree(jnp.asarray(x), jnp.asarray(m), bounds, leaf_size=leaf)
    else:
        tree = build_static_cells_tree(
            jnp.asarray(x),
            jnp.asarray(m),
            bounds,
            leaf_size=leaf,
            leaf_capacity=4 * n // leaf,
            min_level=2,
        )
    tree = tree if hasattr(tree, "node_level") else tree[0]
    level = np.asarray(tree.node_level)
    want = _depths(np.asarray(tree.parent))
    np.testing.assert_array_equal(level, want)
    assert int(tree.num_levels) == int(want.max()) + 1
    order = np.asarray(tree.nodes_by_level)
    np.testing.assert_array_equal(order, np.argsort(want, kind="stable"))
    assert order.dtype == np.asarray(jnp.argsort(jnp.asarray(level))).dtype
    counts = np.bincount(want, minlength=MAX_TREE_LEVELS)[:MAX_TREE_LEVELS]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    np.testing.assert_array_equal(np.asarray(tree.level_offsets), offsets)
