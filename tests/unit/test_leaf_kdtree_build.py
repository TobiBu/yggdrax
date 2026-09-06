"""The leaf KD-tree build's invariants, and its eager/jitted equivalence.

The build's level loop was rewritten to exploit two facts that are properties of
a ceil-median split rather than of the data: the per-level bucket occupancies
follow from ``n`` and the level alone, and the bucket labels are therefore
sorted ascending at every level. That let the counting reduction, its prefix sum
and the final regrouping sort go away, and let the extent reductions declare
``indices_are_sorted``. It also moved the build under ``jit``.

None of that may change the tree. These tests pin the definition -- a
ceil-median split of contiguous buckets, every particle in exactly one leaf,
every split plane separating its subtrees -- and pin the jitted build to the
eager one, so a future rewrite has something to be checked against.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import Tree
from yggdrax.kdtree import (
    _build_leaf_kdtree_topology,
    _leaf_kdtree_depth,
    build_leaf_kdtree,
)

_FIELDS = (
    "particle_indices",
    "node_ranges",
    "parent",
    "left_child",
    "right_child",
    "leaf_nodes",
    "nodes_by_level",
    "split_dim",
    "split_value",
)

# 1001, 4095, 4096 and 4097 straddle the power-of-two leaf-count boundaries where
# buckets stop dividing evenly and the closed-form occupancies differ by one.
_SIZES = [200, 1000, 1001, 4095, 4096, 4097, 10_000]


def _points(n: int, dim: int = 3, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (n, dim), minval=-1.0, maxval=1.0, dtype=jnp.float32)


def _expected_leaf_counts(n: int, depth: int) -> np.ndarray:
    """Bucket occupancies after ``depth`` ceil-median splits, from ``n`` alone."""
    counts = np.array([n], dtype=np.int64)
    for _ in range(depth):
        half = (counts + 1) // 2
        counts = np.stack([half, counts - half], axis=1).reshape(-1)
    return counts


@pytest.mark.parametrize("n", _SIZES)
def test_leaf_occupancies_are_the_ceil_median_recursion(n: int) -> None:
    leaf_size = 64
    tree = build_leaf_kdtree(_points(n), leaf_size=leaf_size)
    depth = _leaf_kdtree_depth(n, leaf_size)
    expected = _expected_leaf_counts(n, depth)

    ranges = np.asarray(tree.node_ranges)
    leaves = np.asarray(tree.leaf_nodes)
    starts = ranges[leaves, 0]
    ends_incl = ranges[leaves, 1]
    counts = ends_incl - starts + 1

    np.testing.assert_array_equal(counts, expected)
    # Contiguous and exactly covering [0, n).
    np.testing.assert_array_equal(
        starts, np.concatenate([[0], np.cumsum(expected)[:-1]])
    )
    assert int(ends_incl[-1]) == n - 1
    # A ceil-median split never empties a leaf.
    assert counts.min() >= 1


@pytest.mark.parametrize("n", _SIZES)
def test_every_particle_lands_in_exactly_one_leaf(n: int) -> None:
    tree = build_leaf_kdtree(_points(n), leaf_size=64)
    perm = np.asarray(tree.particle_indices)
    np.testing.assert_array_equal(np.sort(perm), np.arange(n))


@pytest.mark.parametrize("n", [1000, 4097, 10_000])
def test_split_planes_separate_their_subtrees(n: int) -> None:
    """Left subtree at or below the plane, right subtree at or above it."""
    leaf_size = 64
    points = _points(n)
    tree = build_leaf_kdtree(points, leaf_size=leaf_size)
    pts = np.asarray(points)[np.asarray(tree.particle_indices)]
    ranges = np.asarray(tree.node_ranges)
    split_dim = np.asarray(tree.split_dim)
    split_value = np.asarray(tree.split_value)
    num_internal = int(tree.num_internal_nodes)

    for node in range(num_internal):
        d = int(split_dim[node])
        if d < 0:
            continue
        left, right = 2 * node + 1, 2 * node + 2
        lo_l, hi_l = int(ranges[left, 0]), int(ranges[left, 1])
        lo_r, hi_r = int(ranges[right, 0]), int(ranges[right, 1])
        plane = float(split_value[node])
        assert pts[lo_l : hi_l + 1, d].max() <= plane
        assert pts[lo_r : hi_r + 1, d].min() >= plane


@pytest.mark.parametrize("n", _SIZES)
def test_jitted_build_is_identical_to_the_eager_one(n: int) -> None:
    """``build_leaf_kdtree`` routes through ``jit``; it must change nothing."""
    leaf_size = 64
    points = _points(n)
    eager = _build_leaf_kdtree_topology(points, leaf_size)
    jitted = build_leaf_kdtree(points, leaf_size=leaf_size)
    for field in _FIELDS:
        a = np.asarray(eager[field])
        b = np.asarray(getattr(jitted, field))
        if np.issubdtype(a.dtype, np.floating):
            np.testing.assert_array_equal(a, b)  # NaN-equal by assert_array_equal
        else:
            np.testing.assert_array_equal(a, b)
    assert int(jitted.num_internal_nodes) == int(eager["num_internal"])


def test_num_internal_nodes_stays_a_python_int() -> None:
    """The jit boundary must not turn the static field into a traced array.

    ``num_internal_nodes`` is the topology's one non-array field; letting ``jit``
    return it would make it an ``Array`` and every ``int(...)`` of it a device
    sync.
    """
    tree = build_leaf_kdtree(_points(10_000), leaf_size=64)
    assert isinstance(tree.num_internal_nodes, int)
    assert tree.num_internal_nodes == (1 << _leaf_kdtree_depth(10_000, 64)) - 1


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_build_handles_other_dimensions(dim: int) -> None:
    """The extent reduction packs ``[p, -p]``, so it must follow ``dim``."""
    n = 1000
    tree = build_leaf_kdtree(_points(n, dim=dim), leaf_size=64)
    split_dim = np.asarray(tree.split_dim)
    assert split_dim.max() < dim
    np.testing.assert_array_equal(
        np.sort(np.asarray(tree.particle_indices)), np.arange(n)
    )


def test_particle_tree_build_is_deterministic() -> None:
    """Two builds of the same points give the same tree, jit cache or not."""
    n = 10_000
    points = _points(n)
    masses = jnp.ones((n,), dtype=jnp.float32)
    first = Tree.from_particles(
        points,
        masses,
        tree_type="kdtree",
        build_mode="adaptive",
        leaf_size=64,
        return_reordered=True,
    )
    second = Tree.from_particles(
        points,
        masses,
        tree_type="kdtree",
        build_mode="adaptive",
        leaf_size=64,
        return_reordered=True,
    )
    for field in _FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(first.topology, field)),
            np.asarray(getattr(second.topology, field)),
        )
