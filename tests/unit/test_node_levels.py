"""Per-node depths, against the relaxation they replaced.

``get_node_levels`` used to fall back to a Python loop of ``num_nodes - 1``
rounds whenever a topology carried no ``node_level`` field -- ``O(num_nodes)``
eagerly dispatched primitives, 1.56 ms per node. Only the KD-tree takes that
fallback, and it is why its dual-tree traversal was 50x the radix backend's at
N = 1e5 while doing strictly less device work. It is now a pointer-doubling
``lax.while_loop``: one dispatched computation, ``O(log depth)`` rounds.

These tests pin the depths to the definition -- distance to the root along the
parent chain -- rather than to either implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import Tree
from yggdrax.dtypes import INDEX_DTYPE, as_index
from yggdrax.tree import get_node_levels, node_levels_from_parent


def _levels_by_relaxation(parent: np.ndarray) -> np.ndarray:
    """The pre-change fallback, verbatim: one max-relaxation round per node."""
    parent_j = jnp.asarray(parent, dtype=INDEX_DTYPE)
    num_nodes = int(parent_j.shape[0])
    if num_nodes == 0:
        return np.zeros((0,), dtype=np.int64)
    levels = jnp.zeros((num_nodes,), dtype=INDEX_DTYPE)
    parent_safe = jnp.where(parent_j >= 0, parent_j, as_index(0))
    for _ in range(max(num_nodes - 1, 0)):
        candidate = jnp.where(
            parent_j >= 0, levels[parent_safe] + as_index(1), as_index(0)
        )
        levels = jnp.maximum(levels, candidate)
    return np.asarray(levels)


def _heap_parents(n: int) -> np.ndarray:
    """Parent links of a heap-ordered binary tree of ``n`` nodes."""
    return np.array([-1] + [(i - 1) // 2 for i in range(1, n)], dtype=np.int32)


def _path_parents(n: int) -> np.ndarray:
    """Parent links of a single chain -- the worst case for depth."""
    return np.array([-1] + list(range(n - 1)), dtype=np.int32)


_SIZES = [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 63, 64, 100, 255, 256, 257]


@pytest.mark.parametrize("n", _SIZES)
def test_heap_depths_match_the_relaxation(n: int) -> None:
    parent = _heap_parents(n)
    got = np.asarray(node_levels_from_parent(jnp.asarray(parent, dtype=INDEX_DTYPE)))
    np.testing.assert_array_equal(got, _levels_by_relaxation(parent))
    # A heap's depth is the bit length of the 1-based index.
    expected = np.array([int(i + 1).bit_length() - 1 for i in range(n)])
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("n", _SIZES)
def test_path_depths_match_the_relaxation(n: int) -> None:
    parent = _path_parents(n)
    got = np.asarray(node_levels_from_parent(jnp.asarray(parent, dtype=INDEX_DTYPE)))
    np.testing.assert_array_equal(got, _levels_by_relaxation(parent))
    np.testing.assert_array_equal(got, np.arange(n))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forests_with_padded_nodes_match_the_relaxation(seed: int) -> None:
    """Several roots and unattached nodes: every ``parent < 0`` is depth 0."""
    rng = np.random.default_rng(seed)
    n = 257
    parent = np.array(
        [-1]
        + [
            int(rng.integers(0, max(1, i))) if rng.random() > 0.1 else -1
            for i in range(1, n)
        ],
        dtype=np.int32,
    )
    got = np.asarray(node_levels_from_parent(jnp.asarray(parent, dtype=INDEX_DTYPE)))
    np.testing.assert_array_equal(got, _levels_by_relaxation(parent))
    np.testing.assert_array_equal(got[parent < 0], 0)


def test_empty_topology_returns_empty_levels() -> None:
    class _Empty:
        parent = jnp.zeros((0,), dtype=INDEX_DTYPE)

    assert get_node_levels(_Empty()).shape == (0,)


@pytest.mark.parametrize("backend", ["radix", "octree", "kdtree"])
def test_backend_levels_agree_with_the_relaxation(backend: str) -> None:
    """Every backend's depths, derived and (where carried) declared.

    The KD-tree is the one that matters: it carries no ``node_level``, so it is
    the only backend whose depths ``get_node_levels`` has to derive.
    """
    key = jax.random.PRNGKey(0)
    kp, km = jax.random.split(key)
    positions = jax.random.uniform(
        kp, (2000, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    masses = jax.random.uniform(km, (2000,), minval=0.5, maxval=1.5, dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type=backend,
        build_mode="adaptive",
        leaf_size=64,
        return_reordered=True,
    )
    topo = tree.topology if hasattr(tree, "topology") else tree
    parent = np.asarray(jnp.asarray(topo.parent, dtype=INDEX_DTYPE))
    derived = np.asarray(
        node_levels_from_parent(jnp.asarray(parent, dtype=INDEX_DTYPE))
    )
    np.testing.assert_array_equal(derived, _levels_by_relaxation(parent))
    if hasattr(topo, "node_level"):
        declared = np.asarray(jnp.asarray(topo.node_level, dtype=INDEX_DTYPE))
        np.testing.assert_array_equal(derived, declared)
    np.testing.assert_array_equal(np.asarray(get_node_levels(topo)), derived)
