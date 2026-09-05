"""The KD-tree's heap helpers, against the recursions they replaced.

``_heap_subtree_sizes`` and ``_heap_inorder_nodes_jax`` used to be per-node
Python recursions run at trace time on every KD-tree build (258 ms and 707 ms at
n = 1e6). They are now one vectorised pass per level. These tests pin them to
the definitions rather than to the implementations, so a future rewrite has
something to be checked against.
"""

from __future__ import annotations

import numpy as np
import pytest

from yggdrax.kdtree import (
    _heap_inorder_nodes_jax,
    _heap_inorder_starts,
    _heap_subtree_sizes,
)


def _sizes_by_recursion(n: int) -> list[int]:
    """Subtree sizes, bottom-up, one node at a time."""
    sizes = [1] * n
    for i in range(n - 1, -1, -1):
        total = 1
        if 2 * i + 1 < n:
            total += sizes[2 * i + 1]
        if 2 * i + 2 < n:
            total += sizes[2 * i + 2]
        sizes[i] = total
    return sizes


def _inorder_by_walk(n: int) -> list[int]:
    """Heap indices in inorder, by an explicit stack walk."""
    order: list[int] = []
    stack: list[int] = []
    cur = 0
    while stack or (0 <= cur < n):
        while 0 <= cur < n:
            stack.append(cur)
            cur = 2 * cur + 1
        cur = stack.pop()
        order.append(cur)
        cur = 2 * cur + 2
    return order


_SIZES = list(range(0, 130)) + [255, 256, 257, 1000, 4095, 4096, 4097]


@pytest.mark.parametrize("n", _SIZES)
def test_subtree_sizes_match_the_recursion(n):
    assert np.array_equal(_heap_subtree_sizes(n), np.asarray(_sizes_by_recursion(n)))


@pytest.mark.parametrize("n", _SIZES)
def test_inorder_matches_the_stack_walk(n):
    expected = np.asarray(_inorder_by_walk(n), dtype=np.int32)
    assert np.array_equal(np.asarray(_heap_inorder_nodes_jax(n)), expected)


@pytest.mark.parametrize("n", [1, 2, 7, 8, 100, 1000])
def test_inorder_starts_are_the_subtree_ranges(n):
    """Each node's inorder range is exactly its subtree, contiguously."""
    sizes = _heap_subtree_sizes(n)
    starts = _heap_inorder_starts(n, sizes)
    inorder = np.asarray(_heap_inorder_nodes_jax(n))
    position = np.empty(n, dtype=np.int64)
    position[inorder] = np.arange(n)
    for node in range(n):
        lo, hi = starts[node], starts[node] + sizes[node]
        subtree = {node}
        frontier = [node]
        while frontier:
            i = frontier.pop()
            for child in (2 * i + 1, 2 * i + 2):
                if child < n:
                    subtree.add(child)
                    frontier.append(child)
        assert len(subtree) == sizes[node]
        assert set(position[sorted(subtree)]) == set(range(lo, hi))
