"""Structural invariants for the uniform-octree execution view.

``build_uniform_octree_execution_view`` derives the node-topology fields the FMM
operators consume. This gate checks those fields are self-consistent and in the
natural node layout (root at index 0, no reserved sentinel node) that the fixed
octree FMM kernels require:

* root at index 0, ``num_valid_nodes`` == total node count (all valid),
* child table consistent with ``parent`` and level-major (``parent < child``),
* particle-span rollup: root covers all particles, internal = union of children,
* ``level_offsets`` / ``nodes_by_level`` partition nodes by level exactly once,
* leaf bookkeeping (``leaf_mask`` / ``leaf_indices`` / ``leaf_nodes``) consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from yggdrax.octree_uvwx import (
    build_uniform_octree_execution_view,
    build_uniform_octree_uv,
)


def _points(n, seed, dist):
    rng = np.random.default_rng(seed)
    if dist == "clustered":
        r = rng.uniform(size=n) ** (1.0 / 3.0)
        r = r / (1.0 - 0.85 * r)
        d = rng.standard_normal((n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return r[:, None] * d
    return rng.uniform(-1.0, 1.0, size=(n, 3))


@pytest.mark.parametrize(
    ("n", "L", "dist"),
    [
        (3000, 3, "uniform"),
        (3000, 4, "uniform"),
        (5000, 4, "clustered"),
        (4000, 5, "clustered"),
    ],
)
def test_execution_view_structure(n, L, dist):
    pts = _points(n, 7, dist)
    view = build_uniform_octree_execution_view(pts, L)
    num_nodes = int(view.parent.shape[0])

    # natural node layout: root at 0, all nodes valid, num_valid_nodes == total
    assert view.num_valid_nodes == num_nodes
    assert bool(view.valid_mask.all())
    assert int(view.node_depths[0]) == 0  # index 0 is the root
    assert int(view.parent[0]) == -1
    assert int(view.num_levels) == L + 1

    # child table consistent with parent, and level-major (parent precedes child)
    for c in range(num_nodes):
        p = int(view.parent[c])
        if p >= 0:
            assert p < c, f"node {c}: parent {p} not before child (not level-major)"
            assert c in view.children[p][view.children[p] >= 0].tolist()
    assert np.array_equal(view.child_counts, (view.children >= 0).sum(1))

    # leaves have no children; internal nodes have children
    for c in range(num_nodes):
        has_children = bool((view.children[c] >= 0).any())
        assert has_children != bool(view.leaf_mask[c])

    # particle-span rollup: root covers all particles; internal = union of children.
    n_part = int(view.perm.shape[0])
    assert tuple(view.node_ranges[0]) == (0, n_part - 1)
    for c in sorted(range(num_nodes), key=lambda x: -int(view.node_depths[x])):
        if not view.leaf_mask[c]:
            ch = view.children[c][view.children[c] >= 0]
            assert int(view.node_ranges[c, 0]) == int(view.node_ranges[ch, 0].min())
            assert int(view.node_ranges[c, 1]) == int(view.node_ranges[ch, 1].max())

    # level_offsets / nodes_by_level partition nodes by level exactly once
    assert int(view.level_offsets[0]) == 0
    assert int(view.level_offsets[-1]) == num_nodes
    assert sorted(view.nodes_by_level.tolist()) == list(range(num_nodes))
    for lev in range(int(view.num_levels)):
        s, e = int(view.level_offsets[lev]), int(view.level_offsets[lev + 1])
        block = view.nodes_by_level[s:e]
        assert bool((view.node_depths[block] == lev).all())

    # leaf bookkeeping consistent
    assert view.num_leaf_nodes == int(view.leaf_mask.sum())
    assert view.num_leaf_nodes == len(view.leaf_indices)
    assert bool(view.leaf_mask[view.leaf_indices].all())
    assert np.array_equal(view.leaf_nodes[: view.num_leaf_nodes], view.leaf_indices)


def test_execution_view_lists_match_uv_build():
    """The execution view carries the same U/V lists as the underlying U/V build."""
    pts = _points(3000, 7, "clustered")
    uv = build_uniform_octree_uv(pts, 4)
    view = build_uniform_octree_execution_view(pts, 4)
    assert np.array_equal(view.v_src, uv.v_src)
    assert np.array_equal(view.v_tgt, uv.v_tgt)
    assert np.array_equal(view.u_offsets, uv.u_offsets)
    assert np.array_equal(view.u_neighbors, uv.u_neighbors)
    assert np.array_equal(view.leaf_indices, uv.leaf_indices)
    assert np.array_equal(view.perm, uv.perm)
