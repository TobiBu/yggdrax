"""``summary_tree``: the summary cut plus its ancestors, with child links.

What a two-sided export walk refines on the RECEIVER's side. The claims, each
asserted leaf by leaf rather than inferred from counts:

* the cut cells are exactly the live summary leaves, and the root is index 0;
* every internal summary node has BOTH children, and they are its tree children;
* the live summary leaves tile the live leaves exactly once (so a walk over the
  summary tree reaches every receiver particle exactly once);
* a size-bounded cut works the same way;
* the overflow flag fires.

    JAX_PLATFORMS=cpu pytest tests/distributed/test_summary_tree.py -q
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax._cell_partition import adaptive_cell_leaf_partition
from yggdrax._tree_impl import _build_balanced_bucket_structure
from yggdrax.bounds import infer_bounds
from yggdrax.distributed.summary import occupancy_cut, summary_tree
from yggdrax.morton import morton_encode


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _structure(n=2000, leaf_size=16, capacity=None, seed=0):
    """A cell-leaf tree, optionally padded with empty leaves to ``capacity``."""
    P = jnp.asarray(_plummer(n, seed), jnp.float32)
    codes = jnp.sort(morton_encode(P, infer_bounds(P)))
    k = int(
        adaptive_cell_leaf_partition(codes, leaf_size=leaf_size, capacity=n).num_leaves
    )
    cap = int(capacity or k)
    part = adaptive_cell_leaf_partition(codes, leaf_size=leaf_size, capacity=cap)
    starts = np.asarray(part.leaf_starts).astype(np.int64)
    ends = np.asarray(part.leaf_ends).astype(np.int64)
    parent, left, right, _li, _ri, ranges, *_ = _build_balanced_bucket_structure(
        starts, ends
    )
    I = jnp.int32
    return dict(
        parent=jnp.asarray(parent, I),
        left=jnp.asarray(left, I),
        right=jnp.asarray(right, I),
        ranges=jnp.asarray(ranges, I),
        nint=cap - 1,
        k=k,
        n=n,
    )


def _build(t, *, max_leaves=4, capacity=None, cut_capacity=None, **bound):
    cut = occupancy_cut(
        t["parent"],
        t["ranges"],
        t["nint"],
        max_leaves=max_leaves,
        capacity=cut_capacity or t["k"] + 8,
        **bound,
    )
    st = summary_tree(
        t["parent"],
        t["left"],
        t["right"],
        t["ranges"],
        t["nint"],
        cut,
        capacity=capacity or 2 * (t["k"] + 8),
    )
    return cut, st


def _live_leaves(t):
    nr = np.asarray(t["ranges"])
    live = (nr[:, 1] >= nr[:, 0]) & (nr[:, 0] < t["n"])
    rows = np.flatnonzero(live)
    return rows[rows >= t["nint"]]


def _check(t, cut, st):
    """Every structural claim of the docstring, on one (tree, cut, summary)."""
    assert not bool(st.overflow)
    m = int(st.num_nodes)
    nodes = np.asarray(st.nodes)[:m]
    left = np.asarray(st.left)[:m]
    right = np.asarray(st.right)[:m]
    is_cell = np.asarray(st.is_cell)[:m]
    active = np.asarray(st.active)[:m]
    par = np.asarray(t["parent"])
    nint = t["nint"]
    tl = np.concatenate([np.asarray(t["left"]), np.full(len(par) - nint, -1)])
    tr = np.concatenate([np.asarray(t["right"]), np.full(len(par) - nint, -1)])

    # padding is -1 / False past num_nodes
    assert np.all(np.asarray(st.nodes)[m:] == -1)
    assert not np.any(np.asarray(st.active)[m:])
    # the root is index 0, nodes are distinct
    assert nodes[0] == int(np.argmin(par))
    assert len(set(nodes.tolist())) == m

    # the cut cells are EXACTLY the live summary leaves
    cells = set(np.asarray(cut.cells)[: int(cut.num_cells)].tolist())
    leaf = left < 0
    assert set(nodes[is_cell].tolist()) == cells
    assert set(nodes[leaf & active].tolist()) == cells
    assert np.all(right[leaf] < 0)
    assert np.all(~active[leaf & ~is_cell]), "a non-cell summary leaf must be empty"

    # internal summary nodes: both children present, and they are the tree's
    internal = ~leaf
    assert np.all(right[internal] >= 0)
    assert np.all(nodes[left[internal]] == tl[nodes[internal]])
    assert np.all(nodes[right[internal]] == tr[nodes[internal]])
    assert np.all(active[internal]), "an internal summary node must hold live leaves"

    # every summary node but the root is reached from exactly one parent
    kids = np.concatenate([left[internal], right[internal]])
    assert sorted(kids.tolist()) == list(range(1, m))

    # the live summary leaves tile the live leaves exactly once
    nr = np.asarray(t["ranges"])
    cov = np.zeros(t["n"], np.int32)
    for c in nodes[leaf & active].tolist():
        cov[nr[c, 0] : nr[c, 1] + 1] += 1
    assert cov.min() == 1 and cov.max() == 1
    for lf in _live_leaves(t).tolist():
        s, e = nr[lf]
        hits = [
            c for c in nodes[leaf & active].tolist() if nr[c, 0] <= s and e <= nr[c, 1]
        ]
        assert len(hits) == 1, f"leaf {lf} under {len(hits)} summary leaves"
    return m


@pytest.mark.parametrize("max_leaves", [1, 4, 16])
def test_the_summary_tree_is_the_cut_plus_its_ancestors(max_leaves):
    t = _structure()
    cut, st = _build(t, max_leaves=max_leaves)
    m = _check(t, cut, st)
    # a full binary tree over the cells (no padding here, so no empty children)
    assert m == 2 * int(cut.num_cells) - 1


def test_a_padded_shard_keeps_the_empty_children_as_inactive_leaves():
    """The padding subtree joins the live tree at one node with an EMPTY child.

    That child must be in the summary as an inactive leaf: dropped, its parent would
    read ``left < 0`` (or a -1 right child) and a walk would stop there or skip it.
    """
    k0 = _structure()["k"]
    t = _structure(capacity=int(2 ** np.ceil(np.log2(1.6 * k0))))
    cut, st = _build(t)
    m = _check(t, cut, st)
    inactive_leaves = int(np.sum(~np.asarray(st.active)[:m]))
    assert inactive_leaves >= 1, "vacuous: no empty child reached the summary"
    assert m == 2 * (int(cut.num_cells) + inactive_leaves) - 1


def test_a_size_bounded_cut_works_too():
    t = _structure()
    nr = np.asarray(t["ranges"])
    # any extent that shrinks down the tree will do: the particle count of a node
    extent = jnp.asarray(np.maximum(nr[:, 1] - nr[:, 0] + 1, 0), jnp.float32)
    plain, _ = _build(t)
    cut, st = _build(t, node_extent=extent, max_extent=24.0)
    assert int(cut.num_cells) > int(plain.num_cells), "vacuous: the bound did not bite"
    _check(t, cut, st)


def test_the_whole_tree_in_one_cell_is_a_single_root():
    t = _structure()
    cut, st = _build(t, max_leaves=10 * t["k"])
    assert int(cut.num_cells) == 1
    assert int(st.num_nodes) == 1
    assert int(np.asarray(st.nodes)[0]) == int(np.argmin(np.asarray(t["parent"])))
    assert int(np.asarray(st.left)[0]) == -1 and bool(np.asarray(st.is_cell)[0])


def test_the_overflow_flag_fires():
    t = _structure()
    cut, st = _build(t)
    m = int(st.num_nodes)
    _, short = _build(t, capacity=m - 1)
    assert bool(short.overflow)
    # and a truncated CUT propagates, even with room for the tree
    _, from_short_cut = _build(t, cut_capacity=int(cut.num_cells) - 1)
    assert bool(from_short_cut.overflow)
    # CONTROL: the exact capacity does not fire
    _, exact = _build(t, capacity=m)
    assert not bool(exact.overflow)
