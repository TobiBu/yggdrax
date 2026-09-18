"""``occupancy_cut``: the summary the cross-domain exchange is addressed by.

The defining property is that it is a CUT -- every live leaf has exactly one
ancestor-or-self in it. Everything the exchange does rests on that: a cell's export
list is then a partition for every target inside it, which is what a flat pool of
nodes is not.

    JAX_PLATFORMS=cpu pytest tests/distributed/test_summary.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax._cell_partition import adaptive_cell_leaf_partition
from yggdrax._tree_impl import _build_balanced_bucket_structure
from yggdrax.bounds import infer_bounds
from yggdrax.distributed.summary import occupancy_cut, subtree_leaf_counts
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
    """Parent / ranges / num_internal of a cell-leaf tree, optionally padded."""
    P = jnp.asarray(_plummer(n, seed), jnp.float32)
    codes = jnp.sort(morton_encode(P, infer_bounds(P)))
    probe = adaptive_cell_leaf_partition(codes, leaf_size=leaf_size, capacity=n)
    k = int(probe.num_leaves)
    cap = int(capacity or k)
    part = adaptive_cell_leaf_partition(codes, leaf_size=leaf_size, capacity=cap)
    starts = np.asarray(part.leaf_starts).astype(np.int64)
    ends = np.asarray(part.leaf_ends).astype(np.int64)
    parent, _l, _r, _li, _ri, ranges, *_ = _build_balanced_bucket_structure(
        starts, ends
    )
    return (
        jnp.asarray(parent, jnp.int32),
        jnp.asarray(ranges, jnp.int32),
        cap - 1,
        k,
        n,
    )


def _ancestors(parent, node, root):
    out = [node]
    while node != root:
        node = int(parent[node])
        out.append(node)
    return out


def _live_leaves(ranges, num_internal, n_valid):
    nr = np.asarray(ranges)
    live = (nr[:, 1] >= nr[:, 0]) & (nr[:, 0] < n_valid)
    rows = np.flatnonzero(live)
    return rows[rows >= num_internal]


@pytest.mark.parametrize("max_leaves", [1, 2, 4, 16, 64])
def test_every_live_leaf_has_exactly_one_ancestor_or_self_in_the_cut(max_leaves):
    """The cut property itself, checked leaf by leaf -- not inferred from counts."""
    parent, ranges, nint, k, n = _structure()
    s = occupancy_cut(parent, ranges, nint, max_leaves=max_leaves, capacity=k + 8)
    assert not bool(s.overflow)
    cut = set(np.asarray(s.cells)[: int(s.num_cells)].tolist())
    assert cut
    par = np.asarray(parent)
    root = int(np.argmin(par))
    for leaf in _live_leaves(ranges, nint, n).tolist():
        hits = [a for a in _ancestors(par, leaf, root) if a in cut]
        assert len(hits) == 1, f"leaf {leaf} has {len(hits)} cut ancestors"


@pytest.mark.parametrize("max_leaves", [1, 4, 16])
def test_the_cells_tile_the_particles_and_the_leaf_counts_add_up(max_leaves):
    parent, ranges, nint, k, n = _structure()
    s = occupancy_cut(parent, ranges, nint, max_leaves=max_leaves, capacity=k + 8)
    cells = np.asarray(s.cells)[: int(s.num_cells)]
    nr = np.asarray(ranges)
    cov = np.zeros(n, np.int32)
    for c in cells.tolist():
        cov[nr[c, 0] : nr[c, 1] + 1] += 1
    assert cov.min() == 1 and cov.max() == 1, "cells must tile the particles"
    assert int(np.asarray(s.leaves_per_cell)[: int(s.num_cells)].sum()) == int(
        _live_leaves(ranges, nint, n).size
    )
    assert np.all(np.asarray(s.leaves_per_cell)[: int(s.num_cells)] <= max_leaves)


def test_the_extremes_are_the_leaves_and_the_root():
    parent, ranges, nint, k, n = _structure()
    one = occupancy_cut(parent, ranges, nint, max_leaves=1, capacity=k + 8)
    assert set(np.asarray(one.cells)[: int(one.num_cells)].tolist()) == set(
        _live_leaves(ranges, nint, n).tolist()
    )
    allof = occupancy_cut(parent, ranges, nint, max_leaves=10 * k, capacity=k + 8)
    assert int(allof.num_cells) == 1
    assert int(np.asarray(allof.cells)[0]) == int(np.argmin(np.asarray(parent)))


def test_a_padded_shard_tiles_only_its_live_particles():
    """A capacity-padded tree must summarise its live half and nothing else."""
    k0 = _structure()[3]
    parent, ranges, nint, k, n = _structure(
        capacity=int(2 ** np.ceil(np.log2(1.6 * k0)))
    )
    nr = np.asarray(ranges)
    s = occupancy_cut(parent, ranges, nint, max_leaves=4, capacity=k + 64, num_valid=n)
    assert not bool(s.overflow)
    cells = np.asarray(s.cells)[: int(s.num_cells)]
    cov = np.zeros(n, np.int32)
    for c in cells.tolist():
        cov[nr[c, 0] : nr[c, 1] + 1] += 1
    assert cov.min() == 1 and cov.max() == 1, "must tile exactly the LIVE particles"
    assert np.all(np.asarray(s.leaves_per_cell)[: int(s.num_cells)] <= 4)
    assert int(np.asarray(s.leaves_per_cell)[: int(s.num_cells)].sum()) == int(
        _live_leaves(ranges, nint, n).size
    )
    # NOT a padded-vs-unpadded equality: the two trees have different balanced
    # structures over different leaf-slot counts, so the cut lands on different
    # nodes covering different particle blocks. `num_valid` cannot change that.


def test_num_valid_is_a_no_op_on_this_ranges_convention():
    """Measured, not assumed: here `num_valid` changes nothing, and that is correct.

    The padding leaves this path produces carry ``start > end`` (measured:
    ``[n, n-1]``), so they hold no leaf starts and the ``sub > 0`` guard already
    excludes them. The argument is kept for interface consistency with the rest of
    yggdrax's `num_valid` threading, and because a padding leaf carrying
    ``start == end == n`` is a convention this codebase has elsewhere -- but on THIS
    convention it is inert, and pretending otherwise with a synthetic tree only
    produces a test that passes for an accidental reason.
    """
    k0 = _structure()[3]
    parent, ranges, nint, k, n = _structure(
        capacity=int(2 ** np.ceil(np.log2(1.6 * k0)))
    )
    nr = np.asarray(ranges)
    padding = nr[nint:][nr[nint:, 0] >= n]
    assert padding.size and np.all(padding[:, 0] > padding[:, 1])
    with_nv = occupancy_cut(
        parent, ranges, nint, max_leaves=4, capacity=k + 64, num_valid=n
    )
    without = occupancy_cut(parent, ranges, nint, max_leaves=4, capacity=k + 64)
    assert np.array_equal(np.asarray(with_nv.cells), np.asarray(without.cells))
    assert np.array_equal(
        np.asarray(with_nv.leaves_per_cell), np.asarray(without.leaves_per_cell)
    )


def test_overflow_is_flagged_and_the_prefix_is_still_honest():
    parent, ranges, nint, k, n = _structure()
    full = occupancy_cut(parent, ranges, nint, max_leaves=1, capacity=k + 8)
    small = occupancy_cut(parent, ranges, nint, max_leaves=1, capacity=16)
    assert bool(small.overflow)
    assert int(small.num_cells) == 16
    # what it did return is the true prefix, not a truncated lie about the rest
    assert np.array_equal(np.asarray(small.cells), np.asarray(full.cells)[:16])


def test_subtree_leaf_counts_match_a_direct_walk():
    parent, ranges, nint, k, n = _structure(n=800, leaf_size=16)
    sub = np.asarray(subtree_leaf_counts(ranges, nint))
    par = np.asarray(parent)
    root = int(np.argmin(par))
    ref = np.zeros(par.size, np.int64)
    for leaf in _live_leaves(ranges, nint, n).tolist():
        for a in _ancestors(par, leaf, root):
            ref[a] += 1
    assert np.array_equal(sub, ref)


def test_it_traces():
    """The exchange runs inside shard_map, so the cut must be jittable."""
    parent, ranges, nint, k, n = _structure(n=600)
    f = jax.jit(
        lambda p, r: occupancy_cut(p, r, nint, max_leaves=4, capacity=k + 8),
        static_argnums=(),
    )
    s = f(parent, ranges)
    eager = occupancy_cut(parent, ranges, nint, max_leaves=4, capacity=k + 8)
    assert np.array_equal(np.asarray(s.cells), np.asarray(eager.cells))


@pytest.mark.parametrize("bad", [0, -1])
def test_validation(bad):
    parent, ranges, nint, k, n = _structure(n=400)
    with pytest.raises(ValueError):
        occupancy_cut(parent, ranges, nint, max_leaves=bad, capacity=8)
    with pytest.raises(ValueError):
        occupancy_cut(parent, ranges, nint, max_leaves=4, capacity=bad)
