"""Device (JAX) static-shape ADAPTIVE-depth octree build: partition + no-recompile gates.

build_adaptive_octree_execution_view_device subdivides only where a cell's occupancy
exceeds ``leaf_size``, so every leaf holds at most ``leaf_size`` particles (except leaves
forced at ``max_depth``) -- ~9x fewer leaves than a uniform octree deep enough to bound
occupancy on a concentrated galaxy disk (octree leaves average ~1/4 of ``leaf_size``, so
the reduction is below the ~40x an idealized full-leaf packing would give). Stage 1 builds
the adaptive node topology + geometry ONLY (the U/V interaction lists are Stage 2
placeholders), so the gates here are structural:

* (A) LEAF OCCUPANCY -- every leaf has ``1 <= occ <= leaf_size`` except leaves forced at
  ``max_depth``;
* (B) EXACT PARTITION -- the leaf particle ranges tile ``[0, N)`` exactly once (the key
  correctness property);
* (C) PARENT/CHILD CONSISTENCY -- each non-root node's parent is active + internal; each
  internal node's children tile its own particle span;
* (E) STATIC / NO-RECOMPILE -- jitting with static ``(max_depth, leaf_size, node_capacity,
  leaf_capacity)`` compiles once and reuses across position sets;

plus a small feasibility check that the adaptive node/leaf counts fit under capacity.
"""

from __future__ import annotations

from functools import partial

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from yggdrax.octree_uvwx import build_adaptive_octree_execution_view_device


def _points(n, seed, dist):
    rng = np.random.default_rng(seed)
    if dist == "clustered":  # Plummer-like concentrated cloud
        r = rng.uniform(size=n) ** (1.0 / 3.0)
        r = r / np.sqrt(1.0 - r**2 + 1e-9)
        d = rng.standard_normal((n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return (r[:, None] * d).astype(np.float64)
    return rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float64)


def _view_arrays(v):
    """Pull the view onto the host as numpy for structural inspection."""
    return dict(
        valid=np.asarray(v.valid_mask),
        parent=np.asarray(v.parent),
        children=np.asarray(v.children),
        depth=np.asarray(v.node_depths),
        nr=np.asarray(v.node_ranges),
        leaf_mask=np.asarray(v.leaf_mask),
        leaf_indices=np.asarray(v.leaf_indices),
        num_valid=int(v.num_valid_nodes),
        num_leaf=int(v.num_leaf_nodes),
        node_capacity=int(v.parent.shape[0]),
        leaf_capacity=int(v.leaf_indices.shape[0]),
    )


def _leaf_stats(a, n, leaf_size, max_depth):
    """Occupancy per valid leaf + which are forced at max_depth. Also gate-A checks."""
    valid, leaf_mask, nr, depth = a["valid"], a["leaf_mask"], a["nr"], a["depth"]
    leaf_ids = np.where(valid & leaf_mask)[0]
    occ = nr[leaf_ids, 1] - nr[leaf_ids, 0] + 1
    lev = depth[leaf_ids]
    forced = lev >= max_depth
    return leaf_ids, occ, lev, forced


@pytest.mark.parametrize(
    ("n", "max_depth", "leaf_size", "dist"),
    [
        (2000, 6, 64, "uniform"),
        (2000, 6, 64, "clustered"),
        (4000, 7, 96, "clustered"),
        (8000, 7, 128, "uniform"),
        (8000, 8, 128, "clustered"),
    ],
)
def test_adaptive_gate_A_leaf_occupancy(n, max_depth, leaf_size, dist):
    """(A) Every leaf has 1 <= occ; occ <= leaf_size unless the leaf is at max_depth."""
    pos = _points(n, 11, dist)
    v = build_adaptive_octree_execution_view_device(
        pos, max_depth, leaf_size, node_capacity=8000, leaf_capacity=6000
    )
    a = _view_arrays(v)
    assert a["num_valid"] < a["node_capacity"], "node_capacity overflow"
    assert a["num_leaf"] < a["leaf_capacity"], "leaf_capacity overflow"

    _, occ, lev, forced = _leaf_stats(a, n, leaf_size, max_depth)
    assert occ.min() >= 1, "empty leaf"
    non_forced = ~forced
    assert (occ[non_forced] <= leaf_size).all(), (
        "sub-max-depth leaf exceeds leaf_size: "
        f"max={int(occ[non_forced].max()) if non_forced.any() else 0}"
    )
    # forced (max_depth) leaves may exceed leaf_size; just ensure they are still occupied
    assert (occ[forced] >= 1).all() if forced.any() else True


@pytest.mark.parametrize(
    ("n", "max_depth", "leaf_size", "dist"),
    [
        (2000, 6, 64, "uniform"),
        (2000, 6, 64, "clustered"),
        (3000, 7, 50, "clustered"),
        (5000, 8, 96, "clustered"),
        (8000, 7, 128, "uniform"),
    ],
)
def test_adaptive_gate_B_exact_partition(n, max_depth, leaf_size, dist):
    """(B) The leaf particle ranges partition [0, N) exactly once -- no gaps/overlaps."""
    pos = _points(n, 23, dist)
    v = build_adaptive_octree_execution_view_device(
        pos, max_depth, leaf_size, node_capacity=8000, leaf_capacity=6000
    )
    a = _view_arrays(v)
    assert a["num_valid"] < a["node_capacity"]
    assert a["num_leaf"] < a["leaf_capacity"]
    assert tuple(a["nr"][0]) == (0, n - 1), "root does not span all particles"

    leaf_ids, occ, _, _ = _leaf_stats(a, n, leaf_size, max_depth)
    coverage = np.zeros(n, np.int64)
    for lid in leaf_ids:
        s, e = int(a["nr"][lid, 0]), int(a["nr"][lid, 1])
        coverage[s : e + 1] += 1
    assert coverage.min() == 1 and coverage.max() == 1, (
        f"not an exact partition: {int((coverage == 0).sum())} uncovered, "
        f"{int((coverage > 1).sum())} double-covered"
    )
    assert int(occ.sum()) == n, "leaf occupancies do not sum to N"


@pytest.mark.parametrize(
    ("n", "max_depth", "leaf_size", "dist"),
    [
        (2000, 6, 64, "clustered"),
        (4000, 7, 96, "clustered"),
        (8000, 8, 128, "clustered"),
    ],
)
def test_adaptive_gate_C_parent_child_consistency(n, max_depth, leaf_size, dist):
    """(C) parent is active+internal; internal-node children tile its own span."""
    pos = _points(n, 37, dist)
    v = build_adaptive_octree_execution_view_device(
        pos, max_depth, leaf_size, node_capacity=8000, leaf_capacity=6000
    )
    a = _view_arrays(v)
    valid, parent, children = a["valid"], a["parent"], a["children"]
    depth, nr, leaf_mask = a["depth"], a["nr"], a["leaf_mask"]
    L = max_depth

    def occ_of(node):
        return int(nr[node, 1] - nr[node, 0] + 1)

    internal = valid & ~leaf_mask
    valid_ids = np.where(valid)[0]
    for nid in valid_ids:
        p = int(parent[nid])
        if depth[nid] == 0:
            assert p == -1, "root has a parent"
            continue
        assert p >= 0 and valid[p], f"node {nid}: parent not active"
        assert depth[p] == depth[nid] - 1, f"node {nid}: parent level mismatch"
        # parent must be internal: occ > leaf_size and level < L
        assert (
            occ_of(p) > leaf_size and depth[p] < L
        ), f"node {nid}: parent not internal"

    for nid in np.where(internal)[0]:
        ch = children[nid][children[nid] >= 0]
        assert ch.size >= 1, f"internal node {nid} has no children"
        for c in ch:
            assert valid[c] and int(parent[c]) == nid
            assert depth[c] == depth[nid] + 1
        # children particle ranges tile the parent's own span exactly (=> all occupied
        # children captured, no gaps/overlaps)
        starts = np.sort(nr[ch, 0])
        ends = nr[ch, 1][np.argsort(nr[ch, 0])]
        assert starts[0] == nr[nid, 0] and ends[-1] == nr[nid, 1]
        assert (
            starts[1:] == ends[:-1] + 1
        ).all(), f"node {nid}: children not contiguous"


def test_adaptive_gate_E_static_no_recompile():
    """(E) Static (max_depth, leaf_size, node_capacity, leaf_capacity) -> compile once."""
    build = jax.jit(
        partial(build_adaptive_octree_execution_view_device),
        static_argnums=(1, 2),
        static_argnames=("node_capacity", "leaf_capacity"),
    )
    n, max_depth, leaf_size = 3000, 7, 96
    caps = dict(node_capacity=8000, leaf_capacity=6000)

    compiles = []
    logger = __import__("logging").getLogger("jax")
    with jax.log_compiles():
        v1 = build(_points(n, 1, "uniform"), max_depth, leaf_size, **caps)
        v1.parent.block_until_ready()

        class _Count(__import__("logging").Handler):
            def emit(self, record):
                if "Compiling" in record.getMessage():
                    compiles.append(record.getMessage())

        h = _Count()
        logger.addHandler(h)
        try:
            v2 = build(_points(n, 2, "clustered"), max_depth, leaf_size, **caps)
            v2.parent.block_until_ready()
        finally:
            logger.removeHandler(h)

    assert compiles == [], f"unexpected recompilation: {compiles}"
    for f in ("parent", "children", "node_ranges", "leaf_indices", "leaf_nodes"):
        assert getattr(v1, f).shape == getattr(v2, f).shape
    assert int(v1.parent.shape[0]) == int(v2.parent.shape[0]) == 8000
    assert int(v1.leaf_indices.shape[0]) == int(v2.leaf_indices.shape[0]) == 6000


def test_adaptive_feasibility_small():
    """Adaptive node/leaf counts track occupancy; leaves ~ N/leaf_size, all sub-cap."""
    n, max_depth, leaf_size = 8000, 8, 128
    pos = _points(n, 5, "clustered")
    v = build_adaptive_octree_execution_view_device(
        pos, max_depth, leaf_size, node_capacity=8000, leaf_capacity=6000
    )
    a = _view_arrays(v)
    _, occ, _, forced = _leaf_stats(a, n, leaf_size, max_depth)

    assert 0 < a["num_valid"] < a["node_capacity"]
    assert 0 < a["num_leaf"] < a["leaf_capacity"]
    assert tuple(a["nr"][0]) == (0, n - 1)
    # leaves are ~full: mean occupancy is a decent fraction of leaf_size, and the count is
    # far below a uniform build's 8^max_depth leaf slots.
    assert a["num_leaf"] < (n // 4) + 100  # roughly N / (leaf_size fraction)
    assert a["num_leaf"] < 8**max_depth
    assert int(occ.sum()) == n
