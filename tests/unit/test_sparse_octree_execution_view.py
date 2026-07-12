"""Device (JAX) static-shape SPARSE uniform-octree build: coverage + no-recompile gate.

build_sparse_uniform_octree_execution_view_device stores only the OCCUPIED cells as nodes
(padded to a fixed ``node_capacity`` / ``leaf_capacity``), so the node count scales with the
data's occupancy instead of the dense ``(8^(L+1)-1)/7`` -- which lets a uniform-depth octree
FMM run on clustered data (a concentrated galaxy disk) at a depth deep enough to bound leaf
occupancy without OOMing. This mirrors the dense build's gates: (A) it is jittable with static
depth/capacities and does not retrace when only positions change, (B) the U/V lists form an
exact interaction partition (every particle handled once per occupied leaf), and (C) a small
feasibility check that the sparse node/leaf counts fit under capacity with headroom.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import partial

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from yggdrax.octree_uvwx import build_sparse_uniform_octree_execution_view_device


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
        (2000, 4, "uniform"),
        (2000, 4, "clustered"),
        (2000, 5, "clustered"),
    ],
)
def test_sparse_device_view_coverage(n, L, dist):
    """{self + U neighbours + M2L(V) over ancestors} covers every particle EXACTLY once."""
    pos = _points(n, 7, dist)
    v = build_sparse_uniform_octree_execution_view_device(
        pos, L, node_capacity=4000, leaf_capacity=3000
    )
    parent = np.asarray(v.parent)
    nr = np.asarray(v.node_ranges)
    leaf_idx = np.asarray(v.leaf_indices)
    v_src = np.asarray(v.v_src)
    v_tgt = np.asarray(v.v_tgt)
    u_nbr = np.asarray(v.u_neighbors)
    u_off = np.asarray(v.u_offsets)
    num_nodes = int(v.parent.shape[0])  # node array length == sentinel value
    sentinel = num_nodes
    num_valid = int(v.num_valid_nodes)

    # capacity headroom (no overflow) -- otherwise deepest cells would be dropped
    assert num_valid < num_nodes, "node_capacity overflow"
    assert int(v.num_leaf_nodes) < int(
        v.leaf_indices.shape[0]
    ), "leaf_capacity overflow"

    valid_leaves = [int(c) for c in leaf_idx if c < num_valid]
    off_L = int(min(valid_leaves))  # first (smallest) leaf-level node id
    assert tuple(nr[0]) == (0, n - 1)  # root covers all particles

    def particles(node):
        s, e = nr[node]
        return set(range(int(s), int(e) + 1))

    far_by_tgt = defaultdict(list)
    for s, t in zip(v_src.tolist(), v_tgt.tolist()):
        if s != sentinel and t != sentinel:
            far_by_tgt[int(t)].append(int(s))

    def ancestors(node):
        out = []
        x = int(node)
        while x >= 0:
            out.append(x)
            x = int(parent[x])
        return out

    all_p = set(range(n))
    occ_leaves = [c for c in valid_leaves if (int(nr[c, 1]) - int(nr[c, 0]) + 1) > 0]
    assert occ_leaves, "no occupied leaves"
    for ln in occ_leaves:
        r = ln - off_L
        cnt = Counter()
        cnt.update(particles(ln))  # self block (kernel adds it once)
        for e in u_nbr[u_off[r] : u_off[r] + 26].tolist():
            if e != sentinel:
                cnt.update(particles(int(e) + off_L))
        for a in ancestors(ln):
            for s in far_by_tgt.get(a, []):
                cnt.update(particles(s))
        covered = set(cnt)
        assert covered == all_p, f"leaf {ln}: missing {len(all_p - covered)} particles"
        assert not [p for p, c in cnt.items() if c > 1], f"leaf {ln}: double-counted"


def test_sparse_device_view_jittable_static_shapes():
    """Jitting with static depth/capacities compiles once and reuses across position sets."""
    build = jax.jit(
        partial(build_sparse_uniform_octree_execution_view_device),
        static_argnums=(1,),
        static_argnames=("node_capacity", "leaf_capacity"),
    )
    n, L = 1500, 4
    caps = dict(node_capacity=4000, leaf_capacity=3000)

    compiles = []
    logger = __import__("logging").getLogger("jax")
    with jax.log_compiles():
        v1 = build(_points(n, 1, "uniform"), L, **caps)
        v1.parent.block_until_ready()

        class _Count(__import__("logging").Handler):
            def emit(self, record):
                if "Compiling" in record.getMessage():
                    compiles.append(record.getMessage())

        h = _Count()
        logger.addHandler(h)
        try:
            v2 = build(_points(n, 2, "clustered"), L, **caps)
            v2.parent.block_until_ready()
        finally:
            logger.removeHandler(h)

    assert compiles == [], f"unexpected recompilation: {compiles}"
    for f in (
        "parent",
        "children",
        "node_ranges",
        "v_src",
        "u_neighbors",
        "leaf_indices",
    ):
        assert getattr(v1, f).shape == getattr(v2, f).shape
    assert int(v1.parent.shape[0]) == int(v2.parent.shape[0]) == 4000
    assert int(v1.leaf_indices.shape[0]) == int(v2.leaf_indices.shape[0]) == 3000


def test_sparse_device_view_feasibility_small():
    """Sparse node/leaf counts track occupancy and fit under capacity with headroom."""
    n, L = 4000, 5
    pos = _points(n, 3, "clustered")
    v = build_sparse_uniform_octree_execution_view_device(
        pos, L, node_capacity=6000, leaf_capacity=4096
    )
    num_valid = int(v.num_valid_nodes)
    num_leaf = int(v.num_leaf_nodes)
    nr = np.asarray(v.node_ranges)
    lm = np.asarray(v.leaf_mask)
    max_occ = int((nr[lm, 1] - nr[lm, 0] + 1).max())

    assert tuple(nr[0]) == (0, n - 1)
    assert 0 < num_valid < 6000  # occupied-only, well under dense (8^6-1)/7 = 37449
    assert 0 < num_leaf < 4096
    assert num_valid < (8 ** (L + 1) - 1) // 7  # far fewer than the dense node count
    assert max_occ >= 1
