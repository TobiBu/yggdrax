"""Device (JAX) static-shape ADAPTIVE-octree interaction lists (Stage 2): the O(N) FMM.

``build_adaptive_octree_execution_view_device(..., interactions=True)`` populates the
variable-depth tree's interaction lists: the V-list M2L pairs (``v_src`` / ``v_tgt``, same
level) and the EXTENDED-NEAR list per leaf (``u_offsets`` / ``u_neighbors``: U adjacent +
W finer + X coarser level-jump neighbours, folded into one leaf-leaf P2P list). The tree
is 2:1-balanced first (device ripple) so the near list is bounded. Gates:

* (COVERAGE) the decisive EXACT-PARTITION gate -- for every occupied leaf ``B`` the union of
  {B's own particles} + {extended-near source leaves} + {V-list sources of B AND every
  ancestor of B} equals all N particles, each EXACTLY once (0 missing, 0 double). This is
  the uniform ``test_device_view_coverage`` extended to the variable-level tree, and it is
  what certifies the whole thing is a correct O(N) FMM.
* (PARITY) feeding the completed view through jaccpot's ``octree_fmm_uvwx`` far (V-list M2L
  + L2L) + near (extended-near P2P) matches direct softened N-body to expansion-order.
* (NO-RECOMPILE) jitting with static structural args compiles once across position sets.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from yggdrax.octree_uvwx import build_adaptive_octree_execution_view_device


def _points(n, seed, dist):
    rng = np.random.default_rng(seed)
    if dist == "clustered":  # concentrated Plummer-like cloud
        r = rng.uniform(size=n) ** (1.0 / 3.0)
        r = r / (1.0 - 0.85 * r)
        d = rng.standard_normal((n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pos = r[:, None] * d
    else:
        pos = rng.uniform(-1.0, 1.0, size=(n, 3))
    mass = np.abs(rng.standard_normal(n)) + 0.5
    return pos.astype(np.float64), mass.astype(np.float64)


def _build(pos, max_depth, leaf_size, node_cap, leaf_cap, u_cap):
    return build_adaptive_octree_execution_view_device(
        pos,
        max_depth,
        leaf_size,
        node_capacity=node_cap,
        leaf_capacity=leaf_cap,
        interactions=True,
        u_capacity=u_cap,
    )


@pytest.mark.parametrize(
    ("n", "max_depth", "leaf_size", "dist"),
    [
        (2000, 6, 64, "uniform"),
        (2000, 6, 64, "clustered"),
        (4000, 7, 96, "clustered"),
        (5000, 8, 96, "clustered"),
        (8000, 7, 128, "uniform"),
        (8000, 8, 128, "clustered"),
    ],
)
def test_interaction_exact_partition_coverage(n, max_depth, leaf_size, dist):
    """(COVERAGE, decisive) self + extended-near + V(B and ancestors) == all N, once each."""
    node_cap, leaf_cap, u_cap = 12000, 10000, 256
    pos, _ = _points(n, 7, dist)
    v = _build(pos, max_depth, leaf_size, node_cap, leaf_cap, u_cap)

    parent = np.asarray(v.parent)
    nr = np.asarray(v.node_ranges)
    leaf_idx = np.asarray(v.leaf_indices)
    v_src = np.asarray(v.v_src)
    v_tgt = np.asarray(v.v_tgt)
    u_nbr = np.asarray(v.u_neighbors)
    node_cap_i = int(v.parent.shape[0])
    leaf_cap_i = int(leaf_idx.shape[0])
    sentinel = node_cap_i

    assert int(v.num_valid_nodes) < node_cap_i, "node_capacity overflow"
    assert int(v.num_leaf_nodes) < leaf_cap_i, "leaf_capacity overflow"
    assert tuple(nr[0]) == (0, n - 1), "root does not span all particles"

    W = u_nbr.shape[0] // leaf_cap_i  # fixed CSR width
    starts = nr[:, 0].astype(np.int64)
    ends = nr[:, 1].astype(np.int64)  # inclusive

    far_by_tgt = defaultdict(list)
    for s, t in zip(v_src.tolist(), v_tgt.tolist()):
        if s != sentinel and t != sentinel:
            far_by_tgt[int(t)].append(int(s))

    def ancestors(node):
        out, x = [], int(node)
        while x >= 0:
            out.append(x)
            x = int(parent[x])
        return out

    n_bad = 0
    for r in range(leaf_cap_i):
        ln = int(leaf_idx[r])
        if ln >= node_cap_i:
            continue  # sentinel-padded leaf row
        if ends[ln] - starts[ln] + 1 <= 0:
            continue  # empty (balancing-added) leaf: no self particles to cover
        cov = np.zeros(n, np.int32)
        cov[starts[ln] : ends[ln] + 1] += 1  # self block (kernel adds it once)
        for e in u_nbr[r * W : (r + 1) * W].tolist():
            if e != sentinel and e < leaf_cap_i:
                s_node = int(leaf_idx[e])
                cov[starts[s_node] : ends[s_node] + 1] += 1
        for a in ancestors(ln):
            for s in far_by_tgt.get(a, []):
                cov[starts[s] : ends[s] + 1] += 1
        if int(cov.min()) != 1 or int(cov.max()) != 1:
            n_bad += 1
            if n_bad <= 3:
                print(
                    f"leaf row={r} node={ln}: missing={int((cov == 0).sum())} "
                    f"double={int((cov > 1).sum())}"
                )
    assert n_bad == 0, f"{n_bad} leaves are not an exact partition"


def _far_near_accelerations(view, pos, mass, order, G, soft, node_cap, max_depth):
    """Run the completed adaptive view through jaccpot's far + near FMM passes."""
    import jax.numpy as jnp
    from jaccpot.experimental.octree_fmm_uvwx import (
        _octree_far_field_grad,
        _ulist_near_device,
        octree_execution_data_from_view,
    )

    nr = np.asarray(view.node_ranges)
    leaf_idx = np.asarray(view.leaf_indices)
    real = leaf_idx < node_cap
    occ = (
        nr[np.clip(leaf_idx, 0, node_cap - 1), 1]
        - nr[np.clip(leaf_idx, 0, node_cap - 1), 0]
        + 1
    )
    max_leaf_size = int(occ[real].max())
    loff = np.asarray(view.level_offsets)
    level_batch_width = int(np.diff(loff).max())
    assert loff[max_depth - 1] + level_batch_width <= node_cap

    octree = octree_execution_data_from_view(view)
    perm = np.asarray(view.perm)
    ps = jnp.asarray(pos[perm])
    ms = jnp.asarray(mass[perm])
    far_grad = _octree_far_field_grad(
        view,
        octree,
        ps,
        ms,
        order=order,
        max_leaf_size=max_leaf_size,
        num_levels=max_depth + 1,
        level_batch_width=level_batch_width,
    )
    far_acc = -G * far_grad
    near_acc = _ulist_near_device(
        ps, ms, view, G=G, softening=soft, max_leaf_size=max_leaf_size
    )
    acc_sorted = np.asarray(far_acc + near_acc)
    inv = np.zeros_like(perm)
    inv[perm] = np.arange(perm.shape[0])
    return acc_sorted[inv]


def _direct(pos, mass, G, soft):
    d = pos[:, None, :] - pos[None, :, :]
    d2 = np.sum(d * d, axis=-1) + soft**2
    np.fill_diagonal(d2, np.inf)
    return -G * np.sum(mass[None, :, None] * d * (d2**-1.5)[..., None], axis=1)


@pytest.mark.parametrize(
    ("n", "max_depth", "leaf_size", "dist"),
    [
        (4000, 7, 96, "clustered"),
        (6000, 7, 128, "clustered"),
        (8000, 8, 128, "clustered"),
    ],
)
def test_interaction_parity_through_jaccpot(n, max_depth, leaf_size, dist):
    """(PARITY) far V-list M2L + extended-near P2P == direct softened N-body (order 4)."""
    pytest.importorskip("jaccpot")
    node_cap, leaf_cap, u_cap = 12000, 10000, 256
    G, soft = 1.0, 1e-2
    pos, mass = _points(n, 7, dist)
    v = _build(pos, max_depth, leaf_size, node_cap, leaf_cap, u_cap)
    assert int(v.num_valid_nodes) < node_cap and int(v.num_leaf_nodes) < leaf_cap

    acc = _far_near_accelerations(v, pos, mass, 4, G, soft, node_cap, max_depth)
    ref = _direct(pos, mass, G, soft)
    rel = np.linalg.norm(acc - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)
    med, p90 = float(np.median(rel)), float(np.percentile(rel, 90))
    assert med < 2e-2, f"median rel err {med:.3e}"
    assert p90 < 5e-2, f"p90 rel err {p90:.3e}"


def test_interaction_parity_converges_with_order():
    """(PARITY) order 6 beats order 4 -- confirms it is a real multipole far field."""
    pytest.importorskip("jaccpot")
    node_cap, leaf_cap, u_cap = 8000, 6000, 256
    G, soft = 1.0, 1e-2
    pos, mass = _points(4000, 7, "clustered")
    v = _build(pos, 7, 96, node_cap, leaf_cap, u_cap)
    ref = _direct(pos, mass, G, soft)

    def med(order):
        acc = _far_near_accelerations(v, pos, mass, order, G, soft, node_cap, 7)
        rel = np.linalg.norm(acc - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)
        return float(np.median(rel))

    m4, m6 = med(4), med(6)
    assert m6 < m4, f"order 6 ({m6:.3e}) not better than order 4 ({m4:.3e})"


def test_interaction_static_no_recompile():
    """(NO-RECOMPILE) static (max_depth, leaf_size, node/leaf/u caps) -> compile once."""
    build = jax.jit(
        partial(build_adaptive_octree_execution_view_device),
        static_argnums=(1, 2),
        static_argnames=(
            "node_capacity",
            "leaf_capacity",
            "interactions",
            "u_capacity",
        ),
    )
    n, max_depth, leaf_size = 3000, 7, 96
    caps = dict(
        node_capacity=8000, leaf_capacity=6000, interactions=True, u_capacity=256
    )

    compiles = []
    logger = __import__("logging").getLogger("jax")
    with jax.log_compiles():
        v1 = build(_points(n, 1, "uniform")[0], max_depth, leaf_size, **caps)
        v1.v_src.block_until_ready()

        class _Count(__import__("logging").Handler):
            def emit(self, record):
                if "Compiling" in record.getMessage():
                    compiles.append(record.getMessage())

        h = _Count()
        logger.addHandler(h)
        try:
            v2 = build(_points(n, 2, "clustered")[0], max_depth, leaf_size, **caps)
            v2.v_src.block_until_ready()
        finally:
            logger.removeHandler(h)

    assert compiles == [], f"unexpected recompilation: {compiles}"
    for f in ("v_src", "v_tgt", "u_offsets", "u_neighbors", "leaf_indices"):
        assert getattr(v1, f).shape == getattr(v2, f).shape
    assert int(v1.u_neighbors.shape[0]) == int(v2.u_neighbors.shape[0]) == 6000 * 256
