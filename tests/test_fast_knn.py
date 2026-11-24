"""Exact kNN on the fast path (:mod:`yggdrax.fast_knn`) against cKDTree, on CPU.

The packet traversal (the default) runs its Pallas kernel in interpret mode on
small cases; the walk method (mutual walk + CSR + certificate) runs its XLA
top-k on larger ones. The compiled Triton lowering is exercised on GPU by the
baseline bench.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import cKDTree

from yggdrax.fast_knn import fast_knn, plan_fast_knn


def _uniform(rng, n):
    return rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)


def _clustered(rng, n):
    centres = rng.uniform(0.2, 0.8, (6, 3))
    widths = rng.uniform(0.01, 0.08, 6)
    lab = rng.integers(0, 6, n)
    pts = centres[lab] + widths[lab, None] * rng.standard_normal((n, 3))
    return pts.astype(np.float32)


def _reference(data, queries, k):
    tree = cKDTree(data.astype(np.float64))
    if queries is None:
        d, i = tree.query(data.astype(np.float64), k=k + 1)
        d, i = d.reshape(len(data), k + 1)[:, 1:], i.reshape(len(data), k + 1)[:, 1:]
    else:
        d, i = tree.query(queries.astype(np.float64), k=k)
        d, i = d.reshape(len(queries), k), i.reshape(len(queries), k)
    return d, i


def _run(data, queries, k, *, backend, interpret=False, theta=None, method, **kw):
    q = None if queries is None else jnp.asarray(queries)
    plan = plan_fast_knn(
        jnp.asarray(data),
        q,
        k=k,
        theta=theta,
        walk_backend="flat",
        interpret=interpret,
        method=method,
    )
    fn = jax.jit(
        lambda a, b: fast_knn(
            a,
            b,
            plan=plan,
            backend=backend,
            walk_backend="flat",
            interpret=interpret,
            **kw,
        )
    )
    return plan, fn(jnp.asarray(data), q)


def _assert_exact(res, data, queries, k):
    ref_d, ref_i = _reference(data, queries, k)
    assert not bool(res.overflow)
    assert bool(jnp.all(res.certified))
    d = np.asarray(res.distances)
    np.testing.assert_allclose(d, ref_d, rtol=1e-5, atol=1e-7)
    same = np.all(np.sort(np.asarray(res.indices), 1) == np.sort(ref_i, 1), axis=1)
    assert same.mean() >= 0.999  # float32 near-ties may swap a pair


@pytest.mark.parametrize(
    "n, nq, k, kind",
    [
        (3000, 0, 1, "uniform"),
        (3000, 0, 8, "uniform"),
        (3000, 2000, 32, "uniform"),
        (4000, 0, 8, "clustered"),
        (4000, 3000, 16, "clustered"),
    ],
)
def test_walk_method_matches_ckdtree_xla(n, nq, k, kind):
    rng = np.random.default_rng(n + nq + k)
    gen = _uniform if kind == "uniform" else _clustered
    data = gen(rng, n)
    queries = None if nq == 0 else gen(rng, nq)
    _, res = _run(data, queries, k, backend="xla", method="walk")
    if kind == "clustered" and not bool(jnp.all(res.certified)):
        # the walk's certificate leaves void queries uncertified on clustered
        # data (25 % of them at 4000/3000, k=16, 2026-11-21) -- the reason the
        # traversal is the default. What must hold: every certified row is exact.
        ref_d, _ = _reference(data, queries, k)
        cert = np.asarray(res.certified)
        assert not bool(res.overflow)
        np.testing.assert_allclose(
            np.asarray(res.distances)[cert], ref_d[cert], rtol=1e-5, atol=1e-7
        )
        return
    _assert_exact(res, data, queries, k)


@pytest.mark.parametrize(
    "n, nq, k, kind",
    [
        (700, 0, 8, "uniform"),
        (800, 500, 32, "uniform"),
        (900, 0, 16, "clustered"),
        (900, 700, 8, "clustered"),
        (300, 0, 1, "uniform"),
    ],
)
def test_traversal_pallas_interpret_matches_ckdtree(n, nq, k, kind):
    rng = np.random.default_rng(7 + n + k)
    gen = _uniform if kind == "uniform" else _clustered
    data = gen(rng, n)
    queries = None if nq == 0 else gen(rng, nq)
    _, res = _run(data, queries, k, backend="pallas", interpret=True, method="traverse")
    _assert_exact(res, data, queries, k)


@pytest.mark.parametrize("n, nq, k", [(800, 500, 32), (900, 0, 8), (1000, 900, 16)])
def test_traversal_clustered_matches_ckdtree(n, nq, k):
    # two trees (separate queries) and one tree (self) on clustered points,
    # with k == W (the direct fill) and k < W
    rng = np.random.default_rng(29 + n + k)
    data = _clustered(rng, n)
    queries = None if nq == 0 else _clustered(rng, nq)
    _, res = _run(data, queries, k, backend="pallas", interpret=True, method="traverse")
    _assert_exact(res, data, queries, k)


def test_walk_method_pallas_interpret_matches_ckdtree():
    rng = np.random.default_rng(17)
    data, queries, k = _uniform(rng, 800), _uniform(rng, 500), 8
    _, res = _run(data, queries, k, backend="pallas", interpret=True, method="walk")
    _assert_exact(res, data, queries, k)


def test_certificate_never_vouches_for_a_wrong_row():
    # a loose theta leaves true neighbours outside the near list on purpose
    rng = np.random.default_rng(3)
    data, queries, k = _uniform(rng, 4000), _uniform(rng, 3000), 32
    _, res = _run(data, queries, k, backend="xla", theta=0.85, method="walk")
    ref_d, _ = _reference(data, queries, k)
    exact = np.all(np.isclose(np.asarray(res.distances), ref_d, rtol=1e-5), axis=1)
    cert = np.asarray(res.certified)
    assert cert.mean() < 1.0  # the test needs uncertified rows to mean anything
    assert not np.any(cert & ~exact)


def test_differentiable_distances_and_gradient():
    rng = np.random.default_rng(11)
    data, queries, k = _uniform(rng, 1500), _uniform(rng, 600), 4
    plan = plan_fast_knn(jnp.asarray(data), jnp.asarray(queries), k=k)

    def loss(d, q):
        r = fast_knn(
            d, q, plan=plan, backend="pallas", interpret=True, differentiable=True
        )
        return jnp.sum(r.distances), r

    (val, res), (gd, gq) = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
    )(jnp.asarray(data), jnp.asarray(queries))
    ref_d, _ = _reference(data, queries, k)
    np.testing.assert_allclose(float(val), ref_d.sum(), rtol=1e-5)
    # d/dq sum_j |q - x_j| = sum_j (q - x_j) / |q - x_j| at the found neighbours
    nb = data[np.asarray(res.indices)]
    diff = queries[:, None, :] - nb
    expect = np.sum(diff / np.linalg.norm(diff, axis=-1, keepdims=True), axis=1)
    np.testing.assert_allclose(np.asarray(gq), expect, rtol=1e-4, atol=1e-5)
    assert np.all(np.isfinite(np.asarray(gd)))


def test_plan_shape_mismatch_raises():
    rng = np.random.default_rng(0)
    data = jnp.asarray(_uniform(rng, 500))
    plan = plan_fast_knn(data, None, k=4)
    with pytest.raises(ValueError, match="plan is for"):
        fast_knn(data[:400], None, plan=plan, backend="xla")


@pytest.mark.parametrize("merge_impl", ["replace", "insert"])
@pytest.mark.parametrize(
    "n, nq, k, kind, box",
    [
        (900, 0, 8, "uniform", 1.0),
        (900, 600, 32, "uniform", 2.0),
        (1000, 0, 16, "clustered", 1.0),
        (900, 700, 32, "clustered", None),
    ],
)
def test_traversal_periodic_and_merges_match_ckdtree(n, nq, k, kind, box, merge_impl):
    # periodic: points partly OUTSIDE [0, L) (the driver wraps them), checked
    # against cKDTree(boxsize=L) on the wrapped float64 points
    rng = np.random.default_rng(41 + n + k)
    L = 1.0 if box is None else box
    if kind == "uniform":
        data = rng.uniform(-0.3 * L, 1.3 * L, (n, 3)).astype(np.float32)
        queries = None if nq == 0 else rng.uniform(-0.3 * L, 1.3 * L, (nq, 3))
    else:
        data = _clustered(rng, n) * L
        queries = None if nq == 0 else _clustered(rng, nq) * L
    queries = None if queries is None else queries.astype(np.float32)
    q = None if queries is None else jnp.asarray(queries)
    plan = plan_fast_knn(jnp.asarray(data), q, k=k, box_size=box)
    res = jax.jit(
        lambda a, b: fast_knn(
            a, b, plan=plan, backend="pallas", interpret=True, merge_impl=merge_impl
        )
    )(jnp.asarray(data), q)

    def wrap(x):
        x = x.astype(np.float64)
        return x if box is None else np.where(np.mod(x, L) >= L, 0.0, np.mod(x, L))

    tree = cKDTree(wrap(data), boxsize=box)
    if queries is None:
        ref_d, ref_i = tree.query(wrap(data), k=k + 1)
        ref_d, ref_i = ref_d[:, 1:], ref_i[:, 1:]
    else:
        ref_d, ref_i = tree.query(wrap(queries), k=k)
    assert not bool(res.overflow)
    # float32 wrapping of out-of-box points moves them by ~1 ulp of L
    np.testing.assert_allclose(np.asarray(res.distances), ref_d, rtol=2e-5, atol=1e-6)
    same = np.all(np.sort(np.asarray(res.indices), 1) == np.sort(ref_i, 1), axis=1)
    assert same.mean() >= 0.999


def test_periodic_xla_reference_and_walk_rejection():
    rng = np.random.default_rng(5)
    data = jnp.asarray(_uniform(rng, 700))
    plan = plan_fast_knn(data, None, k=6, box_size=1.0)
    res = fast_knn(data, None, plan=plan, backend="xla")
    ref_d, _ = cKDTree(np.asarray(data, np.float64), boxsize=1.0).query(
        np.asarray(data, np.float64), k=7
    )
    np.testing.assert_allclose(np.asarray(res.distances), ref_d[:, 1:], rtol=1e-5)
    with pytest.raises(ValueError, match="periodic"):
        plan_fast_knn(data, None, k=6, box_size=1.0, method="walk")


@pytest.mark.parametrize("box", [None, 1.0])
@pytest.mark.parametrize("k", [4, 16])
def test_masked_knn_padding_and_roles_match_brute_force(box, k):
    # one padded buffer: rows that are data only, query only, both, and padding
    # (garbage coordinates far outside the box); the Pallas traversal (interpret)
    # must match the dense reference and give padding / non-query rows nothing
    from yggdrax.fast_knn import masked_knn

    rng = np.random.default_rng(100 + k + (box is not None))
    n = 1100
    pts = _clustered(rng, n).astype(np.float64)
    role = rng.integers(0, 4, n)  # 0 pad, 1 data, 2 query, 3 both
    pts[role == 0] = 1e6 * rng.standard_normal((int(np.sum(role == 0)), 3))
    is_d = jnp.asarray((role == 1) | (role == 3))
    is_q = jnp.asarray((role == 2) | (role == 3))
    lo, hi = jnp.zeros(3), jnp.ones(3)
    cap = int(2.5 * n / 32) + 8
    rows, d2, over = jax.jit(
        lambda p: masked_knn(
            p,
            is_d,
            is_q,
            k=k,
            bucket_capacity=cap,
            box_size=box,
            bounds=None if box else (lo - 0.5, hi + 0.5),
            backend="pallas",
            interpret=True,
        )
    )(jnp.asarray(pts))
    ref_r, ref_d2, _ = masked_knn(
        jnp.asarray(pts),
        is_d,
        is_q,
        k=k,
        bucket_capacity=cap,
        box_size=box,
        backend="xla",
    )
    assert not bool(over)
    rows, ref_r = np.asarray(rows), np.asarray(ref_r)
    q = np.asarray(is_q)
    assert np.all(rows[~q] == -1)
    np.testing.assert_allclose(np.asarray(d2)[q], np.asarray(ref_d2)[q], rtol=1e-5)
    same = np.all(np.sort(rows[q], 1) == np.sort(ref_r[q], 1), axis=1)
    assert same.mean() >= 0.999
    # every neighbour is a data row and never the query itself
    nb = rows[q]
    assert np.all(np.asarray(is_d)[nb[nb >= 0]])
    assert not np.any(nb == np.nonzero(q)[0][:, None])


def test_masked_knn_reports_a_bucket_overflow():
    from yggdrax.fast_knn import masked_knn

    rng = np.random.default_rng(1)
    pts = jnp.asarray(_uniform(rng, 600))
    ones = jnp.ones((600,), bool)
    _, _, over = masked_knn(
        pts, ones, ones, k=4, bucket_capacity=4, backend="pallas", interpret=True
    )
    assert bool(over)


def _layout_boxes_cover_their_leaves(pts, is_data, is_query, cap, box):
    import yggdrax.fast_knn as F

    p32 = F._wrap(jnp.asarray(pts, jnp.float32), box)
    frame = F._frame(p32, box) if box is not None else F._bounds(p32)
    b = F._build_radix(p32, 32, cap, bounds=frame, live=jnp.asarray(is_data | is_query))
    t = F._tables(b, jnp.asarray(is_data), jnp.asarray(is_query))
    inf = jnp.asarray(jnp.inf, t.x.dtype)
    src = t._replace(
        x=jnp.where(t.dmask, t.x, inf),
        y=jnp.where(t.dmask, t.y, inf),
        z=jnp.where(t.dmask, t.z, inf),
    )
    pre, leaves = F._radix_layout(b, src._replace(slot=jnp.where(t.dmask, t.slot, -1)))
    lo = np.stack([np.asarray(a) for a in pre[0:3]], 1)
    hi = np.stack([np.asarray(a) for a in pre[3:6]], 1)
    nlo, nhi = np.asarray(pre[8]), np.asarray(pre[9])
    llo = np.stack([np.asarray(a) for a in leaves[0:3]], 1)
    lhi = np.stack([np.asarray(a) for a in leaves[3:6]], 1)
    # per pre-order node, the union of its leaves' boxes (empty leaves: +-inf)
    ok = True
    for i in range(len(nlo)):
        if nhi[i] < nlo[i]:
            continue
        ulo = llo[nlo[i] : nhi[i] + 1].min(0)
        uhi = lhi[nlo[i] : nhi[i] + 1].max(0)
        ok &= bool(np.all(lo[i] <= ulo) and np.all(hi[i] >= uhi))
    return ok


@pytest.mark.parametrize("cap", [1400, 2938, 4096])
def test_radix_layout_node_boxes_cover_every_leaf(cap):
    # One device's local buffer of an 8-device distributed kNN: an eighth of a
    # uniform periodic box by Morton range (what the partition gives a device),
    # half data, half queries, a third of the rows padding. The bottom-up box
    # relaxation once stopped when the leaf RANGES had settled, leaving boxes
    # that missed deep leaves (the root missed a whole y-z slab), and
    # masked_knn dropped true neighbours of 3002 of 12624 queries (cap 2938).
    from yggdrax.morton import morton_encode

    rng = np.random.default_rng(1)
    pts_all = rng.uniform(0, 1, (200_000, 3)).astype(np.float32)
    codes = np.asarray(morton_encode(jnp.asarray(pts_all), (jnp.zeros(3), jnp.ones(3))))
    mine = np.argsort(codes, kind="stable")[3 * 25_000 : 4 * 25_000]
    n = 37_500
    pts = np.zeros((n, 3), np.float32)
    pts[:25_000] = pts_all[mine]
    is_data = np.zeros(n, bool)
    is_query = np.zeros(n, bool)
    role = rng.uniform(size=25_000) < 0.5
    is_data[:25_000] = role
    is_query[:25_000] = ~role
    perm = rng.permutation(n)
    assert _layout_boxes_cover_their_leaves(
        pts[perm], is_data[perm], is_query[perm], cap, (1.0, 1.0, 1.0)
    )
