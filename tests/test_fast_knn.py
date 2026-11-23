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
