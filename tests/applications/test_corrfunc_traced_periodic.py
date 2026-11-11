"""The traced soft pair counts in a periodic box (``box_size=``).

Covers :func:`yggdrax.applications.corrfunc.traced.traced_soft_pair_counts`
with a periodic box: every separation is the minimum-image one, so the result
must equal the O(n^2) minimum-image brute force

* for the window rule (hard and smooth), the geometric rule and ``membership=``;
* with particles on faces, edges and corners, outside the box (wrapped), and a
  cluster wrapped through a corner;
* in a cube and a non-cubic box, log and linear bins;

with gradients matching brute-force autodiff and finite differences, under
``jit``, ``grad`` and ``vmap``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.applications.corrfunc.binning import (
    make_linear_edges,
    make_log_edges,
    soft_bin_weights,
)
from yggdrax.applications.corrfunc.traced import (
    _build_topology,
    _periodic_positions,
    plan_traced_pair_counts,
    traced_soft_pair_counts,
)
from yggdrax.periodic import minimum_image, normalize_box_size

_N = 320
_LEAF = 16
_SHARP = 100.0
_L = 1.0
# Wide log bins (deep interiors, so the window rule accepts far pairs) and
# narrow ones; both stay below L/2 with their eps tails (0.3 e^0.14 = 0.34).
_WIDE = make_log_edges(0.02, 0.3, 3)
_NARROW = make_log_edges(0.02, 0.3, 8)
# Two very wide bins at n = 1000: the window rule accepts ~1e3 far pairs.
_FAR = make_log_edges(0.02, 0.45, 2)
_N_FAR = 1000
_TOL = 1e-6


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _box_array(box):
    return jnp.asarray(normalize_box_size(box, 3), dtype=jnp.float64)


def _faces_and_corners(seed: int = 0, n: int = _N, box=_L) -> jax.Array:
    """Uniform points plus points ON faces, edges and corners, some outside.

    The special points sit at exactly 0, at ``L - 1e-12`` and at ``L`` (which
    wraps to 0), across faces from each other, and a few are given unwrapped
    (``-0.01``, ``L + 0.02``) to exercise the wrap.
    """
    lengths = _box_array(box)
    base = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), dtype=jnp.float64)
    base = base * lengths
    lo, hi = 0.0, 1.0 - 1e-12
    special = jnp.asarray(
        [
            [lo, lo, lo],
            [hi, hi, hi],
            [lo, hi, lo],
            [hi, lo, hi],
            [1.0, 0.5, 0.5],  # on the upper face: wraps to x = 0
            [lo, 0.5, 0.5],
            [0.5, lo, 0.5],
            [0.5, hi, 0.5],
            [0.5, 0.5, lo],
            [0.5, 0.5, hi],
            [lo, lo, 0.3],  # on an edge
            [hi, hi, 0.31],
            [-0.01, 0.2, 0.7],  # outside: wraps
            [1.02, 0.21, 0.71],
            [0.01, 0.99, 0.5],
            [0.99, 0.01, 0.5],
        ],
        dtype=jnp.float64,
    )
    return base.at[: special.shape[0]].set(special * lengths)


def _corner_cluster(seed: int = 0, n: int = _N, box=_L) -> jax.Array:
    """Half the points in a tight blob centred ON the corner, wrapped around it."""
    lengths = _box_array(box)
    k_u, k_c = jax.random.split(jax.random.PRNGKey(seed))
    uniform = jax.random.uniform(k_u, (n - n // 2, 3), dtype=jnp.float64) * lengths
    blob = 0.03 * jax.random.normal(k_c, (n // 2, 3), dtype=jnp.float64)
    return jnp.concatenate([uniform, blob])  # blob has negative coordinates


def _brute(positions, edges, box=_L, sharpness=_SHARP, log=True) -> jax.Array:
    """``sum_{i<j} w_k(|minimum_image(x_i - x_j)|)`` by the O(n^2) double loop."""
    n = positions.shape[0]
    d = minimum_image(positions[:, None, :] - positions[None, :, :], box)
    r = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30)
    w = soft_bin_weights(r, edges, sharpness, log=log)
    upper = jnp.triu(jnp.ones((n, n), dtype=positions.dtype), k=1)
    return jnp.sum(w * upper[..., None], axis=(0, 1))


def _loss(counts: jax.Array) -> jax.Array:
    return jnp.sum(jnp.log1p(counts) * jnp.arange(1.0, counts.shape[0] + 1.0))


def _plan(probes, edges, box=_L, sharpness=_SHARP, log=True, **kw):
    kw.setdefault("leaf_size", _LEAF)
    return plan_traced_pair_counts(
        probes, edges=edges, sharpness=sharpness, log=log, box_size=box, **kw
    )


def _traced(pos, edges, config, box=_L, sharpness=_SHARP, log=True, **kw):
    return traced_soft_pair_counts(
        pos, edges, config, sharpness=sharpness, log=log, box_size=box, **kw
    )


def _max_rel_err(got, ref) -> float:
    return float(np.max(np.abs(np.asarray(got) / np.asarray(ref) - 1.0)))


_CATALOGUES = {"faces": _faces_and_corners, "corner_cluster": _corner_cluster}


# --------------------------------------------------------------------------
# values against the minimum-image brute force
# --------------------------------------------------------------------------


@pytest.mark.parametrize("box", [_L, (1.0, 0.8, 1.2)], ids=["cube", "cuboid"])
@pytest.mark.parametrize("case", sorted(_CATALOGUES))
@pytest.mark.parametrize("edges", [_WIDE, _NARROW], ids=["wide", "narrow"])
def test_window_rule_matches_minimum_image_brute_force(box, case, edges):
    pos = _CATALOGUES[case](seed=1, box=box)
    cfg = _plan(pos, edges, box=box)
    res = jax.jit(lambda p: _traced(p, edges, cfg, box=box))(pos)
    assert not bool(res.overflow)
    ref = _brute(pos, edges, box=box)
    assert _max_rel_err(res.counts, ref) < _TOL
    # the box matters: the open-boundary sum is visibly different
    assert _max_rel_err(_brute(pos, edges, box=None), ref) > 1e-3


def test_window_rule_accepts_far_pairs_across_faces():
    """Wide bins: the far term is live, including pairs that wrap."""
    pos = _faces_and_corners(seed=2, n=_N_FAR)
    cfg = _plan(pos, _FAR)
    res = _traced(pos, _FAR, cfg)
    assert int(res.far_count) > 0
    assert _max_rel_err(res.counts, _brute(pos, _FAR)) < _TOL
    wrapped, _ = _periodic_positions(pos, None, normalize_box_size(_L, 3))
    topo = _build_topology(
        wrapped, cfg, None, _FAR, _SHARP, True, box=normalize_box_size(_L, 3)
    )
    centers = np.asarray(wrapped)[np.asarray(topo.order)]
    ranges = np.asarray(topo.node_ranges)
    count = int(topo.walk.far_count)
    far_a = np.asarray(topo.walk.far_a)[:count]
    far_b = np.asarray(topo.walk.far_b)[:count]

    def com(nodes):
        return np.stack(
            [centers[ranges[i, 0] : ranges[i, 1] + 1].mean(0) for i in nodes]
        )

    raw = com(far_b) - com(far_a)
    # some accepted pairs are closer through a face than inside the box
    assert np.any(np.abs(raw) > 0.5 * _L)


def test_linear_bins_and_sharpness():
    edges = make_linear_edges(0.05, 0.3, 5)
    pos = _corner_cluster(seed=3)
    cfg = _plan(pos, edges, log=False, sharpness=40.0)
    res = _traced(pos, edges, cfg, log=False, sharpness=40.0)
    assert not bool(res.overflow)
    ref = _brute(pos, edges, sharpness=40.0, log=False)
    assert _max_rel_err(res.counts, ref) < _TOL


@pytest.mark.parametrize("case", sorted(_CATALOGUES))
def test_geometric_rule_tiles_every_pair_once(case):
    """A window equal to 1 at every separation counts n(n-1)/2 at any theta.

    The monopole of a constant is exact, so this checks only that within-leaf,
    near and far tile the pairs once, with the min-image far acceptance and no
    periodic guard.
    """
    edges = jnp.asarray([-1.0, 10.0])
    pos = _CATALOGUES[case](seed=4)
    cfg = _plan(pos, edges, log=False, acceptance="geometric", theta=0.6)
    res = _traced(pos, edges, cfg, log=False)
    assert int(res.far_count) > 0
    np.testing.assert_allclose(res.counts[0], _N * (_N - 1) / 2, rtol=1e-12)


def test_geometric_rule_with_only_point_far_pairs_is_exact():
    pos = _faces_and_corners(seed=5)
    cfg = _plan(pos, _NARROW, acceptance="geometric", theta=0.02)
    res = _traced(pos, _NARROW, cfg)
    np.testing.assert_allclose(res.counts, _brute(pos, _NARROW), rtol=1e-12)


def test_geometric_rule_skips_the_walks_periodic_guard():
    """Far pairs beyond L/2 are accepted: no O(n^2) refinement in the corners.

    The walk's own periodic guard would refuse every node pair with
    ``d + r_a + r_b >= L/2`` -- about half of all pairs in a cube -- and leave
    them to the near field. Without it the periodic walk does about the work
    of the open-boundary one.
    """
    n = 1024
    pos = _faces_and_corners(seed=6, n=n)
    edges = jnp.asarray([-1.0, 10.0])
    kw = {"log": False, "acceptance": "geometric", "theta": 0.6}
    res = _traced(pos, edges, _plan(pos, edges, **kw), log=False)
    open_cfg = plan_traced_pair_counts(
        pos, edges=edges, sharpness=_SHARP, leaf_size=_LEAF, **kw
    )
    res_open = traced_soft_pair_counts(
        pos, edges, open_cfg, sharpness=_SHARP, log=False
    )
    np.testing.assert_allclose(res.counts[0], n * (n - 1) / 2, rtol=1e-12)
    assert int(res.near_count) < 2 * int(res_open.near_count)


@pytest.mark.parametrize("acceptance", ["window", "geometric"])
def test_smooth_rule_matches_brute_force(acceptance):
    pos = _corner_cluster(seed=7)
    theta = 0.05 if acceptance == "geometric" else 0.5
    cfg = _plan(pos, _WIDE, acceptance=acceptance, theta=theta, smooth_band=2.0)
    res = _traced(pos, _WIDE, cfg)
    assert not bool(res.overflow)
    assert int(res.band_count) > 0
    tol = _TOL if acceptance == "window" else 1e-3
    assert _max_rel_err(res.counts, _brute(pos, _WIDE)) < tol


def test_large_box_equals_the_open_boundary_estimator():
    """No image within reach: the periodic result is the non-periodic one."""
    pos = jax.random.uniform(jax.random.PRNGKey(8), (_N, 3), dtype=jnp.float64)
    big = 10.0
    cfg = _plan(pos, _WIDE, box=big)
    res = _traced(pos, _WIDE, cfg, box=big)
    ref = _brute(pos, _WIDE, box=None)
    assert _max_rel_err(res.counts, ref) < _TOL


def test_bounds_and_box_are_exclusive():
    pos = _faces_and_corners()
    cfg = _plan(pos, _WIDE)
    with pytest.raises(ValueError, match="bounds or box_size"):
        traced_soft_pair_counts(
            pos,
            _WIDE,
            cfg,
            sharpness=_SHARP,
            box_size=_L,
            bounds=(jnp.zeros(3), jnp.ones(3)),
        )


# --------------------------------------------------------------------------
# membership: the leaf partition from x0, live positions anywhere
# --------------------------------------------------------------------------


def test_membership_with_particles_crossing_faces():
    """x0 inside, x = x0 + step pushing face particles through the face."""
    x0 = _faces_and_corners(seed=9)
    step = 0.004 * jax.random.normal(jax.random.PRNGKey(1), x0.shape)
    x = x0 + step  # face particles leave [0, L): some coordinates < 0 or > L
    assert bool(jnp.any(x < 0)) and bool(jnp.any(x >= _L))
    cfg = _plan([x0, x], _WIDE, smooth_band=2.0)
    res = _traced(x, _WIDE, cfg, membership=x0)
    assert not bool(res.overflow)
    assert _max_rel_err(res.counts, _brute(x, _WIDE)) < _TOL


def test_membership_keeps_a_face_crossing_continuous():
    """Moving one particle through x = 0: no restructuring, so no jump."""
    x0 = _faces_and_corners(seed=10)
    i = 5  # the particle at x = 0 on the lower face
    cfg = _plan(x0, _WIDE, smooth_band=2.0)

    def f(t):
        x = x0.at[i, 0].add(t)
        return _traced(x, _WIDE, cfg, membership=x0).counts

    ts = jnp.linspace(-2e-3, 2e-3, 9)
    vals = np.stack([np.asarray(f(t)) for t in ts])
    second = vals[2:] - 2 * vals[1:-1] + vals[:-2]
    # a jump would show up as a second difference of order eps * pairs
    assert np.max(np.abs(second)) < 1e-6 * np.max(np.abs(vals))


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(_CATALOGUES))
def test_gradient_matches_brute_force_autodiff(case):
    pos = _CATALOGUES[case](seed=11)
    cfg = _plan(pos, _WIDE)
    g = jax.jit(jax.grad(lambda p: _loss(_traced(p, _WIDE, cfg).counts)))(pos)
    g_ref = jax.grad(lambda p: _loss(_brute(p, _WIDE)))(pos)
    scale = float(jnp.abs(g_ref).max())
    np.testing.assert_allclose(g, g_ref, rtol=1e-5, atol=1e-6 * scale)


@pytest.mark.parametrize("smooth_band", [1.0, 2.0], ids=["hard", "smooth"])
def test_gradient_matches_finite_differences(smooth_band):
    """Positions on faces and corners and the edges, central differences.

    Wide bins at n = 1000, so the far monopole's gradient is in the check.
    """
    pos = _faces_and_corners(seed=12, n=_N_FAR)
    cfg = _plan(pos, _FAR, smooth_band=smooth_band)
    assert int(_traced(pos, _FAR, cfg).far_count) > 0

    def f(p, edges):
        return _loss(_traced(p, edges, cfg).counts)

    g_pos, g_edges = jax.jit(jax.grad(f, argnums=(0, 1)))(pos, _FAR)
    f_jit = jax.jit(f)
    h = 1e-6
    # the corner, face and wrapped particles, then a few random ones
    rng = np.random.default_rng(0)
    rows = [0, 1, 4, 5, 12, 13, *rng.integers(16, _N_FAR, 4).tolist()]
    for i in rows:
        for k in range(3):
            fd = (
                float(f_jit(pos.at[i, k].add(h), _FAR))
                - float(f_jit(pos.at[i, k].add(-h), _FAR))
            ) / (2 * h)
            np.testing.assert_allclose(g_pos[i, k], fd, rtol=1e-5, atol=1e-6)
    for k in (1, 2):
        fd = (
            float(f_jit(pos, _FAR.at[k].add(h))) - float(f_jit(pos, _FAR.at[k].add(-h)))
        ) / (2 * h)
        np.testing.assert_allclose(g_edges[k], fd, rtol=1e-5, atol=1e-6)


def test_gradient_with_membership_across_a_face():
    """grad at a point whose particles sit outside [0, L), x0 fixing leaves."""
    x0 = _faces_and_corners(seed=13)
    x = x0.at[5, 0].add(-1e-3).at[1, 2].add(2e-3)  # through x = 0 and z = L
    cfg = _plan([x0, x], _WIDE, smooth_band=2.0)

    def f(p):
        return _loss(_traced(p, _WIDE, cfg, membership=x0).counts)

    g = jax.jit(jax.grad(f))(x)
    g_ref = jax.grad(lambda p: _loss(_brute(p, _WIDE)))(x)
    scale = float(jnp.abs(g_ref).max())
    np.testing.assert_allclose(g, g_ref, rtol=1e-5, atol=1e-6 * scale)
    f_jit = jax.jit(f)
    h = 1e-6
    for i, k in ((5, 0), (1, 2), (0, 1)):
        fd = (float(f_jit(x.at[i, k].add(h))) - float(f_jit(x.at[i, k].add(-h)))) / (
            2 * h
        )
        np.testing.assert_allclose(g[i, k], fd, rtol=1e-5, atol=1e-6)


def test_vmap_over_catalogues_matches_a_loop():
    batch = jnp.stack([_faces_and_corners(seed=s) for s in range(3)])
    cfg = _plan(batch, _WIDE)
    run = jax.jit(jax.vmap(lambda p: _traced(p, _WIDE, cfg).counts))
    got = run(batch)
    for b in range(3):
        np.testing.assert_allclose(
            got[b], _traced(batch[b], _WIDE, cfg).counts, rtol=1e-12
        )
    grads = jax.jit(jax.vmap(jax.grad(lambda p: _loss(_traced(p, _WIDE, cfg).counts))))(
        batch
    )
    assert bool(jnp.all(jnp.isfinite(grads)))


@pytest.mark.parametrize("case", sorted(_CATALOGUES))
def test_sub_blocks_cull_across_faces(case):
    """The culled near field (minimum-image box gaps) equals the dense one."""
    pos = _CATALOGUES[case](seed=14, n=1000)
    culled = _plan(pos, _NARROW, leaf_size=32)
    assert culled.sub_block == 8
    dense = _plan(pos, _NARROW, leaf_size=32, sub_block=0)
    res_c = _traced(pos, _NARROW, culled)
    res_d = _traced(pos, _NARROW, dense)
    assert not bool(res_c.overflow)
    # most sub-block pairs of the near leaf pairs are out of reach
    assert int(res_c.sub_count) < 0.5 * int(res_d.near_count) * 16
    np.testing.assert_allclose(res_c.counts, res_d.counts, rtol=1e-12)
    assert _max_rel_err(res_c.counts, _brute(pos, _NARROW)) < _TOL
    g_c = jax.grad(lambda p: _loss(_traced(p, _NARROW, culled).counts))(pos)
    g_ref = jax.grad(lambda p: _loss(_brute(p, _NARROW)))(pos)
    scale = float(jnp.abs(g_ref).max())
    np.testing.assert_allclose(g_c, g_ref, rtol=1e-5, atol=1e-6 * scale)
