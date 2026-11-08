"""The traced soft pair-count estimator: tree rebuilt inside the traced function.

Covers :mod:`yggdrax.applications.corrfunc.traced`:

* the window acceptance rule (default): per-bin relative error against brute
  force below 1e-6 across opening angles, leaf sizes, clustering and window
  shapes, and gradients matching brute-force autodiff;
* the geometric rule: exact pair coverage, brute-force values when only point
  pairs are far;
* composition with ``jit(grad)``, ``lax.scan`` and ``vmap`` with the rebuild
  inside, and the overflow flag of every capacity.
"""

from __future__ import annotations

import dataclasses

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
    TracedPairCountConfig,
    _build_topology,
    plan_traced_pair_counts,
    traced_soft_pair_counts,
)

_N = 300
_LEAF = 16
_SHARP = 100.0
# Narrow, sharp log bins: almost nothing inside [r_min, r_max] is flat enough
# to accept, so the window rule prunes beyond r_max and sums the rest exactly.
_EDGES = make_log_edges(0.02, 0.8, 8)
# Wide log bins: deep bin interiors exist, so the window rule accepts far pairs
# and all three terms (within-leaf, near, far) are exercised.
_WIDE = make_log_edges(0.02, 0.8, 3)
_TOL = 1e-6  # per-bin relative error the window rule must meet
# Fixed Morton domain, so a finite-difference nudge cannot move the bounds.
_BOUNDS = (jnp.full(3, -0.1), jnp.full(3, 1.1))


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _catalog(seed: int = 0, n: int = _N) -> jax.Array:
    return jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), dtype=jnp.float64)


def _clustered(seed: int = 0, n: int = _N, num_centers: int = 5) -> jax.Array:
    """Points pulled 80 % of the way onto a few centres: strongly clustered."""
    k_base, k_cen, k_pick = jax.random.split(jax.random.PRNGKey(seed), 3)
    base = jax.random.uniform(k_base, (n, 3), dtype=jnp.float64)
    centers = jax.random.uniform(k_cen, (num_centers, 3), dtype=jnp.float64)
    pick = jax.random.randint(k_pick, (n,), 0, num_centers)
    return 0.2 * base + 0.8 * centers[pick]


def _brute(positions, edges=_EDGES, sharpness=_SHARP, log=True) -> jax.Array:
    """``sum_{i<j} w_k(r_ij)`` by the O(n^2) double loop."""
    n = positions.shape[0]
    d = positions[:, None, :] - positions[None, :, :]
    r = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30)
    w = soft_bin_weights(r, edges, sharpness, log=log)
    upper = jnp.triu(jnp.ones((n, n), dtype=positions.dtype), k=1)
    return jnp.sum(w * upper[..., None], axis=(0, 1))


def _loss(counts: jax.Array) -> jax.Array:
    # weights the bins unevenly so the gradient is not a symmetric sum
    return jnp.sum(jnp.log1p(counts) * jnp.arange(1.0, counts.shape[0] + 1.0))


def _traced(
    positions, config, edges=_EDGES, bounds=_BOUNDS, sharpness=_SHARP, log=True
):
    return traced_soft_pair_counts(
        positions, edges, config, sharpness=sharpness, log=log, bounds=bounds
    )


def _plan(positions, edges=_EDGES, sharpness=_SHARP, log=True, **kw):
    kw.setdefault("leaf_size", _LEAF)
    kw.setdefault("bounds", _BOUNDS)
    return plan_traced_pair_counts(
        positions, edges=edges, sharpness=sharpness, log=log, **kw
    )


def _max_rel_err(got, ref) -> float:
    return float(np.max(np.abs(np.asarray(got) / np.asarray(ref) - 1.0)))


# --------------------------------------------------------------------------
# window acceptance (default)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def uniform_500():
    jax.config.update("jax_enable_x64", True)
    pos = _catalog(seed=11, n=500)
    return pos, _brute(pos)


@pytest.mark.parametrize("leaf_size", [16, 32])
@pytest.mark.parametrize("theta", [0.2, 0.4, 0.6, 0.8])
def test_window_rule_matches_brute_force_at_any_theta_and_leaf(
    uniform_500, theta, leaf_size
):
    pos, ref = uniform_500
    cfg = _plan(pos, theta=theta, leaf_size=leaf_size)
    assert cfg.acceptance == "window"
    res = jax.jit(lambda p: _traced(p, cfg))(pos)
    assert not bool(res.overflow)
    assert _max_rel_err(res.counts, ref) < _TOL


def test_window_rule_ignores_theta(uniform_500):
    pos, _ = uniform_500
    counts = [_traced(pos, _plan(pos, theta=theta)).counts for theta in (0.2, 0.8)]
    np.testing.assert_array_equal(counts[0], counts[1])


@pytest.mark.parametrize(
    ("case", "edges", "sharpness", "log"),
    [
        ("clustered", _EDGES, _SHARP, True),
        ("wide_bins", _WIDE, _SHARP, True),
        ("soft_window", _EDGES, 20.0, True),
        ("linear_bins", make_linear_edges(0.05, 0.5, 6), 200.0, False),
        ("leaf_8", _EDGES, _SHARP, True),
    ],
)
def test_window_rule_matches_brute_force_across_problems(case, edges, sharpness, log):
    pos = _clustered(seed=4, n=500) if case == "clustered" else _catalog(seed=5, n=500)
    leaf = 8 if case == "leaf_8" else _LEAF
    cfg = _plan(pos, edges=edges, sharpness=sharpness, log=log, leaf_size=leaf)
    res = _traced(pos, cfg, edges=edges, sharpness=sharpness, log=log)
    assert not bool(res.overflow)
    ref = _brute(pos, edges, sharpness, log)
    assert _max_rel_err(res.counts, ref) < _TOL
    if case == "wide_bins":
        assert int(res.far_count) > 0, "wide bins must exercise the far term"


def test_window_rule_beats_the_geometric_rule_at_equal_theta(uniform_500):
    """The failure mode it fixes: the geometric MAC misbins at theta 0.4."""
    pos, ref = uniform_500
    geo = _traced(pos, _plan(pos, acceptance="geometric", theta=0.4))
    win = _traced(pos, _plan(pos, theta=0.4))
    assert _max_rel_err(geo.counts, ref) > 1e-3
    assert _max_rel_err(win.counts, ref) < _TOL


@pytest.mark.parametrize("edges", [_EDGES, _WIDE], ids=["narrow", "wide"])
def test_window_rule_gradient_matches_brute_force_autodiff(edges):
    pos = _catalog(seed=3)
    cfg = _plan([pos], edges=edges)
    res = _traced(pos, cfg, edges=edges)
    assert not bool(res.overflow)
    if edges is _WIDE:
        assert int(res.far_count) > 0
    g = jax.jit(jax.grad(lambda p: _loss(_traced(p, cfg, edges=edges).counts)))(pos)
    g_ref = jax.grad(lambda p: _loss(_brute(p, edges)))(pos)
    scale = float(jnp.abs(g_ref).max())
    np.testing.assert_allclose(g, g_ref, rtol=_TOL, atol=_TOL * scale)


def test_window_planning_needs_edges_and_sharpness():
    with pytest.raises(ValueError, match="edges and sharpness"):
        plan_traced_pair_counts(_catalog(), leaf_size=_LEAF)


# --------------------------------------------------------------------------
# geometric acceptance
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def near_only_config() -> TracedPairCountConfig:
    """theta so small that the walk accepts no multi-particle node as far.

    A single-particle leaf has radius 0, so the non-strict Dehnen test
    ``(r_a + r_b)^2 <= theta^2 d^2`` accepts a pair of them at any theta; their
    monopole is the exact pair term, so the sum is still exact.
    """
    jax.config.update("jax_enable_x64", True)
    probes = [_catalog(seed=0), _catalog(seed=3)]
    return _plan(probes, acceptance="geometric", theta=0.02)


@pytest.fixture(scope="module")
def geometric_config() -> TracedPairCountConfig:
    jax.config.update("jax_enable_x64", True)
    return _plan(_catalog(), acceptance="geometric", theta=0.6)


def test_geometric_matches_brute_force_when_only_points_are_far(near_only_config):
    pos = _catalog()
    res = jax.jit(lambda p: _traced(p, near_only_config))(pos)
    assert not bool(res.overflow)
    assert int(res.near_count) > 0
    topo = _build_topology(pos, near_only_config, _BOUNDS)
    ranges = np.asarray(topo.node_ranges)
    count = int(topo.walk.far_count)
    for nodes in (topo.walk.far_a, topo.walk.far_b):
        nodes = np.asarray(nodes)[:count]
        np.testing.assert_array_equal(ranges[nodes, 1] - ranges[nodes, 0], 0)
    np.testing.assert_allclose(res.counts, _brute(pos), rtol=1e-12)
    g = jax.jit(jax.grad(lambda p: _loss(_traced(p, near_only_config).counts)))(pos)
    g_ref = jax.grad(lambda p: _loss(_brute(p)))(pos)
    np.testing.assert_allclose(
        g, g_ref, rtol=1e-9, atol=1e-12 * float(jnp.abs(g_ref).max())
    )


def test_geometric_pair_lists_tile_every_pair_once(geometric_config):
    """Within-leaf + near + far cover the n(n-1)/2 pairs exactly once."""
    pos = _catalog()
    topo = _build_topology(pos, geometric_config, _BOUNDS)
    walk = topo.walk
    assert int(walk.far_count) > 0
    ranges = np.asarray(topo.node_ranges)

    def size(nodes):
        return np.maximum(ranges[nodes, 1] - ranges[nodes, 0] + 1, 0)

    far_a = np.asarray(walk.far_a)[: int(walk.far_count)]
    far_b = np.asarray(walk.far_b)[: int(walk.far_count)]
    near_a = np.asarray(walk.near_a)[: int(walk.near_count)]
    near_b = np.asarray(walk.near_b)[: int(walk.near_count)]
    leaves = size(np.arange(topo.num_internal, ranges.shape[0]))
    covered = (
        np.sum(size(far_a) * size(far_b))
        + np.sum(size(near_a) * size(near_b))
        + np.sum(leaves * (leaves - 1) // 2)
    )
    assert covered == _N * (_N - 1) // 2


# --------------------------------------------------------------------------
# composition: jit(grad), scan, vmap, overflow -- on the default window rule
# with wide bins, so within-leaf, near and far terms are all present
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wide_config() -> TracedPairCountConfig:
    jax.config.update("jax_enable_x64", True)
    return _plan([_catalog(s) for s in range(4)], edges=_WIDE)


def _wide(positions, config, bounds=_BOUNDS):
    return _traced(positions, config, edges=_WIDE, bounds=bounds)


def test_gradient_wrt_positions_and_edges_matches_finite_difference(wide_config):
    """Fixed-topology gradient (far terms included) vs central differences."""
    pos = _catalog(seed=1)
    assert int(_wide(pos, wide_config).far_count) > 0

    def f(p, edges):
        res = traced_soft_pair_counts(
            p, edges, wide_config, sharpness=_SHARP, bounds=_BOUNDS
        )
        return _loss(res.counts)

    g_pos, g_edges = jax.jit(jax.grad(f, argnums=(0, 1)))(pos, _WIDE)
    f_jit = jax.jit(f)
    h = 1e-6
    rng = np.random.default_rng(0)
    for i, k in zip(rng.integers(0, _N, 6), rng.integers(0, 3, 6), strict=True):
        fd = (
            float(f_jit(pos.at[i, k].add(h), _WIDE))
            - float(f_jit(pos.at[i, k].add(-h), _WIDE))
        ) / (2 * h)
        np.testing.assert_allclose(g_pos[i, k], fd, rtol=1e-5, atol=1e-6)
    for k in (1, 2):
        fd = (
            float(f_jit(pos, _WIDE.at[k].add(h)))
            - float(f_jit(pos, _WIDE.at[k].add(-h)))
        ) / (2 * h)
        np.testing.assert_allclose(g_edges[k], fd, rtol=1e-5, atol=1e-6)


def test_jit_grad_rebuilds_the_tree_inside(wide_config):
    """One compiled grad serves different catalogues: the tree is not stale."""
    grad_fn = jax.jit(jax.grad(lambda p: _loss(_wide(p, wide_config).counts)))
    jaxpr = str(jax.make_jaxpr(grad_fn)(_catalog()))
    assert "sort" in jaxpr and "while" in jaxpr  # Morton sort + mutual walk

    # Without a rebuild, a permuted catalogue would reuse the first tree's
    # leaf ranges on the wrong particles. With one, grad is equivariant.
    pos = _catalog(seed=2)
    perm = jax.random.permutation(jax.random.PRNGKey(9), _N)
    g = grad_fn(pos)
    g_perm = grad_fn(pos[perm])
    np.testing.assert_allclose(g_perm, g[perm], rtol=1e-10, atol=1e-12)


def test_scan_of_gradient_steps_with_in_loop_rebuilds(wide_config):
    target = _wide(_catalog(seed=0), wide_config).counts

    def loss(p):
        res = _wide(p, wide_config)
        return jnp.mean((jnp.log1p(res.counts) - jnp.log1p(target)) ** 2), res.overflow

    def step(p, _):
        (value, overflow), g = jax.value_and_grad(loss, has_aux=True)(p)
        p = jnp.clip(p - 1e-4 * g, 0.0, 1.0)
        return p, (value, overflow)

    start = _catalog(seed=1)
    final, (losses, overflows) = jax.jit(
        lambda p: jax.lax.scan(step, p, None, length=3)
    )(start)
    assert losses.shape == (3,)
    assert not bool(jnp.any(overflows))
    assert bool(jnp.all(jnp.isfinite(final)))
    assert not np.allclose(final, start)
    np.testing.assert_allclose(losses[0], loss(start)[0], rtol=1e-12)


def test_vmap_over_catalogues_matches_a_loop(wide_config):
    batch = jnp.stack([_catalog(seed=s) for s in range(3)])
    out = jax.jit(jax.vmap(lambda p: _wide(p, wide_config)))(batch)
    assert out.counts.shape == (3, _WIDE.shape[0] - 1)
    assert not bool(jnp.any(out.overflow))
    loss = lambda p: _loss(_wide(p, wide_config).counts)  # noqa: E731
    grads = jax.jit(jax.vmap(jax.grad(loss)))(batch)
    single = jax.jit(lambda p: _wide(p, wide_config).counts)
    single_grad = jax.jit(jax.grad(loss))
    for b in range(3):
        np.testing.assert_allclose(out.counts[b], single(batch[b]), rtol=1e-12)
        np.testing.assert_allclose(
            grads[b], single_grad(batch[b]), rtol=1e-10, atol=1e-12
        )


def test_vmap_with_per_catalogue_bounds(wide_config):
    """bounds=None infers each catalogue's own Morton box inside the vmap."""
    batch = jnp.stack([_catalog(seed=s) for s in range(2)])
    out = jax.vmap(lambda p: _wide(p, wide_config, bounds=None))(batch)
    assert not bool(jnp.any(out.overflow))
    ref = jnp.stack([_brute(p, _WIDE) for p in batch])
    assert _max_rel_err(out.counts, ref) < _TOL


@pytest.mark.parametrize(
    ("field", "flag"),
    [
        ("leaf_capacity", "leaf_overflow"),
        ("max_pair_queue", "queue_overflow"),
        ("far_cap", "far_overflow"),
        ("near_cap", "near_overflow"),
    ],
)
def test_too_small_capacity_raises_its_overflow_flag(wide_config, field, flag):
    small = dataclasses.replace(wide_config, **{field: 4})
    res = jax.jit(lambda p: _wide(p, small))(_catalog())
    assert bool(getattr(res, flag))
    assert bool(res.overflow)
    # the planned config itself is clean
    assert not bool(_wide(_catalog(), wide_config).overflow)


def test_planner_covers_a_batch_and_reports_needs(wide_config):
    for s in range(4):
        res = _wide(_catalog(seed=s), wide_config)
        assert not bool(res.overflow)
        assert int(res.num_leaves) <= wide_config.leaf_capacity
        assert int(res.far_count) <= wide_config.far_cap
        assert int(res.near_count) <= wide_config.near_cap
        assert int(res.peak_wavefront) <= wide_config.max_pair_queue


def test_traced_inference_demo_moves_toward_truth():
    pytest.importorskip("optax")
    from yggdrax.applications.corrfunc.inference_demo import (
        make_cluster_model,
        recover_parameter_traced,
    )

    model = make_cluster_model(jax.random.PRNGKey(0), n=300, num_centers=6)
    res = recover_parameter_traced(
        model,
        make_log_edges(0.01, 0.5, 8),
        a_true=0.5,
        a_init=0.1,
        leaf_size=_LEAF,
        num_steps=25,
        probe_a=(0.0, 0.5, 0.99),
    )
    assert len(res.a_history) == 26
    assert abs(res.a_final - 0.5) < abs(0.1 - 0.5)
    assert res.loss_history[-1] < res.loss_history[0]
