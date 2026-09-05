"""The fused Pallas Stein near field, and the pure-JAX twin it is checked against.

Everything here runs under ``interpret=True``, which executes Pallas with CPU
semantics and no Triton lowering, so the suite is meaningful on a CI runner with
no GPU. The GPU lowering is exercised by the benches; what these tests pin is
the *contract* -- that the kernel, the twin and a naive loop agree, and that the
three things that are easy to get wrong (capacity padding, leaf widths that are
not a power of two, an empty source list) are handled.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import DualTreeTraversalConfig
from yggdrax.applications.svgd.pallas_nearfield import (
    nearfield_stein,
    nearfield_stein_jax,
    nearfield_stein_pallas,
    pallas_stein_nearfield_supported,
)
from yggdrax.applications.svgd.sampler import (
    build_svgd_topology,
    svgd_phi_from_topology,
)

_CFG = DualTreeTraversalConfig(
    max_pair_queue=1 << 18,
    process_block=32,
    max_interactions_per_node=1 << 14,
    max_neighbors_per_leaf=1 << 14,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _naive(leaf_x, leaf_s, leaf_mask, src_offsets, src_leaf, h, include_self=True):
    """Loop-in-Python Stein near field: the reference of last resort."""
    num_leaves, width, dim = leaf_x.shape
    out = np.zeros((num_leaves, width, dim))
    for target in range(num_leaves):
        sources = list(src_leaf[src_offsets[target] : src_offsets[target + 1]])
        if include_self:
            sources = sources + [target]
        for i in range(width):
            if not leaf_mask[target, i]:
                continue
            for source in sources:
                for j in range(width):
                    if not leaf_mask[source, j]:
                        continue
                    diff = leaf_x[target, i] - leaf_x[source, j]
                    kern = np.exp(-np.dot(diff, diff) / (2 * h * h))
                    out[target, i] += kern * leaf_s[source, j] + kern * diff / (h * h)
    return out


def _random_case(num_leaves=6, width=8, dim=2, seed=0, density=0.6):
    """A synthetic leaf-major partition with a SYMMETRIC near-pair list."""
    rng = np.random.default_rng(seed)
    leaf_x = rng.normal(size=(num_leaves, width, dim))
    leaf_s = rng.normal(size=(num_leaves, width, dim))
    mask = rng.random((num_leaves, width)) > 0.25
    mask[:, 0] = True  # every leaf keeps at least one particle
    pairs = set()
    for a in range(num_leaves):
        for b in range(a + 1, num_leaves):
            if rng.random() < density:
                pairs.update({(a, b), (b, a)})
    ordered = sorted(pairs)
    targets = np.array([t for t, _ in ordered], dtype=np.int32)
    sources = np.array([s for _, s in ordered], dtype=np.int32)
    counts = np.bincount(targets, minlength=num_leaves)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return leaf_x, leaf_s, mask, offsets, sources


def _rel(got, want):
    return float(np.abs(np.asarray(got) - np.asarray(want)).max()) / max(
        float(np.abs(np.asarray(want)).max()), 1e-30
    )


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("include_self", [True, False])
def test_twin_and_kernel_both_match_a_naive_loop(dim, include_self):
    """The three implementations agree to round-off, in every dimension."""
    case = _random_case(dim=dim, seed=dim)
    h = 0.7
    want = _naive(*case, h, include_self=include_self)
    args = tuple(jnp.asarray(a) for a in case) + (h,)
    twin = nearfield_stein_jax(*args, include_self=include_self)
    kernel = nearfield_stein_pallas(*args, include_self=include_self, interpret=True)
    assert _rel(twin, want) < 1e-13
    assert _rel(kernel, want) < 1e-13


@pytest.mark.parametrize(
    "num_leaves,width,dim", [(5, 5, 2), (7, 12, 3), (4, 1, 2), (3, 33, 1)]
)
def test_leaf_widths_that_are_not_a_power_of_two(num_leaves, width, dim):
    """Triton needs power-of-two tiles; the padding must not leak into the result."""
    case = _random_case(num_leaves=num_leaves, width=width, dim=dim, seed=width)
    want = _naive(*case, 0.9)
    got = nearfield_stein_pallas(*(jnp.asarray(a) for a in case), 0.9, interpret=True)
    assert _rel(got, want) < 1e-13


def test_capacity_padding_past_the_offsets_is_never_read():
    """A padded source list must give the same answer as an exact one.

    ``assemble_svgd_topology(capacity=...)`` pads the directed pair list so the
    jitted update compiles once instead of once per rebuild. The CSR bounds are
    what stop the kernel reading the padding, so this is the property that lets
    the two features coexist.
    """
    leaf_x, leaf_s, mask, offsets, sources = _random_case(seed=11)
    padded = np.concatenate([sources, np.full((7,), 3, dtype=np.int32)])
    args = (
        jnp.asarray(leaf_x),
        jnp.asarray(leaf_s),
        jnp.asarray(mask),
        jnp.asarray(offsets),
    )
    exact = nearfield_stein_pallas(*args, jnp.asarray(sources), 0.7, interpret=True)
    with_pad = nearfield_stein_pallas(*args, jnp.asarray(padded), 0.7, interpret=True)
    assert _rel(with_pad, exact) < 1e-14
    assert _rel(nearfield_stein_jax(*args, jnp.asarray(padded), 0.7), exact) < 1e-14


def test_an_empty_source_list_leaves_only_the_self_term():
    """A leaf with no near neighbours still has its own block to sum."""
    leaf_x, leaf_s, mask, _, _ = _random_case(num_leaves=4, seed=99)
    offsets = jnp.zeros((5,), dtype=jnp.int32)
    sources = jnp.zeros((0,), dtype=jnp.int32)
    want = _naive(
        leaf_x, leaf_s, mask, np.zeros(5, np.int32), np.zeros(0, np.int32), 0.6
    )
    got = nearfield_stein_pallas(
        jnp.asarray(leaf_x),
        jnp.asarray(leaf_s),
        jnp.asarray(mask),
        offsets,
        sources,
        0.6,
        interpret=True,
    )
    assert _rel(got, want) < 1e-14


def test_the_twin_is_chunk_invariant():
    """Chunking is a memory knob, not a numerical one."""
    case = tuple(jnp.asarray(a) for a in _random_case(seed=5))
    whole = nearfield_stein_jax(*case, 0.7)
    for chunk in (1, 3, 7):
        assert _rel(nearfield_stein_jax(*case, 0.7, chunk_pairs=chunk), whole) == 0.0


@pytest.mark.parametrize("tile", [1, 2, 4, 8])
def test_tile_widths_do_not_change_the_answer(tile):
    """Every (target subtile, source tile) split computes the same sum."""
    case = tuple(jnp.asarray(a) for a in _random_case(seed=7))
    want = nearfield_stein_jax(*case, 0.7)
    got = nearfield_stein_pallas(
        *case, 0.7, target_subtile=tile, source_tile=tile, interpret=True
    )
    assert _rel(got, want) < 1e-13


def test_kernel_reproduces_the_sampler_near_field_end_to_end():
    """On a real partition with no far field, the kernel *is* the Stein update.

    At ``theta = 0`` the walk accepts nothing as far, so
    :func:`svgd_phi_from_topology` reduces to the near field plus the ``1/n``
    normalisation -- which makes it a direct check of the kernel against the
    accumulation it is meant to replace, on a partition the traversal actually
    produced rather than a synthetic one.
    """
    key = jax.random.PRNGKey(0)
    particles = jax.random.normal(key, (192, 2), dtype=jnp.float64)
    scores = -particles
    topo = build_svgd_topology(
        particles, theta=0.0, leaf_size=8, backend="leaf_kdtree", traversal_config=_CFG
    )
    num_leaves = int(topo.leaf_slots.shape[0])

    # CSR offsets over the directed pair list, which np.repeat already emits
    # ascending in target row.
    targets = np.asarray(topo.near_dir_target)
    live = int((np.asarray(topo.near_dir_live) > 0).sum())
    counts = np.bincount(targets[:live], minlength=num_leaves)
    offsets = jnp.asarray(np.concatenate([[0], np.cumsum(counts)]).astype(np.int32))
    sources = jnp.asarray(np.asarray(topo.near_dir_source)[:live].astype(np.int32))

    pos, sco = particles[topo.order], scores[topo.order]
    acc = nearfield_stein_pallas(
        pos[topo.leaf_slots],
        sco[topo.leaf_slots],
        topo.leaf_mask > 0,
        offsets,
        sources,
        0.5,
        interpret=True,
    )
    near = jnp.zeros_like(pos).at[topo.leaf_slots].add(acc * topo.leaf_mask[..., None])
    got = jnp.zeros_like(near).at[topo.order].set(near / particles.shape[0])

    want = svgd_phi_from_topology(particles, scores, 0.5, topo)
    assert _rel(got, want) < 1e-12


def test_the_capability_probe_answers_without_raising():
    """It is called to pick a lane, so it must never raise, GPU or not."""
    assert isinstance(pallas_stein_nearfield_supported(), bool)


# --------------------------------------------------------------------------
# WP1: the hand-written VJP.
#
# A custom derivative rule that is not checked against autodiff of its twin does
# not merit being in the package -- it is the only thing standing between a
# hand-derived gradient and a silently wrong one. These are that check.
# --------------------------------------------------------------------------


def _assert_vjp_matches(case, h, include_self, seed=1234, tol=1e-11):
    """Value *and* every cotangent, against autodiff through the twin."""
    leaf_x, leaf_s, mask, offsets, sources = case
    rng = np.random.default_rng(seed)
    cotangent = jnp.asarray(rng.normal(size=leaf_x.shape))
    static = (jnp.asarray(mask), jnp.asarray(offsets), jnp.asarray(sources))
    x, s, band = jnp.asarray(leaf_x), jnp.asarray(leaf_s), jnp.asarray(h)

    def twin(x_, s_, h_):
        return nearfield_stein_jax(x_, s_, *static, h_, include_self=include_self)

    def fused(x_, s_, h_):
        return nearfield_stein(
            x_, s_, *static, h_, include_self=include_self, interpret=True
        )

    want, vjp_want = jax.vjp(twin, x, s, band)
    got, vjp_got = jax.vjp(fused, x, s, band)
    np.testing.assert_allclose(got, want, rtol=tol, atol=tol)
    for name, a, b in zip(
        ("d/dx", "d/ds", "d/dh"), vjp_got(cotangent), vjp_want(cotangent)
    ):
        np.testing.assert_allclose(a, b, rtol=tol, atol=tol, err_msg=name)


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("include_self", [True, False])
def test_the_custom_vjp_matches_autodiff_of_the_twin(dim, include_self):
    """Positions, scores and the bandwidth, in every dimension."""
    _assert_vjp_matches(_random_case(dim=dim, seed=dim + 40), 0.73, include_self)


@pytest.mark.parametrize(
    "num_leaves,width,dim", [(5, 5, 2), (7, 12, 3), (4, 1, 2), (3, 33, 1)]
)
def test_the_custom_vjp_survives_leaf_padding(num_leaves, width, dim):
    """The gradient must not pick up a contribution from a padded lane."""
    case = _random_case(num_leaves=num_leaves, width=width, dim=dim, seed=width + 7)
    _assert_vjp_matches(case, 0.85, True)


def test_the_custom_vjp_is_correct_under_jit():
    """The rule has to survive being staged out, which is how it will be used."""
    leaf_x, leaf_s, mask, offsets, sources = _random_case(seed=3)
    static = (jnp.asarray(mask), jnp.asarray(offsets), jnp.asarray(sources))
    weights = jnp.asarray(np.random.default_rng(0).normal(size=leaf_x.shape))

    def loss(fn):
        def inner(x_, s_, h_):
            return jnp.sum(weights * fn(x_, s_, *static, h_))

        return inner

    args = (jnp.asarray(leaf_x), jnp.asarray(leaf_s), jnp.asarray(0.73))
    want = jax.grad(loss(nearfield_stein_jax), (0, 1, 2))(*args)
    fused = loss(lambda *a: nearfield_stein(*a, interpret=True))
    for label, grad_fn in (
        ("eager", jax.grad(fused, (0, 1, 2))),
        ("jit", jax.jit(jax.grad(fused, (0, 1, 2)))),
    ):
        got = grad_fn(*args)
        for name, a, b in zip(("d/dx", "d/ds", "d/dh"), got, want):
            np.testing.assert_allclose(
                a, b, rtol=1e-11, atol=1e-11, err_msg=f"{label} {name}"
            )


def test_the_partition_takes_no_gradient():
    """The mask and the CSR are the discrete partition, not differentiable.

    They are ``custom_vjp`` primals all the same, so the rule has to return a
    ``float0`` cotangent for each; getting that wrong is a type error at trace
    time, which is what this pins.
    """
    leaf_x, leaf_s, mask, offsets, sources = _random_case(seed=13)

    def loss(mask_, offsets_, sources_):
        return jnp.sum(
            nearfield_stein(
                jnp.asarray(leaf_x),
                jnp.asarray(leaf_s),
                mask_,
                offsets_,
                sources_,
                0.7,
                interpret=True,
            )
        )

    grads = jax.grad(loss, (0, 1, 2), allow_int=True)(
        jnp.asarray(mask), jnp.asarray(offsets), jnp.asarray(sources)
    )
    for grad in grads:
        assert grad.dtype == jax.dtypes.float0


def test_the_backend_selector_is_explicit():
    """``"jax"`` must be the twin exactly, and a typo must not silently work."""
    case = tuple(jnp.asarray(a) for a in _random_case(seed=17))
    forced = nearfield_stein(*case, 0.7, backend="jax")
    assert _rel(forced, nearfield_stein_jax(*case, 0.7)) == 0.0
    with pytest.raises(ValueError, match="backend must be"):
        nearfield_stein(*case, 0.7, backend="cuda")


def test_the_sampler_accumulations_agree_in_value_and_gradient():
    """``accumulate="interpret"`` must match ``"scatter"`` through the far field.

    ``theta = 0.5`` accepts far pairs, so this covers the monopole step too --
    the kernel replaces the near field only, and this is what pins that it has
    not disturbed anything downstream of it.
    """
    key = jax.random.PRNGKey(1)
    particles = jax.random.normal(key, (192, 2), dtype=jnp.float64)
    scores = -particles
    topo = build_svgd_topology(
        particles, theta=0.5, leaf_size=8, backend="leaf_kdtree", traversal_config=_CFG
    )
    assert int(topo.num_far_pairs) > 0, "this test is pointless with no far field"

    def loss(x, band, how):
        return jnp.sum(
            svgd_phi_from_topology(x, scores, band, topo, accumulate=how) ** 2
        )

    band = jnp.asarray(0.5)
    for how in ("segment", "interpret"):
        np.testing.assert_allclose(
            svgd_phi_from_topology(particles, scores, band, topo, accumulate=how),
            svgd_phi_from_topology(particles, scores, band, topo, accumulate="scatter"),
            rtol=1e-11,
            atol=1e-13,
        )
    want = jax.grad(loss, (0, 1))(particles, band, "scatter")
    got = jax.grad(loss, (0, 1))(particles, band, "interpret")
    np.testing.assert_allclose(got[0], want[0], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-10, atol=1e-13)


def test_an_unknown_accumulation_is_rejected():
    """The name list is a contract; a typo must not fall through to a default."""
    key = jax.random.PRNGKey(2)
    particles = jax.random.normal(key, (64, 2), dtype=jnp.float64)
    topo = build_svgd_topology(
        particles, theta=0.0, leaf_size=8, backend="leaf_kdtree", traversal_config=_CFG
    )
    with pytest.raises(ValueError, match="accumulate must be"):
        svgd_phi_from_topology(particles, -particles, 0.5, topo, accumulate="fused")


def test_the_csr_offsets_bound_the_live_directed_pairs():
    """``near_dir_offsets`` is what stops the kernel reading capacity padding."""
    key = jax.random.PRNGKey(3)
    particles = jax.random.normal(key, (300, 2), dtype=jnp.float64)
    exact = build_svgd_topology(
        particles,
        theta=0.4,
        leaf_size=8,
        backend="leaf_kdtree",
        traversal_config=_CFG,
        capacity="exact",
    )
    padded = build_svgd_topology(
        particles,
        theta=0.4,
        leaf_size=8,
        backend="leaf_kdtree",
        traversal_config=_CFG,
        capacity="pow2",
    )
    live = int((np.asarray(exact.near_dir_live) > 0).sum())
    assert int(np.asarray(exact.near_dir_offsets)[-1]) == live
    # Padding lengthens the arrays but must not move the offsets.
    assert padded.near_dir_source.shape[0] > exact.near_dir_source.shape[0]
    np.testing.assert_array_equal(
        np.asarray(padded.near_dir_offsets), np.asarray(exact.near_dir_offsets)
    )
    np.testing.assert_allclose(
        svgd_phi_from_topology(
            particles, -particles, 0.5, padded, accumulate="interpret"
        ),
        svgd_phi_from_topology(
            particles, -particles, 0.5, exact, accumulate="interpret"
        ),
        rtol=1e-12,
        atol=1e-14,
    )


def test_the_auto_lane_needs_enough_leaves_to_be_worth_launching():
    """One program per leaf, so a small partition cannot fill the device.

    The threshold is measured (see ``_MIN_LEAVES_FOR_KERNEL``); what this pins is
    that it is *applied*, and applied in one place -- an earlier revision had the
    sampler's ``accumulate="auto"`` bypass it by forcing the kernel, which made
    the whole threshold dead code.
    """
    from yggdrax.applications.svgd.pallas_nearfield import (
        _MIN_LEAVES_FOR_KERNEL,
        prefer_fused_nearfield,
    )

    assert not prefer_fused_nearfield(_MIN_LEAVES_FOR_KERNEL - 1)
    # Above the threshold it defers to the capability probe, which is False on a
    # CPU runner -- so the only machine-independent assertion is agreement.
    assert prefer_fused_nearfield(1 << 20) == pallas_stein_nearfield_supported()
