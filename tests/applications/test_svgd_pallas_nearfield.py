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
