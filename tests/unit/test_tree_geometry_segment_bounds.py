"""Leaf bounds by segment reduction equal the padded gather's, values and gradients.

``compute_tree_geometry`` reduces each leaf's particles to a box. The padded
gather holds ``(leaves, max_leaf_size, 3)`` arrays, about five times the
particles for a capacity-padded tree at leaf 64; the segment reduction holds
O(particles + leaves). It is taken whenever the padded gather would be larger
than the particle array, so both must agree exactly.

    JAX_PLATFORMS=cpu pytest tests/unit/test_tree_geometry_segment_bounds.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax._geometry_impl import _padded_leaf_bounds, _segment_leaf_bounds


def _leaves(n, cap, seed):
    """Disjoint runs of at most ``cap`` rows with gaps and empty leaves, shuffled."""
    rng = np.random.default_rng(seed)
    starts, counts, at = [], [], 0
    while at < n:
        c = int(rng.integers(0, cap + 1))
        c = min(c, n - at)
        starts.append(at if c else int(rng.integers(0, n)))
        counts.append(c)
        at += c + int(rng.integers(0, 3))  # rows outside every leaf
    starts += [0, n - 1]  # padding leaves: empty, start anywhere
    counts += [0, 0]
    perm = rng.permutation(len(starts))
    return (
        jnp.asarray(np.asarray(starts)[perm], jnp.int32),
        jnp.asarray(np.asarray(counts)[perm], jnp.int32),
    )


@pytest.mark.parametrize("radius", [False, True], ids=["points", "radius"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_segment_bounds_equal_the_padded_gather(seed, radius):
    n, cap = 997, 16
    rng = np.random.default_rng(10 + seed)
    pos = jnp.asarray(rng.normal(size=(n, 3)))
    # exact ties inside a leaf: the derivative split must agree too
    pos = pos.at[5].set(pos[4]).at[40, 1].set(pos[41, 1])
    rad = jnp.asarray(rng.uniform(0.0, 0.1, n)) if radius else None
    starts, counts = _leaves(n, cap, seed)

    def seg(p, r):
        return _segment_leaf_bounds(p, starts, counts, particle_radius=r)

    def pad(p, r):
        return _padded_leaf_bounds(p, starts, counts, cap, particle_radius=r)

    for a, b in zip(seg(pos, rad), pad(pos, rad)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    w = jnp.asarray(rng.normal(size=(int(starts.shape[0]), 3)))

    def loss(f):
        def inner(p, r):
            lo, hi = f(p, r)
            ok = counts[:, None] > 0
            return jnp.sum(jnp.where(ok, w * lo + 2.0 * w * hi, 0.0))

        return inner

    argnums = (0, 1) if radius else (0,)
    gs = jax.grad(loss(seg), argnums=argnums)(pos, rad)
    gp = jax.grad(loss(pad), argnums=argnums)(pos, rad)
    for a, b in zip(gs, gp):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-15)
    gs = jax.jit(jax.grad(loss(seg), argnums=argnums))(pos, rad)
    for a, b in zip(gs, gp):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-15)
