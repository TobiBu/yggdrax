"""Exactly one device emits each cross-domain pair.

A mutual FMM applies +f/-f to each unordered pair once. The single-device walk gets
that from a canonical `a < b` within one tree; across devices both sides of a
boundary discover the same geometric pair and must agree on which one emits it.

Both ways of getting this wrong are SILENT in the usual momentum check: emit twice
and the pair is double-counted, emit never and it is dropped, and in both cases
+f/-f still cancel within whatever each device did, so every per-device momentum sum
looks perfect. Only a global force comparison would notice. Hence these tests check
the combinatorics directly rather than trusting a downstream assertion.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from yggdrax.distributed.cross_walk import cross_pair_is_owned, cross_pair_owner


def _pairs(n_dev: int, n_node: int):
    """Every cross-device (endpoint, endpoint) pair, as global keys."""
    for (pa, ia), (pb, ib) in itertools.combinations(
        itertools.product(range(n_dev), range(n_node)), 2
    ):
        if pa != pb:  # cross-domain only; intra-domain is the self-walk's job
            yield (pa, ia), (pb, ib)


def test_each_cross_pair_is_emitted_exactly_once():
    """The property the whole design rests on."""
    n_dev, n_node = 4, 7
    for (pa, ia), (pb, ib) in _pairs(n_dev, n_node):
        # device pa walks its own node ia against pb's imported node ib
        a_emits = bool(
            cross_pair_is_owned(
                jnp.asarray(pa), jnp.asarray(ia), jnp.asarray(pb), jnp.asarray(ib)
            )
        )
        # and pb walks ib against pa's imported ia -- the same pair, other way round
        b_emits = bool(
            cross_pair_is_owned(
                jnp.asarray(pb), jnp.asarray(ib), jnp.asarray(pa), jnp.asarray(ia)
            )
        )
        assert (
            a_emits + b_emits == 1
        ), f"pair ({pa},{ia})-({pb},{ib}) emitted {a_emits + b_emits} times"


def test_owner_is_symmetric_under_swapping_the_endpoints():
    """Both devices must compute the SAME owner, seeing the pair in opposite orders."""
    n_dev, n_node = 5, 6
    for (pa, ia), (pb, ib) in _pairs(n_dev, n_node):
        fwd = int(
            cross_pair_owner(
                jnp.asarray(pa), jnp.asarray(ia), jnp.asarray(pb), jnp.asarray(ib)
            )
        )
        rev = int(
            cross_pair_owner(
                jnp.asarray(pb), jnp.asarray(ib), jnp.asarray(pa), jnp.asarray(ia)
            )
        )
        assert fwd == rev, f"asymmetric for ({pa},{ia})-({pb},{ib}): {fwd} vs {rev}"


def test_owner_is_always_one_of_the_two_endpoints():
    """A third device must never be handed the pair -- it holds neither endpoint."""
    n_dev, n_node = 4, 5
    for (pa, ia), (pb, ib) in _pairs(n_dev, n_node):
        own = int(
            cross_pair_owner(
                jnp.asarray(pa), jnp.asarray(ia), jnp.asarray(pb), jnp.asarray(ib)
            )
        )
        assert own in (pa, pb)


def test_work_is_split_between_the_two_domains_not_dumped_on_one():
    """Ordering alone would give a domain-pair's work entirely to one device.

    That is why the rule carries a parity term. With an SFC partition device 0 would
    otherwise own a boundary with each of its neighbours while the last device owned
    almost none, so this pins the balance rather than leaving it to chance.
    """
    n_node = 64
    for pa, pb in ((0, 1), (2, 3), (0, 3), (1, 2)):
        owners = [
            int(
                cross_pair_owner(
                    jnp.asarray(pa), jnp.asarray(ia), jnp.asarray(pb), jnp.asarray(ib)
                )
            )
            for ia in range(n_node)
            for ib in range(n_node)
        ]
        share_a = owners.count(pa) / len(owners)
        assert 0.4 < share_a < 0.6, (
            f"domains {pa}/{pb} split {share_a:.2f}/{1 - share_a:.2f}, "
            "too lopsided to be load-balanced"
        )


def test_vectorises_over_arrays():
    """The walk applies this to whole wavefronts, not scalars."""
    rng = np.random.default_rng(0)
    ia = jnp.asarray(rng.integers(0, 100, 256))
    ib = jnp.asarray(rng.integers(0, 100, 256))
    owned = cross_pair_is_owned(jnp.asarray(1), ia, jnp.asarray(2), ib)
    assert owned.shape == (256,)
    # the complement, seen from the other device, must be exactly the negation
    other = cross_pair_is_owned(jnp.asarray(2), ib, jnp.asarray(1), ia)
    assert bool(jnp.all(owned ^ other)), "some pair emitted twice or not at all"
