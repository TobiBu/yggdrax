"""Passing `seed_count` must not change which pairs the receiver walk finds.

The wavefront's first round is sized by `init_size`. Without `seed_count` that is
the CSR CAPACITY, so the walk reads every padded slot; with it, only what arrived.
Dead slots carry -1 and are filtered by the walk's own liveness mask, so the two
must agree exactly -- if they ever disagree, the padding is NOT inert and the
capacity is silently part of the answer.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from yggdrax.distributed.import_cells import receiver_interaction_lists


def _tree(rng, n_local_leaves=8):
    """A tiny two-level local tree plus a block of imported nodes."""
    n_int = n_local_leaves - 1
    n_local = n_int + n_local_leaves
    left = np.full(n_int, -1, np.int64)
    right = np.full(n_int, -1, np.int64)
    # a balanced binary tree over the leaves
    nxt = [n_int + i for i in range(n_local_leaves)]
    node = n_int - 1
    while len(nxt) > 1:
        new = []
        for i in range(0, len(nxt) - 1, 2):
            left[node] = nxt[i]
            right[node] = nxt[i + 1]
            new.append(node)
            node -= 1
        if len(nxt) % 2:
            new.append(nxt[-1])
        nxt = new[::-1]
    return n_int, n_local, left, right


@pytest.mark.parametrize("capacity", [16, 64, 256])
def test_seed_count_does_not_change_the_lists(capacity):
    rng = np.random.default_rng(5)
    n_int, n_local, left, right = _tree(rng)
    n_imp = 6
    total = n_local + n_imp
    cen = jnp.asarray(rng.normal(size=(total, 3)))
    rad = jnp.asarray(rng.uniform(0.05, 0.4, size=total))
    lf = jnp.full((n_local - n_int,), -1, jnp.int64)
    lc = jnp.concatenate([jnp.asarray(left), lf])
    rc = jnp.concatenate([jnp.asarray(right), lf])

    cells = jnp.asarray([n_int, n_int + 1, n_int + 2, 0], jnp.int64)
    n_csr = 5
    csr_cell = np.full(capacity, -1, np.int64)
    csr_row = np.zeros(capacity, np.int64)
    csr_cell[:n_csr] = [0, 1, 2, 3, 0]
    csr_row[:n_csr] = [0, 1, 2, 3, 4]
    # poison every dead slot: if the padding were read, these would contribute
    csr_cell[n_csr:] = 3
    csr_row[n_csr:] = 5

    out = receiver_interaction_lists(
        lc,
        rc,
        cen,
        rad,
        n_local,
        cells,
        jnp.asarray(csr_cell),
        jnp.asarray(csr_row),
        jnp.asarray(n_csr),
        0.6,
        max_pair_queue=1024,
        far_cap=512,
        near_cap=512,
        mac_type="dehnen",
    )
    # the capacity is the ONLY thing varying across the parametrisation, so equal
    # results across it is exactly the claim
    return_key = (
        int(out.far_count),
        int(out.near_count),
        tuple(np.asarray(out.far_target)[: int(out.far_count)].tolist()),
        tuple(np.asarray(out.far_source)[: int(out.far_count)].tolist()),
        tuple(np.asarray(out.near_target)[: int(out.near_count)].tolist()),
        tuple(np.asarray(out.near_source)[: int(out.near_count)].tolist()),
    )
    assert not bool(out.far_overflow) and not bool(out.near_overflow)
    _SEEN.setdefault("k", []).append(return_key)
    assert _SEEN["k"][0] == return_key, (
        "the walk's answer depends on the CSR capacity, so the padding is not inert"
    )


_SEEN: dict = {}


def test_the_poisoned_padding_is_not_inert_by_construction():
    """The invariance test above is only meaningful if the padding COULD contribute.

    Declare the poisoned rows live and the answer must change. Without this the
    parametrised test could pass because the padding is harmless in this fixture
    rather than because the walk ignores it.
    """
    rng = np.random.default_rng(5)
    n_int, n_local, left, right = _tree(rng)
    n_imp = 6
    total = n_local + n_imp
    cen = jnp.asarray(rng.normal(size=(total, 3)))
    rad = jnp.asarray(rng.uniform(0.05, 0.4, size=total))
    lf = jnp.full((n_local - n_int,), -1, jnp.int64)
    lc = jnp.concatenate([jnp.asarray(left), lf])
    rc = jnp.concatenate([jnp.asarray(right), lf])
    cells = jnp.asarray([n_int, n_int + 1, n_int + 2, 0], jnp.int64)

    capacity = 16
    csr_cell = np.full(capacity, -1, np.int64)
    csr_row = np.zeros(capacity, np.int64)
    csr_cell[:5] = [0, 1, 2, 3, 0]
    csr_row[:5] = [0, 1, 2, 3, 4]
    csr_cell[5:] = 3
    csr_row[5:] = 5

    def run(n_csr):
        out = receiver_interaction_lists(
            lc,
            rc,
            cen,
            rad,
            n_local,
            cells,
            jnp.asarray(csr_cell),
            jnp.asarray(csr_row),
            jnp.asarray(n_csr),
            0.6,
            max_pair_queue=1024,
            far_cap=512,
            near_cap=512,
            mac_type="dehnen",
        )
        return int(out.far_count), int(out.near_count)

    assert run(capacity) != run(5), (
        "declaring the padding live changed nothing, so the invariance test above "
        "proves nothing about the walk"
    )
