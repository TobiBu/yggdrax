"""`build_send_buffers` must preserve every (cell, node) pairing it is given.

The sender's far list is admissible by its own geometry (see
`test_export_walk_admissibility`), yet the receiver rejects ~17 % of the pairs it
gets. Something between emission and re-test changes which node is paired with
which cell. This checks the first stage: dedup, grouping and the double rebase.

The invariant is exact and needs no mesh -- for every live input pair
`(global_cell, node)`, the output must contain a CSR entry whose rebased cell is
`global_cell % max_cells`, in the block of destination `global_cell // max_cells`,
whose `csr_row` points at a `node_rows` slot holding exactly `node`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.distributed.export import build_send_buffers

MC = 32          # max_cells
NDEV = 3         # more than two, so the destination blocks really are distinct
NUM_NODES = 200


def _pairs(rng, n):
    """Random (destination, cell slot, node) triples with deliberate duplicates."""
    dev = rng.integers(0, NDEV, size=n)
    slot = rng.integers(0, MC, size=n)
    node = rng.integers(0, NUM_NODES, size=n)
    # force repeats of the same node to different cells AND the same cell to
    # different nodes, which is what dedup-per-destination has to survive
    node[: n // 4] = node[0]
    slot[n // 4 : n // 2] = slot[0]
    dev[n // 4 : n // 2] = dev[0]
    return dev, slot, node


@pytest.mark.parametrize("n", [1, 7, 64, 501])
def test_every_input_pair_survives_with_its_own_node(n):
    rng = np.random.default_rng(100 + n)
    dev, slot, node = _pairs(rng, n)
    cap = 2048
    cell_global = (dev * MC + slot).astype(np.int64)

    cells = np.full(cap, -1, np.int64)
    nodes = np.zeros(cap, np.int64)
    cells[:n] = cell_global
    nodes[:n] = node
    # poison the padding: if it were read, these pairs would appear in the output
    cells[n:] = 0
    nodes[n:] = NUM_NODES - 1

    sb = build_send_buffers(
        jnp.asarray(cells),
        jnp.asarray(nodes),
        jnp.asarray(n),
        ndev=NDEV,
        max_cells=MC,
        num_nodes=NUM_NODES,
        node_capacity=1024,
        csr_capacity=4096,
    )
    assert not bool(sb.node_overflow) and not bool(sb.csr_overflow)

    node_rows = np.asarray(sb.node_rows)
    node_sizes = np.asarray(sb.node_sizes)
    csr_cell = np.asarray(sb.csr_cell)
    csr_row = np.asarray(sb.csr_row)
    csr_sizes = np.asarray(sb.csr_sizes)
    node_off = np.concatenate([[0], np.cumsum(node_sizes)])
    csr_off = np.concatenate([[0], np.cumsum(csr_sizes)])

    # what the receivers would actually reconstruct, per destination
    got = set()
    for d in range(NDEV):
        for j in range(csr_off[d], csr_off[d + 1]):
            c = int(csr_cell[j])
            r = int(csr_row[j])
            assert 0 <= c < MC, f"rebased cell {c} out of range for destination {d}"
            assert 0 <= r < node_sizes[d], (
                f"csr_row {r} outside destination {d}'s block of {node_sizes[d]}"
            )
            got.add((d, c, int(node_rows[node_off[d] + r])))

    want = {(int(a), int(b), int(c)) for a, b, c in zip(dev, slot, node)}
    assert got == want, (
        f"pairing changed: {len(want - got)} lost, {len(got - want)} invented"
    )


def test_the_poisoned_padding_would_show_up_if_read():
    """Non-vacuity: the padding pairs are distinguishable from the live ones."""
    rng = np.random.default_rng(7)
    n = 32
    dev, slot, node = _pairs(rng, n)
    cap = 256
    cells = np.full(cap, -1, np.int64)
    nodes = np.zeros(cap, np.int64)
    cells[:n] = (dev * MC + slot).astype(np.int64)
    nodes[:n] = node
    cells[n:] = 0
    nodes[n:] = NUM_NODES - 1

    def run(count):
        sb = build_send_buffers(
            jnp.asarray(cells),
            jnp.asarray(nodes),
            jnp.asarray(count),
            ndev=NDEV,
            max_cells=MC,
            num_nodes=NUM_NODES,
            node_capacity=1024,
            csr_capacity=4096,
        )
        return int(jnp.sum(sb.csr_sizes))

    assert run(cap) != run(n), "the padding is inert in this fixture, so the test above is weak"
