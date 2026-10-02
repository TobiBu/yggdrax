"""The cross-domain exchange: two ragged rounds, and the re-offset that is easy to skip.

Run on forced host CPU devices:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_import_cells.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.import_cells import exchange_export_list, rebase_csr
from yggdrax.distributed.sharding import AXIS_NAME


def test_rebase_csr_shifts_each_senders_block():
    """The documented trap, on hand-made sizes so the arithmetic is visible.

    Three senders send 2, 3 and 1 payload rows and 2, 2 and 1 CSR entries. Each
    numbered its rows from zero, so the second sender's row 0 is really row 2 of the
    receive buffer and the third's is row 5. Skipping this reads a plausible but
    wrong node for every entry after the first sender.
    """
    csr_row = jnp.asarray([0, 1, 0, 2, 0, -1, -1])
    csr_sizes = jnp.asarray([2, 2, 1])
    payload_sizes = jnp.asarray([2, 3, 1])
    out = np.asarray(rebase_csr(csr_row, csr_sizes, payload_sizes, ndev=3))
    assert out[:5].tolist() == [0, 1, 2, 4, 5]
    assert out[5:].tolist() == [-1, -1], "padding must stay padding"


def test_rebase_csr_is_the_identity_for_one_sender():
    """The control: with a single block there is nothing to shift."""
    row = jnp.asarray([0, 3, 1, -1])
    out = np.asarray(rebase_csr(row, jnp.asarray([3]), jnp.asarray([4]), ndev=1))
    assert out.tolist() == [0, 3, 1, -1]


pytestmark_mesh = pytest.mark.skipif(
    device_count() < 2, reason="the exchange needs >= 2 devices"
)


@pytestmark_mesh
@pytest.mark.parametrize("ndev", [2, 4])
def test_the_exchange_round_trips_every_reference_to_the_right_payload(ndev):
    """Every (my cell, imported node) reference must resolve to what the sender meant.

    The payload rows carry a tag unique to (sender, node), so resolving
    ``payload[csr_row]`` proves both that the right bytes arrived and that the
    re-offsetting put each sender's entries on its own block. A missing re-offset
    passes every shape and size check and fails exactly here.
    """
    if device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    mesh = make_mesh(ndev)
    rng = np.random.default_rng(0)
    cap_n, cap_c = 64, 128

    # device s sends `rows[s, d]` payload rows and `refs[s, d]` references to d
    rows = rng.integers(1, 6, size=(ndev, ndev))
    refs = rng.integers(1, 8, size=(ndev, ndev))
    np.fill_diagonal(rows, 0)
    np.fill_diagonal(refs, 0)

    TAG = 1000

    def build(s):
        """Sender s's grouped buffers, plus the truth about what it intends."""
        pay, cell, row, truth = [], [], [], {}
        for d in range(ndev):
            base = len(pay)
            for j in range(rows[s, d]):
                pay.append(TAG * (s + 1) + j)  # unique to (sender, its own row j)
            for _ in range(refs[s, d]):
                j = int(rng.integers(0, max(rows[s, d], 1)))
                c = int(rng.integers(0, 16))
                cell.append(c)
                row.append(j)
                if rows[s, d]:
                    truth.setdefault(d, set()).add((c, TAG * (s + 1) + j))
            del base
        pad = lambda a, n: np.asarray(a + [-1] * (n - len(a)), np.int32)
        return (
            pad(pay, cap_n),
            rows[s].astype(np.int32),
            pad(cell, cap_c),
            pad(row, cap_c),
            refs[s].astype(np.int32),
            truth,
        )

    built = [build(s) for s in range(ndev)]
    stack = lambda i: jnp.asarray(np.stack([b[i] for b in built]))

    @jax.jit
    def run(pay, ns, cc, cr, cs):
        def body(pay, ns, cc, cr, cs):
            f = lambda x: x[0]
            got = exchange_export_list(
                f(pay),
                f(ns),
                f(cc),
                f(cr),
                f(cs),
                payload_capacity=cap_n * ndev,
                csr_capacity=cap_c * ndev,
                ndev=ndev,
            )
            return (
                got.payload[None],
                got.csr_cell[None],
                got.csr_row[None],
                got.num_csr[None],
            )

        return shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS_NAME),) * 5,
            out_specs=(P(AXIS_NAME),) * 4,
        )(pay, ns, cc, cr, cs)

    payload, cell, row, ncsr = run(stack(0), stack(1), stack(2), stack(3), stack(4))
    payload, cell, row, ncsr = map(np.asarray, (payload, cell, row, ncsr))

    for d in range(ndev):
        n = int(ncsr[d])
        got = {(int(cell[d, j]), int(payload[d, row[d, j]])) for j in range(n)}
        want = set()
        for s in range(ndev):
            want |= built[s][5].get(d, set())
        assert got == want, f"device {d}: {len(got)} refs against {len(want)} intended"
        assert n == int(refs[:, d].sum())
