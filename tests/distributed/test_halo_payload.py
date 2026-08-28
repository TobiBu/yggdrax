"""The halo import's optional per-PARTICLE payload must arrive with its particle.

``build_coarse_frontier`` already carries an opaque per-LEAF payload, which is the
right channel for anything a MAC or an M2L needs: it rides an ``all_gather`` of the
frontier, so its cost is one row per remote leaf. A per-particle quantity cannot go
that way. The frontier is gathered densely, so publishing ``leaf_size`` columns on it
would ship every remote particle's value to every device -- ``O(N_total)`` traffic,
which is exactly what the demand-driven import exists to avoid.

So a per-particle payload rides round B of the import instead, alongside the positions
and masses, and is therefore sized by the halo rather than by the system. The
consumer this was added for is the distributed mutual lane's block-step rung: a cross
-domain near pair sits at level ``max(rung_i, rung_j)`` and the ``j`` endpoint lives on
another device.

The property worth pinning is the one a caller cannot arrange for itself: payload row
``h`` describes halo particle ``h``. A shift or a per-block transposition would give
every imported particle a plausible neighbour's value, with no shape error and no
obvious signature -- the same failure the frontier payload test guards against, one
level down.

Two things about the encoding, both of which this test got wrong while being written
and both of which produced a failure that read as a misattached payload row:

* ``_GID_STRIDE`` is **imported**, not copied. A local copy read ``1 << 20`` against
  the real ``1 << 40``, so the decode was off by a factor of a million while the
  payload was perfectly correct.
* the columns are integer MULTIPLES of the gid, not fractional offsets from it. At
  ``1 << 40`` a float64's spacing is ~2.4e-4, so a ``+ k/100`` offset survives only to
  ~1e-4 -- a tolerance failure indistinguishable from a real one. Integer multiples are
  exact (``2 * 2**40`` is far inside float64's exact range) and unambiguous, where
  ``gid + 1`` could be mistaken for a neighbouring particle's row.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_halo_payload.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl
from yggdrax.distributed.let import (
    _GID_STRIDE,
    build_coarse_frontier,
    build_remote_coarse_tree,
    import_near_halo,
)
from yggdrax.distributed.local_tree import _build_local_tree, sanitize_padding
from yggdrax.distributed.partition import global_bounds, sfc_partition
from yggdrax.distributed.sharding import AXIS_NAME
from yggdrax.geometry import compute_tree_geometry
from yggdrax.tree_moments import compute_tree_mass_moments

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="a halo import needs >= 2 devices"
)

_NDEV = min(4, device_count())
_LEAF = 8
_PER_DEV = 24
_CAP = 4 * _PER_DEV


def _run(width):
    """Import a halo carrying ``width`` payload columns; return it with the gids.

    The payload VALUE encodes the particle's own global identity, so a misattached
    row is identifiable rather than merely different -- ``gid`` is what the import
    already returns per halo row, so the two can be checked against each other
    without the test having to model the partition.
    """
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    rng = np.random.default_rng(11)
    n = _PER_DEV * _NDEV
    pts = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, 3)))
    mass = jnp.asarray(rng.uniform(0.5, 2.0, size=(n,)))
    max_req = (_NDEV - 1) * (_CAP // _LEAF)

    def fn(pos, m):
        bounds = global_bounds(pos, axis_name=AXIS_NAME)
        p_, m_, _c, cnt = sfc_partition(
            pos, m, _NDEV, output_capacity=_CAP, bounds=bounds, axis_name=AXIS_NAME
        )
        p_, m_ = sanitize_padding(p_, m_, cnt)
        tree, ps, ms = _build_local_tree(
            p_, m_, bounds, tree_type="radix", leaf_size=_LEAF
        )
        geom = compute_tree_geometry(tree, ps, max_leaf_size=_LEAF)
        mom = compute_tree_mass_moments(tree, ps, ms)
        fr = build_coarse_frontier(
            tree, mom.mass, mom.center_of_mass, positions_sorted=ps, max_leaf_size=_LEAF
        )
        rct = build_remote_coarse_tree(fr, _NDEV, bounds=bounds, axis_name=AXIS_NAME)
        res = dual_tree_walk_cross_impl(
            tree,
            geom,
            rct.tree,
            rct.geometry,
            0.5,
            max_interactions_per_node=256,
            max_neighbors_per_leaf=256,
            max_pair_queue=8192,
        )

        me = jax.lax.axis_index(AXIS_NAME)
        gid = me.astype(ps.dtype) * float(_GID_STRIDE) + jnp.arange(
            ps.shape[0], dtype=ps.dtype
        )
        payload = None
        if width is not None:
            payload = gid[:, None] * (1.0 + jnp.arange(width, dtype=ps.dtype)[None, :])

        halo = import_near_halo(
            rct,
            res,
            ps,
            ms,
            _NDEV,
            leaf_size=_LEAF,
            max_req_leaves=max_req,
            max_recv_leaves=max_req,
            payload_sorted=payload,
            axis_name=AXIS_NAME,
        )
        if width is None:
            assert halo.payload is None
            return (halo.gid, jnp.zeros((halo.gid.shape[0], 1), ps.dtype))
        return (halo.gid, halo.payload)

    return shard_map(
        fn,
        mesh=make_mesh(_NDEV),
        in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME), P(AXIS_NAME)),
        check_vma=False,
    )(pts, mass)


def test_a_payload_row_stays_with_its_halo_particle():
    """Row ``h`` must decode to the gid the import reports for row ``h``."""
    width = 2
    gid, payload = (np.asarray(x) for x in _run(width))
    gid = gid.reshape(-1)
    payload = payload.reshape(-1, width)
    assert payload.shape[0] == gid.shape[0]

    checked = 0
    for row in range(gid.shape[0]):
        if gid[row] < 0:  # a padding halo slot carries no identity
            continue
        # EXACT, not a tolerance. The payload is only ever copied -- gathered,
        # concatenated, exchanged -- never arithmetic'd, and every value here is an
        # integer well inside float64's exact range, so any difference at all is a
        # misattached row rather than round-off. A tolerance would be a guess about
        # arithmetic that does not happen.
        want = float(gid[row]) * (1.0 + np.arange(width))
        np.testing.assert_array_equal(
            payload[row],
            want,
            err_msg=(
                f"halo row {row} carries payload {payload[row]} but its gid says "
                f"{gid[row]}"
            ),
        )
        checked += 1
    # Vacuity guard: an all-padding halo would make the loop assert nothing.
    assert checked > 0, "no valid halo particles -- the test would be vacuous"


def test_omitting_the_payload_costs_nothing():
    """``None`` in, ``None`` out -- a caller with no per-particle data pays nothing."""
    gid, _ = (np.asarray(x) for x in _run(None))
    assert gid.reshape(-1).shape[0] > 0
