"""The frontier's opaque per-leaf payload must survive the trip, in order.

A cross-domain far field needs the remote leaf's MULTIPOLE, not just its mass and
centre of mass, and a multipole lives in the caller's basis -- so the frontier carries
an opaque row block instead of the tree library learning about spherical harmonics.

The one property worth pinning is the one a caller cannot arrange for itself: a
payload row stays attached to the coarse particle it describes, through the
``all_gather``, through the drop of the own-domain entries, and through the coarse
tree's own Morton reordering. Getting that wrong attaches every remote leaf's
expansion to the wrong leaf -- a wrong force with no shape error and no obvious
signature, since every row is individually plausible.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_frontier_payload.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.let import build_coarse_frontier, build_remote_coarse_tree
from yggdrax.distributed.local_tree import _build_local_tree, sanitize_padding
from yggdrax.distributed.partition import global_bounds, sfc_partition
from yggdrax.distributed.sharding import AXIS_NAME
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.tree_moments import compute_tree_mass_moments

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="a remote coarse tree needs >= 2 devices"
)

_NDEV = min(4, device_count())
_LEAF = 8
_PER_DEV = 24
_CAP = 4 * _PER_DEV
_WIDTH = 5  # payload columns, standing in for expansion coefficients


def _run():
    """Return, per device, the coarse tree's payload and its own origin tags."""
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    rng = np.random.default_rng(4)
    n = _PER_DEV * _NDEV
    pts = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, 3)))
    mass = jnp.asarray(rng.uniform(0.5, 2.0, size=(n,)))

    def fn(pos, m):
        bounds = global_bounds(pos, axis_name=AXIS_NAME)
        p_, m_, c_, cnt = sfc_partition(
            pos, m, _NDEV, output_capacity=_CAP, bounds=bounds, axis_name=AXIS_NAME
        )
        p_, m_ = sanitize_padding(p_, m_, cnt)
        tree, ps, ms = _build_local_tree(
            p_, m_, bounds, tree_type="radix", leaf_size=_LEAF
        )
        mom = compute_tree_mass_moments(tree, ps, ms)
        num_nodes = mom.mass.shape[0]
        me = jax.lax.axis_index(AXIS_NAME)
        # A payload whose value ENCODES where it came from, so a misattached row is
        # identifiable rather than merely different: row = domain*1000 + node id.
        tag = me.astype(mom.mass.dtype) * 1000.0 + jnp.arange(
            num_nodes, dtype=mom.mass.dtype
        )
        payload = tag[:, None] + jnp.arange(_WIDTH, dtype=mom.mass.dtype)[None, :] / 100
        fr = build_coarse_frontier(
            tree,
            mom.mass,
            mom.center_of_mass,
            positions_sorted=ps,
            max_leaf_size=_LEAF,
            node_payload=payload,
        )
        rct = build_remote_coarse_tree(fr, _NDEV, bounds=bounds, axis_name=AXIS_NAME)
        return rct.payload, rct.tag_domain, rct.tag_node_id

    return shard_map(
        fn,
        mesh=make_mesh(_NDEV),
        in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME),) * 3,
        check_vma=False,
    )(pts, mass)


def test_a_payload_row_stays_with_its_coarse_particle():
    """Every row must equal ``domain*1000 + node_id`` for the tags beside it."""
    payload, domain, node_id = (np.asarray(x) for x in _run())
    ncoarse = payload.shape[0] // _NDEV
    assert payload.shape == (_NDEV * ncoarse, _WIDTH)

    checked = 0
    for row in range(payload.shape[0]):
        if node_id[row] < 0:  # a padding frontier leaf carries no identity
            continue
        want = float(domain[row]) * 1000.0 + float(node_id[row])
        got = payload[row]
        np.testing.assert_allclose(
            got,
            want + np.arange(_WIDTH) / 100,
            rtol=0,
            atol=1e-9,
            err_msg=(
                f"row {row} carries {got[0]} but its tags say "
                f"domain {domain[row]} node {node_id[row]} -> {want}"
            ),
        )
        checked += 1
    # Vacuity guard: all-padding tags would make the loop above assert nothing.
    assert checked > 0, "no non-padding coarse particles -- test would be vacuous"


def test_omitting_the_payload_costs_nothing():
    """``None`` in, ``None`` out -- a monopole-only caller pays no bandwidth."""
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    rng = np.random.default_rng(6)
    n = _PER_DEV * _NDEV
    pts = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, 3)))
    mass = jnp.asarray(rng.uniform(0.5, 2.0, size=(n,)))

    def fn(pos, m):
        bounds = global_bounds(pos, axis_name=AXIS_NAME)
        p_, m_, c_, cnt = sfc_partition(
            pos, m, _NDEV, output_capacity=_CAP, bounds=bounds, axis_name=AXIS_NAME
        )
        p_, m_ = sanitize_padding(p_, m_, cnt)
        tree, ps, ms = _build_local_tree(
            p_, m_, bounds, tree_type="radix", leaf_size=_LEAF
        )
        mom = compute_tree_mass_moments(tree, ps, ms)
        fr = build_coarse_frontier(
            tree, mom.mass, mom.center_of_mass, positions_sorted=ps, max_leaf_size=_LEAF
        )
        assert fr.payload is None
        rct = build_remote_coarse_tree(fr, _NDEV, bounds=bounds, axis_name=AXIS_NAME)
        assert rct.payload is None
        return jnp.sum(rct.masses_sorted)[None].astype(INDEX_DTYPE)

    out = shard_map(
        fn,
        mesh=make_mesh(_NDEV),
        in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
        out_specs=P(AXIS_NAME),
        check_vma=False,
    )(pts, mass)
    assert np.asarray(out).shape == (_NDEV,)
