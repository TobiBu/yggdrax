"""``jacfwd`` / ``vmap`` through the native ragged exchange.

Forward-mode Jacobians and ``vmap`` batch the exchanged rows while the sizes and
offsets stay unbatched; JAX's own batching rule for ``ragged_all_to_all`` asserts on
exactly that case (the b3 forecast row on GPU). ``yggdrax.distributed.comm``
batches it along a feature axis instead.

XLA:CPU has no ``ragged_all_to_all``, so on CPU only the trace is checked; on a GPU
the native result and its derivatives are compared with the all_gather path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # pragma: no cover - older JAX
    from jax.experimental.shard_map import shard_map

from yggdrax.distributed import device_count, make_mesh, ragged_all_to_all_exchange
from yggdrax.distributed.sharding import AXIS_NAME


def _problem(ndev):
    """Send sizes, capacities and per-device operands of a small fitting exchange."""
    s = np.arange(ndev)[:, None]
    i = np.arange(ndev)[None, :]
    full = (1 + ((s + i) % 3)).astype(np.int32)
    c_in = int(full.sum(axis=1).max()) + 2
    cap = int(full.sum(axis=0).max()) + 2
    rng = np.random.default_rng(0)
    ops = rng.normal(size=(ndev * c_in, 3)).astype(np.float32)
    return full, c_in, cap, ops


def _program(ndev, method):
    """theta -> per-device sum of sin(exchanged rows scaled by theta)."""
    full, c_in, cap, ops = _problem(ndev)
    mesh = make_mesh(ndev)

    def body(op, ss, theta):
        x = op * theta[0] + theta[1]
        out, recv, _ = ragged_all_to_all_exchange(
            x, ss, output_capacity=cap, fill_value=0.0, method=method
        )
        live = (jnp.arange(cap) < recv.sum())[:, None]
        return jnp.sum(jnp.where(live, jnp.sin(out), 0.0))[None]

    prog = shard_map(
        body,
        mesh=mesh,
        in_specs=(P(AXIS_NAME), P(AXIS_NAME), P()),
        out_specs=P(AXIS_NAME),
    )
    op, ss = jnp.asarray(ops), jnp.asarray(full.reshape(-1))

    def f(theta):
        return jnp.sum(prog(op, ss, theta))

    return f


_THETA = jnp.asarray([1.3, 0.2], jnp.float32)
_TRANSFORMS = {
    "jacfwd": lambda f: jax.jacfwd(f)(_THETA),
    "vmap": lambda f: jax.vmap(f)(jnp.stack([_THETA, _THETA + 0.5, _THETA - 0.25])),
    "jacrev": lambda f: jax.jacrev(f)(_THETA),
    "vmap_grad": lambda f: jax.vmap(jax.grad(f))(jnp.stack([_THETA, _THETA * 2])),
    "hessian": lambda f: jax.hessian(f)(_THETA),
}


@pytest.mark.parametrize("name", sorted(_TRANSFORMS))
def test_native_exchange_traces_under_batching_transforms(name):
    """Tracing alone used to raise AssertionError for jacfwd / vmap / hessian."""
    ndev = min(2, device_count())
    f = _program(ndev, "native")
    jaxpr = jax.make_jaxpr(lambda: _TRANSFORMS[name](f))()
    assert "ragged_all_to_all" in str(jaxpr)


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="XLA:CPU has no ragged_all_to_all"
)
@pytest.mark.parametrize("name", sorted(_TRANSFORMS))
def test_native_exchange_derivatives_match_buf(name):
    ndev = min(2, device_count())
    got = jax.jit(lambda: _TRANSFORMS[name](_program(ndev, "native")))()
    ref = jax.jit(lambda: _TRANSFORMS[name](_program(ndev, "buf")))()
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-5)
