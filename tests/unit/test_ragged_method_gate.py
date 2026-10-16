"""``ragged_all_to_all_exchange(method="auto")`` must not pick the native collective
where it silently returns its fill value.

Pure resolver tests -- no devices, no compile -- so they run on every CI runner.
The boundary is measured, not read off a changelog: on jax 0.9.0 a
``jit(shard_map(ragged_all_to_all))`` whose input is donated returns ``fill_value``
on 36/40 calls (first call clean); 0.9.1 and 0.10.2 are clean 40/40. See
:data:`yggdrax.distributed.comm.RAGGED_NATIVE_FIXED_JAX`.
"""

from __future__ import annotations

import pytest

import yggdrax.distributed.comm as comm


def test_auto_gates_on_version(monkeypatch):
    monkeypatch.setattr(comm.jax, "default_backend", lambda: "gpu")
    for version, expected in {
        "0.8.3": "buf",
        "0.9.0": "buf",  # the version the forward corruption was measured on
        "0.9.0.1": "buf",
        "0.9.1": "native",  # first fixed release
        "0.10.2": "native",
        "1.0.0": "native",
    }.items():
        monkeypatch.setattr(comm.jax, "__version__", version, raising=False)
        assert comm.resolve_ragged_method("auto") == expected, version
        # an explicit choice is never overridden
        assert comm.resolve_ragged_method("buf") == "buf"
        assert comm.resolve_ragged_method("native") == "native"
    for version in ("0.9.1.dev20260301", "0.10.0rc1", "0.9"):
        monkeypatch.setattr(comm.jax, "__version__", version, raising=False)
        assert comm.resolve_ragged_method("auto") in ("buf", "native")


def test_auto_gates_on_backend(monkeypatch):
    monkeypatch.setattr(comm.jax, "__version__", "0.10.2", raising=False)
    monkeypatch.setattr(comm.jax, "default_backend", lambda: "cpu")
    assert comm.resolve_ragged_method("auto") == "buf"
    for backend in ("gpu", "tpu"):
        monkeypatch.setattr(comm.jax, "default_backend", lambda b=backend: b)
        assert comm.resolve_ragged_method("auto") == "native"


def test_both_gates_are_needed(monkeypatch):
    for version, backend, expected in (
        ("0.10.2", "gpu", "native"),
        ("0.10.2", "cpu", "buf"),
        ("0.9.0", "gpu", "buf"),
        ("0.9.0", "cpu", "buf"),
    ):
        monkeypatch.setattr(comm.jax, "__version__", version, raising=False)
        monkeypatch.setattr(comm.jax, "default_backend", lambda b=backend: b)
        assert comm.resolve_ragged_method("auto") == expected, (version, backend)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be"):
        comm.resolve_ragged_method("ragged")


_ONE_SHOT_OFF = "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false"
_BARRIER = "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl"


@pytest.mark.parametrize(
    "version, flags, broken",
    [
        # XLA's default one-shot kernel: nothing to check
        ("0.11.2", "", False),
        ("0.11.2", f"{_BARRIER}=true", False),
        # the one-flag NCCL setting: safe on 0.10 (barrier off by default), not on 0.11
        ("0.10.2", _ONE_SHOT_OFF, False),
        ("0.11.2", _ONE_SHOT_OFF, True),
        # the barrier decides, on either version
        ("0.11.2", f"{_ONE_SHOT_OFF} {_BARRIER}=false", False),
        ("0.11.2", f"{_ONE_SHOT_OFF} {_BARRIER}=0", False),
        ("0.10.2", f"{_ONE_SHOT_OFF} {_BARRIER}=true", True),
        ("0.10.2", f"{_ONE_SHOT_OFF} {_BARRIER}", True),  # a bare flag means true
        # the later token wins, as in XLA
        ("0.11.2", f"{_ONE_SHOT_OFF} {_BARRIER}=false {_BARRIER}=true", True),
        ("0.11.2", f"{_ONE_SHOT_OFF} {_ONE_SHOT_OFF[:-5]}=true", False),
        ("0.11.2", _ONE_SHOT_OFF.replace("false", "False"), True),
    ],
)
def test_nccl_barrier_hazard(monkeypatch, version, flags, broken):
    """The NCCL send/recv fallback with use_barrier_with_nccl on delivers no
    cross-device rows and deadlocks (measured on jax 0.10.2 and 0.11.2, A100 pair)."""
    monkeypatch.setattr(comm.jax, "__version__", version, raising=False)
    reason = comm.ragged_native_flag_hazard(flags)
    assert (reason is not None) == broken, reason
    if broken:
        assert "use_barrier_with_nccl=false" in reason


def test_hazard_reads_xla_flags_by_default(monkeypatch):
    monkeypatch.setattr(comm.jax, "__version__", "0.11.2", raising=False)
    monkeypatch.setenv("XLA_FLAGS", _ONE_SHOT_OFF)
    assert comm.ragged_native_flag_hazard() is not None
    monkeypatch.setenv("XLA_FLAGS", f"{_ONE_SHOT_OFF} {_BARRIER}=false")
    assert comm.ragged_native_flag_hazard() is None


def test_native_exchange_raises_on_the_hazard(monkeypatch):
    """Traced only (one CPU device, no compile): the guard fires before the op."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.array(jax.devices()[:1]), (comm.AXIS_NAME,))

    def body(x):
        out, _, _ = comm.ragged_all_to_all_exchange(
            x, jnp.ones((1,), jnp.int32), output_capacity=4, method="native"
        )
        return out

    fn = jax.shard_map(
        body, mesh=mesh, in_specs=P(comm.AXIS_NAME), out_specs=P(comm.AXIS_NAME)
    )
    x = jnp.zeros((4, 3), jnp.float32)
    monkeypatch.setattr(comm.jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("XLA_FLAGS", _ONE_SHOT_OFF)
    monkeypatch.setattr(comm.jax, "__version__", "0.11.2", raising=False)
    with pytest.raises(RuntimeError, match="use_barrier_with_nccl"):
        jax.make_jaxpr(fn)(x)
    monkeypatch.setenv("XLA_FLAGS", f"{_ONE_SHOT_OFF} {_BARRIER}=false")
    jax.make_jaxpr(fn)(x)  # the safe setting traces
