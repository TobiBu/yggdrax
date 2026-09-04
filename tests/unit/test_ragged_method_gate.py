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
