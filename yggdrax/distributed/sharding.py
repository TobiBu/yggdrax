"""Device mesh construction for Yggdrax multi-GPU execution.

Phase 0 of the multi-GPU roadmap. This module owns the (single-node for now)
device discovery and ``jax.sharding.Mesh`` construction used by every sharded
Yggdrax op. It is deliberately thin: everything downstream expresses
communication with ``jax.lax`` collectives over the ``AXIS_NAME`` axis, so the
same code runs single-node today and multi-host once
``jax.distributed.initialize`` is wired in (see roadmap Phase 6).
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax

# Single mesh axis name shared by every collective in ``yggdrax.distributed``.
# Keeping it a module constant means callers never have to thread the string
# through, and collectives (``psum``, ``all_gather``, ``ragged_all_to_all``)
# stay consistent with the mesh built here.
AXIS_NAME = "gpus"


def available_devices(
    platform: Optional[str] = None,
) -> list[jax.Device]:
    """Return the JAX devices to shard over.

    When ``platform`` is ``None`` we use the default JAX backend (GPU when a
    CUDA build is installed, otherwise CPU / the host-forced devices used in
    tests). Passing an explicit platform (``"cpu"``/``"gpu"``) is mainly a
    testing convenience.
    """

    if platform is None:
        return list(jax.devices())
    return list(jax.devices(platform))


def device_count(platform: Optional[str] = None) -> int:
    """Number of devices available for sharding."""

    return len(available_devices(platform))


def make_mesh(
    num_devices: Optional[int] = None,
    *,
    axis_name: str = AXIS_NAME,
    devices: Optional[Sequence[jax.Device]] = None,
) -> jax.sharding.Mesh:
    """Build a 1-D device mesh for data-parallel tree/FMM sharding.

    Parameters
    ----------
    num_devices:
        Number of devices along the mesh axis. Defaults to *all* available
        devices. Must not exceed the number of available devices.
    axis_name:
        Name of the single mesh axis; defaults to :data:`AXIS_NAME`.
    devices:
        Explicit device list to place on the mesh. Defaults to the available
        devices (optionally truncated to ``num_devices``).
    """

    pool = list(devices) if devices is not None else available_devices()
    if not pool:
        raise RuntimeError("no JAX devices available to build a mesh")

    n = len(pool) if num_devices is None else int(num_devices)
    if n <= 0:
        raise ValueError(f"num_devices must be positive, got {n}")
    if n > len(pool):
        raise ValueError(f"requested {n} devices but only {len(pool)} are available")

    # Auto axis types keep arrays free of sharding-in-types annotations, so the
    # classic data-parallel ``shard_map`` (Manual inside the body) composes
    # cleanly. Newer JAX defaults ``make_mesh`` to Explicit, which propagates
    # Explicit shardings into the body and conflicts with Manual collectives.
    return jax.make_mesh(
        (n,), (axis_name,), devices=pool[:n], axis_types=(jax.sharding.AxisType.Auto,)
    )


__all__ = [
    "AXIS_NAME",
    "available_devices",
    "device_count",
    "make_mesh",
]
