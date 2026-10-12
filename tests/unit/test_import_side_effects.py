"""Importing yggdrax must not start a JAX backend.

A module-level ``jnp`` array is a device computation: until 2026-10
``_interactions_impl._FILLER_PAIR`` (and, before the x64 change, the Morton masks)
started the backend when yggdrax was imported. After that, ``jax.distributed.
initialize()`` and XLA flags or ``CUDA_VISIBLE_DEVICES`` set by the caller come too
late. Module-level constants are numpy arrays instead.
"""

from __future__ import annotations

import os
import subprocess
import sys

_CODE = """
import importlib, pkgutil
from jax._src import xla_bridge as xb
import yggdrax
names = [m.name for m in pkgutil.walk_packages(yggdrax.__path__, "yggdrax.")]
for name in ["yggdrax", *names]:
    importlib.import_module(name)
    if xb._backends:
        print("STARTED", name)
        break
else:
    print("CLEAN", len(names))
"""


def test_importing_yggdrax_and_its_submodules_starts_no_backend():
    """``import yggdrax`` and every submodule leave JAX's backends uninitialised."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    out = subprocess.run(
        [sys.executable, "-c", _CODE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("CLEAN"), last
