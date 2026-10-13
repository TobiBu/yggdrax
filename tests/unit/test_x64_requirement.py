"""yggdrax needs ``jax_enable_x64`` and no longer switches it on itself.

Until 2026-10 ``yggdrax/__init__.py`` turned x64 on at import, which changed the
default dtypes of every program that imported it and made ``JAX_ENABLE_X64=0``
ineffective. Simply dropping that line was not enough: with x64 off, JAX narrows
``uint64`` to ``uint32`` with at most a warning, so the 63-bit Morton codes lost
their top bits and the trees came out wrong with no error. The helpers that make
64-bit arrays now call :func:`yggdrax.dtypes.require_x64` instead.
"""

from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import build_octree, build_tree, morton_decode, morton_encode
from yggdrax.dtypes import require_x64


@pytest.fixture
def x64_off():
    """Run the test with x64 off, then restore the suite's setting."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _particles(n: int = 64):
    rng = np.random.default_rng(0)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, (n, 3)), dtype=jnp.float32)
    masses = jnp.ones((n,), dtype=jnp.float32)
    return positions, masses


def _bounds():
    return (
        jnp.full((3,), -1.0, dtype=jnp.float32),
        jnp.full((3,), 1.0, dtype=jnp.float32),
    )


def test_importing_yggdrax_leaves_jax_enable_x64_alone():
    """A fresh process keeps x64 off, and the import truncates no 64-bit constant."""
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run(
        [
            sys.executable,
            "-W",
            "error:Explicitly requested dtype",
            "-c",
            "import jax, yggdrax; print(jax.config.jax_enable_x64)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_require_x64_passes_when_x64_is_on():
    if not jax.config.jax_enable_x64:
        pytest.skip("the suite runs with JAX_ENABLE_X64=0 (the x64-off guard sweep)")
    require_x64("anything")


@pytest.mark.parametrize(
    "entry",
    ["morton_encode", "morton_decode", "build_tree", "build_octree"],
)
def test_64_bit_entry_points_raise_without_x64(x64_off, entry):
    """Each public route to 64-bit codes names the flag instead of truncating."""
    positions, masses = _particles()
    calls = {
        "morton_encode": lambda: morton_encode(positions, _bounds()),
        "morton_decode": lambda: morton_decode(
            jnp.zeros((4,), dtype=jnp.uint32), _bounds()
        ),
        "build_tree": lambda: build_tree(positions, masses, leaf_size=8),
        "build_octree": lambda: build_octree(positions, masses, leaf_size=8),
    }
    with pytest.raises(RuntimeError, match="jax_enable_x64"):
        calls[entry]()
