"""Shared helpers for the differentiability benchmark scripts.

Kept deliberately free of a top-level ``import jax`` so that
:func:`select_free_gpu` can run *before* JAX initialises the backend (the
``autocvd`` free-GPU pick must precede ``import jax``). All JAX-using helpers
import JAX lazily inside the function body.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def select_free_gpu(mode: str = "free", *, tag: str = "bench") -> None:
    """Pick a free GPU via ``autocvd`` before JAX is imported.

    No-op when ``mode == "none"``, when ``CUDA_VISIBLE_DEVICES`` is already
    set, or when ``autocvd`` is unavailable (e.g. local CPU runs). Mirrors the
    jaccpot bench convention.

    Args:
        mode: One of ``"free"``, ``"least-used"``, or ``"none"``.
        tag: Short label used in the fallback log line.
    """
    if mode == "none" or "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        from autocvd import autocvd
    except Exception as exc:  # pragma: no cover - env-dependent
        print(f"[{tag}] autocvd unavailable ({exc}); using default device")
        return
    autocvd(num_gpus=1, least_used=(mode == "least-used"))


def _git_commit(repo_root: Path) -> str | None:
    """Return the short git commit of ``repo_root`` or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - env-dependent
        return None


def run_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect reproducibility metadata for a results JSON header."""
    import jax

    dev = jax.devices()[0]
    repo_root = Path(__file__).resolve().parents[2]
    meta: dict[str, Any] = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device_kind": getattr(dev, "device_kind", str(dev)),
        "device_platform": dev.platform,
        "compute_capability": getattr(dev, "compute_capability", None),
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "git_commit": _git_commit(repo_root),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": os.uname().nodename,
    }
    if extra:
        meta.update(extra)
    return meta


@dataclass
class TimingResult:
    """Wall-clock timing summary in seconds."""

    runs: int
    warmup: int
    times_s: list[float] = field(default_factory=list)

    @property
    def min_s(self) -> float:
        return min(self.times_s)

    @property
    def median_s(self) -> float:
        s = sorted(self.times_s)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    @property
    def mean_s(self) -> float:
        return sum(self.times_s) / len(self.times_s)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "warmup": self.warmup,
            "min_s": self.min_s,
            "median_s": self.median_s,
            "mean_s": self.mean_s,
            "times_s": self.times_s,
        }


def time_callable(
    fn: Callable[[], Any],
    *,
    warmup: int = 2,
    runs: int = 5,
) -> TimingResult:
    """Time a zero-arg callable, blocking on JAX results each call.

    The callable is responsible for returning JAX arrays (or pytrees thereof);
    we call ``jax.block_until_ready`` on the result so asynchronous dispatch
    does not corrupt the measurement. JIT compilation should be triggered by
    the warmup iterations.

    Args:
        fn: Zero-argument callable returning JAX arrays to block on.
        warmup: Number of untimed warmup calls (compilation, caches).
        runs: Number of timed calls.

    Returns:
        A :class:`TimingResult` with per-run wall times.
    """
    import jax

    for _ in range(warmup):
        jax.block_until_ready(fn())
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return TimingResult(runs=runs, warmup=warmup, times_s=times)


def dump_json(payload: dict[str, Any], output: str | Path) -> Path:
    """Write ``payload`` as pretty JSON to ``output`` (creating parents)."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[bench] wrote {path}")
    return path


def dump_npz(arrays: dict[str, Any], output: str | Path) -> Path:
    """Write named arrays to a compressed ``.npz`` at ``output``.

    Used for bulk numeric payloads (trajectories, per-step histories) that are
    too large or too array-shaped for the summary JSON. Values are coerced to
    NumPy via :func:`numpy.asarray`, so JAX arrays are accepted directly.

    Args:
        arrays: Mapping of array name to array-like value.
        output: Destination ``.npz`` path (parents are created).

    Returns:
        The path written to.
    """
    import numpy as np

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print(f"[bench] wrote {path}")
    return path
