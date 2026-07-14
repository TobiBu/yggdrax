"""End-to-end smoke tests for the differentiability benchmark scripts.

These assert only that each harness runs to completion on a tiny problem and
writes a well-formed results JSON -- not that the full sweep is correct or
fast. Each script is run as an isolated subprocess so that global side effects
(e.g. ``mac_accuracy`` enabling float64) do not leak into the test session.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPTS = [
    ("scaling", "bench/differentiability/scaling.py", "scaling"),
    (
        "autodiff_overhead",
        "bench/differentiability/autodiff_overhead.py",
        "autodiff_overhead",
    ),
    ("mac_accuracy", "bench/differentiability/mac_accuracy.py", "mac_accuracy"),
]


@pytest.mark.parametrize(
    "name,script,benchmark", _SCRIPTS, ids=[s[0] for s in _SCRIPTS]
)
def test_bench_script_smoke(tmp_path, name, script, benchmark):
    output = tmp_path / f"{name}.json"
    result = subprocess.run(
        [sys.executable, script, "--smoke", "--output", str(output)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert output.exists(), f"{script} did not write {output}"

    payload = json.loads(output.read_text())
    assert payload["benchmark"] == benchmark
    assert payload["records"], f"{script} produced no records"
    assert "metadata" in payload and payload["metadata"]["jax_version"]
