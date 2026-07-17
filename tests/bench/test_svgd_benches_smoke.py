"""End-to-end smoke tests for the SVGD benchmark scripts.

Assert each harness runs to completion on a tiny problem and writes a
well-formed results JSON. Each runs as an isolated subprocess (the scripts
enable float64 globally).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPTS = [
    ("convergence", "bench/svgd/convergence_vs_exact.py", "svgd_convergence"),
    ("scaling", "bench/svgd/scaling.py", "svgd_scaling"),
    (
        "bandwidth",
        "bench/svgd/bandwidth_learning_experiment.py",
        "svgd_bandwidth_learning",
    ),
]


@pytest.mark.parametrize(
    "name,script,benchmark", _SCRIPTS, ids=[s[0] for s in _SCRIPTS]
)
def test_svgd_bench_smoke(tmp_path, name, script, benchmark):
    output = tmp_path / f"{name}.json"
    result = subprocess.run(
        [sys.executable, script, "--smoke", "--output", str(output)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert output.exists(), f"{script} did not write {output}"
    payload = json.loads(output.read_text())
    assert payload["benchmark"] == benchmark
    assert payload["metadata"]["jax_version"]
