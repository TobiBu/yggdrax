"""End-to-end smoke tests for the corrfunc benchmark scripts.

Assert each harness runs to completion on a tiny problem and writes a
well-formed results JSON. Each script runs as an isolated subprocess (the
scripts enable float64 globally, which must not leak into the test session).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPTS = [
    ("validate", "bench/corrfunc/validate_vs_baseline.py", "corrfunc_validation"),
    ("scaling", "bench/corrfunc/scaling.py", "corrfunc_scaling"),
    (
        "bin_sensitivity",
        "bench/corrfunc/bin_width_sensitivity.py",
        "corrfunc_bin_sensitivity",
    ),
    ("recovery", "bench/corrfunc/parameter_recovery_demo.py", "corrfunc_recovery"),
]


@pytest.mark.parametrize(
    "name,script,benchmark", _SCRIPTS, ids=[s[0] for s in _SCRIPTS]
)
def test_corrfunc_bench_smoke(tmp_path, name, script, benchmark):
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
    assert payload["records"], f"{script} produced no records"
    assert payload["metadata"]["jax_version"]
