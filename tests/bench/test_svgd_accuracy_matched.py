"""The accuracy-matched schedule picks the right theta and the right crossover.

``bench/svgd/accuracy_matched.py`` decides a number the paper quotes, from JSON
alone, so its selection rules are checked on synthetic inputs whose answer is
known: the largest theta within tolerance, an N where none qualifies, the
fastest exact time across files, and a crossover that needs the tree to stay
faster at every larger N.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = "bench/svgd/accuracy_matched.py"

# errors[theta][n], tree phi times[theta][n] (s), exact times[theta][n] (s)
_ERRORS = {
    0.2: {1000: 1e-15, 10000: 1e-5, 100000: 4e-4},
    0.3: {1000: 1e-15, 10000: 5e-4, 100000: 2e-3},
    0.4: {1000: 1e-15, 10000: 3e-3, 100000: 9e-3},
}
_PHI = {
    0.2: {1000: 2e-3, 10000: 30e-3, 100000: 400e-3},
    0.3: {1000: 2e-3, 10000: 20e-3, 100000: 200e-3},
    0.4: {1000: 2e-3, 10000: 10e-3, 100000: 100e-3},
}
_EXACT = {1000: 1e-3, 10000: 25e-3, 100000: 300e-3}


def _write_inputs(tmp_path: Path, *, device: str = "A100") -> list[str]:
    paths = []
    for i, theta in enumerate(sorted(_ERRORS)):
        records = []
        for n in (1000, 10000, 100000):
            records.append(
                {
                    "n": n,
                    "rel_error_vs_exact": _ERRORS[theta][n],
                    "phi": {"min_s": _PHI[theta][n]},
                    "value_and_grad": {"min_s": 3 * _PHI[theta][n]},
                    "build_s": 0.1 * _PHI[theta][n],
                    # every file times the same exact update; one is slower
                    "exact_s": _EXACT[n] * (1.5 if i == 0 else 1.0),
                    "accumulate_resolved": "scatter",
                }
            )
        payload = {
            "benchmark": "svgd_scaling",
            "params": {"theta": theta, "dim": 3, "dtype": "float64", "seed": 0},
            "metadata": {
                "jax_version": "0.0",
                "device_kind": device if theta != 0.4 else "A100",
                "git_commit": "abc1234",
                "x64_enabled": True,
            },
            "records": records,
        }
        path = tmp_path / f"theta_{theta}.json"
        path.write_text(json.dumps(payload))
        paths.append(str(path))
    return paths


def _run(paths: list[str], output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, _SCRIPT, "--inputs", *paths, "--output", str(output)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_schedule_and_crossover(tmp_path):
    output = tmp_path / "accuracy_matched.json"
    result = _run(_write_inputs(tmp_path), output)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    tight = payload["schedules"]["0.001"]
    rows = {row["n"]: row for row in tight["records"]}

    # Largest theta within 1e-3; the grid ceiling at N = 1e3 is not a bracket.
    assert rows[1000]["theta"] == 0.4 and not rows[1000]["bracketed"]
    assert rows[10000]["theta"] == 0.3 and rows[10000]["theta_above"] == 0.4
    assert rows[100000]["theta"] == 0.2 and rows[100000]["bracketed"]
    assert tight["unmet_n"] == [] and tight["unbracketed_n"] == [1000]

    # The exact time is the fastest of the per-file measurements.
    assert rows[10000]["exact_s"] == pytest.approx(25e-3)
    # phi: 2 > 1, 20 < 25, 400 > 300 ms -> slower again at the largest N.
    assert tight["crossover"]["forward"]["crossover_n"] is None
    assert tight["crossover"]["forward"]["faster_at_n"] == [10000]

    loose = payload["schedules"]["0.01"]
    rows = {row["n"]: row for row in loose["records"]}
    assert [rows[n]["theta"] for n in (1000, 10000, 100000)] == [0.4, 0.4, 0.4]
    # phi: 2 > 1, 10 < 25, 100 < 300 ms -> faster from N = 1e4 on.
    assert loose["crossover"]["forward"]["crossover_n"] == 10000
    assert loose["crossover"]["forward"]["last_slower_n"] == 1000
    # forward+gradient is 3x phi: 30 > 25 at 1e4, 300 = 300 at 1e5 -> never.
    assert loose["crossover"]["forward_plus_gradient"]["crossover_n"] is None


def test_an_unmet_tolerance_is_a_result_not_a_gap(tmp_path):
    output = tmp_path / "accuracy_matched.json"
    paths = _write_inputs(tmp_path)
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--inputs", *paths, "--tolerances", "1e-4"]
        + ["--output", str(output)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    sched = json.loads(output.read_text())["schedules"]["0.0001"]
    assert sched["unmet_n"] == [100000]
    row = sched["records"][-1]
    assert row["smallest_theta"] == 0.2 and row["smallest_theta_error"] == 4e-4
    assert sched["crossover"]["forward"]["crossover_n"] is None


def test_inputs_from_two_machines_are_refused(tmp_path):
    result = _run(_write_inputs(tmp_path, device="RTX"), tmp_path / "out.json")
    assert result.returncode != 0
    assert "one regime on one machine" in result.stderr
