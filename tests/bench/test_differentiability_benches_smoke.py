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


def test_nn_rebuild_single_smoke(tmp_path):
    """The original single-run harness still writes the legacy payload shape."""
    output = tmp_path / "nn_rebuild.json"
    result = subprocess.run(
        [sys.executable, "bench/differentiability/nn_rebuild.py", "--smoke"]
        + ["--output", str(output)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"nn_rebuild --smoke failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
    )
    payload = json.loads(output.read_text())
    assert payload["benchmark"] == "nn_rebuild"
    # The paper text and the figure notebook read these keys; the sweep must not
    # have changed the single-run schema.
    assert set(payload) == {"benchmark", "params", "metadata", "summary", "trajectory"}
    assert set(payload["summary"]) == {
        "initial_mean_nn",
        "final_mean_nn",
        "initial_loss",
        "final_loss",
        "topology_changes",
    }
    assert payload["summary"]["final_loss"] < payload["summary"]["initial_loss"]
    assert output.with_suffix(".npz").exists()


def test_nn_rebuild_sweep_smoke(tmp_path):
    """The scale-up sweep runs at tiny N and labels what it measured."""
    output = tmp_path / "nn_rebuild_scaling.json"
    result = subprocess.run(
        [sys.executable, "bench/differentiability/nn_rebuild.py", "--smoke-sweep"]
        + ["--output", str(output)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"nn_rebuild --smoke-sweep failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
    )
    payload = json.loads(output.read_text())
    assert payload["benchmark"] == "nn_rebuild_scaling"
    assert payload["records"], "sweep produced no records"
    assert payload["metadata"]["jax_version"]

    # A permutation-switch rate must not be mistaken for an interaction-switch
    # rate, so the label travels with the payload and with every record.
    assert payload["switch_metric"] == "morton_ordering"
    assert all(r["switch_metric"] == "morton_ordering" for r in payload["records"])
    assert set(payload["limitations"]) == {
        "objective",
        "optimizer",
        "switch_metric",
        "not_a_measure_estimate",
        "budget_scaling",
        "extensive_vs_intensive",
    }

    seeds = sorted({r["seed"] for r in payload["records"]})
    assert len(seeds) > 1, "the switch-rate trend needs more than one seed per N"
    for rec in payload["records"]:
        assert rec["free_parameters"] == rec["num_particles"] * rec["dim"]
        summary = rec["summary"]
        changes = summary["topology_changes"]
        assert 0 <= changes <= rec["steps"]
        assert summary["topology_change_fraction"] == changes / rec["steps"]
        assert rec["timing"]["total_s"] > 0

        # The intensive rates are the ones that survive large N; the extensive
        # indicator saturates. Both must be present and mutually consistent.
        for key in (
            "slot_change_fraction_mean",
            "leaf_change_fraction_mean",
            "mean_abs_rank_shift_normalized",
        ):
            assert 0.0 <= summary[key] <= 1.0, key
        # A particle cannot change leaf without changing slot.
        assert summary["leaf_change_fraction_mean"] <= (
            summary["slot_change_fraction_mean"] + 1e-6
        )
        # Any step counted as changed must have moved at least one particle.
        traj = rec["trajectory"]
        for changed, frac in zip(
            traj["topology_changed"], traj["slot_change_fraction"]
        ):
            assert changed == (frac > 0.0)

    # One aggregate entry per N, carrying the spread over seeds.
    assert [a["num_particles"] for a in payload["aggregate"]] == sorted(
        {r["num_particles"] for r in payload["records"]}
    )
    for agg in payload["aggregate"]:
        assert agg["seeds"] == seeds
        assert agg["topology_change_fraction"]["std"] >= 0.0


def test_nn_rebuild_gradient_check_smoke(tmp_path):
    """Autodiff through a rebuild is transparent, and matches FD when pinned."""
    output = tmp_path / "nn_rebuild_gradient_check.json"
    result = subprocess.run(
        [sys.executable, "bench/differentiability/nn_rebuild.py"]
        + ["--smoke-gradient-check", "--output", str(output)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"nn_rebuild --smoke-gradient-check failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
    )
    payload = json.loads(output.read_text())
    assert payload["benchmark"] == "nn_rebuild_gradient_check"
    assert payload["records"]

    checked = [r for r in payload["records"] if "error" not in r]
    assert checked, "no gradient check completed"
    for rec in checked:
        # build_tree returns integers, so differentiating *through* the rebuild
        # must equal differentiating at a frozen ordering -- exactly, not nearly.
        assert rec["rebuild_transparency"]["identical"], rec["num_particles"]
        assert rec["rebuild_transparency"]["max_abs_grad_difference"] == 0.0

    # The correctness gate: at a pinned topology autodiff matches central
    # differences. float64 has the precision to say so tightly.
    f64 = [r for r in checked if r["dtype"] == "float64"]
    assert f64, "float64 check did not run"
    for rec in f64:
        assert rec["best_pinned"]["rel_err_pinned"] < 1e-6, (
            rec["num_particles"],
            rec["best_pinned"],
        )
