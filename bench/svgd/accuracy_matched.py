"""Accuracy-matched Stein-update crossover: per N, the largest theta that meets a tolerance.

``scaling.py`` runs one opening angle per invocation and records
``rel_error_vs_exact`` at every N up to ``--exact-max-n``. In the paper's regime
(fixed cloud, growing density) the tree's error at fixed theta grows with N --
2.6 % at N = 1e5 and theta = 0.5 -- so a crossover measured at fixed theta is a
crossover at a degrading accuracy. This script reads one scaling JSON per theta
and, for each N, picks the **largest** theta whose error is within a tolerance.
Along that schedule it reports the tree's forward, forward-plus-gradient and
build times, the exact baseline's time, and the N from which the tree is faster.

Rules, so the number cannot be flattering by construction:

* inputs must share one regime (dimension, density, dtype, leaf size, backend,
  cloud, seed, accumulation) and one machine (jax version, device, commit, x64);
* the exact baseline's time at each N is the **fastest** of its measurements
  across the input files -- every file times the same exact update;
* an N at which no theta meets the tolerance counts as "not faster", and the
  crossover is the smallest N from which the tree is faster at **every** larger
  measured N;
* the forward-plus-gradient time is compared with the exact *forward* time, as
  the paper does; and a third comparison adds the per-step partition build,
  which a rebuilding sampler pays and the exact update does not.

Reads JSON only; no JAX, no device.

Results -> ``results/svgd/accuracy_matched.json``.

    python bench/svgd/accuracy_matched.py \
        --inputs results/svgd/accuracy_matched/theta_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.differentiability._common import _git_commit, dump_json

#: params that must agree across the inputs: one regime per schedule.
_REGIME_KEYS = (
    "dim",
    "leaf_size",
    "backend",
    "dtype",
    "density",
    "sigma",
    "seed",
    "accumulate",
    "cutoff_bandwidths",
    "exact_block",
)
#: metadata that must agree across the inputs: one machine per schedule.
_MACHINE_KEYS = ("jax_version", "device_kind", "git_commit", "x64_enabled")
#: tree-time comparisons, each against the exact forward update.
_COMPARISONS = {
    "forward": ("phi_s",),
    "forward_plus_gradient": ("value_and_grad_s",),
    "forward_plus_build": ("phi_s", "build_s"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One svgd/scaling.py JSON per theta.",
    )
    p.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2],
        help="Relative-error tolerances; one schedule each.",
    )
    p.add_argument("--output", type=str, default="results/svgd/accuracy_matched.json")
    return p.parse_args()


def _load(paths: list[str]) -> list[dict[str, Any]]:
    """Load the inputs and refuse to mix regimes, machines or thetas."""
    runs = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if payload.get("benchmark") != "svgd_scaling":
            raise SystemExit(f"{path}: not an svgd_scaling JSON")
        runs.append({"path": path, **payload})
    first = runs[0]
    for run in runs[1:]:
        for block, keys in (("params", _REGIME_KEYS), ("metadata", _MACHINE_KEYS)):
            for key in keys:
                a, b = first[block].get(key), run[block].get(key)
                if a != b:
                    raise SystemExit(
                        f"{run['path']}: {block}.{key} = {b!r} but "
                        f"{first['path']} has {a!r}; one schedule needs one "
                        "regime on one machine"
                    )
    thetas = [run["params"]["theta"] for run in runs]
    if len(set(thetas)) != len(thetas):
        raise SystemExit(f"duplicate theta among the inputs: {sorted(thetas)}")
    return sorted(runs, key=lambda run: run["params"]["theta"])


def _table(runs: list[dict[str, Any]]) -> dict[int, dict[float, dict[str, Any]]]:
    """Index every record as ``table[n][theta]``, keeping the fields used here."""
    table: dict[int, dict[float, dict[str, Any]]] = {}
    for run in runs:
        theta = run["params"]["theta"]
        for rec in run["records"]:
            table.setdefault(rec["n"], {})[theta] = {
                "rel_error_vs_exact": rec.get("rel_error_vs_exact"),
                "phi_s": rec["phi"]["min_s"],
                "value_and_grad_s": rec["value_and_grad"]["min_s"],
                "build_s": rec["build_s"],
                "exact_s": rec.get("exact_s"),
                "accumulate_resolved": rec.get("accumulate_resolved"),
                "num_far_pairs": rec.get("num_far_pairs"),
                "num_near_leaf_pairs": rec.get("num_near_leaf_pairs"),
            }
    return dict(sorted(table.items()))


def _schedule(
    table: dict[int, dict[float, dict[str, Any]]], tol: float
) -> list[dict[str, Any]]:
    """Per N, the largest theta whose error is within ``tol``, and its bracket."""
    rows = []
    for n, by_theta in table.items():
        exact_times = [r["exact_s"] for r in by_theta.values() if r["exact_s"]]
        measured = sorted(
            (theta, r["rel_error_vs_exact"])
            for theta, r in by_theta.items()
            if r["rel_error_vs_exact"] is not None
        )
        errors = [err for _, err in measured]
        passing = [theta for theta, err in measured if err <= tol]
        row: dict[str, Any] = {
            "n": n,
            "exact_s": min(exact_times) if exact_times else None,
            "exact_s_spread": (
                max(exact_times) / min(exact_times) if exact_times else None
            ),
            "error_monotone_in_theta": all(
                a <= b for a, b in zip(errors, errors[1:], strict=False)
            ),
            "met": bool(passing),
        }
        if not measured:
            row["note"] = "no theta has rel_error_vs_exact here (n > exact_max_n)"
        elif not passing:
            row["smallest_theta"] = measured[0][0]
            row["smallest_theta_error"] = measured[0][1]
        if passing:
            theta = max(passing)
            above = [(t, e) for t, e in measured if t > theta]
            chosen = by_theta[theta]
            row.update(
                {
                    "theta": theta,
                    "rel_error_vs_exact": chosen["rel_error_vs_exact"],
                    "theta_above": above[0][0] if above else None,
                    "error_above": above[0][1] if above else None,
                    "bracketed": bool(above),
                    "phi_s": chosen["phi_s"],
                    "value_and_grad_s": chosen["value_and_grad_s"],
                    "build_s": chosen["build_s"],
                    "accumulate_resolved": chosen["accumulate_resolved"],
                    "num_far_pairs": chosen["num_far_pairs"],
                    "num_near_leaf_pairs": chosen["num_near_leaf_pairs"],
                }
            )
            if row["exact_s"]:
                for name, fields in _COMPARISONS.items():
                    tree_s = sum(row[f] for f in fields)
                    row[f"speedup_{name}"] = row["exact_s"] / tree_s
        rows.append(row)
    return rows


def _crossover(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Smallest N from which the tree is faster at every larger measured N."""
    key = f"speedup_{name}"
    faster = [(row["n"], row.get(key) is not None and row[key] > 1.0) for row in rows]
    crossover = None
    for i, (n, _) in enumerate(faster):
        if all(ok for _, ok in faster[i:]):
            crossover = n
            break
    below = [n for n, _ in faster if crossover is None or n < crossover]
    return {
        "crossover_n": crossover,
        "last_slower_n": below[-1] if below else None,
        "faster_at_n": [n for n, ok in faster if ok],
        "note": (
            None
            if crossover is not None
            else "the tree is not faster at the largest measured N"
        ),
    }


def main() -> None:
    args = _parse_args()
    runs = _load(args.inputs)
    table = _table(runs)
    first = runs[0]

    schedules = {}
    for tol in args.tolerances:
        rows = _schedule(table, tol)
        schedules[f"{tol:g}"] = {
            "tolerance": tol,
            "records": rows,
            "unmet_n": [row["n"] for row in rows if not row["met"]],
            "unbracketed_n": [
                row["n"] for row in rows if row["met"] and not row["bracketed"]
            ],
            "crossover": {name: _crossover(rows, name) for name in _COMPARISONS},
        }
        print(f"\n[accuracy-matched] tolerance {tol:g}")
        for row in rows:
            if not row["met"]:
                print(f"  n={row['n']:>8d}  no theta meets it")
                continue
            print(
                f"  n={row['n']:>8d}  theta={row['theta']:5.3f} "
                f"err={row['rel_error_vs_exact']:.2e}  "
                f"phi={row['phi_s'] * 1e3:9.2f} ms  "
                f"vgrad={row['value_and_grad_s'] * 1e3:9.2f} ms  "
                + (
                    f"exact={row['exact_s'] * 1e3:9.2f} ms"
                    if row["exact_s"]
                    else "exact=      n/a"
                )
            )
        for name, cross in schedules[f"{tol:g}"]["crossover"].items():
            print(f"  crossover ({name}): {cross['crossover_n']}")

    payload = {
        "benchmark": "svgd_accuracy_matched",
        "params": {
            "inputs": [run["path"] for run in runs],
            "thetas": [run["params"]["theta"] for run in runs],
            "tolerances": args.tolerances,
            "regime": {key: first["params"].get(key) for key in _REGIME_KEYS},
            "exact_max_n": [run["params"].get("exact_max_n") for run in runs],
            "comparisons": {
                name: " + ".join(fields) + " vs exact_s"
                for name, fields in _COMPARISONS.items()
            },
        },
        # The machine the inputs were measured on; this script measured nothing.
        "metadata": {key: first["metadata"].get(key) for key in _MACHINE_KEYS}
        | {
            "derived_by": "bench/svgd/accuracy_matched.py",
            "derived_at_commit": _git_commit(_REPO_ROOT),
            "derived_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "schedules": schedules,
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
