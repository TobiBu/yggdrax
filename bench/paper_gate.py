"""Yggdrax I's accuracy gate: a rerun may move a time, never an accuracy.

The Phase 2 run list (``bench/paper_runlist.sh``) reruns every JSON a figure
or a sentence quotes a *time* from. The performance work claims numerics
identical to ~1e-15 in float64, so wherever a rerun overlaps the committed file
its accuracy and structural fields must match; a change is a bug, not a result.
The run list calls this after each rerun and stops on the first mismatch.

Three checks:

* ``rerun KIND NEW --base REF [--path P]`` compares ``NEW`` with the file at
  ``P`` (default: ``NEW``'s own path) as committed at ``REF``, read with
  ``git show`` -- so the file a rerun overwrites is never its own reference;
* ``pilot NEW --pilot PILOT`` compares one theta of the accuracy-matched sweep
  with the CPU pilot that chose its grid -- error is hardware-independent;
* ``machine FILE ...`` checks Gate 2's metadata: one jax version, one device,
  one commit, x64 on.

Tolerances, and why:

* ``rel_error_vs_exact`` matches to 1e-6 relative or 1e-9 absolute, whichever
  is looser. The accumulation lanes (scatter, segment, the fused kernel) agree
  to ~1e-11 of the update, so they can move an error by ~1e-11 absolutely --
  which is most of a 1e-12 error and a millionth of a 1e-5 one; a changed tree
  moves it by order one. The 1e-9 floor is 100x the lane agreement;
* any other float64 accuracy field matches to 1e-6 relative;
* counts that describe the tree or the partition match exactly;
* series C's descent runs in float32, so its convergence fields get 1e-3.

Exit status 3 on any mismatch -- distinct from a traceback's 1 -- and 0
otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

_ERR_ABS = 1e-9
_REL64 = 1e-6
_REL32 = 1e-3


class _Report:
    """Collects comparisons and prints one line per field."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.failures = 0
        self.checked = 0

    def close(
        self, where: str, field: str, new: Any, old: Any, rel: float, abs_tol=0.0
    ) -> None:
        """Compare two floats to ``rel`` relative or ``abs_tol`` absolute."""
        self.checked += 1
        if new is None or old is None:
            ok = new is None and old is None
        else:
            scale = max(abs(new), abs(old))
            ok = abs(new - old) <= max(rel * scale, abs_tol)
        self._line(ok, where, field, new, old)

    def equal(self, where: str, field: str, new: Any, old: Any) -> None:
        """Compare two counts exactly."""
        self.checked += 1
        self._line(new == old, where, field, new, old)

    def note(self, text: str) -> None:
        print(f"  [note] {text}")

    def _line(self, ok: bool, where: str, field: str, new: Any, old: Any) -> None:
        if not ok:
            self.failures += 1
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {where:<28s} {field:<34s} new={new!r:<24} committed={old!r}")

    def finish(self) -> int:
        verdict = "PASS" if self.failures == 0 else "FAIL"
        print(
            f"[gate] {self.label}: {verdict} -- {self.checked} fields compared, "
            f"{self.failures} mismatched"
        )
        return 0 if self.failures == 0 else 3


def _committed(path: str, base: str) -> dict[str, Any]:
    """The JSON at repository path ``path`` as committed at ``base``."""
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "show", f"{base}:{Path(path).as_posix()}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _svgd_scaling(new: dict, old: dict, rep: _Report) -> None:
    old_by_n = {r["n"]: r for r in old["records"]}
    overlap = [r for r in new["records"] if r["n"] in old_by_n]
    if not overlap:
        rep.note("no overlapping N")
    for rec in overlap:
        prev = old_by_n[rec["n"]]
        where = f"n={rec['n']}"
        if "rel_error_vs_exact" in prev or "rel_error_vs_exact" in rec:
            rep.close(
                where,
                "rel_error_vs_exact",
                rec.get("rel_error_vs_exact"),
                prev.get("rel_error_vs_exact"),
                _REL64,
                _ERR_ABS,
            )
        for key in ("num_far_pairs", "num_near_leaf_pairs", "num_far_contribs"):
            if key in rec and key in prev:
                rep.equal(where, key, rec[key], prev[key])


def _diff_scaling(new: dict, old: dict, rep: _Report) -> None:
    old_by_n = {r["n"]: r for r in old["records"]}
    for rec in new["records"]:
        prev = old_by_n.get(rec["n"])
        if prev is None:
            continue
        for backend, vals in rec["backends"].items():
            before = prev["backends"].get(backend)
            if before is None:
                continue
            for key in ("num_nodes", "num_far_interactions", "num_near_neighbors"):
                if key in vals and key in before:
                    rep.equal(f"n={rec['n']} {backend}", key, vals[key], before[key])


def _corrfunc_scaling(new: dict, old: dict, rep: _Report) -> None:
    old_by_n = {r["n"]: r for r in old["records"]}
    for rec in new["records"]:
        prev = old_by_n.get(rec["n"])
        if prev is None:
            continue
        for key in ("num_far_pairs", "num_near_leaf_pairs", "max_leaf"):
            if key in rec and key in prev:
                rep.equal(f"n={rec['n']}", key, rec[key], prev[key])


_SERIES_C_INITIAL = ("initial_mean_nn", "initial_loss")
_SERIES_C_FINAL = (
    "final_loss",
    "final_mean_nn",
    "loss_ratio",
    "topology_changes",
    "slot_change_fraction_mean",
    "leaf_change_fraction_mean",
    "mean_abs_rank_shift_normalized",
)


def _series_c(new: dict, old: dict, rep: _Report) -> None:
    def key(r):
        return (r["num_particles"], r["seed"])

    old_by = {key(r): r for r in old["records"] if "summary" in r}
    for rec in new["records"]:
        prev = old_by.get(key(rec))
        if prev is None or "summary" not in rec:
            continue
        where = f"N={rec['num_particles']} seed={rec['seed']}"
        for field in ("learning_rate", "target_distance", "steps"):
            rep.close(where, field, rec[field], prev[field], _REL64)
        for field in _SERIES_C_INITIAL:
            rep.close(
                where, field, rec["summary"][field], prev["summary"][field], _REL64
            )
        for field in _SERIES_C_FINAL:
            rep.close(
                where, field, rec["summary"][field], prev["summary"][field], _REL32
            )
    stop_new = (new.get("sweep") or {}).get("stopped_early") or {}
    stop_old = (old.get("sweep") or {}).get("stopped_early") or {}
    rep.note(
        "range: stopped at N="
        f"{stop_new.get('at_num_particles')} (committed: "
        f"{stop_old.get('at_num_particles')}) -- a range change, not an accuracy one"
    )


def _bandwidth(new: dict, old: dict, rep: _Report) -> None:
    for field in ("params",):
        if new[field] != old[field]:
            rep.equal("params", field, new[field], old[field])
    for field in (
        "h_median",
        "h_learned",
        "target_std",
        "std_median_heuristic",
        "std_learned",
    ):
        a, b = new[field], old[field]
        if isinstance(a, list):
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                rep.close(f"{field}[{i}]", field, x, y, _REL64)
        else:
            rep.close("", field, a, b, _REL64)
    for field in ("h_history", "loss_history"):
        a, b = new[field], old[field]
        rep.equal("", f"len({field})", len(a), len(b))
        worst = max(
            range(min(len(a), len(b))),
            key=lambda i: abs(a[i] - b[i]) / max(abs(a[i]), abs(b[i]), 1e-300),
        )
        rep.close(f"worst step {worst}", field, a[worst], b[worst], _REL64)


_RERUN = {
    "svgd_scaling": _svgd_scaling,
    "diff_scaling": _diff_scaling,
    "corrfunc_scaling": _corrfunc_scaling,
    "series_c": _series_c,
    "bandwidth": _bandwidth,
}


def _pilot(new: dict, pilot: dict, rep: _Report) -> None:
    theta = new["params"]["theta"]
    by_n = {r["n"]: r for r in pilot["records"] if r["theta"] == theta}
    if not by_n:
        rep.note(f"the pilot has no theta={theta}")
        rep.failures += 1
    for rec in new["records"]:
        prev = by_n.get(rec["n"])
        if prev is None:
            continue
        where = f"theta={theta} n={rec['n']}"
        rep.close(
            where,
            "rel_error_vs_exact",
            rec.get("rel_error_vs_exact"),
            prev["rel_error_vs_exact"],
            _REL64,
            _ERR_ABS,
        )
        for key in ("num_far_pairs", "num_near_leaf_pairs", "num_far_contribs"):
            rep.equal(where, key, rec[key], prev[key])


def _machine(paths: list[str], rep: _Report) -> None:
    seen: dict[str, set] = {}
    for path in paths:
        meta = json.loads(Path(path).read_text())["metadata"]
        for key in ("jax_version", "device_kind", "git_commit"):
            seen.setdefault(key, set()).add(meta.get(key))
        rep.equal(Path(path).name, "x64_enabled", meta.get("x64_enabled"), True)
    for key, values in seen.items():
        rep.equal("all files", f"one {key}", len(values), 1)
        rep.note(f"{key}: {sorted(map(str, values))}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rerun")
    r.add_argument("kind", choices=sorted(_RERUN))
    r.add_argument("new")
    r.add_argument("--base", required=True, help="Commit the reference is read at.")
    r.add_argument(
        "--path", default=None, help="Repository path of the reference (default: NEW)."
    )
    q = sub.add_parser("pilot")
    q.add_argument("new")
    q.add_argument("--pilot", required=True)
    m = sub.add_parser("machine")
    m.add_argument("paths", nargs="+")
    args = p.parse_args()

    if args.cmd == "rerun":
        rep = _Report(f"{args.kind} {args.new} vs {args.base}")
        new = json.loads(Path(args.new).read_text())
        ref = args.path or str(Path(args.new).resolve().relative_to(_REPO_ROOT))
        _RERUN[args.kind](new, _committed(ref, args.base), rep)
    elif args.cmd == "pilot":
        rep = _Report(f"{args.new} vs {args.pilot}")
        new = json.loads(Path(args.new).read_text())
        _pilot(new, json.loads(Path(args.pilot).read_text()), rep)
    else:
        rep = _Report(f"machine ({len(args.paths)} files)")
        _machine(args.paths, rep)
    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
