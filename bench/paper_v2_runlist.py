"""Yggdrax I, paper v2: every single-GPU number, one command, one provenance.

The referee asked for one re-run on one jax version and one hardware type
instead of results mixed across versions and machines. This driver runs every
single-GPU bench paper v2 quotes, at jax 0.11.2 (CI's pin), from one clean
commit, and stamps each JSON with the same provenance record. The commands,
expected wall times and the environment recipe are in
``bench/paper_v2_runlist.md``. Multi-node scaling (C3) is ``bench/slurm/``.

Usage, from the repository root::

    python bench/paper_v2_runlist.py --list               # rows and their commands
    python bench/paper_v2_runlist.py --smoke              # every row at tiny N (GPU)
    python bench/paper_v2_runlist.py                      # the run
    python bench/paper_v2_runlist.py --rows '^b1_'        # a subset (regex on tags)
    python bench/paper_v2_runlist.py --smoke --platform cpu --out /tmp/rl   # laptop

Rows follow ``bench/slurm/``'s resumable convention. Every row writes its
outputs to ``$OUT/cases/<tag>/``, its log to ``$OUT/logs/<tag>.log`` and, on
success only, ``$OUT/rows/<tag>.json`` (command, wall time, provenance). A rerun
skips every row whose record exists; a failed row leaves ``rows/<tag>.failed``
and runs again next time. ``$OUT/campaign.json`` pins the commit and jax
version of the first run; a later run from another commit or jax refuses to
continue the same ``$OUT`` (start a new one or pass ``--allow-new-commit``).

GPU selection: every GPU row picks its own card(s) with ``autocvd`` right
before it starts (``--gpu-select free``, the org rule on the shared server: it
waits for idle cards), and the bench is passed the same ``--gpu-select``. Rows
marked optional (C2 on 2 and 4 GPUs) wait at most ``--optional-gpu-wait``
seconds and are skipped if the cards are busy; a rerun tries again. If
``CUDA_VISIBLE_DEVICES`` is set, autocvd is not called and rows use those cards.

Provenance: each JSON a row writes gets a top-level ``"runlist"`` block (tag,
arm, walk/backend, jax and jaxlib versions, backend, device kind, the cards'
names and UUIDs, commit, dirty flag, command, XLA flags, wall time). At the end
the driver checks that every finished row shares one jax version, one device
kind and one commit, writes ``$OUT/summary.json`` and exits non-zero on a
failure or a mismatch.

This script never imports JAX itself, so it does not hold a GPU between rows.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXPECT_JAX = "0.11.2"

#: 2^11 .. 2^26: the cell-leaf occupancy is a sawtooth in N (one Morton level per
#: factor 8), so N = 2^(3k+2) are the equal-occupancy sizes of fig_scaling.
POW2 = [str(1 << e) for e in range(11, 27)]


@dataclasses.dataclass(frozen=True)
class Row:
    """One command of the run list.

    ``cmd`` is the full GPU run; ``smoke`` is appended in ``--smoke`` mode (a
    repeated option overrides the earlier one, so these set the tiny sizes), and
    ``cpu`` is appended on ``--platform cpu``. Tokens: ``{py}``, ``{case}`` (the
    row's output directory), ``{platform}`` (gpu|cpu) and ``{gpusel}``.
    """

    tag: str
    arm: str
    walk: str
    cmd: tuple[str, ...]
    smoke: tuple[str, ...] = ()
    cpu: tuple[str, ...] = ()
    ngpu: int = 1
    optional: bool = False
    gpu_only: bool = False
    requires: str | None = None
    env: tuple[tuple[str, str], ...] = ()
    est_gpu_min: float = 0.0
    note: str = ""


def _rows() -> list[Row]:
    diff, knn = "bench/differentiability", "bench/knn_stats"
    fast_walk = "static cell-leaf radix build + dual_tree_walk_mutual (flat, pallas)"
    traced = "traced_soft_pair_counts: static cells + flat mutual walk"
    dist = "distributed stats: SFC partition + halo + mutual walk, exchange=auto"
    g = ("--gpu-select", "{gpusel}")
    a5 = ("--platform", "{platform}", *g)
    tiny = ("--runs", "1", "--warmup", "1")
    # fmt: off
    rows = [
        # -- A1: tree build + traversal scaling, and the autodiff overhead ----------
        Row(
            "a1_scaling",
            "A1",
            "legacy adaptive build + eager dual walk (radix/octree/kdtree); "
            f"{fast_walk}; cKDTree CPU anchor",
            ("{py}", f"{diff}/scaling.py", "--sizes", "1000", "10000", "100000",
             "1000000", "--walk-backends", "flat", "pallas", "--runs", "5",
             "--warmup", "2", *g, "--output", "{case}/scaling.json"),
            smoke=("--sizes", "2000", *tiny),
            cpu=("--pallas-interpret",),
            est_gpu_min=25,
        ),
        Row(
            "a1_scaling_fast",
            "A1",
            fast_walk,
            ("{py}", f"{diff}/scaling.py", "--no-legacy", "--no-scipy", "--sizes",
             *POW2, "--walk-backends", "flat", "pallas", "--runs", "5", "--warmup",
             "2", *g, "--output", "{case}/scaling_fast.json"),
            smoke=("--sizes", "2048", "16384", *tiny),
            cpu=("--pallas-interpret",),
            est_gpu_min=45,
            note="2^26 = 6.7e7 was the 2026-11-07 ceiling on an A100-40GB",
        ),
        Row(
            "a1_autodiff_overhead",
            "A1",
            "build_kdtree + query_neighbors(backend='tree'), value vs grad",
            ("{py}", f"{diff}/autodiff_overhead.py", "--sizes", "1000", "10000",
             "100000", "1000000", "--query-backend", "tree", *g, "--output",
             "{case}/autodiff_overhead.json"),
            smoke=("--sizes", "2000", *tiny),
            est_gpu_min=5,
        ),
        # -- A5: gradient accuracy, switch rates, leapfrog energy (GPU sizes of
        #    docs/differentiating_a_tree_code.md) ------------------------------------
        Row(
            "a5_gradient_accuracy",
            "A5",
            f"{traced}; window/geometric, hard and smooth; kNN: kdtree tree",
            ("{py}", f"{diff}/gradient_accuracy.py", *a5, "--ns",
             "4000,16000,64000", "--out", "{case}/gradient_accuracy.json", "--fig",
             "{case}/gradient_accuracy.png"),
            smoke=("--ns", "500", "--thetas", "0.5", "--epss", "1e-3", "--knn-n",
                   "300"),
            est_gpu_min=40,
            note="exact reference is O(N^2) autodiff; 64000 is the slow point",
        ),
        Row(
            "a5_switch_rates",
            "A5",
            f"{traced}; geometric/window, hard and smooth",
            ("{py}", f"{diff}/switch_rates.py", *a5, "--n", "16000", "--out",
             "{case}/switch_rates.json", "--fig", "{case}/switch_rates.png"),
            smoke=("--n", "300", "--steps", "6"),
            est_gpu_min=30,
        ),
        Row(
            "a5_hmc_energy",
            "A5",
            f"{traced}; exact, hard/smooth geometric, smooth + membership, window",
            ("{py}", f"{diff}/hmc_energy.py", *a5, "--n", "8000", "--out",
             "{case}/hmc_energy.json", "--fig", "{case}/hmc_energy.png"),
            smoke=("--n", "300", "--starts", "2", "--step-factors", "1",
                   "--length", "8"),
            est_gpu_min=30,
        ),
        # -- B1: kNN-CDF scaling, open and periodic ----------------------------------
        Row(
            "b1_knn_cdf_open",
            "B1",
            "knn_stats.knn_cdf: build_kdtree + query_neighbors(backend='tree')",
            ("{py}", f"{knn}/scaling.py", "--sizes", "1000", "10000", "100000",
             "1000000", "10000000", "--brute-force-max-n", "100000", *g,
             "--output", "{case}/knn_cdf_open.json"),
            smoke=("--sizes", "1000", "4000", *tiny, "--brute-force-max-n", "4000"),
            est_gpu_min=20,
        ),
        Row(
            "b1_knn_cdf_periodic",
            "B1",
            "knn_stats.knn_cdf(box_size=1): kdtree tree search, minimum image",
            ("{py}", f"{knn}/scaling.py", "--periodic", "--sizes", "1000", "10000",
             "100000", "1000000", "10000000", "--brute-force-max-n", "100000", *g,
             "--output", "{case}/knn_cdf_periodic.json"),
            smoke=("--sizes", "1000", "4000", *tiny, "--brute-force-max-n", "4000"),
            est_gpu_min=30,
        ),
        # -- B2: periodic xi showcase. SLOT: the benches live on claude/b2-xi
        #    (06c953a), not merged yet; the rows run once the files exist. CLI
        #    taken from that branch; the B2 owner should confirm the GPU sizes. ------
        Row(
            "b2_xi_box",
            "B2",
            "traced_soft_pair_counts(box_size=L), natural estimator; "
            "distributed (1 dev); TreeCorr; Corrfunc",
            ("{py}", "bench/corrfunc/xi_box.py", *a5, "--sizes", "10000", "100000",
             "1000000", "--out", "{case}/xi_box.json", "--fig",
             "{case}/xi_box.png"),
            smoke=("--sizes", "2000", "--repeats", "1"),
            requires="bench/corrfunc/xi_box.py",
            est_gpu_min=30,
            note="B2 SLOT: activates when claude/b2-xi is merged",
        ),
        Row(
            "b2_acceptance_rules",
            "B2",
            "traced pair counts: window eps sweep vs geometric theta sweep",
            ("{py}", "bench/corrfunc/acceptance_rules.py", *a5, "--out",
             "{case}/acceptance_rules.json", "--fig", "{case}/acceptance_rules.png"),
            smoke=("--n", "3000"),
            requires="bench/corrfunc/acceptance_rules.py",
            est_gpu_min=15,
            note="B2 SLOT: activates when claude/b2-xi is merged",
        ),
        # -- B3: point-process forward model, Fisher forecast (docs/point_process.md)
        Row(
            "b3_forecast",
            "B3",
            "distributed_pair_counts / distributed_knn on 1 device (ξ, kNN-CDF)",
            ("{py}", "bench/point_process/forecast.py", *a5, "--n-parents", "2048",
             "--n-children", "16", "--sigma", "0.01", "--n-query-side", "32",
             "--n-cov", "400", "--n-jac", "64", "--n-fit", "50", "--batch-size",
             "8", "--out-dir", "{case}"),
            smoke=("--n-parents", "48", "--n-children", "12", "--sigma", "0.03",
                   "--nbins", "6", "--n-r-knn", "4", "--ks", "1,2",
                   "--n-query-side", "8", "--n-cov", "24", "--n-jac", "3",
                   "--n-fd-jac", "1", "--n-val-alt", "3", "--n-fit", "2",
                   "--fit-steps", "4", "--fd-rel", "1e-2,1e-5",
                   "--batch-size", "1"),
            est_gpu_min=60,
            note="N = 32768; CPU reference at N = 2048 is bench/point_process/results",
        ),
        # -- C2: gradient memory of the distributed statistics, 1/2/4 GPUs ------------
        Row(
            "c2_grad_memory_g1",
            "C2",
            dist,
            ("{py}", "bench/distributed/grad_memory.py", *a5, "--ndev", "1", "--n",
             "1048576", "--out", "{case}/grad_memory_g1.json"),
            smoke=("--n", "1024", "--repeats", "1", "--dtypes", "float64"),
            est_gpu_min=10,
        ),
        Row(
            "c2_grad_memory_g2",
            "C2",
            dist,
            ("{py}", "bench/distributed/grad_memory.py", *a5, "--ndev", "2", "--n",
             "1048576", "--out", "{case}/grad_memory_g2.json"),
            smoke=("--n", "1024", "--repeats", "1", "--dtypes", "float64"),
            ngpu=2,
            optional=True,
            est_gpu_min=10,
        ),
        Row(
            "c2_grad_memory_g4",
            "C2",
            dist,
            ("{py}", "bench/distributed/grad_memory.py", *a5, "--ndev", "4", "--n",
             "1048576", "--out", "{case}/grad_memory_g4.json"),
            smoke=("--n", "1024", "--repeats", "1", "--dtypes", "float64"),
            ngpu=4,
            optional=True,
            est_gpu_min=10,
        ),
        Row(
            "c2_native_ragged_check",
            "C2",
            "distributed ξ/kNN, exchange=native vs buf, forward after gradient",
            ("{py}", "-m", "pytest", "-q", "-o", "addopts=", "-p",
             "no:cacheprovider", "tests/distributed/test_distributed_stats_grad.py",
             "-k", "test_native_ragged_exchange_survives_a_gradient", "-rA",
             "--junitxml", "{case}/junit.xml"),
            ngpu=2,
            optional=True,
            gpu_only=True,
            env=(("YGGDRAX_CHECK_UPSTREAM_RAGGED_FIX", "1"),),
            est_gpu_min=3,
            note="opt-in upstream check; needs >= 2 GPUs",
        ),
        # -- competitor baselines (new) -------------------------------------------------
        Row(
            "base_knn_open",
            "baseline",
            "yggdrax kdtree tree search vs jaxkd, jaxkd-cuda, JZ-Tree, cKDTree",
            ("{py}", "bench/baselines/knn_forward.py", "--sizes", "10000", "100000",
             "1000000", "10000000", "30000000", "--ks", "8", "32", *g, "--output",
             "{case}/knn_forward_open.json"),
            smoke=("--smoke",),
            est_gpu_min=30,
        ),
        Row(
            "base_knn_periodic",
            "baseline",
            "yggdrax kdtree tree search (box) vs JZ-Tree (boxsize), cKDTree",
            ("{py}", "bench/baselines/knn_forward.py", "--periodic", "--sizes",
             "10000", "100000", "1000000", "10000000", "30000000", "--ks", "8",
             "32", *g, "--output", "{case}/knn_forward_periodic.json"),
            smoke=("--smoke",),
            est_gpu_min=30,
        ),
        Row(
            "base_kernel_sums_uniform",
            "baseline",
            f"{traced} (window eps 1e-6) vs KeOps and dense JAX brute force",
            ("{py}", "bench/baselines/kernel_sums.py", "--ic", "uniform", "--sizes",
             "10000", "30000", "100000", "300000", "1000000", *g, "--output",
             "{case}/kernel_sums_uniform.json"),
            smoke=("--smoke",),
            est_gpu_min=25,
        ),
        Row(
            "base_kernel_sums_clustered",
            "baseline",
            f"{traced} (window eps 1e-6) vs KeOps and dense JAX brute force",
            ("{py}", "bench/baselines/kernel_sums.py", "--ic", "clustered",
             "--sizes", "10000", "30000", "100000", "300000", "1000000", *g,
             "--output", "{case}/kernel_sums_clustered.json"),
            smoke=("--smoke",),
            est_gpu_min=25,
        ),
    ]
    # fmt: on
    return rows


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def _git_info() -> dict[str, Any]:
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_files": dirty.splitlines(),
    }


_PROBE = r"""
import json, os, sys
import jax, jaxlib
sys.path.insert(0, sys.argv[1])
import yggdrax
devs = jax.devices()
print("PROBE=" + json.dumps({
    "jax_version": jax.__version__,
    "jaxlib_version": jaxlib.__version__,
    "backend": jax.default_backend(),
    "device_kind": devs[0].device_kind,
    "devices": [str(d) for d in devs],
    "num_devices": len(devs),
    "python": sys.version.split()[0],
    "yggdrax_file": os.path.realpath(yggdrax.__file__),
}))
"""


def _probe(py: str, env: dict[str, str]) -> dict[str, Any]:
    out = subprocess.run(
        [py, "-c", _PROBE, str(REPO)], capture_output=True, text=True, env=env
    )
    for line in out.stdout.splitlines():
        if line.startswith("PROBE="):
            return json.loads(line[len("PROBE=") :])
    raise RuntimeError(f"jax probe failed:\n{out.stdout}\n{out.stderr}")


def _nvidia_smi(cvd: str | None) -> list[dict[str, str]] | None:
    fields = "index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu"
    cmd = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    if cvd:
        cmd += ["-i", cvd]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    keys = fields.split(",")
    return [
        dict(zip(keys, (v.strip() for v in line.split(","))))
        for line in out.stdout.strip().splitlines()
    ]


_PICK = r"""
import os, sys
from autocvd import autocvd
n, least, timeout = int(sys.argv[1]), sys.argv[2] == "1", float(sys.argv[3])
try:
    autocvd(num_gpus=n, least_used=least, timeout=None if timeout <= 0 else int(timeout),
            progress=False)
except TimeoutError:
    print("PICK=timeout"); sys.exit(0)
print("PICK=" + os.environ["CUDA_VISIBLE_DEVICES"])
print("ORDER=" + os.environ.get("CUDA_DEVICE_ORDER", ""))
"""


def _pick_gpus(py: str, n: int, least_used: bool, timeout: float):
    """``(CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER)`` from autocvd, or ``None``."""
    out = subprocess.run(
        [py, "-c", _PICK, str(n), "1" if least_used else "0", str(timeout)],
        capture_output=True,
        text=True,
    )
    pick = order = None
    for line in out.stdout.splitlines():
        if line.startswith("PICK="):
            pick = line[5:]
        elif line.startswith("ORDER="):
            order = line[6:]
    if pick is None:
        raise RuntimeError(f"autocvd failed:\n{out.stdout}\n{out.stderr}")
    return None if pick == "timeout" else (pick, order or "")


def _expand(row: Row, args, case: Path) -> list[str]:
    parts = list(row.cmd)
    if args.smoke:
        parts += row.smoke
    if args.platform == "cpu":
        parts += row.cpu
    subs = {
        "{py}": args.python,
        "{case}": str(case),
        "{platform}": args.platform,
        # on CPU no bench may call autocvd (it raises without nvidia-smi)
        "{gpusel}": args.gpu_select if args.platform == "gpu" else "none",
    }
    out = []
    for p in parts:
        for k, v in subs.items():
            p = p.replace(k, v)
        out.append(p)
    return out


def _stamp(case: Path, block: dict[str, Any]) -> list[str]:
    """Add ``block`` as ``"runlist"`` to every JSON object under ``case``."""
    stamped = []
    for path in sorted(case.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            data["runlist"] = block
            path.write_text(json.dumps(data, indent=1, default=str))
        else:  # a list payload: provenance next to it
            path.with_suffix(".runlist.json").write_text(json.dumps(block, indent=1))
        stamped.append(str(path.relative_to(case)))
    return stamped


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def _parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--smoke", action="store_true", help="every row at tiny N")
    p.add_argument("--platform", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--out", default=None, help="campaign directory (see --list)")
    p.add_argument("--rows", default=".", help="regex: run the rows whose tag matches")
    p.add_argument("--skip-rows", default=None, help="regex: leave these rows out")
    p.add_argument("--python", default=sys.executable, help="interpreter for the rows")
    p.add_argument(
        "--gpu-select",
        choices=("free", "least-used"),
        default="free",
        help="autocvd mode for every GPU row (org rule: free)",
    )
    p.add_argument(
        "--gpu-wait",
        type=float,
        default=0,
        help="seconds a required row waits for free cards (0: forever)",
    )
    p.add_argument(
        "--optional-gpu-wait",
        type=float,
        default=600,
        help="seconds an optional multi-GPU row waits before it is skipped",
    )
    p.add_argument("--expect-jax", default=EXPECT_JAX)
    p.add_argument("--allow-jax-mismatch", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--allow-new-commit", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--list", action="store_true", help="print the rows and exit")
    p.add_argument("--dry-run", action="store_true", help="print commands only")
    args = p.parse_args(argv)
    if args.out is None:
        base = Path(os.environ.get("WORKSPACE", os.path.expanduser("~")))
        name = "yggdrax-paper-v2-runlist"
        if args.smoke:
            name += f"-smoke-{args.platform}"
        elif args.platform == "cpu":
            name += "-cpu"
        args.out = str(base / name)
    return args


def _print_list(rows: list[Row], args) -> None:
    total = 0.0
    for r in rows:
        flags = []
        if r.ngpu > 1:
            flags.append(f"{r.ngpu} GPUs")
        if r.optional:
            flags.append("optional")
        if r.gpu_only:
            flags.append("GPU only")
        if r.requires and not (REPO / r.requires).exists():
            flags.append(f"waiting for {r.requires}")
        total += r.est_gpu_min
        print(f"{r.tag:28s} {r.arm:9s} ~{r.est_gpu_min:4.0f} min  {', '.join(flags)}")
        print(f"    walk: {r.walk}")
        if r.note:
            print(f"    note: {r.note}")
        envs = " ".join(f"{k}={v}" for k, v in r.env)
        cmd = shlex.join(_expand(r, args, Path(args.out) / "cases" / r.tag))
        print(f"    {envs + ' ' if envs else ''}{cmd}")
    print(
        f"\n{len(rows)} rows, ~{total / 60:.1f} h on one A100 (rough; see the README)"
    )


def main(argv=None) -> int:
    args = _parse(argv)
    rows = [
        r
        for r in _rows()
        if re.search(args.rows, r.tag)
        and not (args.skip_rows and re.search(args.skip_rows, r.tag))
    ]
    if args.list or args.dry_run:
        _print_list(rows, args)
        return 0

    out = Path(args.out)
    for sub in ("rows", "logs", "cases"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)

    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + ([base_env["PYTHONPATH"]] if base_env.get("PYTHONPATH") else [])
    )
    base_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # peak = working set; shared
    base_env["PYTHONUNBUFFERED"] = "1"
    if args.platform == "cpu":
        base_env["JAX_PLATFORMS"] = "cpu"
    user_cvd = (
        os.environ.get("CUDA_VISIBLE_DEVICES") if args.platform == "gpu" else None
    )

    # -- preflight: one jax, this checkout, a clean tree, the GPU ---------------------
    git = _git_info()
    probe_env = dict(base_env)
    if args.platform == "gpu" and not user_cvd:
        picked = _pick_gpus(args.python, 1, True, 60)  # least used: just to look
        if picked:
            probe_env["CUDA_VISIBLE_DEVICES"] = picked[0]
    probe = _probe(args.python, probe_env)
    print(f"[runlist] {_now()}  out {out}")
    print(
        f"[runlist] jax {probe['jax_version']} jaxlib {probe['jaxlib_version']} "
        f"backend {probe['backend']} device {probe['device_kind']}"
    )
    print(
        f"[runlist] yggdrax {git['commit'][:10]} ({git['branch']})"
        f"{' DIRTY' if git['dirty'] else ''} from {probe['yggdrax_file']}"
    )
    problems = []
    if probe["jax_version"] != args.expect_jax and not args.allow_jax_mismatch:
        problems.append(f"jax {probe['jax_version']} != {args.expect_jax}")
    if not probe["yggdrax_file"].startswith(str(REPO) + os.sep):
        problems.append("yggdrax is not imported from this checkout")
    if args.platform == "gpu" and probe["backend"] != "gpu":
        problems.append(f"--platform gpu but jax's backend is {probe['backend']}")
    if git["dirty"] and not (args.allow_dirty or args.smoke):
        problems.append("the worktree is not clean (commit first, or --allow-dirty)")
    campaign_path = out / "campaign.json"
    campaign = {
        "started": _now(),
        "git": git,
        "jax_version": probe["jax_version"],
        "jaxlib_version": probe["jaxlib_version"],
        "device_kind": probe["device_kind"],
        "platform": args.platform,
        "smoke": args.smoke,
        "hostname": os.uname().nodename,
        "python": probe["python"],
        "argv": sys.argv,
    }
    if campaign_path.exists():
        prev = json.loads(campaign_path.read_text())
        same = (
            prev["git"]["commit"] == git["commit"]
            and prev["jax_version"] == probe["jax_version"]
        )
        if not same and not args.allow_new_commit:
            problems.append(
                f"{out} holds a campaign at {prev['git']['commit'][:10]} / jax "
                f"{prev['jax_version']}; use a new --out or --allow-new-commit"
            )
        campaign = prev
    if problems:
        print("[runlist] refusing to start: " + "; ".join(problems), file=sys.stderr)
        return 2
    if git["dirty"]:
        print("[runlist] WARNING: dirty worktree (allowed for --smoke/--allow-dirty)")
    if not campaign_path.exists():
        _write(campaign_path, campaign)
        freeze = subprocess.run(
            [args.python, "-m", "pip", "freeze"], capture_output=True, text=True
        ).stdout
        (out / "pip-freeze.txt").write_text(freeze)
        smi = _nvidia_smi(None)
        if smi is not None:
            _write(out / "nvidia-smi.json", smi)

    # -- the rows -------------------------------------------------------------------------
    status: dict[str, str] = {}
    probes: dict[str, dict[str, Any]] = {}
    for row in rows:
        tag = row.tag
        rec_path, fail_path = (
            out / "rows" / f"{tag}.json",
            out / "rows" / f"{tag}.failed",
        )
        if rec_path.exists():
            status[tag] = "done"
            print(f"[runlist] skip {tag} (done)")
            continue
        if row.requires and not (REPO / row.requires).exists():
            status[tag] = f"skipped: {row.requires} not on this branch"
            print(
                f"[runlist] skip {tag}: {row.requires} not on this branch ({row.note})"
            )
            continue
        if row.gpu_only and args.platform != "gpu":
            status[tag] = "skipped: GPU only"
            print(f"[runlist] skip {tag}: GPU only")
            continue

        env = dict(base_env)
        env.update(dict(row.env))
        cvd = None
        if args.platform == "gpu":
            if user_cvd:
                cards = [c for c in user_cvd.split(",") if c.strip()]
                if len(cards) < row.ngpu:
                    status[tag] = (
                        f"skipped: needs {row.ngpu} GPUs, {len(cards)} visible"
                    )
                    print(f"[runlist] skip {tag}: {status[tag][9:]}")
                    continue
                cvd = ",".join(cards[: row.ngpu])
            else:
                wait = args.optional_gpu_wait if row.optional else args.gpu_wait
                print(
                    f"[runlist] {tag}: autocvd {args.gpu_select} x{row.ngpu} ...",
                    flush=True,
                )
                picked = _pick_gpus(
                    args.python, row.ngpu, args.gpu_select == "least-used", wait
                )
                if picked is None:
                    status[tag] = (
                        f"skipped: {row.ngpu} free GPUs not found in {wait:.0f} s"
                    )
                    print(f"[runlist] skip {tag}: {status[tag][9:]}; rerun later")
                    continue
                cvd, order = picked
                if order:
                    env["CUDA_DEVICE_ORDER"] = order
            env["CUDA_VISIBLE_DEVICES"] = cvd
            if row.ngpu > 1:
                # the native ragged all-to-all over NCCL (bench/slurm/stats_scaling_common.sh)
                env["XLA_FLAGS"] = (
                    env.get("XLA_FLAGS", "")
                    + " --xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false"
                ).strip()

        case = out / "cases" / tag
        case.mkdir(parents=True, exist_ok=True)
        cmd = _expand(row, args, case)
        key = f"{cvd}|{env.get('XLA_FLAGS', '')}"
        if key not in probes:
            probes[key] = _probe(args.python, env)
        gpus = _nvidia_smi(cvd) if cvd else None
        log = out / "logs" / f"{tag}.log"
        envs = " ".join(f"{k}={v}" for k, v in row.env)
        print(f"[runlist] === {tag}  {_now()}  CVD={cvd or '-'}", flush=True)
        print(f"    {envs + ' ' if envs else ''}{shlex.join(cmd)}", flush=True)
        t0 = time.time()
        with log.open("w") as fh:
            fh.write(
                f"# {_now()} CUDA_VISIBLE_DEVICES={cvd} XLA_FLAGS="
                f"{env.get('XLA_FLAGS', '')}\n# {shlex.join(cmd)}\n"
            )
            fh.flush()
            rc = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=env
            ).returncode
        wall = time.time() - t0
        block = {
            "tag": tag,
            "arm": row.arm,
            "walk": row.walk,
            "smoke": args.smoke,
            "platform": args.platform,
            **{
                k: probes[key][k]
                for k in (
                    "jax_version",
                    "jaxlib_version",
                    "backend",
                    "device_kind",
                    "devices",
                    "num_devices",
                )
            },
            "gpus": gpus,
            "cuda_visible_devices": cvd,
            "xla_flags": env.get("XLA_FLAGS", ""),
            "env": dict(row.env),
            "git_commit": git["commit"],
            "git_branch": git["branch"],
            "git_dirty": git["dirty"],
            "command": cmd,
            "started": datetime.datetime.fromtimestamp(t0)
            .astimezone()
            .isoformat(timespec="seconds"),
            "wall_s": round(wall, 1),
            "hostname": os.uname().nodename,
        }
        if rc == 0:
            stamped = _stamp(case, block)
            _write(rec_path, {**block, "outputs": stamped, "log": str(log)})
            fail_path.unlink(missing_ok=True)
            status[tag] = "ok"
            print(f"[runlist]     ok in {wall:.0f} s; stamped {len(stamped)} JSON")
        else:
            _write(fail_path, {**block, "returncode": rc, "log": str(log)})
            status[tag] = f"FAILED rc={rc}"
            tail = log.read_text().splitlines()[-15:]
            print(f"[runlist]     FAILED rc={rc} in {wall:.0f} s (log {log}):")
            print("\n".join("      | " + line for line in tail))
            if args.stop_on_failure:
                break

    # -- one machine, one jax, one commit ---------------------------------------------------
    records = {}
    for path in sorted((out / "rows").glob("*.json")):
        records[path.stem] = json.loads(path.read_text())
    mismatch, warnings = [], []
    for field in ("jax_version", "device_kind", "git_commit"):
        values = sorted({str(r.get(field)) for r in records.values()})
        if len(values) > 1:
            # --allow-new-commit (e.g. B2's rows after its merge) accepts a
            # second commit, recorded per row; jax and hardware must still match
            if field == "git_commit" and args.allow_new_commit:
                warnings.append(f"{field}: {values}")
            else:
                mismatch.append(f"{field}: {values}")
    summary = {
        "finished": _now(),
        "campaign": campaign,
        "status": status,
        "rows": {
            t: {
                k: r.get(k)
                for k in (
                    "arm",
                    "wall_s",
                    "jax_version",
                    "device_kind",
                    "git_commit",
                    "cuda_visible_devices",
                    "outputs",
                )
            }
            for t, r in records.items()
        },
        "provenance_mismatch": mismatch,
        "provenance_warnings": warnings,
    }
    _write(out / "summary.json", summary)
    print(f"\n[runlist] summary ({out / 'summary.json'}):")
    for tag, st in status.items():
        wall = records.get(tag, {}).get("wall_s")
        print(f"  {tag:28s} {st}{f'  ({wall:.0f} s)' if wall is not None else ''}")
    failed = [t for t, s in status.items() if s.startswith("FAILED")]
    if warnings:
        print("[runlist] WARNING, more than one commit: " + "; ".join(warnings))
    if mismatch:
        print("[runlist] PROVENANCE MISMATCH across rows: " + "; ".join(mismatch))
        return 4
    if failed:
        print(f"[runlist] {len(failed)} row(s) failed: {' '.join(failed)}")
        return 1
    print(
        f"[runlist] all finished rows share jax {probe['jax_version']}, "
        f"{probe['device_kind']}, commit {git['commit'][:10]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
