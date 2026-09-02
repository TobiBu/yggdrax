"""Gradients across tree rebuilds: nearest-neighbour spacing under descent.

Existence-proof experiment for the differentiability model (paper Sect. 2): the
tree is rebuilt *inside* the differentiated objective at every optimizer step,
yet reverse-mode gradients drive the system to a target nearest-neighbour
spacing across many rebuilds --- including steps on which the discrete tree
ordering changes.

The objective pushes each particle's nearest-neighbour distance toward a target
``r*``::

    L(x) = mean_i ( d_i(x) - r* )^2 ,

where the neighbour candidates of particle ``i`` are read from the *rebuilt*
tree's Morton ordering (the sorted neighbours at offsets +/-1..k). ``build_tree``
returns an integer ordering, so autodiff assigns it zero cotangent: the gradient
that flows is the gradient of the smooth geometry conditioned on the topology the
rebuild just produced. We step plain gradient descent, rebuild every step, and
record the loss/spacing trajectory together with the number of steps on which
the ordering (topology) actually changed.

What the switch counter is and is not
-------------------------------------
The counter compares the Morton **ordering** between consecutive steps, i.e. it
answers "did the permutation change?". A permutation is one component of a
topology, not the whole of it: a reordering can occur with no change in which
pairs actually interact. The rate reported here is therefore an
*upper-bound-shaped* signal for interaction-list change, and must not be plotted
beside an interaction-switch curve as though the two measured the same quantity.
Every results payload carries ``switch_metric = "morton_ordering"`` to keep that
distinction attached to the numbers.

The counter is also a frequency of switch events *along an optimization path*.
It is not an estimate of the measure of the switching set, and
``docs/differentiability_model.md``'s measure-zero statement remains a reasoned
argument rather than a measurement.

Modes
-----
Single run (default) --- one (N, seed), full trajectory plus a position history
for the figure:

  * summary + scalar trajectories -> ``results/differentiability/nn_rebuild.json``
  * position history (for the figure) -> ``results/differentiability/nn_rebuild.npz``

Scale-up sweep (``--sweep-n``) --- many N x seeds, switch statistics per N, no
position history:

  * per-run records -> ``results/differentiability/nn_rebuild_scaling.json``

Production (GPU server):

    micromamba run -n odisseo python bench/differentiability/nn_rebuild.py \
        --num-particles 4096 --steps 200 --gpu-select free

Scale-up sweep (GPU server):

    python bench/differentiability/nn_rebuild.py --gpu-select free \
        --sweep-n 256 1024 4096 16384 65536 --sweep-seeds 0 1 2

Local smoke (CPU):

    conda run -n jaccpot python bench/differentiability/nn_rebuild.py --smoke
    conda run -n jaccpot python bench/differentiability/nn_rebuild.py --smoke-sweep
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.differentiability._common import (
    dump_json,
    dump_npz,
    run_metadata,
    select_free_gpu,
)

_SINGLE_OUTPUT = "results/differentiability/nn_rebuild.json"
_SWEEP_OUTPUT = "results/differentiability/nn_rebuild_scaling.json"

#: What the per-step switch counter actually compares. Carried into every
#: payload so a number lifted out of the JSON keeps its label.
SWITCH_METRIC = "morton_ordering"

#: Limits of the claim these runs support, recorded next to the numbers rather
#: than only in a commit message.
LIMITATIONS: dict[str, str] = {
    "objective": (
        "Geometric: mean nearest-neighbour spacing driven to a target r*. This is "
        "not an inference or likelihood objective, and scaling N does not make it "
        "one. Whatever N is reached, the supported claim stays 'gradient descent "
        "through a per-step-rebuilt tree converges', not 'high-dimensional "
        "inference through rebuilds converges'."
    ),
    "optimizer": (
        "Plain gradient descent by design -- the experiment is an existence proof, "
        "so no adaptive optimizer is substituted to improve the curves. An Adam "
        "run would be a separate series, not a point on this one."
    ),
    "switch_metric": (
        "morton_ordering: the fraction of steps on which the Morton *ordering* "
        "changed. That is a permutation-switch rate, one component of a topology. "
        "A reordering can occur with no change in which pairs interact, so this "
        "rate is >= the rate of interaction-list change. It is an "
        "upper-bound-shaped signal and must not be drawn beside an "
        "interaction-switch curve (e.g. jaccpot's fig19) as the same quantity; "
        "jaccpot/mutual/identity.py separates permutation / tree_shape / "
        "leaf_partition / node_ranges / far_pairs / near_pairs precisely because "
        "they differ. That facility is deliberately not used here: the dependency "
        "edge runs jaccpot -> yggdrax, and yggdrax has no MutualTopology to "
        "fingerprint."
    ),
    "budget_scaling": (
        "The objective is a *mean* over particles, so the per-particle gradient "
        "measured on this problem scales as N^(-4/3) (1/N from the mean, "
        "N^(-1/3) from r*), while the distance each particle must travel scales "
        "as N^(-1/3). Progress per step under a fixed learning rate therefore "
        "scales as 1/N: a fixed (lr, steps) budget is comparable in *cost* across "
        "N but not in *optimizer progress*. A fixed-lr switch-rate curve thus "
        "mixes the switch rate with how far along the path each run got, and is "
        "not an intrinsic N-dependence of rebuild switching. The n-scaled lr mode "
        "(lr proportional to N) holds the descent dynamics N-invariant; it is a "
        "separate series, still plain gradient descent."
    ),
    "not_a_measure_estimate": (
        "These runs measure the frequency of switch events along an optimization "
        "path. They do not estimate the measure of the switching set and must not "
        "be described as having measured it. The measure-zero statement in "
        "docs/differentiability_model.md is a reasoned argument and stays one."
    ),
}


@dataclass(frozen=True)
class DescentConfig:
    """One (N, seed) descent configuration.

    Parameters
    ----------
    num_particles
        Particle count ``N``; the free-parameter count is ``N * dim``.
    dim
        Spatial dimension.
    target_distance
        Target nearest-neighbour spacing ``r*``.
    k_neighbors
        Morton-ordering offsets ``+/-1..k`` used as neighbour candidates.
    leaf_size
        Maximum particles per leaf passed to ``build_tree``.
    steps
        Gradient-descent steps.
    learning_rate
        Gradient-descent learning rate.
    clip
        Positions are clipped to ``[-clip, clip]`` after each step.
    seed
        PRNG seed for the initial positions.
    """

    num_particles: int
    dim: int
    target_distance: float
    k_neighbors: int
    leaf_size: int
    steps: int
    learning_rate: float
    clip: float
    seed: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-particles", type=int, default=256)
    p.add_argument("--dim", type=int, default=3, help="Spatial dimension.")
    p.add_argument("--target-distance", type=float, default=0.12)
    p.add_argument("--k-neighbors", type=int, default=8)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=2.0)
    p.add_argument(
        "--clip",
        type=float,
        default=1.5,
        help="Positions are clipped to [-clip, clip] each step.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            f"Results JSON. Defaults to {_SINGLE_OUTPUT} for a single run and "
            f"{_SWEEP_OUTPUT} for a sweep."
        ),
    )
    p.add_argument("--smoke", action="store_true")

    sweep = p.add_argument_group("scale-up sweep")
    sweep.add_argument(
        "--sweep-n",
        type=int,
        nargs="+",
        default=None,
        help="Particle counts to sweep. Enables sweep mode.",
    )
    sweep.add_argument(
        "--sweep-seeds",
        type=int,
        nargs="+",
        default=(0, 1, 2),
        help="Seeds per N, so the switch-rate trend has error bars.",
    )
    sweep.add_argument(
        "--target-mode",
        choices=("fixed", "density-matched"),
        default="fixed",
        help=(
            "fixed: the same r* at every N (comparable budget, but r* becomes "
            "geometrically unreachable once N particles cannot be spaced r* "
            "apart inside the clip box). density-matched: r* scales as "
            "N^(-1/dim) off --target-reference-n, holding the target at a fixed "
            "multiple of the mean interparticle spacing. Separate series."
        ),
    )
    sweep.add_argument(
        "--target-reference-n",
        type=int,
        default=256,
        help="N at which density-matched r* equals --target-distance.",
    )
    sweep.add_argument(
        "--lr-mode",
        choices=("fixed", "n-scaled"),
        default="fixed",
        help=(
            "fixed: the same learning rate at every N (equal cost, but progress "
            "per step falls as 1/N because the objective is a mean). n-scaled: lr "
            "proportional to N off --lr-reference-n, which holds the descent "
            "dynamics N-invariant. Separate series; still plain gradient descent."
        ),
    )
    sweep.add_argument(
        "--lr-reference-n",
        type=int,
        default=256,
        help="N at which the n-scaled lr equals --learning-rate.",
    )
    sweep.add_argument(
        "--series-label",
        type=str,
        default=None,
        help="Overrides the auto-derived series label in the payload.",
    )
    sweep.add_argument(
        "--time-budget-s",
        type=float,
        default=0.0,
        help="Stop the sweep after this much cumulative wall clock (0 = no cap).",
    )
    sweep.add_argument(
        "--max-run-s",
        type=float,
        default=0.0,
        help="Stop the sweep once a single run exceeds this wall clock (0 = off).",
    )
    sweep.add_argument(
        "--related-series",
        type=str,
        nargs="+",
        default=None,
        help="Companion series files, recorded in the payload so no curve is read alone.",
    )
    sweep.add_argument("--smoke-sweep", action="store_true")
    return p.parse_args()


def _gpu_contention() -> dict[str, Any] | None:
    """Snapshot co-tenant load on the visible GPU, or None when not applicable.

    Wall-clock and peak-memory fields are only interpretable against the load
    the device was already carrying, so the snapshot is recorded alongside them.

    Returns
    -------
    dict or None
        Memory/utilisation/process-count snapshot, or None on CPU runs and when
        ``nvidia-smi`` is unavailable.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return None
    index = visible.split(",")[0].strip()
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                index,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        pids = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
                "-i",
                index,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        used, total, util = (int(v.strip()) for v in gpu.split(","))
    except Exception:  # pragma: no cover - env-dependent
        return None
    return {
        "gpu_index": index,
        "memory_used_mib": used,
        "memory_total_mib": total,
        "utilization_pct": util,
        "compute_process_count": len(pids),
    }


def _memory_snapshot() -> dict[str, Any]:
    """Return device and host memory counters for the current process.

    Returns
    -------
    dict
        ``bytes_in_use`` / ``peak_bytes_in_use`` from the JAX device allocator
        (None when the backend exposes no stats) and the host high-water RSS.
        Both peaks are *process-cumulative*, so within a sweep of increasing N
        they are only tight for the largest N run so far.
    """
    import jax

    device_bytes: int | None = None
    device_peak: int | None = None
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:  # pragma: no cover - env-dependent
        stats = None
    if stats:
        device_bytes = stats.get("bytes_in_use")
        device_peak = stats.get("peak_bytes_in_use")
    return {
        "device_bytes_in_use": device_bytes,
        "device_peak_bytes_in_use": device_peak,
        "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "note": (
            "Peaks are process-cumulative high-water marks, not per-run; within "
            "an increasing-N sweep they are tight only for the largest N so far."
        ),
    }


def _descend(
    cfg: DescentConfig, *, collect_history: bool = False, verbose: bool = True
) -> tuple[dict[str, Any], list[Any] | None]:
    """Run the rebuild-in-the-loop descent for one configuration.

    Parameters
    ----------
    cfg
        The descent configuration.
    collect_history
        Keep every intermediate position array (needed for the figure, skipped
        in the sweep where it would dominate memory at large N).
    verbose
        Print the periodic per-step progress line.

    Returns
    -------
    tuple
        ``(record, positions_history)`` where ``record`` holds the summary,
        trajectory, switch statistics, timing and memory fields, and
        ``positions_history`` is None unless ``collect_history`` is set.
    """
    import jax
    import jax.numpy as jnp

    from yggdrax import build_tree

    target = cfg.target_distance
    k = cfg.k_neighbors
    leaf_size = cfg.leaf_size

    t_setup = time.perf_counter()

    key = jax.random.PRNGKey(cfg.seed)
    key_pos, key_mass = jax.random.split(key)
    positions0 = jax.random.uniform(
        key_pos,
        (cfg.num_particles, cfg.dim),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    # build_tree requires masses; the Morton ordering depends on positions only,
    # so unit masses suffice for a spacing objective.
    masses = jnp.ones((cfg.num_particles,), dtype=jnp.float32)

    def morton_nn_stats(positions_unsorted):
        """Return (loss, nn_dist) using the rebuilt tree's Morton neighbours."""
        tree = build_tree(positions_unsorted, masses, leaf_size=leaf_size)
        pos_sorted = positions_unsorted[tree.particle_indices]
        n = pos_sorted.shape[0]
        idx = jnp.arange(n)
        big = jnp.asarray(1e6, dtype=pos_sorted.dtype)

        def safe_norm(delta):
            return jnp.sqrt(jnp.sum(delta * delta, axis=1) + 1e-12)

        candidate_dists = []
        for off in range(1, k + 1):
            idx_plus = jnp.clip(idx + off, 0, n - 1)
            idx_minus = jnp.clip(idx - off, 0, n - 1)
            dist_plus = safe_norm(pos_sorted - pos_sorted[idx_plus])
            dist_minus = safe_norm(pos_sorted - pos_sorted[idx_minus])
            candidate_dists.append(jnp.where(idx + off < n, dist_plus, big))
            candidate_dists.append(jnp.where(idx - off >= 0, dist_minus, big))

        nn_dist = jnp.min(jnp.stack(candidate_dists, axis=1), axis=1)
        loss = jnp.mean((nn_dist - target) ** 2)
        return loss, nn_dist

    def loss_only(positions_unsorted):
        return morton_nn_stats(positions_unsorted)[0]

    loss_and_grad = jax.jit(jax.value_and_grad(loss_only))
    stats = jax.jit(morton_nn_stats)

    def ordering(positions_unsorted):
        return build_tree(
            positions_unsorted, masses, leaf_size=leaf_size
        ).particle_indices

    ordering_jit = jax.jit(ordering)

    positions = positions0
    positions_history = [positions0] if collect_history else None
    loss_history: list[float] = []
    mean_nn_history: list[float] = []
    topology_changed: list[bool] = []  # per-step: did the ordering change?

    prev_order = ordering_jit(positions)
    _, nn0 = stats(positions)
    initial_mean_nn = float(jnp.mean(nn0))
    setup_s = time.perf_counter() - t_setup

    step_times: list[float] = []
    t_loop = time.perf_counter()
    for step in range(cfg.steps):
        t_step = time.perf_counter()
        loss_val, grad_val = loss_and_grad(positions)
        positions = jnp.clip(
            positions - cfg.learning_rate * grad_val, -cfg.clip, cfg.clip
        )
        if positions_history is not None:
            positions_history.append(positions)

        _, nn_dist = stats(positions)
        loss_history.append(float(loss_val))
        mean_nn_history.append(float(jnp.mean(nn_dist)))

        order = ordering_jit(positions)
        changed = not bool(jnp.array_equal(order, prev_order))
        topology_changed.append(changed)
        prev_order = order
        step_times.append(time.perf_counter() - t_step)

        if verbose and (step % max(1, cfg.steps // 6) == 0 or step == cfg.steps - 1):
            print(
                f"step={step:3d} | loss={loss_history[-1]:.6f} | "
                f"mean_NN={mean_nn_history[-1]:.6f} | "
                f"topo_changed={'yes' if changed else 'no'}"
            )
    loop_s = time.perf_counter() - t_loop

    final_loss, final_nn = stats(positions)
    topology_changes = int(sum(topology_changed))
    half = cfg.steps // 2
    switched_steps = [i for i, c in enumerate(topology_changed) if c]

    record: dict[str, Any] = {
        "num_particles": cfg.num_particles,
        "free_parameters": cfg.num_particles * cfg.dim,
        "dim": cfg.dim,
        "seed": cfg.seed,
        "steps": cfg.steps,
        "target_distance": target,
        "k_neighbors": k,
        "leaf_size": leaf_size,
        "learning_rate": cfg.learning_rate,
        "clip": cfg.clip,
        "switch_metric": SWITCH_METRIC,
        "summary": {
            "initial_mean_nn": initial_mean_nn,
            "final_mean_nn": float(jnp.mean(final_nn)),
            "initial_loss": loss_history[0],
            "final_loss": float(final_loss),
            "loss_ratio": float(final_loss) / loss_history[0],
            "topology_changes": topology_changes,
            "topology_change_fraction": topology_changes / cfg.steps,
            "topology_change_fraction_first_half": (
                sum(topology_changed[:half]) / half if half else None
            ),
            "topology_change_fraction_second_half": (
                sum(topology_changed[half:]) / (cfg.steps - half) if half else None
            ),
            "last_switch_step": switched_steps[-1] if switched_steps else None,
        },
        "timing": {
            "setup_s": setup_s,
            "loop_s": loop_s,
            "total_s": setup_s + loop_s,
            "first_step_s": step_times[0],
            "median_step_s": statistics.median(step_times),
            "note": (
                "setup_s and first_step_s include XLA compilation for this "
                "problem shape; median_step_s is the steady-state per-step cost."
            ),
        },
        "geometry": {
            "clip_box_volume": (2.0 * cfg.clip) ** cfg.dim,
            "volume_at_target": cfg.num_particles * target**cfg.dim,
            "target_packing_ratio": (
                cfg.num_particles * target**cfg.dim / (2.0 * cfg.clip) ** cfg.dim
            ),
            "initial_spacing_over_target": initial_mean_nn / target,
            "note": (
                "target_packing_ratio > 1 means r* cannot be realised inside the "
                "clip box even at perfect packing, so the objective saturates for "
                "box-geometry reasons rather than differentiability ones. "
                "initial_spacing_over_target < 1 means the descent has to *expand* "
                "the cloud rather than compress it, which is a different (and "
                "harder) task at the same nominal budget."
            ),
        },
        "memory": _memory_snapshot(),
        "trajectory": {
            "step": list(range(cfg.steps)),
            "loss": loss_history,
            "mean_nn": mean_nn_history,
            "topology_changed": [bool(c) for c in topology_changed],
        },
    }
    return record, positions_history


def _legacy_payload(cfg: DescentConfig, record: dict[str, Any]) -> dict[str, Any]:
    """Project a descent record onto the original single-run JSON schema.

    The committed ``nn_rebuild.json`` is cited by the paper text and read by the
    figure notebook, so the single-run payload keeps exactly the keys it had
    before the sweep was added.

    Parameters
    ----------
    cfg
        The configuration that produced ``record``.
    record
        A record as returned by :func:`_descend`.

    Returns
    -------
    dict
        The original ``nn_rebuild`` payload shape.
    """
    summary = record["summary"]
    return {
        "benchmark": "nn_rebuild",
        "params": {
            "num_particles": cfg.num_particles,
            "dim": cfg.dim,
            "target_distance": cfg.target_distance,
            "k_neighbors": cfg.k_neighbors,
            "leaf_size": cfg.leaf_size,
            "steps": cfg.steps,
            "learning_rate": cfg.learning_rate,
            "clip": cfg.clip,
            "seed": cfg.seed,
        },
        "metadata": run_metadata(),
        "summary": {
            "initial_mean_nn": summary["initial_mean_nn"],
            "final_mean_nn": summary["final_mean_nn"],
            "initial_loss": summary["initial_loss"],
            "final_loss": summary["final_loss"],
            "topology_changes": summary["topology_changes"],
        },
        "trajectory": record["trajectory"],
    }


def _historical_reference() -> dict[str, Any] | None:
    """Load the committed N = 256 point as a labelled historical reference.

    That point was taken on CPU at a different jax version, so it is *not* the
    first point of a GPU curve: a backend change and a version change would sit
    inside the trend, indistinguishable from an N effect. It is carried along for
    comparison only; the sweep re-measures N = 256 on its own hardware.

    Returns
    -------
    dict or None
        The reference entry, or None when the committed JSON is absent.
    """
    path = _REPO_ROOT / _SINGLE_OUTPUT
    try:
        payload = json.loads(path.read_text())
    except Exception:  # pragma: no cover - env-dependent
        return None
    return {
        "source": _SINGLE_OUTPUT,
        "params": payload.get("params"),
        "metadata": payload.get("metadata"),
        "summary": payload.get("summary"),
        "role": (
            "Historical reference, not a point on this curve: taken on a "
            "different backend and jax version. The sweep re-measures N = 256 on "
            "this run's own hardware so the trend contains only an N effect."
        ),
    }


def _target_for(args: argparse.Namespace, num_particles: int) -> float:
    """Return the target spacing r* for one N under the selected target mode.

    Parameters
    ----------
    args
        Parsed command-line arguments.
    num_particles
        The particle count.

    Returns
    -------
    float
        The target nearest-neighbour spacing.
    """
    if args.target_mode == "fixed":
        return args.target_distance
    scale = (args.target_reference_n / num_particles) ** (1.0 / args.dim)
    return args.target_distance * scale


def _lr_for(args: argparse.Namespace, num_particles: int) -> float:
    """Return the learning rate for one N under the selected lr mode.

    Parameters
    ----------
    args
        Parsed command-line arguments.
    num_particles
        The particle count.

    Returns
    -------
    float
        The gradient-descent learning rate.
    """
    if args.lr_mode == "fixed":
        return args.learning_rate
    return args.learning_rate * (num_particles / args.lr_reference_n)


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce per-seed records to one entry per N with spreads over seeds.

    Parameters
    ----------
    records
        Successful per-run records.

    Returns
    -------
    list
        One entry per N, ordered by N, carrying mean and sample standard
        deviation of the switch fraction, final loss and wall clock.
    """
    by_n: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        if "error" in rec:
            continue
        by_n.setdefault(rec["num_particles"], []).append(rec)

    def spread(values: list[float]) -> dict[str, Any]:
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    out: list[dict[str, Any]] = []
    for num_particles in sorted(by_n):
        group = by_n[num_particles]
        out.append(
            {
                "num_particles": num_particles,
                "free_parameters": group[0]["free_parameters"],
                "target_distance": group[0]["target_distance"],
                "learning_rate": group[0]["learning_rate"],
                "steps": group[0]["steps"],
                "seeds": [r["seed"] for r in group],
                "switch_metric": SWITCH_METRIC,
                "topology_change_fraction": spread(
                    [r["summary"]["topology_change_fraction"] for r in group]
                ),
                "final_loss": spread([r["summary"]["final_loss"] for r in group]),
                "loss_ratio": spread([r["summary"]["loss_ratio"] for r in group]),
                "final_mean_nn": spread([r["summary"]["final_mean_nn"] for r in group]),
                "initial_mean_nn": spread(
                    [r["summary"]["initial_mean_nn"] for r in group]
                ),
                "target_packing_ratio": group[0]["geometry"]["target_packing_ratio"],
                "initial_spacing_over_target": spread(
                    [r["geometry"]["initial_spacing_over_target"] for r in group]
                ),
                "total_s": spread([r["timing"]["total_s"] for r in group]),
                "median_step_s": spread([r["timing"]["median_step_s"] for r in group]),
            }
        )
    return out


def _run_sweep(args: argparse.Namespace) -> None:
    """Sweep N x seeds under a fixed per-run budget and write the scaling JSON.

    Parameters
    ----------
    args
        Parsed command-line arguments with ``sweep_n`` set.
    """
    records: list[dict[str, Any]] = []
    stopped_early: dict[str, Any] | None = None
    t_sweep = time.perf_counter()
    contention_start = _gpu_contention()

    for num_particles in args.sweep_n:
        target = _target_for(args, num_particles)
        learning_rate = _lr_for(args, num_particles)
        for seed in args.sweep_seeds:
            cfg = DescentConfig(
                num_particles=num_particles,
                dim=args.dim,
                target_distance=target,
                k_neighbors=args.k_neighbors,
                leaf_size=args.leaf_size,
                steps=args.steps,
                learning_rate=learning_rate,
                clip=args.clip,
                seed=seed,
            )
            print(
                f"\n[sweep] N={num_particles} seed={seed} "
                f"r*={target:.6g} lr={learning_rate:.6g}"
            )
            t_run = time.perf_counter()
            try:
                record, _ = _descend(cfg, collect_history=False, verbose=False)
            except Exception as exc:  # pragma: no cover - resource-dependent
                elapsed = time.perf_counter() - t_run
                print(f"[sweep] FAILED after {elapsed:.1f}s: {type(exc).__name__}")
                records.append(
                    {
                        "num_particles": num_particles,
                        "free_parameters": num_particles * args.dim,
                        "seed": seed,
                        "steps": args.steps,
                        "target_distance": target,
                        "learning_rate": learning_rate,
                        "switch_metric": SWITCH_METRIC,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:2000],
                        "wall_clock_before_failure_s": elapsed,
                        "memory": _memory_snapshot(),
                    }
                )
                stopped_early = {
                    "reason": "error",
                    "at_num_particles": num_particles,
                    "at_seed": seed,
                    "detail": f"{type(exc).__name__}: {str(exc)[:400]}",
                }
                break
            summary = record["summary"]
            print(
                f"[sweep] N={num_particles} seed={seed} "
                f"loss {summary['initial_loss']:.3e} -> {summary['final_loss']:.3e} "
                f"| mean_NN {summary['initial_mean_nn']:.4f} -> "
                f"{summary['final_mean_nn']:.4f} "
                f"| switches {summary['topology_changes']}/{args.steps} "
                f"({summary['topology_change_fraction']:.2%}) "
                f"| {record['timing']['total_s']:.1f}s"
            )
            records.append(record)

            if args.max_run_s and record["timing"]["total_s"] > args.max_run_s:
                stopped_early = {
                    "reason": "max_run_s",
                    "at_num_particles": num_particles,
                    "at_seed": seed,
                    "detail": (
                        f"run took {record['timing']['total_s']:.1f}s > "
                        f"--max-run-s {args.max_run_s:.1f}s"
                    ),
                }
        if stopped_early is not None:
            break
        elapsed = time.perf_counter() - t_sweep
        if args.time_budget_s and elapsed > args.time_budget_s:
            stopped_early = {
                "reason": "time_budget_s",
                "at_num_particles": num_particles,
                "detail": (
                    f"cumulative {elapsed:.1f}s > --time-budget-s "
                    f"{args.time_budget_s:.1f}s"
                ),
            }
            break

    if stopped_early is not None:
        print(f"\n[sweep] stopped early: {stopped_early}")

    series = (
        args.series_label or f"{args.target_mode}-target/{args.lr_mode}-lr/plain-gd/x64"
    )
    payload = {
        "benchmark": "nn_rebuild_scaling",
        "series": series,
        "switch_metric": SWITCH_METRIC,
        "target_mode": args.target_mode,
        "target_reference_n": args.target_reference_n,
        "lr_mode": args.lr_mode,
        "lr_reference_n": args.lr_reference_n,
        "fixed_budget": {
            "dim": args.dim,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "clip": args.clip,
            "k_neighbors": args.k_neighbors,
            "leaf_size": args.leaf_size,
            "target_distance": args.target_distance,
        },
        "limitations": LIMITATIONS,
        "related_series": (list(args.related_series) if args.related_series else None),
        "historical_reference": _historical_reference(),
        "metadata": run_metadata(
            extra={
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "xla_memory_env": {
                    key: os.environ.get(key)
                    for key in (
                        "XLA_PYTHON_CLIENT_PREALLOCATE",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION",
                        "XLA_PYTHON_CLIENT_ALLOCATOR",
                    )
                },
                "gpu_contention_at_start": contention_start,
                "gpu_contention_at_end": _gpu_contention(),
                "sweep_wall_clock_s": time.perf_counter() - t_sweep,
            }
        ),
        "sweep": {
            "num_particles_requested": list(args.sweep_n),
            "seeds": list(args.sweep_seeds),
            "stopped_early": stopped_early,
        },
        "aggregate": _aggregate(records),
        "records": records,
    }
    dump_json(payload, args.output)


def main() -> None:
    """Run a single descent or a scale-up sweep, per the command line."""
    args = _parse_args()
    if args.smoke:
        args.num_particles = 256
        args.steps = 30
        args.gpu_select = "none"
    if args.smoke_sweep:
        args.sweep_n = [64, 128]
        args.sweep_seeds = [0, 1]
        args.steps = 5
        args.gpu_select = "none"
    sweep_mode = args.sweep_n is not None
    if args.output is None:
        args.output = _SWEEP_OUTPUT if sweep_mode else _SINGLE_OUTPUT

    select_free_gpu(args.gpu_select, tag="nn_rebuild")

    if sweep_mode:
        _run_sweep(args)
        return

    import jax.numpy as jnp

    cfg = DescentConfig(
        num_particles=args.num_particles,
        dim=args.dim,
        target_distance=args.target_distance,
        k_neighbors=args.k_neighbors,
        leaf_size=args.leaf_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        clip=args.clip,
        seed=args.seed,
    )
    record, positions_history = _descend(cfg, collect_history=True, verbose=True)
    summary = record["summary"]
    print(
        f"\ntarget={cfg.target_distance}  "
        f"mean_NN {summary['initial_mean_nn']:.4f} -> "
        f"{summary['final_mean_nn']:.4f}  "
        f"loss {summary['initial_loss']:.3e} -> {summary['final_loss']:.3e}  "
        f"topology_changes={summary['topology_changes']}/{cfg.steps}"
    )

    dump_json(_legacy_payload(cfg, record), args.output)

    assert positions_history is not None
    npz_path = Path(args.output).with_suffix(".npz")
    dump_npz(
        {
            "positions_initial": positions_history[0],
            "positions_final": positions_history[-1],
            "positions_history": jnp.stack(positions_history),
        },
        npz_path,
    )


if __name__ == "__main__":
    main()
