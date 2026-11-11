"""B4: move a particle set until its periodic ξ and kNN-CDFs match a target's.

Every coordinate of an ``N``-particle catalogue is a parameter. The loss
compares the catalogue's ξ (traced soft pair counts, window rule, minimum
image) and kNN-CDFs (random and data queries) with a target catalogue's; an
``optax`` optimiser (Adam or L-BFGS) descends on it with gradients through both
tree estimators. The kNN search and the ξ leaf partition are refreshed every
``--rebuild-every`` steps (:mod:`yggdrax.applications.positions.matching`).

The box grows with ``N`` at fixed number density ``--density``
(``L = (N / density)^(1/3)``), so every length -- the target's cluster size,
the ξ bins, the kNN radii -- is fixed in units of the mean spacing
``l = density^(-1/3)`` and the cost per particle does not change with ``N``.

Target: a Thomas process (:mod:`yggdrax.applications.point_process`) with
``--thomas-children`` per parent and width ``--thomas-sigma`` (in units of
``l``), or ``--target-file`` (``.npy`` positions, or ``.npz`` with
``positions`` and ``box_size``). Start: uniform, or ``--init-file``.

Output (``--out``, default ``bench/positions/results/optimise_<platform>.json``):
the loss history (total and both terms, every step), per-epoch timing,
staleness and displacement, the statistics at the start, at the end and of the
target (soft ξ, soft and hard kNN-CDFs), the loss of an independent target
realisation against the target (the noise floor), a per-component timing of
one step, the environment, and with ``--gradcheck`` a gradient check against
finite differences and against autodiff of a dense ``O(N^2)`` reference. A
``.npz`` next to it keeps thin slabs of the three catalogues (all positions
with ``--save-positions``), and ``--fig`` (default next to the JSON) draws the
summary figure; ``--replot JSON`` redraws it without running.

CPU (a few 1e4):

    python bench/positions/optimise.py --num-particles 30000 --num-steps 600 --gradcheck

Smoke (CPU, seconds):

    python bench/positions/optimise.py --smoke

GPU (one card, autocvd picks a free one -- the org rule):

    python bench/positions/optimise.py --platform gpu --gpu-select free \
        --num-particles 1000000 --num-steps 600
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, os.path.join(_REPO, "bench", "differentiability")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _a5_common import (  # noqa: E402  (path set up above)
    add_common_args,
    environment,
    select_device,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line.

    Parameters
    ----------
    argv:
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p)
    p.add_argument("--num-particles", "--n", type=int, default=30000)
    p.add_argument("--density", type=float, default=3.0e4, help="particles / volume")
    p.add_argument("--target-file", default=None)
    p.add_argument("--init-file", default=None)
    p.add_argument("--thomas-children", type=int, default=16)
    p.add_argument("--thomas-sigma", type=float, default=0.3, help="in units of l")
    p.add_argument("--thomas-beta", type=float, default=1.0)
    p.add_argument("--query-mode", choices=["randoms", "data", "both"], default="both")
    p.add_argument("--num-queries", type=int, default=None, help="default N")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--xi-sharpness", type=float, default=40.0)
    p.add_argument("--knn-sharpness", type=float, default=8.0)
    p.add_argument("--window-eps", type=float, default=1e-6)
    p.add_argument("--candidate-margin", type=int, default=8)
    p.add_argument("--leaf-size", type=int, default=32, help="cell tree of ξ")
    p.add_argument(
        "--weights",
        type=float,
        nargs=2,
        default=None,
        metavar=("XI", "KNN"),
        help="loss weights; default 1 / each term at the start",
    )
    p.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    p.add_argument(
        "--learning-rate", type=float, default=0.05, help="Adam, in units of l"
    )
    p.add_argument(
        "--final-lr-fraction",
        type=float,
        default=0.1,
        help="Adam: cosine decay to this fraction (1 = constant)",
    )
    p.add_argument("--num-steps", type=int, default=600)
    p.add_argument("--rebuild-every", type=int, default=10)
    p.add_argument("--timing-reps", type=int, default=5)
    p.add_argument("--gradcheck", action="store_true")
    p.add_argument("--gradcheck-n", type=int, default=1024, help=">= 800 (grid < L/2)")
    p.add_argument("--save-positions", action="store_true")
    p.add_argument("--no-fig", action="store_true", help="skip the figure")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny CPU run")
    args = p.parse_args(argv)
    if args.smoke:
        args.platform, args.gpu_select = "cpu", "none"
        args.num_particles, args.num_steps, args.rebuild_every = 1024, 20, 5
        args.timing_reps, args.gradcheck_n = 1, 1024
    return args


# --------------------------------------------------------------------------
# problem
# --------------------------------------------------------------------------


def _load(path: str):
    import numpy as np

    if path.endswith(".npz"):
        z = np.load(path)
        box = float(z["box_size"]) if "box_size" in z else None
        return np.asarray(z["positions"], dtype=np.float64), box
    return np.asarray(np.load(path), dtype=np.float64), None


def make_problem(args: argparse.Namespace, n: int, seed: int) -> dict[str, Any]:
    """Target, start, randoms and spec for ``n`` particles.

    Parameters
    ----------
    args:
        Parsed command line.
    n:
        Requested particles (rounded down to whole Thomas clusters).
    seed:
        Seed of the target, start and randoms.

    Returns
    -------
    dict
        ``target``, ``independent`` (a second target realisation, or ``None``
        for a file target), ``init``, ``randoms``, ``spec``, ``box``, ``ell``.

    Raises
    ------
    ValueError
        If a file catalogue lacks its box or the start's size differs.
    """
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    from yggdrax.applications.point_process import draw_thomas, thomas_positions
    from yggdrax.applications.positions import default_spec

    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    independent = None
    if args.target_file:
        tgt_np, box = _load(args.target_file)
        if box is None:
            raise ValueError("a .npy target needs a .npz with box_size instead")
        target = jnp.asarray(tgt_np)
        n = int(target.shape[0])
        ell = box / n ** (1.0 / 3.0)
    else:
        m = int(args.thomas_children)
        n = (n // m) * m
        ell = args.density ** (-1.0 / 3.0)
        box = (n / args.density) ** (1.0 / 3.0)
        theta = jnp.asarray([args.thomas_sigma * ell, args.thomas_beta])
        target = thomas_positions(theta, draw_thomas(keys[0], n // m, m), box)
        independent = thomas_positions(theta, draw_thomas(keys[3], n // m, m), box)
    if args.init_file:
        init_np, _ = _load(args.init_file)
        init = jnp.asarray(init_np)
        if int(init.shape[0]) != n:
            raise ValueError("--init-file must hold as many particles as the target")
    else:
        init = jax.random.uniform(keys[1], (n, 3), dtype=target.dtype) * box
    q = int(args.num_queries) if args.num_queries else n
    randoms = jax.random.uniform(keys[2], (q, 3), dtype=target.dtype) * box
    spec = replace(
        default_spec(
            box, mean_spacing=ell, query_mode=args.query_mode, knn_ks=tuple(args.ks)
        ),
        xi_sharpness=args.xi_sharpness,
        knn_sharpness=args.knn_sharpness,
        window_eps=args.window_eps,
        candidate_margin=args.candidate_margin,
    )
    return {
        "target": target,
        "independent": independent,
        "init": init,
        "randoms": randoms,
        "spec": spec,
        "box": float(box),
        "ell": float(ell),
        "n": n,
    }


def _stats_dict(stats, hard) -> dict[str, Any]:
    import numpy as np

    return {
        "counts": np.asarray(stats.counts).tolist(),
        "xi": np.asarray(stats.xi).tolist(),
        "cdf_soft": np.asarray(stats.cdf).tolist(),
        "cdf_hard": np.asarray(hard).tolist(),
    }


def _summary(stats, hard, target, target_hard, terms) -> dict[str, float]:
    import numpy as np

    lnr = np.log((np.asarray(stats.counts) + 1) / (np.asarray(target.counts) + 1))
    return {
        "loss": float(terms.loss),
        "loss_xi": float(terms.xi),
        "loss_knn": float(terms.knn),
        "max_abs_ln_ratio_1pxi": float(np.max(np.abs(lnr))),
        "max_abs_cdf_soft_error": float(
            np.max(np.abs(np.asarray(stats.cdf) - np.asarray(target.cdf)))
        ),
        "max_abs_cdf_hard_error": float(
            np.max(np.abs(np.asarray(hard) - np.asarray(target_hard)))
        ),
    }


# --------------------------------------------------------------------------
# gradient check
# --------------------------------------------------------------------------


def gradient_check(
    args: argparse.Namespace, log: Callable[[str], None]
) -> dict[str, Any]:
    """Tree gradient vs central differences and vs a dense reference.

    At ``--gradcheck-n`` particles, halfway between the start and the target
    (so clusters and voids are both present): the tree gradient at a frozen
    state against (a) ``jax.grad`` of the dense ``O(N^2)`` estimator, all
    coordinates, and (b) central differences of the tree loss at the frozen
    state and of the full chain (state refreshed at every evaluation), on 24
    random coordinates over a sweep of steps ``h``.

    Parameters
    ----------
    args:
        Parsed command line.
    log:
        Line logger.

    Returns
    -------
    dict
        Errors and settings.
    """
    from dataclasses import replace

    import jax
    import jax.numpy as jnp
    import numpy as np

    from yggdrax.applications.positions import PositionMatcher
    from yggdrax.applications.positions.reference import dense_statistics

    prob = make_problem(args, args.gradcheck_n, args.seed + 101)
    spec = replace(prob["spec"], window_eps=1e-12)
    mid = 0.5 * (prob["init"] + prob["target"])
    m = PositionMatcher.build(
        spec, prob["target"], probes=[prob["init"], mid], randoms=prob["randoms"]
    )
    state = jax.jit(m.refresh)(mid)

    def frozen(x):
        return m.loss(x, state)[0]

    def chain(x):
        return m.loss_from_statistics(m.fresh_statistics(x)).loss

    def dense(x):
        return m.loss_from_statistics(dense_statistics(m, x)).loss

    g_tree = np.asarray(jax.jit(jax.grad(frozen))(mid))
    g_dense = np.asarray(jax.jit(jax.grad(dense))(mid))
    rel_dense = float(np.linalg.norm(g_tree - g_dense) / np.linalg.norm(g_dense))
    val_rel = float(abs(jax.jit(frozen)(mid) / jax.jit(dense)(mid) - 1.0))
    log(
        f"gradcheck: tree vs dense autodiff, rel L2 {rel_dense:.2e}, value {val_rel:.1e}"
    )

    rng = np.random.default_rng(args.seed)
    flat = rng.choice(mid.size, size=24, replace=False)
    rows, cols = flat // 3, flat % 3
    f_frozen, f_chain = jax.jit(frozen), jax.jit(chain)
    sweep = []
    for h_ell in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        h = h_ell * prob["ell"]
        fd_f, fd_c = [], []
        for r, c in zip(rows, cols):
            e = jnp.zeros_like(mid).at[r, c].set(h)
            fd_f.append(float(f_frozen(mid + e) - f_frozen(mid - e)) / (2 * h))
            fd_c.append(float(f_chain(mid + e) - f_chain(mid - e)) / (2 * h))
        g = g_tree[rows, cols]
        scale = float(np.max(np.abs(g)))
        rec = {
            "h_over_ell": h_ell,
            "max_abs_err_frozen_over_max_grad": float(
                np.max(np.abs(np.asarray(fd_f) - g)) / scale
            ),
            "max_abs_err_chain_over_max_grad": float(
                np.max(np.abs(np.asarray(fd_c) - g)) / scale
            ),
        }
        sweep.append(rec)
        log(
            f"gradcheck: h = {h_ell:.0e} l: FD frozen {rec['max_abs_err_frozen_over_max_grad']:.2e}, "
            f"FD full chain {rec['max_abs_err_chain_over_max_grad']:.2e}"
        )
    return {
        "n": prob["n"],
        "window_eps": spec.window_eps,
        "coordinates": [[int(r), int(c)] for r, c in zip(rows, cols)],
        "tree_vs_dense_grad_rel_l2": rel_dense,
        "tree_vs_dense_value_rel": val_rel,
        "fd_sweep": sweep,
    }


# --------------------------------------------------------------------------
# timing of one step's parts
# --------------------------------------------------------------------------


def time_parts(matcher: Any, positions: Any, reps: int) -> dict[str, Any]:
    """Median wall time of a refresh and of value+grad of each loss term.

    Parameters
    ----------
    matcher:
        The matcher.
    positions:
        Where to time (the final, clustered catalogue: the dearer end).
    reps:
        Timed repetitions after one warm-up call.

    Returns
    -------
    dict
        Seconds per part, median and first call (compile included).
    """
    import statistics as st

    import jax

    state = jax.jit(matcher.refresh)(positions)

    def xi_only(x):
        return matcher.loss(x, state)[1].xi

    def knn_only(x):
        return matcher.loss(x, state)[1].knn

    fns = {
        "refresh_knn_search": (jax.jit(matcher.refresh), ()),
        "value_and_grad_total": (
            jax.jit(jax.value_and_grad(lambda x: matcher.loss(x, state)[0])),
            (),
        ),
        "value_and_grad_xi": (jax.jit(jax.value_and_grad(xi_only)), ()),
        "value_and_grad_knn": (jax.jit(jax.value_and_grad(knn_only)), ()),
    }
    out: dict[str, Any] = {}
    for name, (fn, _) in fns.items():
        t0 = time.perf_counter()
        jax.block_until_ready(fn(positions))
        first = time.perf_counter() - t0
        runs = []
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(positions))
            runs.append(time.perf_counter() - t0)
        out[name] = {"median_s": st.median(runs), "first_s": first}
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the optimisation and write JSON, NPZ and figure.

    Parameters
    ----------
    argv:
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    dict
        The JSON payload.
    """
    args = parse_args(argv)
    if args.replot:
        import json

        with open(args.replot) as f:
            payload = json.load(f)
        plot(payload, args.fig or os.path.splitext(args.replot)[0] + ".png")
        return payload

    select_device(args)
    from dataclasses import replace

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from yggdrax.applications.positions import PositionMatcher, optimise_positions

    def log(msg: str) -> None:
        print(msg, flush=True)

    out = args.out or os.path.join(_HERE, "results", f"optimise_{args.platform}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    payload: dict[str, Any] = {
        "benchmark": "positions_optimise",
        "params": vars(args),
        "environment": environment(),
    }
    if args.gradcheck:
        payload["gradcheck"] = gradient_check(args, log)

    prob = make_problem(args, args.num_particles, args.seed)
    n, ell = prob["n"], prob["ell"]
    log(
        f"N = {n}, L = {prob['box']:.4g}, l = {ell:.4g}, Q = {prob['randoms'].shape[0]}"
    )

    t0 = time.perf_counter()
    m = PositionMatcher.build(
        prob["spec"],
        prob["target"],
        probes=[prob["init"]],
        randoms=prob["randoms"],
        leaf_size=args.leaf_size,
    )
    setup_s = time.perf_counter() - t0
    fresh = jax.jit(m.fresh_statistics)
    hard = jax.jit(m.hard_cdf)
    s_init = fresh(prob["init"])
    t_init = m.loss_from_statistics(s_init)
    if args.weights is None:
        w_xi, w_knn = 1.0 / float(t_init.xi), 1.0 / float(t_init.knn)
    else:
        w_xi, w_knn = args.weights
    m = PositionMatcher(
        replace(m.spec, xi_weight=w_xi, knn_weight=w_knn),
        m.traced_config,
        m.randoms,
        m.rr,
        m.target,
        m.n_points,
    )
    t_init = m.loss_from_statistics(s_init)
    log(f"setup {setup_s:.1f} s; weights xi {w_xi:.3g}, knn {w_knn:.3g}")

    if args.optimizer == "adam":
        lr0 = args.learning_rate * ell
        k = args.rebuild_every
        steps = -(-args.num_steps // k) * k
        sched = (
            lr0
            if args.final_lr_fraction >= 1.0
            else optax.cosine_decay_schedule(lr0, steps, alpha=args.final_lr_fraction)
        )
        opt = optax.adam(sched)
    else:
        opt = optax.lbfgs(memory_size=10)

    res = optimise_positions(
        m,
        prob["init"],
        opt,
        num_steps=args.num_steps,
        rebuild_every=args.rebuild_every,
        log=log,
    )
    m = res.matcher
    s_final = fresh(res.positions)
    t_final = m.loss_from_statistics(s_final)
    h_init, h_final, h_target = (
        hard(prob["init"]),
        hard(res.positions),
        hard(prob["target"]),
    )
    summary = {
        "initial": _summary(s_init, h_init, m.target, h_target, t_init),
        "final": _summary(s_final, h_final, m.target, h_target, t_final),
    }
    stats = {
        "initial": _stats_dict(s_init, h_init),
        "final": _stats_dict(s_final, h_final),
        "target": _stats_dict(m.target, h_target),
    }
    if prob["independent"] is not None:
        s_ind = fresh(prob["independent"])
        h_ind = hard(prob["independent"])
        summary["independent_realisation"] = _summary(
            s_ind, h_ind, m.target, h_target, m.loss_from_statistics(s_ind)
        )
        stats["independent_realisation"] = _stats_dict(s_ind, h_ind)
    log(
        "loss {:.3e} -> {:.3e} (independent realisation: {}); "
        "max|hard CDF err| {:.3f} -> {:.3f}; {:.1f} ms/step, compile {:.1f} s".format(
            summary["initial"]["loss"],
            summary["final"]["loss"],
            (
                f"{summary['independent_realisation']['loss']:.3e}"
                if "independent_realisation" in summary
                else "n/a"
            ),
            summary["initial"]["max_abs_cdf_hard_error"],
            summary["final"]["max_abs_cdf_hard_error"],
            res.seconds_per_step * 1e3,
            res.compile_seconds,
        )
    )

    parts = time_parts(m, res.positions, args.timing_reps)
    log(
        "parts (median s): "
        + ", ".join(f"{k} {v['median_s']:.3g}" for k, v in parts.items())
    )
    cfg = m.traced_config
    payload.update(
        {
            "n": n,
            "box_size": prob["box"],
            "mean_spacing": ell,
            "num_queries": int(prob["randoms"].shape[0]),
            "spec": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in vars(m.spec).items()
            },
            "traced_config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
            "steps": len(res.history),
            "setup_s": setup_s,
            "compile_s": res.compile_seconds,
            "seconds_per_step": res.seconds_per_step,
            "replans": res.epochs[-1]["replans"] if res.epochs else 0,
            "step_parts": parts,
            "summary": summary,
            "statistics": stats,
            "history": res.history,
            "epochs": res.epochs,
        }
    )
    write_json(out, payload)

    rng = np.random.default_rng(0)
    slab = {}
    for name, x in (
        ("initial", prob["init"]),
        ("final", res.positions),
        ("target", prob["target"]),
    ):
        x = np.asarray(x)
        sel = np.abs(x[:, 2] - 0.5 * prob["box"]) < 2.0 * ell
        idx = np.flatnonzero(sel)
        if idx.size > 20000:
            idx = rng.choice(idx, 20000, replace=False)
        slab[f"slab_{name}"] = x[idx]
        if args.save_positions:
            slab[f"positions_{name}"] = x
    np.savez_compressed(os.path.splitext(out)[0] + ".npz", **slab)
    if args.no_fig:
        return payload
    try:
        import matplotlib  # noqa: F401  (optional: not in the CI environment)
    except ImportError:
        print("# matplotlib not installed: no figure (--replot JSON draws it later)")
        return payload
    payload["_slabs"] = slab
    plot(payload, args.fig or os.path.splitext(out)[0] + ".png")
    payload.pop("_slabs")
    return payload


def plot(payload: dict[str, Any], path: str) -> None:
    """Loss history, ξ, peaked hard kNN-CDFs and slabs of the catalogues.

    Parameters
    ----------
    payload:
        The JSON payload (slabs from the ``.npz`` next to it when absent).
    path:
        Figure path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    slabs = payload.get("_slabs")
    if slabs is None:
        npz = os.path.splitext(path)[0] + ".npz"
        slabs = dict(np.load(npz)) if os.path.exists(npz) else {}
    st = payload["statistics"]
    spec = payload["spec"]
    ell = payload["mean_spacing"]
    fig, axes = plt.subplots(1, 6, figsize=(26, 4.2))

    ax = axes[0]
    step = np.array([h["step"] for h in payload["history"]])
    for key, lab in (("loss", "total"), ("xi", "ξ term"), ("knn", "kNN term")):
        y = np.array([h[key] for h in payload["history"]])
        if key != "loss":
            w = spec["xi_weight"] if key == "xi" else spec["knn_weight"]
            y = w * y
        ax.semilogy(step, y, label=lab)
    ind = payload["summary"].get("independent_realisation")
    if ind is not None:
        ax.axhline(ind["loss"], color="k", ls=":", label="independent realisation")
    ax.set_xlabel("step")
    ax.set_ylabel("weighted loss")
    ax.legend(fontsize=8)
    ax.set_title(f"N = {payload['n']}, {payload['seconds_per_step'] * 1e3:.0f} ms/step")

    ax = axes[1]
    e = np.asarray(spec["xi_edges"])
    rc = np.sqrt(e[:-1] * e[1:]) / ell
    for name, sty in (("initial", ":"), ("final", "--"), ("target", "-")):
        ax.loglog(rc, 1 + np.asarray(st[name]["xi"]), sty, label=name)
    if "independent_realisation" in st:
        ax.loglog(
            rc,
            1 + np.asarray(st["independent_realisation"]["xi"]),
            "k:",
            lw=0.8,
            label="independent",
        )
    ax.set_xlabel("r / l")
    ax.set_ylabel("1 + ξ")
    ax.legend(fontsize=8)

    r = np.asarray(spec["knn_r_grid"]) / ell
    ks = spec["knn_ks"]
    sets = (["randoms"] if spec["query_mode"] in ("randoms", "both") else []) + (
        ["data"] if spec["query_mode"] in ("data", "both") else []
    )
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.85, len(ks)))
    for s, name in enumerate(sets[:2]):
        ax = axes[2 + s]
        for j, k in enumerate(ks):

            def pk(c):
                c = np.asarray(c)[s, j]
                return np.minimum(c, 1 - c)

            ax.loglog(
                r, pk(st["target"]["cdf_hard"]), color=colors[j], lw=2, label=f"k={k}"
            )
            ax.loglog(r, pk(st["initial"]["cdf_hard"]), color=colors[j], ls=":", lw=1)
            ax.loglog(r, pk(st["final"]["cdf_hard"]), color=colors[j], ls="--", lw=1)
        ax.set_ylim(1e-3, 0.6)
        ax.set_xlabel("r / l")
        ax.set_title(
            f"peaked kNN-CDF, {name} queries\nsolid target, dotted start, dashed final",
            fontsize=9,
        )
        ax.legend(fontsize=7)
    for ax, name in zip(axes[4:], ("final", "target")):
        x = slabs.get(f"slab_{name}")
        if x is not None:
            ax.scatter(x[:, 0] / ell, x[:, 1] / ell, s=0.3, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(f"{name}, |z - L/2| < 2 l")
        ax.set_xlabel("x / l")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"# wrote {path}")


if __name__ == "__main__":
    main()
