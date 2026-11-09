"""Move particles by gradient descent until their kNN-CDFs match a target.

The payoff demo for the differentiable kNN-CDF: start from a uniform (Poisson)
catalogue, take the kNN-CDFs of a clustered one as the target, and descend on
the loss with respect to *every particle coordinate*. The KD-tree is rebuilt
and the neighbours re-found at every step inside the jitted step (the
non-differentiable part); the gradient flows through the recomputed distances
and the smooth step (see :mod:`yggdrax.applications.knn_stats.estimator`).

Target: a Thomas-like process -- ``--num-centers`` uniform cluster centres,
each particle a Gaussian offset of width ``--cluster-sigma`` from one of them,
in a periodic unit box (``--open-boundaries`` turns the box off). Particles
move freely: the kNN search and the distances use the minimum image, and the
positions are wrapped back into the box after each step (the wrap is outside
the gradient). The model and target share one fixed set of
volume-filling random queries (``--query-mode randoms``), use the data points
as queries (``data``), or both (``both``, default): randoms see the voids,
data-centred queries the clumps.

Results -> ``results/knn_stats/cdf_matching.json`` (+ ``.npz`` with the CDFs
and positions, + ``.png`` with ``--plot``).

Local (CPU):

    JAX_PLATFORMS=cpu conda run -n jaccpot python bench/knn_stats/cdf_matching_demo.py --plot

Smoke:

    JAX_PLATFORMS=cpu conda run -n jaccpot python bench/knn_stats/cdf_matching_demo.py --smoke

GPU server (1e6-1e7 particles, later):

    python bench/knn_stats/cdf_matching_demo.py --n 1000000 --num-queries 1000000 \
        --num-centers 2000 --gpu-select free
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.knn_stats._common import (
    dump_json,
    dump_npz,
    run_metadata,
    select_free_gpu,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--num-queries", type=int, default=4000)
    p.add_argument("--query-mode", choices=("randoms", "data", "both"), default="both")
    p.add_argument(
        "--open-boundaries",
        action="store_true",
        help="no periodic box: particles are clipped to the unit cube instead",
    )
    p.add_argument("--num-centers", type=int, default=20)
    p.add_argument("--cluster-sigma", type=float, default=0.03)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--r-min", type=float, default=0.004)
    p.add_argument("--r-max", type=float, default=0.3)
    p.add_argument("--num-r", type=int, default=24)
    p.add_argument("--sharpness", type=float, default=8.0)
    p.add_argument("--peaked", action="store_true", help="match peaked CDFs")
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument("--backend", default="tree", choices=("tree", "tiled", "dense"))
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--num-steps", type=int, default=400)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--output", type=str, default="results/knn_stats/cdf_matching.json")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.n, args.num_queries, args.num_steps = 400, 600, 30
        args.num_centers, args.log_every = 6, 10
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="knn-cdf-demo")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from yggdrax.applications.knn_stats import cdf_distance, hard_knn_cdf, knn_cdf
    from yggdrax.periodic import wrap_positions

    ks = tuple(args.ks)
    r_grid = jnp.geomspace(args.r_min, args.r_max, args.num_r)
    key = jax.random.PRNGKey(args.seed)
    k_init, k_cent, k_assign, k_off, k_q = jax.random.split(key, 5)

    centers = jax.random.uniform(k_cent, (args.num_centers, 3))
    assign = jax.random.randint(k_assign, (args.n,), 0, args.num_centers)
    offsets = args.cluster_sigma * jax.random.normal(k_off, (args.n, 3))
    target_pos = jnp.mod(centers[assign] + offsets, 1.0)
    init_pos = jax.random.uniform(k_init, (args.n, 3))
    randoms = jax.random.uniform(k_q, (args.num_queries, 3))

    query_sets = {
        "randoms": (randoms,),
        "data": (None,),
        "both": (randoms, None),
    }[args.query_mode]
    box = None if args.open_boundaries else 1.0
    kw = dict(backend=args.backend, leaf_size=args.leaf_size, box_size=box)

    def cdfs(pos, *, hard=False):
        """Stack the CDFs of every query set: shape (n_sets, len(ks), num_r)."""
        out = []
        for q in query_sets:
            if hard:
                out.append(hard_knn_cdf(pos, q, ks, r_grid, **kw))
            else:
                out.append(knn_cdf(pos, q, ks, r_grid, sharpness=args.sharpness, **kw))
        return jnp.stack(out)

    target = jax.jit(cdfs)(target_pos)
    target_hard = jax.jit(lambda p: cdfs(p, hard=True))(target_pos)

    def loss_fn(pos):
        return cdf_distance(cdfs(pos), target, peaked=args.peaked)

    # Plain Adam, so the demo needs nothing beyond JAX.
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def step(pos, m, v, t):
        loss, g = jax.value_and_grad(loss_fn)(pos)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        pos = pos - args.learning_rate * mhat / (jnp.sqrt(vhat) + eps)
        # Periodic: wrap back into the box (bookkeeping only; distances use the
        # minimum image either way). Open: keep particles where the queries are.
        pos = wrap_positions(pos, box) if box is not None else jnp.clip(pos, 0.0, 1.0)
        return pos, m, v, loss

    pos = init_pos
    m = jnp.zeros_like(pos)
    v = jnp.zeros_like(pos)
    history = []
    t0 = time.perf_counter()
    jax.block_until_ready(step(pos, m, v, 1.0))  # compile
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for it in range(1, args.num_steps + 1):
        pos, m, v, loss = step(pos, m, v, float(it))
        if it == 1 or it % args.log_every == 0 or it == args.num_steps:
            history.append({"step": it, "loss": float(loss)})
            print(f"step {it:5d}  loss {float(loss):.3e}")
    run_s = time.perf_counter() - t0

    hard_fn = jax.jit(lambda p: cdfs(p, hard=True))
    init_hard, final_hard = hard_fn(init_pos), hard_fn(pos)
    init_loss, final_loss = float(loss_fn(init_pos)), float(loss_fn(pos))
    init_hard_err = float(jnp.max(jnp.abs(init_hard - target_hard)))
    final_hard_err = float(jnp.max(jnp.abs(final_hard - target_hard)))
    print(
        f"soft loss {init_loss:.3e} -> {final_loss:.3e}; "
        f"max |hard CDF - target| {init_hard_err:.3f} -> {final_hard_err:.3f}; "
        f"{run_s / args.num_steps * 1e3:.1f} ms/step (compile {compile_s:.1f} s)"
    )

    out = Path(args.output)
    payload = {
        "benchmark": "knn_cdf_matching_demo",
        "params": {k: v for k, v in vars(args).items()},
        "metadata": run_metadata(),
        "query_sets": [("randoms" if q is not None else "data") for q in query_sets],
        "loss_initial": init_loss,
        "loss_final": final_loss,
        "max_abs_hard_cdf_error_initial": init_hard_err,
        "max_abs_hard_cdf_error_final": final_hard_err,
        "compile_s": compile_s,
        "seconds_per_step": run_s / args.num_steps,
        "history": history,
    }
    dump_json(payload, out)
    dump_npz(
        {
            "r_grid": r_grid,
            "ks": jnp.asarray(ks),
            "target_cdf_hard": target_hard,
            "initial_cdf_hard": init_hard,
            "final_cdf_hard": final_hard,
            "target_pos": target_pos,
            "initial_pos": init_pos,
            "final_pos": pos,
        },
        out.with_suffix(".npz"),
    )
    if args.plot:
        _plot(
            out.with_suffix(".png"),
            r_grid,
            ks,
            payload["query_sets"],
            target_hard,
            init_hard,
            final_hard,
            (init_pos, pos, target_pos),
        )


def _plot(path, r_grid, ks, set_names, target, initial, final, positions) -> None:
    """Peaked CDFs per query set, and a thin slab of each catalogue."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    r = np.asarray(r_grid)
    n_sets = len(set_names)
    fig, axes = plt.subplots(1, n_sets + 3, figsize=(4 * (n_sets + 3), 3.6))
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.85, len(ks)))
    for s, name in enumerate(set_names):
        ax = axes[s]
        for j, k in enumerate(ks):
            pk = lambda c: np.minimum(np.asarray(c[s, j]), 1 - np.asarray(c[s, j]))
            ax.loglog(r, pk(target), color=colors[j], lw=2, label=f"k={k}")
            ax.loglog(r, pk(initial), color=colors[j], ls=":", lw=1)
            ax.loglog(r, pk(final), color=colors[j], ls="--", lw=1)
        ax.set_ylim(1e-3, 0.6)
        ax.set_xlabel("r")
        ax.set_title(
            f"peaked CDF, {name} queries\nsolid target, dotted init, dashed final"
        )
        ax.legend(fontsize=7)
    for ax, p, title in zip(axes[n_sets:], positions, ("initial", "final", "target")):
        p = np.asarray(p)
        slab = np.abs(p[:, 2] - 0.5) < 0.1
        ax.scatter(p[slab, 0], p[slab, 1], s=2)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} (|z-0.5|<0.1)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[bench] wrote {path}")


if __name__ == "__main__":
    main()
