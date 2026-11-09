"""A5c: leapfrog energy error and HMC acceptance with the tree ξ potential.

Potential ``U(x) = λ L(x) + wall(x)``: the ξ loss of ``_a5_common.XiProblem``
times ``λ``, plus a stiff quadratic wall keeping the particles near ``[0, 1]^3``.
Unit masses, plain JAX leapfrog (``lax.scan``), no external sampler.

For each potential -- exact (brute force), each rule of ``_a5_common.rules``
(hard / smooth, geometric / window), and the smooth rules with the leaf
membership FROZEN at each trajectory's start (``membership=x0``; then ``U`` is
``U(.; x0)`` at both ends of the MH test) -- it runs ``--starts`` trajectories
of ``--length`` leapfrog steps from the same starting states and momenta, at a
list of step sizes, and records

* ``dH = H(x_L, p_L) - H(x_0, p_0)`` and the MH acceptance ``min(1, e^{-dH})``;
* the largest ``|H_t - H_0|`` along the trajectory.

The exact potential's ``dH`` scales as ``ε²``; a discontinuous potential adds a
step-size-independent floor (each switch's jump enters ``dH`` whole), which is
what the smooth rule and the frozen membership remove.

Examples::

    python bench/differentiability/hmc_energy.py            # CPU, ~10 min
    python bench/differentiability/hmc_energy.py --smoke
    python bench/differentiability/hmc_energy.py --platform gpu --n 8000 \\
        --gpu-select free
"""

from __future__ import annotations

import argparse
import sys
import time

import _a5_common as C

#: λ of the potential and the reference leapfrog step (N = 500 defaults,
#: tuned on the exact potential): a 20-step trajectory moves each particle by
#: ~0.017 RMS (about the smallest bin edge) with |dH| ~ 0.05. The sharp
#: windows make U stiff -- at ε = 2e-3 the exact leapfrog is already unstable.
DEFAULT_LAMBDA = 30.0
DEFAULT_STEP = 5e-4
WALL = 1e4


class Potential:
    """``U = λ L + wall`` for the exact estimator or a tree config."""

    def __init__(self, problem: "C.XiProblem", lam: float):
        self.problem = problem
        self.lam = float(lam)

    @staticmethod
    def wall(x):
        import jax.numpy as jnp

        out = jnp.maximum(-x, 0.0) ** 2 + jnp.maximum(x - 1.0, 0.0) ** 2
        return WALL * jnp.sum(out)

    def exact(self, x):
        return self.lam * self.problem.exact_loss(x) + self.wall(x)

    def tree(self, x, cfg, membership=None):
        counts = C.tree_counts(self.problem, x, cfg, membership=membership)
        u = self.lam * self.problem.loss_from_counts(counts.counts) + self.wall(x)
        return u, counts.overflow


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.add_common_args(p)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--step", type=float, default=DEFAULT_STEP)
    p.add_argument("--step-factors", default="0.25,0.5,1,2")
    p.add_argument("--length", type=int, default=20)
    p.add_argument("--starts", type=int, default=24)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--sharpness", type=float, default=40.0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.n, args.starts, args.step_factors = 300, 3, "0.5,1"
    return args


def _trajectory(grad_and_value, x0, p0, step, length):
    """Leapfrog; returns final H - H0, max |H_t - H0|, overflow, final x."""
    import jax
    import jax.numpy as jnp

    u0, g0, ov0 = grad_and_value(x0)
    h0 = u0 + 0.5 * jnp.sum(p0 * p0)

    def body(carry, _):
        x, p, g, ov = carry
        p = p - 0.5 * step * g
        x = x + step * p
        u, g, o = grad_and_value(x)
        p = p - 0.5 * step * g
        h = u + 0.5 * jnp.sum(p * p)
        return (x, p, g, ov | o), h - h0

    (x, _, _, ov), dh = jax.lax.scan(body, (x0, p0, g0, ov0), None, length=length)
    return dh[-1], jnp.max(jnp.abs(dh)), ov, x


def run(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from yggdrax.applications.corrfunc.binning import make_log_edges

    edges = make_log_edges(0.02, 0.3, 10)
    problem = C.XiProblem(args.n, edges, args.sharpness, seed=0, margin=0.5)
    pot = Potential(problem, args.lam)
    starts = jnp.stack(
        [C.clustered(args.n, 2000 + s) * 0.9 + 0.05 for s in range(args.starts)]
    )
    momenta = jax.random.normal(jax.random.PRNGKey(1), starts.shape)
    factors = [float(f) for f in args.step_factors.split(",")]
    steps = [args.step * f for f in factors]
    # probes: the starts and a short exact trajectory from each, at the largest
    # step, so the configs cover where the tree potentials go
    probes = [starts[s] for s in range(args.starts)]
    g_exact = jax.jit(
        lambda x: (lambda v, g: (v, g, jnp.asarray(False)))(
            *jax.value_and_grad(pot.exact)(x)
        )
    )
    for s in range(min(args.starts, 6)):
        *_, xl = _trajectory(g_exact, starts[s], momenta[s], max(steps), args.length)
        probes.append(xl)

    potentials = [("exact", None, False)]
    for label, kw in C.rules(args.theta, args.eps, args.beta):
        potentials.append((label, kw, False))
        if kw.get("smooth_band", 1.0) > 1.0:
            potentials.append((label + ", frozen membership", kw, True))

    results = {}
    for label, kw, frozen in potentials:
        if kw is None:
            gv = g_exact

            def traj(x0, p0, step, gv=gv):
                return _trajectory(gv, x0, p0, step, args.length)

        else:
            cfg = C.plan(problem, probes, leaf_size=args.leaf_size, **kw)

            def traj(x0, p0, step, cfg=cfg, frozen=frozen):
                memb = x0 if frozen else None

                def gv(x):
                    (u, ov), g = jax.value_and_grad(
                        lambda y: pot.tree(y, cfg, membership=memb), has_aux=True
                    )(x)
                    return u, g, ov

                return _trajectory(gv, x0, p0, step, args.length)

        run_one = jax.jit(traj)
        t0 = time.perf_counter()
        per_step = {}
        for step in steps:
            dh, mx, ov = [], [], []
            for s in range(args.starts):
                d, m, o, _ = run_one(starts[s], momenta[s], step)
                dh.append(float(d))
                mx.append(float(m))
                ov.append(bool(o))
            dh = np.array(dh)
            per_step[f"{step:.6g}"] = dict(
                step=step,
                dH=dh.tolist(),
                max_abs_dH_along=mx,
                mean_abs_dH=float(np.mean(np.abs(dh))),
                median_abs_dH=float(np.median(np.abs(dh))),
                acceptance=float(np.mean(np.minimum(1.0, np.exp(-dh)))),
                overflow_any=bool(any(ov)),
            )
        results[label] = dict(config=kw, frozen_membership=frozen, steps=per_step)
        line = "  ".join(
            f"ε={v['step']:.2e}: |dH| {v['mean_abs_dH']:.2e} acc {v['acceptance']:.2f}"
            + (" OVERFLOW" if v["overflow_any"] else "")
            for v in per_step.values()
        )
        print(f"{label:52s} {line} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return dict(
        bench="A5c leapfrog energy error and HMC acceptance",
        env=C.environment(),
        args=vars(args),
        edges=[float(e) for e in np.asarray(edges)],
        results=results,
    )


def plot(payload, path):
    import numpy as np

    plt = C.import_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    styles = {
        "exact": dict(color="k", lw=3.0, alpha=0.35, zorder=1),
    }
    for label, rec in payload["results"].items():
        st = sorted(rec["steps"].values(), key=lambda v: v["step"])
        eps = [v["step"] for v in st]
        ls = "--" if ("smooth" not in label and label != "exact") else "-"
        mk = "s" if "frozen" in label else ("" if label == "exact" else "o")
        kw = styles.get(label, dict(zorder=2))
        axes[0].loglog(
            eps, [v["mean_abs_dH"] for v in st], ls, marker=mk, ms=3, label=label, **kw
        )
        axes[1].semilogx(eps, [v["acceptance"] for v in st], ls, marker=mk, ms=3, **kw)
    ref = sorted(payload["results"]["exact"]["steps"].values(), key=lambda v: v["step"])
    e = np.array([v["step"] for v in ref])
    axes[0].loglog(
        e,
        0.5 * ref[0]["mean_abs_dH"] * (e / e[0]) ** 3,
        "k:",
        lw=0.7,
        label="∝ ε³ (fixed step count)",
    )
    axes[0].set_xlabel("leapfrog step ε")
    axes[0].set_ylabel("mean |ΔH| per trajectory")
    axes[1].set_xlabel("leapfrog step ε")
    axes[1].set_ylabel("mean MH acceptance  min(1, e^{-ΔH})")
    axes[1].set_ylim(0.7, 1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle(
        f"N={payload['args']['n']}, {payload['args']['length']} leapfrog steps, "
        f"{payload['args']['starts']} trajectories per point; exact, window and "
        "frozen-membership smooth curves coincide",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path)
    print(f"# wrote {path}")


def main(argv=None) -> int:
    args = _parse(argv)
    if args.replot:
        import json

        with open(args.replot) as f:
            plot(json.load(f), args.fig or args.replot.replace(".json", ".png"))
        return 0
    C.select_device(args)
    payload = run(args)
    C.write_json(args.out or C.default_path("hmc_energy", "json", args), payload)
    plot(payload, args.fig or C.default_path("hmc_energy", "png", args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
