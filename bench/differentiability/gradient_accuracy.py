"""A5a: accuracy of the tree gradient of the ξ payload against exact O(N²) autodiff.

For each N and each acceptance setting -- geometric at a list of θ, window at a
list of ε, each hard and smooth (``smooth_band = β``) -- this measures, against
the brute-force estimator and its autodiff gradient:

* value error: ``max_k |DD_k / DD_k^exact - 1|`` and the relative loss error;
* gradient error of the ξ loss ``L = Σ_k (ξ_k - ξ*_k)²``: relative L2
  ``|g - g_ex| / |g_ex|`` and the max-norm ratio ``max_i |g_i - g_ex,i| /
  max_i |g_ex,i|``;
* an a-posteriori monopole bound (hard rules), from the walk's far list,
  checked against the counts' VJP with the exact cotangent ``c = ∂L/∂DD``
  (``vjp_err_rel_l2``; the loss gradient also changes through ``c`` evaluated
  at the tree's counts, a value-error effect the bound does not cover):

  - value, per bin: ``B_k = Σ_far n_a n_b (max_I w_k - min_I w_k)``, with ``I``
    the pair's separation range ``[d - s, d + s]`` (plus ``ε`` per pruned
    particle pair for the window rule);
  - gradient, per particle ``i``: ``B_i = Σ_{far P ∋ i} n_other (Var_I f' +
    |f'(D)| 4s / (d - s))`` with ``f = Σ_k (∂L/∂DD_k) w_k`` and ``D`` the
    centre-of-mass distance. ``Var`` and ``max`` are evaluated on a 65-point grid
    of ``I``, so the bound is a (tight) estimate, not a certificate. Pruned
    pairs of the window rule are not in it (they are not emitted).

It also checks the kNN-CDF gradient: given the neighbour set, the tree search
returns exactly the dense search's distances, so the gradient is that of the
dense O(N²) reference to round-off.

Examples::

    python bench/differentiability/gradient_accuracy.py            # CPU, ~10 min
    python bench/differentiability/gradient_accuracy.py --smoke    # seconds
    python bench/differentiability/gradient_accuracy.py --platform gpu \\
        --ns 4000,16000,64000 --gpu-select free
"""

from __future__ import annotations

import argparse
import sys
import time

import _a5_common as C


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.add_common_args(p)
    p.add_argument("--ns", default="1000,2000,4000")
    p.add_argument("--thetas", default="0.2,0.3,0.4,0.5,0.6,0.8")
    p.add_argument("--epss", default="1e-1,1e-2,1e-3,1e-4,1e-6,1e-8")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--sharpness", type=float, default=40.0)
    p.add_argument("--rmin", type=float, default=0.02)
    p.add_argument("--rmax", type=float, default=0.3)
    p.add_argument("--nbins", type=int, default=10)
    p.add_argument("--ic", choices=["clustered", "uniform"], default="clustered")
    p.add_argument("--knn-n", type=int, default=2000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.ns, args.thetas, args.epss, args.knn_n = "500", "0.5", "1e-3", 300
    return args


def _timed(fn, *a):
    """Result and wall time of the second call (the first compiles)."""
    import jax

    jax.block_until_ready(fn(*a))
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*a))
    return out, time.perf_counter() - t0


def _bounds(problem, positions, config, coeff):
    """A-posteriori monopole bounds (value per bin, gradient per particle)."""
    import jax
    import jax.numpy as jnp

    from yggdrax.applications.corrfunc.binning import soft_bin_weights
    from yggdrax.applications.corrfunc.traced import (
        _build_topology,
        _node_geometry,
        _range_min_max_table,
        _separation_interval_u,
        _window_min_max,
    )

    topo = _build_topology(
        positions, config, problem.bounds, problem.edges, problem.sharpness, True
    )
    walk = topo.walk
    nf = int(walk.far_count)
    pos_sorted = positions[topo.order]
    ranges = topo.node_ranges
    n = positions.shape[0]
    fa, fb = walk.far_a[:nf], walk.far_b[:nf]
    tables = _range_min_max_table(pos_sorted)
    ca, ra = _node_geometry(tables, ranges, fa)
    cb, rb = _node_geometry(tables, ranges, fb)
    dist_sq = jnp.sum((cb - ca) ** 2, axis=-1)
    d = jnp.sqrt(dist_sq)
    s = ra + rb
    na = (ranges[fa, 1] - ranges[fa, 0] + 1).astype(positions.dtype)
    nb = (ranges[fb, 1] - ranges[fb, 0] + 1).astype(positions.dtype)
    edges_u = jnp.log(problem.edges)
    u_lo, u_hi = _separation_interval_u(dist_sq, ra, rb, True)
    w_min, w_max = _window_min_max(edges_u, problem.sharpness, u_lo, u_hi)
    value_bound = jnp.sum((na * nb)[:, None] * (w_max - w_min), axis=0)
    prune_bound = jnp.zeros_like(value_bound)
    if config.acceptance == "window":
        npairs = n * (n - 1) / 2
        leaf_len = jnp.clip(
            ranges[topo.num_internal :, 1] - ranges[topo.num_internal :, 0] + 1, 0
        )
        within = jnp.sum(leaf_len * (leaf_len - 1) / 2)
        nl = int(walk.near_count)
        la = ranges[walk.near_a[:nl]]
        lb = ranges[walk.near_b[:nl]]
        near = jnp.sum((la[:, 1] - la[:, 0] + 1) * (lb[:, 1] - lb[:, 0] + 1))
        pruned = npairs - within - near - jnp.sum(na * nb)
        prune_bound = prune_bound + config.window_eps * pruned

    # gradient: f(r) = Σ_k c_k w_k(r), sampled on I
    def f(r):
        return jnp.sum(coeff * soft_bin_weights(r, problem.edges, problem.sharpness))

    fp = jax.vmap(jax.grad(f))
    grid = jnp.linspace(0.0, 1.0, 65)
    lo = jnp.maximum(d - s, 1e-6)
    hi = d + s
    rs = lo[:, None] + (hi - lo)[:, None] * grid[None, :]
    fps = fp(rs.reshape(-1)).reshape(rs.shape)
    var = jnp.max(fps, axis=1) - jnp.min(fps, axis=1)
    prefix = jnp.concatenate([jnp.zeros((1, 3)), jnp.cumsum(pos_sorted, axis=0)])

    def com(node):
        st, en = ranges[node, 0], ranges[node, 1]
        return (prefix[en + 1] - prefix[st]) / (en - st + 1)[:, None]

    big_d = jnp.linalg.norm(com(fa) - com(fb), axis=-1)
    # |r_ij - D| <= 2s (every particle and each COM lie in their node's
    # sphere), |r|, |D| >= d - s, and |x/|x| - y/|y|| <= 2|x - y| / max(|x|, |y|)
    angle = jnp.where(d > s, 4.0 * s / jnp.maximum(d - s, 1e-12), jnp.inf)
    per = var + jnp.abs(fp(big_d)) * angle  # per particle pair, |Δ∇|
    # spread n_other * per over each node's slots with a difference array
    diff = jnp.zeros(n + 1)
    for node, other in ((fa, nb), (fb, na)):
        amt = other * per
        diff = diff.at[ranges[node, 0]].add(amt).at[ranges[node, 1] + 1].add(-amt)
    per_slot = jnp.cumsum(diff)[:n]
    grad_bound = jnp.zeros(n).at[topo.order].set(per_slot)
    return value_bound, prune_bound, grad_bound


def _knn_check(n, seed=0):
    """kNN-CDF gradient: tree search vs dense top_k, both differentiated."""
    import jax
    import jax.numpy as jnp

    from yggdrax.kdtree import build_kdtree, query_neighbors

    x = C.clustered(n, seed)
    ks = (1, 2, 4, 8)
    kmax = max(ks)
    grid = jnp.linspace(jnp.log(0.005), jnp.log(0.15), 24)
    s = 10.0

    def cdf_loss(logd):  # (n, kmax) log distances
        cdf = jnp.mean(
            jax.nn.sigmoid(s * (grid[None, None, :] - logd[:, :, None])), axis=0
        )
        sel = cdf[jnp.asarray(ks) - 1]
        return jnp.sum((sel - 0.5) ** 2)

    def tree_loss(p):
        tree = build_kdtree(jax.lax.stop_gradient(p), leaf_size=16)
        idx, _ = query_neighbors(
            tree, jax.lax.stop_gradient(p), k=kmax, exclude_self=True, backend="tree"
        )
        d = jnp.linalg.norm(p[:, None, :] - p[idx], axis=-1)
        return cdf_loss(jnp.log(d)), idx

    def dense_loss(p):
        d2 = jnp.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1)
        d2 = d2 + jnp.diag(jnp.full(p.shape[0], jnp.inf))
        neg, idx = jax.lax.top_k(-d2, kmax)
        return cdf_loss(0.5 * jnp.log(-neg)), idx

    (lt, it), gt = jax.value_and_grad(tree_loss, has_aux=True)(x)
    (ld, idd), gd = jax.value_and_grad(dense_loss, has_aux=True)(x)
    same = bool(jnp.all(jnp.sort(it, axis=1) == jnp.sort(idd, axis=1)))
    return {
        "n": n,
        "ks": list(ks),
        "neighbour_sets_identical": same,
        "loss_rel_diff": float(abs(lt - ld) / abs(ld)),
        "grad_max_rel_diff": float(jnp.max(jnp.abs(gt - gd)) / jnp.max(jnp.abs(gd))),
    }


def run(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from yggdrax.applications.corrfunc.binning import make_log_edges

    edges = make_log_edges(args.rmin, args.rmax, args.nbins)
    ns = [int(v) for v in args.ns.split(",")]
    settings = []
    for th in [float(v) for v in args.thetas.split(",")]:
        for beta in (1.0, args.beta):
            settings.append(dict(acceptance="geometric", theta=th, smooth_band=beta))
    for ep in [float(v) for v in args.epss.split(",")]:
        for beta in (1.0, args.beta):
            settings.append(dict(acceptance="window", window_eps=ep, smooth_band=beta))
    rows = []
    for n in ns:
        problem = C.XiProblem(n, edges, args.sharpness, seed=0)
        x = (C.clustered if args.ic == "clustered" else C.uniform)(n, 7)
        exact_counts = jax.jit(lambda p: C.brute_counts(p, edges, problem.sharpness))(x)
        g_ex = jax.jit(jax.grad(problem.exact_loss))(x)
        loss_ex = float(problem.loss_from_counts(exact_counts))
        coeff = jax.grad(problem.loss_from_counts)(exact_counts)
        # the counts' VJP with the EXACT cotangent: what the monopole bound is
        # about (the loss gradient also moves through d L / d DD at the tree's
        # counts, a value-error effect on every particle)
        gc_ex = jax.jit(
            jax.grad(
                lambda p: jnp.dot(coeff, C.brute_counts(p, edges, problem.sharpness))
            )
        )(x)
        print(f"# N={n}: exact loss {loss_ex:.6e}", flush=True)
        for st in settings:
            cfg = C.plan(problem, x, leaf_size=args.leaf_size, **st)
            fwd = jax.jit(lambda p, c=cfg: C.tree_counts(problem, p, c))
            vg = jax.jit(
                jax.value_and_grad(
                    lambda p, c=cfg: problem.loss_from_counts(
                        C.tree_counts(problem, p, c).counts
                    )
                )
            )
            res, t_fwd = _timed(fwd, x)
            (loss, g), t_vg = _timed(vg, x)
            assert not bool(res.overflow), st
            counts = np.asarray(res.counts)
            dg = np.asarray(g - g_ex)
            row = dict(
                n=n,
                **st,
                value_err=float(np.max(np.abs(counts / np.asarray(exact_counts) - 1))),
                loss_rel_err=float(abs(float(loss) - loss_ex) / abs(loss_ex)),
                grad_rel_l2=float(np.linalg.norm(dg) / np.linalg.norm(g_ex)),
                grad_rel_max=float(np.max(np.abs(dg)) / np.max(np.abs(g_ex))),
                far=int(res.far_count),
                near=int(res.near_count),
                band=int(res.band_count),
                t_fwd_s=t_fwd,
                t_value_and_grad_s=t_vg,
            )
            if st["smooth_band"] == 1.0:
                vb, pb, gb = _bounds(problem, x, cfg, coeff)
                ex = np.asarray(exact_counts)
                row["value_bound_rel"] = float(np.max(np.asarray(vb) / ex))
                row["value_bound_prune_rel"] = float(np.max(np.asarray(pb) / ex))
                row["value_err_far_bound_holds"] = bool(
                    np.all(np.abs(counts - ex) <= np.asarray(vb + pb) * (1 + 1e-9))
                )
                gc = jax.grad(
                    lambda p, c=cfg: jnp.dot(coeff, C.tree_counts(problem, p, c).counts)
                )(x)
                dgc = np.asarray(gc - gc_ex)
                gnorm = np.linalg.norm(np.asarray(gc_ex), axis=1)
                dnorm = np.linalg.norm(dgc, axis=1)
                gbn = np.asarray(gb)
                row["vjp_err_rel_l2"] = float(
                    np.linalg.norm(dgc) / np.linalg.norm(gc_ex)
                )
                row["grad_bound_rel_l2"] = float(
                    np.linalg.norm(gbn) / np.linalg.norm(gc_ex)
                )
                row["grad_bound_rel_max"] = float(np.max(gbn) / np.max(gnorm))
                row["grad_bound_holds_frac"] = float(
                    np.mean(dnorm <= gbn * (1 + 1e-9) + 1e-14 * np.max(gnorm))
                )
            rows.append(row)
            print(
                "  {acceptance:9s} θ={th:<4} ε={eps:<6g} β={smooth_band:<3} "
                "val {value_err:.1e} grad {grad_rel_l2:.1e} (max {grad_rel_max:.1e}) "
                "bound {vb} far {far} near {near} band {band} vg {t_value_and_grad_s:.2f}s".format(
                    th=st.get("theta", "-"),
                    eps=st.get("window_eps", float("nan")),
                    vb=(
                        f"{row['value_bound_rel']:.1e}/{row['grad_bound_rel_l2']:.1e}"
                        if "value_bound_rel" in row
                        else "-"
                    ),
                    **row,
                ),
                flush=True,
            )
    knn = _knn_check(args.knn_n)
    print(f"# kNN-CDF gradient check: {knn}")
    return dict(
        bench="A5a gradient accuracy",
        env=C.environment(),
        args=vars(args),
        edges=[float(e) for e in edges],
        rows=rows,
        knn=knn,
    )


def plot(payload, path):
    import numpy as np

    plt = C.import_pyplot()
    rows = payload["rows"]
    ns = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(ns)))
    for ax, acc, key, xlabel in (
        (axes[0], "geometric", "theta", "opening angle θ"),
        (axes[1], "window", "window_eps", "window tolerance ε"),
    ):
        for n, col in zip(ns, colors):
            for beta, ls in ((1.0, "-"), (payload["args"]["beta"], "--")):
                sel = sorted(
                    (
                        r
                        for r in rows
                        if r["n"] == n
                        and r["acceptance"] == acc
                        and r["smooth_band"] == beta
                    ),
                    key=lambda r: r[key],
                )
                if not sel:
                    continue
                xs = [r[key] for r in sel]
                lab = f"N={n}" if beta == 1.0 else None
                ax.plot(
                    xs,
                    [max(r["grad_rel_l2"], 1e-17) for r in sel],
                    ls,
                    color=col,
                    marker="o",
                    ms=3,
                    label=lab,
                )
                ax.plot(
                    xs,
                    [max(r["value_err"], 1e-17) for r in sel],
                    ls,
                    color=col,
                    marker="x",
                    ms=3,
                    alpha=0.5,
                )
                if beta == 1.0 and n == ns[-1]:
                    ax.plot(
                        xs,
                        [r["grad_bound_rel_l2"] for r in sel],
                        ":",
                        color="k",
                        label=(
                            "gradient bound (hard)"
                            if acc == "geometric"
                            else "gradient bound, far part only (hard)"
                        ),
                    )
        ax.set_yscale("log")
        if acc == "window":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{acc}: o grad rel. L2, x value max rel.")
    axes[0].set_ylabel("relative error")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    ax = axes[2]
    hard = [
        r for r in rows if r["smooth_band"] == 1.0 and r.get("vjp_err_rel_l2", 0) > 0
    ]
    for acc, m in (("geometric", "o"), ("window (far part; pruned not bounded)", "s")):
        sel = [r for r in hard if r["acceptance"] == acc.split()[0]]
        ax.loglog(
            [r["grad_rel_l2"] for r in sel],
            [r["grad_bound_rel_l2"] for r in sel],
            m,
            ms=4,
            label=acc,
        )
    lim = [1e-16, 10]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlabel("measured VJP error, exact cotangent (rel. L2)")
    ax.set_ylabel("a-posteriori bound (rel. L2)")
    ax.set_title("monopole bound vs measurement")
    ax.legend(fontsize=7)
    fig.suptitle("dashed: smooth rule β=%g" % payload["args"]["beta"], fontsize=8)
    fig.tight_layout()
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
    out = args.out or C.default_path("gradient_accuracy", "json", args)
    C.write_json(out, payload)
    plot(payload, args.fig or C.default_path("gradient_accuracy", "png", args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
