"""Shared setup of the differentiability benches (A5): problem, references, I/O.

Every bench in this directory imports this module *after* parsing its command
line and calling :func:`select_device`, because GPU selection (autocvd, the org
rule) must happen before JAX initialises CUDA.

The problem is the paper's ξ payload on a small box: ``N`` points in the unit
cube, soft log bins, the natural estimator ``ξ_k = DD_k / RR_k - 1`` against a
fixed uniform random catalogue of the same size (``RR`` carries no gradient),
and the loss ``L = Σ_k log²((1 + ξ_k) / (1 + ξ*_k))`` against the ξ of a
clustered target -- every bin weighted by its relative error, so the sparse
small-r bins do not swamp the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def add_common_args(p: argparse.ArgumentParser) -> None:
    """``--platform``, ``--gpu-select``, ``--out``, ``--fig``, ``--x64``."""
    p.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    p.add_argument(
        "--gpu-select",
        choices=["free", "least-used", "none"],
        default="free",
        help="GPU pick before JAX starts (autocvd; org rule: 'free')",
    )
    p.add_argument("--gpu-wait-timeout", type=float, default=3600.0)
    p.add_argument("--out", default=None, help="JSON path (default: results/)")
    p.add_argument("--fig", default=None, help="figure path (default: results/)")
    p.add_argument(
        "--x64", action=argparse.BooleanOptionalAction, default=True, help="float64"
    )
    p.add_argument(
        "--replot", default=None, help="redraw the figure from this JSON, no run"
    )


def select_device(args) -> None:
    """Pin the platform and, on GPU, one free card -- before importing JAX."""
    if args.platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if args.gpu_select != "none" and "CUDA_VISIBLE_DEVICES" not in os.environ:
            from autocvd import autocvd  # site tool: sets CUDA_VISIBLE_DEVICES

            picked = autocvd(
                num_gpus=1,
                least_used=args.gpu_select == "least-used",
                timeout=args.gpu_wait_timeout,
            )
            print(f"# autocvd selected GPU: {picked}", file=sys.stderr, flush=True)
    if args.x64:
        os.environ["JAX_ENABLE_X64"] = "1"


def default_path(name: str, ext: str, args) -> str:
    out = os.path.join(_HERE, "results")
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, f"{name}_{args.platform}.{ext}")


def environment() -> dict[str, Any]:
    import jax

    return {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=float)
    print(f"# wrote {path}")


# --------------------------------------------------------------------------
# problem
# --------------------------------------------------------------------------


def uniform(n: int, seed: int):
    import jax

    return jax.random.uniform(jax.random.PRNGKey(seed), (n, 3))


def clustered(n: int, seed: int, num_centers: int = 12, width: float = 0.03):
    """Gaussian blobs (half the points) on a uniform background, in [0, 1]."""
    import jax
    import jax.numpy as jnp

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
    centers = jax.random.uniform(k1, (num_centers, 3), minval=0.15, maxval=0.85)
    pick = jax.random.randint(k2, (n,), 0, num_centers)
    blob = centers[pick] + width * jax.random.normal(k3, (n, 3))
    back = jax.random.uniform(k4, (n, 3))
    take = (jnp.arange(n) % 2 == 0)[:, None]
    return jnp.clip(jnp.where(take, blob, back), 0.0, 1.0)


def brute_counts(positions, edges, sharpness, block: int = 256):
    """Exact ``Σ_{i<j} w_k(r_ij)``: row blocks, checkpointed, differentiable."""
    import jax
    import jax.numpy as jnp

    from yggdrax.applications.corrfunc.binning import soft_bin_weights

    n = positions.shape[0]
    nb = -(-n // block)
    pad = nb * block - n
    idx = jnp.arange(nb * block)
    rows = jnp.concatenate([positions, jnp.zeros((pad, 3), positions.dtype)])

    @jax.checkpoint
    def one(b):
        i = jax.lax.dynamic_slice_in_dim(idx, b * block, block)
        xi = jax.lax.dynamic_slice_in_dim(rows, b * block, block)
        d = xi[:, None, :] - positions[None, :, :]
        r = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30)
        w = soft_bin_weights(r, edges, sharpness)
        keep = (i[:, None] < jnp.arange(n)[None, :]) & (i[:, None] < n)
        return jnp.sum(w * keep[..., None], axis=(0, 1))

    return jnp.sum(jax.lax.map(one, jnp.arange(nb)), axis=0)


class XiProblem:
    """ξ loss against a clustered target; ``RR`` from fixed uniform randoms."""

    def __init__(
        self, n: int, edges, sharpness: float, seed: int = 0, margin: float = 0.05
    ):
        import jax
        import jax.numpy as jnp

        self.n = n
        self.edges = edges
        self.sharpness = float(sharpness)
        self.bounds = (jnp.full(3, -margin), jnp.full(3, 1.0 + margin))
        randoms = uniform(n, seed + 1000)
        self.rr = jax.lax.stop_gradient(brute_counts(randoms, edges, sharpness))
        target = clustered(n, seed + 2000)
        self.xi_target = brute_counts(target, edges, sharpness) / self.rr - 1.0
        self.weights = jnp.ones_like(self.rr)

    def xi(self, counts):
        return counts / self.rr - 1.0

    def loss_from_counts(self, counts):
        import jax.numpy as jnp

        ratio = (1.0 + self.xi(counts)) / (1.0 + self.xi_target)
        return jnp.sum(self.weights * jnp.log(ratio) ** 2)

    def exact_loss(self, positions):
        return self.loss_from_counts(
            brute_counts(positions, self.edges, self.sharpness)
        )


def plan(problem: XiProblem, probes, **kw):
    """A traced config covering ``probes`` (the bench's catalogues)."""
    from yggdrax.applications.corrfunc.traced import plan_traced_pair_counts

    kw.setdefault("leaf_size", 16)
    return plan_traced_pair_counts(
        probes,
        edges=problem.edges,
        sharpness=problem.sharpness,
        bounds=problem.bounds,
        **kw,
    )


def tree_counts(problem: XiProblem, positions, config, membership=None):
    from yggdrax.applications.corrfunc.traced import traced_soft_pair_counts

    return traced_soft_pair_counts(
        positions,
        problem.edges,
        config,
        sharpness=problem.sharpness,
        bounds=problem.bounds,
        membership=membership,
    )


def frozen_counts(problem: XiProblem, positions, config, frozen_at):
    """Counts of ``positions`` on the topology (tree AND walk) of ``frozen_at``.

    For a hard rule this is the smooth continuation of the estimator from
    ``frozen_at``: ``tree(x') - frozen(x', at=x)`` is exactly the jump the step
    ``x -> x'`` picked up from switches.
    """
    from yggdrax.applications.corrfunc.traced import _accumulate, _build_topology

    topo = _build_topology(
        frozen_at, config, problem.bounds, problem.edges, problem.sharpness, True
    )
    return _accumulate(positions, problem.edges, config, topo, problem.sharpness, True)


#: Rules every bench compares: (label, plan kwargs).
def rules(theta: float = 0.5, eps: float = 1e-3, beta: float = 2.0):
    return [
        (f"geometric θ={theta}", dict(acceptance="geometric", theta=theta)),
        (
            f"geometric θ={theta}, smooth β={beta}",
            dict(acceptance="geometric", theta=theta, smooth_band=beta),
        ),
        (f"window ε={eps:g}", dict(acceptance="window", window_eps=eps)),
        (
            f"window ε={eps:g}, smooth β={beta}",
            dict(acceptance="window", window_eps=eps, smooth_band=beta),
        ),
    ]


def leaf_signature(problem: XiProblem, positions, config):
    """Per-particle leaf id (Morton cell of its leaf): membership fingerprint."""
    import jax.numpy as jnp

    from yggdrax.applications.corrfunc.traced import _build_topology

    topo = _build_topology(
        positions, config, problem.bounds, problem.edges, problem.sharpness, True
    )
    ni = topo.num_internal
    ranges = topo.node_ranges[ni:]
    n = positions.shape[0]
    slot = jnp.arange(n)
    # leaf index of each Morton slot, then of each particle
    starts = jnp.where(ranges[:, 1] >= ranges[:, 0], ranges[:, 0], n)
    leaf_of_slot = jnp.searchsorted(jnp.sort(starts), slot, side="right") - 1
    # label a leaf by the smallest particle id in it (order-independent)
    import jax

    smallest = jax.ops.segment_min(
        topo.order, leaf_of_slot, num_segments=int(ranges.shape[0])
    )
    return jnp.zeros(n, topo.order.dtype).at[topo.order].set(smallest[leaf_of_slot])


def pair_lists(problem: XiProblem, positions, config):
    """``(far, near)`` canonical node-pair lists as sorted int arrays."""
    from yggdrax.applications.corrfunc.traced import _build_topology

    topo = _build_topology(
        positions, config, problem.bounds, problem.edges, problem.sharpness, True
    )
    return topo.walk.far_a, topo.walk.far_b, topo.walk.near_a, topo.walk.near_b


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3}
    )
    return plt
