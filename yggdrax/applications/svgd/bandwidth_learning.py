"""Learn the SVGD kernel bandwidth by backpropagating through the sampler.

The differentiability payoff: instead of fixing the RBF bandwidth with the
median heuristic (which systematically under-disperses SVGD on targets with
mismatched scales), we treat the bandwidth as a parameter and learn it by
gradient descent on a validation loss --- the maximum mean discrepancy (MMD)
between the SVGD particles and held-out target samples --- backpropagated
through several tree-accelerated SVGD steps.

Gradients flow through the sampler because each step's Stein update is smooth
in the bandwidth given that step's (discrete, fixed) neighbour indices
(:mod:`yggdrax.applications.svgd.sampler`).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from yggdrax import DualTreeTraversalConfig
from yggdrax.applications.svgd.sampler import (
    build_svgd_topology,
    svgd_phi_from_topology,
)

# Compiled accumulation for the (eager) forward pass that records the frozen
# partitions; collapses per-op dispatch the same way the sampler's step does.
_jit_phi = jax.jit(svgd_phi_from_topology)


def squared_mmd(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    bandwidth: float | Float[Array, ""],
) -> Float[Array, ""]:
    """Unbiased-ish squared MMD between sample sets with an RBF kernel.

    Args:
        x: First sample set, shape ``(n, d)``.
        y: Second sample set, shape ``(m, d)``.
        bandwidth: RBF kernel bandwidth for the MMD statistic.

    Returns:
        Scalar squared MMD.
    """

    def k(a, b):
        d2 = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return jnp.exp(-d2 / (2.0 * bandwidth**2))

    return jnp.mean(k(x, x)) + jnp.mean(k(y, y)) - 2.0 * jnp.mean(k(x, y))


class BandwidthResult(NamedTuple):
    """Outcome of a bandwidth-learning run."""

    h_init: float
    h_final: float
    h_history: list[float]
    loss_history: list[float]


def learn_bandwidth(
    init_particles: Float[Array, "n d"],
    score_fn: Callable[[Float[Array, "n d"]], Float[Array, "n d"]],
    target_samples: Float[Array, "m d"],
    *,
    h_init: float,
    mmd_bandwidth: float,
    step_size: float = 0.3,
    num_svgd_steps: int = 20,
    theta: float = 0.4,
    leaf_size: int = 32,
    backend: str = "leaf_kdtree",
    learning_rate: float = 0.1,
    num_outer_steps: int = 40,
    traversal_config: DualTreeTraversalConfig | None = None,
) -> BandwidthResult:
    """Learn the SVGD bandwidth by minimising MMD to target samples.

    The tree topology is not differentiable, so we cannot backpropagate through
    ``build_svgd_topology``. Instead, at each outer step we run a concrete
    forward pass to record the per-step partitions, then differentiate a
    *replay* of the SVGD steps through those frozen partitions (the
    fixed-topology gradient of section 2). ``h`` is optimised in log space to
    stay positive.

    Args:
        init_particles: Initial SVGD particles, shape ``(n, d)``.
        score_fn: Target score function, ``(n, d) -> (n, d)``.
        target_samples: Held-out samples from the target, shape ``(m, d)``.
        h_init: Initial SVGD kernel bandwidth.
        mmd_bandwidth: Fixed RBF bandwidth of the MMD validation statistic.
        step_size: SVGD step size.
        num_svgd_steps: SVGD steps unrolled per outer gradient step.
        theta: Opening angle for the tree sampler.
        leaf_size: Tree leaf occupancy.
        backend: Tree backend for the partition build (``"radix"``/``"octree"``
            for d<=3, ``"leaf_kdtree"`` for arbitrary dimension).
        learning_rate: Adam learning rate for the (log) bandwidth.
        num_outer_steps: Number of outer optimization steps.
        traversal_config: Optional explicit traversal capacities.

    Returns:
        A :class:`BandwidthResult` with the bandwidth/loss trajectory.
    """

    def forward_topologies(h):
        """Concrete forward pass; return the frozen per-step partitions."""
        p = init_particles
        topos = []
        for _ in range(num_svgd_steps):
            topo = build_svgd_topology(
                p,
                theta=theta,
                leaf_size=leaf_size,
                backend=backend,
                traversal_config=traversal_config,
            )
            topos.append(topo)
            p = p + step_size * _jit_phi(p, score_fn(p), h, topo)
        return topos

    def replay_loss(log_h, topos):
        """Differentiable replay through frozen partitions."""
        h = jnp.exp(log_h)
        p = init_particles
        for topo in topos:
            p = p + step_size * svgd_phi_from_topology(p, score_fn(p), h, topo)
        return squared_mmd(p, target_samples, mmd_bandwidth)

    # Jit the frozen-topology replay's forward+backward pass. The topologies are
    # fixed (non-differentiable) arguments, so the whole unrolled replay fuses
    # into one compiled fwd+bwd kernel instead of dispatching every op eagerly.
    # It (re)compiles only when the padded topology shapes change across outer
    # steps; with a fixed-shape (radix) partition it is compiled once.
    replay_value_and_grad = jax.jit(jax.value_and_grad(replay_loss))

    optimizer = optax.adam(learning_rate)
    log_h = jnp.log(jnp.asarray(h_init, dtype=init_particles.dtype))
    opt_state = optimizer.init(log_h)

    h_hist = [float(jnp.exp(log_h))]
    loss_hist: list[float] = []
    for _ in range(num_outer_steps):
        topos = forward_topologies(jnp.exp(log_h))
        loss, g = replay_value_and_grad(log_h, topos)
        loss_hist.append(float(loss))
        updates, opt_state = optimizer.update(g, opt_state, log_h)
        log_h = jnp.asarray(optax.apply_updates(log_h, updates))
        h_hist.append(float(jnp.exp(log_h)))

    return BandwidthResult(
        h_init=float(h_init),
        h_final=float(jnp.exp(log_h)),
        h_history=h_hist,
        loss_history=loss_hist,
    )
