"""Gradient-based parameter recovery through the differentiable estimator.

A minimal end-to-end demonstration of the payoff of differentiability: recover
a clustering parameter by gradient descent on a loss between a model
correlation function and an observed one, backpropagating through the
tree-accelerated soft pair-count estimator.

Forward model. ``num_centers`` cluster centres and ``n`` uniform base points
are drawn from a fixed PRNG key; each base point is assigned to its nearest
centre (a fixed, discrete assignment). The catalogue is then

    x_i(a) = (1 - a) * base_i + a * centre[assign_i],

so ``a = 0`` is uniform and increasing ``a`` collapses points onto centres,
strengthening small-scale clustering. ``x_i`` is differentiable in ``a`` with
the assignment held fixed.

Recovery. Given the soft pair counts of a catalogue generated at ``a_true``,
we minimise the mean-squared error of the log counts w.r.t. ``a`` by gradient
descent. At each step the pair topology is rebuilt at the current ``a`` (the
non-differentiable part) and the loss is differentiated through the soft
accumulation (the differentiable part), exactly the fixed-topology gradient of
section 2.

Two drivers:

* :func:`recover_parameter_traced` -- the whole optimisation is one jitted
  ``lax.scan``; every step's loss calls
  :func:`~yggdrax.applications.corrfunc.traced.traced_soft_pair_counts`, which
  rebuilds the tree *inside* the differentiated function. Capacities are sized
  once, before the scan, by a probe over catalogues spanning the parameter
  range.
* :func:`recover_parameter` -- the original host loop: the topology is rebuilt
  on the host at each step (``build_pair_topology``, outside ``jax.grad``) and
  passed into the loss. Superseded by the traced driver; kept for comparison.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from yggdrax import DualTreeTraversalConfig
from yggdrax.applications.corrfunc.estimator import (
    build_pair_topology,
    soft_pair_counts_from_topology,
)
from yggdrax.applications.corrfunc.traced import (
    TracedPairCountConfig,
    plan_traced_pair_counts,
    traced_soft_pair_counts,
)
from yggdrax.bounds import infer_bounds


class ClusterModel(NamedTuple):
    """Fixed base points and centres defining the clustering forward model."""

    base: Array  # (n, 3) uniform base positions
    centers: Array  # (m, 3) cluster centres
    assign: Array  # (n,) index of the nearest centre for each base point


def make_cluster_model(key: Array, n: int = 2000, num_centers: int = 8) -> ClusterModel:
    """Draw the fixed base points, centres, and nearest-centre assignment.

    Args:
        key: PRNG key.
        n: Number of catalogue points.
        num_centers: Number of cluster centres.

    Returns:
        A :class:`ClusterModel` with a frozen base/centre/assignment.
    """
    k_base, k_cen = jax.random.split(key)
    base = jax.random.uniform(k_base, (n, 3), dtype=jnp.float64)
    centers = jax.random.uniform(k_cen, (num_centers, 3), dtype=jnp.float64)
    d = base[:, None, :] - centers[None, :, :]
    assign = jnp.argmin(jnp.sum(d * d, axis=-1), axis=1)
    return ClusterModel(base=base, centers=centers, assign=assign)


def catalog(model: ClusterModel, a: Array | float) -> Array:
    """Return the catalogue positions at clustering strength ``a``.

    Args:
        model: The fixed cluster model.
        a: Clustering strength in ``[0, 1)``; differentiable.

    Returns:
        Positions, shape ``(n, 3)``, differentiable in ``a``.
    """
    return (1.0 - a) * model.base + a * model.centers[model.assign]


def _log_counts(
    positions: Array,
    topo,
    edges: Array,
    sharpness: float,
) -> Array:
    counts = soft_pair_counts_from_topology(positions, topo, edges, sharpness)
    return jnp.log(counts + 1.0)


class RecoveryResult(NamedTuple):
    """Outcome of a gradient-based parameter recovery run."""

    a_true: float
    a_init: float
    a_history: list[float]
    loss_history: list[float]
    a_final: float


def recover_parameter(
    model: ClusterModel,
    edges: Array,
    *,
    a_true: float = 0.5,
    a_init: float = 0.1,
    sharpness: float = 100.0,
    theta: float = 0.5,
    leaf_size: int = 16,
    learning_rate: float = 0.05,
    num_steps: int = 80,
    traversal_config: DualTreeTraversalConfig | None = None,
) -> RecoveryResult:
    """Recover ``a_true`` from its correlation function by gradient descent.

    Args:
        model: The fixed cluster forward model.
        edges: Radial bin edges.
        a_true: Ground-truth clustering strength generating the target counts.
        a_init: Initial guess for the optimizer.
        sharpness: Soft-window sharpness.
        theta: Opening angle for the traversal.
        leaf_size: Target leaf occupancy.
        learning_rate: Adam step size.
        num_steps: Number of optimization steps.
        traversal_config: Optional explicit traversal capacities.

    Returns:
        A :class:`RecoveryResult` with the optimization trajectory.
    """
    # Target: log soft counts of the catalogue at a_true.
    pos_true = catalog(model, a_true)
    topo_true = build_pair_topology(
        pos_true, theta=theta, leaf_size=leaf_size, traversal_config=traversal_config
    )
    target = _log_counts(pos_true, topo_true, edges, sharpness)

    def loss_fn(a: Array, topo) -> Array:
        pos = catalog(model, a)
        model_log = _log_counts(pos, topo, edges, sharpness)
        return jnp.mean((model_log - target) ** 2)

    grad_fn = jax.grad(loss_fn, argnums=0)

    # Adam is robust to the steep, varying gradient of the log-count loss.
    optimizer = optax.adam(learning_rate)
    a = jnp.asarray(a_init, dtype=jnp.float64)
    opt_state = optimizer.init(a)
    a_hist: list[float] = [float(a)]
    loss_hist: list[float] = []
    for _ in range(num_steps):
        pos = catalog(model, a)
        # Rebuild the (non-differentiable) topology at the current estimate,
        # then take the fixed-topology gradient of the loss w.r.t. a.
        topo = build_pair_topology(
            pos, theta=theta, leaf_size=leaf_size, traversal_config=traversal_config
        )
        loss_hist.append(float(loss_fn(a, topo)))
        g = grad_fn(a, topo)
        updates, opt_state = optimizer.update(g, opt_state, a)
        a = jnp.clip(jnp.asarray(optax.apply_updates(a, updates)), 0.0, 0.99)
        a_hist.append(float(a))

    return RecoveryResult(
        a_true=float(a_true),
        a_init=float(a_init),
        a_history=a_hist,
        loss_history=loss_hist,
        a_final=float(a),
    )


def recover_parameter_traced(
    model: ClusterModel,
    edges: Array,
    *,
    a_true: float = 0.5,
    a_init: float = 0.1,
    sharpness: float = 100.0,
    theta: float = 0.5,
    leaf_size: int = 16,
    learning_rate: float = 0.05,
    num_steps: int = 80,
    config: TracedPairCountConfig | None = None,
    probe_a: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99),
    acceptance: Literal["window", "geometric"] = "window",
) -> RecoveryResult:
    """Recover ``a_true`` with the tree rebuilt inside the differentiated loss.

    The same problem as :func:`recover_parameter`, but the loss is a single
    traced function of ``a``: ``a -> catalogue -> static cells tree -> mutual
    walk -> soft counts -> loss``. ``jax.value_and_grad`` of it therefore
    differentiates *through* a fresh rebuild at every step, and the whole Adam
    loop runs as one jitted ``lax.scan``. The Morton domain is fixed (the box
    of the base points and centres, which holds every catalogue for ``a`` in
    ``[0, 1]``), so the leaf structure is comparable across steps.

    Args:
        model: The fixed cluster forward model.
        edges: Radial bin edges.
        a_true: Ground-truth clustering strength generating the target counts.
        a_init: Initial guess for the optimizer.
        sharpness: Soft-window sharpness.
        theta: Opening angle; used only by ``acceptance="geometric"``.
        leaf_size: Maximum particles per cell leaf.
        learning_rate: Adam step size.
        num_steps: Number of optimization steps.
        config: Explicit capacities. ``None`` plans them by probing the
            observed catalogue and the model at ``a_init`` and ``probe_a``.
        probe_a: Clustering strengths probed when planning; they should span
            the range the optimizer can reach (``a`` is clipped to ``[0,
            0.99]``).
        acceptance: Far-pair rule of the planned config (see
            :class:`~yggdrax.applications.corrfunc.traced.TracedPairCountConfig`);
            ``"window"`` is accurate to its ``window_eps`` at any leaf size.

    Returns:
        A :class:`RecoveryResult` with the optimization trajectory.

    Raises:
        RuntimeError: If any step overflowed the planned capacities, in which
            case its counts were incomplete. Re-plan with wider ``probe_a`` or
            pass a larger ``config``.
    """
    bounds = infer_bounds(jnp.concatenate([model.base, model.centers], axis=0))
    pos_true = catalog(model, a_true)
    if config is None:
        probes = [pos_true] + [catalog(model, a) for a in (a_init, *probe_a)]
        config = plan_traced_pair_counts(
            probes,
            edges=edges,
            sharpness=sharpness,
            acceptance=acceptance,
            theta=theta,
            leaf_size=leaf_size,
            bounds=bounds,
        )

    def log_counts(positions: Array) -> tuple[Array, Array]:
        res = traced_soft_pair_counts(
            positions, edges, config, sharpness=sharpness, bounds=bounds
        )
        return jnp.log(res.counts + 1.0), res.overflow

    target, target_overflow = jax.jit(log_counts)(pos_true)
    if bool(target_overflow):
        raise RuntimeError("target catalogue overflowed the planned capacities")

    def loss_fn(a: Array) -> tuple[Array, Array]:
        model_log, overflow = log_counts(catalog(model, a))
        return jnp.mean((model_log - target) ** 2), overflow

    optimizer = optax.adam(learning_rate)

    def step(carry, _):
        a, opt_state = carry
        (loss, overflow), g = jax.value_and_grad(loss_fn, has_aux=True)(a)
        updates, opt_state = optimizer.update(g, opt_state, a)
        a_new = jnp.clip(jnp.asarray(optax.apply_updates(a, updates)), 0.0, 0.99)
        return (a_new, opt_state), (a_new, loss, overflow)

    @jax.jit
    def run(a0: Array):
        _, (a_traj, losses, overflows) = jax.lax.scan(
            step, (a0, optimizer.init(a0)), None, length=num_steps
        )
        return a_traj, losses, overflows

    a0 = jnp.asarray(a_init, dtype=jnp.float64)
    a_traj, losses, overflows = run(a0)
    if bool(jnp.any(overflows)):
        first = int(jnp.argmax(overflows))
        raise RuntimeError(
            f"step {first} overflowed the planned capacities {config}; its "
            "counts were incomplete. Re-plan with a wider probe_a."
        )
    a_hist = [float(a0)] + [float(x) for x in a_traj]
    return RecoveryResult(
        a_true=float(a_true),
        a_init=float(a_init),
        a_history=a_hist,
        loss_history=[float(x) for x in losses],
        a_final=a_hist[-1],
    )
