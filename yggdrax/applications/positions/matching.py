"""Move a particle set until its ξ and kNN-CDFs match a target catalogue's.

Every coordinate of an ``(N, 3)`` catalogue in a periodic box is a free
parameter. The loss compares two clustering statistics of the catalogue with
those of a target catalogue of the same size:

* **ξ**, the natural estimator ``DD / RR - 1`` in soft log bins, with ``DD``
  from :func:`~yggdrax.applications.corrfunc.traced.traced_soft_pair_counts`
  (window rule, minimum image, the tree rebuilt inside the differentiated
  function) and the analytic ``RR`` of the box
  (:func:`~yggdrax.applications.point_process.natural_estimator_rr`);
* **kNN-CDFs** (:mod:`yggdrax.applications.knn_stats`) of fixed volume-filling
  random queries, of the particles themselves (own index excluded), or both.

The loss is ``xi_weight * L_xi + knn_weight * L_knn`` with

    L_xi  = mean_b ln^2((DD_b + 1) / (DD*_b + 1)),
    L_knn = mean over query sets, k and r of (CDF - CDF*)^2.

``ln((DD + 1) / (DD* + 1))`` is ``ln((1 + ξ) / (1 + ξ*))`` (both catalogues
have the same ``N`` and box, so the same ``RR``), regularised by one pair so an
empty bin stays finite; every bin then weighs by its relative error, and the
sparse small-r bins do not swamp the rest.

Rebuild cadence
---------------
The optimiser runs in *epochs* of ``rebuild_every`` steps, each one a jitted
``lax.scan``. At the start of an epoch :meth:`PositionMatcher.refresh` freezes
a :class:`MatchState` from the current positions; within the epoch it is held
fixed:

* **ξ.** ``membership=`` the epoch's start positions: the cell tree's leaf
  partition is built from them, every node geometry, acceptance decision and
  separation from the live positions (``docs/differentiating_a_tree_code.md``,
  Section 4). The estimator keeps the window rule's error bound, and a particle
  crossing a cell face inside the epoch no longer restructures the tree, so
  the objective has no restructuring jumps along an epoch (relevant to a line
  search). The traced build and walk still run at every step.
* **kNN.** The KD-tree search runs once per epoch and returns, for every query,
  its ``k_max + candidate_margin`` nearest particles. Every step recomputes the
  live distances to those candidates, sorts them and reads the ``k``-th. This
  is the exact ``k``-th neighbour as long as the true ``k_max`` nearest stay
  among the candidates (a Verlet list); :meth:`PositionMatcher.stale_fraction`
  measures how often they did not, and :func:`optimise_positions` records it
  at every refresh.

So ``rebuild_every`` trades the kNN search (and the membership refresh)
against staleness. ``rebuild_every=1`` is a fresh search every step.

Optimiser
---------
Any ``optax`` transformation works; :func:`lbfgs_positions` is the one to use.
On a Thomas target at ``N = 3e4`` it reaches a given loss in about half the
loss evaluations Adam needs down to ``1e-3``, and goes on past the loss of an
independent target realisation where Adam levels off. Its parts:

* the line search's accepted value and gradient are reused as the next step's
  (``reuse_linesearch_value``), so a step costs about 1.3 evaluations, not
  2.2;
* :func:`cap_displacement` clips each particle's step (``0.25 l`` per step
  bounds the moves to about ``l`` per epoch of 10, against ``5 l`` uncapped,
  at nearly the uncapped rate; ``0.1 l`` is slower);
* the curvature memory is kept across refreshes: clearing it
  (``reset_on_refresh``) leaves the loss 3-9 times higher at a matched number
  of evaluations, and a refresh every step converges no better per evaluation
  than every 10;
* a trial point whose traced walk overflows has an infinite loss, so the line
  search rejects it.

Capacities
----------
The traced ξ has static capacities (``TracedPairCountConfig``), planned by
probing the start and the target catalogue. A catalogue on its way between
them can need more (clustering raises the near-field pair count): every step
reports ``overflow``, and :func:`optimise_positions` re-plans on the epoch's
start and end positions and repeats the epoch when it fires, so no step with a
truncated walk is kept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from yggdrax.applications.corrfunc.traced import (
    TracedPairCountConfig,
    plan_traced_pair_counts,
    traced_soft_pair_counts,
)
from yggdrax.applications.knn_stats.estimator import (
    hard_knn_cdf,
    knn_indices,
    knn_log_distances,
    soft_cdf_from_log_distances,
)
from yggdrax.applications.point_process.statistics import natural_estimator_rr

__all__ = [
    "LossTerms",
    "MatchSpec",
    "MatchState",
    "MatchStatistics",
    "OptimiseResult",
    "PositionMatcher",
    "cap_displacement",
    "default_spec",
    "lbfgs_positions",
    "optimise_positions",
]

QueryMode = Literal["randoms", "data", "both"]


@dataclass(frozen=True)
class MatchSpec:
    """The statistics a catalogue is matched on (static, hashable).

    Attributes
    ----------
    box_size:
        Periodic box length ``L``.
    xi_edges:
        ``(nbins + 1,)`` increasing ξ bin edges; the window's hard cut must stay
        below ``L / 2``.
    knn_ks:
        Neighbour orders of the kNN-CDFs (1-based).
    knn_r_grid:
        Radii at which each kNN-CDF is read, below ``L / 2``.
    xi_sharpness:
        Soft-window sharpness of ξ in ``ln r``.
    window_eps:
        Window tolerance of the traced estimator (and of ``RR``'s hard cut).
    knn_sharpness:
        Soft-step sharpness of the kNN-CDFs in ``ln r``.
    query_mode:
        kNN queries: fixed ``"randoms"``, the particles themselves
        (``"data"``), or ``"both"``.
    candidate_margin:
        Extra neighbours kept per query beyond ``max(knn_ks)`` between kNN
        searches.
    kdtree_leaf_size:
        KD-tree leaf size of the kNN search.
    xi_weight:
        Weight of ``L_xi`` in the loss.
    knn_weight:
        Weight of ``L_knn`` in the loss.
    """

    box_size: float
    xi_edges: tuple[float, ...]
    knn_ks: tuple[int, ...]
    knn_r_grid: tuple[float, ...]
    xi_sharpness: float = 40.0
    window_eps: float = 1e-6
    knn_sharpness: float = 8.0
    query_mode: QueryMode = "both"
    candidate_margin: int = 8
    kdtree_leaf_size: int = 32
    xi_weight: float = 1.0
    knn_weight: float = 1.0

    @property
    def num_candidates(self) -> int:
        """Neighbours kept per query between searches."""
        return max(self.knn_ks) + int(self.candidate_margin)

    @property
    def use_randoms(self) -> bool:
        """Whether random queries are part of the kNN statistic."""
        return self.query_mode in ("randoms", "both")

    @property
    def use_data(self) -> bool:
        """Whether data (self-excluded) queries are part of the kNN statistic."""
        return self.query_mode in ("data", "both")


class MatchState(NamedTuple):
    """What is frozen between two refreshes (no gradient flows into it).

    Attributes
    ----------
    membership:
        ``(N, 3)`` positions that fix the ξ tree's leaf partition.
    random_candidates:
        ``(Q, C)`` candidate neighbours of the random queries (``Q = 0``
        without random queries).
    data_candidates:
        ``(N, C)`` candidate neighbours of each particle, itself excluded
        (zero rows without data queries).
    """

    membership: Array
    random_candidates: Array
    data_candidates: Array


class MatchStatistics(NamedTuple):
    """The statistics of one catalogue.

    Attributes
    ----------
    counts:
        ``(nbins,)`` soft pair counts ``DD``.
    xi:
        ``(nbins,)`` ``DD / RR - 1``.
    cdf:
        ``(n_sets, len(knn_ks), len(knn_r_grid))`` soft kNN-CDFs, randoms
        first when both query sets are used.
    overflow:
        Bool scalar: a traced capacity was too small.
    """

    counts: Array
    xi: Array
    cdf: Array
    overflow: Array


class LossTerms(NamedTuple):
    """The loss and its parts.

    Attributes
    ----------
    loss:
        ``xi_weight * xi + knn_weight * knn``.
    xi:
        ``L_xi``, unweighted.
    knn:
        ``L_knn``, unweighted.
    overflow:
        Bool scalar: a traced capacity was too small.
    """

    loss: Array
    xi: Array
    knn: Array
    overflow: Array


def _sorted_candidate_log_r(
    data: Array,
    queries: Array,
    candidates: Array,
    ks: tuple[int, ...],
    box_size: float,
) -> Array:
    """``ln r_k`` from live distances to frozen candidates, ``(q, len(ks))``."""
    c = int(candidates.shape[1])
    log_r = knn_log_distances(
        data, queries, candidates, tuple(range(1, c + 1)), box_size=box_size
    )
    log_r = jnp.sort(log_r, axis=1)
    return log_r[:, jnp.asarray([k - 1 for k in ks], dtype=jnp.int32)]


class PositionMatcher:
    """The loss of a catalogue against a target's ξ and kNN-CDFs.

    Build it with :meth:`build`. Every method is a pure function of arrays and
    can be called under ``jax.jit``; the static parts (spec, traced
    capacities, ``RR``, queries, target statistics) live on the instance.

    Parameters
    ----------
    spec:
        The statistics.
    traced_config:
        Static capacities of the traced ξ.
    randoms:
        ``(Q, 3)`` fixed random queries, or ``None`` without them.
    rr:
        ``(nbins,)`` analytic ``RR``.
    target:
        The target catalogue's statistics.
    n_points:
        Particles per catalogue.
    """

    def __init__(
        self,
        spec: MatchSpec,
        traced_config: TracedPairCountConfig,
        randoms: Optional[Array],
        rr: np.ndarray,
        target: Optional[MatchStatistics],
        n_points: int,
    ) -> None:
        self.spec = spec
        self.traced_config = traced_config
        self.randoms = randoms
        self.rr = rr
        self.n_points = int(n_points)
        self._target = target

    @property
    def target(self) -> MatchStatistics:
        """The target catalogue's statistics."""
        if self._target is None:
            raise RuntimeError("the matcher has no target statistics yet")
        return self._target

    # ------------------------------------------------------------------ setup

    @classmethod
    def build(
        cls,
        spec: MatchSpec,
        target_positions: Array,
        *,
        probes: Sequence[Array] = (),
        randoms: Optional[Array] = None,
        traced_config: Optional[TracedPairCountConfig] = None,
        leaf_size: int = 32,
    ) -> "PositionMatcher":
        """Plan the capacities and compute the target statistics.

        Parameters
        ----------
        spec:
            The statistics.
        target_positions:
            ``(N, 3)`` target catalogue.
        probes:
            Further catalogues the capacities must cover (the start, say).
        randoms:
            ``(Q, 3)`` fixed random queries; required by
            ``spec.query_mode`` ``"randoms"`` and ``"both"``.
        traced_config:
            Capacities of the traced ξ; planned on the target and ``probes``
            when ``None``.
        leaf_size:
            Cell-tree leaf size of the traced ξ, when planning.

        Returns
        -------
        PositionMatcher
            Ready to evaluate.

        Raises
        ------
        ValueError
            If random queries are needed and missing, a probe's size differs
            from the target's, or a radius reaches ``L / 2``.
        RuntimeError
            If the capacities overflow on the target.
        """
        n = int(target_positions.shape[0])
        if spec.use_randoms and randoms is None:
            raise ValueError(f"query_mode={spec.query_mode!r} needs randoms")
        for p in probes:
            if int(p.shape[0]) != n:
                raise ValueError("every catalogue must have the target's size")
        if max(spec.knn_r_grid) >= 0.5 * spec.box_size:
            raise ValueError("knn_r_grid reaches L / 2")
        if spec.num_candidates > n - 1:
            raise ValueError("max(knn_ks) + candidate_margin must be below N")
        rr = natural_estimator_rr(
            np.asarray(spec.xi_edges),
            box_size=spec.box_size,
            n_points=n,
            sharpness=spec.xi_sharpness,
            eps=spec.window_eps,
        )
        if traced_config is None:
            traced_config = cls.plan(
                spec, [target_positions, *probes], leaf_size=leaf_size
            )
        rq = None if not spec.use_randoms else jnp.asarray(randoms)
        matcher = cls(spec, traced_config, rq, rr, None, n)
        target = jax.jit(matcher.fresh_statistics)(target_positions)
        if bool(target.overflow):
            raise RuntimeError("the planned capacities overflow on the target")
        matcher._target = target
        return matcher

    @staticmethod
    def plan(
        spec: MatchSpec, catalogues: Sequence[Array], *, leaf_size: int = 32
    ) -> TracedPairCountConfig:
        """Plan the traced capacities on ``catalogues``.

        Parameters
        ----------
        spec:
            The statistics.
        catalogues:
            Catalogues the capacities must cover.
        leaf_size:
            Cell-tree leaf size.

        Returns
        -------
        TracedPairCountConfig
            Capacities covering every catalogue.
        """
        dtype = catalogues[0].dtype
        return plan_traced_pair_counts(
            [jnp.asarray(c) for c in catalogues],
            edges=jnp.asarray(spec.xi_edges, dtype),
            sharpness=spec.xi_sharpness,
            window_eps=spec.window_eps,
            leaf_size=leaf_size,
            box_size=spec.box_size,
        )

    def with_traced_config(self, config: TracedPairCountConfig) -> "PositionMatcher":
        """The same matcher with other traced capacities.

        Parameters
        ----------
        config:
            New capacities.

        Returns
        -------
        PositionMatcher
            A copy sharing the target statistics.
        """
        return PositionMatcher(
            self.spec, config, self.randoms, self.rr, self._target, self.n_points
        )

    def replan(self, catalogues: Sequence[Array]) -> "PositionMatcher":
        """Re-plan the traced capacities to cover ``catalogues`` as well.

        Each capacity is the larger of the current one and the new plan's.

        Parameters
        ----------
        catalogues:
            Catalogues that overflowed (or are about to be visited).

        Returns
        -------
        PositionMatcher
            A copy with the larger capacities.
        """
        new = self.plan(self.spec, catalogues, leaf_size=self.traced_config.leaf_size)
        old = self.traced_config
        grown = replace(
            new,
            leaf_capacity=max(new.leaf_capacity, old.leaf_capacity),
            max_pair_queue=max(new.max_pair_queue, old.max_pair_queue),
            far_cap=max(new.far_cap, old.far_cap),
            near_cap=max(new.near_cap, old.near_cap),
            band_cap=max(new.band_cap, old.band_cap),
            sub_pair_cap=max(new.sub_pair_cap, old.sub_pair_cap),
        )
        return self.with_traced_config(grown)

    # ------------------------------------------------------------ evaluation

    def refresh(self, positions: Array) -> MatchState:
        """Freeze the leaf partition and the kNN candidates at ``positions``.

        Parameters
        ----------
        positions:
            ``(N, 3)`` current catalogue (any coordinates; periodic).

        Returns
        -------
        MatchState
            Constants (no gradient).
        """
        spec = self.spec
        pos = jax.lax.stop_gradient(positions)
        c = spec.num_candidates
        leaf, box = spec.kdtree_leaf_size, spec.box_size
        if spec.use_randoms:
            assert self.randoms is not None
            rq = self.randoms.astype(pos.dtype)
            rand_c = knn_indices(pos, rq, c, leaf_size=leaf, box_size=box)
        else:
            rand_c = jnp.zeros((0, c), jnp.int32)
        if spec.use_data:
            data_c = knn_indices(
                pos, pos, c, exclude_self=True, leaf_size=leaf, box_size=box
            )
        else:
            data_c = jnp.zeros((0, c), jnp.int32)
        return MatchState(pos, rand_c, data_c)

    def _log_r(self, positions: Array, state: MatchState) -> list[Array]:
        spec = self.spec
        out = []
        if spec.use_randoms:
            assert self.randoms is not None
            out.append(
                _sorted_candidate_log_r(
                    positions,
                    self.randoms.astype(positions.dtype),
                    state.random_candidates,
                    spec.knn_ks,
                    spec.box_size,
                )
            )
        if spec.use_data:
            out.append(
                _sorted_candidate_log_r(
                    positions,
                    positions,
                    state.data_candidates,
                    spec.knn_ks,
                    spec.box_size,
                )
            )
        return out

    def statistics(self, positions: Array, state: MatchState) -> MatchStatistics:
        """ξ and kNN-CDFs of ``positions`` with ``state`` frozen.

        Parameters
        ----------
        positions:
            ``(N, 3)`` catalogue; differentiable.
        state:
            From :meth:`refresh`.

        Returns
        -------
        MatchStatistics
            The statistics.
        """
        spec = self.spec
        dtype = positions.dtype
        res = traced_soft_pair_counts(
            positions,
            jnp.asarray(spec.xi_edges, dtype),
            self.traced_config,
            sharpness=spec.xi_sharpness,
            membership=state.membership,
            box_size=spec.box_size,
        )
        xi = res.counts / jnp.asarray(self.rr, dtype) - 1.0
        r_grid = jnp.asarray(spec.knn_r_grid, dtype)
        cdf = jnp.stack(
            [
                soft_cdf_from_log_distances(lr, r_grid, spec.knn_sharpness)
                for lr in self._log_r(positions, state)
            ]
        )
        return MatchStatistics(res.counts, xi, cdf, res.overflow)

    def fresh_statistics(self, positions: Array) -> MatchStatistics:
        """Statistics with a state refreshed at ``positions`` (exact kNN).

        Parameters
        ----------
        positions:
            ``(N, 3)`` catalogue.

        Returns
        -------
        MatchStatistics
            The statistics.
        """
        return self.statistics(positions, self.refresh(positions))

    def loss_from_statistics(self, stats: MatchStatistics) -> LossTerms:
        """The loss of ``stats`` against the target.

        Parameters
        ----------
        stats:
            A catalogue's statistics.

        Returns
        -------
        LossTerms
            Total and parts.
        """
        t = self.target
        l_xi = jnp.mean(jnp.log((stats.counts + 1.0) / (t.counts + 1.0)) ** 2)
        l_knn = jnp.mean((stats.cdf - t.cdf) ** 2)
        total = self.spec.xi_weight * l_xi + self.spec.knn_weight * l_knn
        return LossTerms(total, l_xi, l_knn, stats.overflow)

    def loss(self, positions: Array, state: MatchState) -> tuple[Array, LossTerms]:
        """``(loss, terms)``; differentiate with ``has_aux=True``.

        Parameters
        ----------
        positions:
            ``(N, 3)`` catalogue; differentiable.
        state:
            From :meth:`refresh`.

        Returns
        -------
        tuple
            Scalar loss and its :class:`LossTerms`.
        """
        terms = self.loss_from_statistics(self.statistics(positions, state))
        return terms.loss, terms

    def knn_term(self, positions: Array, state: MatchState) -> Array:
        """``L_knn`` alone (unweighted): cheap, no ξ walk.

        Parameters
        ----------
        positions:
            ``(N, 3)`` catalogue.
        state:
            From :meth:`refresh`.

        Returns
        -------
        Array
            Scalar ``L_knn``.
        """
        r_grid = jnp.asarray(self.spec.knn_r_grid, positions.dtype)
        cdf = jnp.stack(
            [
                soft_cdf_from_log_distances(lr, r_grid, self.spec.knn_sharpness)
                for lr in self._log_r(positions, state)
            ]
        )
        return jnp.mean((cdf - self.target.cdf) ** 2)

    def stale_fraction(self, positions: Array, old: MatchState, new: MatchState):
        """Fraction of ``(query, k)`` whose ``r_k`` the old candidates got wrong.

        Parameters
        ----------
        positions:
            ``(N, 3)`` catalogue at which both states are compared.
        old:
            The state the last epoch ran with.
        new:
            A state refreshed at ``positions``.

        Returns
        -------
        Array
            Scalar in ``[0, 1]``: the old candidate list's ``k``-th distance
            exceeds the exact one (a true neighbour had left the list).
        """
        stale = self._log_r(positions, old)
        fresh = self._log_r(positions, new)
        wrong = [
            jnp.mean((s > f + 1e-12).astype(s.dtype)) for s, f in zip(stale, fresh)
        ]
        return jnp.mean(jnp.stack(wrong))

    def hard_cdf(self, positions: Array) -> Array:
        """Exact (step) kNN-CDFs, ``(n_sets, len(ks), len(r_grid))``.

        Parameters
        ----------
        positions:
            ``(N, 3)`` catalogue.

        Returns
        -------
        Array
            The hard CDFs, randoms first.
        """
        spec = self.spec
        r_grid = jnp.asarray(spec.knn_r_grid, positions.dtype)
        leaf, box = spec.kdtree_leaf_size, spec.box_size
        out = []
        queries = []
        if spec.use_randoms:
            assert self.randoms is not None
            queries.append(self.randoms.astype(positions.dtype))
        if spec.use_data:
            queries.append(None)
        for q in queries:
            out.append(
                hard_knn_cdf(
                    positions, q, spec.knn_ks, r_grid, leaf_size=leaf, box_size=box
                )
            )
        return jnp.stack(out)


# --------------------------------------------------------------------------
# the optimisation loop
# --------------------------------------------------------------------------


class OptimiseResult(NamedTuple):
    """What :func:`optimise_positions` returns.

    Attributes
    ----------
    positions:
        ``(N, 3)`` final catalogue, wrapped into ``[0, L)``.
    history:
        One record per step: ``step``, ``loss``, ``xi``, ``knn`` (evaluated at
        the step's start positions) and ``evals``, the loss-and-gradient
        evaluations the step made (one for a plain step; the line search's
        trials for L-BFGS, plus the step's own unless it was reused).
    epochs:
        One record per epoch: ``first_step``, ``seconds``, ``stale_fraction``
        (of the previous epoch's candidates at this epoch's start),
        ``max_displacement`` (largest move within the epoch), ``replans``.
    compile_seconds:
        Ahead-of-time compile time of the epoch function, summed over
        re-plans.
    seconds_per_step:
        Wall time of the kept epochs over their steps (compile excluded).
    matcher:
        The matcher with the capacities the run ended with.
    """

    positions: Array
    history: list[dict[str, float]]
    epochs: list[dict[str, Any]]
    compile_seconds: float
    seconds_per_step: float
    matcher: PositionMatcher


def cap_displacement(max_step: float, per_particle: bool = True) -> Any:
    """An ``optax`` transform capping how far any particle moves in a step.

    Particle ``i``'s 3-vector ``u_i`` is scaled by ``min(1, max_step /
    |u_i|)``: a particle proposed to move further is moved ``max_step`` along
    its own direction, the others are untouched. Chained before a line search
    whose ``max_learning_rate`` is 1 it caps every particle's step at
    ``max_step``, which keeps the frozen leaf partition and kNN candidates of
    an epoch (:meth:`PositionMatcher.refresh`) close to the live positions.

    ``per_particle=False`` scales the whole update by the one factor of its
    largest row instead (the direction is kept exactly). On this loss that
    stalls L-BFGS: the gradient is dominated by a few particles, one of them
    sets the factor, and every other particle barely moves.

    Parameters
    ----------
    max_step:
        Largest displacement of any particle per step (box units).
    per_particle:
        Clip each particle (default) or scale the update uniformly.

    Returns
    -------
    optax.GradientTransformation
        Stateless.
    """
    import optax

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        tiny = jnp.finfo(updates.dtype).tiny
        if per_particle:
            norm = jnp.sqrt(jnp.sum(updates * updates, axis=-1, keepdims=True))
        else:
            norm = jnp.sqrt(jnp.max(jnp.sum(updates * updates, axis=-1)))
        scale = jnp.minimum(1.0, max_step / jnp.maximum(norm, tiny))
        return updates * scale.astype(updates.dtype), state

    return optax.GradientTransformation(init_fn, update_fn)


def lbfgs_positions(
    *,
    max_step: Optional[float] = None,
    per_particle: bool = True,
    memory_size: int = 10,
    max_linesearch_steps: int = 20,
) -> Any:
    """``optax`` L-BFGS for positions, optionally with a displacement cap.

    ``scale_by_lbfgs``, then :func:`cap_displacement` when ``max_step`` is
    set, then a zoom line search starting at step 1 (``max_learning_rate=1``
    with the cap, so the accepted step never exceeds it).

    Parameters
    ----------
    max_step:
        Per-particle displacement cap per step, or ``None``: optax's
        ``lbfgs`` unchanged.
    per_particle:
        See :func:`cap_displacement`.
    memory_size:
        Curvature pairs kept.
    max_linesearch_steps:
        Line-search trials per step.

    Returns
    -------
    optax.GradientTransformationExtraArgs
        Needs ``value``, ``grad`` and ``value_fn`` at every update.
    """
    import optax

    if max_step is None:
        return optax.lbfgs(
            memory_size=memory_size,
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=max_linesearch_steps,
                initial_guess_strategy="one",
            ),
        )
    return optax.chain(
        optax.scale_by_lbfgs(memory_size=memory_size),
        optax.scale(-1.0),
        cap_displacement(max_step, per_particle),
        optax.scale_by_zoom_linesearch(
            max_linesearch_steps=max_linesearch_steps,
            max_learning_rate=1.0,
            initial_guess_strategy="one",
        ),
    )


def _tree_find(tree: Any, name: str) -> Any:
    """The first attribute ``name`` in a (nested) optax state, or ``None``."""
    stack = [tree]
    while stack:
        node = stack.pop(0)
        if hasattr(node, "_fields"):
            if name in node._fields:
                return getattr(node, name)
            stack.extend(getattr(node, f) for f in node._fields)
        elif isinstance(node, (tuple, list)):
            stack.extend(node)
    return None


def _make_epoch(
    matcher: PositionMatcher,
    optimizer: Any,
    rebuild_every: int,
    reuse_linesearch_value: bool,
):
    """The jitted epoch: refresh, then ``rebuild_every`` optimiser steps."""
    import optax

    spec = matcher.spec

    def epoch(pos, opt_state, prev_state):
        state = matcher.refresh(pos)
        stale = matcher.stale_fraction(pos, prev_state, state)
        value_and_grad = jax.value_and_grad(matcher.loss, has_aux=True)

        def value_fn(y):
            # A trial point whose traced walk overflows has a truncated loss;
            # infinity makes a line search reject it.
            value, terms = matcher.loss(y, state)
            return jnp.where(terms.overflow, jnp.inf, value)

        def fresh(x, s):
            del s
            (value, terms), grad = value_and_grad(x, state)
            return value, grad, terms.knn, terms.overflow, jnp.asarray(1, jnp.int32)

        def reused(x, s):
            # The line search's last trial is the accepted point: its value and
            # gradient on this epoch's frozen state are in the optimiser state.
            value = _tree_find(s, "value").astype(x.dtype)
            grad = _tree_find(s, "grad").astype(x.dtype)
            knn = matcher.knn_term(x, state)
            return value, grad, knn, jnp.asarray(False), jnp.asarray(0, jnp.int32)

        def body(carry, i):
            x, s = carry
            if reuse_linesearch_value:
                value, grad, l_knn, ovf, own = jax.lax.cond(i == 0, fresh, reused, x, s)
            else:
                value, grad, l_knn, ovf, own = fresh(x, s)
            l_xi = (value - spec.knn_weight * l_knn) / spec.xi_weight
            updates, s = optimizer.update(
                grad, s, x, value=value, grad=grad, value_fn=value_fn
            )
            ls = _tree_find(s, "num_linesearch_steps")
            evals = own + (
                jnp.asarray(0, jnp.int32) if ls is None else ls.astype(jnp.int32)
            )
            x = optax.apply_updates(x, updates)
            return (x, s), (value, l_xi, l_knn, ovf, evals)

        (pos_end, opt_state), (loss, l_xi, l_knn, ovf, evals) = jax.lax.scan(
            body, (pos, opt_state), jnp.arange(rebuild_every)
        )
        disp = pos_end - pos
        max_disp = jnp.sqrt(jnp.max(jnp.sum(disp * disp, axis=1)))
        return (
            pos_end,
            opt_state,
            state,
            loss,
            l_xi,
            l_knn,
            jnp.any(ovf),
            stale,
            max_disp,
            evals,
        )

    return epoch


def optimise_positions(
    matcher: PositionMatcher,
    initial: Array,
    optimizer: Any,
    *,
    num_steps: int,
    rebuild_every: int = 10,
    max_replans: int = 4,
    reset_on_refresh: bool = False,
    reuse_linesearch_value: Optional[bool] = None,
    log: Optional[Callable[[str], None]] = None,
) -> OptimiseResult:
    """Descend on the matching loss with respect to every particle coordinate.

    The parameters are the raw coordinates; they are never wrapped during the
    run (every estimator is minimum-image, so they need not lie in the box,
    and a wrap would corrupt an L-BFGS curvature history). The result is
    wrapped.

    Parameters
    ----------
    matcher:
        Loss and target (:meth:`PositionMatcher.build`).
    initial:
        ``(N, 3)`` start catalogue.
    optimizer:
        An ``optax`` gradient transformation, e.g. ``optax.adam(lr)`` or
        :func:`lbfgs_positions`; it gets ``value``, ``grad`` and ``value_fn``
        (which is infinite where the traced walk overflows).
    num_steps:
        Steps; rounded up to a multiple of ``rebuild_every``.
    rebuild_every:
        Steps per epoch: the kNN search and the ξ leaf partition are
        refreshed once per epoch (see the module docstring).
    max_replans:
        Re-plans of the traced capacities allowed over the run.
    reset_on_refresh:
        Re-initialise the optimiser state at every epoch, so an L-BFGS
        memory never mixes curvature pairs from two frozen states.
    reuse_linesearch_value:
        Take each step's loss and gradient from the line search's accepted
        trial (all but an epoch's first step), saving one evaluation per
        step. ``None``: when the optimiser state carries a line search.
    log:
        Optional line logger (``print``).

    Returns
    -------
    OptimiseResult
        Final positions, per-step and per-epoch records, timings.

    Raises
    ------
    RuntimeError
        If the capacities still overflow after ``max_replans`` re-plans.
    """
    from yggdrax.periodic import wrap_positions

    k = int(rebuild_every)
    num_epochs = -(-int(num_steps) // k)
    pos = jnp.asarray(initial)
    opt_state = optimizer.init(pos)
    if reuse_linesearch_value is None:
        reuse_linesearch_value = (
            _tree_find(opt_state, "num_linesearch_steps") is not None
        )
    prev_state = jax.jit(matcher.refresh)(pos)

    compile_s = 0.0
    cache: dict[Any, Any] = {}

    def run_epoch(m: PositionMatcher, *args):
        # Compiled ahead of time so the compile is timed apart from the run.
        # An optimiser state can change its leaves' dtypes after the first
        # update (optax's L-BFGS: a weak int64 counter comes back int32), so
        # the executables are cached per argument signature.
        nonlocal compile_s
        sig = (
            id(m),
            jax.tree.structure(args),
            tuple(
                (jnp.shape(a), jnp.result_type(a), getattr(a, "weak_type", False))
                for a in jax.tree.leaves(args)
            ),
        )
        if sig not in cache:
            t0 = time.perf_counter()
            fn = _make_epoch(m, optimizer, k, bool(reuse_linesearch_value))
            cache[sig] = jax.jit(fn).lower(*args).compile()
            compile_s += time.perf_counter() - t0
        return cache[sig](*args)

    history: list[dict[str, float]] = []
    epochs: list[dict[str, Any]] = []
    replans = 0
    run_s = 0.0
    e = 0
    while e < num_epochs:
        if reset_on_refresh and e > 0:
            opt_state = optimizer.init(pos)
        c0, t0 = compile_s, time.perf_counter()
        out = run_epoch(matcher, pos, opt_state, prev_state)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0 - (compile_s - c0)
        pos_end, opt_end, state, loss, l_xi, l_knn, ovf, stale, max_disp, evals = out
        if bool(ovf):
            if replans >= max_replans:
                raise RuntimeError(
                    f"traced capacities overflow at epoch {e} after {replans} re-plans"
                )
            replans += 1
            matcher = matcher.replan([pos, pos_end])
            if log is not None:
                log(f"epoch {e}: overflow, re-planned ({replans}) and repeating")
            continue
        run_s += dt
        loss_h, xi_h, knn_h, ev_h = (np.asarray(a) for a in (loss, l_xi, l_knn, evals))
        for i in range(k):
            history.append(
                {
                    "step": e * k + i,
                    "loss": float(loss_h[i]),
                    "xi": float(xi_h[i]),
                    "knn": float(knn_h[i]),
                    "evals": int(ev_h[i]),
                }
            )
        epochs.append(
            {
                "first_step": e * k,
                "seconds": dt,
                "stale_fraction": float(stale),
                "max_displacement": float(max_disp),
                "replans": replans,
            }
        )
        if log is not None and (e == 0 or (e + 1) % max(1, num_epochs // 20) == 0):
            log(
                f"step {e * k:6d}  loss {loss_h[0]:.4e}  xi {xi_h[0]:.3e}  "
                f"knn {knn_h[0]:.3e}  stale {float(stale):.2e}  "
                f"evals/step {ev_h.mean():.2f}  {dt / k * 1e3:.1f} ms/step"
            )
        pos, opt_state, prev_state = pos_end, opt_end, state
        e += 1

    final = wrap_positions(pos, matcher.spec.box_size)
    return OptimiseResult(
        positions=final,
        history=history,
        epochs=epochs,
        compile_seconds=compile_s,
        seconds_per_step=run_s / (num_epochs * k),
        matcher=matcher,
    )


def default_spec(
    box_size: float,
    *,
    mean_spacing: float,
    query_mode: QueryMode = "both",
    knn_ks: tuple[int, ...] = (1, 2, 4, 8),
) -> MatchSpec:
    """A spec scaled to the mean inter-particle spacing ``l = n^(-1/3)``.

    ξ: 10 log bins over ``[0.12 l, 2.5 l]`` at sharpness 40; kNN-CDFs: 24 log
    radii over ``[0.06 l, 4.5 l]`` at sharpness 8 (``1/s`` about the grid
    spacing in ``ln r``).

    Parameters
    ----------
    box_size:
        Periodic box length.
    mean_spacing:
        ``(L^3 / N)^(1/3)``.
    query_mode:
        kNN query sets.
    knn_ks:
        Neighbour orders.

    Returns
    -------
    MatchSpec
        The spec.
    """
    ell = float(mean_spacing)
    edges = tuple(float(x) for x in np.geomspace(0.12 * ell, 2.5 * ell, 11))
    grid = tuple(float(x) for x in np.geomspace(0.06 * ell, 4.5 * ell, 24))
    return MatchSpec(
        box_size=float(box_size),
        xi_edges=edges,
        knn_ks=tuple(knn_ks),
        knn_r_grid=grid,
        query_mode=query_mode,
    )
