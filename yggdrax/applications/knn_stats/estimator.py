"""Differentiable k-nearest-neighbour CDFs on the yggdrax KD-tree.

The kNN-CDF of Banerjee & Abel (2021, MNRAS 500, 5479) summarises a point
catalogue by the distribution of distances from a set of query points to their
``k``-th nearest data point. For queries ``q = 1..Q`` and ``k = 1..K`` let
``r_k(q)`` be that distance; then

    CDF_k(r) = (1/Q) * sum_q 1[r_k(q) < r].

With volume-filling random queries this is the probability that a sphere of
radius ``r`` holds at least ``k`` data points, so it carries every connected
``n``-point function, not just the two-point one.

Values, not combinatorics
-------------------------
Two things in that definition are not differentiable: *which* point is a
query's ``k``-th neighbour (a discrete choice), and the indicator. They are
handled separately, the same way the pair-count estimator handles its tree.

1. **Neighbour identities are constants.** :func:`knn_indices` runs the exact
   traced KD-tree search (:func:`yggdrax.kdtree.query_neighbors`, ``"tree"``
   backend by default) on ``stop_gradient`` copies of the positions and returns
   integer indices. Nothing differentiates through the tree walk -- which is a
   ``while_loop`` and could not be reverse-differentiated anyway.
2. **Distances are values.** :func:`knn_distances` recomputes ``r_k(q)`` from
   the *live* positions by gathering the chosen neighbours, so gradients flow
   to every data coordinate (and to the query coordinates, if the caller
   differentiates with respect to them). Away from ties -- two neighbours at
   exactly the same distance -- the ``k``-th neighbour distance is a smooth
   function of the positions and this gradient is exact: it is what dense
   ``top_k`` autodiff gives, at tree cost.
3. **The indicator becomes a smooth step** in ``log r``:

       1[r_k < r]  ->  sigmoid(s * (log r - log r_k)).

Bias versus sharpness
---------------------
Averaged over queries, the smooth step convolves the exact CDF -- read as a
function of ``u = log r`` -- with a logistic kernel of scale ``1/s``:

    CDF_k^soft(u) = integral CDF_k(u - t) * s * sigmoid'(s t) dt.

The kernel is symmetric with variance ``pi^2 / (3 s^2)``, so the first-order
term cancels and the bias is

    CDF_k^soft(u) - CDF_k(u) ~ (pi^2 / (6 s^2)) * d^2 CDF_k / du^2,

second order in the smoothing width ``1/s`` (in e-folds of radius). It is
largest where the CDF bends most, at its foot and shoulder, and it is a bias of
the *estimator*, not noise: the same ``s`` applied to model and target cancels
it in a matching loss to leading order.

The price of a large ``s`` is in the gradient. Only queries with
``|log r - log r_k| <~ 1/s`` contribute to ``d CDF_k(r) / d x``, and their
weight scales as ``s``; so the gradient of a grid point rests on about
``Q * pdf_k(u) / s`` queries and its variance grows roughly linearly in ``s``.
The hard limit ``s -> inf`` has an exact zero gradient almost everywhere.

Practical choice: make ``1/s`` comparable to, or a little below, the spacing
of ``log r_grid``. Neighbouring grid points then see overlapping sets of
queries, every query moves some grid point, and the bias stays below the
change of the CDF between grid points. With 20-30 log-spaced points per decade
(``Delta u ~ 0.1``) that is ``s ~ 10-30``. This is the same trade-off as the
pair-count window's sharpness in
:mod:`yggdrax.applications.corrfunc.binning`, which also acts on ``log r``.

Periodic boxes
--------------
``box_size`` (a float ``L`` or one length per axis, static) switches every step
to the minimum-image convention of :mod:`yggdrax.periodic`
(``docs/periodic_boundaries.md``): the KD-tree is built on the data wrapped into
``[0, L)``, :func:`yggdrax.kdtree.query_neighbors` finds minimum-image
neighbours, and :func:`knn_log_distances` recomputes the distances with
:func:`~yggdrax.periodic.minimum_image`. The live positions need not lie in the
box -- a particle that drifts across a face during an optimisation keeps its
neighbours and its gradient -- and the gradient of the minimum image is the
identity away from the measure-zero half-box ties. Keep ``r_grid`` below
``min(L) / 2``, where the ``k``-th neighbour is unambiguous; a concrete grid
past it is refused.

Relation to :func:`yggdrax.distributed.stats.distributed_knn`
-------------------------------------------------------------
This module is the single-device *application* API: a whole catalogue on one
device, any ``ks``, the exact neighbours without a search radius, and ``jit`` /
``vmap`` over catalogues. ``distributed_knn`` is the multi-device *driver* for
catalogues that do not fit on one device: it shards the data over a mesh,
imports halos, searches within ``r_max`` and reduces the CDF with a ``psum``.
The two use the same estimator -- integer neighbours from frozen positions,
distances from live ones, ``sigmoid(s * (log r - log r_k))`` averaged over the
queries -- so on one device, with every ``r_k < r_max``, they give the same
CDF and the same gradient (a test pins this). The reduction is kept separate
on purpose: core code does not import from ``yggdrax.applications``, and the
distributed one has to run inside ``shard_map``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from yggdrax.dtypes import is_traced
from yggdrax.kdtree import build_kdtree, query_neighbors
from yggdrax.periodic import (
    BoxSize,
    minimum_image,
    normalize_box_size,
    wrap_positions,
)

KnnBackend = Literal["tree", "tiled", "dense", "auto"]


def _check_grid_in_box(r_grid: Array, box: Optional[tuple[float, ...]]) -> None:
    """Refuse a concrete radius grid that reaches half the periodic box.

    Parameters
    ----------
    r_grid
        Radii of the CDF grid.
    box
        Normalised periodic box, or ``None``.

    Raises
    ------
    ValueError
        If the box is periodic, ``r_grid`` is concrete and its largest radius
        is at least ``min(L) / 2``.
    """

    if box is None or is_traced(r_grid):
        return
    r_max = float(np.max(np.asarray(r_grid)))
    if r_max >= 0.5 * min(box):
        raise ValueError(
            f"r_grid reaches {r_max} but the k-th neighbour is only unambiguous "
            f"below min(box_size)/2 = {0.5 * min(box)}"
        )


def _normalise_ks(ks: Sequence[int] | int) -> tuple[int, ...]:
    """Return ``ks`` as a tuple of positive Python ints.

    Parameters
    ----------
    ks
        One neighbour order or a sequence of them (1-based, static).

    Returns
    -------
    tuple of int
        The orders, in the order given.

    Raises
    ------
    ValueError
        If ``ks`` is empty or holds an order below 1.
    """

    ks_tuple = (int(ks),) if isinstance(ks, int) else tuple(int(k) for k in ks)
    if not ks_tuple:
        raise ValueError("ks must name at least one neighbour order")
    if min(ks_tuple) < 1:
        raise ValueError(f"neighbour orders are 1-based; received ks={ks_tuple}")
    return ks_tuple


def knn_indices(
    data: Array,
    queries: Array,
    k_max: int,
    *,
    exclude_self: bool = False,
    backend: KnnBackend = "tree",
    leaf_size: int = 32,
    box_size: Optional[BoxSize] = None,
) -> Array:
    """Return the ``k_max`` nearest data points of every query, as constants.

    Builds a KD-tree over ``stop_gradient(data)`` and runs the exact traced
    search on ``stop_gradient(queries)``. The result is an integer array, so
    it carries no gradient; distances are recomputed from live positions by
    :func:`knn_distances`. Traceable: callable under ``jit`` and ``vmap``.

    Parameters
    ----------
    data
        Data points, shape ``(n, dim)``.
    queries
        Query points, shape ``(q, dim)``.
    k_max
        Number of neighbours to find (static), ``1 <= k_max <= n`` (``n - 1``
        when ``exclude_self``).
    exclude_self
        Skip each query's own index; use when ``queries`` *is* ``data``.
    backend
        Search backend of :func:`yggdrax.kdtree.query_neighbors`. ``"tree"``
        (default) is the pruned KD-tree walk; ``"dense"`` and ``"tiled"`` are
        brute-force references.
    leaf_size
        Leaf bucket size of the KD-tree.
    box_size
        Periodic box: a length ``L`` or one per axis (static), or ``None``.
        With a box the search runs on the data wrapped into ``[0, L)`` and
        returns minimum-image neighbours; ``data`` and ``queries`` may lie
        anywhere.

    Returns
    -------
    Array
        Neighbour indices into ``data``, shape ``(q, k_max)``, sorted by
        increasing distance.
    """

    box = normalize_box_size(box_size, int(data.shape[-1]))
    data_c = jax.lax.stop_gradient(data)
    queries_c = jax.lax.stop_gradient(queries)
    if box is not None:
        # The tree must hold points in [0, L); query_neighbors wraps the
        # queries itself. Wrapping moves no rows, so indices still index `data`.
        data_c = wrap_positions(data_c, box)
    tree = build_kdtree(data_c, leaf_size=leaf_size)
    indices, _ = query_neighbors(
        tree,
        queries_c,
        k=int(k_max),
        exclude_self=exclude_self,
        backend=backend,
        return_squared=True,
        box_size=box,
    )
    return jax.lax.stop_gradient(indices)


def knn_log_distances(
    data: Array,
    queries: Array,
    indices: Array,
    ks: Sequence[int] | int,
    *,
    box_size: Optional[BoxSize] = None,
) -> Array:
    """Return ``log r_k`` for each query and order, from live positions.

    The neighbour *identities* in ``indices`` are held fixed; the distances
    are recomputed by gathering ``data[indices]``, so this is differentiable
    in ``data`` and ``queries``. Working with ``0.5 * log(d^2)`` avoids the
    square root's infinite slope at zero separation. With a box the
    differences are minimum-image ones, so the positions may lie anywhere.

    Parameters
    ----------
    data
        Data points, shape ``(n, dim)``.
    queries
        Query points, shape ``(q, dim)``.
    indices
        Neighbour indices from :func:`knn_indices`, shape ``(q, k_max)``.
    ks
        Neighbour orders to return (1-based, static), each ``<= k_max``.
    box_size
        Periodic box (static), or ``None``; the one the indices were found
        with.

    Returns
    -------
    Array
        ``log r_k(q)``, shape ``(q, len(ks))``.
    """

    box = normalize_box_size(box_size, int(data.shape[-1]))
    ks_tuple = _normalise_ks(ks)
    columns = jnp.asarray([k - 1 for k in ks_tuple], dtype=jnp.int32)
    chosen = jnp.take(indices, columns, axis=1)  # (q, nk)
    delta = minimum_image(data[chosen] - queries[:, None, :], box)  # (q, nk, dim)
    d2 = jnp.sum(delta * delta, axis=-1)
    # A coincident pair has log r = -inf; floor it so the value and its
    # (zero) gradient stay finite.
    d2 = jnp.maximum(d2, jnp.finfo(d2.dtype).tiny)
    return 0.5 * jnp.log(d2)


def knn_distances(
    data: Array,
    queries: Array,
    indices: Array,
    ks: Sequence[int] | int,
    *,
    box_size: Optional[BoxSize] = None,
) -> Array:
    """Return ``r_k`` for each query and order, from live positions.

    Parameters
    ----------
    data
        Data points, shape ``(n, dim)``.
    queries
        Query points, shape ``(q, dim)``.
    indices
        Neighbour indices from :func:`knn_indices`, shape ``(q, k_max)``.
    ks
        Neighbour orders to return (1-based, static), each ``<= k_max``.
    box_size
        Periodic box (static), or ``None``.

    Returns
    -------
    Array
        ``r_k(q)``, shape ``(q, len(ks))``.
    """

    return jnp.exp(knn_log_distances(data, queries, indices, ks, box_size=box_size))


def soft_cdf_from_log_distances(
    log_r_k: Array, r_grid: Array, sharpness: float | Array
) -> Array:
    """Smooth empirical CDF of ``r_k`` on ``r_grid``.

    Parameters
    ----------
    log_r_k
        ``log r_k(q)``, shape ``(q, nk)``.
    r_grid
        Radii at which to evaluate the CDF, shape ``(g,)``, all ``> 0``.
    sharpness
        Inverse width ``s`` of the logistic step in ``log r`` (dimensionless;
        ``1/s`` is in e-folds of radius). See the module docstring for the
        bias/variance trade-off.

    Returns
    -------
    Array
        ``CDF_k(r)``, shape ``(nk, g)``.
    """

    log_grid = jnp.log(r_grid)
    gate = jax.nn.sigmoid(sharpness * (log_grid[None, None, :] - log_r_k[..., None]))
    return jnp.mean(gate, axis=0)


def hard_cdf_from_log_distances(log_r_k: Array, r_grid: Array) -> Array:
    """Exact (step-function) empirical CDF of ``r_k`` on ``r_grid``.

    The ``sharpness -> inf`` limit of :func:`soft_cdf_from_log_distances`,
    kept as the reference it converges to. Zero gradient almost everywhere.

    Parameters
    ----------
    log_r_k
        ``log r_k(q)``, shape ``(q, nk)``.
    r_grid
        Radii, shape ``(g,)``, all ``> 0``.

    Returns
    -------
    Array
        Fraction of queries with ``r_k < r``, shape ``(nk, g)``.
    """

    log_grid = jnp.log(r_grid)
    below = log_r_k[..., None] < log_grid[None, None, :]
    return jnp.mean(below.astype(log_r_k.dtype), axis=0)


def peaked_cdf(cdf: Array) -> Array:
    """Peaked CDF, ``min(CDF, 1 - CDF)``.

    Banerjee & Abel plot this to show both tails of each CDF on a log axis.

    Parameters
    ----------
    cdf
        CDF values, any shape.

    Returns
    -------
    Array
        The peaked CDF, same shape.
    """

    return jnp.minimum(cdf, 1.0 - cdf)


def _search_log_distances(
    data: Array,
    queries: Optional[Array],
    ks: Sequence[int] | int,
    r_grid: Array,
    *,
    backend: KnnBackend,
    leaf_size: int,
    box_size: Optional[BoxSize],
) -> Array:
    """Find the neighbours (constants) and return ``log r_k`` from live positions.

    Parameters
    ----------
    data
        Data points, shape ``(n, dim)``.
    queries
        Query points, or ``None`` for the data themselves (self excluded).
    ks
        Neighbour orders (1-based).
    r_grid
        The CDF grid, checked against the periodic box.
    backend
        Neighbour-search backend.
    leaf_size
        KD-tree leaf bucket size.
    box_size
        Periodic box, or ``None``.

    Returns
    -------
    Array
        ``log r_k(q)``, shape ``(q, len(ks))``.
    """

    ks_tuple = _normalise_ks(ks)
    _check_grid_in_box(r_grid, normalize_box_size(box_size, int(data.shape[-1])))
    queries_arr = data if queries is None else queries
    indices = knn_indices(
        data,
        queries_arr,
        max(ks_tuple),
        exclude_self=queries is None,
        backend=backend,
        leaf_size=leaf_size,
        box_size=box_size,
    )
    return knn_log_distances(data, queries_arr, indices, ks_tuple, box_size=box_size)


def knn_cdf(
    data: Array,
    queries: Optional[Array],
    ks: Sequence[int] | int,
    r_grid: Array,
    *,
    sharpness: float | Array,
    backend: KnnBackend = "tree",
    leaf_size: int = 32,
    return_peaked: bool = False,
    box_size: Optional[BoxSize] = None,
) -> Array | tuple[Array, Array]:
    """Differentiable kNN-CDFs of ``data`` seen from ``queries``.

    For every query and every ``k`` in ``ks`` the ``k``-th nearest data point
    is found with the KD-tree (no gradient); its distance is recomputed from
    the live positions and passed through a logistic step in ``log r``
    (see the module docstring). Shapes are static, so the call is ``jit``-,
    ``grad``- and ``vmap``-able as long as ``ks``, ``backend``, ``leaf_size``
    and ``return_peaked`` are static Python values.

    Parameters
    ----------
    data
        Data points, shape ``(n, dim)``.
    queries
        Query points, shape ``(q, dim)`` -- typically volume-filling randoms.
        ``None`` uses the data points themselves with each point's own index
        excluded, so ``CDF_1`` is the nearest-other-point distribution.
    ks
        Neighbour orders (1-based), e.g. ``(1, 2, 4, 8)``.
    r_grid
        Radii at which to evaluate each CDF, shape ``(g,)``, all ``> 0``.
        Typically log-spaced.
    sharpness
        Inverse width ``s`` of the smooth step in ``log r``. Larger is closer
        to the hard CDF (bias ``O(1/s^2)``) with a noisier, more localised
        gradient.
    backend
        Neighbour-search backend; ``"tree"`` is the pruned KD-tree walk.
    leaf_size
        KD-tree leaf bucket size.
    return_peaked
        Also return the peaked CDF, ``min(CDF, 1 - CDF)``.
    box_size
        Periodic box: a length ``L`` or one per axis (static), or ``None`` for
        open boundaries. Distances are minimum-image ones; the positions may
        lie anywhere. ``r_grid`` must stay below ``min(L) / 2`` (a concrete
        grid that reaches it raises ``ValueError``).

    Returns
    -------
    Array or tuple of Array
        ``cdf`` of shape ``(len(ks), g)``, or ``(cdf, peaked)`` when
        ``return_peaked``.
    """

    log_r_k = _search_log_distances(
        data,
        queries,
        ks,
        r_grid,
        backend=backend,
        leaf_size=leaf_size,
        box_size=box_size,
    )
    cdf = soft_cdf_from_log_distances(log_r_k, jnp.asarray(r_grid), sharpness)
    if return_peaked:
        return cdf, peaked_cdf(cdf)
    return cdf


def hard_knn_cdf(
    data: Array,
    queries: Optional[Array],
    ks: Sequence[int] | int,
    r_grid: Array,
    *,
    backend: KnnBackend = "tree",
    leaf_size: int = 32,
    box_size: Optional[BoxSize] = None,
) -> Array:
    """Exact kNN-CDFs (step function), the ``sharpness -> inf`` reference.

    Parameters
    ----------
    data
        Data points, shape ``(n, dim)``.
    queries
        Query points, shape ``(q, dim)``, or ``None`` for the data themselves
        (own index excluded).
    ks
        Neighbour orders (1-based).
    r_grid
        Radii, shape ``(g,)``, all ``> 0``.
    backend
        Neighbour-search backend.
    leaf_size
        KD-tree leaf bucket size.
    box_size
        Periodic box (static), or ``None``.

    Returns
    -------
    Array
        Fraction of queries with ``r_k < r``, shape ``(len(ks), g)``.
    """

    log_r_k = _search_log_distances(
        data,
        queries,
        ks,
        r_grid,
        backend=backend,
        leaf_size=leaf_size,
        box_size=box_size,
    )
    return hard_cdf_from_log_distances(log_r_k, jnp.asarray(r_grid))


def cdf_distance(
    cdf: Array,
    target_cdf: Array,
    *,
    peaked: bool = False,
    weights: Optional[Array] = None,
) -> Array:
    """Weighted mean squared difference between two sets of kNN-CDFs.

    Parameters
    ----------
    cdf
        Model CDFs, shape ``(nk, g)``.
    target_cdf
        Target CDFs, same shape (computed with the same ``ks``, ``r_grid`` and
        ``sharpness``, so the smoothing bias cancels to leading order).
    peaked
        Compare peaked CDFs instead, which weights each CDF's upper tail as
        much as its lower one.
    weights
        Optional non-negative weights broadcastable to ``(nk, g)``; the result
        is ``sum(w * diff^2) / sum(w)``.

    Returns
    -------
    Array
        Scalar loss.
    """

    if peaked:
        cdf = peaked_cdf(cdf)
        target_cdf = peaked_cdf(target_cdf)
    sq = (cdf - target_cdf) ** 2
    if weights is None:
        return jnp.mean(sq)
    w = jnp.broadcast_to(jnp.asarray(weights, dtype=sq.dtype), sq.shape)
    return jnp.sum(w * sq) / jnp.sum(w)


def knn_cdf_loss(
    data: Array,
    queries: Optional[Array],
    target_cdf: Array,
    ks: Sequence[int] | int,
    r_grid: Array,
    *,
    sharpness: float | Array,
    peaked: bool = False,
    weights: Optional[Array] = None,
    backend: KnnBackend = "tree",
    leaf_size: int = 32,
    box_size: Optional[BoxSize] = None,
) -> Array:
    """Loss for matching ``data``'s kNN-CDFs to ``target_cdf``.

    ``cdf_distance(knn_cdf(data, queries, ...), target_cdf)``; differentiate
    with respect to ``data`` (or anything ``data`` is a function of) to move
    the catalogue towards the target statistics.

    Parameters
    ----------
    data
        Data points, shape ``(n, dim)``.
    queries
        Query points, shape ``(q, dim)``, or ``None`` for the data themselves.
    target_cdf
        Target CDFs, shape ``(len(ks), g)``.
    ks
        Neighbour orders (1-based).
    r_grid
        Radii, shape ``(g,)``, all ``> 0``.
    sharpness
        Inverse step width in ``log r``; use the one the target was made with.
    peaked
        Compare peaked CDFs (see :func:`cdf_distance`).
    weights
        Optional weights broadcastable to ``(len(ks), g)``.
    backend
        Neighbour-search backend.
    leaf_size
        KD-tree leaf bucket size.
    box_size
        Periodic box (static), or ``None``.

    Returns
    -------
    Array
        Scalar loss.
    """

    cdf = knn_cdf(
        data,
        queries,
        ks,
        r_grid,
        sharpness=sharpness,
        backend=backend,
        leaf_size=leaf_size,
        box_size=box_size,
    )
    return cdf_distance(cast(Array, cdf), target_cdf, peaked=peaked, weights=weights)


__all__ = [
    "cdf_distance",
    "hard_cdf_from_log_distances",
    "hard_knn_cdf",
    "knn_cdf",
    "knn_cdf_loss",
    "knn_distances",
    "knn_indices",
    "knn_log_distances",
    "peaked_cdf",
    "soft_cdf_from_log_distances",
]
