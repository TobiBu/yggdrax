"""Periodic-box helpers: the minimum-image convention on a box ``[0, L)^d``.

A periodic box is described by ``box_size``: ``None`` (non-periodic, the default
everywhere), one positive length ``L`` for a cube, or one length per dimension.
Particles live in ``[0, L_k)`` along every axis; :func:`wrap_positions` maps
arbitrary coordinates there. Every distance in a periodic walk is the
*minimum-image* distance: each component of a displacement is shifted by a
multiple of ``L_k`` into ``[-L_k / 2, L_k / 2]``.

The walks take ``box_size`` as a *static* argument (it fixes the far-field
acceptance rule at trace time), so it is normalised here to a hashable tuple of
Python floats by :func:`normalize_box_size`. See ``docs/periodic_boundaries.md``
for the semantics of a periodic walk and its limits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from jaxtyping import Array

BoxSize = Union[float, int, Sequence[Union[float, int]]]
"""A periodic box: one length (a cube) or one length per dimension."""


def normalize_box_size(
    box_size: Optional[BoxSize], dim: Optional[int] = None
) -> Optional[tuple[float, ...]]:
    """Return ``box_size`` as a hashable tuple of positive floats, or ``None``.

    Parameters
    ----------
    box_size
        ``None``, a scalar box length, or a sequence of per-dimension lengths
        (a concrete array is accepted too). A traced value is rejected: the box
        is a static property of a walk.
    dim
        Spatial dimension. When given, a scalar is broadcast to ``dim`` lengths
        and a sequence must have exactly ``dim`` entries (or one, which is
        broadcast). When ``None`` the result has the length of the input.

    Returns
    -------
    tuple of float or None
        The per-dimension box lengths, or ``None`` for a non-periodic walk.

    Raises
    ------
    ValueError
        If a length is not positive and finite, or the number of lengths does
        not match ``dim``.
    TypeError
        If ``box_size`` is a traced JAX value.
    """

    if box_size is None:
        return None
    try:
        values = np.asarray(box_size, dtype=np.float64).reshape(-1)
    except Exception as exc:  # a tracer cannot be converted
        raise TypeError(
            "box_size must be a concrete float or sequence of floats (it is a "
            "static property of the walk), not a traced value"
        ) from exc
    if values.size == 0:
        raise ValueError("box_size must hold at least one length")
    if not all(math.isfinite(v) and v > 0.0 for v in values.tolist()):
        raise ValueError(f"box_size lengths must be positive and finite: {box_size}")
    lengths = tuple(float(v) for v in values.tolist())
    if dim is not None:
        if len(lengths) == 1:
            lengths = lengths * int(dim)
        elif len(lengths) != int(dim):
            raise ValueError(
                f"box_size has {len(lengths)} lengths but the positions have "
                f"dimension {dim}"
            )
    return lengths


def _box_array(box_size: BoxSize, like: Array) -> Array:
    lengths = normalize_box_size(box_size)
    assert lengths is not None
    return jnp.asarray(lengths, dtype=jnp.asarray(like).dtype)


def minimum_image(delta: Array, box_size: Optional[BoxSize]) -> Array:
    """Map displacements to their minimum image.

    Parameters
    ----------
    delta
        Displacements of shape ``(..., d)``.
    box_size
        Periodic box; ``None`` returns ``delta`` unchanged (the same array).

    Returns
    -------
    Array
        ``delta - L * round(delta / L)`` per component: every component lies in
        ``[-L_k / 2, L_k / 2]``. Differentiable in ``delta`` (unit Jacobian
        away from the measure-zero half-box ties).
    """

    if box_size is None:
        return delta
    box = _box_array(box_size, delta)
    return delta - box * jnp.round(delta / box)


def periodic_displacement(
    x_from: Array, x_to: Array, box_size: Optional[BoxSize]
) -> Array:
    """Minimum-image displacement ``x_to - x_from``.

    Parameters
    ----------
    x_from
        Start points of shape ``(..., d)``.
    x_to
        End points, broadcastable against ``x_from``.
    box_size
        Periodic box; ``None`` gives the plain difference.

    Returns
    -------
    Array
        Minimum-image displacement of the broadcast shape.
    """

    return minimum_image(jnp.asarray(x_to) - jnp.asarray(x_from), box_size)


def periodic_distance_sq(x_a: Array, x_b: Array, box_size: Optional[BoxSize]) -> Array:
    """Squared minimum-image distance between ``x_a`` and ``x_b``.

    Parameters
    ----------
    x_a
        Points of shape ``(..., d)``.
    x_b
        Points broadcastable against ``x_a``.
    box_size
        Periodic box; ``None`` gives the plain squared distance.

    Returns
    -------
    Array
        Squared distances of shape ``(...)``.
    """

    delta = periodic_displacement(x_a, x_b, box_size)
    return jnp.sum(delta * delta, axis=-1)


def pairwise_periodic_distance_sq(
    x_a: Array, x_b: Array, box_size: Optional[BoxSize]
) -> Array:
    """All-pairs squared minimum-image distances, for near-field blocks.

    Parameters
    ----------
    x_a
        Points of shape ``(n, d)``.
    x_b
        Points of shape ``(m, d)``.
    box_size
        Periodic box; ``None`` gives plain squared distances.

    Returns
    -------
    Array
        ``(n, m)`` squared distances.
    """

    return periodic_distance_sq(
        jnp.asarray(x_a)[:, None, :], jnp.asarray(x_b)[None, :, :], box_size
    )


def wrap_positions(positions: Array, box_size: BoxSize) -> Array:
    """Wrap positions into the periodic box ``[0, L)^d``.

    Parameters
    ----------
    positions
        Coordinates of shape ``(..., d)``, anywhere in space.
    box_size
        Periodic box.

    Returns
    -------
    Array
        Coordinates in ``[0, L_k)`` along every axis. A value that rounds to
        ``L_k`` in floating point is mapped to ``0``, so the half-open interval
        holds exactly.
    """

    positions = jnp.asarray(positions)
    box = _box_array(box_size, positions)
    wrapped = positions - box * jnp.floor(positions / box)
    wrapped = jnp.where(wrapped >= box, wrapped - box, wrapped)
    return jnp.where(wrapped < 0, jnp.zeros_like(wrapped), wrapped)


def periodic_bounds(
    box_size: BoxSize, dim: int, dtype: DTypeLike = jnp.float64
) -> tuple[Array, Array]:
    """The fixed tree box ``([0]*d, [L_k])`` of a periodic build.

    Parameters
    ----------
    box_size
        Periodic box.
    dim
        Spatial dimension.
    dtype
        Coordinate dtype.

    Returns
    -------
    tuple of Array
        ``(min_corner, max_corner)``, each of shape ``(dim,)``.
    """

    lengths = normalize_box_size(box_size, dim)
    assert lengths is not None
    return jnp.zeros((dim,), dtype=dtype), jnp.asarray(lengths, dtype=dtype)


def far_pair_is_unambiguous(
    dist_sq: Array, radius_a: Array, radius_b: Array, box_size: BoxSize
) -> Array:
    """Whether a node pair's minimum image is the minimum image of all its pairs.

    With ``d`` the minimum-image distance of the node centres and ``r_a``,
    ``r_b`` radii that bound every particle's distance from its node centre,
    every particle pair's displacement through the *same* image has length at
    most ``d + r_a + r_b``. When that is below ``min_k L_k / 2`` each component is
    below ``L_k / 2``, so it is that particle pair's minimum image and no other
    image is closer: the node pair can be treated as one well-separated pair.
    Otherwise the walk refuses far acceptance and refines (or, for two leaves,
    emits a near pair evaluated particle by particle).

    Parameters
    ----------
    dist_sq
        Squared minimum-image distance of the node centres.
    radius_a
        Bounding radius of the first node.
    radius_b
        Bounding radius of the second node.
    box_size
        Periodic box.

    Returns
    -------
    Array
        Boolean mask, strict: ``r_a + r_b + d < min(L) / 2``.
    """

    lengths = normalize_box_size(box_size)
    assert lengths is not None
    half = jnp.asarray(0.5 * min(lengths), dtype=jnp.asarray(dist_sq).dtype)
    reach = radius_a + radius_b + jnp.sqrt(jnp.maximum(dist_sq, 0.0))
    return reach < half


__all__ = [
    "BoxSize",
    "far_pair_is_unambiguous",
    "minimum_image",
    "normalize_box_size",
    "pairwise_periodic_distance_sq",
    "periodic_bounds",
    "periodic_displacement",
    "periodic_distance_sq",
    "wrap_positions",
]
