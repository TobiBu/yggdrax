"""Dehnen (2014) eq (16a) as a per-pair test inside the flat mutual walks.

The error-controlled acceptance criterion of Dehnen (2014, arXiv:1405.2255) needs
each node's multipole powers, which a tree library does not compute. So the
criterion is split in two:

* the CALLER builds, once per walk, a ``[nodes, W]`` table holding everything
  eq (16a) reads about a node, in the column layout fixed here: ``G M`` at
  :data:`WALK_TABLE_GM`, the node's acceptance threshold ``eps * min_b f_b`` (a
  force) at :data:`WALK_TABLE_THRESHOLD`, and the normalised multipole powers
  ``s_n = P_n / (M rho^n)``, ``n = 1..p`` (eq 12, ``rho`` the walk's own MAC
  radius), from :data:`WALK_TABLE_POWER0` on. ``W`` is
  :func:`walk_table_width`, a power of two (Triton splits only those). jaccpot's
  ``dehnen_walk_table`` is such a builder.
* :func:`dehnen_pair_accept` evaluates eq (15)/(16a) for one pair, symmetrised,
  from two gathered rows. It is elementwise only -- no gathers, no reductions --
  so the same function runs inside the Pallas walk
  (:func:`yggdrax.pallas.mutual_walk.mutual_walk_pallas`, ``error_order > 0``) on
  a block of lanes and in :func:`yggdrax.interactions.dual_tree_walk_mutual`
  (through a ``pair_accept`` callable) on a whole round, so the two walks make the
  same decisions.

Eq (15) in the normalised form: with ``a = rho_A / r`` and ``b = rho_B / r``,

    E_{A->B} = 8 max(rho_A, rho_B) / (rho_A + rho_B)
               * sum_{n=0}^{p} C(p, n) s_n(A) a^n b^(p-n),        s_0 = 1,

which equals the paper's ``sum C(p,n) P_n rho_B^(p-n) / (M_A r^p)`` term by term.
``s_n`` lies in ``[0, 1]`` whenever ``rho`` bounds the node's particles about its
expansion centre (``P_n <= M rho^n`` by the triangle inequality), so the sum is
well scaled in float32 at any node size; the paper's form needs ``r^p`` and
``rho^n`` separately. eq (16a) then accepts when, in BOTH directions,

    G M_A E_{A->B} / r^2 < thr_B   and   rho_A + rho_B < theta_max r.

Symmetrised because the flat walk visits each unordered pair once and uses one
decision for both directions. There is no ``d > 0`` guard: at ``r = 0`` the
convergence clause already fails.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import jax.numpy as jnp
from jaxtyping import Array

__all__ = [
    "WALK_TABLE_GM",
    "WALK_TABLE_POWER0",
    "WALK_TABLE_THRESHOLD",
    "dehnen_pair_accept",
    "walk_table_width",
]

#: column of ``G M`` in the walk table
WALK_TABLE_GM = 0
#: column of the node's acceptance threshold ``eps * min_b f_b`` (a force)
WALK_TABLE_THRESHOLD = 1
#: column of ``s_1``; ``s_n`` sits at ``WALK_TABLE_POWER0 + n - 1``
WALK_TABLE_POWER0 = 2

#: the improvement factor of eq (15): ``8 max(rho) / (rho_A + rho_B)``
_IMPROVEMENT = 8.0
#: floor of every division
_TINY = 1e-24


def walk_table_width(order: int) -> int:
    """Columns of the eq (16a) walk table for expansion order ``order``.

    ``G M``, the threshold and ``s_1..s_p``, padded to a power of two.

    Parameters
    ----------
    order
        Expansion order ``p``.

    Returns
    -------
    int
        The table width, a power of two ``>= order + 2``.
    """
    need = int(order) + WALK_TABLE_POWER0
    return 1 << max(0, (need - 1).bit_length())


def _powers(x: Array, p: int) -> list[Array]:
    out = [jnp.ones_like(x)]
    for _ in range(p):
        out.append(out[-1] * x)
    return out


def dehnen_pair_accept(
    *,
    row_a: Sequence[Array],
    row_b: Sequence[Array],
    radius_a: Array,
    radius_b: Array,
    dist_sq: Array,
    order: int,
    theta_max: Any,
) -> Array:
    """Eq (16a), symmetrised, for pairs given their two gathered table rows.

    Elementwise in every argument, so it runs on a block of lanes inside Pallas
    as well as on a whole round in XLA.

    Parameters
    ----------
    row_a
        Node A's table columns ``0..p+1`` (layout in the module docstring).
    row_b
        Node B's table columns.
    radius_a
        A's MAC radius.
    radius_b
        B's MAC radius.
    dist_sq
        Squared centre distance.
    order
        Expansion order ``p``. Static.
    theta_max
        The convergence clause's bound (the paper's 1).

    Returns
    -------
    Array
        True where both directions pass eq (16a) and the expansion converges.
    """
    p = int(order)
    dtype = dist_sq.dtype
    tiny = jnp.asarray(_TINY, dtype)
    r = jnp.sqrt(dist_sq)
    rsum = radius_a + radius_b
    convergent = rsum < theta_max * r
    inv_r = 1.0 / jnp.maximum(r, tiny)
    a = radius_a * inv_r
    b = radius_b * inv_r
    pa = _powers(a, p)
    pb = _powers(b, p)
    e_ab = pb[p]  # n = 0: s_0 = 1
    e_ba = pa[p]
    for n in range(1, p + 1):
        c = float(math.comb(p, n))
        e_ab = e_ab + c * row_a[WALK_TABLE_POWER0 + n - 1] * pa[n] * pb[p - n]
        e_ba = e_ba + c * row_b[WALK_TABLE_POWER0 + n - 1] * pb[n] * pa[p - n]
    improvement = (
        _IMPROVEMENT * jnp.maximum(radius_a, radius_b) / jnp.maximum(rsum, tiny)
    )
    scale = improvement * inv_r * inv_r
    ok_ab = row_a[WALK_TABLE_GM] * e_ab * scale < row_b[WALK_TABLE_THRESHOLD]
    ok_ba = row_b[WALK_TABLE_GM] * e_ba * scale < row_a[WALK_TABLE_THRESHOLD]
    return convergent & ok_ab & ok_ba
