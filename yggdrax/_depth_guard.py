"""The radix tree's depth bound, said loudly, and the domain numbers that explain it.

A radix tree's level tables (``level_offsets``, ``nodes_by_level``) hold
:data:`~yggdrax.tree.MAX_TREE_LEVELS` levels and ``num_levels`` is clipped to it:
a node deeper than that is in no level, so every level-ordered sweep (M2M, L2L)
skips it, SILENTLY. Over 63-bit Morton codes a tree of unique cell codes is at
most 63 deep; it gets deeper when many leaves share one level-21 cell (a wide
level-21 cell split into leaves of ``leaf_size``: each doubling of those leaves
adds a level). That happens when the Morton box is so large that a level-21
cell (box / 2**21) is wide compared with the dense centre -- the 1e8 Sgr IC's
sampler tails out to 30 Mpc made a 44 Mpc box, 22 pc cells and a tree 68 deep.

* :func:`domain_report`: the Morton box, the level-21 cell, the farthest
  particle and the 99.9th-percentile radius about the median centre -- the
  numbers a user needs to see that a few outliers set the box.
* :func:`check_tree_depth`: raises :class:`TreeDepthError` with them when a
  CONCRETE tree is deeper than the bound. Under a trace it returns ``None``
  (the caller folds ``depth > bound`` into its capacity verdict, as jaccpot's
  traced rebuild does): no program changes.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array

#: Morton bits per axis (63-bit codes): the finest cell is ``box / 2**21``.
MORTON_LEVELS = 21

__all__ = [
    "MORTON_LEVELS",
    "TreeDepthError",
    "check_tree_depth",
    "domain_report",
    "format_domain_report",
]


class TreeDepthError(ValueError):
    """A radix tree deeper than ``MAX_TREE_LEVELS``: its level tables cannot hold it.

    ``depth`` (the tree's number of levels, ``max(node_level) + 1``), ``bound``
    (``MAX_TREE_LEVELS``) and ``report`` (:func:`domain_report` of the positions,
    or the box alone) are attributes.
    """

    def __init__(self, message: str, *, depth: int, bound: int, report: dict):
        """Record the depth, the bound and the domain report.

        Parameters
        ----------
        message
            The error message.
        depth
            The tree's number of levels.
        bound
            The bound it passed.
        report
            The domain report.
        """
        super().__init__(message)
        self.depth = int(depth)
        self.bound = int(bound)
        self.report = dict(report)


def _is_concrete(x: Any) -> bool:
    return not isinstance(x, jax_core.Tracer)


def domain_report(
    positions: Array,
    bounds: Optional[tuple[Array, Array]] = None,
    *,
    quantile: float = 0.999,
) -> dict:
    """The Morton domain of ``positions``: box, level-21 cell, outliers.

    Host values. The centre is the per-axis MEDIAN position (robust against the
    outliers this report is about); radii are measured from it.

    Parameters
    ----------
    positions
        ``[N, 3]`` concrete positions.
    bounds
        The Morton box ``(lower, upper)``; ``None``: the positions' AABB (what
        ``infer_bounds`` gives, before any cubic or padded widening).
    quantile
        The radius quantile reported as ``r_quantile`` (default the 99.9th
        percentile).

    Returns
    -------
    dict
        ``box_lower``, ``box_upper`` (lists), ``box_extent`` (the largest side),
        ``cell21`` (``box_extent / 2**21``), ``centre`` (the median),
        ``r_max`` and ``farthest`` (index and position of the farthest particle),
        ``quantile`` and ``r_quantile``, ``dominance`` (``box_extent / (2
        r_quantile)``: how much larger the box is than the bulk's diameter),
        ``n_beyond`` (particles farther than ``10 r_quantile``) and ``n``.

    Raises
    ------
    ValueError
        If ``positions`` is not ``[N, 3]`` with ``N >= 1``.
    """
    pos = np.asarray(jax.device_get(positions), dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] == 0:
        raise ValueError(f"positions must be [N, 3] with N >= 1, got {pos.shape}")
    if bounds is None:
        lower, upper = pos.min(axis=0), pos.max(axis=0)
    else:
        lower = np.asarray(jax.device_get(bounds[0]), dtype=np.float64).reshape(3)
        upper = np.asarray(jax.device_get(bounds[1]), dtype=np.float64).reshape(3)
    extent = float(np.max(upper - lower))
    centre = np.median(pos, axis=0)
    radius = np.sqrt(np.sum((pos - centre) ** 2, axis=1))
    far = int(np.argmax(radius))
    r_q = float(np.quantile(radius, quantile))
    return {
        "n": int(pos.shape[0]),
        "box_lower": lower.tolist(),
        "box_upper": upper.tolist(),
        "box_extent": extent,
        "cell21": extent / float(1 << MORTON_LEVELS),
        "centre": centre.tolist(),
        "r_max": float(radius[far]),
        "farthest": {"index": far, "position": pos[far].tolist()},
        "quantile": float(quantile),
        "r_quantile": r_q,
        "dominance": extent / (2.0 * r_q) if r_q > 0 else float("inf"),
        "n_beyond": int(np.sum(radius > 10.0 * r_q)),
    }


def format_domain_report(report: dict) -> str:
    """One sentence of :func:`domain_report`'s numbers, for error messages.

    Parameters
    ----------
    report
        :func:`domain_report`.

    Returns
    -------
    str
        The box, the level-21 cell, the farthest particle and the quantile radius.
    """
    q = 100.0 * float(report["quantile"])
    far = report["farthest"]
    pos = ", ".join(f"{v:.6g}" for v in far["position"])
    return (
        f"Morton box {report['box_extent']:.6g} across (level-21 cell "
        f"{report['cell21']:.6g}); farthest particle #{far['index']} at "
        f"({pos}), {report['r_max']:.6g} from the median centre; the "
        f"{q:g}th-percentile radius is {report['r_quantile']:.6g}, so the box is "
        f"{report['dominance']:.3g}x the bulk's diameter, and "
        f"{report['n_beyond']} of {report['n']} particles lie beyond 10x that "
        "radius"
    )


def check_tree_depth(
    tree: Any,
    positions: Optional[Array] = None,
    *,
    bound: Optional[int] = None,
) -> Optional[int]:
    """Raise :class:`TreeDepthError` when a concrete tree is deeper than the bound.

    Parameters
    ----------
    tree
        A radix topology (``node_level``, or ``parent`` links).
    positions
        The particles the tree was built on: the error then names the box, the
        level-21 cell and the outliers (:func:`domain_report`, over the tree's
        own box when it has ``bounds_min`` / ``bounds_max``).
    bound
        The depth bound; ``None``: ``MAX_TREE_LEVELS``.

    Returns
    -------
    Optional[int]
        The tree's number of levels, or ``None`` when it is traced (nothing is
        checked then: the caller's traced verdict must fold the depth).

    Raises
    ------
    TreeDepthError
        If the tree has more levels than ``bound``.
    """
    from .tree import MAX_TREE_LEVELS, get_node_levels

    limit = int(MAX_TREE_LEVELS if bound is None else bound)
    levels = get_node_levels(tree)
    if not _is_concrete(levels) or int(levels.shape[0]) == 0:
        return None if not _is_concrete(levels) else 1
    depth = int(jax.device_get(jnp.max(levels))) + 1
    if depth <= limit:
        return depth
    report: dict = {}
    detail = ""
    lo = getattr(tree, "bounds_min", None)
    hi = getattr(tree, "bounds_max", None)
    box = None
    if lo is not None and hi is not None and _is_concrete(lo) and _is_concrete(hi):
        box = (lo, hi)
    if positions is not None and _is_concrete(positions):
        report = domain_report(positions, box)
        detail = " " + format_domain_report(report) + "."
    elif box is not None:
        extent = float(
            np.max(
                np.asarray(jax.device_get(box[1]), np.float64)
                - np.asarray(jax.device_get(box[0]), np.float64)
            )
        )
        report = {"box_extent": extent, "cell21": extent / float(1 << MORTON_LEVELS)}
        detail = (
            f" Morton box {extent:.6g} across (level-21 cell "
            f"{report['cell21']:.6g})."
        )
    raise TreeDepthError(
        f"radix tree {depth} levels deep, past the bound of {limit} "
        "(MAX_TREE_LEVELS): its level tables hold no node below that, so every "
        "level-ordered sweep (M2M, L2L) would silently skip them. Many leaves "
        "share a level-21 Morton cell: the box is too large for the dense "
        f"centre.{detail} Clip or remove the outlying particles (escapers), or "
        "build on a smaller explicit box.",
        depth=depth,
        bound=limit,
        report=report,
    )
