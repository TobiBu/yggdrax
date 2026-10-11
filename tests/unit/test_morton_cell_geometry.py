"""``compute_tree_geometry``'s Morton-cell path (``use_morton_geometry=True``).

A backend that knows each leaf's Morton cell (``leaf_codes`` + ``leaf_depths``,
the optional ``MortonLeafBoundsProtocol`` of ``docs/backend_contract.md``) may set
``use_morton_geometry`` and get leaf boxes from the cells instead of from the
particles. Until the 2026-10 cleanup (Y2) the fixed-depth builders were the
in-tree producers of such topologies and these tests built their inputs with
them. No yggdrax builder sets the flag any more, so the inputs are now an
adaptive radix tree whose leaves are given their Morton cells explicitly.
"""

import jax.numpy as jnp
import numpy as np

from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.geometry import _MAX_MORTON_LEVEL, compute_tree_geometry
from yggdrax.morton import _compact3_u64
from yggdrax.tree import build_tree

_MORTON_BITS = 3 * _MAX_MORTON_LEVEL


def _with_morton_cell_leaves(tree, leaf_depths):
    """Return ``tree``'s topology with each leaf assigned its depth-``d`` cell.

    The cell code is the leaf's first particle code with the bits below depth
    ``d`` cleared. The caller is responsible for every particle of a leaf lying
    in that cell; :func:`_assert_leaves_lie_in_their_cells` checks it.
    """

    topology = tree.topology
    num_internal = int(topology.num_internal_nodes)
    first_codes = np.asarray(topology.morton_codes)[
        np.asarray(topology.node_ranges)[num_internal:, 0]
    ].astype(np.uint64)
    depths = np.asarray(leaf_depths, dtype=np.int64)
    shift = (_MORTON_BITS - 3 * depths).astype(np.uint64)
    cell_codes = (first_codes >> shift) << shift
    return topology._replace(
        leaf_codes=jnp.asarray(cell_codes, dtype=jnp.uint64),
        leaf_depths=jnp.asarray(depths, dtype=INDEX_DTYPE),
        use_morton_geometry=jnp.asarray(True),
    )


def _assert_leaves_lie_in_their_cells(topology):
    num_internal = int(topology.num_internal_nodes)
    codes = np.asarray(topology.morton_codes).astype(np.uint64)
    ranges = np.asarray(topology.node_ranges)[num_internal:]
    depths = np.asarray(topology.leaf_depths, dtype=np.int64)
    cells = np.asarray(topology.leaf_codes).astype(np.uint64)
    for (start, end), depth, cell in zip(ranges, depths, cells, strict=True):
        shift = np.uint64(_MORTON_BITS - 3 * int(depth))
        assert np.all((codes[int(start) : int(end) + 1] >> shift) << shift == cell)


def test_morton_cell_geometry_matches_decoded_cells():
    rng = np.random.default_rng(42)
    pos = rng.uniform(low=0.2, high=0.3, size=(32, 3))
    masses = np.ones((32,), dtype=np.float64)
    bounds = (jnp.zeros(3), jnp.ones(3))

    tree, pos_sorted, *_ = build_tree(
        jnp.asarray(pos),
        jnp.asarray(masses),
        bounds,
        leaf_size=1,
        return_reordered=True,
    )
    num_internal = int(tree.num_internal_nodes)
    num_leaves = int(tree.parent.shape[0]) - num_internal
    # One particle per leaf, so any depth is a cell holding the leaf; vary it
    # per leaf (1..5) so the decode is exercised at every depth at once.
    topology = _with_morton_cell_leaves(tree, 1 + np.arange(num_leaves) % 5)
    _assert_leaves_lie_in_their_cells(topology)

    geometry = compute_tree_geometry(topology, pos_sorted)

    leaf_depths = np.asarray(topology.leaf_depths).astype(np.int32)
    assert np.all(leaf_depths >= 0)

    leaf_half_extents = np.asarray(geometry.half_extent)[num_internal:]
    leaf_centers = np.asarray(geometry.center)[num_internal:]

    domain = np.asarray(bounds[1] - bounds[0])
    bounds_min = np.asarray(bounds[0])

    cell_sizes = domain[None, :] / (2.0 ** leaf_depths[:, None])
    expected_half_extents = 0.5 * cell_sizes

    np.testing.assert_allclose(
        leaf_half_extents,
        expected_half_extents,
        rtol=0,
        atol=1e-12,
    )

    leaf_codes = np.asarray(topology.leaf_codes, dtype=np.uint64)
    shift = (_MAX_MORTON_LEVEL - leaf_depths).astype(np.uint64)
    x_coords = _compact3_u64(leaf_codes)
    y_coords = _compact3_u64(leaf_codes >> np.uint64(1))
    z_coords = _compact3_u64(leaf_codes >> np.uint64(2))
    indices = np.stack(
        [x_coords >> shift, y_coords >> shift, z_coords >> shift],
        axis=1,
    ).astype(domain.dtype)
    expected_centers = bounds_min[None, :] + (indices + 0.5) * cell_sizes

    np.testing.assert_allclose(
        leaf_centers,
        expected_centers,
        rtol=0,
        atol=1e-12,
    )


def test_morton_cell_geometry_contains_particles():
    coords = jnp.linspace(-0.75, 0.75, 4)
    grid = jnp.meshgrid(coords, coords, coords)
    positions = jnp.stack(grid, axis=-1).reshape(-1, 3)
    masses = jnp.ones((positions.shape[0],))

    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    # Each octant of the 4^3 grid holds 2^3 points, contiguous in Morton order,
    # so leaves of 8 are exactly the eight depth-1 cells.
    tree, positions_sorted, *_ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=8,
        return_reordered=True,
    )
    num_internal = int(tree.num_internal_nodes)
    num_leaves = int(tree.parent.shape[0]) - num_internal
    assert num_leaves == 8
    topology = _with_morton_cell_leaves(tree, np.ones(num_leaves, dtype=np.int64))
    _assert_leaves_lie_in_their_cells(topology)

    geometry = compute_tree_geometry(topology, positions_sorted)
    leaf_ranges = np.asarray(topology.node_ranges[num_internal:])
    centers = np.asarray(geometry.center[num_internal:])
    half_extent = np.asarray(geometry.half_extent[num_internal:])
    positions_np = np.asarray(positions_sorted)
    eps = 1e-9

    np.testing.assert_allclose(half_extent, 0.5, rtol=0, atol=1e-12)
    for idx, (start, end) in enumerate(leaf_ranges):
        pts = positions_np[int(start) : int(end) + 1]
        assert pts.size > 0
        cell_min = centers[idx] - half_extent[idx]
        cell_max = centers[idx] + half_extent[idx]
        assert np.all(pts >= cell_min - eps)
        assert np.all(pts <= cell_max + eps)

    domain_half_extent = 0.5 * (bounds[1] - bounds[0])
    root_extent = geometry.max_extent[0]
    assert jnp.allclose(
        root_extent,
        jnp.max(domain_half_extent),
        rtol=1e-5,
        atol=1e-5,
    )
