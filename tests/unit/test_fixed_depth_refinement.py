import jax.numpy as jnp
import numpy as np

from yggdrax.geometry import _MAX_MORTON_LEVEL, compute_tree_geometry
from yggdrax.morton import _compact3_u64
from yggdrax.tree import build_fixed_depth_tree


def make_thin_slab(N=2000, length=1.0, thickness=1e-3):
    """Return a highly anisotropic point cloud used for refinement tests."""

    rng = np.random.default_rng(0)
    xs = np.linspace(0.0, length, N)
    ys = rng.normal(scale=thickness, size=N)
    zs = rng.normal(scale=thickness, size=N)
    pos = np.stack([xs, ys, zs], axis=1)
    masses = np.ones((N,), dtype=np.float64)
    return jnp.asarray(pos), jnp.asarray(masses)


def test_local_refinement_can_increase_leaf_depth():
    """Local refinement should split elongated Morton buckets when enabled."""

    pos, masses = make_thin_slab(N=512, length=1.0, thickness=1e-3)
    bounds = (jnp.min(pos, axis=0), jnp.max(pos, axis=0))

    tree_no_refine, *_ = build_fixed_depth_tree(
        pos,
        masses,
        bounds,
        return_reordered=True,
        target_leaf_particles=8,
        refine_local=False,
    )
    tree_refined, *_ = build_fixed_depth_tree(
        pos,
        masses,
        bounds,
        return_reordered=True,
        target_leaf_particles=8,
        refine_local=True,
        max_refine_levels=3,
        aspect_threshold=4.0,
    )

    assert int(np.max(np.asarray(tree_refined.leaf_depths))) >= int(
        np.max(np.asarray(tree_no_refine.leaf_depths))
    )


def test_fixed_depth_geometry_matches_morton_cells():
    rng = np.random.default_rng(42)
    pos = rng.uniform(low=0.2, high=0.3, size=(32, 3))
    masses = np.ones((32,), dtype=np.float64)
    bounds = (jnp.zeros(3), jnp.ones(3))

    tree, pos_sorted, *_ = build_fixed_depth_tree(
        jnp.asarray(pos),
        jnp.asarray(masses),
        bounds,
        target_leaf_particles=1,
        return_reordered=True,
        refine_local=False,
    )

    geometry = compute_tree_geometry(tree, pos_sorted)

    leaf_depths = np.asarray(tree.leaf_depths).astype(np.int32)
    assert np.all(leaf_depths >= 0)

    num_internal = int(tree.num_internal_nodes)
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

    leaf_codes = np.asarray(tree.leaf_codes, dtype=np.uint64)
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
