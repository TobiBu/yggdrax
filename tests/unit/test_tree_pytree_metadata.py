"""``leaf_size`` is static, and must survive every builder as a Python ``int``.

The tree pytree registration files ``topology.leaf_size`` as *aux* data rather
than among the children, so that ``int(topology.leaf_size)`` keeps working on a
tree that has been through ``jax.jit``. Aux data is the treedef, and the treedef
is part of every ``jax.jit`` cache key, so nothing there may be a JAX array.

Four builder paths broke that, each by handing its work to a ``jax.jit`` that
returns a *build result* (a topology, or a tuple starting with one) rather than a
``Tree``: the topology is a ``NamedTuple``, so ``leaf_size`` is an ordinary
pytree child of that output and JAX converts it to a device array on the way out.
A static ``leaf_size=8`` came back as ``Array(8, dtype=int64, weak_type=True)``.

What it costs is a jit cache lookup taken inside a trace: two trees built the
same way hold two *distinct* ``Array(8)`` objects, so the treedef comparison
falls past the identity fast path into ``Array.__eq__``, which -- with an outer
trace live -- stages a ``bool[]`` tracer rather than returning a bool. JAX
surfaces that as ``ValueError: Exception raised while checking equality of
metadata fields of pytree``. It cost jaccpot a red CI shard (TobiBu/jaccpot#330),
and it read as flaky there because one tree reused holds the same array object on
both sides and the identity check hides it.

``leaf_size=None`` needs nothing -- ``None`` is pytree structure, not a leaf --
which is why the fixed-depth paths were never affected. They are parametrised in
anyway: the assertion is "static", and ``None`` is static.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import yggdrax.tree as tree_api
from yggdrax.tree import KDParticleTree, OctreeTree, RadixTree


def _sample_problem(n: int = 64):
    key_pos, key_mass = jax.random.split(jax.random.PRNGKey(17))
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    return positions, masses


def _topology_of(built):
    """Return the topology from whatever a builder handed back."""

    obj = built[0] if isinstance(built, tuple) else built
    return getattr(obj, "topology", obj)


# Every entry point that can produce a tree with a non-``None`` leaf size, plus
# the fixed-depth ones for completeness. Keyed by name so a failure names the
# builder rather than an index.
_BUILDERS = {
    "build_tree": lambda p, m: tree_api.build_tree(
        p, m, leaf_size=8, return_reordered=True
    ),
    "build_tree_jit": lambda p, m: tree_api.build_tree_jit(
        p, m, leaf_size=8, return_reordered=True
    ),
    "build_octree": lambda p, m: tree_api.build_octree(
        p, m, leaf_size=8, return_reordered=True
    ),
    "build_octree_jit": lambda p, m: tree_api.build_octree_jit(
        p, m, leaf_size=8, return_reordered=True
    ),
    "build_static_radix_tree": lambda p, m: tree_api.build_static_radix_tree(
        p, m, leaf_size=8, return_reordered=True
    ),
    "build_fixed_depth_tree": lambda p, m: tree_api.build_fixed_depth_tree(
        p, m, return_reordered=True
    ),
    "build_fixed_depth_tree_jit": lambda p, m: tree_api.build_fixed_depth_tree_jit(
        p, m, return_reordered=True
    ),
    "build_fixed_depth_octree": lambda p, m: tree_api.build_fixed_depth_octree(
        p, m, return_reordered=True
    ),
    "build_fixed_depth_octree_jit": (
        lambda p, m: tree_api.build_fixed_depth_octree_jit(p, m, return_reordered=True)
    ),
    "RadixTree.adaptive": lambda p, m: RadixTree.from_particles(
        p, m, build_mode="adaptive", leaf_size=8
    ),
    "RadixTree.fixed_depth": lambda p, m: RadixTree.from_particles(
        p, m, build_mode="fixed_depth", leaf_size=8
    ),
    "RadixTree.static_radix": lambda p, m: RadixTree.from_particles(
        p, m, build_mode="static_radix", leaf_size=8
    ),
    "OctreeTree.adaptive": lambda p, m: OctreeTree.from_particles(
        p, m, build_mode="adaptive", leaf_size=8
    ),
    "OctreeTree.fixed_depth": lambda p, m: OctreeTree.from_particles(
        p, m, build_mode="fixed_depth", leaf_size=8
    ),
    "KDParticleTree.adaptive": lambda p, m: KDParticleTree.from_particles(
        p, m, leaf_size=8
    ),
}


@pytest.mark.parametrize("builder", sorted(_BUILDERS), ids=sorted(_BUILDERS))
def test_leaf_size_survives_every_builder_as_a_static_value(builder):
    positions, masses = _sample_problem()
    topology = _topology_of(_BUILDERS[builder](positions, masses))

    leaf_size = getattr(topology, "leaf_size", None)

    assert not isinstance(leaf_size, (jax.Array, np.ndarray)), (
        f"{builder} returned leaf_size={leaf_size!r}, a JAX array. It is filed as "
        "pytree aux data and so lands in every jit cache key; restore the static "
        "value on the way out of the jit that produced it."
    )
    assert leaf_size is None or isinstance(leaf_size, int)


def _metadata_holding_arrays(value):
    """Return ``(path, node type, metadata repr)`` per array found in metadata."""

    def _reaches_array(meta, depth=0):
        if isinstance(meta, (jax.Array, np.ndarray)):
            return True
        if depth > 4:
            return False
        if isinstance(meta, (tuple, list, set, frozenset)):
            return any(_reaches_array(item, depth + 1) for item in meta)
        if isinstance(meta, dict):
            return any(_reaches_array(item, depth + 1) for item in meta.values())
        return False

    def _walk(treedef, path):
        found = []
        node_data = treedef.node_data()
        if node_data is not None and _reaches_array(node_data[1]):
            found.append((path, node_data[0].__name__, repr(node_data[1])[:200]))
        for index, child in enumerate(treedef.children()):
            found.extend(_walk(child, f"{path}.{index}"))
        return found

    return _walk(jax.tree_util.tree_structure(value), "tree")


@pytest.mark.parametrize("tree_cls", [RadixTree, OctreeTree, KDParticleTree])
def test_tree_treedef_carries_no_arrays_in_its_metadata(tree_cls):
    """The general statement the leaf-size assertion above is one instance of.

    Walks the structure rather than the leaves, because that is exactly where
    such a value hides: it is invisible to ``jax.tree.leaves`` by construction.
    """

    positions, masses = _sample_problem()
    tree = tree_cls.from_particles(positions, masses, leaf_size=8)

    assert _metadata_holding_arrays(tree) == []


def test_two_independently_built_octrees_are_jittable():
    """The failure in its original form, and the reason for each part of the setup.

    Three things all have to be true, which is why it went unnoticed for so long.

    *Two trees, built separately.* One tree reused holds the same aux object in
    both treedefs, and the identity fast path in the comparison never reaches
    ``Array.__eq__``.

    *Nested jits.* The comparison has to happen while a trace is live, or two
    concrete ``Array(8)`` compare to a concrete ``True`` and nothing goes wrong.
    So the tree is passed to an *inner* jit from inside an *outer* one -- exactly
    the shape jaccpot hit, where a jitted evaluation kernel was called from a
    jitted solver entry point.

    *A fresh outer wrapper per tree.* Sharing one makes the second call an outer
    cache hit that never re-traces, so the inner lookup never happens.
    """

    positions, masses = _sample_problem()
    first = OctreeTree.from_particles(positions, masses, leaf_size=8)
    second = OctreeTree.from_particles(positions, masses, leaf_size=8)

    inner = jax.jit(lambda tree: jnp.sum(tree.positions_sorted))

    def total_through_nested_jit(tree):
        return jax.jit(lambda t: inner(t))(tree)

    assert np.allclose(
        np.asarray(total_through_nested_jit(first)),
        np.asarray(total_through_nested_jit(second)),
    )
