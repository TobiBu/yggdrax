"""Tests for traversal policy contracts."""

from yggdrax.interactions import DualTreeTraversalConfig
from yggdrax.policies import TraversalPolicy


def test_traversal_policy_defaults_description():
    config = DualTreeTraversalConfig(
        max_pair_queue=1024,
        process_block=128,
        max_interactions_per_node=256,
        max_neighbors_per_leaf=128,
    )
    policy = TraversalPolicy(config=config)

    assert policy.config == config
    assert policy.description == "Explicit dual-tree traversal capacities."


def test_traversal_policy_accepts_custom_description():
    config = DualTreeTraversalConfig(
        max_pair_queue=2048,
        process_block=64,
        max_interactions_per_node=512,
        max_neighbors_per_leaf=256,
    )
    policy = TraversalPolicy(config=config, description="tuned-for-bench")

    assert policy.config.max_pair_queue == 2048
    assert policy.description == "tuned-for-bench"
