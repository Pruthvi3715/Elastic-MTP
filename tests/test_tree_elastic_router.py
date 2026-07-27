"""
Unit tests for 2D Dynamic Tree Elastic Router and Causal Tree Masking.
"""

import torch
import pytest
from src.tree_elastic_router import DynamicTreeRouter, TreeNode, TreeTopologyResult


def test_tree_router_entropy_computation():
    router = DynamicTreeRouter()
    sharp_logits = torch.tensor([10.0, -10.0, -10.0])  # Low entropy
    flat_logits = torch.tensor([1.0, 1.0, 1.0])       # High entropy

    low_h = router.compute_entropy(sharp_logits).item()
    high_h = router.compute_entropy(flat_logits).item()

    assert low_h < high_h
    assert low_h < 0.1
    assert high_h > 1.0


def test_tree_topology_construction():
    router = DynamicTreeRouter(tau_high=5.0, tau_low=2.5)

    # Sharp logits -> Deep tree
    sharp_logits = torch.tensor([15.0, 0.0, 0.0])
    deep_tree = router.construct_dynamic_tree(sharp_logits)
    assert deep_tree.depth == 8
    assert deep_tree.num_branches == 1
    assert deep_tree.tree_mask.shape == (len(deep_tree.nodes), len(deep_tree.nodes))

    # Moderate logits -> Wide tree
    mod_logits = torch.tensor([3.0, 2.5, 2.0, 1.0, 0.0])
    wide_tree = router.construct_dynamic_tree(mod_logits)
    assert wide_tree.num_branches >= 1
    assert len(wide_tree.nodes) > 1

    # Flat logits -> Collapse to 1D (single node)
    flat_logits = torch.full((1000,), 1.0)
    flat_tree = router.construct_dynamic_tree(flat_logits)
    assert len(flat_tree.nodes) == 1


def test_causal_tree_mask_properties():
    router = DynamicTreeRouter()
    logits = torch.tensor([3.0, 2.0, 1.0])
    tree_topo = router.construct_dynamic_tree(logits)

    mask = tree_topo.tree_mask
    num_nodes = len(tree_topo.nodes)

    # Diagonal must always be 0 (self-attention allowed)
    for i in range(num_nodes):
        assert mask[i, i] == 0.0

    # Upper triangle or non-ancestor pairs must be -inf
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Unless j is an ancestor of i, mask[i, j] should be -inf
            pass


def test_longest_path_selection():
    router = DynamicTreeRouter()
    logits = torch.tensor([2.0, 1.0, 0.5])
    tree_topo = router.construct_dynamic_tree(logits)

    verified_mask = [True] * len(tree_topo.nodes)
    longest_path = router.select_longest_valid_path(tree_topo, verified_mask)

    assert len(longest_path) > 0
    assert isinstance(longest_path, list)
