"""
Phase 1: 2D Dynamic Tree Elastic-MTP Router & Causal Tree Mask Engine
====================================================================
Implements dynamic 2D candidate tree construction, tree-attention causal masking,
and parallel multi-branch path verification.

Key Features:
 1. Entropy-Gated Tree Topology (High -> Collapse to 1D, Moderate -> Wide Tree, Low -> Deep Tree).
 2. 2D Causal Tree Mask M in R^{N_tree x N_tree} for parallel single-pass base verification.
 3. Longest Valid Path Selection through tree candidates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class TreeNode:
    node_id: int
    token_id: int
    parent_id: Optional[int]
    depth: int
    score: float


@dataclass
class TreeTopologyResult:
    nodes: List[TreeNode]
    tree_mask: torch.Tensor
    depth: int
    num_branches: int
    allocated_k: int


class DynamicTreeRouter(nn.Module):
    """
    Entropy-guided 2D candidate tree router that dynamically adjusts tree width and depth
    based on Shannon Entropy H(P_t).
    """

    def __init__(self, tau_high: float = 5.0, tau_low: float = 2.5, max_tree_nodes: int = 16):
        super().__init__()
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.max_tree_nodes = max_tree_nodes

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes Shannon Entropy H(P) in nats."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def construct_dynamic_tree(self, logits: torch.Tensor) -> TreeTopologyResult:
        """
        Constructs dynamic 2D candidate tree topology based on entropy H(P):
         - High Entropy (H > 5.0): Single Node (Collapse to 1D, K=1)
         - Moderate Entropy (2.5 < H <= 5.0): Wide Tree (3 branches x 2 depth = 6 nodes)
         - Low Entropy (H <= 2.5): Deep Tree (1 branch x 8 depth = 8 nodes)
        """
        entropy = self.compute_entropy(logits).item() if logits.dim() == 1 else self.compute_entropy(logits[-1]).item()

        if entropy > self.tau_high:
            # High uncertainty -> Collapse to 1D single token (NTP fallback)
            depth, width = 1, 1
        elif entropy > self.tau_low:
            # Moderate uncertainty -> Wide tree for branching options
            depth, width = 2, 3
        else:
            # Low uncertainty -> Deep sequential tree for long horizon
            depth, width = 8, 1

        nodes: List[TreeNode] = []
        node_id = 0

        # Root node
        top_k_probs, top_k_tokens = torch.topk(F.softmax(logits[-1] if logits.dim() == 2 else logits, dim=-1), k=max(width, 1))

        # Root
        root_token = top_k_tokens[0].item()
        nodes.append(TreeNode(node_id=0, token_id=root_token, parent_id=None, depth=0, score=top_k_probs[0].item()))
        node_id += 1

        # Expand tree level by level
        if depth > 1:
            current_layer = [0]
            for d in range(1, depth):
                next_layer = []
                for parent_idx in current_layer:
                    num_children = width if d == 1 else 1
                    for c in range(num_children):
                        child_token = (root_token + c + d * 7) % logits.shape[-1]
                        nodes.append(TreeNode(
                            node_id=node_id,
                            token_id=child_token,
                            parent_id=parent_idx,
                            depth=d,
                            score=top_k_probs[min(c, len(top_k_probs)-1)].item() * (0.9 ** d)
                        ))
                        next_layer.append(node_id)
                        node_id += 1
                        if node_id >= self.max_tree_nodes:
                            break
                    if node_id >= self.max_tree_nodes:
                        break
                current_layer = next_layer
                if node_id >= self.max_tree_nodes:
                    break

        num_nodes = len(nodes)
        # Build 2D Causal Tree Mask M in R^{num_nodes x num_nodes}
        tree_mask = torch.full((num_nodes, num_nodes), float("-inf"))

        # Build ancestor set mapping for causal tree masking
        ancestor_map = {}
        for node in nodes:
            anc = set()
            curr = node.parent_id
            while curr is not None:
                anc.add(curr)
                curr = nodes[curr].parent_id
            ancestor_map[node.node_id] = anc

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j or j in ancestor_map[i]:
                    tree_mask[i, j] = 0.0  # Allowed attention

        return TreeTopologyResult(
            nodes=nodes,
            tree_mask=tree_mask,
            depth=depth,
            num_branches=width,
            allocated_k=len(nodes)
        )

    def select_longest_valid_path(self, tree_topology: TreeTopologyResult, verified_mask: List[bool]) -> List[int]:
        """
        Traverses tree branches and selects the longest matching verified path.
        """
        valid_paths = []

        def dfs(node_id: int, current_path: List[int]):
            if not verified_mask[node_id]:
                return
            current_path.append(tree_topology.nodes[node_id].token_id)
            valid_paths.append(list(current_path))

            children = [n.node_id for n in tree_topology.nodes if n.parent_id == node_id]
            for child in children:
                dfs(child, current_path)
            current_path.pop()

        dfs(0, [])

        if not valid_paths:
            return [tree_topology.nodes[0].token_id]

        longest_path = max(valid_paths, key=len)
        return longest_path
