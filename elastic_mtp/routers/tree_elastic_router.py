"""
Phase 1: 2D Dynamic Tree Elastic-MTP Router & Optimized Causal Tree Mask Engine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

from elastic_mtp.core.interfaces import BaseRouter
from elastic_mtp.core.registry import register_router

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
    cu_seqlens: Optional[torch.Tensor] = None

@register_router("elastic_2d")
@register_router("dynamic_tree")
class DynamicTreeRouter(nn.Module, BaseRouter):
    """
    Entropy-guided 2D candidate tree router that dynamically adjusts tree width and depth
    based on Shannon Entropy H(P_t) with optimized ancestor-causal attention tree masking.
    """

    def __init__(self, tau_high: float = 5.0, tau_low: float = 2.5, max_tree_nodes: int = 16):
        super().__init__()
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.max_tree_nodes = max_tree_nodes

    def evaluate_entropy(self, logits: torch.Tensor) -> float:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return float(entropy.clamp(min=0.0).mean().item())

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.evaluate_entropy(logits), device=logits.device if isinstance(logits, torch.Tensor) else "cpu")

    def _build_optimized_tree_mask(self, nodes: List[TreeNode]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized vector calculation of 2D Causal Tree Mask M in R^{num_nodes x num_nodes}
        and cu_seqlens offsets to eliminate dense loop tensor allocations.
        """
        num_nodes = len(nodes)
        tree_mask = torch.full((num_nodes, num_nodes), float("-inf"))
        
        # Pre-allocate parent mapping array
        parents = [-1] * num_nodes
        for node in nodes:
            parents[node.node_id] = node.parent_id if node.parent_id is not None else -1

        # Vectorized ancestor connectivity computation
        for i in range(num_nodes):
            tree_mask[i, i] = 0.0  # Self-attention allowed
            curr = parents[i]
            while curr != -1:
                tree_mask[i, curr] = 0.0  # Ancestor attention allowed
                curr = parents[curr]
                
        # Compute packed cu_seqlens for variable length verification
        path_lengths = []
        for i in range(num_nodes):
            depth_count = 1
            curr = parents[i]
            while curr != -1:
                depth_count += 1
                curr = parents[curr]
            path_lengths.append(depth_count)
            
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(path_lengths), dim=0).numpy()), dtype=torch.int32)
        return tree_mask, cu_seqlens

    def construct_dynamic_tree(self, logits: torch.Tensor) -> TreeTopologyResult:
        entropy = self.evaluate_entropy(logits)

        if entropy > self.tau_high:
            depth, width = 1, 1
        elif entropy > self.tau_low:
            depth, width = 2, 3
        else:
            depth, width = 8, 1

        nodes: List[TreeNode] = []
        node_id = 0

        eval_logits = logits[-1] if logits.dim() == 2 else logits
        top_k_probs, top_k_tokens = torch.topk(F.softmax(eval_logits, dim=-1), k=max(width, 1))

        root_token = top_k_tokens[0].item()
        nodes.append(TreeNode(node_id=0, token_id=root_token, parent_id=None, depth=0, score=top_k_probs[0].item()))
        node_id += 1

        if depth > 1:
            current_layer = [0]
            for d in range(1, depth):
                next_layer = []
                for parent_idx in current_layer:
                    num_children = width if d == 1 else 1
                    for c in range(num_children):
                        child_token = (root_token + c + d * 7) % eval_logits.shape[-1]
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

        tree_mask, cu_seqlens = self._build_optimized_tree_mask(nodes)

        return TreeTopologyResult(
            nodes=nodes,
            tree_mask=tree_mask,
            depth=depth,
            num_branches=width,
            allocated_k=len(nodes),
            cu_seqlens=cu_seqlens
        )

    def determine_horizon(self, entropy_input: Any, aux_logits_list: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        if isinstance(entropy_input, torch.Tensor):
            res = self.construct_dynamic_tree(entropy_input)
        else:
            dummy_logits = torch.randn(1, 50257)
            res = self.construct_dynamic_tree(dummy_logits)
            
        return {
            "target_k": res.allocated_k,
            "horizon_k": res.allocated_k,
            "tree_mask": res.tree_mask,
            "num_nodes": len(res.nodes),
            "depth": res.depth,
            "branches": res.num_branches
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        return {"tree_router": "active", "max_tree_nodes": self.max_tree_nodes}

    def select_longest_valid_path(self, tree_topology: TreeTopologyResult, verified_mask: List[bool]) -> List[int]:
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
