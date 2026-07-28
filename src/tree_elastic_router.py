"""
Backward compatibility layer for tree_elastic_router.py.
"""
from elastic_mtp.routers.tree_elastic_router import (
    DynamicTreeRouter,
    TreeNode,
    TreeTopologyResult
)

__all__ = ["DynamicTreeRouter", "TreeNode", "TreeTopologyResult"]
