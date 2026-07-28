"""
Backward compatibility layer for elastic_horizon_router.py.
"""
from elastic_mtp.routers.elastic_horizon_router import (
    DynamicHorizonRouter,
    ElasticHorizonRouter,
    UncertaintyAwareHorizonFilter,
    RouterResult
)

__all__ = [
    "DynamicHorizonRouter",
    "ElasticHorizonRouter",
    "UncertaintyAwareHorizonFilter",
    "RouterResult"
]
