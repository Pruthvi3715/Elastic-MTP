"""
Backward compatibility layer for vllm_elastic_plugin.py.
"""
from elastic_mtp.engine.vllm_elastic_plugin import (
    FusedCUDAEntropyRouter,
    ElasticvLLMBlockAllocator,
    ElasticvLLMServingEngine,
    vLLMBatchRequest
)

__all__ = [
    "FusedCUDAEntropyRouter",
    "ElasticvLLMBlockAllocator",
    "ElasticvLLMServingEngine",
    "vLLMBatchRequest"
]
