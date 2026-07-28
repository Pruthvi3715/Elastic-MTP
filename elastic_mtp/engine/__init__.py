"""
Elastic-MTP Inference & vLLM Serving Engines.
"""
from elastic_mtp.engine.inference_engine import ElasticMTPInferenceEngine, SyntheticLM
from elastic_mtp.engine.vllm_elastic_plugin import (
    ElasticvLLMServingEngine,
    FusedCUDAEntropyRouter,
    ElasticvLLMBlockAllocator,
    vLLMBatchRequest
)
from elastic_mtp.engine.kv_cache_manager import SpeculativeKVCache

__all__ = [
    "ElasticMTPInferenceEngine",
    "SyntheticLM",
    "ElasticvLLMServingEngine",
    "FusedCUDAEntropyRouter",
    "ElasticvLLMBlockAllocator",
    "vLLMBatchRequest",
    "SpeculativeKVCache"
]
