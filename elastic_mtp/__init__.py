"""
Elastic-MTP: Uncertainty-Aware Dynamic Horizon Multi-Token Prediction Engine.
"""

from elastic_mtp.config import ElasticMTPConfig
from elastic_mtp.core import BaseRouter, BaseAdapter, BaseCompressor, build_router, build_compressor
from elastic_mtp.routers import DynamicHorizonRouter, DynamicTreeRouter, FusedEntropyRouter, QuantizationAwareCalibrator
from elastic_mtp.adapters import MTPGLoRAModule
from elastic_mtp.compressors import TurboQuantKVCompressor
from elastic_mtp.engine import ElasticMTPInferenceEngine, ElasticvLLMServingEngine, SpeculativeKVCache
from elastic_mtp.daemon import AutoResearchManager

__version__ = "2.0.0"

__all__ = [
    "ElasticMTPConfig",
    "BaseRouter",
    "BaseAdapter",
    "BaseCompressor",
    "build_router",
    "build_compressor",
    "DynamicHorizonRouter",
    "DynamicTreeRouter",
    "FusedEntropyRouter",
    "QuantizationAwareCalibrator",
    "MTPGLoRAModule",
    "TurboQuantKVCompressor",
    "ElasticMTPInferenceEngine",
    "ElasticvLLMServingEngine",
    "SpeculativeKVCache",
    "AutoResearchManager"
]
