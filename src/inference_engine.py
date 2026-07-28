"""
Backward compatibility layer for inference_engine.py.
"""
from elastic_mtp.engine.inference_engine import ElasticMTPInferenceEngine, SyntheticLM

__all__ = ["ElasticMTPInferenceEngine", "SyntheticLM"]
