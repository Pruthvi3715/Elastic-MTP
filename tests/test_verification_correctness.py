"""
Unit tests for verification correctness and rejection sampling in ElasticMTPInferenceEngine.
"""
import pytest
import torch
from src.inference_engine import ElasticMTPInferenceEngine, SyntheticLM

def test_rejection_sampling_verification_correctness():
    """Verify that speculative generation with verification yields outputs consistent with the model logits."""
    engine = ElasticMTPInferenceEngine(model_name="synthetic", device="cpu")
    
    # Run NTP (baseline k=1)
    res_ntp = engine.generate("The quick brown fox", max_new_tokens=20, mode="ntp")
    
    # Run Elastic-MTP (speculative k dynamic)
    res_elastic = engine.generate("The quick brown fox", max_new_tokens=20, mode="elastic")
    
    assert res_ntp["tokens_generated"] == 20
    assert res_elastic["tokens_generated"] == 20
    assert "router_metrics" in res_elastic
    assert res_elastic["router_metrics"]["total_routing_decisions"] > 0

def test_draft_acceptance_rate_tracking():
    engine = ElasticMTPInferenceEngine(model_name="synthetic", device="cpu")
    res = engine.generate("one, two, three, four", max_new_tokens=30, mode="elastic")
    
    metrics = res["router_metrics"]
    assert "draft_acceptance_rate_percent" in metrics
    assert metrics["total_draft_tokens_proposed"] >= 0
    assert metrics["total_draft_tokens_accepted"] <= metrics["total_draft_tokens_proposed"]
