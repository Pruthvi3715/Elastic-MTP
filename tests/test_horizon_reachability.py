"""
Unit tests for horizon reachability and allocation formulas in ElasticHorizonRouter.
"""
import pytest
import torch
from src.elastic_horizon_router import DynamicHorizonRouter
from src.config import ElasticMTPConfig

def test_k_max_reachable_at_zero_entropy():
    router = DynamicHorizonRouter(tau_entropy=1.50, max_k=8)
    
    # Test zero entropy input
    res = router.determine_horizon(0.0)
    assert res["target_k"] == 8, f"Expected K=8 at entropy=0.0, got K={res['target_k']}"

def test_full_k_spectrum_reachability():
    router = DynamicHorizonRouter(tau_entropy=1.50, max_k=8)
    observed_ks = set()
    
    # Sweep entropy from 0.0 to 1.50 in small steps
    for step in range(101):
        entropy = (step / 100.0) * 1.50
        res = router.determine_horizon(entropy)
        observed_ks.add(res["target_k"])
        
    # Check that K=8 and all intermediate K values [1..8] are reachable
    assert 8 in observed_ks, "K=8 was never reached during entropy sweep"
    assert 1 in observed_ks, "K=1 was never reached during high entropy"
    assert len(observed_ks) == 8, f"Expected all 8 horizon levels to be reachable, got {sorted(list(observed_ks))}"

def test_high_entropy_fallback_to_k1():
    router = DynamicHorizonRouter(tau_entropy=1.50, max_k=8)
    res = router.determine_horizon(2.50)
    assert res["target_k"] == 1
    assert "HIGH_ENTROPY" in res["reason"]
