"""
Unit tests for Uncertainty-Aware Horizon Filter and Entropy Evaluator.
"""
import unittest
import torch
import torch.nn.functional as F
from src.entropy_evaluator import EntropyEvaluator
from src.elastic_horizon_router import UncertaintyAwareHorizonFilter, ElasticHorizonRouter

class TestElasticMTPCore(unittest.TestCase):

    def test_shannon_entropy_deterministic(self):
        """Zero uncertainty (100% on single token) -> Entropy must be 0.0"""
        logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
        entropy = EntropyEvaluator.compute_shannon_entropy(logits)
        self.assertAlmostEqual(entropy.item(), 0.0, places=4)

    def test_shannon_entropy_uniform(self):
        """Max uncertainty (equal prob over 4 tokens) -> Entropy must be ln(4) ≈ 1.3863"""
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        entropy = EntropyEvaluator.compute_shannon_entropy(logits)
        expected = torch.log(torch.tensor(4.0)).item()
        self.assertAlmostEqual(entropy.item(), expected, places=4)

    def test_numerical_stability(self):
        """Extreme negative values must not cause NaN or Inf"""
        logits = torch.tensor([[-1e9, -1e9, -1e9, 100.0]])
        entropy = EntropyEvaluator.compute_shannon_entropy(logits)
        self.assertFalse(torch.isnan(entropy).any())
        self.assertFalse(torch.isinf(entropy).any())

    def test_horizon_routing(self):
        router = UncertaintyAwareHorizonFilter(tau_entropy=1.85, max_k=4)
        
        # Low entropy -> k > 1 (with matching aux logits)
        p_low = torch.tensor([[10.0, -10.0, -10.0]])
        k_low, meta_low = router.determine_horizon(p_low, [p_low, p_low, p_low])
        self.assertGreater(k_low, 1)
        
        # High entropy (>1.85) -> k = 1
        k_high, meta_high = router.determine_horizon(torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))
        self.assertEqual(k_high, 1)
        self.assertEqual(meta_high["reason"], "HIGH_ENTROPY_NTP_FALLBACK")

    def test_contradiction_detection(self):
        router = UncertaintyAwareHorizonFilter(tau_divergence=0.45)
        
        # Primary logits pick token 0, Aux logits pick token 1 (High divergence)
        p_logits = torch.tensor([[10.0, -10.0, -10.0]])
        q_logits = torch.tensor([[-10.0, 10.0, -10.0]])
        
        k_val, meta = router.determine_horizon(p_logits, [q_logits])
        self.assertTrue(meta["is_contradiction"])
        self.assertEqual(k_val, 1)

if __name__ == "__main__":
    unittest.main()
