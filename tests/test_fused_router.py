"""
Unit tests for Fused Entropy Router and Quantization Noise Stress-Test Harness.
"""
import unittest
import torch
from src.entropy_evaluator import EntropyEvaluator
from src.fused_entropy_router import FusedEntropyRouter
from quantization_bench import simulate_quantization_noise

class TestFusedRouterAndQuantization(unittest.TestCase):

    def test_fused_entropy_equivalence(self):
        fused_router = FusedEntropyRouter()
        logits = torch.randn(2, 500)
        
        baseline_entropy = EntropyEvaluator.compute_shannon_entropy(logits)
        fused_entropy = fused_router.fused_shannon_entropy(logits)
        
        # Must match to within 1e-4 precision
        torch.testing.assert_close(fused_entropy, baseline_entropy, rtol=1e-4, atol=1e-4)

    def test_quantization_noise_simulation(self):
        logits = torch.tensor([[10.0, -5.0, 2.5, 0.0]])
        fp16_out = simulate_quantization_noise(logits, "fp16")
        int8_out = simulate_quantization_noise(logits, "int8")
        int4_out = simulate_quantization_noise(logits, "int4")
        
        self.assertEqual(fp16_out.shape, logits.shape)
        self.assertEqual(int8_out.shape, logits.shape)
        self.assertEqual(int4_out.shape, logits.shape)
        self.assertFalse(torch.isnan(int4_out).any())

if __name__ == "__main__":
    unittest.main()
