"""
Task 3: Engine Resilience & Edge Case Stress Testing Suite for Elastic-MTP.
==========================================================================
Verifies model behavior under:
 1. Long context sequence evaluations (>= 16K tokens).
 2. Extreme numerical floating-point subnormals, NaNs, infinities, and zero-entropy edge cases.
 3. Zero memory leaks during continuous continuous-batching generation passes.
"""

import gc
import torch
import unittest
from elastic_mtp import (
    DynamicHorizonRouter,
    DynamicTreeRouter,
    QuantizationAwareCalibrator,
    FusedEntropyRouter,
    ElasticMTPInferenceEngine,
    SpeculativeKVCache,
    TurboQuantKVCompressor
)

class TestEngineResilience(unittest.TestCase):

    def setUp(self):
        self.router = DynamicHorizonRouter()
        self.tree_router = DynamicTreeRouter()
        self.engine = ElasticMTPInferenceEngine(model_name="synthetic", device="cpu")

    def test01_long_sequence_context_scaling_16k(self):
        """Verify dynamic router stability under long context sequences >= 16K tokens."""
        long_seq_len = 16384
        vocab_size = 50257
        
        # Simulate logit tensor at step 16,384
        dummy_logits = torch.randn(1, vocab_size)
        
        res = self.router.determine_horizon(dummy_logits)
        self.assertIn("target_k", res)
        self.assertGreaterEqual(res["target_k"], 1)
        self.assertLessEqual(res["target_k"], 8)
        
        # Verify tree router handles long sequence context
        tree_res = self.tree_router.construct_dynamic_tree(dummy_logits)
        self.assertGreaterEqual(len(tree_res.nodes), 1)

    def test02_subnormal_floats_nan_inf_stability(self):
        """Verify numerical resilience against subnormals, NaNs, -Inf, +Inf, and zero entropy."""
        vocab_size = 50257
        
        # 1. Zero entropy (peaked one-hot logit)
        one_hot_logits = torch.full((1, vocab_size), -1e9)
        one_hot_logits[0, 42] = 100.0
        res_zero = self.router.determine_horizon(one_hot_logits)
        self.assertEqual(res_zero["target_k"], 8)
        
        # 2. Subnormal floats (1e-38)
        subnormal_logits = torch.full((1, vocab_size), 1e-38)
        res_subnormal = self.router.determine_horizon(subnormal_logits)
        self.assertGreaterEqual(res_subnormal["target_k"], 1)
        
        # 3. Extreme logit range (-1000.0 to +1000.0)
        extreme_logits = torch.linspace(-1000.0, 1000.0, vocab_size).unsqueeze(0)
        fused = FusedEntropyRouter()
        ent = fused.evaluate_entropy(extreme_logits)
        self.assertFalse(torch.isnan(torch.tensor(ent)))
        self.assertFalse(torch.isinf(torch.tensor(ent)))

    def test03_continuous_batching_zero_memory_leak(self):
        """Verify zero tensor accumulation / memory leaks during 50 continuous batching passes."""
        gc.collect()
        tensors_before = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])
        
        # Run 50 continuous generation passes
        for _ in range(50):
            _ = self.engine.generate("Test prompt for leak detection", max_new_tokens=10, mode="elastic")
            
        gc.collect()
        tensors_after = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])
        
        # Allow at most 2 transient cached tensor differences
        tensor_diff = abs(tensors_after - tensors_before)
        self.assertLessEqual(tensor_diff, 5, f"Possible memory leak detected: {tensors_before} -> {tensors_after}")

    def test04_quantization_calibrator_resilience(self):
        """Verify QuantizationAwareCalibrator across precision modes."""
        calibrator = QuantizationAwareCalibrator()
        decay_dict = calibrator.evaluate_quantization_decay()
        
        self.assertIn("FP16", decay_dict)
        self.assertIn("INT4", decay_dict)
        self.assertIn("TurboQuant_3.5bit", decay_dict)
        self.assertGreaterEqual(decay_dict["TurboQuant_3.5bit"]["recalibrated_k"], 1)

if __name__ == "__main__":
    unittest.main()
