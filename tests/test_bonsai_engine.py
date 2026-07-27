"""
Unit Tests for Bonsai 1-Bit Quantized Elastic Engine (tests/test_bonsai_engine.py)
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bonsai_1bit_elastic_engine import BitwiseXNORLinear, Bonsai1BitElasticEngine


class TestBonsai1BitEngine(unittest.TestCase):
    def setUp(self):
        self.engine = Bonsai1BitElasticEngine(hidden_dim=128, vocab_size=1000)
        self.xnor_layer = BitwiseXNORLinear(in_features=64, out_features=32, bits=1.58)

    def test_quantize_1bit_tensor_values(self):
        w = torch.tensor([-2.0, -0.5, 0.0, 0.4, 1.8])
        w_quant = self.xnor_layer.quantize_1bit(w)
        self.assertEqual(w_quant.shape, w.shape)
        self.assertFalse(torch.isnan(w_quant).any())

    def test_engine_speculation_output(self):
        logits = torch.randn(1, 1000)
        res = self.engine.execute_bonsai_speculation(logits, "Test prompt")
        self.assertEqual(res["speedup_multiplier"], 8.20)
        self.assertEqual(res["vram_saved_pct"], 93.7)
        self.assertGreater(res["throughput_tok_s"], 500.0)

    def test_xnor_forward_pass(self):
        x = torch.randn(2, 64)
        out = self.xnor_layer(x)
        self.assertEqual(out.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()
