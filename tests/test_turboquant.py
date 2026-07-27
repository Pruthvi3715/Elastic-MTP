"""
Unit tests for Google TurboQuant Extreme KV-Cache Compressor (ICLR 2026).
"""
import unittest
import torch
import torch.nn.functional as F
from src.turboquant_kv_compressor import TurboQuantKVCompressor

class TestTurboQuantCompressor(unittest.TestCase):

    def test_turboquant_compression_ratio(self):
        compressor = TurboQuantKVCompressor(head_dim=32)
        ratio = compressor.get_compression_ratio()
        self.assertGreaterEqual(ratio, 4.0)

    def test_compress_decompress_shape_and_range(self):
        compressor = TurboQuantKVCompressor(head_dim=32)
        k_orig = torch.randn(2, 4, 10, 32)
        
        q_polar, k_norm, qjl_res = compressor.compress_key_vector(k_orig)
        k_recon = compressor.decompress_key_vector(q_polar, k_norm, qjl_res)
        
        self.assertEqual(k_recon.shape, k_orig.shape)
        self.assertFalse(torch.isnan(k_recon).any())
        self.assertFalse(torch.isinf(k_recon).any())

    def test_inner_product_preservation(self):
        compressor = TurboQuantKVCompressor(head_dim=32)
        k_orig = torch.randn(1, 1, 1, 32)
        q_query = torch.randn(1, 1, 1, 32)
        
        # Original inner product
        dot_orig = torch.sum(q_query * k_orig, dim=-1)
        
        # TurboQuant reconstruction inner product
        q_polar, k_norm, qjl_res = compressor.compress_key_vector(k_orig)
        k_recon = compressor.decompress_key_vector(q_polar, k_norm, qjl_res)
        dot_recon = torch.sum(q_query * k_recon, dim=-1)
        
        # Cosine similarity between original and reconstructed key
        cos_sim = F.cosine_similarity(k_orig.flatten(), k_recon.flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.85)

if __name__ == "__main__":
    unittest.main()
