"""
Unit tests for KV-Cache Manager (Google Speculative Decoding Pattern).
"""
import unittest
import torch
from src.kv_cache_manager import SpeculativeKVCache

class TestKVCacheManager(unittest.TestCase):

    def test_kv_cache_initialization(self):
        cache = SpeculativeKVCache(num_layers=4, num_heads=2, head_dim=16)
        self.assertEqual(cache.num_layers, 4)
        self.assertEqual(cache.get_memory_bytes(), 0)

    def test_update_layer_cache(self):
        cache = SpeculativeKVCache(num_layers=2, num_heads=2, head_dim=16)
        k1 = torch.randn(1, 2, 5, 16)
        v1 = torch.randn(1, 2, 5, 16)
        
        full_k, full_v = cache.update_layer_cache(0, k1, v1)
        self.assertEqual(full_k.shape, (1, 2, 5, 16))
        
        # Append 1 new token
        k2 = torch.randn(1, 2, 1, 16)
        v2 = torch.randn(1, 2, 1, 16)
        full_k2, full_v2 = cache.update_layer_cache(0, k2, v2)
        self.assertEqual(full_k2.shape, (1, 2, 6, 16))

    def test_rollback_cache(self):
        cache = SpeculativeKVCache(num_layers=2, num_heads=2, head_dim=16)
        k = torch.randn(1, 2, 10, 16)
        v = torch.randn(1, 2, 10, 16)
        cache.update_layer_cache(0, k, v)
        
        # Roll back 3 rejected draft tokens
        cache.rollback_cache(3)
        self.assertEqual(cache.key_caches[0].shape[-2], 7)

if __name__ == "__main__":
    unittest.main()
