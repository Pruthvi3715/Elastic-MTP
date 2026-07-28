"""
Speculative KV Cache Manager with Memory Rollback & Quantization.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

class SpeculativeKVCache:
    def __init__(self, num_layers: int = 12, num_heads: int = 4, head_dim: int = 32, device: str = "cpu"):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        self.key_caches: List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_caches: List[Optional[torch.Tensor]] = [None] * num_layers

    def reset_cache(self):
        self.key_caches = [None] * self.num_layers
        self.value_caches = [None] * self.num_layers

    def update_layer_cache(self, layer_idx: int, new_keys: torch.Tensor, new_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key_caches[layer_idx] is None:
            self.key_caches[layer_idx] = new_keys
            self.value_caches[layer_idx] = new_values
        else:
            self.key_caches[layer_idx] = torch.cat([self.key_caches[layer_idx], new_keys], dim=-2)
            self.value_caches[layer_idx] = torch.cat([self.value_caches[layer_idx], new_values], dim=-2)
        return self.key_caches[layer_idx], self.value_caches[layer_idx]

    def rollback_cache(self, num_rejected_tokens: int):
        if num_rejected_tokens <= 0:
            return
        for idx in range(self.num_layers):
            if self.key_caches[idx] is not None:
                seq_len = self.key_caches[idx].shape[-2]
                keep_len = max(0, seq_len - num_rejected_tokens)
                if keep_len == 0:
                    self.key_caches[idx] = None
                    self.value_caches[idx] = None
                else:
                    self.key_caches[idx] = self.key_caches[idx][..., :keep_len, :]
                    self.value_caches[idx] = self.value_caches[idx][..., :keep_len, :]

    def get_memory_bytes(self) -> int:
        total = 0
        for k in self.key_caches:
            if k is not None:
                total += k.element_size() * k.nelement()
        for v in self.value_caches:
            if v is not None:
                total += v.element_size() * v.nelement()
        return total
