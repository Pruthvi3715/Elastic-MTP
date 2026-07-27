"""
KV-Cache Management Module for Speculative Multi-Token Prediction (Google Speculative Decoding Pattern).

Manages PyTorch Dynamic Key-Value Cache state across prefill and speculative decode passes.
Eliminates O(N^2) attention re-computation overhead during token generation.
"""
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any

class SpeculativeKVCache(nn.Module):
    """
    Dynamic Key-Value Cache tensor container.
    Stores past key and value tensors for each transformer layer:
    key_cache: (batch_size, num_heads, seq_len, head_dim)
    value_cache: (batch_size, num_heads, seq_len, head_dim)
    """
    def __init__(self, 
                 num_layers: int = 12, 
                 num_heads: int = 4, 
                 head_dim: int = 32, 
                 max_seq_len: int = 2048,
                 device: str = "cpu"):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.reset_cache()

    def reset_cache(self):
        """Resets cache sequence length pointer and clears cached tensors."""
        self.seq_len = 0
        self.key_caches: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self.value_caches: List[Optional[torch.Tensor]] = [None] * self.num_layers

    def update_layer_cache(self, 
                           layer_idx: int, 
                           new_key: torch.Tensor, 
                           new_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new_key and new_value to layer_idx cache.
        Returns full concatenated key and value tensors up to current seq_len.
        """
        if self.key_caches[layer_idx] is None:
            self.key_caches[layer_idx] = new_key
            self.value_caches[layer_idx] = new_value
        else:
            self.key_caches[layer_idx] = torch.cat([self.key_caches[layer_idx], new_key], dim=-2)
            self.value_caches[layer_idx] = torch.cat([self.value_caches[layer_idx], new_value], dim=-2)
            
        return self.key_caches[layer_idx], self.value_caches[layer_idx]

    def rollback_cache(self, num_tokens_to_discard: int):
        """
        Rolls back cache sequence pointer if a draft sequence is rejected.
        Truncates cached keys/values to length = current_len - num_tokens_to_discard.
        """
        if num_tokens_to_discard <= 0:
            return
            
        for i in range(self.num_layers):
            if self.key_caches[i] is not None:
                curr_len = self.key_caches[i].shape[-2]
                new_len = max(0, curr_len - num_tokens_to_discard)
                self.key_caches[i] = self.key_caches[i][..., :new_len, :]
                self.value_caches[i] = self.value_caches[i][..., :new_len, :]

    def get_memory_bytes(self) -> int:
        """Returns total memory occupied by KV cache in bytes."""
        total_bytes = 0
        for i in range(self.num_layers):
            if self.key_caches[i] is not None:
                total_bytes += self.key_caches[i].element_size() * self.key_caches[i].nelement()
                total_bytes += self.value_caches[i].element_size() * self.value_caches[i].nelement()
        return total_bytes
