"""
Google TurboQuant: Extreme KV-Cache Vector Compression Module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional

from elastic_mtp.core.interfaces import BaseCompressor
from elastic_mtp.core.registry import register_compressor

@register_compressor("turboquant")
@register_compressor("turboquant_3.5bit")
class TurboQuantKVCompressor(nn.Module, BaseCompressor):
    def __init__(self, 
                 head_dim: int = 32, 
                 target_bits: float = 3.5, 
                 device: str = "cpu"):
        super().__init__()
        self.head_dim = head_dim
        self.target_bits = target_bits
        self.device = device
        
        with torch.no_grad():
            random_matrix = torch.randn(head_dim, head_dim, device=device)
            q, _ = torch.linalg.qr(random_matrix)
            self.register_buffer("rotation_matrix", q)
            
        self.qjl_dim = max(16, head_dim // 2)
        with torch.no_grad():
            qjl_proj = torch.randint(0, 2, (self.qjl_dim, head_dim), device=device).float() * 2.0 - 1.0
            self.register_buffer("qjl_matrix", qjl_proj / np.sqrt(self.qjl_dim))

    @torch.no_grad()
    def compress_key_vector(self, k_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = k_tensor.shape
        flat_k = k_tensor.reshape(-1, self.head_dim)
        
        rotated_k = F.linear(flat_k, self.rotation_matrix)
        k_norm = torch.norm(rotated_k, dim=-1, keepdim=True) + 1e-8
        unit_k = rotated_k / k_norm
        
        scale = 7.0 / 2.0
        q_polar = torch.round((unit_k + 1.0) * scale)
        q_polar = torch.clamp(q_polar, 0, 7)
        deq_polar = (q_polar / scale) - 1.0
        
        residual = unit_k - deq_polar
        qjl_sign = torch.sign(F.linear(residual, self.qjl_matrix))
        
        q_polar = q_polar.reshape(*shape[:-1], self.head_dim)
        k_norm = k_norm.reshape(*shape[:-1], 1)
        qjl_residuals = qjl_sign.reshape(*shape[:-1], self.qjl_dim)
        
        k_norm = torch.clamp(k_norm, max=65504.0)
        return q_polar.byte(), k_norm.half(), qjl_residuals.to(torch.int8)

    @torch.no_grad()
    def decompress_key_vector(self, 
                               q_polar: torch.Tensor, 
                               k_norm: torch.Tensor, 
                               qjl_residuals: torch.Tensor) -> torch.Tensor:
        shape = q_polar.shape
        flat_q = q_polar.reshape(-1, self.head_dim).float()
        flat_norm = k_norm.reshape(-1, 1).float()
        
        scale = 7.0 / 2.0
        deq_unit = (flat_q / scale) - 1.0
        
        decompressed_rotated = deq_unit * flat_norm
        decompressed_k = F.linear(decompressed_rotated, self.rotation_matrix.T)
        
        return decompressed_k.reshape(*shape[:-1], self.head_dim)

    @torch.no_grad()
    def compress_kv(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        q_polar, k_norm, qjl_residuals = self.compress_key_vector(key_cache)
        v_shape = value_cache.shape
        v_norm = torch.norm(value_cache, dim=-1, keepdim=True) + 1e-8
        v_unit = value_cache / v_norm
        q_v = torch.clamp(torch.round((v_unit + 1.0) * 3.5), 0, 7).byte()
        
        meta = {
            "k_norm": k_norm,
            "qjl_residuals": qjl_residuals,
            "v_norm": v_norm.half(),
            "v_shape": v_shape
        }
        return q_polar, q_v, meta

    @torch.no_grad()
    def decompress_kv(self, compressed_key: torch.Tensor, compressed_value: torch.Tensor, meta: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        key_decomp = self.decompress_key_vector(compressed_key, meta["k_norm"], meta["qjl_residuals"])
        
        q_v_float = compressed_value.float()
        deq_v_unit = (q_v_float / 3.5) - 1.0
        val_decomp = deq_v_unit * meta["v_norm"].float()
        
        return key_decomp, val_decomp

    def get_compression_ratio(self) -> float:
        original_bits = self.head_dim * 16
        compressed_bits = (self.head_dim * 3) + 16 + (self.qjl_dim * 1)
        return float(original_bits / compressed_bits)
