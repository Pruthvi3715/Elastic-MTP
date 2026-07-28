"""
Multi-Layer Latent Feature Fusion MTP-GLoRA Adapter for Elastic-MTP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from elastic_mtp.core.interfaces import BaseAdapter
from elastic_mtp.core.registry import register_adapter

@register_adapter("mtp_glora_fused")
class MTPGLoRAHead(nn.Module, BaseAdapter):
    """
    Multi-Layer Latent Feature Fusion MTP-GLoRA Head.
    Fuses hidden representations from an intermediate layer (L/2) and top layer (L):
        h_fused = SiLU(W_f * [z_{L/2}; z_L])
    Passes h_fused into gated low-rank adapter projections (B_k * A_k).
    """
    def __init__(self, hidden_dim: int = 128, vocab_size: int = 50257, rank: int = 16, num_aux_heads: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.rank = rank
        self.num_aux_heads = num_aux_heads

        # Fusion projection: [batch_size, seq_len, 2 * hidden_dim] -> [batch_size, seq_len, hidden_dim]
        self.W_f = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)

        self.A_projections = nn.ModuleList([
            nn.Linear(hidden_dim, rank, bias=False) for _ in range(num_aux_heads)
        ])
        self.B_projections = nn.ModuleList([
            nn.Linear(rank, hidden_dim, bias=False) for _ in range(num_aux_heads)
        ])
        self.gate_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_aux_heads)
        ])
        self.lm_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size, bias=False) for _ in range(num_aux_heads)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_f.weight, a=1.0)
        for A, B, gate in zip(self.A_projections, self.B_projections, self.gate_projections):
            nn.init.kaiming_uniform_(A.weight, a=1.0)
            nn.init.zeros_(B.weight)
            nn.init.constant_(gate.bias, -2.0)

    def fuse_latent_features(self, z_mid: torch.Tensor, z_top: torch.Tensor) -> torch.Tensor:
        """
        Fuses intermediate layer activation z_mid and top layer activation z_top.
        Args:
            z_mid: Tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            z_top: Tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        Returns:
            h_fused: Tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        """
        if z_mid.dim() != z_top.dim():
            raise ValueError(f"Shape mismatch: z_mid dim {z_mid.dim()} != z_top dim {z_top.dim()}")
            
        z_concat = torch.cat([z_mid, z_top], dim=-1)
        h_fused = F.silu(self.W_f(z_concat))
        return h_fused

    def forward_offset(self, h_fused: torch.Tensor, offset_idx: int) -> torch.Tensor:
        """
        Forward pass for a single future prediction offset.
        Args:
            h_fused: Tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            offset_idx: Integer index of prediction head in range [0, num_aux_heads - 1]
        Returns:
            logits: Tensor of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
        """
        idx = max(0, min(offset_idx, self.num_aux_heads - 1))
        gate = torch.sigmoid(self.gate_projections[idx](h_fused))
        delta_h = self.B_projections[idx](self.A_projections[idx](h_fused))
        adapted_h = h_fused + gate * delta_h
        logits = self.lm_heads[idx](adapted_h)
        return logits

    def forward(self, z: torch.Tensor, z_mid: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass producing logits for all auxiliary prediction heads.
        Args:
            z: Top layer hidden states of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            z_mid: Optional intermediate layer hidden states of shape [batch_size, seq_len, hidden_dim]
        Returns:
            aux_logits: List of Tensors each of shape [batch_size, seq_len, vocab_size]
        """
        if z_mid is None:
            z_mid = z
            
        h_fused = self.fuse_latent_features(z_mid, z)
        outputs = []
        for i in range(self.num_aux_heads):
            outputs.append(self.forward_offset(h_fused, i))
        return outputs
