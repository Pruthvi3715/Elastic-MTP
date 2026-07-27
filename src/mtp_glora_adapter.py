"""
Gated-LoRA Multi-Token Prediction Adapter (MTP-GLoRA) Module.

Implements parameter-efficient auxiliary prediction heads for offset k in {1..K}:
W_effective^(k) = W_0 + g_k * (B_k @ A_k)
g_k = sigmoid(W_g @ [z_t; e(y_{t+k-1})])

Includes gradient detachment between backbone hidden state z_t and auxiliary necks
to prevent autograd graph corruption across ranks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

class GatedLoRAPredictionHead(nn.Module):
    def __init__(self, 
                 hidden_dim: int = 1536, 
                 vocab_size: int = 151936, 
                 rank: int = 8, 
                 head_offset: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.rank = rank
        self.head_offset = head_offset
        
        # Low-rank matrices A_k and B_k
        self.lora_A = nn.Parameter(torch.randn(rank, hidden_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(vocab_size, rank))
        
        # Information-dependent Gating Network: W_g @ [z_t; e(y_{t+k-1})]
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_act = nn.Sigmoid()
        
        # Unembedding projection head (Memory-Efficient Bottleneck for large vocabs)
        if vocab_size > 32000:
            self.head_proj = nn.Sequential(
                nn.Linear(hidden_dim, 128, bias=False),
                nn.Linear(128, vocab_size, bias=False)
            )
        else:
            self.head_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.out_head = None  # Tied dynamically at runtime if provided

    def forward(self, z_t: torch.Tensor, prev_token_emb: torch.Tensor = None) -> torch.Tensor:
        """
        z_t: (batch_size, hidden_dim) - Base model hidden state (Detached)
        prev_token_emb: (batch_size, hidden_dim) - Embedding of offset token
        """
        z_detached = z_t.detach()
        if prev_token_emb is None:
            prev_token_emb = torch.zeros_like(z_detached)
            
        # Compute dynamic gating vector g_k
        concat_input = torch.cat([z_detached, prev_token_emb], dim=-1)
        g_k = self.gate_act(self.gate_proj(concat_input))
        
        # Modulate hidden representation: z_modulated = z_detached * g_k
        z_modulated = z_detached * g_k
        
        if self.out_head is not None:
            return self.out_head(z_modulated)
            
        logits_base = self.head_proj(z_modulated)
        lora_hidden = F.linear(z_detached, self.lora_A)
        logits_lora = F.linear(lora_hidden, self.lora_B)
        return logits_base + logits_lora

class MTPGLoRAModule(nn.Module):
    def __init__(self, 
                 hidden_dim: int = 1536, 
                 vocab_size: int = 151936, 
                 num_aux_heads: int = 3, 
                 rank: int = 8,
                 lambda_0: float = 0.3,
                 gamma: float = 0.8):
        super().__init__()
        self.num_aux_heads = num_aux_heads
        self.lambda_0 = lambda_0
        self.gamma = gamma
        
        # Auxiliary MTP Heads for offsets k = 1..K
        self.aux_heads = nn.ModuleList([
            GatedLoRAPredictionHead(
                hidden_dim=hidden_dim, 
                vocab_size=vocab_size, 
                rank=rank, 
                head_offset=k+1
            ) for k in range(num_aux_heads)
        ])

    def get_aux_weights(self) -> List[float]:
        return [self.lambda_0 * (self.gamma ** i) for i in range(self.num_aux_heads)]

    def compute_composite_mtp_loss(self, 
                                    primary_loss: torch.Tensor, 
                                    aux_logits_list: List[torch.Tensor], 
                                    target_tokens_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes weighted composite loss: L_MTP = L_NTP + sum(lambda_i * CE(P_i, y_{t+1+i}))
        """
        weights = self.get_aux_weights()
        total_loss = primary_loss
        loss_dict = {"primary_loss": primary_loss.item()}
        
        for idx, (aux_logits, targets) in enumerate(zip(aux_logits_list, target_tokens_list)):
            w = weights[idx]
            aux_ce = F.cross_entropy(aux_logits, targets)
            total_loss = total_loss + (w * aux_ce)
            loss_dict[f"aux_head_{idx+1}_loss"] = aux_ce.item()
            loss_dict[f"aux_head_{idx+1}_weight"] = w
            
        loss_dict["composite_total_loss"] = total_loss.item()
        return total_loss, loss_dict
