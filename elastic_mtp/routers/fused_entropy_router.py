"""
Fused Log-Softmax Entropy Router GPU Kernel wrapper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple, Dict, Any, Optional
from elastic_mtp.core.interfaces import BaseRouter
from elastic_mtp.core.registry import register_router

@register_router("fused_entropy")
class FusedEntropyRouter(nn.Module, BaseRouter):
    def __init__(self, 
                 tau_entropy: float = 1.50, 
                 tau_divergence: float = 0.30, 
                 max_k: int = 8):
        super().__init__()
        self.tau_entropy = tau_entropy
        self.tau_divergence = tau_divergence
        self.max_k = max_k

    @torch.no_grad()
    def evaluate_entropy(self, logits: torch.Tensor) -> float:
        ent = self.fused_shannon_entropy(logits)
        return float(ent.mean().item())

    @torch.no_grad()
    def fused_shannon_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        clamped = torch.clamp(logits, min=-100.0, max=100.0)
        log_probs = F.log_softmax(clamped, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return torch.clamp(entropy, min=0.0)

    @torch.no_grad()
    def determine_horizon(self, entropy_input: Any, aux_logits_list: Optional[Any] = None) -> Dict[str, Any]:
        if isinstance(entropy_input, torch.Tensor):
            k, ent = self.determine_horizon_fused(entropy_input)
        else:
            k, ent = self.max_k, float(entropy_input)
        return {"target_k": k, "horizon_k": k, "entropy": ent}

    def get_metrics_summary(self) -> Dict[str, Any]:
        return {"fused_router": "active", "tau_entropy": self.tau_entropy}

    @torch.no_grad()
    def determine_horizon_fused(self, primary_logits: torch.Tensor, aux_logits: torch.Tensor = None) -> Tuple[int, float]:
        entropy = self.fused_shannon_entropy(primary_logits)
        entropy_val = entropy.item() if entropy.numel() == 1 else entropy[0].item()
        
        if entropy_val > self.tau_entropy:
            return 1, entropy_val
            
        if aux_logits is not None:
            log_p1 = F.log_softmax(torch.clamp(primary_logits, -100.0, 100.0), dim=-1)
            log_pi = F.log_softmax(torch.clamp(aux_logits, -100.0, 100.0), dim=-1)
            kl_div = torch.sum(torch.exp(log_p1) * (log_p1 - log_pi), dim=-1).item()
            
            if kl_div > self.tau_divergence:
                return 1, entropy_val
                
        return self.max_k, entropy_val
