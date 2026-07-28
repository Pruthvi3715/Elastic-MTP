"""
Uncertainty-Aware Dynamic Horizon Router for Elastic Multi-Token Prediction (Elastic-MTP).
"""
import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Union, Optional

from elastic_mtp.config import ElasticMTPConfig
from elastic_mtp.core.interfaces import BaseRouter
from elastic_mtp.core.registry import register_router

class RouterResult(dict):
    """Dictionary subclass supporting tuple unpacking (k, meta) for backward compatibility."""
    def __init__(self, k: int, meta: dict):
        super().__init__(meta)
        self["target_k"] = k
        self["horizon_k"] = k
        self.k = k

    def __iter__(self):
        yield self.k
        yield dict(self)

@register_router("elastic_1d")
@register_router("dynamic_1d")
class DynamicHorizonRouter(BaseRouter):
    def __init__(self, 
                 tau_entropy: Optional[float] = None,
                 tau_divergence: Optional[float] = None, 
                 max_k: Optional[int] = None,
                 entropy_threshold: Optional[float] = None,
                 divergence_threshold: Optional[float] = None,
                 max_horizon: Optional[int] = None):
        default_tau_e = ElasticMTPConfig.ENTROPY_LOW_THRESHOLD
        default_tau_d = ElasticMTPConfig.CONTRADICTION_THRESHOLD
        default_k_max = ElasticMTPConfig.K_MAX

        self.tau_entropy = entropy_threshold if entropy_threshold is not None else (tau_entropy if tau_entropy is not None else default_tau_e)
        self.tau_divergence = divergence_threshold if divergence_threshold is not None else (tau_divergence if tau_divergence is not None else default_tau_d)
        self.max_k = max_horizon if max_horizon is not None else (max_k if max_k is not None else default_k_max)
        self.max_horizon = self.max_k
        
        # Metrics tracking for research evaluation
        self.total_draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.contradiction_events = 0
        self.routing_decisions = []

    def evaluate_entropy(self, logits: torch.Tensor) -> float:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return float(torch.clamp(entropy, min=0.0).mean().item())

    def determine_horizon(self, entropy_input: Union[float, torch.Tensor], aux_logits_list: Optional[List[torch.Tensor]] = None) -> RouterResult:
        if isinstance(entropy_input, torch.Tensor):
            if entropy_input.dim() > 0 and entropy_input.shape[-1] > 1:
                entropy_nats = self.evaluate_entropy(entropy_input)
                base_logits = entropy_input
            else:
                entropy_nats = float(entropy_input.item()) if entropy_input.numel() == 1 else 0.0
                base_logits = None
        else:
            entropy_nats = float(entropy_input)
            base_logits = None

        divergence_detected = False
        if aux_logits_list is not None and len(aux_logits_list) > 0:
            primary = base_logits if base_logits is not None else aux_logits_list[0]
            aux = aux_logits_list[-1]
            if primary is not aux:
                base_p = F.softmax(primary, dim=-1)
                aux_p = F.softmax(aux, dim=-1)
                kl_div = torch.sum(base_p * (torch.log(base_p + 1e-8) - torch.log(aux_p + 1e-8)), dim=-1).item()
                if kl_div > self.tau_divergence:
                    divergence_detected = True
                    self.contradiction_events += 1

        if entropy_nats > self.tau_entropy or divergence_detected:
            reason = "HIGH_ENTROPY_NTP_FALLBACK" if entropy_nats > self.tau_entropy else "DIVERGENCE_SAFEGUARD_FALLBACK"
            meta = {
                "reason": reason,
                "entropy": entropy_nats,
                "divergence_detected": divergence_detected,
                "is_contradiction": divergence_detected
            }
            self.routing_decisions.append({
                "k": 1,
                "entropy": entropy_nats,
                "contradiction": divergence_detected,
                "reason": reason
            })
            return RouterResult(1, meta)

        ratio = max(0.0, min(1.0, (self.tau_entropy - entropy_nats) / self.tau_entropy))
        allocated_k = max(1, min(self.max_k, int(round(1 + ratio * (self.max_k - 1)))))
        
        self.total_draft_tokens += (allocated_k - 1)
        
        meta = {
            "reason": f"LOW_ENTROPY_DYNAMIC_SPECULATION (K={allocated_k})",
            "entropy": entropy_nats,
            "divergence_detected": False,
            "is_contradiction": False
        }
        self.routing_decisions.append({
            "k": allocated_k,
            "entropy": entropy_nats,
            "contradiction": False,
            "reason": meta["reason"]
        })
        return RouterResult(allocated_k, meta)
    
    def record_draft_acceptance(self, num_accepted: int, num_proposed: int):
        self.accepted_draft_tokens += num_accepted
        self.total_draft_tokens += num_proposed
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        dar = (self.accepted_draft_tokens / self.total_draft_tokens * 100.0) if self.total_draft_tokens > 0 else 0.0
        contradiction_rate = (self.contradiction_events / len(self.routing_decisions) * 100.0) if self.routing_decisions else 0.0
        
        k_counts = {}
        for decision in self.routing_decisions:
            k_val = decision["k"]
            k_counts[k_val] = k_counts.get(k_val, 0) + 1
        
        return {
            "draft_acceptance_rate_percent": round(dar, 2),
            "contradiction_rate_percent": round(contradiction_rate, 2),
            "total_routing_decisions": len(self.routing_decisions),
            "total_draft_tokens_proposed": self.total_draft_tokens,
            "total_draft_tokens_accepted": self.accepted_draft_tokens,
            "contradiction_events": self.contradiction_events,
            "k_distribution": k_counts,
            "avg_k": sum(d["k"] for d in self.routing_decisions) / len(self.routing_decisions) if self.routing_decisions else 0.0
        }
    
    def reset_metrics(self):
        self.total_draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.contradiction_events = 0
        self.routing_decisions = []

    def evaluate_and_route(self, logits: torch.Tensor, aux_logits_list: Optional[List[torch.Tensor]] = None) -> RouterResult:
        return self.determine_horizon(logits, aux_logits_list)

# Backward Compatibility Aliases
ElasticHorizonRouter = DynamicHorizonRouter
UncertaintyAwareHorizonFilter = DynamicHorizonRouter
