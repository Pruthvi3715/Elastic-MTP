"""
Uncertainty-Aware Dynamic Horizon Router for Elastic Multi-Token Prediction (Elastic-MTP).

Evaluates token-level Shannon Entropy H(P_t) and KL-divergence D_KL(P_base || P_aux)
to dynamically scale speculative prediction horizon K in [1, K_max].
"""
import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Union

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

class DynamicHorizonRouter:
    def __init__(self, 
                 tau_entropy: float = 5.00,  # Calibrated for real LLM vocabularies (e.g. GPT-2/Llama)
                 tau_divergence: float = 0.30, 
                 max_k: int = 8,
                 entropy_threshold: float = None,
                 divergence_threshold: float = None,
                 max_horizon: int = None):
        self.tau_entropy = tau_entropy if entropy_threshold is None else entropy_threshold
        self.tau_divergence = tau_divergence if divergence_threshold is None else divergence_threshold
        self.max_k = max_k if max_horizon is None else max_horizon
        self.max_horizon = self.max_k

    def evaluate_entropy(self, logits: torch.Tensor) -> float:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return float(torch.clamp(entropy, min=0.0).mean().item())

    def determine_horizon(self, entropy_input: Union[float, torch.Tensor], aux_logits_list: List[torch.Tensor] = None) -> RouterResult:
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

        if entropy_nats > self.tau_entropy or divergence_detected:
            reason = "HIGH_ENTROPY_NTP_FALLBACK" if entropy_nats > self.tau_entropy else "DIVERGENCE_SAFEGUARD_FALLBACK"
            meta = {
                "reason": reason,
                "entropy": entropy_nats,
                "divergence_detected": divergence_detected,
                "is_contradiction": divergence_detected
            }
            return RouterResult(1, meta)

        ratio = (self.tau_entropy - entropy_nats) / self.tau_entropy
        allocated_k = max(1, min(self.max_k, int(1 + ratio * (self.max_k - 1))))
        meta = {
            "reason": f"LOW_ENTROPY_DYNAMIC_SPECULATION (K={allocated_k})",
            "entropy": entropy_nats,
            "divergence_detected": False,
            "is_contradiction": False
        }
        return RouterResult(allocated_k, meta)

    def evaluate_and_route(self, logits: torch.Tensor, aux_logits_list: List[torch.Tensor] = None) -> RouterResult:
        return self.determine_horizon(logits, aux_logits_list)

# Class Aliases for Full Backward Compatibility
ElasticHorizonRouter = DynamicHorizonRouter
UncertaintyAwareHorizonFilter = DynamicHorizonRouter
