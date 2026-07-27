"""
AutoResearch Train Sandbox Module (Agent Editable).

Defines experimental parameters, filter hyperparameters, and logit clamping thresholds.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# === EXPERIMENTAL HYPERPARAMETERS ===
HYPERPARAMS = {
    "TAU_ENTROPY": 1.85,
    "TAU_DIVERGENCE": 0.45,
    "MAX_K": 4,
    "LOGIT_CLAMP_MIN": -100.0,
    "LOGIT_CLAMP_MAX": 100.0,
    "AUX_LOSS_DECAY_GAMMA": 0.8
}

class SandboxHorizonFilter(nn.Module):
    def __init__(self, hp=HYPERPARAMS):
        super().__init__()
        self.tau_entropy = hp["TAU_ENTROPY"]
        self.tau_divergence = hp["TAU_DIVERGENCE"]
        self.max_k = hp["MAX_K"]
        self.clamp_min = hp["LOGIT_CLAMP_MIN"]
        self.clamp_max = hp["LOGIT_CLAMP_MAX"]

    @torch.no_grad()
    def compute_stable_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        clamped_logits = torch.clamp(logits, min=self.clamp_min, max=self.clamp_max)
        log_probs = F.log_softmax(clamped_logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return torch.clamp(entropy, min=0.0)

    @torch.no_grad()
    def compute_kl_divergence(self, primary_logits: torch.Tensor, aux_logits: torch.Tensor) -> torch.Tensor:
        clamped_primary = torch.clamp(primary_logits, min=self.clamp_min, max=self.clamp_max)
        clamped_aux = torch.clamp(aux_logits, min=self.clamp_min, max=self.clamp_max)
        log_p1 = F.log_softmax(clamped_primary, dim=-1)
        log_pi = F.log_softmax(clamped_aux, dim=-1)
        p1 = torch.exp(log_p1)
        kl_div = torch.sum(p1 * (log_p1 - log_pi), dim=-1)
        return torch.clamp(kl_div, min=0.0)

    def determine_horizon(self, primary_logits: torch.Tensor, aux_logits_list: list = None):
        entropy_val = self.compute_stable_entropy(primary_logits).item()
        if entropy_val > self.tau_entropy:
            return 1, {"entropy": entropy_val, "reason": "HIGH_ENTROPY_NTP_FALLBACK", "is_contradiction": False}
        
        accepted_k = 1
        is_contradiction = False
        if aux_logits_list:
            for aux_logits in aux_logits_list:
                kl_val = self.compute_kl_divergence(primary_logits, aux_logits).item()
                if kl_val <= self.tau_divergence:
                    accepted_k += 1
                else:
                    is_contradiction = True
                    break
        return min(accepted_k, self.max_k), {"entropy": entropy_val, "reason": "DYNAMIC_SPECULATION", "is_contradiction": is_contradiction}
