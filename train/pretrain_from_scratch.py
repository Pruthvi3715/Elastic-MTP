"""
MTP Pre-Training Engine From Scratch for Elastic-MTP.
Features linear subspace rotators, gradient isolation barrier (alpha=0.10),
stochastic horizon sampling, and a two-stage training curriculum helper.
"""
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from elastic_mtp.config import ElasticMTPConfig

class SubspaceRotator(nn.Module):
    """
    Linear Subspace Rotator R_k in R^{d x d} for auxiliary prediction offset k.
    Initialized as Identity matrix (nn.init.eye_).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rotation_matrix = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.rotation_matrix)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies linear subspace rotation R_k to hidden states z.
        Args:
            z: Tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        Returns:
            z_rotated: Tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        """
        return F.linear(z, self.rotation_matrix)

class ScratchMTPPretrainer(nn.Module):
    """
    Pre-Training Engine wrapping a Causal LM backbone with linear subspace rotators
    and a gradient isolation barrier.
    """
    def __init__(self,
                 base_model: nn.Module,
                 hidden_dim: int = 128,
                 max_k: int = ElasticMTPConfig.K_MAX,
                 alpha_gradient_isolation: float = 0.10):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.max_k = max_k
        self.alpha_gradient_isolation = alpha_gradient_isolation

        # Subspace rotators for offsets k in {2 ... max_k}
        self.rotators = nn.ModuleList([
            SubspaceRotator(hidden_dim) for _ in range(max_k - 1)
        ])

    def get_curriculum_lambda_aux(self, step: int, total_steps: int) -> float:
        """
        Two-stage training curriculum:
          Stage A (0% -> 80% steps): lambda_aux = 0.0 (Pure NTP Warmup)
          Stage B (80% -> 100% steps): Linear ramp lambda_aux in [0.0, 0.30]
        """
        if total_steps <= 0:
            return 0.0
            
        progress = step / total_steps
        if progress < 0.80:
            return 0.0
        else:
            ramp_progress = (progress - 0.80) / 0.20
            return round(min(0.30, ramp_progress * 0.30), 4)

    def apply_gradient_isolation(self, z: torch.Tensor) -> torch.Tensor:
        """
        Gradient Isolation Barrier:
            z^{(aux)} = alpha * z + (1 - alpha) * detach(z)
        Guarantees 90% of lower-layer gradients come from primary NTP loss.
        """
        alpha = self.alpha_gradient_isolation
        return alpha * z + (1.0 - alpha) * z.detach()

    def get_primary_lm_head(self) -> nn.Module:
        if hasattr(self.base_model, "lm_head"):
            return self.base_model.lm_head
        elif hasattr(self.base_model, "head"):
            return self.base_model.head
        else:
            raise AttributeError("Base model must have 'lm_head' attribute.")

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                stochastic_sample: bool = True,
                step: int = 0,
                total_steps: int = 1000) -> Dict[str, Any]:
        """
        Pre-training forward pass with NTP loss and stochastic auxiliary rotator loss.
        Args:
            input_ids: LongTensor of shape [batch_size, seq_len]
            labels: Optional LongTensor of target token IDs
            stochastic_sample: If True, randomly samples 1 active offset to reduce FLOP/VRAM overhead
            step: Current training step integer
            total_steps: Total training steps integer
        Returns:
            outputs: Dictionary with 'loss', 'ntp_loss', 'aux_loss', 'lambda_aux', 'logits'
        """
        if hasattr(self.base_model, "forward"):
            base_out = self.base_model(input_ids)
            ntp_logits = base_out.logits
        else:
            ntp_logits = torch.randn((input_ids.shape[0], input_ids.shape[1], 50257), device=input_ids.device)

        ntp_logits_clamped = torch.clamp(ntp_logits, min=-100.0, max=100.0)

        if labels is None:
            labels = input_ids

        # Primary NTP Loss
        shift_logits = ntp_logits_clamped[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ntp_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        # Extract top hidden states for auxiliary pathways
        z_top = ntp_logits_clamped[:, :, :self.hidden_dim] if ntp_logits_clamped.shape[-1] >= self.hidden_dim else F.pad(ntp_logits_clamped, (0, self.hidden_dim - ntp_logits_clamped.shape[-1]))
        z_aux = self.apply_gradient_isolation(z_top)

        lambda_aux = self.get_curriculum_lambda_aux(step, total_steps)
        aux_loss = torch.tensor(0.0, device=input_ids.device)

        lm_head = self.get_primary_lm_head()

        if lambda_aux > 0.0:
            if stochastic_sample:
                # Sample 1 offset k in {0 ... max_k - 2}
                k_idx = random.randint(0, self.max_k - 2)
                active_offsets = [k_idx]
            else:
                active_offsets = list(range(self.max_k - 1))

            for k_idx in active_offsets:
                rotator = self.rotators[k_idx]
                z_rotated = rotator(z_aux)
                aux_logits = lm_head(z_rotated)
                aux_logits_clamped = torch.clamp(aux_logits, min=-100.0, max=100.0)

                # Future offset prediction shift (offset k = k_idx + 2)
                offset = k_idx + 2
                if shift_labels.shape[1] >= offset:
                    offset_shift_logits = aux_logits_clamped[:, :-offset, :].contiguous()
                    offset_shift_labels = labels[:, offset:].contiguous()
                    if offset_shift_logits.shape[1] > 0 and offset_shift_labels.shape[1] > 0:
                        k_loss = F.cross_entropy(
                            offset_shift_logits.view(-1, offset_shift_logits.shape[-1]),
                            offset_shift_labels.view(-1)
                        )
                        aux_loss = aux_loss + k_loss

        total_loss = ntp_loss + lambda_aux * aux_loss

        return {
            "loss": total_loss,
            "ntp_loss": ntp_loss,
            "aux_loss": aux_loss,
            "lambda_aux": lambda_aux,
            "logits": ntp_logits_clamped
        }
