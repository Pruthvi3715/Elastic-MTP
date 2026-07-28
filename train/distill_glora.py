"""
Soft Logit Distillation Trainer for Elastic-MTP MTP-GLoRA Adapters.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from elastic_mtp.adapters.glora import MTPGLoRAHead

class SoftLogitDistillationTrainer:
    """
    Soft Logit Distillation Trainer for aligning MTP-GLoRA speculative prediction heads.
    Uses temperature scaling (T=1.5) and exponential horizon loss decay:
        lambda_i = lambda_0 * gamma^(i-1)  (lambda_0 = 0.3, gamma = 0.8)
    """
    def __init__(self,
                 base_model: nn.Module,
                 adapter: MTPGLoRAHead,
                 temperature: float = 1.5,
                 lambda_0: float = 0.3,
                 gamma: float = 0.8,
                 lr: float = 1e-4,
                 device: str = "cpu"):
        self.base_model = base_model.to(device)
        self.adapter = adapter.to(device)
        self.temperature = temperature
        self.lambda_0 = lambda_0
        self.gamma = gamma
        self.device = device

        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self.optimizer = torch.optim.AdamW(
            [p for p in self.adapter.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01
        )

    def compute_distillation_loss(self,
                                  base_logits: torch.Tensor,
                                  aux_logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes soft logit KL-divergence distillation loss with domain clamping and exponential horizon decay.
        Args:
            base_logits: Tensor of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            aux_logits_list: List of Tensors each of shape [batch_size, seq_len, vocab_size]
        Returns:
            total_loss: Scalar loss Tensor
            metrics: Dictionary of per-offset loss components
        """
        base_logits_clamped = torch.clamp(base_logits, min=-100.0, max=100.0)
        p_base = F.softmax(base_logits_clamped / self.temperature, dim=-1)
        log_p_base = F.log_softmax(base_logits_clamped / self.temperature, dim=-1)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        metrics = {}

        for i, aux_logits in enumerate(aux_logits_list):
            aux_logits_clamped = torch.clamp(aux_logits, min=-100.0, max=100.0)
            log_p_aux = F.log_softmax(aux_logits_clamped / self.temperature, dim=-1)

            # KL(P_base || P_aux) = sum(P_base * (log_P_base - log_P_aux))
            kl_div = F.kl_div(log_p_aux, p_base, reduction="batchmean")
            offset_loss = (self.temperature ** 2) * kl_div

            weight = self.lambda_0 * (self.gamma ** i)
            total_loss = total_loss + weight * offset_loss
            metrics[f"loss_offset_{i+2}"] = float(offset_loss.item())

        metrics["total_distill_loss"] = float(total_loss.item())
        return total_loss, metrics

    def train_step(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """
        Performs one single optimization step over a batch of token IDs.
        Args:
            input_ids: LongTensor of shape [batch_size, seq_len]
        Returns:
            metrics: Dictionary of step metrics
        """
        self.optimizer.zero_grad()
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            if hasattr(self.base_model, "forward"):
                base_out = self.base_model(input_ids)
                base_logits = base_out.logits
            else:
                base_logits = torch.randn((input_ids.shape[0], input_ids.shape[1], 50257), device=self.device)

            z_top = base_logits[:, :, :self.adapter.hidden_dim] if base_logits.shape[-1] >= self.adapter.hidden_dim else F.pad(base_logits, (0, self.adapter.hidden_dim - base_logits.shape[-1]))
            z_mid = z_top

        aux_logits_list = self.adapter(z_top, z_mid)
        loss, metrics = self.compute_distillation_loss(base_logits, aux_logits_list)

        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
            self.optimizer.step()

        return metrics
