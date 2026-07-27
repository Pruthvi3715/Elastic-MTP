"""
Entropy Evaluator Module for Elastic-MTP.

Calculates numerical stable Shannon Entropy H(P) and probability divergence metrics
to drive dynamic prediction horizon routing and hallucination filtering.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Dict

class EntropyEvaluator:
    """
    Evaluates probability distribution entropy and uncertainty metrics.
    """
    
    @staticmethod
    def compute_shannon_entropy(logits: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """
        Computes numerically stable Shannon Entropy H(P) = -sum(P * log(P))
        
        Args:
            logits: Tensor of shape (..., vocab_size)
            eps: Small epsilon for numerical clamping
            
        Returns:
            entropy: Tensor of shape (...) containing entropy in nats
        """
        # Convert logits to log-probabilities safely
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Clamp log_probs to avoid 0 * log(0) -> NaN
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return torch.clamp(entropy, min=0.0)

    @staticmethod
    def compute_normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
        """
        Computes normalized entropy H_norm(P) = H(P) / log(|V|)
        Scale: [0.0, 1.0]
        """
        vocab_size = logits.size(-1)
        raw_entropy = EntropyEvaluator.compute_shannon_entropy(logits)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=logits.dtype, device=logits.device))
        return raw_entropy / max_entropy

    @staticmethod
    def compute_kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        """
        Computes KL Divergence D_KL(P || Q) between primary head and auxiliary head logit distributions.
        Used as a contradiction / hallucination detector.
        """
        p_log_probs = F.log_softmax(p_logits, dim=-1)
        q_log_probs = F.log_softmax(q_logits, dim=-1)
        p_probs = torch.exp(p_log_probs)
        
        kl_div = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1)
        return torch.clamp(kl_div, min=0.0)

    @staticmethod
    def evaluate_token_uncertainty(logits: torch.Tensor) -> Dict[str, float]:
        """
        Extracts summary metrics for a single token logit distribution.
        """
        entropy = EntropyEvaluator.compute_shannon_entropy(logits)
        probs = F.softmax(logits, dim=-1)
        top1_prob, top1_idx = torch.max(probs, dim=-1)
        top2_prob = torch.topk(probs, k=2, dim=-1).values[..., 1]
        
        margin = top1_prob - top2_prob
        
        return {
            "entropy": entropy.item() if entropy.numel() == 1 else entropy.tolist(),
            "top1_prob": top1_prob.item() if top1_prob.numel() == 1 else top1_prob.tolist(),
            "margin": margin.item() if margin.numel() == 1 else margin.tolist()
        }
