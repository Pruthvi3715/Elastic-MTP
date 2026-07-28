"""
Rejection Diagnostic Analyzer for Elastic-MTP Speculative Verification.
"""
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from elastic_mtp.config import ElasticMTPConfig

class RejectionAnalyzer:
    """
    Intercepts and categorizes speculative draft token rejections into 3 failure modes:
      1. High Context Entropy (H(P_t) > tau_entropy)
      2. Low-Margin Flips (P(top1) - P(top2) < 0.10)
      3. Syntax / Punctuation Boundaries
    """
    PUNCTUATION_CHARS = {".", ",", ";", ":", "(", ")", "{", "}", "[", "]", "\n", "\t", "=", "+", "-", "*", "/"}

    def __init__(self, tau_entropy: float = ElasticMTPConfig.TAU_ENTROPY, margin_threshold: float = 0.10):
        self.tau_entropy = tau_entropy
        self.margin_threshold = margin_threshold

        self.total_rejections = 0
        self.category_counts = {
            "high_entropy": 0,
            "low_margin_flip": 0,
            "syntax_boundary": 0,
            "other": 0
        }

    def analyze_rejection(self,
                          target_logits: torch.Tensor,
                          draft_token_id: int,
                          rejected_position: int,
                          token_text: Optional[str] = None) -> str:
        """
        Analyzes a single draft rejection event and categorizes the root cause.
        Args:
            target_logits: Target model logits Tensor of shape [vocab_size] or [1, vocab_size]
            draft_token_id: Integer ID of proposed draft token
            rejected_position: Integer sequence position index of rejection
            token_text: Optional decoded string text of rejected token
        Returns:
            category: String name of categorized rejection mode
        """
        if target_logits.dim() > 1:
            logits = target_logits.squeeze(0)
        else:
            logits = target_logits

        logits_clamped = torch.clamp(logits, min=-100.0, max=100.0)
        probs = F.softmax(logits_clamped, dim=-1)
        log_probs = F.log_softmax(logits_clamped, dim=-1)

        # 1. Check Entropy
        entropy = -torch.sum(probs * log_probs).item()
        if entropy > self.tau_entropy:
            category = "high_entropy"
        else:
            # 2. Check Margin
            top_probs, _ = torch.topk(probs, k=2)
            margin = (top_probs[0] - top_probs[1]).item()
            if margin < self.margin_threshold:
                category = "low_margin_flip"
            elif token_text is not None and any(ch in token_text for ch in self.PUNCTUATION_CHARS):
                # 3. Check Syntax Boundary
                category = "syntax_boundary"
            else:
                category = "other"

        self.total_rejections += 1
        self.category_counts[category] += 1
        return category

    def get_rejection_summary(self) -> Dict[str, Any]:
        """
        Returns summary statistics and percentage breakdown across rejection modes.
        Returns:
            summary: Dictionary with count and percentage metrics
        """
        total = max(1, self.total_rejections)
        percentages = {
            f"{k}_percent": round((v / total) * 100.0, 2)
            for k, v in self.category_counts.items()
        }
        return {
            "total_rejections": self.total_rejections,
            "counts": dict(self.category_counts),
            "percentages": percentages
        }
