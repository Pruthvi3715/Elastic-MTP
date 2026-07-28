"""
Unit Tests for Phase 1 Model & Draft Architecture Upgrades.
Tests MTPGLoRAHead Feature Fusion, SoftLogitDistillationTrainer, and RejectionAnalyzer.
"""
import unittest
import torch
import torch.nn as nn
from elastic_mtp.adapters.glora import MTPGLoRAHead
from elastic_mtp.engine.rejection_analyzer import RejectionAnalyzer
from elastic_mtp.engine.inference_engine import SyntheticLM
from train.distill_glora import SoftLogitDistillationTrainer

class TestModelUpgrades(unittest.TestCase):

    def test01_feature_fusion_tensor_shapes(self):
        """Verify concatenated [z_{L/2}; z_L] feature fusion produces correct tensor shapes [batch, seq, hidden_dim]."""
        head = MTPGLoRAHead(hidden_dim=128, vocab_size=1000, rank=16, num_aux_heads=3)
        z_mid = torch.randn((2, 16, 128))
        z_top = torch.randn((2, 16, 128))

        h_fused = head.fuse_latent_features(z_mid, z_top)
        self.assertEqual(h_fused.shape, (2, 16, 128))

        aux_logits = head(z_top, z_mid)
        self.assertEqual(len(aux_logits), 3)
        for logits in aux_logits:
            self.assertEqual(logits.shape, (2, 16, 1000))

    def test02_soft_distillation_loss_convergence(self):
        """Verify soft logit distillation loss converges under temperature T=1.5."""
        base_model = SyntheticLM(vocab_size=1000)
        adapter = MTPGLoRAHead(hidden_dim=128, vocab_size=1000, rank=16, num_aux_heads=3)

        trainer = SoftLogitDistillationTrainer(
            base_model=base_model,
            adapter=adapter,
            temperature=1.5,
            lambda_0=0.3,
            gamma=0.8,
            lr=1e-3
        )

        input_ids = torch.randint(0, 1000, (4, 16))
        initial_metrics = trainer.train_step(input_ids)

        for _ in range(5):
            metrics = trainer.train_step(input_ids)

        self.assertLessEqual(metrics["total_distill_loss"], initial_metrics["total_distill_loss"] + 1e-3)
        self.assertIn("loss_offset_2", metrics)
        self.assertIn("loss_offset_3", metrics)

    def test03_rejection_analyzer_categorization(self):
        """Verify RejectionAnalyzer correctly classifies failure modes into high entropy, low margin flip, and syntax boundary."""
        analyzer = RejectionAnalyzer(tau_entropy=1.5, margin_threshold=0.10)

        # High Entropy Logits
        flat_logits = torch.zeros(1000)
        cat_high = analyzer.analyze_rejection(flat_logits, draft_token_id=5, rejected_position=0)
        self.assertEqual(cat_high, "high_entropy")

        # Low Margin Flip Logits
        peaked_logits = torch.full((1000,), -20.0)
        peaked_logits[10] = 5.0
        peaked_logits[11] = 4.95  # Delta = 0.05 < 0.10
        cat_margin = analyzer.analyze_rejection(peaked_logits, draft_token_id=11, rejected_position=1)
        self.assertEqual(cat_margin, "low_margin_flip")

        # Syntax Boundary Logits
        syntax_logits = torch.full((1000,), -20.0)
        syntax_logits[10] = 10.0
        cat_syntax = analyzer.analyze_rejection(syntax_logits, draft_token_id=10, rejected_position=2, token_text=",")
        self.assertEqual(cat_syntax, "syntax_boundary")

        summary = analyzer.get_rejection_summary()
        self.assertEqual(summary["total_rejections"], 3)
        self.assertEqual(summary["counts"]["high_entropy"], 1)
        self.assertEqual(summary["counts"]["low_margin_flip"], 1)
        self.assertEqual(summary["counts"]["syntax_boundary"], 1)

if __name__ == "__main__":
    unittest.main()
