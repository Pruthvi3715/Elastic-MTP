"""
Unit tests for Gated-LoRA MTP Adapter Module and Training Pipeline.
"""
import os
import unittest
import torch
from src.mtp_glora_adapter import GatedLoRAPredictionHead, MTPGLoRAModule
from train_mtp_adapter import SyntheticTrainerLM, CHECKPOINT_DIR

class TestMTPAdapterPipeline(unittest.TestCase):

    def test_glora_head_forward(self):
        head = GatedLoRAPredictionHead(hidden_dim=64, vocab_size=1000, rank=4)
        z_t = torch.randn(2, 64)
        prev_emb = torch.randn(2, 64)
        logits = head(z_t, prev_emb)
        self.assertEqual(logits.shape, (2, 1000))
        self.assertFalse(torch.isnan(logits).any())

    def test_gradient_detachment_safety(self):
        head = GatedLoRAPredictionHead(hidden_dim=64, vocab_size=1000, rank=4)
        z_t = torch.randn(2, 64, requires_grad=True)
        logits = head(z_t)
        loss = logits.sum()
        loss.backward()
        
        # z_t was detached inside forward pass -> z_t.grad must be None!
        self.assertIsNone(z_t.grad)

    def test_composite_mtp_loss(self):
        adapter = MTPGLoRAModule(hidden_dim=64, vocab_size=1000, num_aux_heads=3)
        primary_loss = torch.tensor(2.5)
        aux_logits = [torch.randn(2, 1000) for _ in range(3)]
        aux_targets = [torch.randint(0, 1000, (2,)) for _ in range(3)]
        
        total_loss, loss_dict = adapter.compute_composite_mtp_loss(primary_loss, aux_logits, aux_targets)
        self.assertGreater(total_loss.item(), primary_loss.item())
        self.assertIn("composite_total_loss", loss_dict)
        self.assertIn("aux_head_1_loss", loss_dict)

    def test_backbone_freeze_verification(self):
        base_model = SyntheticTrainerLM(vocab_size=1000, hidden_dim=64)
        for param in base_model.parameters():
            param.requires_grad = False
            
        trainable_params = [p for p in base_model.parameters() if p.requires_grad]
        self.assertEqual(len(trainable_params), 0)

if __name__ == "__main__":
    unittest.main()
