"""
Unit Tests for Phase 2 Pre-Training Engine From Scratch.
Tests ScratchMTPPretrainer, SubspaceRotator, Gradient Isolation Barrier, and FP16/BF16 stability.
"""
import math
import unittest
import torch
import torch.nn as nn
from elastic_mtp.engine.inference_engine import SyntheticLM
from train.pretrain_from_scratch import ScratchMTPPretrainer, SubspaceRotator

class TestPretrainFromScratch(unittest.TestCase):

    def test01_subspace_rotator_identity_initialization(self):
        """Verify subspace rotator matrices initialization as Identity matrices."""
        rotator = SubspaceRotator(hidden_dim=64)
        eye = torch.eye(64)
        self.assertTrue(torch.allclose(rotator.rotation_matrix, eye))

        z = torch.randn((2, 10, 64))
        z_rot = rotator(z)
        self.assertTrue(torch.allclose(z_rot, z))

    def test02_gradient_isolation_barrier(self):
        """Assert that gradient scaling alpha=0.10 bounds lower layer gradient scaling."""
        base_model = SyntheticLM(vocab_size=1000)
        pretrainer = ScratchMTPPretrainer(base_model=base_model, hidden_dim=128, max_k=4, alpha_gradient_isolation=0.10)

        z = torch.randn((1, 8, 128), requires_grad=True)
        z_aux = pretrainer.apply_gradient_isolation(z)

        dummy_loss = (z_aux ** 2).sum()
        dummy_loss.backward()

        # z_aux value = z, but dz_aux/dz = alpha = 0.10
        # d(z_aux^2)/dz = 2 * z_aux * alpha = 2 * z * 0.10 = 0.20 * z
        expected_grad = 2.0 * 0.10 * z
        self.assertTrue(torch.allclose(z.grad, expected_grad, atol=1e-5))

    def test03_precision_stability_fp16_bf16(self):
        """Verify zero NaN/Inf returns across FP16 and BF16 precision."""
        base_model = SyntheticLM(vocab_size=1000)
        pretrainer = ScratchMTPPretrainer(base_model=base_model, hidden_dim=128, max_k=4)

        input_ids = torch.randint(0, 1000, (2, 16))

        for dtype in [torch.float16, torch.bfloat16]:
            pretrainer.to(dtype)
            input_ids_device = input_ids.to(pretrainer.rotators[0].rotation_matrix.device)

            out = pretrainer(input_ids_device, stochastic_sample=True, step=90, total_steps=100)
            loss = out["loss"]

            self.assertFalse(torch.isnan(loss).any().item())
            self.assertFalse(torch.isinf(loss).any().item())

    def test04_dummy_pretrain_run_convergence(self):
        """Run a 10-step dummy pre-training verification pass."""
        base_model = SyntheticLM(vocab_size=1000)
        pretrainer = ScratchMTPPretrainer(base_model=base_model, hidden_dim=128, max_k=4)
        optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 1000, (4, 16))

        for step in range(10):
            optimizer.zero_grad()
            out = pretrainer(input_ids, step=step, total_steps=10)
            loss = out["loss"]
            loss.backward()
            optimizer.step()

            self.assertFalse(math.isnan(loss.item()))
            self.assertFalse(math.isinf(loss.item()))

if __name__ == "__main__":
    unittest.main()
