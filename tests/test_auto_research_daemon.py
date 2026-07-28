"""
Unit and Integration Tests for AutoResearch Daemon (src/auto_research_daemon.py).
"""
import os
import copy
import shutil
import unittest
import torch
import torch.nn as nn

from src.auto_research_daemon import AutoResearchManager, TelemetrySample
from src.mtp_glora_adapter import MTPGLoRAModule
from src.inference_engine import ElasticMTPInferenceEngine, SyntheticLM


class DummyEngine:
    def __init__(self, vocab_size=1000, hidden_dim=64, device="cpu"):
        self.device = device
        self.base_model = SyntheticLM(vocab_size=vocab_size).to(device)
        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.adapter_stack = MTPGLoRAModule(hidden_dim=128, vocab_size=vocab_size, num_aux_heads=3).to(device)

    def generate_telemetry(self, prompt: str, max_new_tokens: int = 50):
        return {
            "drafted_tokens": 50,
            "accepted_tokens": 40,
            "acceptance_rate": 80.0
        }


class TestAutoResearchDaemon(unittest.TestCase):

    def setUp(self):
        self.test_checkpoint_dir = "test_checkpoints"
        os.makedirs(self.test_checkpoint_dir, exist_ok=True)
        self.engine = DummyEngine()

    def tearDown(self):
        if os.path.exists(self.test_checkpoint_dir):
            shutil.rmtree(self.test_checkpoint_dir, ignore_errors=True)

    def test_telemetry_sample_capture(self):
        daemon = AutoResearchManager(
            engine=self.engine,
            adapter_stack=self.engine.adapter_stack,
            min_buffer_size=5,
            checkpoint_dir=self.test_checkpoint_dir
        )
        self.assertEqual(len(daemon.telemetry_buffer), 0)

        prompt_ids = torch.tensor([1, 2, 3, 4, 5])
        daemon.capture_rejection(prompt_ids, rejected_offset=1, target_token_id=42)

        self.assertEqual(len(daemon.telemetry_buffer), 1)
        sample = daemon.telemetry_buffer[0]
        self.assertIsInstance(sample, TelemetrySample)
        self.assertEqual(sample.rejected_offset, 1)
        self.assertEqual(sample.target_token_id, 42)

    def test_evaluate_dar(self):
        daemon = AutoResearchManager(
            engine=self.engine,
            adapter_stack=self.engine.adapter_stack,
            eval_prompts=["test prompt 1", "test prompt 2"],
            checkpoint_dir=self.test_checkpoint_dir
        )
        dar = daemon.evaluate_dar()
        self.assertAlmostEqual(dar, 80.0, places=2)

    def test_backbone_freeze_enforcement(self):
        daemon = AutoResearchManager(
            engine=self.engine,
            adapter_stack=self.engine.adapter_stack,
            min_buffer_size=2,
            checkpoint_dir=self.test_checkpoint_dir
        )

        for param in self.engine.base_model.parameters():
            self.assertFalse(param.requires_grad)

    def test_self_improvement_cycle_promotion(self):
        daemon = AutoResearchManager(
            engine=self.engine,
            adapter_stack=self.engine.adapter_stack,
            min_buffer_size=2,
            dar_improvement_threshold=0.0,
            checkpoint_dir=self.test_checkpoint_dir,
            enable_async=False
        )

        # Mock run_unit_tests to return True
        daemon.run_unit_tests = lambda: True

        # Fill buffer with failure samples
        prompt_ids = torch.randint(0, 1000, (10,))
        daemon.capture_rejection(prompt_ids, rejected_offset=1, target_token_id=15)
        daemon.capture_rejection(prompt_ids, rejected_offset=2, target_token_id=30)

        # Force baseline DAR low so candidate beats it
        daemon.best_dar = 50.0
        daemon.evaluate_dar = lambda: 85.0

        daemon.run_self_improvement_cycle()

        # Verify candidate promotion
        self.assertEqual(daemon.best_dar, 85.0)
        self.assertEqual(len(daemon.telemetry_buffer), 0)
        ckpt_path = os.path.join(self.test_checkpoint_dir, "auto_tuned_glora_best.pt")
        self.assertTrue(os.path.exists(ckpt_path))

    def test_self_improvement_cycle_rollback(self):
        daemon = AutoResearchManager(
            engine=self.engine,
            adapter_stack=self.engine.adapter_stack,
            min_buffer_size=2,
            dar_improvement_threshold=5.0,
            checkpoint_dir=self.test_checkpoint_dir,
            enable_async=False
        )

        # Mock unit test failure
        daemon.run_unit_tests = lambda: False

        prompt_ids = torch.randint(0, 1000, (10,))
        daemon.capture_rejection(prompt_ids, rejected_offset=1, target_token_id=15)
        daemon.capture_rejection(prompt_ids, rejected_offset=2, target_token_id=30)

        initial_dar = daemon.best_dar
        daemon.run_self_improvement_cycle()

        # Best DAR must remain unchanged due to unit test rollback
        self.assertEqual(daemon.best_dar, initial_dar)
        self.assertEqual(len(daemon.telemetry_buffer), 0)

    def test_gradient_clipping_and_no_nan(self):
        daemon = AutoResearchManager(
            engine=self.engine,
            adapter_stack=self.engine.adapter_stack,
            min_buffer_size=1,
            max_grad_norm=0.5,
            checkpoint_dir=self.test_checkpoint_dir,
            enable_async=False
        )
        daemon.run_unit_tests = lambda: True
        daemon.evaluate_dar = lambda: 90.0

        prompt_ids = torch.randint(0, 1000, (8,))
        daemon.capture_rejection(prompt_ids, rejected_offset=1, target_token_id=99)

        daemon.run_self_improvement_cycle()

        for param in daemon.adapter_stack.parameters():
            self.assertFalse(torch.isnan(param).any())
            self.assertFalse(torch.isinf(param).any())

    def test_integration_with_inference_engine(self):
        engine = ElasticMTPInferenceEngine(model_name="synthetic", device="cpu")
        adapter = MTPGLoRAModule(hidden_dim=128, vocab_size=50257, num_aux_heads=3).to("cpu")

        daemon = AutoResearchManager(
            engine=engine,
            adapter_stack=adapter,
            min_buffer_size=100,
            checkpoint_dir=self.test_checkpoint_dir
        )
        engine.auto_research = daemon
        engine.adapter_stack = adapter

        res = engine.generate("explain quantization in deep learning", max_new_tokens=20, mode="elastic")
        self.assertIn("generated_text", res)
        # Verify telemetry capture works during generate
        self.assertGreaterEqual(len(daemon.telemetry_buffer), 0)


if __name__ == "__main__":
    unittest.main()
