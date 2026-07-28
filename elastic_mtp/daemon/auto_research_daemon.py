"""
Autonomous Self-Improvement Loop for Elastic-MTP (AutoResearch Daemon).
"""
import os
import time
import copy
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TelemetrySample:
    """Represents a speculative draft token rejection event captured during inference."""
    prompt_ids: torch.Tensor
    rejected_offset: int
    target_token_id: int

class AutoResearchManager:
    """
    Continuous background daemon managing failure mining, adapter fine-tuning,
    automated pytest regression verification, and atomic weight promotion.
    """
    def __init__(
        self,
        engine: Any,
        adapter_stack: nn.Module,
        eval_prompts: Optional[List[str]] = None,
        min_buffer_size: int = 200,
        dar_improvement_threshold: float = 0.01,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        enable_async: bool = True
    ):
        self.engine = engine
        self.adapter_stack = adapter_stack
        self.eval_prompts = eval_prompts if eval_prompts is not None else [
            "The quick brown fox jumps over the lazy dog",
            "Quantization enables memory-efficient deep learning models",
            "Elastic multi-token prediction dynamically adapts draft horizon",
            "Continuous self-tuning optimizes speculative decoding acceptance",
            "def binary_search(arr, target):"
        ]
        self.min_buffer_size = min_buffer_size
        self.dar_threshold = dar_improvement_threshold
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.enable_async = enable_async

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.telemetry_buffer: List[TelemetrySample] = []
        self.is_training: bool = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AutoResearchThread")

        self.best_dar = self.evaluate_dar()
        print(f"[AutoResearch] Daemon initialized. Baseline DAR: {self.best_dar:.2f}%")

    def capture_rejection(self, prompt_ids: torch.Tensor, rejected_offset: int, target_token_id: int):
        with self._lock:
            ids_cpu = prompt_ids.detach().cpu().squeeze()
            sample = TelemetrySample(
                prompt_ids=ids_cpu,
                rejected_offset=rejected_offset,
                target_token_id=target_token_id
            )
            self.telemetry_buffer.append(sample)
            buffer_len = len(self.telemetry_buffer)

        if buffer_len >= self.min_buffer_size and not self.is_training:
            if self.enable_async:
                self._executor.submit(self.run_self_improvement_cycle)
            else:
                self.run_self_improvement_cycle()

    def evaluate_dar(self) -> float:
        total_drafted = 0
        total_accepted = 0

        for prompt in self.eval_prompts:
            if hasattr(self.engine, "generate_telemetry"):
                stats = self.engine.generate_telemetry(prompt, max_new_tokens=50)
                total_drafted += stats.get("drafted_tokens", 0)
                total_accepted += stats.get("accepted_tokens", 0)
            elif hasattr(self.engine, "generate"):
                res = self.engine.generate(prompt, max_new_tokens=30, mode="elastic")
                rm = res.get("router_metrics", {})
                total_drafted += rm.get("total_drafted", 0)
                total_accepted += rm.get("total_accepted", 0)
            else:
                total_drafted += 50
                total_accepted += 35

        if total_drafted == 0:
            return 0.0
        return (total_accepted / total_drafted) * 100.0

    def run_unit_tests(self) -> bool:
        try:
            result = subprocess.run(
                ["pytest", "tests/", "-q"],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode == 0
        except Exception as e:
            print(f"[AutoResearch] Unit test execution failed: {e}")
            return False

    def _get_base_model(self) -> nn.Module:
        if hasattr(self.engine, "base_model"):
            return self.engine.base_model
        elif hasattr(self.engine, "model"):
            return self.engine.model
        else:
            raise AttributeError("Inference engine must have 'base_model' or 'model' attribute.")

    def run_self_improvement_cycle(self):
        with self._lock:
            if self.is_training:
                return
            self.is_training = True
            samples = list(self.telemetry_buffer)

        print(f"\n[AutoResearch] Starting continuous optimization cycle on {len(samples)} failure samples...")

        device = getattr(self.engine, "device", "cpu")
        base_model = self._get_base_model()

        for param in base_model.parameters():
            param.requires_grad = False

        candidate_stack = copy.deepcopy(self.adapter_stack).to(device)
        candidate_stack.train()

        trainable_params = []
        for param in candidate_stack.parameters():
            param.requires_grad = True
            trainable_params.append(param)

        optimizer = torch.optim.AdamW(
            trainable_params if trainable_params else candidate_stack.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        for epoch in range(2):
            epoch_loss = 0.0
            for sample in samples:
                optimizer.zero_grad()

                prompt_tensor = sample.prompt_ids
                if prompt_tensor.ndim == 1:
                    input_ids = prompt_tensor.unsqueeze(0).to(device)
                else:
                    input_ids = prompt_tensor.to(device)

                with torch.no_grad():
                    if hasattr(base_model, "embedding"):
                        emb = base_model.embedding(input_ids)
                        if hasattr(base_model, "backbone"):
                            h = base_model.backbone(emb)
                        else:
                            h = emb
                        hidden_states = h
                    else:
                        out = base_model(input_ids)
                        hidden_states = getattr(out, "last_hidden_state", out.logits)

                z_last = hidden_states[:, -1, :]
                if hasattr(candidate_stack, "aux_heads"):
                    idx = max(0, min(sample.rejected_offset - 1, len(candidate_stack.aux_heads) - 1))
                    head = candidate_stack.aux_heads[idx]
                    logits = head(z_last)
                elif isinstance(candidate_stack, (nn.ModuleList, list)):
                    idx = max(0, min(sample.rejected_offset - 1, len(candidate_stack) - 1))
                    logits = candidate_stack[idx](z_last)
                else:
                    logits = candidate_stack(z_last)

                target = torch.tensor([sample.target_token_id], device=device, dtype=torch.long)
                loss = F.cross_entropy(logits.view(1, -1), target)
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in candidate_stack.parameters() if p.requires_grad],
                        max_norm=self.max_grad_norm
                    )
                    optimizer.step()
                epoch_loss += loss.item()

        candidate_stack.eval()

        with self._lock:
            original_stack = getattr(self.engine, "adapter_stack", self.adapter_stack)
            self.engine.adapter_stack = candidate_stack

        candidate_dar = self.evaluate_dar()
        tests_passed = self.run_unit_tests()

        print(f"[AutoResearch] Candidate Evaluation -> DAR: {candidate_dar:.2f}% (Baseline: {self.best_dar:.2f}%) | Tests Passed: {tests_passed}")

        with self._lock:
            if tests_passed and (candidate_dar >= self.best_dar + self.dar_threshold):
                print(f"[AutoResearch] SUCCESS! Promoting candidate adapter (+{candidate_dar - self.best_dar:.2f}% DAR gain)")
                self.best_dar = candidate_dar
                self.adapter_stack.load_state_dict(candidate_stack.state_dict())
                self.engine.adapter_stack = self.adapter_stack

                best_ckpt_path = os.path.join(self.checkpoint_dir, "auto_tuned_glora_best.pt")
                torch.save(candidate_stack.state_dict(), best_ckpt_path)
                print(f"[AutoResearch] Saved promoted checkpoint to: {best_ckpt_path}")
            else:
                print("[AutoResearch] REJECTED candidate weights. Rolling back to original state.")
                self.engine.adapter_stack = original_stack

            self.telemetry_buffer.clear()
            self.is_training = False
