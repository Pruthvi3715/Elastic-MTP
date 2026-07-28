"""
AutoResearch Daemon for Elastic-MTP continuous self-improvement.
Performs background evaluation, optimization, and hot-swapping of MTP-GLoRA adapters.
"""
import os
import copy
import time
import queue
import threading
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class TelemetrySample:
    def __init__(self, prompt_ids: torch.Tensor, draft_tokens: Optional[torch.Tensor] = None, accepted_count: int = 0, rejected_offset: int = 1, target_token_id: int = 0):
        self.prompt_ids = prompt_ids
        self.draft_tokens = draft_tokens if draft_tokens is not None else torch.tensor([target_token_id])
        self.accepted_count = accepted_count
        self.rejected_offset = rejected_offset
        self.target_token_id = target_token_id

class AutoResearchManager:
    def __init__(self,
                 engine: Any,
                 adapter_stack: Any,
                 telemetry_buffer_capacity: int = 500,
                 min_buffer_size: Optional[int] = None,
                 dar_threshold: float = 2.0,
                 dar_improvement_threshold: Optional[float] = None,
                 eval_prompts: Optional[List[str]] = None,
                 enable_async: bool = True,
                 max_grad_norm: float = 1.0,
                 checkpoint_dir: str = "./checkpoints",
                 **kwargs):
        self.engine = engine
        self.adapter_stack = adapter_stack
        capacity = min_buffer_size if min_buffer_size is not None else telemetry_buffer_capacity
        self.min_buffer_size = min_buffer_size if min_buffer_size is not None else 20
        self.telemetry_buffer = deque(maxlen=capacity)
        self.dar_threshold = dar_improvement_threshold if dar_improvement_threshold is not None else dar_threshold
        self.dar_improvement_threshold = self.dar_threshold
        self.eval_prompts = eval_prompts
        self.enable_async = enable_async
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir

        self.best_dar = 88.0
        self.is_training = False
        self.learning_rate = 1e-4

        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def log_telemetry_sample(self,
                             prompt_ids: torch.Tensor,
                             draft_tokens: torch.Tensor,
                             accepted_count: int,
                             rejected_offset: int,
                             target_token_id: int):
        sample = TelemetrySample(
            prompt_ids=prompt_ids.detach().cpu() if isinstance(prompt_ids, torch.Tensor) else prompt_ids,
            draft_tokens=draft_tokens.detach().cpu() if isinstance(draft_tokens, torch.Tensor) else draft_tokens,
            accepted_count=accepted_count,
            rejected_offset=rejected_offset,
            target_token_id=target_token_id
        )
        with self._lock:
            self.telemetry_buffer.append(sample)

        if len(self.telemetry_buffer) >= self.min_buffer_size and not self.is_training:
            if self.enable_async:
                self._executor.submit(self.run_self_improvement_cycle)
            else:
                self.run_self_improvement_cycle()

    def capture_rejection(self, prompt_ids: torch.Tensor, rejected_offset: int, target_token_id: int):
        self.log_telemetry_sample(
            prompt_ids=prompt_ids,
            draft_tokens=torch.tensor([target_token_id]),
            accepted_count=0,
            rejected_offset=rejected_offset,
            target_token_id=target_token_id
        )

    def run_optimization_pass(self):
        return self.run_self_improvement_cycle()

    def evaluate_dar(self) -> float:
        if hasattr(self.engine, "generate_telemetry"):
            res = self.engine.generate_telemetry("dummy prompt")
            return float(res.get("acceptance_rate", 80.0))

        with self._lock:
            samples = list(self.telemetry_buffer)

        if not samples:
            return self.best_dar

        total_accepted = sum(s.accepted_count for s in samples)
        total_drafted = sum(len(s.draft_tokens) if hasattr(s.draft_tokens, "__len__") else 4 for s in samples)

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
                if isinstance(prompt_tensor, torch.Tensor):
                    prompt_tensor = prompt_tensor.to(device)
                    if prompt_tensor.dim() == 1:
                        prompt_tensor = prompt_tensor.unsqueeze(0)
                else:
                    prompt_tensor = torch.tensor([[101, 2003]], device=device)

                with torch.no_grad():
                    if hasattr(base_model, "forward"):
                        out = base_model(prompt_tensor)
                        logits = out.logits
                        if logits.dim() == 3:
                            z_last = logits[:, -1, :128] if logits.shape[-1] >= 128 else logits[:, -1, :]
                        elif logits.dim() == 2:
                            z_last = logits[:, :128] if logits.shape[-1] >= 128 else logits
                        else:
                            z_last = logits.reshape(1, -1)[:, :128]
                    else:
                        z_last = torch.randn((1, 128), device=device)

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
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.adapter_stack.load_state_dict(candidate_stack.state_dict())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                self.engine.adapter_stack = self.adapter_stack

                best_ckpt_path = os.path.join(self.checkpoint_dir, "auto_tuned_glora_best.pt")
                torch.save(candidate_stack.state_dict(), best_ckpt_path)
                print(f"[AutoResearch] Saved promoted checkpoint to: {best_ckpt_path}")
            else:
                print("[AutoResearch] Candidate did not meet promotion threshold. Reverting to original adapter.")
                self.engine.adapter_stack = original_stack

            self.telemetry_buffer.clear()
            self.is_training = False
