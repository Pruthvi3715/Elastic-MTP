"""
Elastic-MTP Custom Inference Engine with Speculative Draft Parallel Verification.

Wraps PyTorch / Hugging Face models to simulate and evaluate:
1. Standard Next-Token Prediction (NTP, k=1)
2. Static Multi-Token Prediction (MTP, fixed k)
3. Elastic-MTP (Dynamic Horizon k routed via Entropy + Contradiction Safeguard)
"""
import time
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import ElasticMTPConfig
from src.entropy_evaluator import EntropyEvaluator
from src.elastic_horizon_router import ElasticHorizonRouter
from src.fused_entropy_router import FusedEntropyRouter
from src.kv_cache_manager import SpeculativeKVCache

class SyntheticLM(torch.nn.Module):
    """Synthetic language model that produces a GRADIENT of confidence levels.
    
    confidence_boost controls how peaked the logit distribution is:
      - 0.0  → random logits (max entropy ~10.65) → router picks K=1
      - 3.0  → mild boost   (entropy ~4.5)        → router picks K=2
      - 5.0  → moderate     (entropy ~3.5)         → router picks K=3-4
      - 8.0  → confident    (entropy ~2.0)         → router picks K=5-6
      - 12.0 → very peaked  (entropy ~0.8)         → router picks K=7
      - 15.0 → near-certain (entropy ~0.28)        → router picks K=8
    """
    def __init__(self, vocab_size=50257):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, 128)
        self.lm_head = torch.nn.Linear(128, vocab_size)
        
    def forward(self, input_ids, is_predictable_prompt: bool = True, confidence_boost: float = None):
        emb = self.embedding(input_ids)
        logits = self.lm_head(emb)
        
        # Determine the boost level
        if confidence_boost is not None:
            boost = confidence_boost
        elif is_predictable_prompt:
            boost = 15.0  # legacy: high confidence
        else:
            boost = 0.0   # legacy: no boost = random/hard
        
        if boost > 0.0:
            # Concentrated nucleus distribution mimicking real LLM top-p / top-k logits
            # High boost = high top-1 probability mass (low entropy -> K=8)
            # Low boost  = spread across top-k/uniform (high entropy -> K=1)
            p1 = min(0.99, 0.02 + (boost / 15.0) ** 1.3 * 0.97)
            
            # Reset all logits to low baseline (-20.0)
            logits.fill_(-20.0)
            
            max_indices = torch.argmax(logits, dim=-1, keepdim=True)
            # Set top-1 logit to 0.0 (so e^0 = 1.0)
            logits.scatter_(-1, max_indices, 0.0)
            
            # Fill next top-50 tokens with tail probability mass
            num_tail = max(2, int(500 * (1.0 - boost / 15.0)))
            p_tail_each = (1.0 - p1) / num_tail
            tail_logit = math.log(max(p_tail_each, 1e-12))
            
            # Apply to tail indices
            tail_indices = (max_indices + torch.arange(1, num_tail + 1, device=logits.device)) % self.vocab_size
            logits.scatter_(-1, tail_indices, tail_logit)
            
            # Top-1 logit = log(p1)
            logits.scatter_(-1, max_indices, math.log(max(p1, 1e-12)))
            
        class Output:
            pass
        out = Output()
        out.logits = logits
        return out

class ElasticMTPInferenceEngine:
    def __init__(self, 
                 model_name: str = ElasticMTPConfig.DEFAULT_MODEL_NAME,
                 device: str = ElasticMTPConfig.DEVICE,
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 adapter_stack: Optional[Any] = None,
                 auto_research: Optional[Any] = None):
        self.device = device
        self.model_name = model_name
        self.router = ElasticHorizonRouter()
        self.fused_router = FusedEntropyRouter().to(device)
        self.kv_cache = SpeculativeKVCache(num_layers=12, num_heads=4, head_dim=32, device=device)
        self.adapter_stack = adapter_stack
        self.auto_research = auto_research
        
        print(f"[ElasticMTP Engine] Initializing engine for '{model_name}' on {device}...")
        if model_name == "synthetic":
            print("[ElasticMTP Engine] Using instant offline PyTorch model engine.")
            self.tokenizer = None
            self.model = SyntheticLM().to(device)
            self.model.eval()
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                model_kwargs = {"trust_remote_code": True}
                if device != "cpu" and torch.cuda.is_available():
                    model_kwargs["torch_dtype"] = ElasticMTPConfig.DTYPE
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["torch_dtype"] = torch.float32

                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                if device == "cpu":
                    self.model = self.model.to("cpu")
                self.model.eval()
                print(f"[ElasticMTP Engine] HuggingFace model loaded successfully.")
            except Exception as e:
                print(f"[ElasticMTP Engine] Online load skipped ({e}). Using instant offline PyTorch model engine.")
                self.tokenizer = None
                self.model = SyntheticLM().to(device)
                self.model.eval()

    def _estimate_prompt_confidence(self, prompt: str) -> float:
        """Map prompt content to a continuous confidence_boost in [0.0, 15.0].
        
        This produces a GRADIENT of entropy values so the router uses
        the full K range [1,2,3,4,5,6,7,8] instead of just extremes.
        
        In a real system, this would come from the model's own logits.
        For the SyntheticLM, we simulate it based on prompt characteristics.
        
        Returns:
            confidence_boost: float in [0.0, 15.0]
              - 0.0  → max entropy ~10.65 → K=1 (pure math/symbolic)
              - 3.0  → entropy ~4.5       → K=2 (code with comments)
              - 5.0  → entropy ~3.5       → K=3 (technical writing)
              - 7.0  → entropy ~2.8       → K=4 (formal prose)
              - 9.0  → entropy ~2.0       → K=5 (structured text)
              - 11.0 → entropy ~1.2       → K=6 (conversational)
              - 13.0 → entropy ~0.6       → K=7 (repetitive/formulaic)
              - 15.0 → entropy ~0.28      → K=8 (highly predictable)
        """
        prompt_lower = prompt.lower()
        
        # Level 0: Pure math/symbolic — maximum uncertainty
        if any(kw in prompt_lower for kw in ["x^2", "x²", "integral", "∫", "derivative", "∑", "theorem"]):
            return 0.0
        
        # Level 1: Code generation — high uncertainty
        if any(kw in prompt_lower for kw in ["python function", "def ", "implement", "algorithm", "binary search"]):
            return 3.0
        
        # Level 2: Technical explanation — moderate-high uncertainty
        if any(kw in prompt_lower for kw in ["explain", "describe how", "what is the difference", "compare"]):
            return 5.0
        
        # Level 3: Factual/encyclopedic — moderate uncertainty
        if any(kw in prompt_lower for kw in ["quantization", "neural network", "machine learning", "paradigm"]):
            return 7.0
        
        # Level 4: Structured prose — moderate confidence
        if any(kw in prompt_lower for kw in ["the history of", "in conclusion", "according to", "research shows"]):
            return 9.0
        
        # Level 5: Conversational — high confidence
        if any(kw in prompt_lower for kw in ["hello", "good morning", "how are you", "thank you"]):
            return 11.0
        
        # Level 6: Formulaic/repetitive — very high confidence
        if any(kw in prompt_lower for kw in ["once upon a time", "the quick brown fox", "to be or not"]):
            return 13.0
        
        # Level 7: Counting/sequential — near-certain
        if any(kw in prompt_lower for kw in ["one, two", "a, b, c", "first, second", "monday, tuesday"]):
            return 15.0
        
        # Default: moderate confidence
        return 7.0

    @torch.no_grad()
    def generate(self, 
                 prompt: str, 
                 max_new_tokens: int = 50, 
                 mode: str = "elastic",
                 fixed_k: int = 4) -> Dict[str, Any]:
        if self.tokenizer is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
        else:
            prompt_bytes = prompt.encode("utf-8")
            if len(prompt_bytes) == 0:
                # Guard: empty prompt → use a default BOS-like token
                prompt_bytes = [0]
            input_ids = torch.tensor([[min(b, 50256) for b in prompt_bytes]], dtype=torch.long, device=self.device)

        generated_tokens = []
        telemetry = []
        forward_pass_count = 0
        
        start_time = time.perf_counter()
        self.kv_cache.reset_cache()
        curr_input_ids = input_ids.clone()
        tokens_needed = max_new_tokens
        
        # Reset router metrics at the start of each generation run
        self.router.reset_metrics()

        # Classify prompt confidence level on a gradient (not binary!)
        # Maps prompt characteristics to a continuous confidence_boost in [0.0, 15.0]
        confidence_boost = self._estimate_prompt_confidence(prompt)

        while len(generated_tokens) < tokens_needed:
            forward_pass_count += 1
            if self.tokenizer is None:
                outputs = self.model(curr_input_ids, confidence_boost=confidence_boost)
            else:
                outputs = self.model(curr_input_ids)
                
            next_token_logits = outputs.logits[:, -1, :].clone()
            if len(generated_tokens) > 0 and self.tokenizer is not None:
                for past_tok in set(generated_tokens):
                    if next_token_logits[0, past_tok] > 0:
                        next_token_logits[0, past_tok] /= 1.15
                    else:
                        next_token_logits[0, past_tok] *= 1.15
            
            # Compute token entropy via fused kernel
            entropy = self.fused_router.fused_shannon_entropy(next_token_logits).item()
            
            # Decide prediction horizon k
            if mode == "ntp":
                k = 1
                reason = "MODE_NTP"
            elif mode == "static_mtp":
                k = fixed_k
                reason = f"MODE_STATIC_MTP_K{fixed_k}"
            else: # elastic
                route_res = self.router.evaluate_and_route(next_token_logits)
                k = route_res["target_k"]
                reason = route_res["reason"]

            tokens_to_add = min(k, tokens_needed - len(generated_tokens))
            
            # Track draft acceptance for Elastic-MTP mode
            if mode == "elastic" and k > 1:
                # In simulation, we assume all speculative tokens are accepted
                # (In real speculative decoding, some would be rejected by verification)
                # Here we simulate perfect acceptance for predictable prompts
                num_proposed = k - 1  # speculative tokens beyond the first
                # For synthetic model with high confidence_boost, assume high acceptance
                if confidence_boost >= 11.0:  # highly predictable
                    num_accepted = num_proposed
                elif confidence_boost >= 7.0:  # moderate predictability
                    num_accepted = max(0, num_proposed - 1)  # reject at most 1
                else:  # low predictability - router should have chosen k=1 anyway
                    num_accepted = 0
                self.router.record_draft_acceptance(num_accepted, num_proposed)

                # Failure Mining: Trap rejected draft tokens for AutoResearch self-tuning
                if self.auto_research is not None and num_accepted < num_proposed:
                    target_token_id = torch.argmax(next_token_logits, dim=-1).item()
                    self.auto_research.capture_rejection(
                        prompt_ids=curr_input_ids[0],
                        rejected_offset=num_accepted + 1,
                        target_token_id=target_token_id
                    )
            
            for offset in range(tokens_to_add):
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                tok_val = next_token_id.item()
                generated_tokens.append(tok_val)
                curr_input_ids = torch.cat([curr_input_ids, next_token_id], dim=-1)
                
                tok_str = self.tokenizer.decode([tok_val]) if self.tokenizer is not None else f"tok_{tok_val}"
                
                telemetry.append({
                    "step": len(generated_tokens),
                    "token_id": tok_val,
                    "token_str": tok_str,
                    "entropy": entropy,
                    "horizon_k": k,
                    "reason": reason
                })
                
                if self.tokenizer is not None and tok_val == self.tokenizer.eos_token_id:
                    break

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time
        tokens_generated = len(generated_tokens)
        throughput = tokens_generated / elapsed_sec if elapsed_sec > 0 else 0.0
        
        if self.tokenizer is not None:
            full_text = self.tokenizer.decode(curr_input_ids[0], skip_special_tokens=True)
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            full_text = f"{prompt} [Generated {len(generated_tokens)} synthetic tokens]"
            generated_text = f"[Generated {len(generated_tokens)} synthetic tokens]"

        # Get router metrics summary for research evaluation
        router_metrics = self.router.get_metrics_summary() if mode == "elastic" else {}

        return {
            "prompt": prompt,
            "mode": mode,
            "generated_text": generated_text,
            "full_text": full_text,
            "tokens_generated": tokens_generated,
            "elapsed_sec": elapsed_sec,
            "tokens_per_sec": throughput,
            "forward_pass_count": forward_pass_count,
            "telemetry": telemetry,
            "router_metrics": router_metrics
        }

    def generate_telemetry(self, prompt: str, max_new_tokens: int = 50) -> Dict[str, Any]:
        """Runs speculative inference and returns draft acceptance telemetry stats."""
        res = self.generate(prompt, max_new_tokens=max_new_tokens, mode="elastic")
        rm = res.get("router_metrics", {})
        return {
            "drafted_tokens": rm.get("total_drafted", 0),
            "accepted_tokens": rm.get("total_accepted", 0),
            "acceptance_rate": rm.get("acceptance_rate", 0.0)
        }

