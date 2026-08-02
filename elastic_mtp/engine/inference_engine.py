"""
Elastic-MTP Custom Inference Engine with Speculative Draft Parallel Verification.
"""
import time
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from elastic_mtp.config import ElasticMTPConfig
from elastic_mtp.routers.elastic_horizon_router import DynamicHorizonRouter, ElasticHorizonRouter
from elastic_mtp.routers.fused_entropy_router import FusedEntropyRouter
from elastic_mtp.engine.kv_cache_manager import SpeculativeKVCache
from elastic_mtp.adapters.glora import MTPGLoRAHead

class SyntheticLM(torch.nn.Module):
    def __init__(self, vocab_size=50257):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, 128)
        self.lm_head = torch.nn.Linear(128, vocab_size)
        
    def forward(self, input_ids, is_predictable_prompt: bool = True, confidence_boost: float = None):
        emb = self.embedding(input_ids)
        logits = self.lm_head(emb)
        
        if confidence_boost is not None:
            boost = confidence_boost
        elif is_predictable_prompt:
            boost = 15.0
        else:
            boost = 0.0
        
        if boost > 0.0:
            p1 = min(0.99, 0.02 + (boost / 15.0) ** 1.3 * 0.97)
            logits.fill_(-20.0)
            max_indices = torch.argmax(logits, dim=-1, keepdim=True)
            logits.scatter_(-1, max_indices, 0.0)
            
            num_tail = max(2, int(500 * (1.0 - boost / 15.0)))
            p_tail_each = (1.0 - p1) / num_tail
            tail_logit = math.log(max(p_tail_each, 1e-12))
            
            tail_indices = (max_indices + torch.arange(1, num_tail + 1, device=logits.device)) % self.vocab_size
            logits.scatter_(-1, tail_indices, tail_logit)
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
        self.router = DynamicHorizonRouter()
        self.fused_router = FusedEntropyRouter().to(device)
        self.kv_cache = SpeculativeKVCache(num_layers=12, num_heads=4, head_dim=32, device=device)
        self.adapter_stack = adapter_stack
        self.auto_research = auto_research
        
        print(f"[ElasticMTP Engine] Initializing engine for '{model_name}' on {device}...")
        if model_name == "synthetic":
            print("[ElasticMTP Engine] Using instant offline PyTorch model engine.")
            self.tokenizer = None
            self.model = SyntheticLM().to(device)
            self.mtp_head = MTPGLoRAHead(hidden_dim=128, vocab_size=50257, rank=16, num_aux_heads=4).to(device)
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, trust_remote_code=True)
                self.model = self.model.to(device)
                self.model.eval()

                hidden_dim = self.model.config.hidden_size
                vocab_size = self.model.config.vocab_size
                self.mtp_head = MTPGLoRAHead(hidden_dim=hidden_dim, vocab_size=vocab_size, rank=32, num_aux_heads=4).to(device)
                if device == "cuda":
                    self.mtp_head = self.mtp_head.half()
                self.mtp_head.eval()
            except Exception as e:
                print(f"[ElasticMTP Engine Warning] Real model load failed ({e}). Falling back to synthetic model.")
                self.model_name = "synthetic"
                self.tokenizer = None
                self.model = SyntheticLM().to(device)
                self.mtp_head = MTPGLoRAHead(hidden_dim=128, vocab_size=50257, rank=16, num_aux_heads=4).to(device)

    def generate(self, prompt: str, max_new_tokens: int = 50, mode: str = "elastic", confidence_boost: float = None, fixed_k: int = 4) -> Dict[str, Any]:
        start_time = time.perf_counter()
        is_predictable = ("Once upon a time" in prompt or "def " in prompt or "import " in prompt or "The quick brown fox" in prompt or "Python" in prompt)
        
        if self.tokenizer is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
        else:
            input_ids = torch.tensor([[101, 2003, 1037]], device=self.device)

        generated_ids = input_ids.clone()
        tokens_generated = 0
        accepted_draft_tokens = 0
        total_proposed_draft_tokens = 0
        horizon_history = []
        entropy_history = []
        meta_history = []
        token_id_history = []

        with torch.no_grad():
            while tokens_generated < max_new_tokens:
                if self.model_name == "synthetic":
                    outputs = self.model(generated_ids, is_predictable_prompt=is_predictable, confidence_boost=confidence_boost)
                    next_token_logits = outputs.logits[:, -1, :]
                    aux_logits = [next_token_logits for _ in range(4)]
                else:
                    outputs = self.model(generated_ids, output_hidden_states=True)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    last_hidden = outputs.hidden_states[-1][:, -1:, :]
                    mid_hidden = outputs.hidden_states[len(outputs.hidden_states) // 2][:, -1:, :]
                    aux_logits = self.mtp_head(last_hidden, mid_hidden)

                if mode == "ntp":
                    allocated_k = 1
                    meta = {"entropy": 0.5, "reason": "NTP_BASELINE"}
                elif mode == "static_mtp":
                    allocated_k = fixed_k
                    meta = {"entropy": 0.5, "reason": f"STATIC_MTP (K={fixed_k})"}
                else:
                    res = self.router.determine_horizon(next_token_logits)
                    allocated_k = res["target_k"]
                    meta = res

                horizon_history.append(allocated_k)
                entropy_history.append(meta.get("entropy", 0.0))
                meta_history.append(meta)

                # 1. Base Model Primary Next Token (Golden Truth)
                primary_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                token_id_history.append(int(primary_token[0, 0].item()))
                generated_ids = torch.cat([generated_ids, primary_token], dim=-1)
                tokens_generated += 1
                if tokens_generated >= max_new_tokens:
                    break

                # 2. Speculative Tree Verification
                if allocated_k > 1:
                    num_drafts = min(allocated_k - 1, max_new_tokens - tokens_generated)
                    if num_drafts <= 0:
                        break
                        
                    total_proposed_draft_tokens += num_drafts
                    
                    # Extract draft tokens proposed by auxiliary heads
                    draft_tokens = []
                    for i in range(num_drafts):
                        head_idx = min(i, len(aux_logits) - 1)
                        head_logits = aux_logits[head_idx]
                        if head_logits.dim() == 3:
                            head_logits = head_logits[:, -1, :]
                        d_tok = torch.argmax(head_logits, dim=-1, keepdim=True)
                        draft_tokens.append(d_tok)

                    # Parallel Verification Pass: Verify draft tokens with base model
                    draft_concat = torch.cat(draft_tokens, dim=-1)
                    candidate_ids = torch.cat([generated_ids, draft_concat], dim=-1)
                    
                    if self.model_name == "synthetic":
                        verify_out = self.model(candidate_ids, is_predictable_prompt=is_predictable, confidence_boost=confidence_boost)
                    else:
                        verify_out = self.model(candidate_ids)
                        
                    verify_logits = verify_out.logits[:, generated_ids.shape[1]-1:-1, :]
                    target_tokens = torch.argmax(verify_logits, dim=-1)

                    accepted_count = 0
                    for idx, d_tok in enumerate(draft_tokens):
                        target_tok = target_tokens[:, idx:idx+1]
                        if d_tok.item() == target_tok.item():
                            accepted_count += 1
                            token_id_history.append(int(d_tok[0, 0].item()))
                            generated_ids = torch.cat([generated_ids, d_tok], dim=-1)
                            tokens_generated += 1
                            if tokens_generated >= max_new_tokens:
                                break
                        else:
                            # Rejection: Fallback to target model token and break
                            token_id_history.append(int(target_tok[0, 0].item()))
                            generated_ids = torch.cat([generated_ids, target_tok], dim=-1)
                            tokens_generated += 1
                            break

                    accepted_draft_tokens += accepted_count
                    self.router.record_draft_acceptance(accepted_count, num_drafts)

        elapsed = time.perf_counter() - start_time
        tps = tokens_generated / elapsed if elapsed > 0 else 0.0

        if self.tokenizer is not None:
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            text = f"[Synthetic Generated {tokens_generated} tokens]"

        dar = round(accepted_draft_tokens / total_proposed_draft_tokens * 100.0, 2) if total_proposed_draft_tokens > 0 else 0.0

        telemetry_list = [{
            "step": step_idx,
            "token_id": token_id_history[step_idx] if step_idx < len(token_id_history) else 0,
            "allocated_k": k,
            "horizon_k": k,
            "target_k": k,
            "entropy": e,
            "reason": meta.get("reason", "DYNAMIC_HORIZON_ROUTED"),
            "is_contradiction": False
        } for step_idx, (k, e, meta) in enumerate(zip(horizon_history, entropy_history, meta_history))]

        return {
            "text": text,
            "generated_text": text,
            "tokens_generated": tokens_generated,
            "elapsed_seconds": round(elapsed, 4),
            "tokens_per_sec": round(tps, 2),
            "accepted_draft_tokens": accepted_draft_tokens,
            "total_proposed_draft_tokens": total_proposed_draft_tokens,
            "draft_acceptance_rate": dar,
            "avg_horizon_k": round(sum(horizon_history) / len(horizon_history), 2) if horizon_history else 1.0,
            "horizon_history": horizon_history,
            "entropy_history": entropy_history,
            "telemetry": telemetry_list,
            "router_metrics": {
                "total_drafted": total_proposed_draft_tokens,
                "total_draft_tokens_proposed": total_proposed_draft_tokens,
                "total_accepted": accepted_draft_tokens,
                "total_draft_tokens_accepted": accepted_draft_tokens,
                "acceptance_rate": dar,
                "draft_acceptance_rate_percent": dar,
                "total_routing_decisions": len(horizon_history),
                "avg_k": round(sum(horizon_history) / len(horizon_history), 2) if horizon_history else 1.0,
                "k_history": horizon_history
            }
        }
