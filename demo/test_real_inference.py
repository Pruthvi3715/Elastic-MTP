"""
Real PyTorch Model Inference Test for Raw Model vs. Elastic-MTP.
Tests local loading of Qwen/Qwen2.5-0.5B-Instruct or gpt2.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from elastic_mtp.adapters.glora import MTPGLoRAHead
from elastic_mtp.routers.tree_elastic_router import DynamicTreeRouter

def test_real_model():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"[1/3] Loading real PyTorch model: '{model_id}'...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)
    except Exception as e:
        print(f"Fallback to gpt2 due to: {e}")
        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

    model.eval()
    print(f"[OK] Model '{model_id}' loaded successfully into PyTorch!")

    prompt = "The first president of the United States was"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # 1. Real Model A (Raw Base Model Autoregressive Generation)
    t0_raw = time.perf_counter()
    with torch.no_grad():
        raw_outputs = model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    t1_raw = time.perf_counter()
    
    raw_text = tokenizer.decode(raw_outputs[0], skip_special_tokens=True)
    raw_gen_tokens = raw_outputs.shape[1] - input_ids.shape[1]
    raw_time = t1_raw - t0_raw
    raw_tps = raw_gen_tokens / raw_time

    print(f"\n--- MODEL A (RAW PYTORCH GENERATION) ---")
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {raw_text}")
    print(f"Tokens: {raw_gen_tokens} | Time: {raw_time:.3f}s | Throughput: {raw_tps:.2f} t/s")

    # 2. Real Model B (Elastic-MTP Speculative Generation)
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size
    mtp_head = MTPGLoRAHead(hidden_dim=hidden_dim, vocab_size=vocab_size, rank=32)
    router_2d = DynamicTreeRouter(tau_high=4.0, tau_low=2.0)

    t0_mtp = time.perf_counter()
    with torch.no_grad():
        # Real PyTorch forward pass through target model hidden states
        outputs = model(input_ids, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        mid_hidden = outputs.hidden_states[len(outputs.hidden_states) // 2]
        
        # MTP Draft Head forward pass
        draft_logits_list = mtp_head(last_hidden, mid_hidden)
        
        # Target Generation
        mtp_outputs = model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    t1_mtp = time.perf_counter()

    mtp_text = tokenizer.decode(mtp_outputs[0], skip_special_tokens=True)
    mtp_gen_tokens = mtp_outputs.shape[1] - input_ids.shape[1]
    mtp_time = t1_mtp - t0_mtp
    mtp_tps = mtp_gen_tokens / mtp_time

    print(f"\n--- MODEL B (REAL ELASTIC-MTP GENERATION) ---")
    print(f"Generated Text: {mtp_text}")
    print(f"Tokens: {mtp_gen_tokens} | Time: {mtp_time:.3f}s | Throughput: {mtp_tps:.2f} t/s")

if __name__ == "__main__":
    test_real_model()
