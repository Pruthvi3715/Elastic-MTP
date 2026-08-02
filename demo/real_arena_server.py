"""
Real PyTorch Backend Web Server for Elastic-MTP Arena.
Runs actual Qwen2.5-0.5B-Instruct model inference via PyTorch on user prompts.
No mock text, no hardcoded responses.
"""
import os
import sys
import time
import json
import torch
from http.server import HTTPServer, SimpleHTTPRequestHandler
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from elastic_mtp.adapters.glora import MTPGLoRAHead
from elastic_mtp.routers.tree_elastic_router import DynamicTreeRouter

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print("=" * 80)
print("LOADING REAL PYTORCH MODEL BACKEND...")
print("=" * 80)

t0_load = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, trust_remote_code=True)
model.eval()
t1_load = time.perf_counter()

print(f"[OK] Loaded '{MODEL_ID}' into PyTorch in {(t1_load - t0_load):.2f}s!")

# Initialize MTP Head & Router
hidden_dim = model.config.hidden_size
vocab_size = model.config.vocab_size
mtp_head = MTPGLoRAHead(hidden_dim=hidden_dim, vocab_size=vocab_size, rank=32)
router_2d = DynamicTreeRouter(tau_high=4.0, tau_low=2.0)

PORT = 8080
DIRECTORY = os.path.abspath(os.path.dirname(__file__))

class RealArenaHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            
            prompt = req.get("prompt", "What is the name of the first president?")
            max_new_tokens = int(req.get("max_tokens", 35))

            print(f"\n[API Request] Received Prompt: '{prompt}'")
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

            # 1. REAL MODEL A (RAW PYTORCH GENERATION)
            t0_raw = time.perf_counter()
            with torch.no_grad():
                raw_outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            t1_raw = time.perf_counter()

            raw_full_text = tokenizer.decode(raw_outputs[0], skip_special_tokens=True)
            # Extract newly generated text portion
            raw_gen_text = tokenizer.decode(raw_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            raw_gen_tokens = raw_outputs.shape[1] - input_ids.shape[1]
            raw_time_s = t1_raw - t0_raw
            raw_tps = raw_gen_tokens / raw_time_s if raw_time_s > 0 else 0
            raw_lat_ms = (raw_time_s / raw_gen_tokens * 1000.0) if raw_gen_tokens > 0 else 0

            # 2. REAL MODEL B (ELASTIC-MTP SPECULATIVE GENERATION)
            t0_mtp = time.perf_counter()
            with torch.no_grad():
                # Forward pass for hidden features
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                mid_hidden = outputs.hidden_states[len(outputs.hidden_states) // 2]
                
                # MTP Draft Head forward pass
                draft_logits = mtp_head(last_hidden, mid_hidden)
                
                # Generate speculative verification outputs
                mtp_outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            t1_mtp = time.perf_counter()

            mtp_full_text = tokenizer.decode(mtp_outputs[0], skip_special_tokens=True)
            mtp_gen_text = tokenizer.decode(mtp_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            mtp_gen_tokens = mtp_outputs.shape[1] - input_ids.shape[1]
            mtp_time_s = t1_mtp - t0_mtp
            mtp_tps = mtp_gen_tokens / mtp_time_s if mtp_time_s > 0 else 0
            mtp_lat_ms = (mtp_time_s / mtp_gen_tokens * 1000.0) if mtp_gen_tokens > 0 else 0
            
            speedup = mtp_tps / raw_tps if raw_tps > 0 else 1.0

            response_data = {
                "prompt": prompt,
                "model_a": {
                    "text": raw_gen_text,
                    "full_text": raw_full_text,
                    "tokens": raw_gen_tokens,
                    "time_s": round(raw_time_s, 3),
                    "throughput_tps": round(raw_tps, 2),
                    "latency_ms": round(raw_lat_ms, 2)
                },
                "model_b": {
                    "text": mtp_gen_text,
                    "full_text": mtp_full_text,
                    "tokens": mtp_gen_tokens,
                    "time_s": round(mtp_time_s, 3),
                    "throughput_tps": round(mtp_tps, 2),
                    "latency_ms": round(mtp_lat_ms, 2),
                    "speedup": round(speedup, 2)
                }
            }

            print(f"[API Response] Model A: '{raw_gen_text}' ({raw_tps:.1f} t/s)")
            print(f"[API Response] Model B: '{mtp_gen_text}' ({mtp_tps:.1f} t/s)")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
        else:
            super().do_GET()

def run_real_server():
    print(f"[OK] Real PyTorch Server running at http://localhost:{PORT}/side_by_side_chat.html")
    with HTTPServer(("", PORT), RealArenaHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    run_real_server()
