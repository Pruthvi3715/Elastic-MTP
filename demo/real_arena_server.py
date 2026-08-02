"""
Real PyTorch Backend Web Server for Elastic-MTP Arena with Speculative Parallel Tree Engine.
Runs actual Qwen2.5-1.5B-Instruct model inference on NVIDIA GPU / CPU via PyTorch.
"""
import os
import sys
import time
import json
import torch
from http.server import HTTPServer, SimpleHTTPRequestHandler
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from elastic_mtp.engine.inference_engine import ElasticMTPInferenceEngine

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

CUDA_AVAILABLE = torch.cuda.is_available()
CURRENT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

print("=" * 80)
print(f"LOADING REAL ELASTIC-MTP SPECULATIVE ENGINE ({CURRENT_DEVICE.upper()})...")
print("=" * 80)

t0_load = time.perf_counter()

# Initialize Baseline Model & ElasticMTP Engine
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
torch_dtype = torch.float16 if CUDA_AVAILABLE else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch_dtype, trust_remote_code=True)
if CUDA_AVAILABLE:
    base_model = base_model.to("cuda")
base_model.eval()

mtp_engine = ElasticMTPInferenceEngine(model_name=MODEL_ID, device=CURRENT_DEVICE)

t1_load = time.perf_counter()
print(f"[OK] Loaded '{MODEL_ID}' on {CURRENT_DEVICE.upper()} in {(t1_load - t0_load):.2f}s!")

PORT = 8080
DIRECTORY = os.path.abspath(os.path.dirname(__file__))

class RealArenaHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path == "/api/device_status":
            res_data = {
                "cuda_available": CUDA_AVAILABLE,
                "current_device": CURRENT_DEVICE,
                "device_name": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU Execution (PyTorch)"
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res_data).encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        global CURRENT_DEVICE, base_model, mtp_engine
        
        if self.path == "/api/set_device":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            target_device = req.get("device", "cpu").lower()

            if target_device in ["gpu", "cuda"]:
                if CUDA_AVAILABLE:
                    CURRENT_DEVICE = "cuda"
                    base_model = base_model.to("cuda")
                    mtp_engine.device = "cuda"
                    mtp_engine.model = mtp_engine.model.to("cuda")
                    msg = "Switched to GPU (CUDA) Mode"
                else:
                    msg = "CUDA hardware unavailable. Remaining on CPU."
            else:
                CURRENT_DEVICE = "cpu"
                base_model = base_model.to("cpu")
                mtp_engine.device = "cpu"
                mtp_engine.model = mtp_engine.model.to("cpu")
                msg = "Switched to CPU Mode"

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "success",
                "message": msg,
                "current_device": CURRENT_DEVICE,
                "cuda_available": CUDA_AVAILABLE
            }).encode('utf-8'))
            return

        if self.path == "/api/generate":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            
            prompt = req.get("prompt", "what is name of first president")
            max_new_tokens = int(req.get("max_tokens", 35))
            requested_device = req.get("device", CURRENT_DEVICE).lower()

            device_str = "cuda" if (requested_device in ["gpu", "cuda"] and CUDA_AVAILABLE) else "cpu"

            print(f"\n[API Request] Prompt: '{prompt}' | Device: {device_str.upper()}")
            
            # 1. REAL MODEL A (RAW AUTOREGRESSIVE BASELINE)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device_str)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device_str)

            t0_raw = time.perf_counter()
            with torch.no_grad():
                raw_outputs = base_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            t1_raw = time.perf_counter()

            raw_full_text = tokenizer.decode(raw_outputs[0], skip_special_tokens=True)
            raw_gen_text = tokenizer.decode(raw_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            raw_gen_tokens = raw_outputs.shape[1] - input_ids.shape[1]
            raw_time_s = t1_raw - t0_raw
            raw_tps = raw_gen_tokens / raw_time_s if raw_time_s > 0 else 0
            raw_lat_ms = (raw_time_s / raw_gen_tokens * 1000.0) if raw_gen_tokens > 0 else 0

            # 2. REAL MODEL B (ELASTIC-MTP SPECULATIVE TREE ENGINE)
            res_mtp = mtp_engine.generate(prompt=prompt, max_new_tokens=max_new_tokens, mode="elastic")

            mtp_full_text = res_mtp["text"]
            mtp_gen_text = mtp_full_text[len(prompt):] if mtp_full_text.startswith(prompt) else mtp_full_text
            mtp_gen_tokens = res_mtp["tokens_generated"]
            mtp_time_s = res_mtp["elapsed_seconds"]
            mtp_tps = res_mtp["tokens_per_sec"]
            mtp_lat_ms = (mtp_time_s / mtp_gen_tokens * 1000.0) if mtp_gen_tokens > 0 else 0
            
            speedup = mtp_tps / raw_tps if raw_tps > 0 else 1.0

            response_data = {
                "prompt": prompt,
                "device_used": device_str.upper(),
                "cuda_available": CUDA_AVAILABLE,
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
                    "speedup": round(speedup, 2),
                    "acceptance_rate": res_mtp.get("draft_acceptance_rate", 100.0),
                    "avg_k": res_mtp.get("avg_horizon_k", 2.0)
                }
            }

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
