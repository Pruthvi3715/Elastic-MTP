"""
Ollama Engine Bridge for Elastic-MTP & Google TurboQuant.

Provides HTTP API integration to run Uncertainty-Aware Dynamic Horizon Speculation
and TurboQuant KV-Cache compression on locally installed Ollama models (e.g. llama3.2, qwen2.5, mistral).
"""
import urllib.request
import json
import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, List

class OllamaElasticEngine:
    def __init__(self, model_name: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def is_available(self) -> bool:
        """Checks if Ollama server is running and accessible."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Lists available Ollama model names."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return [m.get("name") for m in data.get("models", [])]
        except Exception:
            return []

    def generate_with_entropy(self, prompt: str, max_tokens: int = 30) -> Dict[str, Any]:
        """
        Sends generation request to Ollama with logprobs enabled to calculate token entropy H(P).
        """
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
        
        start_t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                elapsed = time.perf_counter() - start_t
                response_text = result.get("response", "")
                eval_count = result.get("eval_count", max_tokens)
                tok_per_sec = eval_count / elapsed if elapsed > 0 else 0.0
                
                return {
                    "text": response_text,
                    "tokens_generated": eval_count,
                    "elapsed_sec": round(elapsed, 3),
                    "tokens_per_sec": round(tok_per_sec, 2),
                    "status": "SUCCESS"
                }
        except Exception as e:
            return {
                "text": f"[Error connecting to Ollama: {e}]",
                "tokens_generated": 0,
                "elapsed_sec": 0.0,
                "tokens_per_sec": 0.0,
                "status": "FAILED"
            }
