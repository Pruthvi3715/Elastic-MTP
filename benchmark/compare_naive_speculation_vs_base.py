"""
Elastic-MTP vs Naive Unverified Speculation Comparison Script (Real GPT-2 Weights)
==================================================================================
Demonstrates what happens when you do "naive speculation" (predicting K tokens
greedily without base model verification) vs Base Model Ground Truth vs Elastic-MTP.
"""

import os
import sys
import time
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference_engine import ElasticMTPInferenceEngine


def run_naive_vs_verified_comparison():
    print("=" * 85)
    print("NAIVE UNVERIFIED SPECULATION vs BASE MODEL GROUND TRUTH vs ELASTIC-MTP")
    print("Backbone: Real GPT-2 Neural Transformer Weights")
    print("=" * 85)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Initializing real GPT-2 model engine...")
    engine = ElasticMTPInferenceEngine(model_name="gpt2", device=device)

    test_prompts = [
        {"category": "Formulaic Story", "prompt": "Once upon a time in a faraway land,"},
        {"category": "General Knowledge", "prompt": "The capital of France is Paris and its main river is"},
        {"category": "Dialogue", "prompt": "Hello, how are you feeling today?"},
        {"category": "Python Code", "prompt": "def calculate_factorial(n):\n    if n <= 1:"},
        {"category": "Math Reasoning", "prompt": "To solve 3x + 12 = 45, we subtract 12 from both sides:"},
    ]

    print("\nComparing generated sentences across 3 decoding strategies...\n")

    for idx, item in enumerate(test_prompts, 1):
        prompt = item["prompt"]
        category = item["category"]
        print(f"[{idx}/5] Category: {category}")
        print(f"     Prompt: \"{prompt}\"")

        # 1. Base Model Ground Truth (NTP)
        res_base = engine.generate(prompt=prompt, max_new_tokens=15, mode="ntp")
        base_text = res_base["generated_text"].replace("\n", " ")

        # 2. Static MTP (Simulated draft without safety fallback)
        res_naive = engine.generate(prompt=prompt, max_new_tokens=15, mode="static_mtp")
        naive_text = res_naive["generated_text"].replace("\n", " ")

        # 3. Elastic-MTP (Verified Speculation with Dynamic Router)
        res_elastic = engine.generate(prompt=prompt, max_new_tokens=15, mode="elastic")
        elastic_text = res_elastic["generated_text"].replace("\n", " ")

        # Metrics
        base_words = base_text.split()
        naive_words = naive_text.split()
        elastic_words = elastic_text.split()

        naive_matches = sum(1 for w1, w2 in zip(base_words, naive_words) if w1 == w2)
        naive_match_pct = (naive_matches / len(base_words) * 100.0) if base_words else 100.0

        elastic_matches = sum(1 for w1, w2 in zip(base_words, elastic_words) if w1 == w2)
        elastic_match_pct = (elastic_matches / len(base_words) * 100.0) if base_words else 100.0

        print("-" * 85)
        print(f"  1. Base Ground Truth (NTP)   : \"{base_text}\"")
        print(f"  2. Static Speculation (K=4)  : \"{naive_text}\" (Match: {naive_match_pct:.1f}%)")
        print(f"  3. Elastic-MTP (Verified)    : \"{elastic_text}\" (Match: {elastic_match_pct:.1f}%)")

        if naive_match_pct < 100.0:
            print("  [!] STATIC/NAIVE SPECULATION DIVERGED! (Error cascade without entropy routing)")
        else:
            print("  [OK] Matched ground truth (Predictable sequence)")
        print("-" * 85 + "\n")


if __name__ == "__main__":
    run_naive_vs_verified_comparison()
