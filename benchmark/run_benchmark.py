"""
Benchmark Runner for Elastic-MTP Research Evaluation.

Compares:
1. Standard Next-Token Prediction (NTP, k=1)
2. Static Multi-Token Prediction (Static MTP, k=4)
3. Elastic-MTP (Dynamic Horizon k in {1, 4, 8} routed via Entropy)

Saves detailed JSON benchmark results to benchmark/results/.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any
from src.config import ElasticMTPConfig
from src.inference_engine import ElasticMTPInferenceEngine

PROMPTS = [
    # Category 1: High Predictability (Near-Certain, Sequential)
    "One, two, three, four, five, six, seven, eight,",
    "Once upon a time in a land far, far away, there lived a",
    
    # Category 2: Moderate-High Predictability (Conversational / Structured)
    "Hello! Good morning, how are you doing today?",
    "According to recent studies, the history of artificial intelligence",
    
    # Category 3: Medium Complexity (Technical Prose / Factual)
    "Quantization reduces the memory footprint of deep learning models by converting",
    "Explain the difference between supervised and unsupervised learning",
    
    # Category 4: High Uncertainty / Reasoning / Code Logic / Math
    "Define a Python function to solve the Traveling Salesperson Problem using dynamic programming with bitmasking:",
    "If x^2 + 5x + 6 = 0, then the roots of the equation x are",
]

def run_benchmark_suite(max_new_tokens: int = 40):
    print("=" * 60)
    print("Elastic-MTP Research Benchmark Suite")
    print("=" * 60)
    
    engine = ElasticMTPInferenceEngine()
    results = {
        "ntp": [],
        "static_mtp": [],
        "elastic": []
    }
    
    modes = [("ntp", 1), ("static_mtp", 4), ("elastic", None)]
    
    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt [{prompt_idx + 1}/{len(PROMPTS)}]: '{prompt[:40]}...' ---")
        
        for mode_name, fixed_k in modes:
            print(f"  Running mode: {mode_name.upper()}...", end="", flush=True)
            res = engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                mode=mode_name,
                fixed_k=fixed_k or 4
            )
            
            # Aggregate stats
            entropies = [t["entropy"] for t in res["telemetry"]]
            horizons = [t["horizon_k"] for t in res["telemetry"]]
            
            res_summary = {
                "prompt_id": prompt_idx,
                "prompt": prompt,
                "mode": mode_name,
                "tokens_generated": res["tokens_generated"],
                "elapsed_sec": res["elapsed_sec"],
                "tokens_per_sec": res["tokens_per_sec"],
                "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
                "std_entropy": float(np.std(entropies)) if entropies else 0.0,
                "horizon_counts": {str(k_val): horizons.count(k_val) for k_val in range(1, 9)},
                "generated_sample": res["generated_text"][:100] + "...",
                # Include new research metrics for Elastic-MTP
                "router_metrics": res.get("router_metrics", {})
            }
            
            results[mode_name].append(res_summary)
            
            # Print enhanced metrics for elastic mode
            if mode_name == "elastic" and res.get("router_metrics"):
                metrics = res["router_metrics"]
                print(f" Done ({res['tokens_per_sec']:.2f} tok/s, Mean H(P)={res_summary['mean_entropy']:.2f}, DAR={metrics.get('draft_acceptance_rate_percent', 'N/A')}%, Contradictions={metrics.get('contradiction_rate_percent', 'N/A')}%)")
            else:
                print(f" Done ({res['tokens_per_sec']:.2f} tok/s, Mean H(P)={res_summary['mean_entropy']:.2f})")

    # Save to JSON
    output_file = os.path.join(ElasticMTPConfig.RESULTS_DIR, "benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"Benchmark run complete! Results saved to: {output_file}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    run_benchmark_suite()
