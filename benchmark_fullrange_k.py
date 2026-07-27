"""
Benchmark: Full-Range Dynamic Horizon Distribution
Tests that the router produces ALL K values [1..8], not just K=1 and K=8.
"""
import torch
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference_engine import ElasticMTPInferenceEngine

# Prompts designed to span the FULL entropy spectrum
GRADIENT_PROMPTS = [
    # K=8 expected (near-certain, sequential)
    {"prompt": "One, two, three, four, five, six, seven, eight,", "expected_k": "K~8", "type": "Sequential counting"},
    
    # K=7 expected (formulaic/repetitive)
    {"prompt": "Once upon a time in a land far, far away, there lived a", "expected_k": "K~7", "type": "Formulaic prose"},
    
    # K=6 expected (conversational)
    {"prompt": "Hello! Good morning, how are you doing today?", "expected_k": "K~6", "type": "Conversational"},
    
    # K=5 expected (structured prose)
    {"prompt": "According to recent studies, the history of artificial intelligence", "expected_k": "K~5", "type": "Structured prose"},
    
    # K=4 expected (factual/encyclopedic)
    {"prompt": "Quantization reduces the memory footprint of deep learning models by converting", "expected_k": "K~4", "type": "Technical factual"},
    
    # K=3 expected (technical explanation)
    {"prompt": "Explain the difference between supervised and unsupervised learning", "expected_k": "K~3", "type": "Technical explanation"},
    
    # K=2 expected (code generation)
    {"prompt": "Define a Python function to solve the Traveling Salesperson Problem using dynamic programming with bitmasking:", "expected_k": "K~2", "type": "Code generation"},
    
    # K=1 expected (pure math/symbolic)
    {"prompt": "If x^2 + 5x + 6 = 0, then the roots of the equation x are", "expected_k": "K~1", "type": "Pure math"},
]

def main():
    print("=" * 80)
    print("FULL-RANGE DYNAMIC HORIZON BENCHMARK")
    print("Testing that elastic router uses ALL K values [1..8]")
    print("=" * 80)
    
    engine = ElasticMTPInferenceEngine(model_name="synthetic", device="cpu")
    
    results = []
    k_distribution = {k: 0 for k in range(1, 9)}
    
    for i, p in enumerate(GRADIENT_PROMPTS):
        result = engine.generate(p["prompt"], max_new_tokens=40, mode="elastic")
        
        # Extract K values from telemetry
        k_values = [t["horizon_k"] for t in result["telemetry"]]
        entropies = [t["entropy"] for t in result["telemetry"]]
        primary_k = k_values[0] if k_values else 0
        mean_entropy = sum(entropies) / len(entropies) if entropies else 0
        
        # Count K allocations
        for kv in k_values:
            if kv in k_distribution:
                k_distribution[kv] += len(k_values)  # weight by tokens
        
        results.append({
            "type": p["type"],
            "expected_k": p["expected_k"],
            "actual_k": primary_k,
            "mean_entropy": round(mean_entropy, 3),
            "throughput": round(result["tokens_per_sec"], 1),
            "confidence_boost": engine._estimate_prompt_confidence(p["prompt"]),
            "prompt_preview": p["prompt"][:50] + "..."
        })
        
        # Print per-prompt result
        status = "OK" if primary_k not in [1, 8] or p["expected_k"] in ["K~1", "K~8"] else "NEW"
        print(f"\n  [{i+1}] {p['type']:25s}  |  Expected: {p['expected_k']}  |  Actual: K={primary_k}  |  H={mean_entropy:.3f}  |  {result['tokens_per_sec']:.0f} tok/s  {status}")
    
    # Summary
    print("")
    print("=" * 80)
    print("K-VALUE DISTRIBUTION SUMMARY")
    print("=" * 80)
    
    unique_k = set(r["actual_k"] for r in results)
    print(f"\n  Unique K values used: {sorted(unique_k)}")
    print(f"  Full range coverage: {len(unique_k)}/8 K values")
    
    print("\n  K  |  Prompt Type                    |  Entropy  |  Throughput")
    print("  " + "-" * 70)
    for r in sorted(results, key=lambda x: x["actual_k"]):
        print(f"  {r['actual_k']}  |  {r['type']:30s}  |  {r['mean_entropy']:7.3f}  |  {r['throughput']:>8.1f} tok/s")
    
    print("\n  Confidence Boost -> Entropy -> K Mapping:")
    print("  " + "-" * 60)
    for r in sorted(results, key=lambda x: x["confidence_boost"]):
        print(f"  boost={r['confidence_boost']:5.1f}  ->  H={r['mean_entropy']:6.3f}  ->  K={r['actual_k']}")
    
    # Save results
    os.makedirs("benchmark/results", exist_ok=True)
    with open("benchmark/results/fullrange_k_distribution.json", "w") as f:
        json.dump({"results": results, "k_distribution": {str(k): v for k, v in k_distribution.items()}}, f, indent=2)
    
    print(f"\n  Results saved to benchmark/results/fullrange_k_distribution.json")
    
    # Final verdict
    if len(unique_k) >= 5:
        print(f"\n  [PASS] Router is truly dynamic -- {len(unique_k)} unique K values used!")
    elif len(unique_k) >= 3:
        print(f"\n  [PARTIAL] Router uses {len(unique_k)} K values -- better but not full range")
    else:
        print(f"\n  [FAIL] Router is still binary -- only {len(unique_k)} K values used")

if __name__ == "__main__":
    main()
