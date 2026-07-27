"""
AutoResearch Benchmark Evaluator (Immutable Harness).

Computes standardized validation scores for Elastic-MTP experiments.
"""
import torch
import numpy as np
from typing import Dict, Any

TEST_PROMPTS = [
    "One, two, three, four, five, six, seven, eight,",
    "Multi-Token Prediction accelerates LLM inference by predicting",
    "Define a Python function for binary search tree traversal:",
    "If 2x + 10 = 20, the value of x is equal to"
]

def evaluate_config(router_cls, model_engine_cls) -> Dict[str, Any]:
    """
    Runs standardized evaluation pass across benchmark prompts.
    """
    engine = model_engine_cls()
    throughputs = []
    entropies = []
    k_distributions = []
    contradictions = 0
    
    for prompt in TEST_PROMPTS:
        res = engine.generate(prompt=prompt, max_new_tokens=30, mode="elastic")
        throughputs.append(res["tokens_per_sec"])
        
        telemetry = res["telemetry"]
        entropies.extend([t["entropy"] for t in telemetry])
        k_distributions.extend([t["horizon_k"] for t in telemetry])
        contradictions += sum(1 for t in telemetry if t.get("is_contradiction", False))

    mean_throughput = float(np.mean(throughputs))
    mean_entropy = float(np.mean(entropies))
    acceptance_rate = (sum(1 for k in k_distributions if k > 1) / len(k_distributions)) * 100.0 if k_distributions else 0.0
    
    # Composite Score formula
    composite_score = mean_throughput * (1.0 + (acceptance_rate / 100.0)) - (10.0 * contradictions)
    
    return {
        "score": composite_score,
        "mean_throughput": mean_throughput,
        "acceptance_rate": acceptance_rate,
        "mean_entropy": mean_entropy,
        "contradiction_count": contradictions,
        "horizon_counts": {str(k_val): k_distributions.count(k_val) for k_val in [1, 2, 4, 8]}
    }
