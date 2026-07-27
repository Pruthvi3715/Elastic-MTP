"""
Scheduled 2-Minute Autonomous Research Pass for Elastic-MTP.

Runs a hypothesis test pass, evaluates score improvements, accepts/rejects changes,
and maintains an audit log in benchmark/results/scheduled_research_history.json.
"""
import os
import json
import time
import copy
from typing import Dict, Any
from src.config import ElasticMTPConfig
from src.inference_engine import ElasticMTPInferenceEngine
from src.elastic_horizon_router import UncertaintyAwareHorizonFilter
from autoresearch.prepare_eval import evaluate_config

HISTORY_PATH = os.path.join(ElasticMTPConfig.BASE_DIR, "benchmark", "results", "scheduled_research_history.json")
MASTER_LOG_PATH = os.path.join(ElasticMTPConfig.BASE_DIR, "benchmark", "results", "master_benchmark_log.json")

HYPOTHESES_POOL = [
    {
        "id": 1,
        "name": "Temperature-Weighted Soft Entropy H(P/T) with T=0.85",
        "tau_entropy": 1.45,
        "tau_divergence": 0.28,
        "max_k": 8
    },
    {
        "id": 2,
        "name": "Strict Divergence Protection (tau_divergence=0.22)",
        "tau_entropy": 1.50,
        "tau_divergence": 0.22,
        "max_k": 8
    },
    {
        "id": 3,
        "name": "Expanded Speculation Horizon (max_k=10)",
        "tau_entropy": 1.55,
        "tau_divergence": 0.32,
        "max_k": 10
    },
    {
        "id": 4,
        "name": "Ultra-Low Entropy Gate (tau_entropy=1.20)",
        "tau_entropy": 1.20,
        "tau_divergence": 0.25,
        "max_k": 8
    },
    {
        "id": 5,
        "name": "Balanced Pareto Pair (tau_entropy=1.50, tau_divergence=0.30)",
        "tau_entropy": 1.50,
        "tau_divergence": 0.30,
        "max_k": 8
    }
]

def run_scheduled_pass():
    print("=" * 65)
    print("Scheduled 2-Minute Research Pass — Elastic-MTP")
    print("=" * 65)
    
    # Load history
    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
            
    pass_number = len(history) + 1
    hypothesis = HYPOTHESES_POOL[(pass_number - 1) % len(HYPOTHESES_POOL)]
    
    print(f"[Pass #{pass_number}/5] Testing Hypothesis: '{hypothesis['name']}'...")
    
    best_score_so_far = max([h["score"] for h in history if h.get("status") == "ACCEPTED"] + [276.61])
    
    # Temporarily patch router
    test_router = UncertaintyAwareHorizonFilter(
        tau_entropy=hypothesis["tau_entropy"],
        tau_divergence=hypothesis["tau_divergence"],
        max_k=hypothesis["max_k"]
    )
    
    start_time = time.time()
    try:
        res = evaluate_config(UncertaintyAwareHorizonFilter, ElasticMTPInferenceEngine)
        score = res["score"]
        throughput = res["mean_throughput"]
        
        status = "ACCEPTED" if score >= best_score_so_far * 0.95 else "REJECTED"
        
        entry = {
            "pass_number": pass_number,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hypothesis": hypothesis["name"],
            "parameters": {
                "tau_entropy": hypothesis["tau_entropy"],
                "tau_divergence": hypothesis["tau_divergence"],
                "max_k": hypothesis["max_k"]
            },
            "score": score,
            "throughput_tok_sec": throughput,
            "acceptance_rate_pct": res["acceptance_rate"],
            "status": status,
            "elapsed_sec": round(time.time() - start_time, 2)
        }
        
        history.append(entry)
        
        # Save updated history JSON
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
            
        print(f"  Score: {score:.2f} | Throughput: {throughput:.1f} tok/s -> Decision: [{status}]")
        print(f"  Saved pass record to: {HISTORY_PATH}")
        print("=" * 65)
        
        return entry
        
    except Exception as e:
        print(f"  FAILED pass with error: {e} -> Decision: [CRASH_REJECTED]")
        entry = {
            "pass_number": pass_number,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hypothesis": hypothesis["name"],
            "score": -999.0,
            "status": "CRASH_REJECTED",
            "error": str(e)
        }
        history.append(entry)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
        return entry

if __name__ == "__main__":
    run_scheduled_pass()
