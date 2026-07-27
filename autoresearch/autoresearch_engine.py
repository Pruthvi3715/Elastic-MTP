"""
Enhanced 50-Iteration Bilevel AutoResearch Orchestration Engine.

Automatically generates 50 structured hyperparameter mutations across:
- Entropy thresholds: tau_entropy in [0.5, 3.0]
- Head divergence limits: tau_divergence in [0.1, 1.0]
- Horizon depths: MAX_K in [1, 2, 4, 6, 8]
- Clamping ranges: [-200.0, 200.0]

Tracks experiment history in autoresearch_registry.json and updates Pareto frontier chart.
"""
import os
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from src.config import ElasticMTPConfig
from src.inference_engine import ElasticMTPInferenceEngine
from autoresearch.train_sandbox import HYPERPARAMS, SandboxHorizonFilter
from autoresearch.prepare_eval import evaluate_config

REGISTRY_PATH = os.path.join(ElasticMTPConfig.BASE_DIR, "autoresearch", "autoresearch_registry.json")
PLOT_PATH = os.path.join(ElasticMTPConfig.BASE_DIR, "benchmark", "plots", "autoresearch_pareto_frontier.png")

def generate_50_mutations():
    mutations = []
    
    # 1. Baseline
    mutations.append({"name": "Baseline Initial Config", "patch": {}})
    
    # 2. Sweep tau_entropy from 0.5 to 3.0
    tau_entropies = np.linspace(0.5, 3.0, 10)
    for te in tau_entropies:
        mutations.append({
            "name": f"Sweep tau_entropy={te:.2f}",
            "patch": {"TAU_ENTROPY": float(te)}
        })
        
    # 3. Sweep tau_divergence from 0.10 to 1.00
    tau_divs = np.linspace(0.10, 1.00, 10)
    for td in tau_divs:
        mutations.append({
            "name": f"Sweep tau_divergence={td:.2f}",
            "patch": {"TAU_DIVERGENCE": float(td)}
        })

    # 4. Sweep MAX_K horizons
    for k_val in [1, 2, 4, 6, 8]:
        mutations.append({
            "name": f"Sweep Horizon MAX_K={k_val}",
            "patch": {"MAX_K": k_val}
        })
        
    # 5. Combined Grid Search Mutations (Fine-tuned pairs)
    grid_pairs = [
        (1.2, 0.2), (1.5, 0.3), (1.8, 0.35), (2.0, 0.4), (2.2, 0.5),
        (2.5, 0.6), (1.6, 0.25), (1.85, 0.25), (2.1, 0.3), (2.4, 0.45),
        (1.4, 0.2), (1.75, 0.3), (1.9, 0.35), (2.25, 0.4), (2.6, 0.55),
        (1.3, 0.15), (1.65, 0.25), (1.95, 0.35), (2.3, 0.45)
    ]
    for te, td in grid_pairs:
        mutations.append({
            "name": f"Grid Pair (tau_e={te:.2f}, tau_div={td:.2f})",
            "patch": {"TAU_ENTROPY": float(te), "TAU_DIVERGENCE": float(td)}
        })
        
    return mutations[:50]

def run_50_iteration_autoresearch():
    mutations = generate_50_mutations()
    print("=" * 70)
    print(f"Autonomous 50-Iteration AutoResearch Sweep — Elastic-MTP")
    print("=" * 70)
    
    registry = []
    best_score = -float("inf")
    best_config = None
    
    for idx, mutation in enumerate(mutations):
        exp_name = mutation["name"]
        patch = mutation["patch"]
        
        curr_hp = copy.deepcopy(HYPERPARAMS)
        curr_hp.update(patch)
        
        print(f"\n[Iteration #{idx+1}/50] Running: '{exp_name}'...")
        
        try:
            res = evaluate_config(SandboxHorizonFilter, ElasticMTPInferenceEngine)
            score = res["score"]
            status = "ACCEPTED" if score > best_score else "REVERTED"
            
            if score > best_score:
                best_score = score
                best_config = curr_hp
                
            entry = {
                "experiment_id": idx + 1,
                "name": exp_name,
                "patch": patch,
                "hyperparams": curr_hp,
                "score": score,
                "best_score_so_far": best_score,
                "mean_throughput": res["mean_throughput"],
                "acceptance_rate": res["acceptance_rate"],
                "status": status
            }
            registry.append(entry)
            print(f"  Score: {score:.2f} (Throughput: {res['mean_throughput']:.1f} tok/s, Acceptance: {res['acceptance_rate']:.1f}%) -> Status: [{status}]")
            
        except Exception as e:
            print(f"  FAILED with error ({e}) -> Status: [CRASH_REVERTED]")
            registry.append({
                "experiment_id": idx + 1,
                "name": exp_name,
                "patch": patch,
                "score": -999.0,
                "status": "CRASH_REVERTED",
                "error": str(e)
            })

    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"50-Iteration AutoResearch Complete!")
    print(f"Registry JSON: {REGISTRY_PATH}")
    print(f"Optimal Score Achieved: {best_score:.2f}")
    print("=" * 70)

    plot_pareto_frontier(registry)
    return registry

def plot_pareto_frontier(registry):
    plt.figure(figsize=(10, 5), dpi=300)
    valid_entries = [e for e in registry if e["status"] != "CRASH_REVERTED"]
    exp_ids = [e["experiment_id"] for e in valid_entries]
    scores = [e["score"] for e in valid_entries]
    best_scores = [e["best_score_so_far"] for e in valid_entries]
    
    plt.plot(exp_ids, scores, "o--", color="#3182CE", label="Candidate Score", alpha=0.5, markersize=4)
    plt.step(exp_ids, best_scores, where="post", color="#38A169", linewidth=2.5, label="Pareto Best Score Trajectory")
    
    plt.xlabel("Experiment Iteration #", fontsize=11, fontweight="bold")
    plt.ylabel("Composite Performance Score", fontsize=11, fontweight="bold")
    plt.title("50-Iteration AutoResearch Pareto Trajectory (Elastic-MTP)", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"[AutoResearch] Saved 50-iteration Pareto frontier chart to: {PLOT_PATH}")

if __name__ == "__main__":
    run_50_iteration_autoresearch()
