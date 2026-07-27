"""
Quantization Noise Stress-Test Harness for Elastic-MTP.

Evaluates how simulated INT8 and INT4 AWQ quantization noise impacts:
1. Shannon Entropy H(P) variance and shift
2. Dynamic horizon allocation K
3. Draft Acceptance Rate (DAR) decay
"""
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from src.config import ElasticMTPConfig
from src.entropy_evaluator import EntropyEvaluator
from src.elastic_horizon_router import UncertaintyAwareHorizonFilter

PLOT_DIR = os.path.join(ElasticMTPConfig.BASE_DIR, "benchmark", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
PLOT_PATH = os.path.join(PLOT_DIR, "quantization_decay_curve.png")

def simulate_quantization_noise(logits: torch.Tensor, precision: str = "fp16") -> torch.Tensor:
    """
    Simulates precision reduction noise on logits.
    - fp16: baseline FP16 precision
    - int8: 8-bit uniform quantization with scale factor
    - int4: 4-bit zero-point AWQ-style quantization noise
    """
    if precision == "fp16":
        return logits.half().float() if logits.dtype != torch.half else logits.float()
        
    elif precision == "int8":
        # 8-bit quantization scale: [-128, 127]
        max_val = torch.max(torch.abs(logits))
        scale = max_val / 127.0 if max_val > 0 else 1.0
        q_logits = torch.round(logits / scale)
        q_logits = torch.clamp(q_logits, -128, 127)
        return q_logits * scale
        
    elif precision == "int4":
        # 4-bit quantization scale: [-8, 7]
        max_val = torch.max(torch.abs(logits))
        scale = max_val / 7.0 if max_val > 0 else 1.0
        q_logits = torch.round(logits / scale)
        q_logits = torch.clamp(q_logits, -8, 7)
        return q_logits * scale
        
    else:
        raise ValueError(f"Unknown precision: {precision}")

def run_quantization_stress_test(num_samples: int = 100) -> Dict[str, Any]:
    print("=" * 65)
    print("Quantization Noise Stress-Test Harness — Elastic-MTP")
    print("=" * 65)
    
    router = UncertaintyAwareHorizonFilter(
        tau_entropy=ElasticMTPConfig.ENTROPY_LOW_THRESHOLD,
        tau_divergence=ElasticMTPConfig.CONTRADICTION_THRESHOLD,
        max_k=ElasticMTPConfig.K_MAX
    )
    
    precisions = ["fp16", "int8", "int4"]
    results = {p: {"entropy_mean": 0.0, "accepted_k_mean": 0.0, "accuracy_retention": 0.0} for p in precisions}
    
    torch.manual_seed(42)
    vocab_size = 50257
    
    for precision in precisions:
        entropy_list = []
        horizon_list = []
        match_count = 0
        
        for _ in range(num_samples):
            # Generate primary logits
            clean_logits = torch.randn(1, vocab_size) * 3.0
            
            # Apply precision noise
            noisy_logits = simulate_quantization_noise(clean_logits, precision=precision)
            
            # Evaluate entropy and router horizon
            entropy = EntropyEvaluator.compute_shannon_entropy(noisy_logits).item()
            entropy_list.append(entropy)
            
            # Compute auxiliary logits with noise
            aux_logits = simulate_quantization_noise(clean_logits + torch.randn_like(clean_logits) * 0.1, precision=precision)
            
            target_k, meta = router.determine_horizon(noisy_logits, [aux_logits])
            horizon_list.append(target_k)
            
            clean_top = torch.argmax(clean_logits, dim=-1).item()
            noisy_top = torch.argmax(noisy_logits, dim=-1).item()
            if clean_top == noisy_top:
                match_count += 1
                
        results[precision]["entropy_mean"] = float(np.mean(entropy_list))
        results[precision]["accepted_k_mean"] = float(np.mean(horizon_list))
        results[precision]["accuracy_retention"] = (match_count / num_samples) * 100.0
        
        print(f"[{precision.upper()}] Mean H(P): {results[precision]['entropy_mean']:.3f} | Mean Horizon K: {results[precision]['accepted_k_mean']:.2f} | Accuracy Retention: {results[precision]['accuracy_retention']:.1f}%")

    print("\n[Quantization] Generating Decay Curve Plot...")
    plot_quantization_decay(results)
    
    return results

def plot_quantization_decay(results: Dict[str, Any]):
    precisions = ["FP16 Baseline", "INT8 Uniform", "INT4 AWQ"]
    mean_k = [results["fp16"]["accepted_k_mean"], results["int8"]["accepted_k_mean"], results["int4"]["accepted_k_mean"]]
    retention = [results["fp16"]["accuracy_retention"], results["int8"]["accuracy_retention"], results["int4"]["accuracy_retention"]]
    
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    
    color = "#3182CE"
    ax1.set_xlabel("Quantization Precision", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Speculative Horizon (K)", color=color, fontsize=11, fontweight="bold")
    bars = ax1.bar(precisions, mean_k, color=color, alpha=0.75, width=0.4, label="Mean Horizon K")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 8.5)
    
    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"K={height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

    ax2 = ax1.twinx()
    color = "#E53E3E"
    ax2.set_ylabel("Top-1 Accuracy Retention (%)", color=color, fontsize=11, fontweight="bold")
    ax2.plot(precisions, retention, color=color, marker="o", linewidth=2.5, markersize=8, label="Accuracy Retention")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(50, 105)
    
    plt.title("Quantization Precision vs Speculative Horizon & Accuracy Retention", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"[Quantization] Chart saved to: {PLOT_PATH}")

if __name__ == "__main__":
    run_quantization_stress_test()
