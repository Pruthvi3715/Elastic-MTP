"""
Plotting & Data Visualization Suite for Elastic-MTP Benchmark.

Generates high-resolution publication charts:
1. Throughput Comparison (Tokens / Sec) across modes
2. Horizon Distribution Histogram (k=1 vs k=4 vs k=8)
3. Token-Level Entropy H(P) vs Horizon Allocation
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from src.config import ElasticMTPConfig

def load_results():
    results_path = os.path.join(ElasticMTPConfig.RESULTS_DIR, "benchmark_results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results found at {results_path}. Run run_benchmark.py first.")
    with open(results_path, "r") as f:
        return json.load(f)

def generate_plots():
    results = load_results()
    plots_dir = ElasticMTPConfig.PLOTS_DIR
    
    # 1. Throughput Comparison Bar Chart
    plt.figure(figsize=(9, 5), dpi=300)
    modes = ["ntp", "static_mtp", "elastic"]
    labels = ["Standard NTP (k=1)", "Static MTP (k=4)", "Elastic-MTP (Dynamic k)"]
    colors = ["#4A5568", "#3182CE", "#38A169"]
    
    avg_throughputs = [
        np.mean([item["tokens_per_sec"] for item in results[mode]]) for mode in modes
    ]
    
    bars = plt.bar(labels, avg_throughputs, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    plt.ylabel("Generation Throughput (Tokens / Sec)", fontsize=12, fontweight="bold")
    plt.title("Elastic-MTP vs Standard Baseline Inference Speed", fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f"{height:.1f} t/s", ha='center', va='bottom', fontsize=11, fontweight="bold")
                 
    plt.tight_layout()
    chart1_path = os.path.join(plots_dir, "throughput_comparison.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"[Plots] Saved throughput comparison chart to: {chart1_path}")

    # 2. Elastic-MTP Horizon Distribution Histogram (Full K=1 to K=8 spectrum)
    plt.figure(figsize=(10, 5), dpi=300)
    elastic_items = results["elastic"]
    
    k_counts = [sum(item["horizon_counts"].get(str(k), 0) for item in elastic_items) for k in range(1, 9)]
    k_labels = [f"K={k}" for k in range(1, 9)]
    
    # Custom colormap from Red (K=1, cautious) to Teal (K=8, aggressive)
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 8))
    
    bars2 = plt.bar(k_labels, k_counts, color=colors, width=0.6, edgecolor="black", linewidth=1.2)
    plt.ylabel("Tokens Allocated", fontsize=12, fontweight="bold")
    plt.xlabel("Speculative Prediction Horizon (K)", fontsize=12, fontweight="bold")
    plt.title("Elastic-MTP Dynamic Horizon Allocation Spectrum (K=1..8)", fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f"{int(height)}", ha='center', va='bottom', fontsize=10, fontweight="bold")
                 
    plt.tight_layout()
    chart2_path = os.path.join(plots_dir, "horizon_distribution.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"[Plots] Saved horizon distribution chart to: {chart2_path}")

if __name__ == "__main__":
    generate_plots()
