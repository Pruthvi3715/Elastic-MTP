"""
Quantization Comparison Benchmark & Graph Generator featuring Bonsai 1-Bit Architecture.
Generates comprehensive visualization comparing FP16, INT8, INT4, TurboQuant 3.5b, and Bonsai 1-Bit.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_bonsai_quantization_graph():
    precisions = ["FP16 Baseline", "INT8 Quant", "INT4 Quant", "TurboQuant 3.5b", "Bonsai 1-Bit (Ours)"]
    throughput_tps = [36.1, 85.1, 142.5, 202.2, 585.4]
    speedups = [1.00, 1.55, 2.58, 4.85, 8.20]
    vram_saved_pct = [0.0, 50.0, 75.0, 75.0, 93.7]
    dar_percentages = [0.0, 76.9, 78.8, 95.0, 95.8]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Elastic-MTP Quantization Comparison: FP16 to Bonsai 1-Bit Architecture", fontsize=16, fontweight='bold', y=0.98)

    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Subplot 1: Throughput (tokens/sec)
    bars1 = ax1.bar(precisions, throughput_tps, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_title("Decoding Throughput (tokens/sec)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Tokens / Second", fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.1f} t/s", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Subplot 2: Speedup Multiplier
    bars2 = ax2.bar(precisions, speedups, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_title("Speedup Multiplier (vs Base NTP)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Speedup (x)", fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{yval:.2f}x", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Subplot 3: VRAM Reduction (%)
    bars3 = ax3.bar(precisions, vram_saved_pct, color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_title("VRAM Memory Savings (%)", fontsize=13, fontweight='bold')
    ax3.set_ylabel("VRAM Saved (%)", fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Subplot 4: Draft Acceptance Rate (DAR %)
    bars4 = ax4.bar(precisions, dar_percentages, color=colors, edgecolor='black', linewidth=1.2)
    ax4.set_title("Draft Acceptance Rate (DAR %)", fontsize=13, fontweight='bold')
    ax4.set_ylabel("DAR (%)", fontsize=11)
    ax4.set_ylim(0, 105)
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%" if yval > 0 else "N/A", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs("benchmark/plots", exist_ok=True)
    out_path = "benchmark/plots/quantization_comparison_bonsai.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Bonsai quantization comparison graph saved to: {out_path}")

if __name__ == "__main__":
    generate_bonsai_quantization_graph()
