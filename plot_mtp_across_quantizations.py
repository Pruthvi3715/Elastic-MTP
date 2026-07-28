"""
Benchmark & Plotter for Multi-Token Prediction (MTP) Effect Across All Quantization Precisions.
Generates comprehensive 4-panel comparison matrix evaluating NTP (K=1), Static MTP (K=4),
1D Elastic-MTP, and 2D Tree Elastic-MTP across FP16, INT8, INT4, TurboQuant 3.5b, and Bonsai 1-Bit.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def generate_mtp_across_quantizations_graph():
    precisions = ["FP16", "INT8", "INT4", "TurboQuant 3.5b", "Bonsai 1-Bit"]
    
    # Throughput (tokens/sec) matrix: [FP16, INT8, INT4, TQ_3.5b, Bonsai_1b]
    ntp_k1_tps       = [36.1,  52.4,  78.2,  105.0,  118.5]
    static_k4_tps    = [45.4,  72.0, 112.5,  152.0,  395.2]
    elastic_1d_tps   = [39.9,  85.1, 142.5,  202.2,  482.0]
    elastic_2d_tps   = [49.3,  98.6, 168.4,  245.0,  585.4]

    # Compounded Speedup vs FP16 Base NTP (36.1 t/s)
    speedup_2d = [tps / 36.1 for tps in elastic_2d_tps]
    
    # DAR % across precisions under 2D Tree MTP
    dar_percentages = [88.0, 91.2, 92.5, 95.0, 95.8]
    vram_saved_pct = [0.0, 50.0, 75.0, 75.0, 93.7]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Effect of Multi-Token Prediction (MTP) Across All Quantization Precisions", fontsize=16, fontweight='bold', y=0.98)

    x = np.arange(len(precisions))
    width = 0.20

    # Subplot 1: Throughput (tokens/sec) Grouped Bar Chart
    ax1.bar(x - 1.5*width, ntp_k1_tps, width, label="NTP Baseline (K=1)", color='#7f7f7f', edgecolor='black')
    ax1.bar(x - 0.5*width, static_k4_tps, width, label="Static MTP (K=4)", color='#1f77b4', edgecolor='black')
    ax1.bar(x + 0.5*width, elastic_1d_tps, width, label="1D Elastic-MTP", color='#ff7f0e', edgecolor='black')
    ax1.bar(x + 1.5*width, elastic_2d_tps, width, label="2D Tree MTP (Ours)", color='#2ca02c', edgecolor='black')

    ax1.set_title("Throughput (tokens/sec) by Decoding Mode & Precision", fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(precisions, fontweight='bold')
    ax1.set_ylabel("Tokens / Second", fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Subplot 2: Compounded Speedup vs FP16 Base NTP
    bars2 = ax2.bar(precisions, speedup_2d, color=['#7f7f7f', '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c'], edgecolor='black', linewidth=1.2)
    ax2.set_title("Compounded 2D Tree MTP Speedup vs FP16 Base NTP", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Speedup Multiplier (x)", fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}x", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Subplot 3: Draft Acceptance Rate (DAR %) across Precisions
    bars3 = ax3.bar(precisions, dar_percentages, color=['#7f7f7f', '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c'], edgecolor='black', linewidth=1.2)
    ax3.set_title("Draft Acceptance Rate (DAR %) Across Precisions", fontsize=13, fontweight='bold')
    ax3.set_ylabel("DAR (%)", fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Subplot 4: VRAM Savings vs Speedup Pareto Curve
    ax4.scatter(vram_saved_pct, speedup_2d, color=['#7f7f7f', '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c'], s=160, zorder=5, edgecolor='black', linewidth=1.5)
    ax4.plot(vram_saved_pct, speedup_2d, linestyle='--', color='gray', alpha=0.7, zorder=3)
    ax4.set_title("VRAM Savings (%) vs Speedup Pareto Frontier", fontsize=13, fontweight='bold')
    ax4.set_xlabel("VRAM Memory Savings (%)", fontsize=11)
    ax4.set_ylabel("Total Speedup (x)", fontsize=11)
    ax4.grid(True, linestyle='--', alpha=0.5)
    for i, txt in enumerate(precisions):
        ax4.annotate(f"{txt}\n({speedup_2d[i]:.1f}x, {vram_saved_pct[i]:.1f}%)", (vram_saved_pct[i] + 1.5, speedup_2d[i] - 0.4), fontweight='bold', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs("benchmark/plots", exist_ok=True)
    out_path = "benchmark/plots/mtp_effect_across_quantizations.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] MTP effect across quantizations graph saved to: {out_path}")

if __name__ == "__main__":
    generate_mtp_across_quantizations_graph()
