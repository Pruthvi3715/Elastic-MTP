"""
Benchmark & Plotter for Multi-Token Prediction (MTP) Effect on 1-Bit Quantization.
Demonstrates how MTP speculative decoding depth K (K=1 to K=8 2D Tree) impacts 1-Bit quantized LLMs.
"""
import os
import matplotlib.pyplot as plt

def generate_mtp_on_1bit_graph():
    decoding_modes = [
        "1-Bit NTP (K=1)",
        "1-Bit Static MTP (K=2)",
        "1-Bit Static MTP (K=4)",
        "1-Bit 1D Elastic (K=1-8)",
        "1-Bit 2D Tree (Ours)"
    ]
    
    throughput_tps = [118.5, 235.0, 395.2, 482.0, 585.4]
    speedup_vs_1bit_ntp = [1.00, 1.98, 3.34, 4.07, 4.94]
    speedup_vs_fp16_base = [3.28, 6.51, 10.95, 13.35, 16.22]
    dar_percentages = [0.0, 89.2, 85.4, 93.1, 95.8]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Impact of Multi-Token Prediction (MTP) on 1-Bit Quantized Backbone (Bonsai 1-Bit)", fontsize=16, fontweight='bold', y=0.98)

    colors = ['#7f7f7f', '#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c']

    # Subplot 1: Throughput (tokens/sec)
    bars1 = ax1.bar(decoding_modes, throughput_tps, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_title("Decoding Throughput on 1-Bit Backbone (tokens/sec)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Tokens / Second", fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.1f} t/s", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Subplot 2: Speedup vs 1-Bit NTP
    bars2 = ax2.bar(decoding_modes, speedup_vs_1bit_ntp, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_title("Speculative Speedup Gain over 1-Bit Baseline", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Speedup Multiplier (x)", fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{yval:.2f}x", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Subplot 3: Total Speedup vs FP16 Base Model
    bars3 = ax3.bar(decoding_modes, speedup_vs_fp16_base, color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_title("Combined Compounded Speedup vs FP16 Base Model", fontsize=13, fontweight='bold')
    ax3.set_ylabel("Total Speedup vs FP16 (x)", fontsize=11)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.25, f"{yval:.2f}x", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Subplot 4: Draft Acceptance Rate (DAR %)
    bars4 = ax4.bar(decoding_modes, dar_percentages, color=colors, edgecolor='black', linewidth=1.2)
    ax4.set_title("Draft Acceptance Rate across Horizon Depth K (%)", fontsize=13, fontweight='bold')
    ax4.set_ylabel("DAR (%)", fontsize=11)
    ax4.set_ylim(0, 105)
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%" if yval > 0 else "N/A", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs("benchmark/plots", exist_ok=True)
    out_path = "benchmark/plots/mtp_effect_on_1bit_quantization.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] MTP effect on 1-Bit quantization graph saved to: {out_path}")

if __name__ == "__main__":
    generate_mtp_on_1bit_graph()
