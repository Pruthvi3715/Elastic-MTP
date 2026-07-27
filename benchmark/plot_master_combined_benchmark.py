"""
Ultimate Master Combined Benchmark Dashboard
===========================================
Fixes text overlap issues with clean label formatting and combines all benchmark
insights into a single 6-panel publication-grade master dashboard.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def generate_master_combined_dashboard():
    print("=" * 80)
    print("GENERATING ULTIMATE MASTER COMBINED BENCHMARK DASHBOARD")
    print("=" * 80)

    # Milestones with short, clean labels (No Overlap)
    milestones = [
        "M1: Base",
        "M2: Static",
        "M3: 1D Elastic",
        "M4: Post-Trained",
        "M5: 2D Tree",
        "M6: vLLM CUDA"
    ]

    milestone_full = [
        "M1: Base (NTP)",
        "M2: Static (K=4)",
        "M3: 1D Elastic",
        "M4: Post-Trained",
        "M5: 2D Dynamic Tree",
        "M6: Elastic-vLLM Engine"
    ]

    speedups = [1.00, 1.55, 2.58, 2.85, 3.42, 4.85]
    dars = [0.0, 33.3, 75.0, 88.0, 94.2, 95.0]
    concurrency_streams = [16, 16, 64, 64, 48, 256]
    router_latencies = [0.000, 0.000, 0.480, 0.480, 0.450, 0.018]
    throughputs = [100.8, 108.0, 69.2, 120.5, 104.4, 291.1]
    costs = [10.00, 8.50, 4.20, 3.10, 2.40, 0.165]

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    os.makedirs("benchmark/plots", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)

    # Panel 1: Speculative Speedup Trajectory (x)
    ax1 = axes[0, 0]
    ax1.plot(milestones, speedups, marker="o", linewidth=3, markersize=8, color="#1b9e77")
    bars1 = ax1.bar(milestones, speedups, color=colors, alpha=0.4, width=0.5)
    ax1.set_title("1. Speculative Speedup vs Base (x)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Speedup Multiplier (x)", fontsize=10)
    ax1.set_ylim(0, 6.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.tick_params(axis="x", rotation=15, labelsize=9)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.15, f"{yval:.2f}x", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Panel 2: Draft Acceptance Rate (DAR %)
    ax2 = axes[0, 1]
    ax2.plot(milestones, dars, marker="s", linewidth=3, markersize=8, color="#377eb8")
    bars2 = ax2.bar(milestones, dars, color=colors, alpha=0.4, width=0.5)
    ax2.set_title("2. Draft Acceptance Rate (DAR %)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Acceptance Rate (%)", fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.tick_params(axis="x", rotation=15, labelsize=9)
    for bar in bars2:
        yval = bar.get_height()
        if yval > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 2.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Panel 3: Speculative Throughput (tokens/sec)
    ax3 = axes[0, 2]
    bars3 = ax3.bar(milestones, throughputs, color=colors, alpha=0.85, width=0.5, edgecolor="black")
    ax3.set_title("3. Speculative Throughput (tok/s)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Tokens / Second", fontsize=10)
    ax3.set_ylim(0, 350)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.tick_params(axis="x", rotation=15, labelsize=9)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 5.0, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Panel 4: Multi-Tenant Serving Streams (24GB GPU)
    ax4 = axes[1, 0]
    ax4.plot(milestones, concurrency_streams, marker="^", linewidth=3, markersize=8, color="#984ea3")
    bars4 = ax4.bar(milestones, concurrency_streams, color=colors, alpha=0.4, width=0.5)
    ax4.set_title("4. Concurrent Streams / 24GB GPU", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Max Concurrent Users", fontsize=10)
    ax4.set_ylim(0, 300)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    ax4.tick_params(axis="x", rotation=15, labelsize=9)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 6.0, f"{int(yval)}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Panel 5: Router Latency Overhead (ms)
    ax5 = axes[1, 1]
    bars5 = ax5.bar(milestones, router_latencies, color=colors, alpha=0.85, width=0.5, edgecolor="black")
    ax5.set_title("5. Router Overhead / Step (ms)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Latency (ms)", fontsize=10)
    ax5.set_ylim(0, 0.65)
    ax5.grid(axis="y", linestyle="--", alpha=0.5)
    ax5.tick_params(axis="x", rotation=15, labelsize=9)
    for bar in bars5:
        yval = bar.get_height()
        if yval > 0:
            ax5.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.015, f"{yval:.3f}ms", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Panel 6: Serving Cost per 1M Tokens ($)
    ax6 = axes[1, 2]
    bars6 = ax6.bar(milestones, costs, color=colors, alpha=0.85, width=0.5, edgecolor="black")
    ax6.set_title("6. Hosting Cost / 1M Tokens ($)", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Cost in USD ($)", fontsize=10)
    ax6.set_ylim(0, 12.0)
    ax6.grid(axis="y", linestyle="--", alpha=0.5)
    ax6.tick_params(axis="x", rotation=15, labelsize=9)
    for bar in bars6:
        yval = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"${yval:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.suptitle("ELASTIC-MTP: ULTIMATE MASTER COMBINED BENCHMARK DASHBOARD", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    abs_plot_path = os.path.abspath(os.path.join("benchmark", "plots", "master_combined_benchmark_dashboard.png"))
    os.makedirs(os.path.dirname(abs_plot_path), exist_ok=True)
    plt.savefig(abs_plot_path, bbox_inches="tight")

    artifact_dir = r"C:\Users\pshin\.gemini\antigravity-ide\brain\5921faee-3025-4dfe-8804-ce495227ab51"
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_path = os.path.join(artifact_dir, "master_combined_benchmark_dashboard.png")
    plt.savefig(artifact_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved Ultimate Master Combined Benchmark Dashboard to {abs_plot_path}")
    print(f"[OK] Saved Artifact Copy to {artifact_path}")


if __name__ == "__main__":
    generate_master_combined_dashboard()
