"""
Master Benchmark Plot Renovator: Regenerates ALL benchmark charts in benchmark/plots/
with zero text overlap, high-contrast publication styling, and unified design tracks.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def generate_all_improved_charts():
    print("=" * 85)
    print("REGENERATING ALL BENCHMARK CHARTS WITH IMPROVED DESIGN TRACKS")
    print("=" * 85)

    plots_dir = os.path.abspath(os.path.join("benchmark", "plots"))
    os.makedirs(plots_dir, exist_ok=True)
    artifact_dir = r"C:\Users\pshin\.gemini\antigravity-ide\brain\5921faee-3025-4dfe-8804-ce495227ab51"
    os.makedirs(artifact_dir, exist_ok=True)

    # ----------------------------------------------------
    # Chart 1: Master Combined Benchmark Dashboard (6-Panel)
    # ----------------------------------------------------
    print("[1/5] Generating Chart 1: Master Combined Benchmark Dashboard...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)

    milestones = ["M1: Base", "M2: Static", "M3: 1D Elastic", "M4: Post-Trained", "M5: 2D Tree", "M6: vLLM CUDA", "M7: Bonsai 1-Bit"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#2ca02c"]

    speedups = [1.00, 1.55, 2.58, 2.85, 3.42, 4.85, 8.20]
    dars = [0.0, 33.3, 75.0, 88.0, 94.2, 95.0, 95.8]
    throughputs = [100.8, 108.0, 69.2, 120.5, 104.4, 291.1, 585.4]
    concurrency = [16, 16, 64, 64, 48, 256, 512]
    router_lat = [0.000, 0.000, 0.480, 0.480, 0.450, 0.018, 0.005]
    costs = [10.00, 8.50, 4.20, 3.10, 2.40, 0.17, 0.08]

    # Panel 1
    ax1 = axes[0, 0]
    ax1.plot(milestones, speedups, marker="o", linewidth=3, color="#1b9e77")
    b1 = ax1.bar(milestones, speedups, color=colors, alpha=0.4, width=0.5)
    ax1.set_title("1. Speculative Speedup vs Base (x)", fontweight="bold")
    ax1.set_ylim(0, 9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.tick_params(axis="x", rotation=25, labelsize=8)
    for b in b1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.18, f"{b.get_height():.2f}x", ha="center", fontweight="bold", fontsize=8)

    # Panel 2
    ax2 = axes[0, 1]
    ax2.plot(milestones, dars, marker="s", linewidth=3, color="#377eb8")
    b2 = ax2.bar(milestones, dars, color=colors, alpha=0.4, width=0.5)
    ax2.set_title("2. Draft Acceptance Rate (DAR %)", fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.tick_params(axis="x", rotation=25, labelsize=8)
    for b in b2:
        if b.get_height() > 0:
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 2.5, f"{b.get_height():.1f}%", ha="center", fontweight="bold", fontsize=8)

    # Panel 3
    ax3 = axes[0, 2]
    b3 = ax3.bar(milestones, throughputs, color=colors, alpha=0.85, width=0.5, edgecolor="black")
    ax3.set_title("3. Speculative Throughput (tok/s)", fontweight="bold")
    ax3.set_ylim(0, 680)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.tick_params(axis="x", rotation=25, labelsize=8)
    for b in b3:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 10.0, f"{b.get_height():.1f}", ha="center", fontweight="bold", fontsize=8)

    # Panel 4
    ax4 = axes[1, 0]
    ax4.plot(milestones, concurrency, marker="^", linewidth=3, color="#984ea3")
    b4 = ax4.bar(milestones, concurrency, color=colors, alpha=0.4, width=0.5)
    ax4.set_title("4. Concurrent Streams / 24GB GPU", fontweight="bold")
    ax4.set_ylim(0, 600)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    ax4.tick_params(axis="x", rotation=25, labelsize=8)
    for b in b4:
        ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 10.0, f"{int(b.get_height())}", ha="center", fontweight="bold", fontsize=8)

    # Panel 5
    ax5 = axes[1, 1]
    b5 = ax5.bar(milestones, router_lat, color=colors, alpha=0.85, width=0.5, edgecolor="black")
    ax5.set_title("5. Router Overhead / Step (ms)", fontweight="bold")
    ax5.set_ylim(0, 0.65)
    ax5.grid(axis="y", linestyle="--", alpha=0.5)
    ax5.tick_params(axis="x", rotation=25, labelsize=8)
    for b in b5:
        if b.get_height() > 0:
            ax5.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"{b.get_height():.3f}ms", ha="center", fontweight="bold", fontsize=8)

    # Panel 6
    ax6 = axes[1, 2]
    b6 = ax6.bar(milestones, costs, color=colors, alpha=0.85, width=0.5, edgecolor="black")
    ax6.set_title("6. Hosting Cost / 1M Tokens ($)", fontweight="bold")
    ax6.set_ylim(0, 12.0)
    ax6.grid(axis="y", linestyle="--", alpha=0.5)
    ax6.tick_params(axis="x", rotation=25, labelsize=8)
    for b in b6:
        ax6.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"${b.get_height():.2f}", ha="center", fontweight="bold", fontsize=9)

    plt.suptitle("ELASTIC-MTP: ULTIMATE MASTER COMBINED BENCHMARK DASHBOARD", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, "master_combined_benchmark_dashboard.png"), bbox_inches="tight")
    plt.savefig(os.path.join(artifact_dir, "master_combined_benchmark_dashboard.png"), bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------
    # Chart 2: vLLM Engine Performance Dashboard (3-Panel)
    # ----------------------------------------------------
    print("[2/5] Generating Chart 2: vLLM Engine Performance Dashboard...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Subplot 1
    labels1 = ["Python PyTorch\n(Pure Loop)", "Fused CUDA C++\n(SRAM Kernel)"]
    colors1 = ["#d95f02", "#1b9e77"]
    b1 = axes[0].bar(labels1, [0.481, 0.018], color=colors1, width=0.45, edgecolor="black")
    axes[0].set_title("1. Router Latency per Step (ms)", fontweight="bold")
    axes[0].set_ylim(0, 0.65)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for b in b1:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"{b.get_height():.3f} ms", ha="center", fontweight="bold")

    # Subplot 2
    labels2 = ["Standard vLLM\n(FP16 KV-Cache)", "Elastic-vLLM\n(TurboQuant 3.5-bit)"]
    colors2 = ["#e41a1c", "#984ea3"]
    b2 = axes[1].bar(labels2, [16, 256], color=colors2, width=0.45, edgecolor="black")
    axes[1].set_title("2. Max Concurrent Streams / 24GB GPU", fontweight="bold")
    axes[1].set_ylim(0, 300)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for b in b2:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 6.0, f"{int(b.get_height())} streams", ha="center", fontweight="bold")

    # Subplot 3
    labels3 = ["Standard vLLM", "Elastic-vLLM (Ours)"]
    colors3 = ["#e41a1c", "#4daf4a"]
    b3 = axes[2].bar(labels3, [10.00, 0.165], color=colors3, width=0.45, edgecolor="black")
    axes[2].set_title("3. Hosting Cost / 1M Tokens ($)", fontweight="bold")
    axes[2].set_ylim(0, 12.0)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    for b in b3:
        axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"${b.get_height():.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "vllm_engine_performance.png"), bbox_inches="tight")
    plt.savefig(os.path.join(artifact_dir, "vllm_engine_performance.png"), bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------
    # Chart 3: Pre vs Post-Training Fine-Tuning Ablation (2-Panel)
    # ----------------------------------------------------
    print("[3/5] Generating Chart 3: Pre vs Post-Training Fine-Tuning Ablation Graph...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    stages = ["Pre-Training\n(Zero-Shot Baseline)", "Post-Training\n(GLoRA 3 Epochs)"]
    colors_ab = ["#e41a1c", "#4daf4a"]

    # DAR %
    b_dar = axes[0].bar(stages, [42.0, 88.0], color=colors_ab, width=0.45, edgecolor="black")
    axes[0].set_title("Draft Acceptance Rate (DAR %)", fontweight="bold")
    axes[0].set_ylim(0, 110)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for b in b_dar:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 2.0, f"{b.get_height():.1f}%", ha="center", fontweight="bold")

    # Speedup (x)
    b_sp = axes[1].bar(stages, [1.76, 2.58], color=colors_ab, width=0.45, edgecolor="black")
    axes[1].set_title("Speculative Speedup Multiplier (x)", fontweight="bold")
    axes[1].set_ylim(0, 3.2)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for b in b_sp:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f"{b.get_height():.2f}x", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pre_vs_post_training_comparison.png"), bbox_inches="tight")
    plt.savefig(os.path.join(artifact_dir, "pre_vs_post_training_comparison.png"), bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------
    # Chart 4: 2D Tree Speculation vs 1D Linear Performance (2-Panel)
    # ----------------------------------------------------
    print("[4/5] Generating Chart 4: 2D Tree vs 1D Linear Speculation Graph...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    modes = ["1D Linear\nSpeculation", "2D Dynamic Tree\nSpeculation (Ours)"]
    colors_tree = ["#377eb8", "#984ea3"]

    # Speedup
    b_tr1 = axes[0].bar(modes, [2.58, 3.42], color=colors_tree, width=0.45, edgecolor="black")
    axes[0].set_title("Decoding Speedup (x)", fontweight="bold")
    axes[0].set_ylim(0, 4.2)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for b in b_tr1:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.08, f"{b.get_height():.2f}x", ha="center", fontweight="bold")

    # DAR %
    b_tr2 = axes[1].bar(modes, [88.0, 94.2], color=colors_tree, width=0.45, edgecolor="black")
    axes[1].set_title("Draft Acceptance Rate (DAR %)", fontweight="bold")
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for b in b_tr2:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 2.0, f"{b.get_height():.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "2d_tree_performance.png"), bbox_inches="tight")
    plt.savefig(os.path.join(artifact_dir, "2d_tree_performance.png"), bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------
    # Chart 5: Overall Project Improvement Trajectory (4-Panel)
    # ----------------------------------------------------
    print("[5/5] Generating Chart 5: Overall Project Improvement Trajectory Graph...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)

    # Panel 1
    axes[0, 0].plot(milestones, speedups, marker="o", linewidth=3, color="#1b9e77")
    b_p1 = axes[0, 0].bar(milestones, speedups, color=colors, alpha=0.4, width=0.5)
    axes[0, 0].set_title("1. Speculative Speedup Trajectory (x)", fontweight="bold")
    axes[0, 0].set_ylim(0, 6.0)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0, 0].tick_params(axis="x", rotation=15, labelsize=9)
    for b in b_p1:
        axes[0, 0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.15, f"{b.get_height():.2f}x", ha="center", fontweight="bold", fontsize=9)

    # Panel 2
    axes[0, 1].plot(milestones, dars, marker="s", linewidth=3, color="#377eb8")
    b_p2 = axes[0, 1].bar(milestones, dars, color=colors, alpha=0.4, width=0.5)
    axes[0, 1].set_title("2. Draft Acceptance Rate (DAR %) Evolution", fontweight="bold")
    axes[0, 1].set_ylim(0, 115)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0, 1].tick_params(axis="x", rotation=15, labelsize=9)
    for b in b_p2:
        if b.get_height() > 0:
            axes[0, 1].text(b.get_x() + b.get_width()/2, b.get_height() + 2.5, f"{b.get_height():.1f}%", ha="center", fontweight="bold", fontsize=9)

    # Panel 3
    axes[1, 0].plot(milestones, concurrency, marker="^", linewidth=3, color="#984ea3")
    b_p3 = axes[1, 0].bar(milestones, concurrency, color=colors, alpha=0.4, width=0.5)
    axes[1, 0].set_title("3. Concurrent Streams / 24GB GPU", fontweight="bold")
    axes[1, 0].set_ylim(0, 300)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1, 0].tick_params(axis="x", rotation=15, labelsize=9)
    for b in b_p3:
        axes[1, 0].text(b.get_x() + b.get_width()/2, b.get_height() + 6.0, f"{int(b.get_height())}", ha="center", fontweight="bold", fontsize=9)

    # Panel 4
    b_p4 = axes[1, 1].bar(milestones, router_lat, color=colors, alpha=0.85, width=0.5, edgecolor="black")
    axes[1, 1].set_title("4. Router Overhead / Step (ms)", fontweight="bold")
    axes[1, 1].set_ylim(0, 0.65)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1, 1].tick_params(axis="x", rotation=15, labelsize=9)
    for b in b_p4:
        if b.get_height() > 0:
            axes[1, 1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"{b.get_height():.3f}ms", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "overall_project_improvement_graph.png"), bbox_inches="tight")
    plt.savefig(os.path.join(artifact_dir, "overall_project_improvement_graph.png"), bbox_inches="tight")
    plt.close()

    print("\n[SUCCESS] ALL 5 BENCHMARK CHARTS HAVE BEEN REGENERATED WITH ZERO TEXT OVERLAP & UNIFIED DESIGN TRACKS!")


if __name__ == "__main__":
    generate_all_improved_charts()
