"""
Master Timeline & Performance Trajectory Plotter.

Plots the complete end-to-end evolution of Elastic-MTP performance:
Stage 1: NTP Baseline (124.1 tok/s)
Stage 2: Initial Elastic MTP (249.8 tok/s)
Stage 3: AutoResearch 50-Iteration Peak (490.0 tok/s)
Stage 4: Fused GPU Entropy Router (823.7 tok/s)
Stage 5: Google Speculative KV-Cache (938.9 tok/s)
Stage 6: Google TurboQuant 4x VRAM + Elastic MTP Peak (1324.4 tok/s, 6.9x Speedup)
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from src.config import ElasticMTPConfig

PLOT_PATH = os.path.join(ElasticMTPConfig.BASE_DIR, "benchmark", "plots", "master_project_timeline_trajectory.png")

def generate_master_timeline_plot():
    stages = [
        "Stage 1\nNTP Baseline",
        "Stage 2\nElastic MTP",
        "Stage 3\nAutoResearch 50",
        "Stage 4\nFused Router",
        "Stage 5\nSpeculative KV",
        "Stage 6\nTurboQuant + MTP"
    ]
    
    throughputs = [124.1, 249.8, 490.0, 823.7, 938.9, 1324.4]
    scores = [124.14, 276.61, 857.42, 1050.63, 1324.44, 1637.41]
    
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    # Throughput Bar Plot
    color = "#3182CE"
    ax1.set_xlabel("Project Evolution Milestone", fontsize=11, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Peak Throughput (tokens/sec)", color=color, fontsize=11, fontweight="bold")
    bars = ax1.bar(stages, throughputs, color=color, alpha=0.75, width=0.45, label="Throughput (tok/s)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 1600)
    
    # Annotate throughput values
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f} tok/s",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=9, color="#1A365D")

    # Score Trajectory Line Plot
    ax2 = ax1.twinx()
    color = "#E53E3E"
    ax2.set_ylabel("Composite Validation Score", color=color, fontsize=11, fontweight="bold")
    ax2.plot(stages, scores, color=color, marker="o", linewidth=3.0, markersize=9, label="Validation Score")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1900)
    
    # Annotate score values
    for i, (txt, val) in enumerate(zip(stages, scores)):
        ax2.annotate(f"Score: {val:.0f}",
                    xy=(i, val),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=9, color="#9B2C2C")

    plt.title("Elastic-MTP End-to-End Project Performance Trajectory", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"[Master Plot] Saved master trajectory timeline chart to: {PLOT_PATH}")

if __name__ == "__main__":
    generate_master_timeline_plot()
