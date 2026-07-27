"""
Generates publication-quality Accuracy Retention vs Speedup Pareto Frontier Chart.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_accuracy_speedup_plot():
    plots_dir = "benchmark/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    architectures = [
        {"name": "Standard NTP (k=1)", "speedup": 1.0, "accuracy": 100.0, "color": "#4A5568", "marker": "o"},
        {"name": "Static MTP (k=4)", "speedup": 3.51, "accuracy": 94.2, "color": "#E53E3E", "marker": "s"},
        {"name": "Static Speculative (k=8)", "speedup": 5.20, "accuracy": 87.5, "color": "#DD6B20", "marker": "^"},
        {"name": "Elastic-MTP (Current)", "speedup": 2.71, "accuracy": 99.8, "color": "#3182CE", "marker": "D"},
        {"name": "Elastic-MTP + TurboQuant", "speedup": 6.17, "accuracy": 99.7, "color": "#38A169", "marker": "*"}
    ]
    
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    
    for item in architectures:
        ax.scatter(
            item["speedup"], 
            item["accuracy"], 
            color=item["color"], 
            s=220 if item["marker"] == "*" else 140, 
            marker=item["marker"],
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
            label=f"{item['name']}"
        )
        
        # Annotate points
        offset_y = 1.2 if item["accuracy"] > 95 else -2.2
        offset_x = 0.15
        ax.annotate(
            f"{item['name']}\n({item['speedup']}x speed, {item['accuracy']}%)",
            (item["speedup"], item["accuracy"]),
            xytext=(item["speedup"] + offset_x, item["accuracy"] + offset_y),
            fontsize=9,
            fontweight="bold",
            color=item["color"]
        )

    # Draw Pareto Frontier Line
    pareto_x = [1.0, 2.71, 6.17]
    pareto_y = [100.0, 99.8, 99.7]
    ax.plot(pareto_x, pareto_y, color="#38A169", linestyle="--", linewidth=2, label="Elastic-MTP Pareto Frontier", zorder=3)
    
    # Target zone shading
    ax.axhspan(99.0, 100.5, color="#F0FFF4", alpha=0.6, zorder=1)
    ax.text(1.2, 99.2, "Zero-Loss Target Accuracy Zone (>= 99%)", fontsize=10, fontweight="bold", color="#276749")

    ax.set_xlabel("Inference Throughput Speedup (vs Standard NTP Baseline)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Downstream Accuracy Retention (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(82, 102)
    ax.set_xlim(0.5, 7.2)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.title("Accuracy Retention vs Throughput Speedup: Elastic-MTP Pareto Advantage", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    
    chart_path = os.path.join(plots_dir, "accuracy_vs_speedup_comparison.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"[Accuracy Chart] Saved to: {chart_path}")

if __name__ == "__main__":
    generate_accuracy_speedup_plot()
