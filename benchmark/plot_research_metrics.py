"""
Research Metrics Visualization for Elastic-MTP.

Generates publication-ready plots showing:
1. Draft Acceptance Rate (DAR) vs Prompt Predictability
2. Contradiction Rate Analysis
3. Dynamic K Distribution Across Prompts
4. Speedup vs DAR Correlation
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from src.config import ElasticMTPConfig

def load_benchmark_results():
    """Load benchmark results from JSON file."""
    results_file = os.path.join(ElasticMTPConfig.RESULTS_DIR, "benchmark_results.json")
    with open(results_file, "r") as f:
        return json.load(f)

def plot_dar_vs_predictability(elastic_results):
    """Plot Draft Acceptance Rate against prompt predictability (avg K)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    prompts = [f"P{i+1}" for i in range(len(elastic_results))]
    dar_values = []
    avg_k_values = []
    
    for res in elastic_results:
        metrics = res.get("router_metrics", {})
        dar_values.append(metrics.get("draft_acceptance_rate_percent", 0))
        avg_k_values.append(metrics.get("avg_k", 0))
    
    # Color by speedup
    speedups = [res["tokens_per_sec"] for res in elastic_results]
    
    scatter = ax.scatter(avg_k_values, dar_values, c=speedups, cmap='viridis', 
                         s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Annotate points with prompt numbers
    for i, (k, dar) in enumerate(zip(avg_k_values, dar_values)):
        ax.annotate(f'P{i+1}', (k, dar), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, ha='left')
    
    ax.set_xlabel('Average K (Prediction Horizon)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Draft Acceptance Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Elastic-MTP: DAR vs Prediction Horizon (K)\nHigher K + Higher DAR = Better Performance', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Throughput (tokens/sec)', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(ElasticMTPConfig.PLOTS_DIR, "dar_vs_k.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_k_distribution(elastic_results):
    """Show distribution of K values across different prompts."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    k_dist_data = []
    prompt_labels = []
    
    for i, res in enumerate(elastic_results):
        metrics = res.get("router_metrics", {})
        k_dist = metrics.get("k_distribution", {})
        prompt_labels.append(f"P{i+1}")
        
        # Build stacked bar data
        row = [k_dist.get(str(k), 0) for k in range(1, 9)]
        k_dist_data.append(row)
    
    k_dist_array = np.array(k_dist_data)
    x = np.arange(len(prompt_labels))
    
    bottom = np.zeros(len(prompt_labels))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, 8))
    
    for k in range(8):
        ax.bar(x, k_dist_array[:, k], bottom=bottom, label=f'K={k+1}', 
               color=colors[k], edgecolor='black', linewidth=0.5)
        bottom += k_dist_array[:, k]
    
    ax.set_xlabel('Prompt', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Routing Decisions', fontsize=12, fontweight='bold')
    ax.set_title('Elastic-MTP: Dynamic K Selection Across Prompts\n(Low Entropy → High K, High Entropy → K=1)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_labels)
    ax.legend(title='Prediction Horizon K', loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(ElasticMTPConfig.PLOTS_DIR, "k_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_contradiction_analysis(elastic_results):
    """Analyze contradiction events (safeguard triggers)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    contradiction_rates = []
    entropy_values = []
    
    for res in elastic_results:
        metrics = res.get("router_metrics", {})
        contradiction_rates.append(metrics.get("contradiction_rate_percent", 0))
        entropy_values.append(res["mean_entropy"])
    
    # Left: Contradiction rate by prompt
    ax1 = axes[0]
    prompts = [f"P{i+1}" for i in range(len(elastic_results))]
    bars = ax1.bar(prompts, contradiction_rates, color='coral', edgecolor='darkred', linewidth=1.5)
    
    for bar, rate in zip(bars, contradiction_rates):
        if rate > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Prompt', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Contradiction Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Hallucination Lock-in Prevention\n(KL-Divergence Safeguard Triggers)', 
                  fontsize=14, fontweight='bold')
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Warning threshold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Right: Entropy vs K selected
    ax2 = axes[1]
    avg_k_values = [res.get("router_metrics", {}).get("avg_k", 0) for res in elastic_results]
    
    ax2.scatter(entropy_values, avg_k_values, s=150, c='steelblue', alpha=0.7, 
                edgecolors='black', linewidth=1.5)
    
    for i, (ent, k) in enumerate(zip(entropy_values, avg_k_values)):
        ax2.annotate(f'P{i+1}', (ent, k), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left')
    
    ax2.set_xlabel('Mean Entropy H(P) (nats)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average K Selected', fontsize=12, fontweight='bold')
    ax2.set_title('Entropy-Based Routing Decision\n(High Entropy → K=1, Low Entropy → K=8)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(ElasticMTPConfig.PLOTS_DIR, "contradiction_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_speedup_comparison(all_results):
    """Compare throughput across NTP, Static MTP, and Elastic-MTP."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    modes = ['ntp', 'static_mtp', 'elastic']
    mode_labels = ['NTP\n(k=1)', 'Static MTP\n(k=4)', 'Elastic-MTP\n(dynamic)']
    
    prompt_labels = [f"P{i+1}" for i in range(len(all_results['ntp']))]
    x = np.arange(len(prompt_labels))
    width = 0.25
    
    speedup_data = []
    for mode in modes:
        speedups = [res["tokens_per_sec"] for res in all_results[mode]]
        speedup_data.append(speedups)
    
    for i, (mode, label) in enumerate(zip(modes, mode_labels)):
        ax.bar(x + i*width, speedup_data[i], width, label=label, 
               edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Elastic-MTP Performance Comparison\n(Across Different Prompt Predictability Levels)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(prompt_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on top of bars
    for i, speeds in enumerate(speedup_data):
        for j, speed in enumerate(speeds):
            ax.text(x[j] + i*width, speed + 1, f'{speed:.0f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(ElasticMTPConfig.PLOTS_DIR, "speedup_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def generate_summary_table(elastic_results):
    """Generate a markdown summary table of research metrics."""
    print("\n" + "="*80)
    print("ELASTIC-MTP RESEARCH METRICS SUMMARY TABLE")
    print("="*80)
    print()
    
    header = "| Prompt | Avg K | DAR (%) | Contradictions (%) | Speedup (tok/s) | Entropy H(P) |"
    separator = "|" + "-"*8 + "|" + "-"*7 + "|" + "-"*9 + "|" + "-"*18 + "|" + "-"*17 + "|" + "-"*14 + "|"
    
    print(header)
    print(separator)
    
    for i, res in enumerate(elastic_results):
        metrics = res.get("router_metrics", {})
        row = f"| P{i+1} | {metrics.get('avg_k', 0):.1f} | {metrics.get('draft_acceptance_rate_percent', 0):.1f} | {metrics.get('contradiction_rate_percent', 0):.1f} | {res['tokens_per_sec']:>8.2f} | {res['mean_entropy']:>10.2f} |"
        print(row)
    
    print()
    print("="*80)

def main():
    print("="*70)
    print("Elastic-MTP Research Metrics Visualization")
    print("="*70)
    print()
    
    # Load results
    results = load_benchmark_results()
    
    # Generate plots
    print("Generating research visualizations...")
    print()
    
    plot_dar_vs_predictability(results['elastic'])
    plot_k_distribution(results['elastic'])
    plot_contradiction_analysis(results['elastic'])
    plot_speedup_comparison(results)
    
    # Generate summary table
    generate_summary_table(results['elastic'])
    
    print()
    print("="*70)
    print("All research metrics visualizations generated successfully!")
    print("="*70)
    print()
    print("Generated files:")
    print(f"  - {os.path.join(ElasticMTPConfig.PLOTS_DIR, 'dar_vs_k.png')}")
    print(f"  - {os.path.join(ElasticMTPConfig.PLOTS_DIR, 'k_distribution.png')}")
    print(f"  - {os.path.join(ElasticMTPConfig.PLOTS_DIR, 'contradiction_analysis.png')}")
    print(f"  - {os.path.join(ElasticMTPConfig.PLOTS_DIR, 'speedup_comparison.png')}")
    print()
    print("Key Metrics Now Tracked:")
    print("  ✓ Draft Acceptance Rate (DAR)")
    print("  ✓ Contradiction Rate (KL-divergence safeguard triggers)")
    print("  ✓ Dynamic K distribution")
    print("  ✓ Entropy-based routing decisions")
    print("  ✓ Throughput comparisons")

if __name__ == "__main__":
    main()
