# Master Research Findings & Project Summary Dossier

**Project Title**: Elastic-MTP: Uncertainty-Aware Dynamic Horizon Multi-Token Prediction for Hardware-Constrained Devices  
**Last Updated**: 2026-07-27  
**Master Log JSON**: [master_benchmark_log.json](file:///c:/Users/pshin/CODEE/research/benchmark/results/master_benchmark_log.json)

---

## Executive Summary

Elastic-MTP modifies multi-token prediction by dynamically tuning the speculative prediction horizon $K$ in real-time based on token Shannon Entropy $H(P_t)$ and Kullback-Leibler (KL) Divergence head safety checks.

---

## Key Experimental Results

1. **Inference Acceleration**:
   * Standard Next-Token Prediction (NTP, K=1): 249.3 tok/s
   * Static Multi-Token Prediction (MTP, K=4): **823.7 tok/s (3.30× Speedup)**
   * Elastic-MTP: Dynamically adjusts horizon length, achieving up to **2.27× real speedup** on HuggingFace `gpt2` models while protecting output accuracy.

2. **AutoResearch 50-Iteration Optimization**:
   * Evaluated 50 candidate hyperparameter mutations.
   * Baseline Composite Score: 124.14
   * **Optimal Composite Score**: **276.61 (+122.8% Improvement)**
   * Winning Hyperparameter Set:
     - `Entropy Threshold = 1.50`
     - `Divergence Threshold = 0.30`
     - `Max Speculative Horizon = 8`

3. **Quantization Noise Robustness**:
   * **FP16 Baseline**: 100.0% accuracy retention.
   * **INT8 Uniform**: **97.0% accuracy retention** with stable entropy routing.
   * **INT4 AWQ**: 50.0% retention under severe noise, triggering automatic router collapse to $K=1$ to protect generation accuracy.

4. **Fused GPU Router Kernel**:
   * Fused max-shifted log-softmax and entropy reduction into a single tensor pass.
   * **Execution Time**: 0.4516 ms per call.
   * **Intermediate VRAM Allocation**: 0 Bytes.

5. **Gated-LoRA Adapter Architecture**:
   * Parameter overhead: **<3.5%** ($<5\text{M}$ params vs $932\text{M}$ params for naive linear heads).
   * Gradient detachment verified (`z_t.grad is None`).
   * Trained checkpoint saved to `checkpoints/mtp_glora_adapter.pt`.

---

## Unit Test Status

- **Total Tests**: 11
- **Passed**: 11/11 (100%)
- **Execution Time**: 0.037s

---

## Saved Visual Plots & Logs

- Master Log: [master_benchmark_log.json](file:///c:/Users/pshin/CODEE/research/benchmark/results/master_benchmark_log.json)
- AutoResearch Registry: [autoresearch_registry.json](file:///c:/Users/pshin/CODEE/research/autoresearch/autoresearch_registry.json)
- Quantization Decay Curve: [quantization_decay_curve.png](file:///c:/Users/pshin/CODEE/research/benchmark/plots/quantization_decay_curve.png)
- Throughput Bar Chart: [throughput_comparison.png](file:///c:/Users/pshin/CODEE/research/benchmark/plots/throughput_comparison.png)
- Horizon Distribution: [horizon_distribution.png](file:///c:/Users/pshin/CODEE/research/benchmark/plots/horizon_distribution.png)
- AutoResearch Pareto Trajectory: [autoresearch_pareto_frontier.png](file:///c:/Users/pshin/CODEE/research/benchmark/plots/autoresearch_pareto_frontier.png)
