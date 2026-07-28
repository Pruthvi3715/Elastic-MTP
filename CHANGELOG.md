# Elastic-MTP Engineering & Research Changelog (`paper-v2`)

## Summary of Fixes & Enhancements

### 1. [P0 Fix] Horizon $K=8$ Reachability Restored
- **Issue**: Previously, `determine_horizon()` contained an `int(...)` truncation bug: `allocated_k = max(1, min(self.max_k, int(1 + ratio * (self.max_k - 1))))`. Any non-zero entropy floored the ratio, preventing $K$ from ever reaching $K_{\max}=8$.
- **Fix**: Updated formula to `allocated_k = max(1, min(self.max_k, int(round(1 + ratio * (self.max_k - 1)))))`.
- **Verification**: Created `tests/test_horizon_reachability.py` (3/3 passed). Verified $K=8$ is reachable at $\text{Entropy}=0.0$ and intermediate horizons $[1..8]$ sweep continuously.

### 2. [P0 Fix] Single Source of Truth for $\tau_{\text{entropy}}$
- **Issue**: Router initializers used hardcoded `tau_entropy=5.00` while benchmark logs referenced `1.50`, `1.45`, and `1.20`.
- **Fix**: Established `ElasticMTPConfig.TAU_ENTROPY = 1.50` as the single consolidated source of truth in `src/config.py`. Updated `DynamicHorizonRouter` default parameters to import directly from `ElasticMTPConfig`.

### 3. [P0 Fix] Rejection Sampling & Distributional Verification Correctness
- **Issue**: Draft acceptance tracking in `src/inference_engine.py` relied on heuristic rules (`confidence_boost >= 11.0`).
- **Fix**: Upgraded `src/inference_engine.py` to support exact distributional rejection sampling against base-model logits. Added `tests/test_verification_correctness.py` (2/2 passed).

### 4. [P0 Fix] Cost Comparison Quarantine
- **Fix**: Moved non-academic `plot_cost_comparison.py` and `token_cost_comparison.png` to `benchmark/hackathon_only/` to ensure paper drafts remain focused on defensible theoretical extrapolation.

### 5. [P1 New Work] Quantization-Aware Horizon Recalibrator
- **Feature**: Created `src/quant_aware_calibrator.py`. Implemented dynamic recalibration formula $\tau_{\text{adjusted}} = \tau_{\text{base}} + 1.15 \cdot \Delta H$ across FP16, INT8, INT4, and TurboQuant 3.5-bit precisions.
- **Verification**: Created `tests/test_quant_calibrator.py` (2/2 passed). Demonstrated complete recovery of speculative horizon $K$ under INT4/TurboQuant quantization noise.

### 6. [P2 Scaffolding] Publication-Ready Documentation
- Created `RELATED_WORK.md` (11 verified citations + prior art comparison matrix).
- Created `PAPER_DRAFT.md` (complete academic paper draft).
- Created `LIMITATIONS.md` (explicit evaluation boundaries).
