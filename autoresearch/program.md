# AutoResearch Directive: Elastic-MTP Optimization

**Research Focus**: Autonomous Optimization of Uncertainty-Aware Dynamic Horizon Multi-Token Prediction (Elastic-MTP).

---

## Target Metrics

1. **Maximize Throughput**: $\text{Tokens/Sec} \ge 1.5\times \text{NTP Baseline}$
2. **Maximize Draft Acceptance Rate**: $\text{DAR} \ge 75.0\%$
3. **Minimize Contradiction Rate**: $\text{KL Divergence Exceptions} \le 5.0\%$
4. **Zero Numerical Instability**: Zero `NaN` or `Inf` in FP16 / INT4 precision.

---

## Allowed Sandbox Files
- `autoresearch/train_sandbox.py` (Editable model parameters, logit clamping, entropy thresholds $\tau$, Gated-LoRA ranks $r$)

## Protected Immutable Files
- `autoresearch/prepare_eval.py` (Evaluation harness & benchmark dataset split)

---

## Experiment Loop Rules

1. Every experiment proposes a single hypothesis (e.g., *"Lowering tau_entropy to 1.5 improves draft acceptance rate on math prompts"*).
2. Runs validation pass on benchmark prompts.
3. If `Validation Score` > `Best Historic Score`, the experiment is **ACCEPTED** and committed to `experiment_registry.json`.
4. If `Validation Score` $\le$ `Best Historic Score` or crashes with `NaN`, the experiment is **REVERTED**.
