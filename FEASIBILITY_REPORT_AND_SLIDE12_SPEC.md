# Feasibility Report & Slide 12 Specification

**Project**: Uncertainty-Aware Dynamic Horizon Multi-Token Prediction (Elastic-MTP)  
**Document Purpose**: Full technical assessment, mathematical formulations, hardware matrix, risk breakdown, and Slide 12 problem specification.

---

## 1. Theoretical Architecture & Math Formulations

### 1.1 Multi-Token Prediction Objective
Standard Next-Token Prediction (NTP) optimizes a single cross-entropy loss. MTP appends auxiliary heads / Gated-LoRA adapter necks to predict $k$ future tokens simultaneously:

$$\mathcal{L}_{MTP} = \mathcal{L}_{NTP} + \sum_{i=1}^{k} \lambda_i \cdot \text{CE}(P_i, y_{t+i})$$

Where $\lambda_i$ represents the weighting coefficient for the $i$-th auxiliary future token head.

### 1.2 Logit Entropy & Uncertainty Quantification
Uncertainty at token step $t$ is calculated via Shannon Entropy $H(P)$ over vocabulary $V$:

$$H(P) = -\sum_{v \in V} P(v) \log P(v)$$

### 1.3 Stabilized Log-Domain Operations
To prevent underflow, log-of-zero operations, or `NaN` gradients in BF16/FP8 precision regimes, logits are clamped before `log_softmax`:

$$\text{Logits}_{\text{clamped}} = \text{clamp}(\mathbf{z}, -100.0, 100.0)$$
$$\log P(v) = \text{log\_softmax}(\text{Logits}_{\text{clamped}})$$

---

## 2. Hardware Constraints & Memory Matrix

| Hardware Profile | Target Model Scale | MTP Approach | Memory Required | Feasibility & Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Node Cluster** (4×GH200 / 8×H800) | 8B to 70B+ Base | Full Block Cloning + FSDP2 | $\ge 80\text{GB/GPU}$ | Feasible. Bottleneck: Inter-node all-reduce sync. |
| **Single Workstation** (NVIDIA L4 24GB / RTX 4090) | 1.5B to 8B Base | Gated LoRA (MTP-GLoRA) + Quant Base | 16–24GB | Feasible with constraints. Bottleneck: Activation memory on long sequences. |
| **Single Desktop** (RTX 3090 16GB) | 1.5B to 3B Base | Mini-Qwen / Sparse Probing | 12–16GB | Feasible strictly for micro-models. Bottleneck: Severe batch size limit. |
| **Edge Workstation** (Apple M4 Pro / Strix Halo) | 2B to 4B Base | Native MTP Inference Only (llama.cpp) | 18–36GB Unified | Feasible for inference eval. Bottleneck: Memory bandwidth. |

> **Advisor Rule**: Undergrad scope MUST stick to adapter-based fine-tuning on sub-8B models or inference-only speculative decoding evaluations.

---

## 3. Failure Mode & Risk Matrix

| Failure Mode / Risk | Severity | Probability | Structural Root Cause | Project Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Shared Parameter Deadlock** | Critical | High | Unresolved all-reduce across shared embeddings in PyTorch pipeline ranks. | Training silently diverges or communication times out. |
| **Activation VRAM Exhaustion** | High | High | Unbounded accumulation of hidden states across auxiliary heads. | OOM crash during sequence length increase. |
| **Numerical Instability (NaN)** | High | High | Unclamped logit entropy under FP16/BF16 precision. | Loss yields `NaN`, zeroing adapter gradient updates. |
| **Kernel / Driver Failure** | Medium | High | Triton dependency / CUDA driver version mismatch. | Development halts due to compilation errors. |
| **Scope Creep** | High | Medium | Attempting full MTP blocks + CUDA kernels + distributed routers at once. | Student overwhelmed by infrastructure bugs. |

---

## 4. MVP Roadmap & Fallback Guardrails

- **Phase 1**: Lightweight adapter head (Gated-LoRA) on compact sub-3B model (Qwen 1.5B) in single-GPU environment.
- **Phase 2**: Offline entropy logging & draft correlation analysis (verify entropy correlates with error rate).
- **Phase 3**: Live uncertainty-gated engine integration in vLLM / llama.cpp.
- **Pivot Trigger (Midpoint Guardrail)**: If PyTorch autograd instability or OOM crashes persist by project midpoint $\rightarrow$ **Pivot to Fallback A**: Post-hoc uncertainty profiling on pre-trained native MTP checkpoints (Gemma4/Qwen-MTP) in inference-only mode.

---

## 5. Slide 12 Specification (Key Problems with Standard MTP)

Place these 4 core points directly onto **Slide 12** of your presentation deck:

1. **Fixed Prediction Depth ($k$)**: Current MTP architectures always predict a fixed number of future tokens (e.g., $k=4$) regardless of prompt difficulty. Wastes compute on complex math/logic and too conservative for simple prose.
2. **Hallucination Cascades (Error Propagation)**: If the model makes a mistake on the first guessed token, auxiliary heads blindly try to justify that mistake, locking the AI into an unrecoverable hallucination path.
3. **Draft Acceptance Penalty under Quantization**: When models are compressed to 4-bit or 8-bit to fit on consumer laptops, numerical noise causes prediction heads to make fuzzier guesses, dropping draft acceptance rates.
4. **High Post-Training Overhead**: Upgrading an existing language model to support native MTP usually requires massive pre-training from scratch rather than a lightweight adapter conversion.
