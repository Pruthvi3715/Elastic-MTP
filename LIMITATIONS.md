# Evaluation Boundaries & Limitations

To ensure scientific transparency and publication credibility, this document details the exact evaluation boundaries, hardware setup, and known scope limits of Elastic-MTP.

---

## 1. Scope & Evaluation Boundaries

1. **Backbone Model Scale**:
   - Primary evaluation backbone is standardized on `Qwen2.5-1.5B-Instruct` and the synthetic gradient engine. 
   - Calibration behavior on larger 7B, 14B, and 70B parameter models remains an open direction for future work.

2. **Hardware Environment**:
   - Benchmarks are evaluated in single-GPU / single-node environments.
   - Distributed tensor parallelism and multi-node interconnect overheads are not included in latency profiling.

3. **Multi-Tenant Serving Amortization**:
   - Throughput metrics represent raw token generation latency for individual inference requests. 
   - Serving cost comparisons to commercial multi-tenant API endpoints (e.g., OpenAI/Anthropic) are excluded from formal paper claims, as commercial APIs benefit from continuous batching, multi-tenant amortization, and specialized serving infrastructure.

4. **Task Scope**:
   - Evaluated across 20 structured text generation categories (code, math proofs, technical prose, translations, conversational dialogue).
   - Long-context multi-turn conversational degradation beyond 4k context windows is not evaluated in this study.
