"""
Continuous 2-Hour Autonomous AutoResearch Daemon Loop for Elastic-MTP.

Runs continuous inference, harvests hard prompt failures, fine-tunes candidate GLoRA adapters,
validates zero regressions via pytest & DAR metrics, and hot-swaps improved checkpoints atomically.
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference_engine import ElasticMTPInferenceEngine
from src.mtp_glora_adapter import MTPGLoRAModule
from src.auto_research_daemon import AutoResearchManager


PROMPT_POOL = [
    "x^2 + y^2 = z^2 integral calculus theorem proof and math derivation",
    "def complex_graph_search(adj_matrix, start_node, visited_set):",
    "Quantization enables memory-efficient deep learning models on edge devices",
    "Elastic multi-token prediction dynamically adapts draft horizon based on entropy",
    "Continuous self-tuning optimizes speculative decoding acceptance rates",
    "Explain quantum entanglement, superdense coding, and decoherence paradigms",
    "Implement a lock-free concurrent queue in C++ with memory order guarantees",
    "Analyze the asymptotic complexity of dynamic programming matrix chain multiplication",
    "The quick brown fox jumps over the lazy dog repeatedly across multiple sentences",
    "System architecture for autonomous self-improving neural network inference engines"
]


def run_continuous_daemon(duration_seconds: float = 7200.0):
    start_time = time.time()
    end_time = start_time + duration_seconds

    print("=" * 80)
    print(f"Starting 2-Hour Autonomous AutoResearch Loop ({duration_seconds / 3600:.1f} Hours)")
    print(f"Target Completion Time: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    print("=" * 80)

    device = "cpu"
    engine = ElasticMTPInferenceEngine(model_name="synthetic", device=device)
    adapter = MTPGLoRAModule(hidden_dim=128, vocab_size=50257, num_aux_heads=3, rank=8).to(device)

    daemon = AutoResearchManager(
        engine=engine,
        adapter_stack=adapter,
        eval_prompts=PROMPT_POOL[:4],
        min_buffer_size=10,
        dar_improvement_threshold=0.0,
        learning_rate=1e-4,
        checkpoint_dir="checkpoints"
    )

    engine.auto_research = daemon
    engine.adapter_stack = adapter

    cycle_count = 0
    prompts_processed = 0

    while time.time() < end_time:
        cycle_count += 1
        elapsed = time.time() - start_time
        remaining = end_time - time.time()

        print(f"\n--- [Cycle #{cycle_count} | Elapsed: {elapsed/60:.1f}m | Remaining: {remaining/60:.1f}m] ---")

        # Pick prompt from pool
        prompt = PROMPT_POOL[prompts_processed % len(PROMPT_POOL)]
        prompts_processed += 1

        print(f"Running live inference on prompt [{prompts_processed}]: '{prompt[:60]}...'")
        res = engine.generate(prompt, max_new_tokens=40, mode="elastic")

        buffer_len = len(daemon.telemetry_buffer)
        print(f"  -> Generated {res['tokens_generated']} tokens | Buffer Size: {buffer_len}/{daemon.min_buffer_size}")

        if buffer_len >= daemon.min_buffer_size and not daemon.is_training:
            print("  -> Buffer capacity reached! Triggering self-improvement optimization cycle...")
            daemon.run_self_improvement_cycle()
            print(f"  -> Current Best DAR: {daemon.best_dar:.2f}%")

        time.sleep(5.0)

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Completed 2-Hour AutoResearch Self-Improvement Loop!")
    print(f"Total Cycles: {cycle_count} | Prompts Processed: {prompts_processed}")
    print(f"Final Best DAR Achieved: {daemon.best_dar:.2f}%")
    print(f"Promoted Checkpoint Saved: checkpoints/auto_tuned_glora_best.pt")
    print("=" * 80)


if __name__ == "__main__":
    # Target duration: 2 hours = 7200 seconds
    target_duration = 7200.0
    if len(sys.argv) > 1:
        try:
            target_duration = float(sys.argv[1])
        except ValueError:
            pass
    run_continuous_daemon(target_duration)
