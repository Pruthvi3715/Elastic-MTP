"""
Unit tests for vLLM & CUDA Enterprise Engine Plugin Module.
"""

import torch
import pytest
from src.vllm_elastic_plugin import (
    FusedCUDAEntropyRouter,
    ElasticvLLMBlockAllocator,
    ElasticvLLMServingEngine,
    vLLMBatchRequest,
)


def test_fused_cuda_router_execution():
    cuda_router = FusedCUDAEntropyRouter(tau_high=5.0, tau_low=2.5)
    logits = torch.randn(1, 1000)

    entropy, k_val = cuda_router.forward_cuda_kernel(logits)

    assert isinstance(entropy, torch.Tensor)
    assert isinstance(k_val, torch.Tensor)
    assert k_val.item() in [1, 4, 8]


def test_vllm_block_allocator_memory_management():
    allocator = ElasticvLLMBlockAllocator(num_blocks=100, block_size=16)

    # Allocate blocks for request
    blocks = allocator.allocate("req_1", num_tokens=32)
    assert len(blocks) == 2
    assert len(allocator.free_blocks) == 98

    # Free blocks
    allocator.free("req_1")
    assert len(allocator.free_blocks) == 100


def test_vllm_continuous_batching_engine():
    engine = ElasticvLLMServingEngine(max_concurrency=256)
    requests = [
        vLLMBatchRequest(request_id=f"req_{i}", prompt_tokens=torch.tensor([1, 2]), allocated_k=4, kv_block_ids=[i])
        for i in range(16)
    ]

    batch_res = engine.process_continuous_batch(requests)

    assert batch_res["active_batch_size"] == 16
    assert batch_res["total_tokens_generated"] > 0
    assert batch_res["cuda_kernel_latency_ms"] >= 0.0
    assert batch_res["vram_memory_saved_pct"] == 75.0
