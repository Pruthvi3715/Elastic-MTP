/*
 * Fused Log-Softmax Shannon Entropy CUDA Kernel for Elastic-MTP.
 * Calculates max-shifted log-softmax and Shannon Entropy reduction
 * across vocab dimensions in warp-synchronous register memory.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>

template <typename scalar_t>
__global__ void fused_entropy_cuda_kernel(
    const scalar_t* __restrict__ logits,
    scalar_t* __restrict__ entropy_out,
    const int batch_size,
    const int vocab_size
) {
    int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;

    const scalar_t* batch_logits = logits + batch_idx * vocab_size;

    // Step 1: Find max logit for numerical stability
    scalar_t max_val = -1e20;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        scalar_t val = batch_logits[v];
        if (val > max_val) max_val = val;
    }

    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(0xffffffff, max_val, offset);
        if (other > max_val) max_val = other;
    }

    // Broadcast max_val
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // Step 2: Sum exp(logits - max_val) and compute entropy numerator
    scalar_t sum_exp = 0.0;
    scalar_t sum_x_exp = 0.0;

    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        scalar_t x = batch_logits[v] - max_val;
        scalar_t exp_x = expf(x);
        sum_exp += exp_x;
        sum_x_exp += x * exp_x;
    }

    // Warp reductions
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        sum_x_exp += __shfl_down_sync(0xffffffff, sum_x_exp, offset);
    }

    // Thread 0 computes final Shannon entropy for this sequence
    if (threadIdx.x == 0) {
        scalar_t log_sum_exp = logf(sum_exp + 1e-10f);
        // H(P) = log(sum_exp) - (sum_x_exp / sum_exp)
        scalar_t entropy = log_sum_exp - (sum_x_exp / (sum_exp + 1e-10f));
        entropy_out[batch_idx] = (entropy > 0.0f) ? entropy : 0.0f;
    }
}

torch::Tensor fused_entropy_cuda(torch::Tensor logits) {
    auto batch_size = logits.size(0);
    auto vocab_size = logits.size(1);
    auto options = torch::TensorOptions().dtype(logits.dtype()).device(logits.device());
    auto entropy_out = torch::zeros({batch_size}, options);

    dim3 threads(32, 4);
    dim3 blocks((batch_size + 3) / 4);

    AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "fused_entropy_cuda_kernel", ([&] {
        fused_entropy_cuda_kernel<scalar_t><<<blocks, threads>>>(
            logits.data_ptr<scalar_t>(),
            entropy_out.data_ptr<scalar_t>(),
            batch_size,
            vocab_size
        );
    }));

    return entropy_out;
}
