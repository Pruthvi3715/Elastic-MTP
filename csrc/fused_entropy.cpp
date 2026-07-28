/*
 * PyTorch C++ Extension Bindings for Fused Entropy CUDA/CPU Router.
 */

#include <torch/extension.h>

#ifdef WITH_CUDA
torch::Tensor fused_entropy_cuda(torch::Tensor logits);
#endif

torch::Tensor fused_entropy(torch::Tensor logits) {
#ifdef WITH_CUDA
    if (logits.is_cuda()) {
        return fused_entropy_cuda(logits);
    }
#endif
    // CPU fallback path
    auto log_probs = torch::log_softmax(logits, -1);
    auto probs = torch::exp(log_probs);
    auto entropy = -torch::sum(probs * log_probs, -1);
    return torch::clamp(entropy, 0.0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_entropy", &fused_entropy, "Fused Log-Softmax Shannon Entropy (CUDA/CPU)");
}
