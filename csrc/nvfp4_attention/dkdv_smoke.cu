#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

__global__ void add_one_kernel(const float* in, float* out, int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + 1.0f;
    }
}

torch::Tensor smoke_add_one(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Float, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    const int64_t n = input.numel();
    if (n == 0) {
        return output;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    add_one_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smoke_add_one", &smoke_add_one);
}
