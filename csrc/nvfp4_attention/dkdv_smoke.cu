#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

#if defined(__has_include)
#if __has_include(<cutlass/cutlass.h>) && __has_include(<cute/config.hpp>)
#include <cutlass/cutlass.h>
#include <cute/config.hpp>
#define AXOLOTL_NVFP4_HAS_CUTLASS 1
#endif
#endif

#ifndef AXOLOTL_NVFP4_HAS_CUTLASS
#define AXOLOTL_NVFP4_HAS_CUTLASS 0
#endif

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

__global__ void dkdv_signature_probe_kernel(
    const uint8_t* qnv,
    const uint8_t* qsc,
    const uint8_t* qtnv,
    const uint8_t* qtsc,
    const uint8_t* donv,
    const uint8_t* dosc,
    const uint8_t* dotnv,
    const uint8_t* dotsc,
    const uint8_t* knv,
    const uint8_t* ksc,
    const uint8_t* vnv,
    const uint8_t* vsc,
    const float* bias,
    const float* lse,
    const float* delta,
    float* dk,
    float* dv,
    float scaling,
    int64_t seed,
    int64_t sq,
    int64_t sq_pad,
    int64_t skv,
    int d,
    int h_count,
    int hk,
    int64_t sb_z,
    int64_t sdk_n,
    int64_t sdv_n,
    bool has_bias,
    bool causal,
    bool sr,
    bool sr_p_dv,
    int block_m,
    int block_n
) {
    (void)scaling;
    (void)seed;
    (void)causal;
    (void)sr;
    (void)sr_p_dv;
    (void)block_m;

    if (threadIdx.x != 0) {
        return;
    }

    const int64_t pid_n = static_cast<int64_t>(blockIdx.x);
    const int64_t pid_zh = static_cast<int64_t>(blockIdx.y);
    const int64_t n = pid_n * static_cast<int64_t>(block_n);
    if (n >= skv) {
        return;
    }

    const int64_t z = pid_zh / h_count;
    const int64_t qh = pid_zh % h_count;
    const int64_t groups = h_count / hk;
    const int64_t zhk = z * hk + (qh / groups);
    const int64_t dp2 = d / 2;
    const int64_t dp16 = d / 16;
    const int64_t sq2 = sq_pad / 2;
    const int64_t sq16 = sq_pad / 16;

    const uint8_t dk_mix =
        qnv[pid_zh * sq * dp2] ^
        qsc[pid_zh * sq * dp16] ^
        qtnv[pid_zh * d * sq2] ^
        qtsc[pid_zh * d * sq16] ^
        knv[zhk * skv * dp2 + n * dp2] ^
        ksc[zhk * skv * dp16 + n * dp16];
    const uint8_t dv_mix =
        donv[pid_zh * sq * dp2] ^
        dosc[pid_zh * sq * dp16] ^
        dotnv[pid_zh * d * sq2] ^
        dotsc[pid_zh * d * sq16] ^
        vnv[zhk * skv * dp2 + n * dp2] ^
        vsc[zhk * skv * dp16 + n * dp16];

    float state = lse[pid_zh * sq] + delta[pid_zh * sq];
    if (has_bias) {
        state += bias[z * sb_z + n];
    }

    dk[pid_zh * skv * sdk_n + n * sdk_n] = 1000.0f + static_cast<float>(dk_mix) + state;
    dv[pid_zh * skv * sdv_n + n * sdv_n] = 2000.0f + static_cast<float>(dv_mix) + state;
}

void check_cuda_byte(torch::Tensor const& tensor, char const* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Byte, name, " must be uint8");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_float(torch::Tensor const& tensor, char const* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void dkdv_signature_probe(
    torch::Tensor qnv,
    torch::Tensor qsc,
    torch::Tensor qtnv,
    torch::Tensor qtsc,
    torch::Tensor donv,
    torch::Tensor dosc,
    torch::Tensor dotnv,
    torch::Tensor dotsc,
    torch::Tensor knv,
    torch::Tensor ksc,
    torch::Tensor vnv,
    torch::Tensor vsc,
    torch::Tensor bias,
    torch::Tensor lse,
    torch::Tensor delta,
    torch::Tensor dk,
    torch::Tensor dv,
    double scaling,
    int64_t seed,
    int64_t sq,
    int64_t sq_pad,
    int64_t skv,
    int64_t d,
    int64_t h_count,
    int64_t hk,
    int64_t sb_z,
    int64_t sdk_n,
    int64_t sdv_n,
    bool has_bias,
    bool causal,
    bool sr,
    bool sr_p_dv,
    int64_t block_m,
    int64_t block_n
) {
    check_cuda_byte(qnv, "qnv");
    check_cuda_byte(qsc, "qsc");
    check_cuda_byte(qtnv, "qtnv");
    check_cuda_byte(qtsc, "qtsc");
    check_cuda_byte(donv, "donv");
    check_cuda_byte(dosc, "dosc");
    check_cuda_byte(dotnv, "dotnv");
    check_cuda_byte(dotsc, "dotsc");
    check_cuda_byte(knv, "knv");
    check_cuda_byte(ksc, "ksc");
    check_cuda_byte(vnv, "vnv");
    check_cuda_byte(vsc, "vsc");
    check_cuda_float(lse, "lse");
    check_cuda_float(delta, "delta");
    check_cuda_float(dk, "dk");
    check_cuda_float(dv, "dv");
    if (has_bias) {
        check_cuda_float(bias, "bias");
    }

    TORCH_CHECK(d > 0 && d % 16 == 0, "D must be positive and divisible by 16");
    TORCH_CHECK(h_count > 0 && hk > 0 && h_count % hk == 0, "H must be divisible by HK");
    TORCH_CHECK(block_n > 0 && block_m > 0, "block sizes must be positive");
    TORCH_CHECK(sq > 0 && sq_pad >= sq && skv > 0, "invalid sequence lengths");
    TORCH_CHECK(dk.dim() == 3 && dv.dim() == 3, "dk and dv must be [Z*H, Skv, D]");
    TORCH_CHECK(dk.size(0) == dv.size(0), "dk/dv leading dimensions must match");
    TORCH_CHECK(dk.size(1) >= skv && dv.size(1) >= skv, "dk/dv Skv dimension is too small");
    TORCH_CHECK(dk.size(2) >= d && dv.size(2) >= d, "dk/dv D dimension is too small");
    TORCH_CHECK(dk.size(0) % h_count == 0, "dk leading dimension must be a multiple of H");

    c10::cuda::CUDAGuard device_guard(qnv.device());
    constexpr int threads = 128;
    const int blocks_n = static_cast<int>((skv + block_n - 1) / block_n);
    const int blocks_zh = static_cast<int>(dk.size(0));
    if (blocks_n == 0 || blocks_zh == 0) {
        return;
    }

    dkdv_signature_probe_kernel<<<
        dim3(blocks_n, blocks_zh),
        threads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        qnv.data_ptr<uint8_t>(),
        qsc.data_ptr<uint8_t>(),
        qtnv.data_ptr<uint8_t>(),
        qtsc.data_ptr<uint8_t>(),
        donv.data_ptr<uint8_t>(),
        dosc.data_ptr<uint8_t>(),
        dotnv.data_ptr<uint8_t>(),
        dotsc.data_ptr<uint8_t>(),
        knv.data_ptr<uint8_t>(),
        ksc.data_ptr<uint8_t>(),
        vnv.data_ptr<uint8_t>(),
        vsc.data_ptr<uint8_t>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        lse.data_ptr<float>(),
        delta.data_ptr<float>(),
        dk.data_ptr<float>(),
        dv.data_ptr<float>(),
        static_cast<float>(scaling),
        seed,
        sq,
        sq_pad,
        skv,
        static_cast<int>(d),
        static_cast<int>(h_count),
        static_cast<int>(hk),
        sb_z,
        sdk_n,
        sdv_n,
        has_bias,
        causal,
        sr,
        sr_p_dv,
        static_cast<int>(block_m),
        static_cast<int>(block_n)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool has_cutlass_headers() {
    return AXOLOTL_NVFP4_HAS_CUTLASS != 0;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smoke_add_one", &smoke_add_one);
    m.def("dkdv_signature_probe", &dkdv_signature_probe);
    m.def("has_cutlass_headers", &has_cutlass_headers);
}
