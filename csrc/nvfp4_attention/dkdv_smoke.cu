#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

#if defined(__has_include)
#if __has_include(<cutlass/cutlass.h>) && __has_include(<cute/config.hpp>)
#include <cutlass/cutlass.h>
#include <cute/config.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>
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

#if AXOLOTL_NVFP4_HAS_CUTLASS
__global__ void fp4_mma_compile_capability_kernel(uint8_t* out) {
    if (threadIdx.x == 0) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(CUTLASS_ARCH_MMA_SM120A_ENABLED)
        out[0] = 1;
#else
        out[0] = 0;
#endif
    }
}

__global__ void fp4_mma_microbench_kernel(float* out, int64_t iterations) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && \
    (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(CUTLASS_ARCH_MMA_SM120A_ENABLED))
    using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
        cute::float_e2m1_t,
        cute::float_e2m1_t,
        float,
        cute::float_ue4m3_t,
        16>;

    const uint32_t lane = static_cast<uint32_t>(threadIdx.x & 31);
    const uint32_t tile = static_cast<uint32_t>(blockIdx.x);
    const uint32_t seed = 0x9e3779b9u * (tile + 1u) ^ (lane * 0x7f4a7c15u);
    const uint32_t a0 = seed ^ 0x11111111u;
    const uint32_t a1 = seed ^ 0x22222222u;
    const uint32_t a2 = seed ^ 0x44444444u;
    const uint32_t a3 = seed ^ 0x88888888u;
    const uint32_t b0 = seed ^ 0x13579bdfu;
    const uint32_t b1 = seed ^ 0x2468ace0u;
    const uint32_t sf = 0x38383838u;

    float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    for (int64_t i = 0; i < iterations; ++i) {
        float d0;
        float d1;
        float d2;
        float d3;
        Mma::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sf, sf);
        c0 = d0;
        c1 = d1;
        c2 = d2;
        c3 = d3;
    }
    out[static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x] =
        c0 + c1 + c2 + c3;
#else
    if (threadIdx.x == 0) {
        out[static_cast<int64_t>(blockIdx.x) * blockDim.x] = 0.0f;
    }
#endif
}
#endif

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

bool sm120_mxf4nvf4_ue4m3_available() {
#if AXOLOTL_NVFP4_HAS_CUTLASS
    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 12) {
        return false;
    }
    auto flag = torch::empty(
        {1},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8)
    );
    fp4_mma_compile_capability_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        flag.data_ptr<uint8_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto host_flag = flag.cpu();
    return host_flag.item<uint8_t>() != 0;
#else
    return false;
#endif
}

torch::Tensor fp4_mma_microbench(int64_t tiles, int64_t iterations) {
    TORCH_CHECK(tiles > 0, "tiles must be positive");
    TORCH_CHECK(iterations > 0, "iterations must be positive");
    TORCH_CHECK(
        sm120_mxf4nvf4_ue4m3_available(),
        "SM120 mxf4nvf4 ue4m3 MMA requires CUTLASS/CuTe headers and a compute capability 12.x CUDA device"
    );

    auto output = torch::empty(
        {tiles, 32},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)
    );
#if AXOLOTL_NVFP4_HAS_CUTLASS
    fp4_mma_microbench_kernel<<<
        static_cast<unsigned int>(tiles),
        32,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(output.data_ptr<float>(), iterations);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
#else
    return output;
#endif
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

__device__ __forceinline__ float decode_e2m1(uint8_t nibble) {
    constexpr float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return values[nibble & 0x0f];
}

__device__ __forceinline__ float decode_e4m3fn(uint8_t byte) {
    const int sign = byte & 0x80;
    const int exp = (byte >> 3) & 0x0f;
    const int mant = byte & 0x07;
    if (exp == 0) {
        float value = static_cast<float>(mant) * 0.001953125f;
        return sign ? -value : value;
    }
    if (exp == 0x0f && mant == 0x07) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float value = ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
    return sign ? -value : value;
}

__device__ __forceinline__ float load_nvfp4_d(
    const uint8_t* q,
    const uint8_t* s,
    int64_t row_base_q,
    int64_t row_base_s,
    int d
) {
    const uint8_t packed = q[row_base_q + d / 2];
    const uint8_t nibble = (d & 1) ? (packed >> 4) : (packed & 0x0f);
    const float scale = decode_e4m3fn(s[row_base_s + d / 16]);
    return decode_e2m1(nibble) * scale;
}

__device__ __forceinline__ uint8_t encode_e4m3fn_positive(float x) {
    x = fminf(fmaxf(x, 1.5258789e-05f), 448.0f);
    if (x < 0.015625f) {
        int mant = static_cast<int>(rintf(x * 512.0f));
        mant = max(1, min(7, mant));
        return static_cast<uint8_t>(mant);
    }

    int exp2 = 0;
    float frac = frexpf(x, &exp2);
    exp2 -= 1;
    int exp = exp2 + 7;
    float norm = ldexpf(frac, 1);
    int mant = static_cast<int>(rintf((norm - 1.0f) * 8.0f));
    if (mant == 8) {
        mant = 0;
        ++exp;
    }
    if (exp >= 15) {
        exp = 15;
        mant = min(mant, 6);
    }
    exp = max(1, exp);
    return static_cast<uint8_t>((exp << 3) | (mant & 0x07));
}

__device__ __forceinline__ uint8_t quantize_e2m1_rtn(float x) {
    constexpr float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    x = fminf(fmaxf(x, -6.0f), 6.0f);
    uint8_t best = 0;
    float best_err = fabsf(x - values[0]);
    for (uint8_t i = 1; i < 16; ++i) {
        const float err = fabsf(x - values[i]);
        if (err < best_err) {
            best = i;
            best_err = err;
        }
    }
    return best;
}

__device__ __forceinline__ uint8_t load_nibble(
    const uint8_t* q,
    int64_t row_base_q,
    int k
) {
    const uint8_t packed = q[row_base_q + k / 2];
    return (k & 1) ? (packed >> 4) : (packed & 0x0f);
}

__device__ __forceinline__ uint32_t pack8_nibbles_from_nvfp4(
    const uint8_t* q,
    int64_t row_base_q,
    int k0
) {
    if ((k0 & 1) == 0) {
        const int64_t byte0 = row_base_q + k0 / 2;
        return static_cast<uint32_t>(q[byte0]) |
               (static_cast<uint32_t>(q[byte0 + 1]) << 8) |
               (static_cast<uint32_t>(q[byte0 + 2]) << 16) |
               (static_cast<uint32_t>(q[byte0 + 3]) << 24);
    }
    uint32_t out = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out |= static_cast<uint32_t>(load_nibble(q, row_base_q, k0 + i)) << (4 * i);
    }
    return out;
}

__device__ __forceinline__ uint32_t pack4_scale_bytes(
    const uint8_t* s,
    int64_t row_base_s,
    int group0
) {
    return static_cast<uint32_t>(s[row_base_s + group0]) |
           (static_cast<uint32_t>(s[row_base_s + group0 + 1]) << 8) |
           (static_cast<uint32_t>(s[row_base_s + group0 + 2]) << 16) |
           (static_cast<uint32_t>(s[row_base_s + group0 + 3]) << 24);
}

__device__ __forceinline__ uint32_t pack8_nibbles_from_tile(
    const float* tile,
    const float* inv_scale,
    int row,
    int k0
) {
    uint32_t out = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int k = k0 + i;
        const uint8_t nibble = quantize_e2m1_rtn(tile[row * 64 + k] * inv_scale[row * 4 + k / 16]);
        out |= static_cast<uint32_t>(nibble) << (4 * i);
    }
    return out;
}

#if AXOLOTL_NVFP4_HAS_CUTLASS
__launch_bounds__(1024, 1)
__global__ void dkdv_mma_rtn_kernel(
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
    int64_t sq,
    int64_t sq_pad,
    int64_t skv,
    int h_count,
    int hk,
    int64_t sb_z,
    int64_t sdk_n,
    int64_t sdv_n,
    bool has_bias,
    bool causal
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && \
    (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(CUTLASS_ARCH_MMA_SM120A_ENABLED))
    using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
        cute::float_e2m1_t,
        cute::float_e2m1_t,
        float,
        cute::float_ue4m3_t,
        16>;

    __shared__ float score_tile[16 * 64];
    __shared__ float dp_tile[16 * 64];
    __shared__ float p_inv_scale[16 * 4];
    __shared__ float ds_inv_scale[16 * 4];
    __shared__ uint8_t p_scale[16 * 4];
    __shared__ uint8_t ds_scale[16 * 4];
    __shared__ uint32_t p_pack[32 * 4];
    __shared__ uint32_t ds_pack[32 * 4];
    __shared__ uint32_t p_scale_pack[32];
    __shared__ uint32_t ds_scale_pack[32];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int lane_r = lane >> 2;
    const int lane_kb = (lane & 3) * 8;
    const int lane_n2 = (lane & 3) * 2;
    const int n_base = static_cast<int>(blockIdx.x) * 16;
    const int pid_zh = static_cast<int>(blockIdx.z);
    const int z = pid_zh / h_count;
    const int qh = pid_zh % h_count;
    const int groups = h_count / hk;
    const int zhk = z * hk + (qh / groups);
    constexpr int d_count = 256;
    constexpr int dp2 = d_count / 2;
    constexpr int dp16 = d_count / 16;
    const int64_t sq2 = sq_pad / 2;
    const int64_t sq16 = sq_pad / 16;

    const int d_tile = warp * 8;
    float dv0 = 0.0f, dv1 = 0.0f, dv2 = 0.0f, dv3 = 0.0f;
    float dk0 = 0.0f, dk1 = 0.0f, dk2 = 0.0f, dk3 = 0.0f;

    int64_t lo = 0;
    if (causal) {
        const int64_t shift = skv - sq;
        if (n_base > shift) {
            lo = ((static_cast<int64_t>(n_base) - shift) / 64) * 64;
        }
    }

    for (int64_t start_m = lo; start_m < sq; start_m += 64) {
        if (warp < 8) {
            const int m_sub = warp * 8;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;

            #pragma unroll
            for (int d0 = 0; d0 < d_count; d0 += 64) {
                const int64_t k_row0_q = (static_cast<int64_t>(zhk) * skv + n_base + lane_r) * dp2;
                const int64_t k_row1_q = (static_cast<int64_t>(zhk) * skv + n_base + lane_r + 8) * dp2;
                const int64_t k_row_s = (static_cast<int64_t>(zhk) * skv + n_base) * dp16;
                const int64_t q_row_q = static_cast<int64_t>(pid_zh) * sq * dp2;
                const int64_t q_row_s = static_cast<int64_t>(pid_zh) * sq * dp16;

                uint32_t ka0 = 0, ka1 = 0, ka2 = 0, ka3 = 0;
                uint32_t va0 = 0, va1 = 0, va2 = 0, va3 = 0;
                uint32_t ksf = 0x38383838u;
                uint32_t vsf = 0x38383838u;
                const int sfa_m = lane_r + ((lane & 1) ? 8 : 0);
                if (n_base + lane_r < skv) {
                    ka0 = pack8_nibbles_from_nvfp4(knv, k_row0_q, d0 + lane_kb);
                    ka2 = pack8_nibbles_from_nvfp4(knv, k_row0_q, d0 + 32 + lane_kb);
                    va0 = pack8_nibbles_from_nvfp4(vnv, k_row0_q, d0 + lane_kb);
                    va2 = pack8_nibbles_from_nvfp4(vnv, k_row0_q, d0 + 32 + lane_kb);
                }
                if (n_base + lane_r + 8 < skv) {
                    ka1 = pack8_nibbles_from_nvfp4(knv, k_row1_q, d0 + lane_kb);
                    ka3 = pack8_nibbles_from_nvfp4(knv, k_row1_q, d0 + 32 + lane_kb);
                    va1 = pack8_nibbles_from_nvfp4(vnv, k_row1_q, d0 + lane_kb);
                    va3 = pack8_nibbles_from_nvfp4(vnv, k_row1_q, d0 + 32 + lane_kb);
                }
                if (n_base + sfa_m < skv) {
                    const int64_t srow = k_row_s + static_cast<int64_t>(sfa_m) * dp16;
                    ksf = pack4_scale_bytes(ksc, srow, d0 / 16);
                    vsf = pack4_scale_bytes(vsc, srow, d0 / 16);
                }

                const int b_row = static_cast<int>(start_m) + m_sub + lane_r;
                uint32_t qb0 = 0, qb1 = 0, dob0 = 0, dob1 = 0;
                uint32_t qsf = 0x38383838u;
                uint32_t dosf = 0x38383838u;
                if (b_row < sq) {
                    const int64_t brow_q = q_row_q + static_cast<int64_t>(b_row) * dp2;
                    const int64_t brow_s = q_row_s + static_cast<int64_t>(b_row) * dp16;
                    qb0 = pack8_nibbles_from_nvfp4(qnv, brow_q, d0 + lane_kb);
                    qb1 = pack8_nibbles_from_nvfp4(qnv, brow_q, d0 + 32 + lane_kb);
                    dob0 = pack8_nibbles_from_nvfp4(donv, brow_q, d0 + lane_kb);
                    dob1 = pack8_nibbles_from_nvfp4(donv, brow_q, d0 + 32 + lane_kb);
                    qsf = pack4_scale_bytes(qsc, brow_s, d0 / 16);
                    dosf = pack4_scale_bytes(dosc, brow_s, d0 / 16);
                }

                float ns0, ns1, ns2, ns3;
                float np0, np1, np2, np3;
                Mma::fma(ns0, ns1, ns2, ns3, ka0, ka1, ka2, ka3, qb0, qb1, s0, s1, s2, s3, ksf, qsf);
                Mma::fma(np0, np1, np2, np3, va0, va1, va2, va3, dob0, dob1, p0, p1, p2, p3, vsf, dosf);
                s0 = ns0; s1 = ns1; s2 = ns2; s3 = ns3;
                p0 = np0; p1 = np1; p2 = np2; p3 = np3;
            }

            const int r0 = lane_r;
            const int r1 = lane_r + 8;
            const int c0 = m_sub + lane_n2;
            const int c1 = c0 + 1;
            score_tile[r0 * 64 + c0] = s0;
            score_tile[r0 * 64 + c1] = s1;
            score_tile[r1 * 64 + c0] = s2;
            score_tile[r1 * 64 + c1] = s3;
            dp_tile[r0 * 64 + c0] = p0;
            dp_tile[r0 * 64 + c1] = p1;
            dp_tile[r1 * 64 + c0] = p2;
            dp_tile[r1 * 64 + c1] = p3;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < 16 * 64; idx += blockDim.x) {
            const int n = idx / 64;
            const int m = idx - n * 64;
            const int64_t n_abs = static_cast<int64_t>(n_base + n);
            const int64_t m_abs = start_m + m;
            bool valid = n_abs < skv && m_abs < sq;
            float s = score_tile[idx] * scaling;
            if (has_bias && n_abs < skv) {
                s += bias[static_cast<int64_t>(z) * sb_z + n_abs];
            }
            if (causal && n_abs > (m_abs + (skv - sq))) {
                valid = false;
            }
            float p = 0.0f;
            float ds = 0.0f;
            if (valid) {
                p = expf(s - lse[static_cast<int64_t>(pid_zh) * sq + m_abs]);
                ds = p * (dp_tile[idx] - delta[static_cast<int64_t>(pid_zh) * sq + m_abs]) * scaling;
            }
            score_tile[idx] = p;
            dp_tile[idx] = ds;
        }
        __syncthreads();

        if (threadIdx.x < 16 * 4) {
            const int n = threadIdx.x / 4;
            const int g = threadIdx.x - n * 4;
            float p_amax = 0.0f;
            float ds_amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                p_amax = fmaxf(p_amax, fabsf(score_tile[n * 64 + g * 16 + i]));
                ds_amax = fmaxf(ds_amax, fabsf(dp_tile[n * 64 + g * 16 + i]));
            }
            const uint8_t ps = encode_e4m3fn_positive(p_amax / 6.0f);
            const uint8_t dss = encode_e4m3fn_positive(ds_amax / 6.0f);
            p_scale[threadIdx.x] = ps;
            ds_scale[threadIdx.x] = dss;
            p_inv_scale[threadIdx.x] = 1.0f / decode_e4m3fn(ps);
            ds_inv_scale[threadIdx.x] = 1.0f / decode_e4m3fn(dss);
        }
        __syncthreads();

        if (threadIdx.x < 32) {
            const uint32_t pa0 = pack8_nibbles_from_tile(score_tile, p_inv_scale, lane_r, lane_kb);
            const uint32_t pa1 = pack8_nibbles_from_tile(score_tile, p_inv_scale, lane_r + 8, lane_kb);
            const uint32_t pa2 = pack8_nibbles_from_tile(score_tile, p_inv_scale, lane_r, 32 + lane_kb);
            const uint32_t pa3 = pack8_nibbles_from_tile(score_tile, p_inv_scale, lane_r + 8, 32 + lane_kb);
            const uint32_t dsa0 = pack8_nibbles_from_tile(dp_tile, ds_inv_scale, lane_r, lane_kb);
            const uint32_t dsa1 = pack8_nibbles_from_tile(dp_tile, ds_inv_scale, lane_r + 8, lane_kb);
            const uint32_t dsa2 = pack8_nibbles_from_tile(dp_tile, ds_inv_scale, lane_r, 32 + lane_kb);
            const uint32_t dsa3 = pack8_nibbles_from_tile(dp_tile, ds_inv_scale, lane_r + 8, 32 + lane_kb);
            const int sfa_m = lane_r + ((lane & 1) ? 8 : 0);
            p_pack[lane * 4 + 0] = pa0;
            p_pack[lane * 4 + 1] = pa1;
            p_pack[lane * 4 + 2] = pa2;
            p_pack[lane * 4 + 3] = pa3;
            ds_pack[lane * 4 + 0] = dsa0;
            ds_pack[lane * 4 + 1] = dsa1;
            ds_pack[lane * 4 + 2] = dsa2;
            ds_pack[lane * 4 + 3] = dsa3;
            p_scale_pack[lane] =
                static_cast<uint32_t>(p_scale[sfa_m * 4 + 0]) |
                (static_cast<uint32_t>(p_scale[sfa_m * 4 + 1]) << 8) |
                (static_cast<uint32_t>(p_scale[sfa_m * 4 + 2]) << 16) |
                (static_cast<uint32_t>(p_scale[sfa_m * 4 + 3]) << 24);
            ds_scale_pack[lane] =
                static_cast<uint32_t>(ds_scale[sfa_m * 4 + 0]) |
                (static_cast<uint32_t>(ds_scale[sfa_m * 4 + 1]) << 8) |
                (static_cast<uint32_t>(ds_scale[sfa_m * 4 + 2]) << 16) |
                (static_cast<uint32_t>(ds_scale[sfa_m * 4 + 3]) << 24);
        }
        __syncthreads();

        if (d_tile < d_count) {
            const uint32_t pa0 = p_pack[lane * 4 + 0];
            const uint32_t pa1 = p_pack[lane * 4 + 1];
            const uint32_t pa2 = p_pack[lane * 4 + 2];
            const uint32_t pa3 = p_pack[lane * 4 + 3];
            const uint32_t dsa0 = ds_pack[lane * 4 + 0];
            const uint32_t dsa1 = ds_pack[lane * 4 + 1];
            const uint32_t dsa2 = ds_pack[lane * 4 + 2];
            const uint32_t dsa3 = ds_pack[lane * 4 + 3];
            const uint32_t psf = p_scale_pack[lane];
            const uint32_t dssf = ds_scale_pack[lane];

            const int d_row = d_tile + lane_r;
            const int64_t dot_base = static_cast<int64_t>(pid_zh) * d_count * sq2 + static_cast<int64_t>(d_row) * sq2;
            const int64_t qti_base = static_cast<int64_t>(pid_zh) * d_count * sq2 + static_cast<int64_t>(d_row) * sq2;
            const uint32_t dob0 = pack8_nibbles_from_nvfp4(dotnv, dot_base, static_cast<int>(start_m) + lane_kb);
            const uint32_t dob1 = pack8_nibbles_from_nvfp4(dotnv, dot_base, static_cast<int>(start_m) + 32 + lane_kb);
            const uint32_t qb0 = pack8_nibbles_from_nvfp4(qtnv, qti_base, static_cast<int>(start_m) + lane_kb);
            const uint32_t qb1 = pack8_nibbles_from_nvfp4(qtnv, qti_base, static_cast<int>(start_m) + 32 + lane_kb);
            const int64_t dots_base = static_cast<int64_t>(pid_zh) * d_count * sq16 + static_cast<int64_t>(d_row) * sq16;
            const uint32_t dosf = pack4_scale_bytes(dotsc, dots_base, static_cast<int>(start_m / 16));
            const uint32_t qsf = pack4_scale_bytes(qtsc, dots_base, static_cast<int>(start_m / 16));

            float ndv0, ndv1, ndv2, ndv3;
            float ndk0, ndk1, ndk2, ndk3;
            Mma::fma(ndv0, ndv1, ndv2, ndv3, pa0, pa1, pa2, pa3, dob0, dob1, dv0, dv1, dv2, dv3, psf, dosf);
            Mma::fma(ndk0, ndk1, ndk2, ndk3, dsa0, dsa1, dsa2, dsa3, qb0, qb1, dk0, dk1, dk2, dk3, dssf, qsf);
            dv0 = ndv0; dv1 = ndv1; dv2 = ndv2; dv3 = ndv3;
            dk0 = ndk0; dk1 = ndk1; dk2 = ndk2; dk3 = ndk3;
        }
        __syncthreads();
    }

    if (d_tile < d_count) {
        const int n0 = n_base + lane_r;
        const int n1 = n_base + lane_r + 8;
        const int d0 = d_tile + lane_n2;
        const int d1 = d0 + 1;
        if (n0 < skv && d1 < d_count) {
            const int64_t off = static_cast<int64_t>(pid_zh) * skv * sdk_n + static_cast<int64_t>(n0) * sdk_n;
            dk[off + d0] = dk0;
            dk[off + d1] = dk1;
            const int64_t voff = static_cast<int64_t>(pid_zh) * skv * sdv_n + static_cast<int64_t>(n0) * sdv_n;
            dv[voff + d0] = dv0;
            dv[voff + d1] = dv1;
        }
        if (n1 < skv && d1 < d_count) {
            const int64_t off = static_cast<int64_t>(pid_zh) * skv * sdk_n + static_cast<int64_t>(n1) * sdk_n;
            dk[off + d0] = dk2;
            dk[off + d1] = dk3;
            const int64_t voff = static_cast<int64_t>(pid_zh) * skv * sdv_n + static_cast<int64_t>(n1) * sdv_n;
            dv[voff + d0] = dv2;
            dv[voff + d1] = dv3;
        }
    }
#endif
}
#endif

__global__ void dkdv_reference_kernel(
    const uint8_t* qnv,
    const uint8_t* qsc,
    const uint8_t* donv,
    const uint8_t* dosc,
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
    int64_t sq,
    int64_t skv,
    int d_count,
    int h_count,
    int hk,
    int64_t sb_z,
    int64_t sdk_n,
    int64_t sdv_n,
    bool has_bias,
    bool causal
) {
    const int64_t n = static_cast<int64_t>(blockIdx.x);
    const int64_t pid_zh = static_cast<int64_t>(blockIdx.y);
    const int d = threadIdx.x;
    if (n >= skv || d >= d_count) {
        return;
    }

    const int64_t z = pid_zh / h_count;
    const int64_t qh = pid_zh % h_count;
    const int64_t groups = h_count / hk;
    const int64_t zhk = z * hk + (qh / groups);
    const int64_t dp2 = d_count / 2;
    const int64_t dp16 = d_count / 16;

    extern __shared__ float scratch[];
    float* score_partials = scratch;
    float* dp_partials = scratch + blockDim.x;

    const int64_t q_head_base_q = pid_zh * (sq * dp2);
    const int64_t q_head_base_s = pid_zh * (sq * dp16);
    const int64_t kv_head_base_q = zhk * (skv * dp2);
    const int64_t kv_head_base_s = zhk * (skv * dp16);
    const int64_t do_head_base_q = q_head_base_q;
    const int64_t do_head_base_s = q_head_base_s;
    const int64_t kn_row_q = kv_head_base_q + n * dp2;
    const int64_t kn_row_s = kv_head_base_s + n * dp16;
    const int64_t vn_row_q = kv_head_base_q + n * dp2;
    const int64_t vn_row_s = kv_head_base_s + n * dp16;

    const float kd = load_nvfp4_d(knv, ksc, kn_row_q, kn_row_s, d);
    const float vd = load_nvfp4_d(vnv, vsc, vn_row_q, vn_row_s, d);
    float dk_acc = 0.0f;
    float dv_acc = 0.0f;

    for (int64_t m = 0; m < sq; ++m) {
        const int64_t qm_row_q = q_head_base_q + m * dp2;
        const int64_t qm_row_s = q_head_base_s + m * dp16;
        const int64_t dom_row_q = do_head_base_q + m * dp2;
        const int64_t dom_row_s = do_head_base_s + m * dp16;
        const float qd = load_nvfp4_d(qnv, qsc, qm_row_q, qm_row_s, d);
        const float dod = load_nvfp4_d(donv, dosc, dom_row_q, dom_row_s, d);

        score_partials[d] = qd * kd;
        dp_partials[d] = dod * vd;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (d < stride) {
                score_partials[d] += score_partials[d + stride];
                dp_partials[d] += dp_partials[d + stride];
            }
            __syncthreads();
        }

        float p = 0.0f;
        float ds = 0.0f;
        if (d == 0) {
            float score = score_partials[0] * scaling;
            bool valid = true;
            if (has_bias) {
                score += bias[z * sb_z + n];
            }
            if (causal && n > (m + (skv - sq))) {
                valid = false;
            }
            if (valid) {
                p = expf(score - lse[pid_zh * sq + m]);
                ds = p * (dp_partials[0] - delta[pid_zh * sq + m]) * scaling;
            }
            score_partials[0] = p;
            dp_partials[0] = ds;
        }
        __syncthreads();

        p = score_partials[0];
        ds = dp_partials[0];
        dv_acc += p * dod;
        dk_acc += ds * qd;
        __syncthreads();
    }

    dk[pid_zh * (skv * sdk_n) + n * sdk_n + d] = dk_acc;
    dv[pid_zh * (skv * sdv_n) + n * sdv_n + d] = dv_acc;
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

void dkdv_reference(
    torch::Tensor qnv,
    torch::Tensor qsc,
    torch::Tensor donv,
    torch::Tensor dosc,
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
    int64_t sq,
    int64_t skv,
    int64_t d,
    int64_t h_count,
    int64_t hk,
    int64_t sb_z,
    int64_t sdk_n,
    int64_t sdv_n,
    bool has_bias,
    bool causal
) {
    check_cuda_byte(qnv, "qnv");
    check_cuda_byte(qsc, "qsc");
    check_cuda_byte(donv, "donv");
    check_cuda_byte(dosc, "dosc");
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
    TORCH_CHECK(d == 128 || d == 256, "reference dK/dV kernel supports D in {128, 256}");
    TORCH_CHECK(qnv.dim() == 3 && donv.dim() == 3, "qnv/donv must be [Z*H, Sq, D/2]");
    TORCH_CHECK(knv.dim() == 3 && vnv.dim() == 3, "knv/vnv must be [Z*HK, Skv, D/2]");
    TORCH_CHECK(dk.dim() == 3 && dv.dim() == 3, "dk and dv must be [Z*H, Skv, D]");
    TORCH_CHECK(qnv.size(0) == dk.size(0), "qnv and dk leading dimensions must match");
    TORCH_CHECK(qnv.size(1) >= sq && donv.size(1) >= sq, "Sq dimension is too small");
    TORCH_CHECK(knv.size(1) >= skv && vnv.size(1) >= skv, "Skv dimension is too small");
    TORCH_CHECK(dk.size(1) >= skv && dv.size(1) >= skv, "dk/dv Skv dimension is too small");
    TORCH_CHECK(dk.size(2) >= d && dv.size(2) >= d, "dk/dv D dimension is too small");
    TORCH_CHECK(h_count > 0 && hk > 0 && h_count % hk == 0, "H must be divisible by HK");

    c10::cuda::CUDAGuard device_guard(qnv.device());
    const dim3 grid(static_cast<unsigned int>(skv), static_cast<unsigned int>(dk.size(0)));
    const int threads = static_cast<int>(d);
    const size_t smem = static_cast<size_t>(threads) * 2 * sizeof(float);
    dkdv_reference_kernel<<<grid, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
        qnv.data_ptr<uint8_t>(),
        qsc.data_ptr<uint8_t>(),
        donv.data_ptr<uint8_t>(),
        dosc.data_ptr<uint8_t>(),
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
        sq,
        skv,
        static_cast<int>(d),
        static_cast<int>(h_count),
        static_cast<int>(hk),
        sb_z,
        sdk_n,
        sdv_n,
        has_bias,
        causal
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dkdv_mma_rtn(
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
    bool causal
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
    TORCH_CHECK(d == 256, "MMA dK/dV kernel currently supports D=256 only");
    TORCH_CHECK(sq > 0 && sq_pad >= sq && sq_pad % 64 == 0, "Sq_pad must be a multiple of 64");
    TORCH_CHECK(skv > 0, "Skv must be positive");
    TORCH_CHECK(h_count > 0 && hk > 0 && h_count % hk == 0, "H must be divisible by HK");
    TORCH_CHECK(dk.dim() == 3 && dv.dim() == 3, "dk and dv must be [Z*H, Skv, D]");
    TORCH_CHECK(dk.size(0) == qnv.size(0), "dk leading dimension must match qnv");
    TORCH_CHECK(dk.size(1) >= skv && dv.size(1) >= skv, "dk/dv Skv dimension is too small");
    TORCH_CHECK(dk.size(2) >= d && dv.size(2) >= d, "dk/dv D dimension is too small");
    TORCH_CHECK(
        sm120_mxf4nvf4_ue4m3_available(),
        "SM120 mxf4nvf4 ue4m3 MMA requires CUTLASS/CuTe headers and a compute capability 12.x CUDA device"
    );

    c10::cuda::CUDAGuard device_guard(qnv.device());
#if AXOLOTL_NVFP4_HAS_CUTLASS
    const dim3 grid(
        static_cast<unsigned int>((skv + 15) / 16),
        1,
        static_cast<unsigned int>(dk.size(0))
    );
    if (grid.x == 0 || grid.z == 0) {
        return;
    }
        dkdv_mma_rtn_kernel<<<grid, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
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
        sq,
        sq_pad,
        skv,
        static_cast<int>(h_count),
        static_cast<int>(hk),
        sb_z,
        sdk_n,
        sdv_n,
        has_bias,
        causal
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

bool has_cutlass_headers() {
    return AXOLOTL_NVFP4_HAS_CUTLASS != 0;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smoke_add_one", &smoke_add_one);
    m.def("dkdv_signature_probe", &dkdv_signature_probe);
    m.def("dkdv_reference", &dkdv_reference);
    m.def("dkdv_mma_rtn", &dkdv_mma_rtn);
    m.def("has_cutlass_headers", &has_cutlass_headers);
    m.def("sm120_mxf4nvf4_ue4m3_available", &sm120_mxf4nvf4_ue4m3_available);
    m.def("fp4_mma_microbench", &fp4_mma_microbench);
}
