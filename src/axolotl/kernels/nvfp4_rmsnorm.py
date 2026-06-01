"""Fused RMSNorm -> NVFP4 quantization for consumer Blackwell (sm_120).

MSLK ships a fused rms+quant kernel (`triton_scale_nvfp4_quant_rms`) but it uses a
1D persistent thread-grid sized from the sm_100 SM count and faults with an illegal
memory access on sm_120. The plain quant kernel (`triton_quantize_nvfp4`) uses a 2D
M/N tile grid and runs fine on sm_120, so we rebuild the fusion on that grid instead.

RMS is a full-row reduction (needs all K columns) which a 64-col tile can't see, so we
take the reciprocal-norm factor in torch (one cheap read of x) and fuse the per-row
normalize + per-column gamma + NVFP4 quant into the tile kernel. The kernel emits the
normalized bf16 activation (for the LoRA adapter / next op) alongside the packed FP4
qdata and swizzled e4m3 scales consumed directly by ``torch._scaled_mm``.
"""

from __future__ import annotations

import torch
import triton

from mslk.quantize.triton.fp4_quantize import (
    convert_fp32_to_fp4_packed,
    nvfp4_scale_swizzle,
)

import triton.language as tl


@triton.jit
def _fused_rmsnorm_nvfp4_kernel(
    x_ptr,
    rms_ptr,  # [M] reciprocal rms factor, fp32
    w_ptr,  # [N] rmsnorm gamma
    y_ptr,  # [M, N] normalized activation out, bf16
    q_ptr,  # [M, N//2] packed fp4
    s_ptr,  # swizzled e4m3 scales
    stride_xm,
    stride_xn,
    M,
    N,
    M_PER_BLOCK: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    E4M3_EPS = 1.5258789e-05
    FP8_E4M3_MAX = 448.0
    FP4_E2M1_MAX = 6.0

    NUM_ELEM_PER_LAYOUT = 128 * 4
    NUM_N_BLOCKS = tl.cdiv(N, 64)

    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    # Tail block that only zeros the padded scales when M < 128.
    if M_PER_BLOCK != 128 and pid_m * M_PER_BLOCK >= M:
        tl.device_assert(pid_m == 1, "pid_m != 1 when M_PER_BLOCK != 128")
        layout_off = pid_n * NUM_ELEM_PER_LAYOUT
        offs_m = tl.arange(0, 128)[:, None]
        scale_offs = layout_off + nvfp4_scale_swizzle(offs_m)
        oob_mask = (offs_m >= M) & tl.full((4,), True, dtype=tl.int1)[None, :]
        zero_scales = tl.full([128, 4], 0, dtype=tl.float8e4nv)
        tl.store(s_ptr + scale_offs, zero_scales, mask=oob_mask)
        return

    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]

    if USE_MASK:
        mask = (offs_m < M) & (offs_n < N)
        other = 0.0
    else:
        mask = None
        other = None

    load_offsets = offs_m * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptr + load_offsets, mask=mask, other=other).to(tl.float32)

    # Fused RMSNorm: per-row reciprocal factor * per-column gamma.
    rms = tl.load(rms_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    y = x * rms * w  # [M_PER_BLOCK, 64] normalized activation

    # Emit normalized activation for the adapter / residual-consuming op.
    tl.store(y_ptr + (offs_m * N + offs_n), y.to(y_ptr.dtype.element_ty), mask=mask)

    y_blocks = y.reshape(M_PER_BLOCK, 4, 16)
    block_amax = tl.max(tl.abs(y_blocks), axis=2)  # [M_PER_BLOCK, 4]

    scales = tl.div_rn(block_amax, FP4_E2M1_MAX)
    scales = tl.clamp(scales, E4M3_EPS, FP8_E4M3_MAX)
    scales = scales.to(tl.float8e4nv)

    total_scale = tl.div_rn(1.0, scales.to(tl.float32)[:, :, None])
    y_blocks = y_blocks * total_scale

    if USE_MASK:
        scale_offs_n = pid_n * 4 + tl.arange(0, 4)[None, :]
        scale_mask = (offs_m < M) & (scale_offs_n < (N // 16))
        scales = tl.where(scale_mask, scales, 0.0)

    swz_m = (pid_m * M_PER_BLOCK % 128) + tl.arange(0, M_PER_BLOCK)[:, None]
    layout_off = (
        (pid_m * M_PER_BLOCK) // 128
    ) * NUM_N_BLOCKS * NUM_ELEM_PER_LAYOUT + pid_n * NUM_ELEM_PER_LAYOUT
    scale_offs = layout_off + nvfp4_scale_swizzle(swz_m)
    tl.store(s_ptr + scale_offs, scales)

    x_fp4x2 = convert_fp32_to_fp4_packed(y_blocks.reshape(M_PER_BLOCK, 32, 2).split())
    q_offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    q_offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    if USE_MASK:
        q_mask = (q_offs_m < M) & (q_offs_n < N // 2)
    else:
        q_mask = None
    tl.store(q_ptr + (q_offs_m * (N // 2) + q_offs_n), x_fp4x2, mask=q_mask)


def fused_rmsnorm_nvfp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm(x) * weight, returning (normalized bf16, packed fp4, swizzled scales).

    Single-level (activation) NVFP4: global_scale is implicitly 1.0. The fp4/scale
    outputs feed ``torch._scaled_mm`` directly (TN layout, contraction %32).
    """
    orig_dims, K = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, K)
    M, N = x2.shape
    assert N % 16 == 0, "K must be divisible by 16 for NVFP4 quantization"

    rms = torch.rsqrt(x2.float().pow(2).mean(-1, keepdim=True) + eps).reshape(M)

    num_scales = N // 16
    n_row_blocks = triton.cdiv(M, 128)
    n_col_blocks = triton.cdiv(num_scales, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    xq = x2.new_empty(M, N // 2, dtype=torch.uint8)
    scales = x2.new_empty(padded_rows, padded_cols, dtype=torch.float8_e4m3fn)
    y = x2.new_empty(M, N, dtype=x.dtype)

    M_PER_BLOCK = min(triton.next_power_of_2(M), 128)
    USE_MASK = M % M_PER_BLOCK != 0 or N % 64 != 0

    grid = (triton.cdiv(N, 64), triton.cdiv(M, M_PER_BLOCK))
    if M_PER_BLOCK != 128:
        grid = (grid[0], grid[1] + 1)

    _fused_rmsnorm_nvfp4_kernel[grid](
        x2,
        rms,
        weight,
        y,
        xq,
        scales,
        x2.stride(0),
        x2.stride(1),
        M,
        N,
        M_PER_BLOCK=M_PER_BLOCK,
        USE_MASK=USE_MASK,
    )

    return (
        y.view(*orig_dims, N),
        xq.view(torch.float4_e2m1fn_x2).view(*orig_dims, N // 2),
        scales,
    )
