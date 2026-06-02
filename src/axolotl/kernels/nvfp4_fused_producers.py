"""Fused NVFP4 producers in the ``tl.dot_scaled`` row-major layout.

These feed the packed-input path of ``attn_nvfp4_flash.nvfp4_flash_attention_packed``
so no standalone pre-quant HBM round-trip of Q/K/V is paid: the NVFP4 pack rides
along with the op that already touches the operand.

  * ``fused_rope_quant_qk``    — partial RoPE (Qwen3.5 rotate_half, ``rotary_dim`` of
    ``head_dim``) on Q,K + NVFP4 pack along head_dim, one pass. The pass-through
    tail (``head_dim - rotary_dim`` dims) is copied unrotated then quantized with
    the rotated head, so the whole head_dim is one contiguous packed row.
  * ``quant_v_keyaxis``        — V (no RoPE) packed along the KEY axis (V^T
    ``[.,D,Skv]`` layout the PV-GEMM consumes), via strided reads (no transpose
    copy). This is the v_proj epilogue stand-in; it reads V once and emits packed.

Layout (matches attn_nvfp4_flash._quant_nvfp4 / tl.dot_scaled, e4m3 group-16):
  * packed operand : ``[rows, K//2]`` uint8  (2 e2m1 nibbles/byte, low first)
  * scale          : ``[rows, K//16]`` float8_e4m3fn  (row-major, NOT swizzled)

The quant core (block amax -> e4m3 scale -> hw FP4 pack) is identical to
``attn_nvfp4_flash._quant_nvfp4_kernel``; only the per-tile transform differs (RoPE
for Q/K, identity for V). Single-level scaling (no per-tensor global amax), matching
what the flash kernel's own pre-quant uses, so parity is apples-to-apples.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mslk.quantize.triton.fp4_quantize import convert_fp32_to_fp4_packed

_E4M3_EPS = tl.constexpr(1.5258789e-05)
_F8E4M3_MAX = tl.constexpr(448.0)
_F4_MAX = tl.constexpr(6.0)


# ---------------------------------------------------------------------------
# Fused partial-RoPE -> NVFP4 pack of Q (or K), quantized along head_dim D.
# Grid: (Z*H, cdiv(S, BLOCK_R)). One program owns a [BLOCK_R, D] tile of one head.
# cos/sin are [Z, S, rotary_dim] (already mrope-resolved by the HF rotary module);
# they are broadcast over heads and indexed by the seq row.
# ---------------------------------------------------------------------------
@triton.jit
def _rope_quant_kernel(
    x_ptr,            # [Z*H, S, D]  high-precision (normed) Q or K
    cos_ptr, sin_ptr, # [Z, S, ROT]  (ROT = rotary_dim)
    q_ptr, s_ptr,     # [Z*H, S, D//2] uint8, [Z*H, S, D//16] e4m3
    Z, H, S,
    s_xn, s_xr,       # x: per-(z*h) stride, per-row(seq) stride; col stride = 1
    s_cz, s_cr,       # cos/sin: per-z stride, per-row stride; col stride = 1
    s_qn, s_qr,       # q packed: per-(z*h) stride, per-row stride
    s_sn, s_sr,       # scale: per-(z*h) stride, per-row stride
    D: tl.constexpr, ROT: tl.constexpr, HALF: tl.constexpr,
    BLOCK_R: tl.constexpr,
    DP2: tl.constexpr, DP16: tl.constexpr, NG: tl.constexpr,
):
    pid_n = tl.program_id(0)   # z*h
    pid_r = tl.program_id(1)
    z = pid_n // H

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = offs_r < S
    offs_d = tl.arange(0, D)

    xbase = pid_n * s_xn
    x = tl.load(
        x_ptr + xbase + offs_r[:, None] * s_xr + offs_d[None, :],
        mask=rmask[:, None], other=0.0,
    ).to(tl.float32)

    # partial RoPE on the first ROT dims; tail [ROT, D) passes through unrotated.
    # rotate_half over the ROT block: low half partner is +HALF (negated), high -HALF.
    is_rot = offs_d < ROT
    is_low = offs_d < HALF
    partner = tl.where(is_low, offs_d + HALF, offs_d - HALF)
    xp = tl.load(
        x_ptr + xbase + offs_r[:, None] * s_xr + partner[None, :],
        mask=rmask[:, None] & (partner[None, :] < ROT), other=0.0,
    ).to(tl.float32)
    rot = tl.where(is_low, -xp, xp)

    cbase = z * s_cz
    # cos/sin only defined for d < ROT; clamp the col index, gate by is_rot.
    cd = tl.where(is_rot, offs_d, 0)
    cos = tl.load(
        cos_ptr + cbase + offs_r[:, None] * s_cr + cd[None, :],
        mask=rmask[:, None], other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + cbase + offs_r[:, None] * s_cr + cd[None, :],
        mask=rmask[:, None], other=0.0,
    ).to(tl.float32)

    x_rot = tl.where(is_rot[None, :], x * cos + rot * sin, x)

    # NVFP4 pack along D, group-16, single-level scale.
    xb = x_rot.reshape(BLOCK_R, NG, 16)
    amax = tl.max(tl.abs(xb), axis=2)
    sc = tl.clamp(amax / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
    xn = xb / sc.to(tl.float32)[:, :, None]
    pairs = xn.reshape(BLOCK_R * DP2, 2).split()
    qpk = convert_fp32_to_fp4_packed(pairs).reshape(BLOCK_R, DP2)

    offs_qk = tl.arange(0, DP2)
    tl.store(
        q_ptr + pid_n * s_qn + offs_r[:, None] * s_qr + offs_qk[None, :],
        qpk, mask=rmask[:, None],
    )
    offs_sk = tl.arange(0, DP16)
    tl.store(
        s_ptr + pid_n * s_sn + offs_r[:, None] * s_sr + offs_sk[None, :],
        sc.to(tl.uint8, bitcast=True), mask=rmask[:, None],
    )


def fused_rope_quant_qk(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3.5 partial RoPE to ``x`` and emit it NVFP4-packed (along D).

    Args:
        x: ``[Z, H, S, D]`` high precision (already q_norm/k_norm'd), D in {128,256}.
        cos, sin: ``[Z, S, rotary_dim]`` (mrope-resolved); rotary_dim <= D, even.

    Returns:
        (packed uint8 ``[Z*H, S, D//2]``, scale float8_e4m3fn ``[Z*H, S, D//16]``)
        in the row-major tl.dot_scaled layout.
    """
    z, h, s, d = x.shape
    rot = cos.shape[-1]
    assert d % 16 == 0 and rot % 2 == 0 and rot <= d
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    xn = x.reshape(z * h, s, d)
    q = x.new_empty(z * h, s, d // 2, dtype=torch.uint8)
    sc = x.new_empty(z * h, s, d // 16, dtype=torch.uint8)

    BLOCK_R = 64
    grid = (z * h, triton.cdiv(s, BLOCK_R))
    _rope_quant_kernel[grid](
        xn, cos, sin, q, sc,
        z, h, s,
        xn.stride(0), xn.stride(1),
        cos.stride(0), cos.stride(1),
        q.stride(0), q.stride(1),
        sc.stride(0), sc.stride(1),
        D=d, ROT=rot, HALF=rot // 2,
        BLOCK_R=BLOCK_R,
        DP2=d // 2, DP16=d // 16, NG=d // 16,
    )
    return q, sc.view(torch.float8_e4m3fn)


def quant_v_keyaxis(
    value: torch.Tensor, block_n: int = 128
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack V (no RoPE) along the KEY axis -> V^T tl.dot_scaled layout.

    The v_proj epilogue stand-in: reads V ``[Z, Hk, Skv, D]`` once via strided reads
    (no transpose copy) and emits ``vnv [Z*Hk, D, Skv_pad//2]`` /
    ``vsc [Z*Hk, D, Skv_pad//16]`` (e4m3), key axis padded to a multiple of block_n.

    Returns (vnv, vsc, Skv_pad).
    """
    from axolotl.kernels.attn_nvfp4_flash import _next_mult, _quant_nvfp4

    z, hk, s_kv, d = value.shape
    v2 = value.reshape(z * hk, s_kv, d)
    s_kv_pad = _next_mult(s_kv, block_n)
    vnv, vsc = _quant_nvfp4(v2, transpose=True, k_pad=s_kv_pad)
    return vnv, vsc, s_kv_pad
