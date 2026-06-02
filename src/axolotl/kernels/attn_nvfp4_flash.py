"""Fused native-NVFP4 flash attention (forward only) for sm_120 Blackwell.

Both attention GEMMs run as real 5th-gen FP4 tensor-core ops via Triton
``tl.dot_scaled`` (e2m1-packed operands + e4m3 group-16 block scales — native
NVFP4, NOT MXFP4/e8m0). Verified in PTX to emit ``mma.sync...kind::mxf4nvf4...ue4m3``.

Measured on RTX 5090 (sm_120), Qwen3.5-9B-like shapes (Z1 H16 Hk4 D256 causal),
flash compute only (pre-quant excluded) vs bf16 cuDNN FLASH SDPA:
  S=2048 1.31x, S=4096 1.33x, S=8192 1.82x — the FP4 win grows with seq length.
The standalone pre-quant (3 full-tensor HBM round-trips) costs ~as much as the
whole bf16 attention, so the END-TO-END pipeline does NOT beat cuDNN unless the
quant is fused into the producing ops (RoPE for Q/K, the v_proj epilogue for V)
so no extra round-trip is paid — see nvfp4_quant_fusion_proto.py.

Quantize-ONCE design:
  * A cheap Triton pre-pass packs Q, K (along head_dim D) and V (along the key
    seq axis) into NVFP4 exactly once. The flash kernel then LOADS pre-packed
    tiles; it never re-quantizes K/V per Q-block (the trap that made a prior
    MXFP4 attempt 5x slower).
  * The ONLY in-kernel quant is the P (softmax-probs) tile, packed to NVFP4
    along the key axis fused into the softmax/exp step (Sage-3 style), so it can
    feed the P@V ``tl.dot_scaled`` directly.

Flash inner loop (one program per Q-block, grid over Z*H heads x Q-blocks):
  load packed Q tile once; loop K/V blocks:
    QK = tl.dot_scaled(Qnv, Knv)        -> S tile in SRAM (fp32)
    scale + causal mask + key-pad bias
    online softmax (running max/sum, rescale acc)
    pack P tile to NVFP4 along key axis  (in-kernel)
    acc += tl.dot_scaled(Pnv, Vnv)
  final normalize.

tl.dot_scaled layout (per the validated /tmp microbenches), e4m3 group-16:
  * packed operand: ``[rows, K//2]`` uint8 (2 e2m1 nibbles/byte, low nibble first)
  * scale:          ``[rows, K//16]`` float8_e4m3fn
  * ``tl.dot_scaled(a, asc, "e2m1", b.T, bsc, "e2m1")`` computes ``a @ b`` with
    ``b`` loaded as ``[N, K//2]`` and transposed inside.

Forward only, no autograd. GQA handled by mapping each query head to its KV head
in-kernel (no repeat_kv materialization).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mslk.quantize.triton.fp4_quantize import convert_fp32_to_fp4_packed

_E4M3_EPS = tl.constexpr(1.5258789e-05)
_F8E4M3_MAX = tl.constexpr(448.0)
_F4_MAX = tl.constexpr(6.0)
_NEG_INF = tl.constexpr(float("-inf"))


# ---------------------------------------------------------------------------
# Pre-pass: quantize a [R, K] tile to NVFP4 (group-16 along K), row-major
# tl.dot_scaled layout. One kernel handles Q/K (quant along D) and V^T (quant
# along the key axis) — caller just lays the input out as [rows, contraction].
# ---------------------------------------------------------------------------
@triton.jit
def _quant_nvfp4_kernel(
    x_ptr, q_ptr, s_ptr,
    R, K, K_READ,
    s_xb, s_qb, s_sb,   # per-batch strides
    s_xr, s_xk,         # input row / contraction-col strides (transpose w/o copy)
    s_qr, s_sr,         # per-row strides (= K//2, K//16)
    BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rmask = offs_r < R
    kmask = offs_k < K_READ
    x = tl.load(
        x_ptr + pid_b * s_xb + offs_r[:, None] * s_xr + offs_k[None, :] * s_xk,
        mask=rmask[:, None] & kmask[None, :], other=0.0,
    ).to(tl.float32)

    NG: tl.constexpr = BLOCK_K // 16
    xb = x.reshape(BLOCK_R, NG, 16)
    amax = tl.max(tl.abs(xb), axis=2)
    sc = tl.clamp(amax / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
    xn = xb / sc.to(tl.float32)[:, :, None]
    pairs = xn.reshape(BLOCK_R * (BLOCK_K // 2), 2).split()
    qpk = convert_fp32_to_fp4_packed(pairs).reshape(BLOCK_R, BLOCK_K // 2)

    offs_qk = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(
        q_ptr + pid_b * s_qb + offs_r[:, None] * s_qr + offs_qk[None, :],
        qpk, mask=rmask[:, None] & (offs_qk[None, :] < s_qr),
    )
    offs_sk = pid_k * NG + tl.arange(0, NG)
    tl.store(
        s_ptr + pid_b * s_sb + offs_r[:, None] * s_sr + offs_sk[None, :],
        sc.to(tl.uint8, bitcast=True), mask=rmask[:, None] & (offs_sk[None, :] < s_sr),
    )


def _quant_nvfp4(
    x: torch.Tensor, transpose: bool = False, k_pad: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to NVFP4 group-16 along the contraction axis (tl.dot_scaled layout).

    Returns (packed uint8 [B, R, K//2], scale float8_e4m3fn [B, R, K//16]).

    Args:
        x: ``[B, R, K]`` (no transpose) — quantize along K (rows=R, contraction=K).
        transpose: if True, ``x`` is ``[B, K, R]`` and the contraction axis is
            quantized via STRIDED reads (no physical transpose/copy) — used for V,
            laid out ``[B, Skv, D]`` and quantized along Skv to produce V^T.
        k_pad: pad the contraction axis to this length (padded groups -> amax 0 ->
            eps scale, zero packed values), for the PV-GEMM key-axis multiple-of-16.

    The contraction is tiled by a power-of-2 ``BLOCK_K`` (multiple of 16) so a
    non-power-of-2 key axis is supported.
    """
    if transpose:
        B, K_read, R = x.shape   # contraction axis is the middle dim (Skv)
        s_xr, s_xk = x.stride(2), x.stride(1)
        s_xb = x.stride(0)
    else:
        B, R, K_read = x.shape
        x = x.contiguous()
        s_xr, s_xk = x.stride(1), x.stride(2)
        s_xb = x.stride(0)
    K = k_pad if k_pad is not None else K_read
    assert K % 16 == 0, "padded contraction dim must be a multiple of 16"
    q = x.new_empty(B, R, K // 2, dtype=torch.uint8)
    s = x.new_zeros(B, R, K // 16, dtype=torch.uint8)
    BLOCK_R = 64
    BLOCK_K = min(triton.next_power_of_2(K), 256)
    grid = (B, triton.cdiv(R, BLOCK_R), triton.cdiv(K, BLOCK_K))
    _quant_nvfp4_kernel[grid](
        x, q, s, R, K, K_read,
        s_xb, q.stride(0), s.stride(0),
        s_xr, s_xk,
        K // 2, K // 16,
        BLOCK_R=BLOCK_R, BLOCK_K=BLOCK_K,
    )
    return q, s.view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Fused flash forward. Grid: (num_q_blocks, Z*H). Each program owns one Q-block
# of one (z, head); it indexes the matching KV head for GQA.
# ---------------------------------------------------------------------------
@triton.jit
def _flash_fwd_kernel(
    qnv_ptr, qsc_ptr,          # [Z*H, Sq, D//2], [Z*H, Sq, D//16]
    knv_ptr, ksc_ptr,          # [Z*Hk, Skv, D//2], [Z*Hk, Skv, D//16]
    vnv_ptr, vsc_ptr,          # [Z*Hk, D, Skv//2], [Z*Hk, D, Skv//16]  (V^T, quant on key)
    bias_ptr,                  # [Z, Skv] fp32 additive key-pad bias, or 0
    out_ptr,                   # [Z*H, Sq, D]  out_dtype
    scaling,
    Sq, Skv,
    D: tl.constexpr,
    H: tl.constexpr, HK: tl.constexpr,
    sq_qn, sq_sn,
    sk_kn, sk_sn,
    sv_kn, sv_sn,
    sb_z,
    so_n,
    HAS_BIAS: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DP2: tl.constexpr, DP16: tl.constexpr,   # D//2, D//16
    NP2: tl.constexpr, NP16: tl.constexpr,   # BLOCK_N//2, BLOCK_N//16
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    zhk = z * HK + (h // (H // HK))   # GQA: query head -> kv head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)

    # load packed Q tile once: [BLOCK_M, D//2] + scale [BLOCK_M, D//16]
    qbase = pid_zh * (Sq * sq_qn)
    qscbase = pid_zh * (Sq * sq_sn)
    mmask = offs_m < Sq
    qnv = tl.load(
        qnv_ptr + qbase + offs_m[:, None] * sq_qn + offs_dp[None, :],
        mask=mmask[:, None], other=0,
    )
    qsc = tl.load(
        qsc_ptr + qscbase + offs_m[:, None] * sq_sn + offs_dsc[None, :],
        mask=mmask[:, None], other=0,
    ).to(tl.float8e4nv, bitcast=True)

    m_i = tl.full((BLOCK_M,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    kbase = zhk * (Skv * sk_kn)
    kscbase = zhk * (Skv * sk_sn)
    vbase = zhk * (D * sv_kn)
    vscbase = zhk * (D * sv_sn)

    offs_n0 = tl.arange(0, BLOCK_N)
    offs_np = tl.arange(0, NP2)
    offs_nsc = tl.arange(0, NP16)
    offs_d = tl.arange(0, D)

    # causal: key j attends iff j <= i + (Skv - Sq). Cap the loop to the last
    # relevant key block for the max query row in this block.
    if CAUSAL:
        hi = tl.minimum(Skv, (pid_m * BLOCK_M + BLOCK_M) + (Skv - Sq))
    else:
        hi = Skv

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n0
        nmask = offs_n < Skv
        # load packed K tile [BLOCK_N, D//2] + scale [BLOCK_N, D//16]
        knv = tl.load(
            knv_ptr + kbase + offs_n[:, None] * sk_kn + offs_dp[None, :],
            mask=nmask[:, None], other=0,
        )
        ksc = tl.load(
            ksc_ptr + kscbase + offs_n[:, None] * sk_sn + offs_dsc[None, :],
            mask=nmask[:, None], other=0,
        ).to(tl.float8e4nv, bitcast=True)
        # QK^T via native NVFP4: [BLOCK_M, BLOCK_N]
        s = tl.dot_scaled(qnv, qsc, "e2m1", knv.T, ksc, "e2m1")
        s = s * scaling

        if HAS_BIAS:
            b = tl.load(bias_ptr + z * sb_z + offs_n, mask=nmask, other=_NEG_INF)
            s = s + b[None, :]
        s = tl.where(nmask[None, :], s, _NEG_INF)
        if CAUSAL:
            causal_ok = offs_n[None, :] <= (offs_m[:, None] + (Skv - Sq))
            s = tl.where(causal_ok, s, _NEG_INF)

        # online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])         # [BLOCK_M, BLOCK_N], >=0
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # in-kernel NVFP4 pack of P along the key axis (group-16)
        pb = p.reshape(BLOCK_M, NP16, 16)
        pamax = tl.max(pb, axis=2)             # P>=0
        psc = tl.clamp(pamax / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
        pn = pb / psc.to(tl.float32)[:, :, None]
        ppairs = pn.reshape(BLOCK_M * NP2, 2).split()
        pq = convert_fp32_to_fp4_packed(ppairs).reshape(BLOCK_M, NP2)

        # load packed V^T tile: [D, BLOCK_N//2] + scale [D, BLOCK_N//16]
        vnv = tl.load(
            vnv_ptr + vbase + offs_d[:, None] * sv_kn + (start_n // 2 + offs_np)[None, :],
        )
        vsc = tl.load(
            vsc_ptr + vscbase + offs_d[:, None] * sv_sn + (start_n // 16 + offs_nsc)[None, :],
        ).to(tl.float8e4nv, bitcast=True)
        # P @ V via native NVFP4: a=P [BLOCK_M, BLOCK_N], b=V^T loaded [D, BLOCK_N//2]
        acc = tl.dot_scaled(pq, psc, "e2m1", vnv.T, vsc, "e2m1", acc=acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    obase = pid_zh * (Sq * so_n)
    tl.store(
        out_ptr + obase + offs_m[:, None] * so_n + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty), mask=mmask[:, None],
    )


def _next_mult(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _run_flash_packed(
    qnv, qsc, knv, ksc, vnv, vsc,
    z, h, hk, s_q, s_kv, d,
    scaling, causal, bias, out,
    block_m, block_n, num_warps, num_stages,
):
    """Launch the flash kernel on already-packed (tl.dot_scaled layout) Q/K/V.

    qnv/knv: ``[Z*H or Z*Hk, S, D//2]`` uint8;  qsc/ksc: ``[., S, D//16]`` e4m3.
    vnv: ``[Z*Hk, D, Skv_pad//2]`` uint8;  vsc: ``[Z*Hk, D, Skv_pad//16]`` e4m3
    (V^T, quantized along the key axis; key axis padded to a multiple of block_n).
    """
    qnv_v = qnv.view(torch.uint8)
    knv_v = knv.view(torch.uint8)
    vnv_v = vnv.view(torch.uint8)
    qsc_v = qsc.view(torch.uint8)
    ksc_v = ksc.view(torch.uint8)
    vsc_v = vsc.view(torch.uint8)

    grid = (triton.cdiv(s_q, block_m), z * h)
    _flash_fwd_kernel[grid](
        qnv_v, qsc_v, knv_v, ksc_v, vnv_v, vsc_v,
        bias if bias is not None else qnv_v,
        out,
        scaling,
        s_q, s_kv,
        D=d,
        H=h, HK=hk,
        sq_qn=qnv_v.stride(1), sq_sn=qsc_v.stride(1),
        sk_kn=knv_v.stride(1), sk_sn=ksc_v.stride(1),
        sv_kn=vnv_v.stride(1), sv_sn=vsc_v.stride(1),
        sb_z=bias.stride(0) if bias is not None else 0,
        so_n=out.stride(1),
        HAS_BIAS=bias is not None,
        CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=block_n,
        DP2=d // 2, DP16=d // 16,
        NP2=block_n // 2, NP16=block_n // 16,
        num_warps=num_warps, num_stages=num_stages,
    )


def nvfp4_flash_attention_packed(
    qnv: torch.Tensor,
    qsc: torch.Tensor,
    knv: torch.Tensor,
    ksc: torch.Tensor,
    vnv: torch.Tensor,
    vsc: torch.Tensor,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    scaling: float,
    out_dtype: torch.dtype,
    causal: bool = False,
    key_pad_bias: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """Flash forward on Q/K/V ALREADY in the NVFP4 tl.dot_scaled layout.

    Skips the internal pre-quant entirely — operands are expected to arrive packed
    from the fused producers (RoPE for Q/K, v_proj epilogue / key-axis quant for V).
    V's key axis must be padded to a multiple of ``block_n`` (padded keys contribute
    nothing: masked to -inf and eps-scaled zero columns).

    Returns ``[Z, H, S_q, D]`` in ``out_dtype``.
    """
    bias = None
    if key_pad_bias is not None:
        bias = key_pad_bias.to(torch.float32).contiguous()
    out = torch.empty(z * h, s_q, d, device=qnv.device, dtype=out_dtype)
    _run_flash_packed(
        qnv, qsc, knv, ksc, vnv, vsc,
        z, h, hk, s_q, s_kv, d,
        scaling, causal, bias, out,
        block_m, block_n, num_warps, num_stages,
    )
    return out.reshape(z, h, s_q, d)


def nvfp4_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    causal: bool = False,
    num_key_value_groups: int = 1,
    key_pad_bias: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
):
    """Fused native-NVFP4 flash attention, forward only.

    Args:
        query: ``[Z, H, Sq, D]`` high precision (bf16/fp16/fp32). D in {128, 256}.
        key/value: ``[Z, Hk, Skv, D]`` (pre-repeat_kv GQA).
        scaling: softmax scale on QK^T scores (e.g. ``1/sqrt(D)``).
        causal: lower-triangular causal mask (bottom-right aligned if Sq != Skv).
        num_key_value_groups: ``H // Hk``.
        key_pad_bias: optional ``[Z, Skv]`` additive bias on the key axis
            (0 for real tokens, -inf for padding), broadcast over heads/queries.
        block_m, block_n: flash tile sizes.

    Returns:
        ``[Z, H, Sq, D]`` in ``query.dtype``.
    """
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    assert h % hk == 0 and h // hk == num_key_value_groups
    assert d % 16 == 0 and d in (128, 256)
    out_dtype = query.dtype

    q2 = query.reshape(z * h, s_q, d)
    k2 = key.reshape(z * hk, s_kv, d)
    v2 = value.reshape(z * hk, s_kv, d)

    # pre-quant (quantize-once): Q,K along D; V along the key axis (-> V^T [.,D,Skv]).
    # V is quantized via strided reads (no physical transpose) and its key axis is
    # padded to a multiple of block_n: padded keys are masked to -inf (P weight 0)
    # and the eps-scaled zero V columns contribute nothing, so the result is exact.
    qnv, qsc = _quant_nvfp4(q2)
    knv, ksc = _quant_nvfp4(k2)
    s_kv_pad = _next_mult(s_kv, block_n)
    vnv, vsc = _quant_nvfp4(v2, transpose=True, k_pad=s_kv_pad)

    bias = None
    if key_pad_bias is not None:
        bias = key_pad_bias.to(torch.float32).contiguous()

    out = torch.empty(z * h, s_q, d, device=query.device, dtype=out_dtype)

    qnv_v = qnv.view(torch.uint8)
    knv_v = knv.view(torch.uint8)
    vnv_v = vnv.view(torch.uint8)
    qsc_v = qsc.view(torch.uint8)
    ksc_v = ksc.view(torch.uint8)
    vsc_v = vsc.view(torch.uint8)

    grid = (triton.cdiv(s_q, block_m), z * h)

    def _run():
        _flash_fwd_kernel[grid](
            qnv_v, qsc_v, knv_v, ksc_v, vnv_v, vsc_v,
            bias if bias is not None else qnv_v,  # dummy ptr when no bias
            out,
            scaling,
            s_q, s_kv,
            D=d,
            H=h, HK=hk,
            sq_qn=qnv_v.stride(1), sq_sn=qsc_v.stride(1),
            sk_kn=knv_v.stride(1), sk_sn=ksc_v.stride(1),
            sv_kn=vnv_v.stride(1), sv_sn=vsc_v.stride(1),
            sb_z=bias.stride(0) if bias is not None else 0,
            so_n=out.stride(1),
            HAS_BIAS=bias is not None,
            CAUSAL=causal,
            BLOCK_M=block_m, BLOCK_N=block_n,
            DP2=d // 2, DP16=d // 16,
            NP2=block_n // 2, NP16=block_n // 16,
            num_warps=num_warps, num_stages=num_stages,
        )

    _run()
    return out.reshape(z, h, s_q, d)
