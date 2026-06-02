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

A native-NVFP4 BACKWARD (``nvfp4_flash_attn_func`` / ``_NVFP4FlashAttn``) wraps
this into a full ``torch.autograd.Function``. It recomputes S/P in SRAM and runs
all four grad GEMMs (dV, dP, dK, dQ) as ``tl.dot_scaled`` FP4 ops, quantizing the
gradient operands (P, dS, dO) with stochastic rounding (the convergence knob from
``utils/nvfp4_training``). The seq-axis contractions (dV=P^T@dO, dK=dS^T@Q,
dQ=dS@K) are the large-K FP4-friendly ones; dP=dO@V^T contracts over head_dim.

The FP4 packs that are reused across the score-recompute loops are quantized ONCE
in two cheap pack-prep passes instead of being re-quantized every loop iteration
(the round of per-iter SR philox draws was the backward's dominant cost):
  * pack-prep (m-block): Q/dO and their transposes -> dK/dV pass operands.
  * K-side pack-prep (n-block): K/V along D and K^T along N -> dQ pass operands.
Only the two genuinely (n,m)-dependent SR packs (pT, dSt) remain in the dK/dV
loop. With the loop footprint shrunk, both passes run narrow key tiles + deep
pipelining. Net: backward ~2x (S=2048) / ~1.45x (S=4096) slower than bf16 cuDNN
on sm_120, down from ~10x. Validated on Qwen3.5-2B: a 120-step SR-on training run
tracks bf16 attention to within loss noise (no divergence).

GQA handled by mapping each query head to its KV head in-kernel (no repeat_kv
materialization).
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
# In-kernel NVFP4 pack of a [ROWS, K] fp32 tile along K (group-16), returning the
# tl.dot_scaled operands (packed uint8 [ROWS, K//2], e4m3 scale [ROWS, K//16]).
# Used by the backward to quantize the gradient-side operands (P, dS, dO, Q, K)
# right where they are produced — no HBM round-trip, no per-block re-quant.
#
# STOCHASTIC_ROUND: mirror utils/nvfp4_training._sr_dither — add uniform noise of
# width = one FP4 step (in the per-block-scaled domain) before round-to-nearest,
# which realizes unbiased stochastic rounding. This is the convergence-critical
# knob for the gradient operands. The PRNG is a cheap in-kernel philox draw keyed
# by a per-launch seed + the tile's flat element offset (decorrelated across the
# tile and across recompute), so no extra global memory traffic is paid.
# ---------------------------------------------------------------------------
@triton.jit
def _pack_nvfp4_along_k(
    x, base_off, seed,
    ROWS: tl.constexpr, K: tl.constexpr,
    STOCHASTIC: tl.constexpr,
):
    NG: tl.constexpr = K // 16
    xb = x.reshape(ROWS, NG, 16)
    amax = tl.max(tl.abs(xb), axis=2)
    sc = tl.clamp(amax / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
    scf = sc.to(tl.float32)[:, :, None]
    xn = xb / scf
    if STOCHASTIC:
        # one FP4 step in the scaled domain: 2^clamp(floor(log2|xn|),0,2). Adding
        # U(-.5,.5)*step before round-to-nearest realizes unbiased SR. The step is a
        # 3-level staircase in |xn| (<2 ->1, <4 ->2, else 4) so it is computed with two
        # compares instead of log2/floor/exp2 (~3 transcendentals/elt) — the SR math,
        # not the GEMMs, dominated this kernel.
        ax = tl.abs(xn)
        step = tl.where(ax < 2.0, 1.0, tl.where(ax < 4.0, 2.0, 4.0))
        off = base_off + tl.arange(0, ROWS)[:, None, None] * K + \
            tl.arange(0, NG)[None, :, None] * 16 + tl.arange(0, 16)[None, None, :]
        u = tl.rand(seed, off) - 0.5
        xn = xn + u * step
    xn = tl.clamp(xn, -_F4_MAX, _F4_MAX)
    pairs = xn.reshape(ROWS * (K // 2), 2).split()
    q = convert_fp32_to_fp4_packed(pairs).reshape(ROWS, K // 2)
    return q, sc


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
    lse_ptr,                   # [Z*H, Sq] fp32 logsumexp, written iff STORE_LSE
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
    STORE_LSE: tl.constexpr,
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
    if STORE_LSE:
        # Persist logsumexp so the backward prep can skip its full QK^T recompute.
        # Match the prep kernel's safe handling for all-masked rows (l_i==0).
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        m_fin = tl.where(m_i == _NEG_INF, 0.0, m_i)
        tl.store(lse_ptr + pid_zh * Sq + offs_m, m_fin + tl.log(l_safe), mask=mmask)


def _next_mult(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


# Forward flash-tile defaults per head_dim, swept (parity-gated) on sm_120 across
# S=2048/4096/8192. D=256 is SMEM-tight: BLOCK_M64/BLOCK_N128/warps8/stages3 already
# wins (wider M-tiles or shallower pipelines spill or regress). D=128 has SMEM
# headroom — a full BLOCK_M128/BLOCK_N128 tile with 2 pipeline stages is 1.5x (S2048)
# to 2.0x (S8192) the D=256-style narrow default. (block_m, block_n, num_warps, num_stages).
_FWD_TILE = {256: (64, 128, 8, 3), 128: (128, 128, 8, 2)}
_FWD_TILE_DEFAULT = (64, 128, 8, 3)


def _resolve_fwd_tiles(d, block_m, block_n, num_warps, num_stages):
    """Pick head_dim-tuned forward tiles when the caller left the generic defaults.

    Explicit non-default tiles are honored as-is (autotune / manual override).
    """
    if (block_m, block_n, num_warps, num_stages) == _FWD_TILE_DEFAULT:
        return _FWD_TILE.get(d, _FWD_TILE_DEFAULT)
    return block_m, block_n, num_warps, num_stages


# ---------------------------------------------------------------------------
# Backward prep: LSE + D_i.  One program per (z, h, query-block m). Loops over all
# keys (FP4 QK^T recompute) to get the row logsumexp, then D_i = rowsum(dO * O).
# Both are the standard FlashAttention-2 backward preliminaries.
# ---------------------------------------------------------------------------
@triton.jit
def _flash_bwd_prep_kernel(
    q_ptr, k_ptr,
    do_ptr, o_ptr, bias_ptr,
    lse_ptr, delta_ptr,
    scaling, seed, Sq, Skv,
    D: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
    sq_n, sk_n, sdo_n, so_n, sb_z,
    HAS_BIAS: tl.constexpr, CAUSAL: tl.constexpr, HAVE_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    zhk = z * HK + (h // (H // HK))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mmask = offs_m < Sq

    # LSE is either reused from the forward (HAVE_LSE) or recomputed here via a full
    # FP4 QK^T pass. Reuse skips that pass entirely (the forward already had m_i/l_i).
    if HAVE_LSE:
        lse = tl.load(lse_ptr + pid_zh * Sq + offs_m, mask=mmask, other=0.0)
    else:
        q = tl.load(
            q_ptr + pid_zh * (Sq * sq_n) + offs_m[:, None] * sq_n + offs_d[None, :],
            mask=mmask[:, None], other=0.0,
        ).to(tl.float32)
        qnv, qsc = _pack_nvfp4_along_k(q, 0, seed, BLOCK_M, D, False)

        m_i = tl.full((BLOCK_M,), _NEG_INF, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        if CAUSAL:
            hi = tl.minimum(Skv, (pid_m * BLOCK_M + BLOCK_M) + (Skv - Sq))
        else:
            hi = Skv
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            nmask = offs_n < Skv
            k = tl.load(
                k_ptr + zhk * (Skv * sk_n) + offs_n[:, None] * sk_n + offs_d[None, :],
                mask=nmask[:, None], other=0.0,
            ).to(tl.float32)
            knv, ksc = _pack_nvfp4_along_k(k, 0, seed, BLOCK_N, D, False)
            s = tl.dot_scaled(qnv, qsc, "e2m1", knv.T, ksc, "e2m1") * scaling
            if HAS_BIAS:
                b = tl.load(bias_ptr + z * sb_z + offs_n, mask=nmask, other=_NEG_INF)
                s = s + b[None, :]
            s = tl.where(nmask[None, :], s, _NEG_INF)
            if CAUSAL:
                causal_ok = offs_n[None, :] <= (offs_m[:, None] + (Skv - Sq))
                s = tl.where(causal_ok, s, _NEG_INF)
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == _NEG_INF, 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(tl.exp(s - m_safe[:, None]), axis=1)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        m_final = tl.where(m_i == _NEG_INF, 0.0, m_i)
        lse = m_final + tl.log(l_safe)

    do = tl.load(
        do_ptr + pid_zh * (Sq * sdo_n) + offs_m[:, None] * sdo_n + offs_d[None, :],
        mask=mmask[:, None], other=0.0,
    ).to(tl.float32)
    o = tl.load(
        o_ptr + pid_zh * (Sq * so_n) + offs_m[:, None] * so_n + offs_d[None, :],
        mask=mmask[:, None], other=0.0,
    ).to(tl.float32)
    delta = tl.sum(do * o, axis=1)

    tl.store(lse_ptr + pid_zh * Sq + offs_m, lse, mask=mmask)
    tl.store(delta_ptr + pid_zh * Sq + offs_m, delta, mask=mmask)


# ---------------------------------------------------------------------------
# Backward pack-prep: quantize the m-block-local operands of the dK/dV pass ONCE
# per (z, h, query-block m) and store the FP4 packs to HBM, so the dK/dV pass
# (which loops query blocks for every key block) loads them instead of
# re-quantizing each one ~Skv/BLOCK_N times. Hoists 4 of the 6 per-iteration packs
# — including 2 of the 4 stochastic-rounding (philox) packs that dominated the
# kernel — out of the n*m loop into an m-only pass:
#   q  along D (RTN)   -> sT recompute operand
#   qT along M (RTN)   -> dK GEMM operand (Q^T)
#   do along D (SR)    -> dPt GEMM operand
#   doT along M (SR)   -> dV GEMM operand (dO^T)
# The two genuinely (n,m)-dependent SR packs (pT, dSt) stay in the dK/dV loop.
# ---------------------------------------------------------------------------
@triton.jit
def _flash_bwd_packprep_kernel(
    q_ptr, do_ptr,
    qnv_ptr, qsc_ptr, qtnv_ptr, qtsc_ptr,
    donv_ptr, dosc_ptr, dotnv_ptr, dotsc_ptr,
    seed, Sq, Sq_pad,
    D: tl.constexpr,
    sq_n, sdo_n,
    SR: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mmask = offs_m < Sq

    q = tl.load(
        q_ptr + pid_zh * (Sq * sq_n) + offs_m[:, None] * sq_n + offs_d[None, :],
        mask=mmask[:, None], other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + pid_zh * (Sq * sdo_n) + offs_m[:, None] * sdo_n + offs_d[None, :],
        mask=mmask[:, None], other=0.0,
    ).to(tl.float32)

    # Distinct philox stream per (m-block, operand) so the SR noise on dO is
    # decorrelated across query blocks. Without the pid_m term every m-block would
    # draw the same dither pattern (a hidden bias in the gradient estimate).
    mblk = pid_m * (BLOCK_M * D)
    # along D (rows = m tile)
    qnv, qsc = _pack_nvfp4_along_k(q, 0, seed, BLOCK_M, D, False)
    donv, dosc = _pack_nvfp4_along_k(do, 2 * Sq + mblk, seed, BLOCK_M, D, SR)
    # along M (rows = D): transpose the tile in SRAM, pack the BLOCK_M axis
    qT = tl.trans(q)
    doT = tl.trans(do)
    qtnv, qtsc = _pack_nvfp4_along_k(qT, 4 * Sq, seed, D, BLOCK_M, False)
    dotnv, dotsc = _pack_nvfp4_along_k(doT, Sq + mblk, seed, D, BLOCK_M, SR)

    DP2: tl.constexpr = D // 2
    DP16: tl.constexpr = D // 16
    MP2: tl.constexpr = BLOCK_M // 2
    MP16: tl.constexpr = BLOCK_M // 16
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)
    offs_mp = pid_m * MP2 + tl.arange(0, MP2)
    offs_msc = pid_m * MP16 + tl.arange(0, MP16)

    # store along-D packs: [zh, Sq, D//2] + [zh, Sq, D//16]
    tl.store(
        qnv_ptr + pid_zh * (Sq * DP2) + offs_m[:, None] * DP2 + offs_dp[None, :],
        qnv, mask=mmask[:, None],
    )
    tl.store(
        qsc_ptr + pid_zh * (Sq * DP16) + offs_m[:, None] * DP16 + offs_dsc[None, :],
        qsc.to(tl.uint8, bitcast=True), mask=mmask[:, None],
    )
    tl.store(
        donv_ptr + pid_zh * (Sq * DP2) + offs_m[:, None] * DP2 + offs_dp[None, :],
        donv, mask=mmask[:, None],
    )
    tl.store(
        dosc_ptr + pid_zh * (Sq * DP16) + offs_m[:, None] * DP16 + offs_dsc[None, :],
        dosc.to(tl.uint8, bitcast=True), mask=mmask[:, None],
    )
    # store along-M packs: [zh, D, Sq_pad//2] + [zh, D, Sq_pad//16]
    sq2 = Sq_pad // 2
    sq16 = Sq_pad // 16
    tl.store(
        qtnv_ptr + pid_zh * (D * sq2) + offs_d[:, None] * sq2 + offs_mp[None, :],
        qtnv,
    )
    tl.store(
        qtsc_ptr + pid_zh * (D * sq16) + offs_d[:, None] * sq16 + offs_msc[None, :],
        qtsc.to(tl.uint8, bitcast=True),
    )
    tl.store(
        dotnv_ptr + pid_zh * (D * sq2) + offs_d[:, None] * sq2 + offs_mp[None, :],
        dotnv,
    )
    tl.store(
        dotsc_ptr + pid_zh * (D * sq16) + offs_d[:, None] * sq16 + offs_msc[None, :],
        dotsc.to(tl.uint8, bitcast=True),
    )


# ---------------------------------------------------------------------------
# Backward K-side pack-prep: quantize K/V ONCE per (z, kv-head, key-block) so the
# dQ pass (which re-packs K along D, V along D, and K^T along N for every query
# block it loops over) and the dK/dV pass load them instead. All RTN (forward-path
# operands). knv/vnv: [z*hk, Skv, D//2]+scale along D; kTnv: [z*hk, D, Skv_pad//2]
# +scale along N (K^T for the dQ GEMM, padded to a multiple of BLOCK_N).
# ---------------------------------------------------------------------------
@triton.jit
def _flash_bwd_kprep_kernel(
    k_ptr, v_ptr,
    knv_ptr, ksc_ptr, vnv_ptr, vsc_ptr, ktnv_ptr, ktsc_ptr,
    seed, Skv, Skv_pad,
    D: tl.constexpr,
    sk_n, sv_n,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_zhk = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    nmask = offs_n < Skv

    k = tl.load(
        k_ptr + pid_zhk * (Skv * sk_n) + offs_n[:, None] * sk_n + offs_d[None, :],
        mask=nmask[:, None], other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + pid_zhk * (Skv * sv_n) + offs_n[:, None] * sv_n + offs_d[None, :],
        mask=nmask[:, None], other=0.0,
    ).to(tl.float32)
    knv, ksc = _pack_nvfp4_along_k(k, 0, seed, BLOCK_N, D, False)
    vnv, vsc = _pack_nvfp4_along_k(v, 0, seed, BLOCK_N, D, False)
    kTnv, kTsc = _pack_nvfp4_along_k(tl.trans(k), 0, seed, D, BLOCK_N, False)

    DP2: tl.constexpr = D // 2
    DP16: tl.constexpr = D // 16
    NP2: tl.constexpr = BLOCK_N // 2
    NP16: tl.constexpr = BLOCK_N // 16
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)
    offs_np = pid_n * NP2 + tl.arange(0, NP2)
    offs_nsc = pid_n * NP16 + tl.arange(0, NP16)

    tl.store(
        knv_ptr + pid_zhk * (Skv * DP2) + offs_n[:, None] * DP2 + offs_dp[None, :],
        knv, mask=nmask[:, None],
    )
    tl.store(
        ksc_ptr + pid_zhk * (Skv * DP16) + offs_n[:, None] * DP16 + offs_dsc[None, :],
        ksc.to(tl.uint8, bitcast=True), mask=nmask[:, None],
    )
    tl.store(
        vnv_ptr + pid_zhk * (Skv * DP2) + offs_n[:, None] * DP2 + offs_dp[None, :],
        vnv, mask=nmask[:, None],
    )
    tl.store(
        vsc_ptr + pid_zhk * (Skv * DP16) + offs_n[:, None] * DP16 + offs_dsc[None, :],
        vsc.to(tl.uint8, bitcast=True), mask=nmask[:, None],
    )
    sk2 = Skv_pad // 2
    sk16 = Skv_pad // 16
    tl.store(
        ktnv_ptr + pid_zhk * (D * sk2) + offs_d[:, None] * sk2 + offs_np[None, :],
        kTnv,
    )
    tl.store(
        ktsc_ptr + pid_zhk * (D * sk16) + offs_d[:, None] * sk16 + offs_nsc[None, :],
        kTsc.to(tl.uint8, bitcast=True),
    )


# ---------------------------------------------------------------------------
# Native-NVFP4 flash BACKWARD, dK/dV pass. One program per (z, h, key-block n).
# Works entirely in the TRANSPOSED [N, M] score frame so NOT A SINGLE operand is
# transposed in-kernel (the old [M,N] frame paid four tl.trans of fp32 tiles per
# inner step — the dominant backward cost). Q and dO arrive in BOTH packings'
# native layouts: q/do as [.,Sq,D] (contract D) and qt/dot as [.,D,Sq] (contract M,
# pre-transposed once in HBM), so each grad GEMM loads its operand already laid out
# along its own contraction axis — the "pre-pack each operand in the layout each
# GEMM needs" lever.
#
#   sT[n,m] = scale * K[n,:].Q[m,:]          (recompute, contract D)
#   pT[n,m] = exp(sT - lse[m])               (column-softmax via precomputed lse)
#   dV[n,:] += sum_m pT[n,m] * dO[m,:]       (contract M, large-K FP4-friendly)
#   dPt[n,m] = sum_d V[n,d] * dO[m,d]        (contract D, small-K)
#   dSt[n,m] = pT*(dPt - delta[m])*scale
#   dK[n,:] += sum_m dSt[n,m] * Q[m,:]       (contract M, large-K FP4-friendly)
# Gradient operands (pT, dSt, dO) quantized with STOCHASTIC ROUNDING (SR=True).
# ---------------------------------------------------------------------------
@triton.jit
def _flash_bwd_dkdv_kernel(
    qnv_ptr, qsc_ptr, qtnv_ptr, qtsc_ptr,
    donv_ptr, dosc_ptr, dotnv_ptr, dotsc_ptr,
    k_ptr, v_ptr, bias_ptr,
    lse_ptr, delta_ptr,
    dk_ptr, dv_ptr,
    scaling, seed, Sq, Sq_pad, Skv,
    D: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
    sk_n, sv_n, sb_z, sdk_n, sdv_n,
    HAS_BIAS: tl.constexpr, CAUSAL: tl.constexpr, SR: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    zhk = z * HK + (h // (H // HK))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    nmask = offs_n < Skv

    k = tl.load(
        k_ptr + zhk * (Skv * sk_n) + offs_n[:, None] * sk_n + offs_d[None, :],
        mask=nmask[:, None], other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + zhk * (Skv * sv_n) + offs_n[:, None] * sv_n + offs_d[None, :],
        mask=nmask[:, None], other=0.0,
    ).to(tl.float32)
    # K,V packed along D once for this n-block (sT recompute + dPt). RTN (forward
    # path operands, not gradients).
    knv, ksc = _pack_nvfp4_along_k(k, 0, seed, BLOCK_N, D, False)
    vnv, vsc = _pack_nvfp4_along_k(v, 0, seed, BLOCK_N, D, False)

    dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    if CAUSAL:
        lo = tl.maximum(((pid_n * BLOCK_N - (Skv - Sq)) // BLOCK_M) * BLOCK_M, 0)
    else:
        lo = 0

    DP2: tl.constexpr = D // 2
    DP16: tl.constexpr = D // 16
    MP2: tl.constexpr = BLOCK_M // 2
    MP16: tl.constexpr = BLOCK_M // 16
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)
    offs_mp0 = tl.arange(0, MP2)
    offs_msc0 = tl.arange(0, MP16)
    sq2 = Sq_pad // 2
    sq16 = Sq_pad // 16
    for start_m in range(lo, Sq, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mmask = offs_m < Sq
        # load precomputed FP4 packs (quantized once in the pack-prep pass)
        qnv = tl.load(
            qnv_ptr + pid_zh * (Sq * DP2) + offs_m[:, None] * DP2 + offs_dp[None, :],
            mask=mmask[:, None], other=0,
        )
        qsc = tl.load(
            qsc_ptr + pid_zh * (Sq * DP16) + offs_m[:, None] * DP16 + offs_dsc[None, :],
            mask=mmask[:, None], other=0,
        ).to(tl.float8e4nv, bitcast=True)
        mp = (start_m // 2) + offs_mp0
        msc = (start_m // 16) + offs_msc0
        qtnv = tl.load(
            qtnv_ptr + pid_zh * (D * sq2) + offs_d[:, None] * sq2 + mp[None, :],
        )
        qtsc = tl.load(
            qtsc_ptr + pid_zh * (D * sq16) + offs_d[:, None] * sq16 + msc[None, :],
        ).to(tl.float8e4nv, bitcast=True)
        donv = tl.load(
            donv_ptr + pid_zh * (Sq * DP2) + offs_m[:, None] * DP2 + offs_dp[None, :],
            mask=mmask[:, None], other=0,
        )
        dosc = tl.load(
            dosc_ptr + pid_zh * (Sq * DP16) + offs_m[:, None] * DP16 + offs_dsc[None, :],
            mask=mmask[:, None], other=0,
        ).to(tl.float8e4nv, bitcast=True)
        dotnv = tl.load(
            dotnv_ptr + pid_zh * (D * sq2) + offs_d[:, None] * sq2 + mp[None, :],
        )
        dotsc = tl.load(
            dotsc_ptr + pid_zh * (D * sq16) + offs_d[:, None] * sq16 + msc[None, :],
        ).to(tl.float8e4nv, bitcast=True)
        lse = tl.load(lse_ptr + pid_zh * Sq + offs_m, mask=mmask, other=0.0)
        delta = tl.load(delta_ptr + pid_zh * Sq + offs_m, mask=mmask, other=0.0)

        # recompute scores transposed: sT[n,m] = scale * K[n,:] . Q[m,:]
        sT = tl.dot_scaled(knv, ksc, "e2m1", qnv.T, qsc, "e2m1") * scaling
        if HAS_BIAS:
            b = tl.load(bias_ptr + z * sb_z + offs_n, mask=nmask, other=_NEG_INF)
            sT = sT + b[:, None]
        sT = tl.where(nmask[:, None] & mmask[None, :], sT, _NEG_INF)
        if CAUSAL:
            causal_ok = offs_n[:, None] <= (offs_m[None, :] + (Skv - Sq))
            sT = tl.where(causal_ok, sT, _NEG_INF)
        pT = tl.exp(sT - lse[None, :])
        pT = tl.where(sT == _NEG_INF, 0.0, pT)

        # dV += pT @ dO^T.T  (contract M). pT [BLOCK_N, BLOCK_M] (SR), dO^T precomputed.
        pT_q, pT_s = _pack_nvfp4_along_k(pT, start_m, seed, BLOCK_N, BLOCK_M, SR)
        dv = tl.dot_scaled(pT_q, pT_s, "e2m1", dotnv.T, dotsc, "e2m1", acc=dv)

        # dPt[n,m] = sum_d V[n,d] dO[m,d]  (contract D). dO precomputed (SR).
        dpT = tl.dot_scaled(vnv, vsc, "e2m1", donv.T, dosc, "e2m1")

        dsT = pT * (dpT - delta[None, :]) * scaling
        dsT = tl.where(sT == _NEG_INF, 0.0, dsT)

        # dK += dSt @ Q^T.T  (contract M). dSt [BLOCK_N, BLOCK_M] (SR), Q^T precomputed (RTN).
        dsT_q, dsT_s = _pack_nvfp4_along_k(dsT, start_m + 3 * Sq, seed, BLOCK_N, BLOCK_M, SR)
        dk = tl.dot_scaled(dsT_q, dsT_s, "e2m1", qtnv.T, qtsc, "e2m1", acc=dk)

    tl.store(
        dk_ptr + pid_zh * (Skv * sdk_n) + offs_n[:, None] * sdk_n + offs_d[None, :],
        dk.to(dk_ptr.dtype.element_ty), mask=nmask[:, None],
    )
    tl.store(
        dv_ptr + pid_zh * (Skv * sdv_n) + offs_n[:, None] * sdv_n + offs_d[None, :],
        dv.to(dv_ptr.dtype.element_ty), mask=nmask[:, None],
    )


# ---------------------------------------------------------------------------
# Native-NVFP4 flash BACKWARD, dQ pass. One program per (z, h, query-block m).
# Loops keys; dQ[m,:] += sum_n dS[m,n] * K[n,:]  (contract N=keys). dS packed along
# N (SR), K packed along N (i.e. K^T [D, BLOCK_N]).
# ---------------------------------------------------------------------------
@triton.jit
def _flash_bwd_dq_kernel(
    q_ptr, do_ptr, bias_ptr,
    knv_ptr, ksc_ptr, vnv_ptr, vsc_ptr, ktnv_ptr, ktsc_ptr,
    lse_ptr, delta_ptr, dq_ptr,
    scaling, seed, Sq, Skv, Skv_pad,
    D: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
    sq_n, sdo_n, sb_z, sdq_n,
    HAS_BIAS: tl.constexpr, CAUSAL: tl.constexpr, SR: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    zhk = z * HK + (h // (H // HK))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mmask = offs_m < Sq

    q = tl.load(
        q_ptr + pid_zh * (Sq * sq_n) + offs_m[:, None] * sq_n + offs_d[None, :],
        mask=mmask[:, None], other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + pid_zh * (Sq * sdo_n) + offs_m[:, None] * sdo_n + offs_d[None, :],
        mask=mmask[:, None], other=0.0,
    ).to(tl.float32)
    lse = tl.load(lse_ptr + pid_zh * Sq + offs_m, mask=mmask, other=0.0)
    delta = tl.load(delta_ptr + pid_zh * Sq + offs_m, mask=mmask, other=0.0)
    qnv, qsc = _pack_nvfp4_along_k(q, 0, seed, BLOCK_M, D, False)
    do_q, do_s = _pack_nvfp4_along_k(do, pid_m * BLOCK_M, seed, BLOCK_M, D, SR)

    DP2: tl.constexpr = D // 2
    DP16: tl.constexpr = D // 16
    NP2: tl.constexpr = BLOCK_N // 2
    NP16: tl.constexpr = BLOCK_N // 16
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)
    offs_np0 = tl.arange(0, NP2)
    offs_nsc0 = tl.arange(0, NP16)
    sk2 = Skv_pad // 2
    sk16 = Skv_pad // 16

    dq = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    if CAUSAL:
        hi = tl.minimum(Skv, (pid_m * BLOCK_M + BLOCK_M) + (Skv - Sq))
    else:
        hi = Skv

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < Skv
        # precomputed K/V packs (K-side pack-prep): knv/vnv along D, K^T along N
        knv = tl.load(
            knv_ptr + zhk * (Skv * DP2) + offs_n[:, None] * DP2 + offs_dp[None, :],
            mask=nmask[:, None], other=0,
        )
        ksc = tl.load(
            ksc_ptr + zhk * (Skv * DP16) + offs_n[:, None] * DP16 + offs_dsc[None, :],
            mask=nmask[:, None], other=0,
        ).to(tl.float8e4nv, bitcast=True)
        vnv = tl.load(
            vnv_ptr + zhk * (Skv * DP2) + offs_n[:, None] * DP2 + offs_dp[None, :],
            mask=nmask[:, None], other=0,
        )
        vsc = tl.load(
            vsc_ptr + zhk * (Skv * DP16) + offs_n[:, None] * DP16 + offs_dsc[None, :],
            mask=nmask[:, None], other=0,
        ).to(tl.float8e4nv, bitcast=True)
        np_ = (start_n // 2) + offs_np0
        nsc = (start_n // 16) + offs_nsc0
        kTnv = tl.load(
            ktnv_ptr + zhk * (D * sk2) + offs_d[:, None] * sk2 + np_[None, :],
        )
        kTsc = tl.load(
            ktsc_ptr + zhk * (D * sk16) + offs_d[:, None] * sk16 + nsc[None, :],
        ).to(tl.float8e4nv, bitcast=True)

        s = tl.dot_scaled(qnv, qsc, "e2m1", knv.T, ksc, "e2m1") * scaling
        if HAS_BIAS:
            b = tl.load(bias_ptr + z * sb_z + offs_n, mask=nmask, other=_NEG_INF)
            s = s + b[None, :]
        s = tl.where(nmask[None, :] & mmask[:, None], s, _NEG_INF)
        if CAUSAL:
            causal_ok = offs_n[None, :] <= (offs_m[:, None] + (Skv - Sq))
            s = tl.where(causal_ok, s, _NEG_INF)
        p = tl.exp(s - lse[:, None])
        p = tl.where(s == _NEG_INF, 0.0, p)

        dp = tl.dot_scaled(do_q, do_s, "e2m1", vnv.T, vsc, "e2m1")
        ds = p * (dp - delta[:, None]) * scaling
        ds = tl.where(s == _NEG_INF, 0.0, ds)

        # dQ += dS @ K  (contract N). dS [BLOCK_M, BLOCK_N] (SR), K^T precomputed.
        ds_q, ds_s = _pack_nvfp4_along_k(ds, start_n + pid_m * Skv, seed, BLOCK_M, BLOCK_N, SR)
        dq = tl.dot_scaled(ds_q, ds_s, "e2m1", kTnv.T, kTsc, "e2m1", acc=dq)

    tl.store(
        dq_ptr + pid_zh * (Sq * sdq_n) + offs_m[:, None] * sdq_n + offs_d[None, :],
        dq.to(dq_ptr.dtype.element_ty), mask=mmask[:, None],
    )


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
        out,  # dummy lse ptr (STORE_LSE=False)
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
        STORE_LSE=False,
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
    block_m, block_n, num_warps, num_stages = _resolve_fwd_tiles(
        d, block_m, block_n, num_warps, num_stages
    )
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
    return_lse: bool = False,
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
        return_lse: also return the per-row logsumexp ``[Z*H, Sq]`` (fp32) so the
            backward can skip recomputing it (the FA2 backward prep's QK^T pass).

    Returns:
        ``[Z, H, Sq, D]`` in ``query.dtype`` (and the LSE tensor if ``return_lse``).
    """
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    assert h % hk == 0 and h // hk == num_key_value_groups
    assert d % 16 == 0 and d in (128, 256)
    block_m, block_n, num_warps, num_stages = _resolve_fwd_tiles(
        d, block_m, block_n, num_warps, num_stages
    )
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
    lse = (
        torch.empty(z * h, s_q, device=query.device, dtype=torch.float32)
        if return_lse else out
    )

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
            lse,
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
            STORE_LSE=return_lse,
            BLOCK_M=block_m, BLOCK_N=block_n,
            DP2=d // 2, DP16=d // 16,
            NP2=block_n // 2, NP16=block_n // 16,
            num_warps=num_warps, num_stages=num_stages,
        )

    _run()
    if return_lse:
        return out.reshape(z, h, s_q, d), lse
    return out.reshape(z, h, s_q, d)


# ---------------------------------------------------------------------------
# Full forward + native-NVFP4 backward as a torch.autograd.Function.
# ---------------------------------------------------------------------------
def _run_bwd(
    q, k, v, do, o, bias,
    z, h, hk, s_q, s_kv, d, scaling, causal, sr,
    block_m, block_n, num_warps, num_stages,
    lse=None,
):
    """Native-NVFP4 backward. q/do/o: [Z*H,Sq,D]; k/v: [Z*Hk,Skv,D] (hp). Returns
    dq [Z*H,Sq,D], dk/dv [Z*H,Skv,D] (per query head; GQA-reduced by the caller).

    If ``lse`` (the forward's per-row logsumexp, [Z*H,Sq]) is supplied, the prep
    kernel reuses it instead of recomputing it with a full FP4 QK^T pass."""
    have_lse = lse is not None
    dq = torch.empty(z * h, s_q, d, device=q.device, dtype=torch.float32)
    dk = torch.empty(z * h, s_kv, d, device=q.device, dtype=torch.float32)
    dv = torch.empty(z * h, s_kv, d, device=q.device, dtype=torch.float32)
    if not have_lse:
        lse = torch.empty(z * h, s_q, device=q.device, dtype=torch.float32)
    delta = torch.empty(z * h, s_q, device=q.device, dtype=torch.float32)
    seed = torch.randint(0, 2**31 - 1, (1,), device=q.device).item() if sr else 0
    # The recompute kernels hold several fp32 [BLOCK, D] tiles + their FP4 packs in
    # SRAM at once; D=256 needs small query tiles to fit the 99KB budget.
    block_m = min(block_m, 64)
    # dkdv loops over query blocks and is the backward hotspot. With the m-block
    # operands (Q/dO and their transposes) pre-packed once in the pack-prep pass,
    # the loop body only holds the per-n K/V packs plus two small in-loop SR packs
    # (pT, dSt), so a NARROW key tile (BLOCK_N=32) + 8 warps + deep pipelining
    # (3 stages) is dramatically faster than the old wide-tile config: the small
    # footprint lets the scheduler overlap the SR-pack ALU with the FP4 GEMMs and
    # the K/V loads instead of spilling. ~8x faster than BLOCK_N=128 / 16 warps.
    dkdv_block_n = 32 if d <= 256 else 32
    dkdv_warps = 8
    dkdv_stages = 3
    # dq loops over key blocks (already cheap); keep the conservative key tile. Two
    # pipeline stages overlap the per-block K/V loads + dS SR-pack with the FP4 GEMMs
    # (the loop holds only narrow tiles, so the extra stage's SRAM fits) — ~15% faster
    # than the single-stage launch; deeper pipelines give nothing more here.
    dq_block_n = 64 if d >= 256 else min(block_n, 128)
    dq_warps = max(num_warps, 8)
    dq_stages = 2

    bdummy = bias if bias is not None else q
    sb_z = bias.stride(0) if bias is not None else 0
    has_bias = bias is not None

    _flash_bwd_prep_kernel[(triton.cdiv(s_q, block_m), z * h)](
        q, k, do, o, bdummy, lse, delta,
        scaling, seed, s_q, s_kv,
        D=d, H=h, HK=hk,
        sq_n=q.stride(1), sk_n=k.stride(1), sdo_n=do.stride(1), so_n=o.stride(1),
        sb_z=sb_z,
        HAS_BIAS=has_bias, CAUSAL=causal, HAVE_LSE=have_lse,
        BLOCK_M=block_m, BLOCK_N=dq_block_n,
        num_warps=dq_warps, num_stages=num_stages,
    )

    # Pack-prep: quantize the dK/dV pass's m-block-local operands ONCE here (q/qT
    # RTN, do/doT SR) instead of re-quantizing each Skv/BLOCK_N times in the loop.
    # Along-M (qT/doT) buffers are padded to a multiple of BLOCK_M so the dkdv loop
    # can load full m-tiles without M-axis masking; padded rows are masked out of
    # every GEMM via the sT==-inf (pT/dsT=0) path.
    # A narrow BLOCK_M (32) + 2 pipeline stages packs the most elements/SM and keeps
    # the philox SR ALU overlapped with the stores; the along-M/along-D pack layouts
    # are group-16 so independent of this producing tile size (the dkdv loop reads
    # them at its own BLOCK_M). ~7x faster than the wide-tile config.
    pp_block_m = 32
    s_q_pad = _next_mult(s_q, max(block_m, pp_block_m))
    qnv_p = q.new_empty(z * h, s_q, d // 2, dtype=torch.uint8)
    qsc_p = q.new_zeros(z * h, s_q, d // 16, dtype=torch.uint8)
    donv_p = q.new_empty(z * h, s_q, d // 2, dtype=torch.uint8)
    dosc_p = q.new_zeros(z * h, s_q, d // 16, dtype=torch.uint8)
    qtnv_p = q.new_empty(z * h, d, s_q_pad // 2, dtype=torch.uint8)
    qtsc_p = q.new_zeros(z * h, d, s_q_pad // 16, dtype=torch.uint8)
    dotnv_p = q.new_empty(z * h, d, s_q_pad // 2, dtype=torch.uint8)
    dotsc_p = q.new_zeros(z * h, d, s_q_pad // 16, dtype=torch.uint8)
    _flash_bwd_packprep_kernel[(triton.cdiv(s_q, pp_block_m), z * h)](
        q, do,
        qnv_p, qsc_p, qtnv_p, qtsc_p,
        donv_p, dosc_p, dotnv_p, dotsc_p,
        seed, s_q, s_q_pad,
        D=d, sq_n=q.stride(1), sdo_n=do.stride(1),
        SR=sr, BLOCK_M=pp_block_m,
        num_warps=8, num_stages=2,
    )
    qsc_p = qsc_p.view(torch.float8_e4m3fn)
    dosc_p = dosc_p.view(torch.float8_e4m3fn)
    qtsc_p = qtsc_p.view(torch.float8_e4m3fn)
    dotsc_p = dotsc_p.view(torch.float8_e4m3fn)

    # K-side pack-prep: pack K/V once per kv-head (knv/vnv along D, K^T along N) so
    # the dQ pass loads them instead of re-quantizing K/V/K^T for every query block.
    kprep_block_n = 64
    s_kv_pad = _next_mult(s_kv, kprep_block_n)
    knv_p = k.new_empty(z * hk, s_kv, d // 2, dtype=torch.uint8)
    ksc_p = k.new_zeros(z * hk, s_kv, d // 16, dtype=torch.uint8)
    vnv_p = k.new_empty(z * hk, s_kv, d // 2, dtype=torch.uint8)
    vsc_p = k.new_zeros(z * hk, s_kv, d // 16, dtype=torch.uint8)
    ktnv_p = k.new_empty(z * hk, d, s_kv_pad // 2, dtype=torch.uint8)
    ktsc_p = k.new_zeros(z * hk, d, s_kv_pad // 16, dtype=torch.uint8)
    _flash_bwd_kprep_kernel[(triton.cdiv(s_kv, kprep_block_n), z * hk)](
        k, v,
        knv_p, ksc_p, vnv_p, vsc_p, ktnv_p, ktsc_p,
        seed, s_kv, s_kv_pad,
        D=d, sk_n=k.stride(1), sv_n=v.stride(1),
        BLOCK_N=kprep_block_n,
        num_warps=dq_warps, num_stages=num_stages,
    )
    ksc_pv = ksc_p.view(torch.uint8)
    vsc_pv = vsc_p.view(torch.uint8)
    ktsc_pv = ktsc_p.view(torch.uint8)

    _flash_bwd_dkdv_kernel[(triton.cdiv(s_kv, dkdv_block_n), z * h)](
        qnv_p, qsc_p.view(torch.uint8), qtnv_p, qtsc_p.view(torch.uint8),
        donv_p, dosc_p.view(torch.uint8), dotnv_p, dotsc_p.view(torch.uint8),
        k, v, bdummy, lse, delta, dk, dv,
        scaling, seed, s_q, s_q_pad, s_kv,
        D=d, H=h, HK=hk,
        sk_n=k.stride(1), sv_n=v.stride(1),
        sb_z=sb_z, sdk_n=dk.stride(1), sdv_n=dv.stride(1),
        HAS_BIAS=has_bias, CAUSAL=causal, SR=sr,
        BLOCK_M=block_m, BLOCK_N=dkdv_block_n,
        num_warps=dkdv_warps, num_stages=dkdv_stages,
    )
    _flash_bwd_dq_kernel[(triton.cdiv(s_q, block_m), z * h)](
        q, do, bdummy,
        knv_p, ksc_pv, vnv_p, vsc_pv, ktnv_p, ktsc_pv,
        lse, delta, dq,
        scaling, seed, s_q, s_kv, s_kv_pad,
        D=d, H=h, HK=hk,
        sq_n=q.stride(1), sdo_n=do.stride(1),
        sb_z=sb_z, sdq_n=dq.stride(1),
        HAS_BIAS=has_bias, CAUSAL=causal, SR=sr,
        BLOCK_M=block_m, BLOCK_N=dq_block_n,
        num_warps=dq_warps, num_stages=dq_stages,
    )
    return dq, dk, dv


class _NVFP4FlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, key, value, scaling, causal, num_key_value_groups,
        key_pad_bias, sr, block_m, block_n, num_warps, num_stages,
    ):
        z, h, s_q, d = query.shape
        _, hk, s_kv, _ = key.shape
        out, lse = nvfp4_flash_attention(
            query, key, value, scaling, causal=causal,
            num_key_value_groups=num_key_value_groups, key_pad_bias=key_pad_bias,
            block_m=block_m, block_n=block_n, num_warps=num_warps, num_stages=num_stages,
            return_lse=True,
        )
        bias = None
        if key_pad_bias is not None:
            bias = key_pad_bias.to(torch.float32).contiguous()
        ctx.save_for_backward(
            query.reshape(z * h, s_q, d).contiguous(),
            key.reshape(z * hk, s_kv, d).contiguous(),
            value.reshape(z * hk, s_kv, d).contiguous(),
            out.reshape(z * h, s_q, d),
            bias if bias is not None else torch.empty(0, device=query.device),
            lse,
        )
        ctx.dims = (z, h, hk, s_q, s_kv, d)
        ctx.scaling = scaling
        ctx.causal = causal
        ctx.sr = sr
        ctx.tiles = (block_m, block_n, num_warps, num_stages)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, o, bias, lse = ctx.saved_tensors
        z, h, hk, s_q, s_kv, d = ctx.dims
        block_m, block_n, num_warps, num_stages = ctx.tiles
        bias = bias if ctx.has_bias else None
        do = grad_out.reshape(z * h, s_q, d).contiguous()
        dq, dk, dv = _run_bwd(
            q, k, v, do, o, bias,
            z, h, hk, s_q, s_kv, d, ctx.scaling, ctx.causal, ctx.sr,
            block_m, block_n, 4, 1, lse=lse,
        )
        dq = dq.reshape(z, h, s_q, d).to(grad_out.dtype)
        ng = h // hk
        if ng > 1:
            dk = dk.reshape(z, hk, ng, s_kv, d).sum(2).to(grad_out.dtype)
            dv = dv.reshape(z, hk, ng, s_kv, d).sum(2).to(grad_out.dtype)
        else:
            dk = dk.reshape(z, hk, s_kv, d).to(grad_out.dtype)
            dv = dv.reshape(z, hk, s_kv, d).to(grad_out.dtype)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def nvfp4_flash_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    causal: bool = False,
    num_key_value_groups: int = 1,
    key_pad_bias: torch.Tensor | None = None,
    stochastic_rounding: bool = True,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """Native-NVFP4 flash attention with a differentiable native-NVFP4 backward.

    Forward and all four backward GEMMs (dV, dP, dK, dQ) run as real 5th-gen FP4
    ``tl.dot_scaled`` ops. Gradient operands (P, dS, dO) are quantized with
    stochastic rounding when ``stochastic_rounding`` (the convergence-critical
    knob — see ``utils/nvfp4_training``). q:[Z,H,Sq,D], k/v:[Z,Hk,Skv,D]; D in
    {128,256}; supports causal and GQA. Returns [Z,H,Sq,D] in query.dtype.
    """
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    assert h % hk == 0 and h // hk == num_key_value_groups
    assert d in (128, 256)
    return _NVFP4FlashAttn.apply(
        query, key, value, scaling, causal, num_key_value_groups,
        key_pad_bias, stochastic_rounding, block_m, block_n, num_warps, num_stages,
    )
