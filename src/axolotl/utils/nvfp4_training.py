"""NVFP4-GEMM training: real FP4 forward + backward GEMMs on Blackwell.

This is a throughput feature, not a memory feature: master weights and
optimizer state stay bf16/fp32; only the GEMM operands are NVFP4. It mirrors
the FP8 training surface (module-swap on ``nn.Linear``) but is self-owned —
accelerate/torchao have no NVFP4 training path.

Foundations verified on sm_120 (RTX 5090, torchao 0.18):
  * the three GEMMs (fprop, dgrad, wgrad) run via ``torch._scaled_mm`` over
    NVFP4 operands and match dequant-matmul exactly;
  * two-level (per-tensor) scaling is REQUIRED — gradients otherwise underflow
    the per-block e4m3 scale floor (2^-9) and round to zero;
  * the quant->GEMM boundary compiles with zero graph breaks (the 2.9-3.6x
    speedup over bf16 only materializes under torch.compile).

Convergence recipe knobs (stochastic rounding on gradients, random Hadamard
transform on wgrad inputs) attach at the ``_quantize`` seam — see
``QuantPolicy``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

LOG = logging.getLogger(__name__)

# Blackwell: B200/B300 = sm_100/sm_103, consumer (RTX 50xx, RTX PRO 6000) = sm_120.
# FP4 tensor cores exist on both families; do NOT blanket-gate on sm_100 only
# (the recurring hazard where the consumer SM is silently dropped).
_MIN_FP4_CAPABILITIES = ((10, 0), (12, 0))

_BLOCK_SIZE = 16  # NVFP4 is fixed at block_size 16
# torch._scaled_mm packs 2 FP4/byte and requires the packed contraction dim
# (= logical/2) to be divisible by 16, so logical contraction must be a
# multiple of 32. The token dimension (M) is padded to this alignment.
_GEMM_ALIGN = 32

# Paper ablation sweet spot: 16x16 random Hadamard (4x4 worse, 128x128 marginal).
# 16 | 32 (_GEMM_ALIGN) so the block tiles the padded contraction dim cleanly.
_HAD_DIM = 16
# FP4 E2M1 normalized-mantissa exponent range: levels are 2^e * {1, 1.5} for
# e in {0,1,2} (above the denormal step), so the half-step granularity at which
# SR dithers lives in this clamp.
_FP4_EXP_LO, _FP4_EXP_HI = 0, 2


def nvfp4_supported() -> tuple[bool, str]:
    """Return (ok, reason). ok=False means refuse with `reason`."""
    if not torch.cuda.is_available():
        return False, "NVFP4 training requires CUDA"
    cap = torch.cuda.get_device_capability()
    major = cap[0]
    # sm_100/103 (Blackwell datacenter) and sm_120 (Blackwell consumer) both
    # carry 5th-gen FP4 tensor cores. Gate on major>=10 with the known minors.
    if major < 10:
        return (
            False,
            f"NVFP4 training requires Blackwell FP4 tensor cores "
            f"(compute capability >= 10.0), got {cap[0]}.{cap[1]}",
        )
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import (  # noqa: F401
            NVFP4Tensor,
            _addmm_nvfp4_dispatch,
            per_tensor_amax_to_scale,
        )
    except ImportError as exc:
        return False, f"NVFP4 training requires torchao >= 0.18 ({exc})"
    if torch.__version__ < "2.8":
        return False, "NVFP4 training requires torch >= 2.8"
    return True, ""


@dataclass
class QuantPolicy:
    """How a tensor is quantized for one operand of one GEMM.

    The convergence-recipe agent extends ``stochastic`` (gradient SR) and
    ``hadamard`` (RHT on wgrad inputs); the base path implements round-to-
    nearest with mandatory two-level per-tensor scaling.
    """

    stochastic: bool = False
    hadamard: bool = False


def _build_base_dh() -> torch.Tensor:
    """Orthonormal 16x16 Hadamard (H_16/4) times a fixed random ±1 diagonal.

    (D H)^T (D H) = H^T D^T D H = H^T H = 16 I, so dividing by sqrt(16)=4 gives
    an orthonormal rotation: applied to the contraction dim of both wgrad
    operands it cancels, while the FP4 quant in between sees Gaussian-ized values.
    """
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < _HAD_DIM:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    h = h / (_HAD_DIM**0.5)
    gen = torch.Generator().manual_seed(0)
    sign = torch.randint(0, 2, (_HAD_DIM,), generator=gen).to(torch.float64) * 2 - 1
    return sign.unsqueeze(1) * h  # D @ H, orthonormal


# Built once at import (the Generator call must stay out of the traced region —
# a torch.Generator() inside the backward graph forces a dynamo break under
# fullgraph). The hot path only casts this constant to device/dtype, which is
# fully traceable.
_BASE_DH = _build_base_dh()


def _apply_rht(t: torch.Tensor) -> torch.Tensor:
    """Block-Hadamard rotation along the last (contraction) dim of ``t``.

    Last dim is the contraction axis for both wgrad operands as fed to
    ``_quantize`` (``gt`` directly, ``xp`` via ``b.t()``), so the same rotation
    on both cancels the product.
    """
    dh = _BASE_DH.to(device=t.device, dtype=t.dtype)
    lead = t.shape[:-1]
    blocks = t.shape[-1] // _HAD_DIM
    return (t.reshape(*lead, blocks, _HAD_DIM) @ dh.t()).reshape(*lead, t.shape[-1])


def _sr_dither(t: torch.Tensor, per_tensor_scale: torch.Tensor) -> torch.Tensor:
    """Uniform dither over one FP4 step so the subsequent RTN realizes SR.

    Adding uniform noise of width = one quantization step before RTN is exactly
    unbiased stochastic rounding. The step is computed in the original domain by
    mirroring torchao's two-level scaling: rotate to the per-block-scaled value
    ``v``, take the FP4 half-step ``2^floor(log2|v|)`` (mantissa is 1 bit, levels
    are 2^e*{1,1.5}; clamp e to the E2M1 normal range), then de-scale.
    """
    from torchao.prototype.mx_formats.constants import (
        F4_E2M1_MAX,
        F8E4M3_MAX,
    )

    eps = torch.finfo(torch.float8_e4m3fn).tiny
    x = t.float().reshape(t.shape[0], -1, _BLOCK_SIZE)
    block_scale = torch.amax(torch.abs(x), dim=-1) / F4_E2M1_MAX
    scaled_block = block_scale / per_tensor_scale
    sbf8 = (
        torch.clamp(scaled_block, min=eps, max=F8E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    recip = (1.0 / per_tensor_scale) / sbf8  # original -> fp4-grid multiplier
    v = x * recip.unsqueeze(-1)
    e = torch.floor(torch.log2(v.abs().clamp(min=eps))).clamp(_FP4_EXP_LO, _FP4_EXP_HI)
    step_orig = (2.0**e) / recip.unsqueeze(-1)  # one FP4 step, original domain
    u = (torch.rand_like(v) - 0.5) * step_orig
    return (x + u).reshape(t.shape).to(t.dtype)


# Row-chunk size for the load-time quant of large frozen weights. torchao's
# to_nvfp4 upcasts the WHOLE tensor to f32 at once (~2x the bf16 size of scratch),
# which OOMs lm_head/embedding-sized weights on a near-full card. Quantizing in
# row blocks (with a globally-fixed per-tensor scale so the result is bit-
# identical) bounds that scratch to one block. 128 = torchao's swizzle row tile,
# so block-row qdata/scale concatenate exactly to the whole-tensor layout.
_QUANT_CHUNK_ROWS = 8192
_QUANT_CHUNK_MIN_ROWS = 2 * _QUANT_CHUNK_ROWS  # below this the scratch is small


def _to_nvfp4_chunked(t, per_tensor_scale, act_quant_kwargs):
    """``NVFP4Tensor.to_nvfp4`` over a 2D weight, row-block by row-block.

    The f32 quant scratch is bounded to ``_QUANT_CHUNK_ROWS`` rows instead of the
    whole tensor. Bit-identical to the single-shot quant: NVFP4 blocks lie along
    the last dim (rows are independent) and the per-tensor scale is fixed across
    blocks, so concatenating the per-block qdata/scale reproduces the full layout
    (block rows align to the 128-row swizzle tile). Falls back to a single call
    for small tensors / non-2D inputs.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    def _one(x):
        return NVFP4Tensor.to_nvfp4(
            x.contiguous(),
            block_size=_BLOCK_SIZE,
            per_tensor_scale=per_tensor_scale,
            act_quant_kwargs=act_quant_kwargs,
        )

    if t.dim() != 2 or t.shape[0] < _QUANT_CHUNK_MIN_ROWS:
        return _one(t)

    qparts, sparts, ctx = [], [], None
    for i in range(0, t.shape[0], _QUANT_CHUNK_ROWS):
        c = _one(t[i : i + _QUANT_CHUNK_ROWS])
        if ctx is None:
            ctx = c.__tensor_flatten__()[1]
        qparts.append(c.qdata)
        sparts.append(c.scale)
        del c
    inner = {
        "qdata": torch.cat(qparts, dim=0),
        "scale": torch.cat(sparts, dim=0),
        "per_tensor_scale": per_tensor_scale,
    }
    return NVFP4Tensor.__tensor_unflatten__(inner, ctx, None, None)


def _quantize(t: torch.Tensor, policy: QuantPolicy):
    """Quantize a high-precision tensor to an NVFP4Tensor (along its last dim).

    Two-level scaling (per-tensor fp32 scale + per-block e4m3 scale) is always
    applied: without it small-magnitude tensors (gradients) underflow to zero.
    ``policy.hadamard`` rotates the contraction dim (RHT, wgrad inputs);
    ``policy.stochastic`` dithers for stochastic rounding (gradient operands).
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    t = t.contiguous()
    if policy.hadamard:
        t = _apply_rht(t).contiguous()
    per_tensor_scale = per_tensor_amax_to_scale(torch.max(torch.abs(t)))
    if policy.stochastic:
        t = _sr_dither(t, per_tensor_scale).contiguous()
    # RHT/SR rewrite the whole tensor up front (no per-block-row independence), so
    # only the plain frozen-weight quant takes the memory-bounded chunked path.
    if policy.hadamard or policy.stochastic:
        return NVFP4Tensor.to_nvfp4(
            t, block_size=_BLOCK_SIZE, per_tensor_scale=per_tensor_scale
        )
    return _to_nvfp4_chunked(t, per_tensor_scale, None)


def _fp4_mm(a_hp: torch.Tensor, b_hp: torch.Tensor, a_pol, b_pol) -> torch.Tensor:
    """C[M,N] = a_hp[M,K] @ b_hp[K,N] with both operands quantized to NVFP4.

    ``_scaled_mm`` wants TN layout: ``a`` row-major, ``b`` such that
    ``b.qdata.t()`` is contiguous. The latter is produced by quantizing the
    [N,K] form (contiguous, along K) and transposing.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

    a_q = _quantize(a_hp, a_pol)
    b_q = _quantize(b_hp.t().contiguous(), b_pol).t()
    return _addmm_nvfp4_dispatch(a_q, b_q, torch.ops.aten.mm.default)


def _pad_to_block(t: torch.Tensor, dim: int) -> tuple[torch.Tensor, int]:
    """Zero-pad ``dim`` up to a multiple of 32; return (padded, original_size).

    The token dimension (M) is the wgrad contraction axis and is rarely a
    multiple of 32. Zero rows contribute zero to the wgrad accumulation and are
    sliced back off the forward output / dgrad.
    """
    n = t.shape[dim]
    rem = n % _GEMM_ALIGN
    if rem == 0:
        return t, n
    pad = _GEMM_ALIGN - rem
    pad_spec = [0, 0] * (t.dim() - dim - 1) + [0, pad]
    return torch.nn.functional.pad(t, pad_spec), n


class NVFP4LinearFunction(torch.autograd.Function):
    """Linear with FP4 GEMMs in forward (fprop) and backward (dgrad + wgrad).

    Master weight stays high-precision (the saved ``weight``); only GEMM
    operands are quantized. Per the NVFP4 recipe, gradient operands get
    stochastic rounding and wgrad inputs get RHT (via ``QuantPolicy``).
    """

    @staticmethod
    def forward(ctx, x, weight, bias, recipe):
        # x:[*, K]  weight:[N, K]  out:[*, N]
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x2d_p, m = _pad_to_block(x2d, 0)

        w_pol = QuantPolicy()
        a_pol = QuantPolicy()
        out = _fp4_mm(x2d_p, weight.t(), a_pol, w_pol)[:m]  # [M,K]@[K,N]->[M,N]
        if bias is not None:
            out = out + bias

        ctx.save_for_backward(x2d, weight)
        ctx.recipe = recipe
        ctx.has_bias = bias is not None
        ctx.x_shape = orig_shape
        return out.reshape(*orig_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight = ctx.saved_tensors  # x2d:[M,K] weight:[N,K]
        recipe = ctx.recipe
        g = grad_out.reshape(-1, weight.shape[0])  # [M, N]

        grad_x = grad_w = grad_bias = None

        # gradient operands use stochastic rounding; wgrad inputs use RHT
        g_pol = QuantPolicy(stochastic=recipe.stochastic_rounding)
        rht_pol = QuantPolicy(
            stochastic=recipe.stochastic_rounding, hadamard=recipe.hadamard
        )

        if ctx.needs_input_grad[0]:
            # dgrad: grad_x[M,K] = g[M,N] @ weight[N,K]   (contraction N)
            g_p, m = _pad_to_block(g, 0)
            grad_x = _fp4_mm(g_p, weight, g_pol, QuantPolicy())[:m]
            grad_x = grad_x.reshape(ctx.x_shape)

        if ctx.needs_input_grad[1]:
            # wgrad: grad_w[N,K] = g.t()[N,M] @ x[M,K]    (contraction M, RHT)
            gt, _ = _pad_to_block(g.t().contiguous(), 1)  # [N, M_pad]
            xp, _ = _pad_to_block(x2d, 0)  # [M_pad, K]
            grad_w = _fp4_mm(gt, xp, rht_pol, rht_pol)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = g.sum(dim=0)

        return grad_x, grad_w, grad_bias, None


@dataclass
class NVFP4Recipe:
    """Training-precision recipe (distinct from the QAT/PTQ quantization block)."""

    stochastic_rounding: bool = True
    hadamard: bool = True


class NVFP4Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` whose GEMMs run in NVFP4.

    Weight/bias remain ``nn.Parameter`` in their original dtype; quantization
    happens inside each GEMM. Requires ``in_features`` and ``out_features``
    divisible by 16 — callers must exclude layers that violate this (lm_head,
    odd-vocab embeddings) from the swap.
    """

    def __init__(self, weight, bias, recipe: NVFP4Recipe):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.recipe = recipe
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, x):
        return NVFP4LinearFunction.apply(x, self.weight, self.bias, self.recipe)

    @classmethod
    def from_linear(cls, linear: nn.Linear, recipe: NVFP4Recipe) -> "NVFP4Linear":
        return cls(linear.weight, linear.bias, recipe)


class NVFP4FrozenBaseFunction(torch.autograd.Function):
    """Forward GEMM against a pre-quantized FROZEN weight; dgrad only.

    For the NVFP4-QLoRA path: the base weight is stored packed in FP4 (no
    high-precision master copy → ~3.5x weight memory savings) and is frozen, so
    there is no weight gradient — only dgrad to propagate into the trainable
    LoRA adapters and earlier layers.
    """

    @staticmethod
    def forward(ctx, x, w_q, recipe):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x2d_p, m = _pad_to_block(x2d, 0)
        # w_q is the stored FP4 weight ([N,K], blocked along K); w_q.t() is the
        # [K,N] B operand. Route through _addmm (NOT F.linear): torchao's
        # dynamic-act F.linear path can't carry the two-level per-tensor scale
        # (it asserts per_tensor_scale is None on both operands).
        out = _addmm_nvfp4_dispatch(
            _quantize(x2d_p, QuantPolicy()), w_q.t(), torch.ops.aten.mm.default
        )[:m]
        ctx.w_q = w_q
        ctx.recipe = recipe
        ctx.x_shape = orig_shape
        return out.reshape(*orig_shape[:-1], w_q.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        w_q = ctx.w_q
        grad_x = None
        if ctx.needs_input_grad[0]:
            g = grad_out.reshape(-1, w_q.shape[0])
            g_p, m = _pad_to_block(g, 0)
            # dgrad = g @ W; dequant the stored weight for the contraction-along-N
            # GEMM (the stored layout is quantized along K for the forward).
            w_hp = w_q.dequantize(torch.bfloat16)
            grad_x = _fp4_mm(
                g_p, w_hp, QuantPolicy(stochastic=ctx.recipe.stochastic_rounding),
                QuantPolicy(),
            )[:m]
            grad_x = grad_x.reshape(ctx.x_shape)
        return grad_x, None, None


_FSDP_NVFP4_CLS = None


def _fsdp_nvfp4_class():
    """torchao's NVFP4Tensor subclassed with FSDP2 all-gather hooks (cached).

    Built lazily so importing this module never needs the torchao.prototype
    NVFP4Tensor symbol at top level.
    """
    global _FSDP_NVFP4_CLS
    if _FSDP_NVFP4_CLS is not None:
        return _FSDP_NVFP4_CLS
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    class FSDPNVFP4Tensor(NVFP4Tensor):
        # The frozen FP4 base shards along dim 0: qdata and scale both split by
        # row; the global per_tensor_scale (computed once over the whole weight,
        # identical on every rank) is replicated. Reconstruction concatenates the
        # row-shards — verified bit-exact against the unsharded tensor.
        def fsdp_pre_all_gather(self, mesh):
            return (self.qdata, self.scale), (
                self.__tensor_flatten__()[1],
                self.per_tensor_scale,
            )

        def fsdp_post_all_gather(
            self, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            qdata, scale = all_gather_outputs
            ctx, per_tensor_scale = metadata
            if out is not None:
                return
            inner = {"qdata": qdata, "scale": scale, "per_tensor_scale": per_tensor_scale}
            rebuilt = type(self).__tensor_unflatten__(inner, ctx, None, None)
            return rebuilt, (qdata, scale)

    _FSDP_NVFP4_CLS = FSDPNVFP4Tensor
    return _FSDP_NVFP4_CLS


def _to_fsdp_nvfp4(w_q):
    """Re-wrap an NVFP4Tensor as the FSDP-hooked subclass (same inner data)."""
    sub = _fsdp_nvfp4_class()
    ctx = w_q.__tensor_flatten__()[1]
    inner = {
        "qdata": w_q.qdata,
        "scale": w_q.scale,
        "per_tensor_scale": w_q.per_tensor_scale,
    }
    return sub.__tensor_unflatten__(inner, ctx, None, None)


class NVFP4FrozenBaseLinear(nn.Module):
    """Frozen base linear whose weight is stored packed in FP4 (QLoRA base).

    Unlike ``NVFP4Linear`` (which keeps a high-precision trainable master weight
    — throughput only), this drops the master copy for ~3.5x weight memory
    savings. It is FROZEN (no weight gradient) and is meant to sit under LoRA
    adapters. Bias, if any, stays high-precision.
    """

    def __init__(self, w_q, bias, recipe: NVFP4Recipe):
        super().__init__()
        # Buffer (not Parameter): frozen/no-grad, but still enters state_dict so
        # the FP4-packed base survives save/load — otherwise resume silently
        # reinitializes the base. NVFP4Tensor round-trips through torch.save.
        self.register_buffer("w_q", w_q)
        self.bias = bias  # already an nn.Parameter from the source linear
        self.recipe = recipe
        self.in_features = w_q.shape[1]
        self.out_features = w_q.shape[0]

    def forward(self, x):
        out = NVFP4FrozenBaseFunction.apply(x, self.w_q, self.recipe)
        return out if self.bias is None else out + self.bias

    @property
    def weight(self) -> torch.Tensor:
        # Read-only dequantized [N,K] view for PEFT (DoRA weight-norm, delta
        # compute, base_layer.weight reads). Writes to it (in-process merge:
        # base_layer.weight.data += delta) do NOT persist into the FP4 store —
        # NVFP4 LoRA bases must merge via the offline CLI (axolotl merge-lora),
        # which the patch_manager enforces by skipping the FP4 swap under merge.
        return self.w_q.dequantize(torch.bfloat16)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, recipe: NVFP4Recipe, *, fsdp: bool = False
    ) -> "NVFP4FrozenBaseLinear":
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            QuantizeTensorToNVFP4Kwargs,
            per_tensor_amax_to_scale,
        )

        w = linear.weight.detach()
        pts = per_tensor_amax_to_scale(torch.max(torch.abs(w)))
        w_q = _to_nvfp4_chunked(
            w.contiguous(),
            pts,
            QuantizeTensorToNVFP4Kwargs(block_size=_BLOCK_SIZE),
        )
        # FSDP2 needs the all-gather hooks to shard the FP4 base by row.
        if fsdp:
            w_q = _to_fsdp_nvfp4(w_q)
        return cls(w_q, linear.bias, recipe)


def _embedding_to_nvfp4(weight: torch.Tensor):
    """Quantize an [vocab, hidden] embedding weight to a stored NVFP4Tensor.

    Blocked along the hidden dim (last), two-level scaled — same packing as the
    frozen base linear. The result is the SHARED store routed to both the
    embedding lookup and (when tied) the lm_head GEMM.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        QuantizeTensorToNVFP4Kwargs,
        per_tensor_amax_to_scale,
    )

    w = weight.detach().contiguous()
    pts = per_tensor_amax_to_scale(torch.max(torch.abs(w)))
    return _to_nvfp4_chunked(
        w, pts, QuantizeTensorToNVFP4Kwargs(block_size=_BLOCK_SIZE)
    )


def _nvfp4_embedding_gather(w_q, input):
    """Embedding lookup that dequantizes only the gathered rows of ``w_q``.

    Avoids materializing the full [vocab, hidden] bf16 table (lm_head-sized) just
    to gather a handful of rows. NVFP4 blocks lie along the hidden dim, so each
    vocab row is a self-contained slice of qdata/scale; gathering the rows of the
    packed buffers and dequantizing that subset is bit-identical to dequantizing
    the whole table and gathering. Frozen weight, so padding_idx (gradient-only)
    is a no-op in forward. Returns None (caller falls back) when the layout isn't
    safe to row-slice (swizzled scales) or anything is unexpected.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    # Under torch.compile keep the plain full dequant: the subtensor rebuild and
    # row gather graph-break, and the table dequant is what the compiled graph
    # already fuses. The memory win matters at load/eager, not in the hot graph.
    if torch.compiler.is_compiling():
        return None

    try:
        names, ctx = w_q.__tensor_flatten__()
        if ctx.get("is_swizzled_scales"):
            return None
        flat = input.reshape(-1)
        sub = NVFP4Tensor.__tensor_unflatten__(
            {
                "qdata": w_q.qdata[flat],
                "scale": w_q.scale[flat],
                "per_tensor_scale": w_q.per_tensor_scale,
            },
            ctx,
            None,
            None,
        )
        rows = sub.dequantize(torch.bfloat16)
        return rows.reshape(*input.shape, rows.shape[-1])
    except Exception:  # any layout/version surprise -> full-dequant fallback
        return None


class NVFP4Embedding(nn.Module):
    """Input embedding whose weight is stored packed in FP4 (W4A16 lookup).

    The lookup gathers rows by integer index — no activation quant — so forward
    is ``F.embedding`` over the dequantized weight. FROZEN only: an FP4-stored
    weight has no high-precision master to receive gradients (use QAT for a
    trainable FP4 embedding). Hidden dim must be divisible by 16.
    """

    def __init__(self, w_q, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        # Buffer (not Parameter): frozen, but must enter the state_dict so the
        # FP4-packed embedding survives save/load (NVFP4Tensor round-trips).
        self.register_buffer("w_q", w_q)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(self, input):
        # Dequantize ONLY the gathered rows, not the whole [vocab, hidden] table:
        # the full bf16 dequant (+ its f32 scratch) is lm_head-sized and OOMs on a
        # near-full card. NVFP4 blocks lie along the hidden dim so each row is
        # self-contained — slicing qdata/scale by the unique looked-up rows and
        # dequantizing that subset is bit-identical to gathering after a full
        # dequant. Falls back to the full path if the scales are swizzled (row
        # slicing would break the tile) or anything is unexpected.
        gathered = _nvfp4_embedding_gather(self.w_q, input)
        if gathered is not None:
            return gathered
        w = self.w_q.dequantize(torch.bfloat16)
        return torch.nn.functional.embedding(input, w, self.padding_idx)

    @property
    def weight(self) -> torch.Tensor:
        # Read-only dequantized [vocab, hidden] view; writes don't persist.
        return self.w_q.dequantize(torch.bfloat16)

    @classmethod
    def from_embedding(cls, emb: nn.Embedding) -> "NVFP4Embedding":
        return cls.from_weight(
            emb.weight, emb.num_embeddings, emb.embedding_dim, emb.padding_idx
        )

    @classmethod
    def from_weight(
        cls, weight, num_embeddings, embedding_dim, padding_idx=None
    ) -> "NVFP4Embedding":
        return cls(
            _embedding_to_nvfp4(weight), num_embeddings, embedding_dim, padding_idx
        )


class NVFP4TiedLMHead(nn.Module):
    """lm_head GEMM over a SHARED FP4 store (tied-embedding case).

    Holds no weight of its own: it reads the SAME ``NVFP4Tensor`` that backs the
    input :class:`NVFP4Embedding`, so the dequantized weight is bit-identical for
    the lookup and the GEMM. Frozen (no wgrad); dgrad flows via the storage-mode
    frozen-base function.
    """

    def __init__(self, embedding: NVFP4Embedding, bias, recipe: NVFP4Recipe):
        super().__init__()
        # Reference the embedding's buffer directly so the two roles always read
        # the same FP4 store (the embedding owns it in the state_dict).
        self._embedding = embedding
        self.bias = bias
        self.recipe = recipe
        self.in_features = embedding.w_q.shape[1]
        self.out_features = embedding.w_q.shape[0]

    @property
    def w_q(self):
        return self._embedding.w_q

    def forward(self, x):
        out = NVFP4FrozenBaseFunction.apply(x, self.w_q, self.recipe)
        return out if self.bias is None else out + self.bias

    @property
    def weight(self) -> torch.Tensor:
        return self.w_q.dequantize(torch.bfloat16)


class NVFP4ComputeBaseFunction(torch.autograd.Function):
    """Frozen-base linear with FP4 compute on BOTH base GEMMs, zero per-step
    base-quant prologue.

    The base is frozen, so its NVFP4 operands are quantized ONCE at load (not
    per step like FFT). fprop needs the base blocked along K (contraction) and
    dgrad needs it blocked along N (contraction), so two pre-quantized layouts
    are stored. Each is a real ``torch._scaled_mm`` FP4 GEMM; only the activation
    / incoming-gradient operand is quantized per step. No wgrad (frozen).
    """

    @staticmethod
    def forward(ctx, x, w_fprop, w_dgrad, recipe):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x2d_p, m = _pad_to_block(x2d, 0)
        out = _addmm_nvfp4_dispatch(
            _quantize(x2d_p, QuantPolicy()), w_fprop, torch.ops.aten.mm.default
        )[:m]
        ctx.w_dgrad = w_dgrad
        ctx.recipe = recipe
        ctx.x_shape = orig_shape
        ctx.out_features = w_fprop.shape[1]
        return out.reshape(*orig_shape[:-1], ctx.out_features)

    @staticmethod
    def backward(ctx, grad_out):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

        grad_x = None
        if ctx.needs_input_grad[0]:
            g = grad_out.reshape(-1, ctx.out_features)
            g_p, m = _pad_to_block(g, 0)
            # dgrad: gx[M,K] = g[M,N] @ W[N,K]; W pre-quantized along N (contraction)
            g_q = _quantize(g_p, QuantPolicy(stochastic=ctx.recipe.stochastic_rounding))
            grad_x = _addmm_nvfp4_dispatch(
                g_q, ctx.w_dgrad, torch.ops.aten.mm.default
            )[:m]
            grad_x = grad_x.reshape(ctx.x_shape)
        return grad_x, None, None, None


class NVFP4ComputeBaseLinear(nn.Module):
    """Frozen LoRA base with FP4 compute on fprop+dgrad and no per-step base quant.

    Stores two pre-quantized NVFP4 layouts of the frozen weight (fprop: blocked
    along K; dgrad: blocked along N) as buffers. ~1.75x weight memory vs bf16
    (two FP4 copies + scales) but the base GEMMs run pure FP4 with the quant
    prologue paid once at load — faster than re-quantizing the base every step.
    Adapters (added by PEFT around this base_layer) stay high-precision.
    """

    def __init__(self, w_fprop, w_dgrad, bias, recipe: NVFP4Recipe):
        super().__init__()
        self.register_buffer("w_fprop", w_fprop)
        self.register_buffer("w_dgrad", w_dgrad)
        self.bias = bias
        self.recipe = recipe
        self.in_features = w_fprop.shape[0]
        self.out_features = w_fprop.shape[1]

    def forward(self, x):
        out = NVFP4ComputeBaseFunction.apply(
            x, self.w_fprop, self.w_dgrad, self.recipe
        )
        return out if self.bias is None else out + self.bias

    @property
    def weight(self) -> torch.Tensor:
        # Read-only dequantized [N,K] for PEFT; writes don't persist (see
        # NVFP4FrozenBaseLinear.weight). w_fprop is _quantize(W).t() ([K,N]),
        # so .t().dequantize() recovers [N,K].
        return self.w_fprop.t().dequantize(torch.bfloat16)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, recipe: NVFP4Recipe
    ) -> "NVFP4ComputeBaseLinear":
        w = linear.weight.detach()  # [N, K]
        # fprop b-operand represents W.T ([K,N]): quantize W then transpose.
        w_fprop = _quantize(w, QuantPolicy()).t()
        # dgrad b-operand represents W ([N,K]) blocked along N: quantize W.T then transpose.
        w_dgrad = _quantize(w.t().contiguous(), QuantPolicy()).t()
        return cls(w_fprop, w_dgrad, linear.bias, recipe)


# NVFP4 two-level global scale target: map a tensor's amax onto the product of
# the FP4 and FP8 maxima so the per-block e4m3 scales use their full range.
_NVFP4_GLOBAL_AMAX = 448.0 * 6.0  # F8E4M3_MAX * F4_E2M1_MAX
_MSLK_AVAILABLE: bool | None = None


def _mslk_available() -> bool:
    """Whether the MSLK fused NVFP4 quant kernel is importable (cached).

    MSLK lives only in the TE/perf venv, not the base experimental venv, so the
    fast path is strictly optional — callers fall back to the torchao quantizer.
    """
    global _MSLK_AVAILABLE
    import os

    if os.environ.get("AXOLOTL_NVFP4_NO_MSLK") == "1":
        return False
    if _MSLK_AVAILABLE is None:
        try:
            from mslk.quantize.triton.fp4_quantize import (  # noqa: F401
                triton_quantize_nvfp4,
            )

            _MSLK_AVAILABLE = True
        except Exception:  # ImportError or a triton/runtime probe failure
            _MSLK_AVAILABLE = False
    return _MSLK_AVAILABLE


def _swizzled_scale_shape(m: int, k: int) -> tuple[int, int]:
    """Shape of MSLK's swizzled e4m3 block-scale tensor for a [m, k] input.

    Rows pad to 128, columns (= k/16 block scales) pad to 4 — the tcgen05 MMA
    scale-factor tile. Needed by the custom-op fake impls (the scale tensor's
    size can't be derived from the qdata under tracing).
    """
    rounded_m = (m + 127) // 128 * 128
    n_blocks = k // _BLOCK_SIZE
    rounded_k = (n_blocks + 3) // 4 * 4
    return rounded_m, rounded_k


# MSLK's Triton quant kernels registered as opaque custom ops. Inductor's
# decompose_triton_kernel_wrapper_functional pass crashes if it traces INTO a raw
# Triton kernel in the dynamo graph; a registered op with a fake impl is a black
# box it compiles AROUND (no decompose, no graph break, no eager fallback). The
# concrete impl runs the kernel eagerly under compile — only the GEMM and the
# surrounding pointwise ops fuse, which is the win (the quant is already ~0.03ms).
@torch.library.custom_op("axolotl_nvfp4::quantize_two_level", mutates_args=())
def _mslk_quantize_op(
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from mslk.quantize.triton.fp4_quantize import triton_quantize_nvfp4

    t = t.contiguous()
    amax = torch.amax(torch.abs(t)).to(torch.float32)
    global_scale = _NVFP4_GLOBAL_AMAX / torch.clamp(amax, min=1e-12)
    q, s = triton_quantize_nvfp4(t, global_scale)
    return (
        q.view(torch.float4_e2m1fn_x2),
        s.view(torch.float8_e4m3fn),
        (1.0 / global_scale).to(t.dtype),
    )


@_mslk_quantize_op.register_fake
def _(t):
    m, k = t.shape
    rm, rk = _swizzled_scale_shape(m, k)
    return (
        t.new_empty(m, k // 2, dtype=torch.float4_e2m1fn_x2),
        t.new_empty(rm, rk, dtype=torch.float8_e4m3fn),
        t.new_empty((), dtype=t.dtype),
    )


@torch.library.custom_op("axolotl_nvfp4::quantize_single_level", mutates_args=())
def _mslk_quantize_sl_op(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from mslk.quantize.triton.fp4_quantize import triton_quantize_nvfp4

    q, s = triton_quantize_nvfp4(t.contiguous(), None)
    return q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e4m3fn)


@_mslk_quantize_sl_op.register_fake
def _(t):
    m, k = t.shape
    rm, rk = _swizzled_scale_shape(m, k)
    return (
        t.new_empty(m, k // 2, dtype=torch.float4_e2m1fn_x2),
        t.new_empty(rm, rk, dtype=torch.float8_e4m3fn),
    )


def _mslk_quantize(t: torch.Tensor):
    """Quantize ``t`` (along its last dim) to NVFP4 via MSLK's fused Triton kernel.

    Returns ``(qdata, scale, inv_global_scale)`` ready for ``torch._scaled_mm``:
    ``qdata`` is ``float4_e2m1fn_x2`` packed, ``scale`` is the swizzled e4m3 block
    scale, and ``inv_global_scale = 1/global_scale`` is folded back into the GEMM
    output (two-level scaling). Routed through a registered custom op so it stays
    opaque-in-graph under torch.compile (see ``_mslk_quantize_op``).
    """
    return _mslk_quantize_op(t)


def _mslk_dequant(qdata, scale, inv_gs, shape, dtype=torch.bfloat16) -> torch.Tensor:
    """Dequantize MSLK-packed FP4 buffers back to a [*shape] hp tensor.

    MSLK emits swizzled e4m3 block scales (same layout as torchao's triton
    quant) and folds the two-level rescale into ``inv_gs = 1/global_scale``, so
    wrap into an NVFP4Tensor with ``per_tensor_scale=inv_gs`` and reuse torchao's
    dequant. ``shape`` is the logical [M,K] of the original hp tensor (qdata is
    [M,K/2] packed); used only to sanity-check the unpacked size.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    t = NVFP4Tensor(
        qdata,
        scale,
        _BLOCK_SIZE,
        dtype,
        per_tensor_scale=inv_gs.to(torch.float32),
        is_swizzled_scales=True,
    )
    out = t.dequantize(dtype)
    return out.reshape(shape)


def _mslk_scaled_mm(aq, a_scale, a_inv_gs, bq, b_scale, b_inv_gs, out_dtype):
    """``A @ B`` in FP4 from pre-quantized MSLK operands, two-level rescaled.

    ``aq`` is [M, K/2], ``bq`` is the B operand quantized as [N, K/2] so ``bq.t()``
    gives the [K, N] contraction layout ``_scaled_mm`` wants (TN).
    """
    out = torch._scaled_mm(
        aq,
        bq.t(),
        bias=None,
        out_dtype=out_dtype,
        scale_a=a_scale,
        scale_b=b_scale,
    )
    return out * (a_inv_gs * b_inv_gs)


def _prequant_sl(x_orig: torch.Tensor, x2d: torch.Tensor):
    """Single-level activation quant: reuse a fused-norm's pre-quantized result
    when ``x_orig`` is that norm's output (identity cache hit), else quantize
    ``x2d`` now. The cache stores the 2D ([M, K/2]) quant, drop-in for the
    ``_mslk_quantize_sl`` output.

    The cache is an identity-keyed WeakTensorKeyDictionary (untraceable) and only
    populated by the fuse_rmsnorm path (default OFF). Under torch.compile skip the
    lookup and quantize directly so the quant stays an opaque op in-graph; eager
    keeps the cache fast-path.
    """
    if torch.compiler.is_compiling():
        return _mslk_quantize_sl(x2d)
    from axolotl.kernels.nvfp4_rmsnorm import get_prequant

    cached = get_prequant(x_orig)
    if cached is not None:
        return cached
    return _mslk_quantize_sl(x2d)


def _mslk_quantize_sl(t: torch.Tensor):
    """Single-level NVFP4 quant (``global_scale=None``): MSLK fuses the amax into
    the kernel and bakes it into the block scales, so there is no separate
    per-tensor scale to compute (no amax reduction) or fold. ~3.6x faster than
    the two-level path. Safe ONLY for the forward ACTIVATION — its magnitudes are
    large enough that the e4m3 block scales don't underflow. Small weights and
    gradients DO underflow single-level, so they keep two-level. Routed through a
    registered custom op so it stays opaque-in-graph under torch.compile."""
    return _mslk_quantize_sl_op(t)


def _mslk_fprop_mm(aq, a_scale, bq, b_scale, b_inv_gs, out_dtype):
    """fprop GEMM: single-level activation (``aq``/``a_scale``, no per-tensor
    scale) @ two-level weight (``bq``/``b_scale``/``b_inv_gs``). Only the weight's
    per-tensor scale is folded post-GEMM (the activation has none)."""
    out = torch._scaled_mm(
        aq, bq.t(), bias=None, out_dtype=out_dtype, scale_a=a_scale, scale_b=b_scale
    )
    return out * b_inv_gs


class NVFP4FastComputeBaseFunction(torch.autograd.Function):
    """Compute-base fprop/dgrad using MSLK-quantized FP4 operands.

    Same math as :class:`NVFP4ComputeBaseFunction` (frozen base, two pre-quantized
    layouts, fprop + dgrad as ``_scaled_mm`` FP4 GEMMs, no wgrad) but the per-step
    activation/gradient quant uses MSLK's fused Triton kernel instead of the
    torchao quantizer — the prologue that otherwise dominates an unfused (whole-
    model-compiled) step.
    """

    @staticmethod
    def forward(ctx, x, wq_f, wsc_f, w_inv_f, wq_d, wsc_d, w_inv_d, out_features):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        # fprop: single-level activation (no per-step amax) @ two-level weight.
        xq, xsc = _prequant_sl(x, x2d)
        out = _mslk_fprop_mm(xq, xsc, wq_f, wsc_f, w_inv_f, x.dtype)
        ctx.dgrad = (wq_d, wsc_d, w_inv_d)
        ctx.x_shape = orig_shape
        ctx.out_features = out_features
        return out.reshape(*orig_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = None
        if ctx.needs_input_grad[0]:
            wq_d, wsc_d, w_inv_d = ctx.dgrad
            g = grad_out.reshape(-1, ctx.out_features)
            gq, gsc, g_inv = _mslk_quantize(g)
            grad_x = _mslk_scaled_mm(
                gq, gsc, g_inv, wq_d, wsc_d, w_inv_d, grad_out.dtype
            )
            grad_x = grad_x.reshape(ctx.x_shape)
        return grad_x, None, None, None, None, None, None, None


class NVFP4FastComputeBaseLinear(nn.Module):
    """Frozen LoRA base with MSLK-fused FP4 compute on fprop + dgrad.

    Drop-in for :class:`NVFP4ComputeBaseLinear` (same memory: two FP4 weight
    layouts + scales) chosen when MSLK is available. ``recipe`` SR/RHT are not
    applied here (MSLK's plain quant is round-to-nearest); on sm_120 the recipe
    is disabled anyway, and the win is throughput.
    """

    def __init__(self, w_f, w_d, bias, out_features, recipe: NVFP4Recipe):
        super().__init__()
        wq_f, wsc_f, w_inv_f = w_f  # fprop weight: two-level
        wq_d, wsc_d, w_inv_d = w_d  # dgrad weight: two-level
        self.register_buffer("wq_f", wq_f)
        self.register_buffer("wsc_f", wsc_f)
        self.register_buffer("w_inv_f", w_inv_f)
        self.register_buffer("wq_d", wq_d)
        self.register_buffer("wsc_d", wsc_d)
        self.register_buffer("w_inv_d", w_inv_d)
        self.bias = bias
        self.recipe = recipe
        self.out_features = out_features
        # wq_f packs the fprop layout [N, K/2]; K = in_features = 2 * packed cols.
        self.in_features = wq_f.shape[-1] * 2

    def forward(self, x):
        out = NVFP4FastComputeBaseFunction.apply(
            x,
            self.wq_f,
            self.wsc_f,
            self.w_inv_f,
            self.wq_d,
            self.wsc_d,
            self.w_inv_d,
            self.out_features,
        )
        return out if self.bias is None else out + self.bias

    @property
    def weight(self) -> torch.Tensor:
        # Read-only dequantized [N,K] for PEFT; writes don't persist (see
        # NVFP4FrozenBaseLinear.weight). Rebuilt from the fprop MSLK buffers.
        return _mslk_dequant(
            self.wq_f, self.wsc_f, self.w_inv_f, (self.out_features, self.in_features)
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, recipe: NVFP4Recipe
    ) -> "NVFP4FastComputeBaseLinear":
        w = linear.weight.detach()  # [N, K]
        # Both layouts two-level (small weights underflow single-level). fprop
        # B = wq_f.t(): quant W ([N,K]); dgrad B = wq_d.t(): quant W.T.
        w_f = _mslk_quantize(w)
        w_d = _mslk_quantize(w.t().contiguous())
        return cls(w_f, w_d, linear.bias, w.shape[0], recipe)


class NVFP4FastFrozenBaseFunction(torch.autograd.Function):
    """Storage-mode fprop/dgrad using MSLK-quantized FP4 operands.

    Same math as :class:`NVFP4FrozenBaseFunction` (single FP4 weight layout, ~3.5x
    weight memory; dgrad dequantizes the weight) but the per-step activation /
    gradient quant uses MSLK's fused Triton kernel instead of the torchao
    quantizer. The weight is stored ONLY in the fprop layout; dgrad dequantizes it
    to bf16 then re-quantizes both operands — so this keeps the single-layout
    memory win of storage mode, trading dgrad FLOPs for memory (vs the two-layout
    compute mode which stores a second FP4 dgrad layout to skip the dequant).
    """

    @staticmethod
    def forward(ctx, x, wq, wsc, w_inv, out_features, in_features):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        xq, xsc, x_inv = _mslk_quantize(x2d)
        out = _mslk_scaled_mm(xq, xsc, x_inv, wq, wsc, w_inv, x.dtype)
        ctx.wstore = (wq, wsc, w_inv)
        ctx.x_shape = orig_shape
        ctx.out_features = out_features
        ctx.in_features = in_features
        return out.reshape(*orig_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = None
        if ctx.needs_input_grad[0]:
            wq, wsc, w_inv = ctx.wstore
            g = grad_out.reshape(-1, ctx.out_features)
            # dgrad = g @ W; the stored layout is blocked along K (fprop), so
            # dequantize to bf16 for the contraction-along-N GEMM.
            w_hp = _mslk_dequant(
                wq, wsc, w_inv, (ctx.out_features, ctx.in_features), grad_out.dtype
            )
            gp, m = _pad_to_block(g, 0)
            gq, gsc, g_inv = _mslk_quantize(gp)
            wdq, wdsc, wd_inv = _mslk_quantize(w_hp.t().contiguous())  # B = W.t()
            grad_x = _mslk_scaled_mm(
                gq, gsc, g_inv, wdq, wdsc, wd_inv, grad_out.dtype
            )[:m]
            grad_x = grad_x.reshape(ctx.x_shape)
        return grad_x, None, None, None, None, None


class NVFP4FastFrozenBaseLinear(nn.Module):
    """Storage-mode frozen base (single FP4 weight, ~3.5x memory) with MSLK-fused
    per-step quant.

    Drop-in for :class:`NVFP4FrozenBaseLinear`, chosen when MSLK is available. Same
    single-layout FP4 storage; dgrad dequantizes the weight (no second layout).
    ``recipe`` SR/RHT are not applied (MSLK's quant is round-to-nearest), matching
    :class:`NVFP4FastComputeBaseLinear`.
    """

    def __init__(self, w_store, bias, out_features, in_features, recipe: NVFP4Recipe):
        super().__init__()
        wq, wsc, w_inv = w_store
        self.register_buffer("wq", wq)
        self.register_buffer("wsc", wsc)
        self.register_buffer("w_inv", w_inv)
        self.bias = bias
        self.recipe = recipe
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        out = NVFP4FastFrozenBaseFunction.apply(
            x, self.wq, self.wsc, self.w_inv, self.out_features, self.in_features
        )
        return out if self.bias is None else out + self.bias

    @property
    def weight(self) -> torch.Tensor:
        # Read-only dequantized [N,K] for PEFT; writes don't persist (see
        # NVFP4FrozenBaseLinear.weight).
        return _mslk_dequant(
            self.wq, self.wsc, self.w_inv, (self.out_features, self.in_features)
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, recipe: NVFP4Recipe
    ) -> "NVFP4FastFrozenBaseLinear":
        w = linear.weight.detach()  # [N, K]
        w_store = _mslk_quantize(w)  # single fprop layout, blocked along K
        return cls(w_store, linear.bias, w.shape[0], w.shape[1], recipe)


# Base modules whose forward is an FP4 GEMM (no high-precision .weight to read).
# The fused LoRA kernels detect these to route the base GEMM through FP4 instead
# of reading base_layer.weight (which only NVFP4Linear, the hp mode, exposes).
def _nvfp4_base_classes() -> tuple:
    return (
        NVFP4Linear,
        NVFP4FrozenBaseLinear,
        NVFP4FastFrozenBaseLinear,
        NVFP4ComputeBaseLinear,
        NVFP4FastComputeBaseLinear,
    )


def is_nvfp4_base(module) -> bool:
    return isinstance(module, _nvfp4_base_classes())


def nvfp4_base_fprop(x: torch.Tensor, base) -> torch.Tensor:
    """``x @ W.T`` in FP4 for any native NVFP4 base module (2D ``x`` [M, K]).

    Mirrors each module's forward GEMM but as a plain (no-autograd) call so the
    fused LoRA autograd Functions can invoke it inside their own forward. Pads M
    to the FP4 alignment and slices it back.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

    xp, m = _pad_to_block(x, 0)
    if isinstance(base, NVFP4FastComputeBaseLinear):
        xq, xsc = _mslk_quantize_sl(xp)
        out = _mslk_fprop_mm(xq, xsc, base.wq_f, base.wsc_f, base.w_inv_f, x.dtype)
    elif isinstance(base, NVFP4FastFrozenBaseLinear):
        xq, xsc = _mslk_quantize_sl(xp)
        out = _mslk_fprop_mm(xq, xsc, base.wq, base.wsc, base.w_inv, x.dtype)
    elif isinstance(base, NVFP4ComputeBaseLinear):
        out = _addmm_nvfp4_dispatch(
            _quantize(xp, QuantPolicy()), base.w_fprop, torch.ops.aten.mm.default
        )
    elif isinstance(base, NVFP4FrozenBaseLinear):
        out = _addmm_nvfp4_dispatch(
            _quantize(xp, QuantPolicy()), base.w_q.t(), torch.ops.aten.mm.default
        )
    else:  # NVFP4Linear (hp): high-precision master weight, per-step requant
        out = _fp4_mm(xp, base.weight.t(), QuantPolicy(), QuantPolicy())
    return out[:m]


def nvfp4_base_dgrad(g: torch.Tensor, base) -> torch.Tensor:
    """``g @ W`` in FP4 (the base contribution to the input gradient), 2D ``g``."""
    from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

    gp, m = _pad_to_block(g, 0)
    sr = QuantPolicy(stochastic=base.recipe.stochastic_rounding)
    if isinstance(base, NVFP4FastComputeBaseLinear):
        gq, gsc, g_inv = _mslk_quantize(gp)
        out = _mslk_scaled_mm(
            gq, gsc, g_inv, base.wq_d, base.wsc_d, base.w_inv_d, g.dtype
        )
    elif isinstance(base, NVFP4ComputeBaseLinear):
        out = _addmm_nvfp4_dispatch(
            _quantize(gp, sr), base.w_dgrad, torch.ops.aten.mm.default
        )
    elif isinstance(base, NVFP4FastFrozenBaseLinear):
        # single FP4 layout: dequantize the stored weight for the dgrad GEMM.
        w_hp = _mslk_dequant(
            base.wq, base.wsc, base.w_inv, (base.out_features, base.in_features), g.dtype
        )
        out = _fp4_mm(gp, w_hp, sr, QuantPolicy())
    elif isinstance(base, NVFP4FrozenBaseLinear):
        out = _fp4_mm(gp, base.w_q.dequantize(g.dtype), sr, QuantPolicy())
    else:  # NVFP4Linear (hp)
        out = _fp4_mm(gp, base.weight, sr, QuantPolicy())
    return out[:m]


def _is_swappable(module: nn.Linear) -> bool:
    # Both in and out are contraction dims across fprop/dgrad, so both must meet
    # the _scaled_mm packed-contraction rule (logical %32, not just block %16) —
    # an out_features of 16 packs to 8 and trips "trailing dim divisible by 16".
    return (
        module.in_features % _GEMM_ALIGN == 0
        and module.out_features % _GEMM_ALIGN == 0
    )


def _embedding_swappable(emb: nn.Embedding) -> bool:
    # Only the hidden dim is a quant block axis for the lookup (no GEMM), so the
    # NVFP4 block_size 16 is the only constraint — vocab is unrestricted.
    return emb.embedding_dim % _BLOCK_SIZE == 0


# Decoder-block norm attributes whose output feeds an NVFP4 base linear (qkv /
# gate-up). q_norm/k_norm (head-dim, feed attention) and the final norm (feeds
# lm_head, not an NVFP4 base) are intentionally excluded.
_FUSE_NORM_NAMES = frozenset(
    {
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    }
)


def convert_norms_to_nvfp4_fused(model: nn.Module) -> int:
    """Swap decoder-block RMSNorms for the fused RMSNorm->NVFP4-quant module.

    The fused norm emits the normalized activation AND its NVFP4 quant in one
    kernel, so the consuming base linear skips re-quantizing. The gamma convention
    (plain ``weight`` vs zero-centered ``1 + weight``) is detected per norm, and
    every swap is VERIFIED against the original on a probe before committing —
    a mismatch reverts the swap (never silently diverges). No-op if MSLK is missing.
    """
    try:
        from axolotl.kernels.nvfp4_rmsnorm import NVFP4FusedRMSNorm
    except Exception as exc:  # MSLK / triton unavailable
        LOG.warning("NVFP4 fused RMSNorm unavailable (%s); skipping norm fusion", exc)
        return 0

    parents = dict(model.named_modules())
    swapped = 0
    skipped = 0
    for name, module in list(model.named_modules()):
        attr = name.rsplit(".", 1)[-1]
        if attr not in _FUSE_NORM_NAMES:
            continue
        if not type(module).__name__.endswith("RMSNorm") or not hasattr(
            module, "weight"
        ):
            continue
        w = module.weight
        if not w.is_cuda:  # the fused kernel + the verify probe need CUDA
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = parents.get(parent_name)
        if parent is None:
            continue
        fused = NVFP4FusedRMSNorm.from_norm(module)
        # Verify the fused norm reproduces the original before committing — guards
        # against any unhandled gamma convention silently corrupting the forward.
        with torch.no_grad():
            probe = torch.randn(8, w.shape[-1], device=w.device, dtype=w.dtype)
            ref = module(probe).float()
            rel = (fused(probe).float() - ref).norm() / (ref.norm() + 1e-9)
        if rel > 0.05:
            LOG.warning("NVFP4: skip fused norm %s (rel-err %.3f)", name, rel.item())
            skipped += 1
            continue
        setattr(parent, attr, fused)
        swapped += 1
    LOG.info(
        "NVFP4 training: fused %d decoder RMSNorms (%d skipped on verify)",
        swapped,
        skipped,
    )
    return swapped


def convert_to_nvfp4_training(
    model: nn.Module,
    recipe: NVFP4Recipe | None = None,
    *,
    exclude: tuple[str, ...] = ("lm_head", "embed_tokens"),
    skip_first_n_blocks: int = 0,
    skip_last_n_blocks: int = 0,
) -> int:
    """Swap eligible ``nn.Linear`` layers for ``NVFP4Linear`` in place.

    Sensitive layers stay high-precision per the convergence recipe: ``exclude``
    name fragments (lm_head/embeddings) plus the first/last N transformer blocks
    (block index parsed from the ``...layers.<i>...`` path). Returns the number
    of layers swapped.

    NOTE: ``skip_first_n_blocks``/``skip_last_n_blocks`` need the total block
    count; that policy is finalized by the integration layer. Here we only honor
    explicit name-fragment exclusion and dims%16 eligibility; block-range
    exclusion is applied by the caller via ``exclude`` for now.
    """
    recipe = recipe or NVFP4Recipe()
    swapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if any(frag in name for frag in exclude):
            continue
        if not _is_swappable(module):
            LOG.warning(
                "NVFP4: skipping %s (in=%d out=%d not both divisible by 32)",
                name,
                module.in_features,
                module.out_features,
            )
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, leaf, NVFP4Linear.from_linear(module, recipe))
        swapped += 1
    LOG.info("NVFP4 training: swapped %d linear layers", swapped)
    return swapped


def swap_frozen_linear_to_nvfp4(
    model: nn.Module,
    name: str,
    recipe: NVFP4Recipe | None = None,
    *,
    base_mode: str = "compute",
) -> bool:
    """Swap a single bare frozen ``nn.Linear`` (e.g. an un-targeted lm_head in a
    LoRA run) for the matching NVFP4 base module.

    The LoRA base converter only touches ``lora.Linear`` modules; a frozen
    lm_head that isn't a LoRA target stays a bare ``nn.Linear`` and is invisible
    to it. This swaps that bare module in place using the same three base modes:
    ``compute`` (default), ``storage``, or ``hp``. Returns True if swapped.
    """
    recipe = recipe or NVFP4Recipe()
    try:
        module = model.get_submodule(name)
    except AttributeError:
        return False
    if not isinstance(module, nn.Linear) or not _is_swappable(module):
        return False

    fast = _mslk_available()
    if base_mode == "compute":
        cls = NVFP4FastComputeBaseLinear if fast else NVFP4ComputeBaseLinear
        build = lambda src: cls.from_linear(src, recipe)  # noqa: E731
    elif base_mode == "storage":
        if fast:
            build = lambda src: NVFP4FastFrozenBaseLinear.from_linear(  # noqa: E731
                src, recipe
            )
        else:
            build = lambda src: NVFP4FrozenBaseLinear.from_linear(  # noqa: E731
                src, recipe, fsdp=False
            )
    else:
        # hp mode keeps a high-precision trainable master weight (no quant), so
        # there is no FP4 transient to stream around; swap in place.
        module.weight.requires_grad_(False)
        _set_submodule(model, name, NVFP4Linear.from_linear(module, recipe))
        LOG.info("NVFP4 training: swapped frozen %s (mode=%s)", name, base_mode)
        return True

    _stream_quantize_swap(model, name, module, build)
    LOG.info("NVFP4 training: swapped frozen %s (mode=%s)", name, base_mode)
    return True


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    setattr(parent, name.rsplit(".", 1)[-1], new_module)


def _reclaim_gpu() -> None:
    """Sync then release freed GPU blocks back to the allocator.

    Sync first: the quant kernels still read the source weight async, and
    expandable_segments would otherwise hand a freed-but-in-flight block to the
    next allocation (use-after-free / illegal access).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _stream_quantize_swap(model, name, source, build):
    """Quantize ``source`` to its NVFP4 module keeping the transient peak small.

    The lm_head/embed/vision swaps run AFTER the full model is on the GPU, so on a
    near-full card the source bf16 + the new FP4 layouts (+ quant scratch) coexist
    and OOM. The torchao quant scratch is already bounded by the chunked quantizer
    (see ``_to_nvfp4_chunked``); this drops the bf16 source the moment it is no
    longer needed and reclaims it before the next swap, so the resident bf16 isn't
    carried alongside the FP4 copies — mirroring the LoRA base streaming. Quant
    stays on the GPU (CPU quant is NOT bit-identical: torchao's e2m1 rounding
    diverges between CPU and CUDA).
    """
    new_module = build(source)
    _set_submodule(model, name, new_module)
    # Free the now-replaced bf16 source. Sync first: the quant kernels may still
    # read it async, and expandable_segments would otherwise hand the freed block
    # to the next allocation (use-after-free / illegal access).
    if isinstance(source, nn.Module) and source is not new_module:
        if getattr(source, "weight", None) is not None:
            source.weight = None
        if getattr(source, "bias", None) is not None:
            source.bias = None
    _reclaim_gpu()
    return new_module


def swap_frozen_embedding_to_nvfp4(
    model: nn.Module, name: str
) -> NVFP4Embedding | None:
    """Swap a FROZEN ``nn.Embedding`` for :class:`NVFP4Embedding` in place.

    Skips (returns None) a trainable embedding — an FP4-stored weight can't carry
    gradients — or a hidden dim not divisible by 16. Returns the new module on
    success so the tied path can route the lm_head through the same store.
    """
    try:
        module = model.get_submodule(name)
    except AttributeError:
        return None
    if not isinstance(module, nn.Embedding):
        return None
    if module.weight is not None and module.weight.requires_grad:
        LOG.warning(
            "nvfp4_training.quantize_embeddings: %s is trainable; skipping (an "
            "FP4-stored embedding has no high-precision master for gradients).",
            name,
        )
        return None
    if not _embedding_swappable(module):
        LOG.warning(
            "nvfp4_training.quantize_embeddings: %s hidden dim %d not divisible "
            "by %d; keeping it in high precision.",
            name,
            module.embedding_dim,
            _BLOCK_SIZE,
        )
        return None
    new_module = _stream_quantize_swap(
        model, name, module, lambda src: NVFP4Embedding.from_embedding(src)
    )
    LOG.info("NVFP4 training: swapped frozen embedding %s", name)
    return new_module


def swap_tied_embedding_and_lm_head_to_nvfp4(
    model: nn.Module,
    embed_name: str,
    lm_head_name: str,
    recipe: NVFP4Recipe | None = None,
) -> bool:
    """Quantize a tied (shared) FROZEN weight ONCE and route both roles to it.

    The shared weight becomes a single :class:`NVFP4Embedding` store; the input
    embedding reads it for the lookup and the lm_head reads the SAME store for
    its GEMM (:class:`NVFP4TiedLMHead`), so the dequantized weight is identical
    for both. No-op (False) if the shared weight is trainable or not eligible —
    the caller must keep RAISING on a trainable tied weight.
    """
    recipe = recipe or NVFP4Recipe()
    try:
        embed = model.get_submodule(embed_name)
        lm_head = model.get_submodule(lm_head_name)
    except AttributeError:
        return False
    if not isinstance(embed, nn.Embedding) or not isinstance(lm_head, nn.Linear):
        return False
    if not _embedding_swappable(embed) or not _is_swappable(lm_head):
        LOG.warning(
            "nvfp4_training: tied embedding/lm_head dims not NVFP4-eligible "
            "(hidden %%16 and lm_head %%32); keeping both in high precision."
        )
        return False
    # Capture the lm_head bias before streaming frees the shared weight, then
    # quantize the shared weight ONCE via the embedding and route the lm_head GEMM
    # at the SAME store. The shared weight is held by BOTH modules, so drop the
    # lm_head's reference too — otherwise streaming the embedding can't reclaim it.
    lm_head_bias = lm_head.bias
    lm_head.weight = None
    new_embed = _stream_quantize_swap(
        model, embed_name, embed, lambda src: NVFP4Embedding.from_embedding(src)
    )
    _set_submodule(
        model, lm_head_name, NVFP4TiedLMHead(new_embed, lm_head_bias, recipe)
    )
    LOG.info(
        "NVFP4 training: tied embedding/lm_head quantized once (shared FP4 store)"
    )
    return True


# Vision-tower submodule names / class-name fragments to locate the encoder.
_VISION_ATTR_NAMES = ("visual", "vision_tower", "vision_model")
_VISION_CLASS_FRAGS = ("Vision",)
# Linears under the vision tower that are NOT encoder GEMMs: the merger /
# patch-embed projection feed the language model, not the attention/MLP stack.
_VISION_SKIP_FRAGS = ("merger", "patch_embed", "deepstack")


def _find_vision_tower(model: nn.Module):
    """Return (name, module) of the vision encoder, or (None, None).

    Prefers the conventional attribute names (``visual``/``vision_tower``/
    ``vision_model``, possibly one level under ``model``); falls back to the
    first submodule whose class name contains "Vision".
    """
    for prefix in ("", "model."):
        for attr in _VISION_ATTR_NAMES:
            name = f"{prefix}{attr}"
            try:
                mod = model.get_submodule(name)
            except AttributeError:
                continue
            if isinstance(mod, nn.Module):
                return name, mod
    for name, mod in model.named_modules():
        if not name:
            continue
        cls = type(mod).__name__
        if any(frag in cls for frag in _VISION_CLASS_FRAGS) and any(
            isinstance(c, nn.Linear) for c in mod.modules()
        ):
            return name, mod
    return None, None


def convert_vision_tower_to_nvfp4(
    model: nn.Module,
    recipe: NVFP4Recipe | None = None,
    *,
    base_mode: str = "compute",
) -> int:
    """Swap eligible FROZEN ``nn.Linear`` layers under the vision tower to NVFP4.

    Scopes strictly to linears under the located vision encoder (attn qkv/proj,
    mlp fc1/fc2); the merger / patch-embed projection and any %32-ineligible or
    trainable linear are skipped. Warns and returns 0 if no vision tower is found
    (text-only model). Returns the count swapped.
    """
    recipe = recipe or NVFP4Recipe()
    vt_name, vt = _find_vision_tower(model)
    if vt is None:
        LOG.warning(
            "nvfp4_training.quantize_vision_tower: no vision tower found "
            "(visual/vision_tower/vision_model or a *Vision* module); skipping."
        )
        return 0

    swapped = 0
    for name, module in list(vt.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(frag in name for frag in _VISION_SKIP_FRAGS):
            continue
        if module.weight is not None and module.weight.requires_grad:
            continue
        if not _is_swappable(module):
            LOG.warning(
                "nvfp4_training.quantize_vision_tower: skipping %s.%s "
                "(in=%d out=%d not both divisible by %d)",
                vt_name,
                name,
                module.in_features,
                module.out_features,
                _GEMM_ALIGN,
            )
            continue
        fast = _mslk_available()
        if base_mode == "compute":
            cls = NVFP4FastComputeBaseLinear if fast else NVFP4ComputeBaseLinear
            build = lambda src, cls=cls: cls.from_linear(src, recipe)  # noqa: E731
        elif base_mode == "storage":
            if fast:
                build = lambda src: NVFP4FastFrozenBaseLinear.from_linear(  # noqa: E731
                    src, recipe
                )
            else:
                build = lambda src: NVFP4FrozenBaseLinear.from_linear(  # noqa: E731
                    src, recipe, fsdp=False
                )
        else:
            module.weight.requires_grad_(False)
            _set_submodule(vt, name, NVFP4Linear.from_linear(module, recipe))
            swapped += 1
            continue
        _stream_quantize_swap(vt, name, module, build)
        swapped += 1
    LOG.info(
        "nvfp4_training.quantize_vision_tower: swapped %d linears under %s "
        "(mode=%s)",
        swapped,
        vt_name,
        base_mode,
    )
    return swapped


def te_nvfp4_available() -> tuple[bool, str]:
    """Return (ok, reason) for the Transformer Engine NVFP4 backend."""
    try:
        import transformer_engine.pytorch  # noqa: F401
        from transformer_engine.common.recipe import NVFP4BlockScaling  # noqa: F401
    except Exception as exc:  # ImportError or the cuBLAS-symbol OSError
        return False, (
            f"Transformer Engine NVFP4 backend unavailable ({type(exc).__name__}: "
            f"{exc}). Install axolotl[transformer-engine] (source build; on sm_120 "
            "use NVTE_CUDA_ARCHS=120 and preload the system cuBLAS)."
        )
    return True, ""


def te_nvfp4_recipe(recipe: "NVFP4Recipe"):
    """Build a TE NVFP4BlockScaling recipe. On consumer Blackwell (sm_120) the
    RHT/SR/2D fusion kernels do not run, so disable them and warn — TE there is
    a recipe-less FP4 GEMM. On sm_100 (B200) the full recipe runs."""
    from transformer_engine.common.recipe import NVFP4BlockScaling

    cap = torch.cuda.get_device_capability()
    if cap == (12, 0):
        LOG.warning(
            "TE NVFP4 on sm_120 (consumer Blackwell): RHT/stochastic-rounding/2D "
            "kernels do not run here, disabling them — convergence recipe is OFF "
            "(unproven at scale). Use backend=native for the full recipe on sm_120."
        )
        return NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
            disable_2d_quantization=True,
        )
    return NVFP4BlockScaling(
        disable_rht=not recipe.hadamard,
        disable_stochastic_rounding=not recipe.stochastic_rounding,
    )


def convert_to_te_nvfp4_training(
    model: nn.Module,
    recipe: NVFP4Recipe | None = None,
    *,
    exclude: tuple[str, ...] = ("lm_head", "embed_tokens"),
) -> int:
    """Swap eligible ``nn.Linear`` for ``transformer_engine.pytorch.Linear`` so
    they run NVFP4 GEMMs under TE's ``NVFP4BlockScaling`` recipe (FFT only).

    The caller must wrap the training step in ``te.fp8_autocast(fp8_recipe=
    te_nvfp4_recipe(recipe))``. Weights are copied into the TE linear; dims must
    be divisible by 16.
    """
    import transformer_engine.pytorch as te

    recipe = recipe or NVFP4Recipe()
    swapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or any(f in name for f in exclude):
            continue
        if not _is_swappable(module):
            LOG.warning("NVFP4(te): skipping %s (dims not divisible by 16)", name)
            continue
        te_lin = te.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            params_dtype=module.weight.dtype,
        )
        with torch.no_grad():
            te_lin.weight.copy_(module.weight)
            if module.bias is not None:
                te_lin.bias.copy_(module.bias)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], te_lin)
        swapped += 1
    LOG.info("NVFP4 training (te backend): swapped %d linear layers", swapped)
    return swapped


def convert_lora_base_to_te_nvfp4(
    model: nn.Module,
    recipe: NVFP4Recipe | None = None,
    *,
    exclude: tuple[str, ...] = ("lm_head", "embed_tokens"),
) -> int:
    """Swap the FROZEN base_layer inside each PEFT ``lora.Linear`` for a
    ``transformer_engine.pytorch.Linear`` so the base GEMM runs NVFP4 under TE's
    ``fp8_autocast`` (the trainer wraps the step via ``te_nvfp4_recipe``).

    The trainable LoRA adapters stay high-precision ``nn.Linear`` and are left
    untouched; only the frozen base runs FP4, so TE computes its dgrad (input
    gradient) but no wgrad — which also sidesteps TE's wgrad token-dim %32
    constraint. Weights are copied in; dims must be divisible by 16.
    """
    import transformer_engine.pytorch as te

    from peft.tuners.lora import Linear as LoraLinear

    recipe = recipe or NVFP4Recipe()
    swapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoraLinear):
            continue
        if any(frag in name for frag in exclude):
            continue
        base = module.base_layer
        if not isinstance(base, nn.Linear) or not _is_swappable(base):
            continue
        te_lin = te.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            params_dtype=base.weight.dtype,
        )
        with torch.no_grad():
            te_lin.weight.copy_(base.weight)
            if base.bias is not None:
                te_lin.bias.copy_(base.bias)
        for p in te_lin.parameters():
            p.requires_grad_(False)
        module.base_layer = te_lin
        swapped += 1
    LOG.info("NVFP4 training (te backend): swapped %d LoRA base layers", swapped)
    return swapped


def convert_lora_base_to_nvfp4(
    model: nn.Module,
    recipe: NVFP4Recipe | None = None,
    *,
    quantized_storage: bool = False,
    compute_base: bool = False,
    fsdp: bool = False,
    exclude: tuple[str, ...] = ("lm_head", "embed_tokens"),
) -> int:
    """Swap the FROZEN base_layer inside each PEFT ``lora.Linear`` for an NVFP4
    linear, leaving the trainable adapters in high precision.

    Three base modes (mutually exclusive, checked in priority order):

    - ``compute_base=True`` (LoRA + FP4 compute, recommended): base_layer ->
      NVFP4ComputeBaseLinear. The frozen base is pre-quantized ONCE into two
      NVFP4 layouts; fprop+dgrad run as pure FP4 GEMMs with no per-step base
      quant prologue. ~1.75x weight memory and the fastest base compute.
    - ``quantized_storage=True`` (NVFP4-QLoRA): base_layer ->
      NVFP4FrozenBaseLinear, base stored packed in FP4 (~3.5x weight memory);
      backward dequantizes to bf16. Max memory, modest speed. When MSLK is
      available (and not FSDP) the MSLK-fused NVFP4FastFrozenBaseLinear is used
      instead — same single-layout storage, fast per-step quant.
    - neither (default): base_layer -> NVFP4Linear, base kept high-precision and
      re-quantized each step. FP4 base GEMM, no memory win.

    Returns the number of base layers swapped. Requires PEFT-wrapped adapters.
    """
    from peft.tuners.lora import Linear as LoraLinear

    recipe = recipe or NVFP4Recipe()
    swapped = 0
    skipped_offloaded = 0
    # Stream the swap. For the NVFP4 adapter path the loader keeps the base on
    # CPU (so the full bf16 model never sits on the GPU and device_map can't
    # strand weights on meta). Move each base weight to the GPU just-in-time,
    # quantize to FP4, and free the bf16 — the GPU only ever holds the FP4 base
    # plus one transient layer. Weights already on the GPU are quantized in place.
    # A materialized named_modules() list would pin every base_layer and defeat
    # the per-layer free, so keep only the lora.Linear references.
    target = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    lora_modules = [
        (n, m) for n, m in model.named_modules() if isinstance(m, LoraLinear)
    ]
    streamed = False
    for name, module in lora_modules:
        if any(frag in name for frag in exclude):
            continue
        base = module.base_layer
        if not isinstance(base, nn.Linear) or not _is_swappable(base):
            continue
        # A meta weight has no data to quantize (device_map offloaded it because
        # even the CPU-streamed load couldn't place it). Can't support it.
        if base.weight is None or base.weight.is_meta:
            skipped_offloaded += 1
            continue
        if base.weight.device != target:
            base = base.to(target)  # stream this layer to the GPU
            streamed = True
        if compute_base:
            fast = _mslk_available()
            if fast and (recipe.stochastic_rounding or recipe.hadamard):
                LOG.warning_once(
                    "NVFP4 MSLK-fast compute base uses round-to-nearest; the "
                    "stochastic_rounding/hadamard convergence recipe is NOT applied "
                    "(harmless on sm_120 where the recipe is off anyway, but on "
                    "sm_100 this drops the recipe — use base_mode without mslk for it)."
                )
            cls = NVFP4FastComputeBaseLinear if fast else NVFP4ComputeBaseLinear
            module.base_layer = cls.from_linear(base, recipe)
        elif quantized_storage:
            # MSLK-fused storage has no FSDP all-gather hooks yet, so keep the
            # torchao NVFP4FrozenBaseLinear (which carries them) under FSDP.
            if _mslk_available() and not fsdp:
                module.base_layer = NVFP4FastFrozenBaseLinear.from_linear(base, recipe)
            else:
                module.base_layer = NVFP4FrozenBaseLinear.from_linear(
                    base, recipe, fsdp=fsdp
                )
        else:
            base.weight.requires_grad_(False)
            module.base_layer = NVFP4Linear.from_linear(base, recipe)
        swapped += 1
        # Free the now-replaced bf16 base immediately. Otherwise the full bf16
        # model stays resident while the FP4 copies accumulate on top (the swap
        # transiently doubles memory) — which OOMs large models on load even
        # though the FP4 base itself is far smaller. hp mode keeps its weight.
        if not (compute_base or quantized_storage):
            continue
        # Drop the bf16 weight now so peak stays near the FP4 footprint, not
        # bf16's. Sync first: the quant kernels still read this weight async, and
        # expandable_segments would hand the freed block to the next layer's
        # quant — a use-after-free that corrupts the kernel (illegal access).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        base.weight = None
        base.bias = None
        del base
    # The FP4 base is in place; move the rest (embeddings, norms, lm_head, the
    # LoRA adapters) onto the GPU now that the heavy weights are quantized.
    if streamed:
        model.to(target)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if compute_base:
        mode = "compute (mslk-fast)" if _mslk_available() else "compute"
    elif quantized_storage:
        mode = "storage (mslk-fast)" if (_mslk_available() and not fsdp) else "storage"
    else:
        mode = "hp"
    LOG.info("NVFP4 training: swapped %d LoRA base layers (mode=%s)", swapped, mode)
    if skipped_offloaded:
        raise RuntimeError(
            f"nvfp4_training: {skipped_offloaded} base weight(s) are on meta with no "
            "data to quantize — the model didn't fully materialize even via the "
            "CPU-streamed load. Use a smaller base, add VRAM/GPUs, or adapter: qlora."
        )
    return swapped
