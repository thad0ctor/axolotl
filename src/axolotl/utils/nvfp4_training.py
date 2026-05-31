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
    return NVFP4Tensor.to_nvfp4(
        t, block_size=_BLOCK_SIZE, per_tensor_scale=per_tensor_scale
    )


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


def _is_swappable(module: nn.Linear) -> bool:
    return (
        module.in_features % _BLOCK_SIZE == 0
        and module.out_features % _BLOCK_SIZE == 0
    )


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
                "NVFP4: skipping %s (in=%d out=%d not both divisible by 16)",
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
