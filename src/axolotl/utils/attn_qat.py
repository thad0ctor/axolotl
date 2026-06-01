"""FP4-attention quantization-aware training (Attn-QAT, arXiv 2603.00040).

QAT for FP4 attention: during SFT the attention operands (Q, K, V and the
softmax probabilities P) are FAKE-quantized to NVFP4 — quantized then immediately
dequantized, with all matmuls left in bf16 — so the model learns to tolerate real
FP4 attention at inference. This is an ACCURACY feature, not a speed one: a training
step stays ~bf16 cost/precision; the payoff is FP4-attention robustness for serving.

The fake-quant uses a straight-through estimator (STE): forward returns the
dequantized value, backward passes the gradient through unchanged. v1 is an EAGER
attention (P is materialized), which is correct for QAT — the STE makes the
backward exact without the paper's auxiliary high-precision O' reconstruction.

Follow-up: a fused Triton kernel would recompute P inside a FlashAttention-style
backward and would then need the paper's O' identity (eq. 9: P^T dP = dO^T O') to
avoid storing P. Eager has P in hand, so v1 does not need it.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

LOG = logging.getLogger(__name__)

ATTN_QAT_IMPL_NAME = "nvfp4_qat"

_BLOCK_SIZE = 16  # NVFP4 block, taken along the last (head_dim) axis


def _nvfp4_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 fake-quant φ over the last axis in blocks of 16: per-block scale
    ``amax/6`` (e4m3), round to the E2M1 grid, dequantize back to ``x``'s dtype.

    Reuses torchao's ``NVFP4Tensor`` round-trip, which implements exactly this
    (amax/6 e4m3 block scale + round-to-nearest-even on the E2M1 grid). The
    softmax probabilities P are quantized along the key axis whose length is the
    sequence length and need not be a multiple of 16, so zero-pad the trailing
    block and slice it back — the zeros sit in their own partial block and do not
    perturb the real blocks' scales.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    orig_shape = x.shape
    orig_dtype = x.dtype
    n = orig_shape[-1]
    pad = (-n) % _BLOCK_SIZE
    x2d = x.reshape(-1, n)
    if pad:
        x2d = torch.nn.functional.pad(x2d, (0, pad))
    q = NVFP4Tensor.to_nvfp4(x2d.contiguous(), block_size=_BLOCK_SIZE)
    out = q.dequantize(orig_dtype)
    if pad:
        out = out[:, :n]
    return out.reshape(orig_shape)


def fake_quant_nvfp4(x: torch.Tensor) -> torch.Tensor:
    """Straight-through NVFP4 fake-quant: forward = quant→dequant, backward = id.

    ``x_fq = x + (φ(x) - x).detach()`` so the value seen downstream is the FP4
    round-trip while the gradient flows back as if no quantization happened.
    """
    return x + (_nvfp4_quant_dequant(x) - x).detach()


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA broadcast: [B, n_kv, S, D] -> [B, n_kv*n_rep, S, D] (mirrors HF)."""
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


def nvfp4_qat_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention with NVFP4 fake-quant on Q, K, V and the softmax weights P.

    Structurally identical to HF's ``eager_attention_forward`` (same GQA repeat,
    scaling, additive mask, fp32 softmax, dropout) with the four fake-quant calls
    from Algorithm 2 inserted. Registered in ``ALL_ATTENTION_FUNCTIONS`` under
    :data:`ATTN_QAT_IMPL_NAME` and selected via ``config._attn_implementation``.
    """
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    q_fq = fake_quant_nvfp4(query)
    k_fq = fake_quant_nvfp4(key_states)
    v_fq = fake_quant_nvfp4(value_states)

    attn_weights = torch.matmul(q_fq, k_fq.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., : k_fq.shape[-2]]

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    p_fq = fake_quant_nvfp4(attn_weights)

    attn_output = torch.matmul(p_fq, v_fq)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def register_nvfp4_qat_attention() -> None:
    """Register the QAT attention in the transformers attention-function registry.

    Also registers the matching mask builder: ``create_causal_mask`` skips mask
    construction (returns None -> bidirectional attention, future-token leakage)
    for any impl absent from ``ALL_MASK_ATTENTION_FUNCTIONS``, so map this key to
    ``eager_mask``, which emits the additive 4D causal+padding mask this eager
    attention consumes.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(ATTN_QAT_IMPL_NAME, nvfp4_qat_attention_forward)
    if ATTN_QAT_IMPL_NAME not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_QAT_IMPL_NAME, eager_mask)


def apply_nvfp4_qat_attention(model: nn.Module) -> None:
    """Register + select NVFP4 fake-quant attention on ``model`` (opt-in).

    Registers the function, then routes the model's attention dispatch through it
    via ``set_attn_implementation`` (validates the key and propagates to every
    submodel). The model must follow the HF AttentionInterface dispatch pattern.
    """
    register_nvfp4_qat_attention()
    set_impl = getattr(model, "set_attn_implementation", None)
    if set_impl is None:
        raise RuntimeError(
            "fp4_attention_qat: model has no set_attn_implementation; this "
            "transformers version predates the AttentionInterface registry."
        )
    set_impl(ATTN_QAT_IMPL_NAME)
    LOG.info(
        "fp4_attention_qat: NVFP4 fake-quant attention active (impl=%s)",
        ATTN_QAT_IMPL_NAME,
    )
