"""End-to-end native-NVFP4 attention for Qwen3.5 full-attention layers.

Replaces the full softmax-attention forward with a fused-producer NVFP4 pipeline:

  q_proj/k_proj/v_proj (+ q_norm/k_norm, unchanged bf16)
    -> fused RoPE -> NVFP4 pack of Q,K   (one pass, no standalone quant round-trip)
    -> v_proj output -> NVFP4 pack of V along the key axis
    -> packed-input native-NVFP4 flash forward (tl.dot_scaled, e2m1 + e4m3/group-16)
    -> attn_output * sigmoid(gate) -> o_proj

The fused producers eliminate the standalone Q/K pre-quant round-trips (the trap
that made the materialized-attention pipeline lose to bf16): the pack rides along
with the RoPE pass Q/K already pay. V is still a separate key-axis quant pass (the
real v_proj GEMM epilogue would absorb it; here it is a standalone read).

Qwen3.5 is HYBRID — only ``full_attention`` layers (head_dim 256 softmax GQA) are
patched; ``linear_attention`` layers are left untouched. Inference/forward only
(no backward). Opt-in: call ``patch_qwen3_5_nvfp4_attention(model)``; default OFF.

Per-key padding / non-causal-with-mask batches fall back to the model's original
forward (correctness over speed). The fast path serves dense causal prefill and
single-step / full decode.
"""

from __future__ import annotations

import torch
from torch import nn

from axolotl.kernels.attn_nvfp4_flash import nvfp4_flash_attention_packed
from axolotl.kernels.nvfp4_fused_producers import (
    fused_rope_quant_qk,
    quant_v_keyaxis,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_BLOCK_N = 128


def _mask_is_dense_causal_or_full(
    attention_mask: torch.Tensor | None, q_len: int, kv_len: int
) -> str | None:
    """Return 'causal' / 'full' if the mask is a dense causal/full mask, else None.

    None -> there is per-key padding (or a mask shape we can't densely represent),
    so the caller must fall back to the original forward.
    """
    if attention_mask is None:
        return "causal" if q_len == kv_len else "full"
    if attention_mask.dim() != 4:
        return None
    m = attention_mask[..., :kv_len]
    if m.shape[-1] != kv_len:
        return None
    neg = torch.finfo(m.dtype).min
    keep = m > neg / 2
    if keep.shape[1] != 1:
        if not bool((keep == keep[:, :1]).all()):
            return None
        keep = keep[:, :1]
    keep = keep[:, 0]
    if bool(keep.all()):
        return "full"
    if q_len == kv_len:
        causal = torch.tril(
            torch.ones(q_len, kv_len, dtype=torch.bool, device=keep.device)
        )[None]
        if bool((keep == causal).all()):
            return "causal"
    return None


def _nvfp4_attention(
    module: nn.Module,
    query_states: torch.Tensor,   # [Z, H, S, D]  post q_norm, PRE-RoPE
    key_states: torch.Tensor,     # [Z, Hk, S, D] post k_norm, PRE-RoPE
    value_states: torch.Tensor,   # [Z, Hk, Skv, D]
    cos: torch.Tensor,            # [Z, S, rotary_dim]
    sin: torch.Tensor,
    scaling: float,
    causal: bool,
) -> torch.Tensor:
    """Fused-producer NVFP4 attention. Returns ``[Z, S, H, D]`` (HF attn_output)."""
    z, h, s_q, d = query_states.shape
    hk = key_states.shape[1]
    s_kv = key_states.shape[2]

    qnv, qsc = fused_rope_quant_qk(query_states, cos, sin)
    knv, ksc = fused_rope_quant_qk(key_states, cos, sin)
    vnv, vsc, _ = quant_v_keyaxis(value_states, block_n=_BLOCK_N)

    out = nvfp4_flash_attention_packed(
        qnv, qsc, knv, ksc, vnv, vsc,
        z=z, h=h, hk=hk, s_q=s_q, s_kv=s_kv, d=d,
        scaling=scaling,
        out_dtype=query_states.dtype,
        causal=causal,
        block_n=_BLOCK_N,
    )  # [Z, H, S, D]
    return out.transpose(1, 2)  # [Z, S, H, D]


def make_nvfp4_forward(orig_forward):
    """Build a patched ``Qwen3_5Attention.forward`` that uses NVFP4 attention.

    Mirrors the stock forward (chunk q/gate, q_norm/k_norm, RoPE, attention, gate,
    o_proj) but routes the attention through the fused-producer NVFP4 path. Falls
    back to ``orig_forward`` for: grad enabled, KV-cache use (decode with cache),
    per-key padding, or output_attentions.
    """
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if torch.is_grad_enabled() or kwargs.get("output_attentions"):
            return orig_forward(
                self, hidden_states, position_embeddings, attention_mask,
                past_key_values, **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2, dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q_len = query_states.shape[2]

        # NVFP4 fast path: prefill only (no prior cached context) on a dense
        # causal/full mask. Cached decode reuses prior roped K/V from the cache,
        # which the fused-RoPE producer can't reconstruct, so it goes to stock.
        has_cache_context = (
            past_key_values is not None
            and past_key_values.get_seq_length(self.layer_idx) > 0
        )
        kind = None
        if not has_cache_context:
            kind = _mask_is_dense_causal_or_full(attention_mask, q_len, q_len)

        if kind is not None:
            # write the (roped) K/V to the cache so subsequent decode steps see
            # prefill context, then run NVFP4 on the same roped K/V.
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                apply_rotary_pos_emb,
            )

            if past_key_values is not None:
                k_roped, _ = apply_rotary_pos_emb(
                    key_states, key_states, cos, sin
                )
                past_key_values.update(k_roped, value_states, self.layer_idx)

            attn_output = _nvfp4_attention(
                self, query_states, key_states, value_states, cos, sin,
                self.scaling, causal=(kind == "causal"),
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            return self.o_proj(attn_output), None

        return orig_forward(
            self, hidden_states, position_embeddings, attention_mask,
            past_key_values, **kwargs,
        )

    return forward


def patch_qwen3_5_nvfp4_attention(model: nn.Module) -> int:
    """Patch every Qwen3.5 FULL-attention layer's forward to use NVFP4 attention.

    Leaves linear-attention layers untouched. Returns the count of patched layers.
    Idempotent: re-patching a model is a no-op on already-patched modules.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    patched = 0
    seen_forward = None
    for module in model.modules():
        if isinstance(module, Qwen3_5Attention):
            if getattr(module, "_nvfp4_patched", False):
                continue
            orig = type(module).forward
            if seen_forward is None:
                seen_forward = make_nvfp4_forward(orig)
            module.forward = seen_forward.__get__(module, type(module))
            module._nvfp4_patched = True
            patched += 1
    LOG.info("nvfp4 attention: patched %d Qwen3.5 full-attention layers", patched)
    return patched
