"""Native-NVFP4 attention for Qwen3-VL dense full-attention (LM) layers.

Qwen3-VL uses STANDARD (non-gated) softmax attention with multimodal mRoPE,
unlike Qwen3.5's gated DeltaNet-hybrid attention — so it needs its own forward
rather than the Qwen3.5 patch (which assumes q_proj -> head_dim*2 + sigmoid gate
and Qwen3.5's plain RoPE). This patches ``Qwen3VLTextAttention`` to run native
NVFP4 attention on the dense causal/full, cache-free PREFILL path in both grad
(training) and no-grad (eval / first generation token) modes — fed VL's
already-roped Q/K via the model's own ``apply_rotary_pos_emb`` (rope-agnostic op).
It falls back to the model's configured attention (flash_attention_2) for KV-cache
decode, non-dense / per-key-padded masks, and ``output_attentions``. head_dim 128
is supported by the kernel (d in {128, 256}). Vision attention
(Qwen3VLVisionBlock) is left untouched.

Opt-in via ``nvfp4_training.attention`` on a qwen3_vl model; default OFF.
"""

from __future__ import annotations

import torch
from torch import nn

from axolotl.kernels.attn_nvfp4_flash import nvfp4_flash_attn_func
from axolotl.monkeypatch.attention.nvfp4_flash_attn import (
    _custom_op_enabled,
    _mask_is_dense_causal_or_full,
    _packed_position_ids_info,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def make_nvfp4_vl_forward(orig_forward):
    """Build a patched ``Qwen3VLTextAttention.forward`` using NVFP4 attention.

    Runs NVFP4 on the dense causal/full, cache-free PREFILL path in BOTH grad
    (training) and no-grad (eval / first generation token) modes — mirroring the
    Qwen3.5 patch, so the two passes are numerically consistent (matters under
    gradient checkpointing). On the no-grad path it writes the roped K/V to the
    cache so subsequent decode steps see prefill context. Falls back to
    ``orig_forward`` (flash_attention_2) for: KV-cache decode (prior cached
    context — a dedicated FP4 decode kernel is a proven loser), per-key-padded /
    non-dense masks, ``output_attentions``, and grad mode without a trained
    backward. The Qwen3.5-only inference fast paths (fuse_vproj / fp4_projections)
    are intentionally omitted here.
    """

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if kwargs.get("output_attentions"):
            return orig_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )

        grad_enabled = torch.is_grad_enabled()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_len = input_shape[-1]

        # NVFP4 fast path is PREFILL only (no prior cached context) on a dense
        # causal/full mask. Cached decode reuses prior roped K/V and FP4 decode is
        # a proven loser, so it goes to stock attention.
        has_cache_context = (
            past_key_values is not None
            and past_key_values.get_seq_length(self.layer_idx) > 0
        )
        kind = None
        if not has_cache_context:
            kind = _mask_is_dense_causal_or_full(attention_mask, q_len, q_len)
        if kind == "causal" and attention_mask is None:
            # Sample-packed (multipack) batch: boundaries live in position_ids.
            # TODO: wire the varlen (cu_seqlens) NVFP4 path like the Qwen3.5
            # patch once VL mRoPE packed position_ids semantics (vision tokens
            # share/repeat temporal positions) are validated against the
            # position==0 reset detection. Until then NEVER dense-attend a
            # packed batch — fall back to the original (FA2 varlen) forward.
            pkind, _ = _packed_position_ids_info(kwargs.get("position_ids"), q_len)
            if pkind is not None:
                kind = None
        if kind is None:
            return orig_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )
        # Grad mode needs the trained backward; a cache during grad is unexpected.
        if grad_enabled and (
            not getattr(self, "_nvfp4_train_backward", False)
            or past_key_values is not None
        ):
            return orig_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
        )

        # Standard Qwen3-VL projections (no gate): q/k_norm on head_dim, then mRoPE.
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_roped, key_roped = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # No-grad prefill: stash roped K/V so later decode steps see this context.
        if not grad_enabled and past_key_values is not None:
            past_key_values.update(key_roped, value_states, self.layer_idx)

        ng = query_states.shape[1] // key_states.shape[1]
        sr = getattr(self, "_nvfp4_stochastic_rounding", True)
        grad_sr = sr and not getattr(self, "_nvfp4_backward_rtn_grad_packs", False)
        save_packs = getattr(self, "_nvfp4_save_backward_packs", False)
        dkdv_bf16 = getattr(self, "_nvfp4_dkdv_scratch_bf16", False)
        hp_grad_dots = getattr(self, "_nvfp4_bf16_grad_dots", None)

        # Same NVFP4 op for grad and no-grad (consistent under checkpointing). The
        # backward SR knobs are inert in no-grad. Under compile the opaque custom op
        # keeps Inductor from tracing tl.dot_scaled (decompose crash); the graph
        # breaks give it the eager boundary.
        if _custom_op_enabled(self):
            from axolotl.kernels.attn_nvfp4_custom_op import (
                nvfp4_flash_attn_train_custom_op,
            )

            torch._dynamo.graph_break()
            attn_output = nvfp4_flash_attn_train_custom_op(
                query_roped,
                key_roped,
                value_states,
                self.scaling,
                causal=(kind == "causal"),
                num_key_value_groups=ng,
                stochastic_rounding=sr,
                backward_p_dv_stochastic_rounding=grad_sr,
                backward_dot_dv_stochastic_rounding=grad_sr,
                backward_ds_dq_stochastic_rounding=grad_sr,
                dkdv_scratch_bf16=dkdv_bf16,
                backward_bf16_grad_dots=hp_grad_dots,
                save_backward_packs=save_packs,
                out_layout="zshd",
            )  # [Z, S, H, D]
            torch._dynamo.graph_break()
        else:
            attn_output = nvfp4_flash_attn_func(
                query_roped,
                key_roped,
                value_states,
                self.scaling,
                causal=(kind == "causal"),
                num_key_value_groups=ng,
                stochastic_rounding=sr,
                backward_p_dv_stochastic_rounding=grad_sr,
                backward_dot_dv_stochastic_rounding=grad_sr,
                backward_ds_dq_stochastic_rounding=grad_sr,
                save_backward_packs=save_packs,
                dkdv_scratch_bf16=dkdv_bf16,
                backward_bf16_grad_dots=hp_grad_dots,
            ).transpose(1, 2)  # [Z, H, S, D] -> [Z, S, H, D]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None

    return forward


def _set_attrs(module, *, train_backward, backward_rtn_grad_packs, save_backward_packs,
               dkdv_scratch_bf16, bf16_grad_dots, compile_custom_op,
               stochastic_rounding):
    module._nvfp4_train_backward = train_backward
    module._nvfp4_backward_rtn_grad_packs = backward_rtn_grad_packs
    module._nvfp4_save_backward_packs = save_backward_packs
    module._nvfp4_dkdv_scratch_bf16 = dkdv_scratch_bf16
    module._nvfp4_bf16_grad_dots = bf16_grad_dots
    module._nvfp4_compile_custom_op = compile_custom_op
    module._nvfp4_stochastic_rounding = stochastic_rounding


def patch_qwen3_vl_nvfp4_attention(
    model: nn.Module,
    *,
    train_backward: bool = False,
    backward_rtn_grad_packs: bool = False,
    save_backward_packs: bool = False,
    dkdv_scratch_bf16: bool = False,
    bf16_grad_dots: bool | None = None,
    compile_custom_op: bool = False,
    stochastic_rounding: bool = True,
) -> int:
    """Patch every Qwen3-VL LM full-attention layer's forward to use NVFP4.

    Leaves vision-encoder attention untouched. Idempotent. Returns patched count.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    patched = 0
    seen_forward = None
    for module in model.modules():
        if isinstance(module, Qwen3VLTextAttention):
            kw = dict(
                train_backward=train_backward,
                backward_rtn_grad_packs=backward_rtn_grad_packs,
                save_backward_packs=save_backward_packs,
                dkdv_scratch_bf16=dkdv_scratch_bf16,
                bf16_grad_dots=bf16_grad_dots,
                compile_custom_op=compile_custom_op,
                stochastic_rounding=stochastic_rounding,
            )
            if getattr(module, "_nvfp4_patched", False):
                _set_attrs(module, **kw)
                continue
            orig = type(module).forward
            if seen_forward is None:
                seen_forward = make_nvfp4_vl_forward(orig)
            module.forward = seen_forward.__get__(module, type(module))
            module._nvfp4_patched = True
            _set_attrs(module, **kw)
            patched += 1
    LOG.info(
        "nvfp4 attention: patched %d Qwen3-VL full-attention layers "
        "(train_backward=%s, compile_custom_op=%s)",
        patched,
        train_backward,
        compile_custom_op,
    )
    return patched
