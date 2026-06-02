"""End-to-end native-NVFP4 attention for Qwen3.5 full-attention layers.

Replaces the full softmax-attention forward with a fused-producer NVFP4 pipeline:

  q_proj/k_proj/v_proj (+ q_norm/k_norm, unchanged bf16)
    -> fused RoPE -> NVFP4 pack of Q,K   (one pass, no standalone quant round-trip)
    -> v_proj output -> NVFP4 pack of V along the key axis
    -> packed-input native-NVFP4 flash forward (tl.dot_scaled, e2m1 + e4m3/group-16)
    -> attn_output * sigmoid(gate) -> o_proj

The fused producers eliminate the standalone Q/K pre-quant round-trips (the trap
that made the materialized-attention pipeline lose to bf16): the pack rides along
with the RoPE pass Q/K already pay. V can optionally be produced by a native-NVFP4
v_proj GEMM with a key-axis pack epilogue.

Qwen3.5 is HYBRID — only ``full_attention`` layers (head_dim 256 softmax GQA) are
patched; ``linear_attention`` layers are left untouched. The fast fused-producer
path is inference/forward-only; an explicitly enabled training path uses
``nvfp4_flash_attn_func`` with native NVFP4 backward. Opt-in: call
``patch_qwen3_5_nvfp4_attention(model)``; default OFF.

Per-key padding / non-causal-with-mask batches fall back to the model's original
forward (correctness over speed). The fast path serves dense causal prefill and
single-step / full decode.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from axolotl.kernels.attn_nvfp4_flash import (
    nvfp4_flash_attention_packed,
    nvfp4_flash_attn_func,
)
from axolotl.kernels.nvfp4_fused_producers import (
    fused_rope_quant_qk,
    fused_vproj_quant_v_keyaxis,
    prepack_vproj_weight,
    quant_v_keyaxis,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_BLOCK_N = 128

# Fuse the V NVFP4-quant into a native-NVFP4 v_proj GEMM epilogue (no bf16 V, no
# transpose copy, no standalone quant read). Makes v_proj itself FP4, so it trades a
# little parity (attn-op cos ~0.979 -> ~0.967) for ~1.4-1.5x on the V producer and
# +0.1-0.2x end-to-end (v_proj counted). Real-model logit cos stays >=0.998. OFF by
# default; flip via patch_qwen3_5_nvfp4_attention(model, fuse_vproj=True).
_FUSE_VPROJ = False
_LAYER_AUTOGRAD_ENV = "AXOLOTL_NVFP4_QWEN35_LAYER_AUTOGRAD"
_CUSTOM_OP_ENV = "AXOLOTL_NVFP4_QWEN35_ATTENTION_CUSTOM_OP"


def _layer_autograd_enabled(module: nn.Module) -> bool:
    if getattr(module, "_nvfp4_layer_autograd", False):
        return True
    return os.environ.get(_LAYER_AUTOGRAD_ENV, "").lower() in {"1", "true", "yes"}


def _custom_op_enabled(module: nn.Module) -> bool:
    if getattr(module, "_nvfp4_compile_custom_op", False):
        return True
    return os.environ.get(_CUSTOM_OP_ENV, "").lower() in {"1", "true", "yes"}


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


def _get_vproj_packed(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazily prepack v_proj and refresh if an optimizer updated the weight."""
    if type(module.v_proj) is not nn.Linear:
        raise TypeError("NVFP4 fused v_proj requires a plain nn.Linear v_proj")
    weight = module.v_proj.weight
    key = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)
    cached = getattr(module, "_nvfp4_vproj_packed", None)
    if cached is None or cached[0] != key:
        wnv, wsc = prepack_vproj_weight(weight.detach())
        module._nvfp4_vproj_packed = (key, wnv, wsc)
        cached = module._nvfp4_vproj_packed
    return cached[1], cached[2]


def _can_fuse_vproj(module: nn.Module) -> bool:
    return type(module.v_proj) is nn.Linear


def _nvfp4_attention(
    module: nn.Module,
    query_states: torch.Tensor,   # [Z, H, S, D]  post q_norm, PRE-RoPE
    key_states: torch.Tensor,     # [Z, Hk, S, D] post k_norm, PRE-RoPE
    value_states: torch.Tensor,   # [Z, Hk, Skv, D]
    cos: torch.Tensor,            # [Z, S, rotary_dim]
    sin: torch.Tensor,
    scaling: float,
    causal: bool,
    hidden_states: torch.Tensor | None = None,  # [Z, S, hidden], for fused v_proj
) -> torch.Tensor:
    """Fused-producer NVFP4 attention. Returns ``[Z, S, H, D]`` (HF attn_output)."""
    z, h, s_q, d = query_states.shape
    hk = key_states.shape[1]
    s_kv = key_states.shape[2]

    qnv, qsc = fused_rope_quant_qk(query_states, cos, sin)
    knv, ksc = fused_rope_quant_qk(key_states, cos, sin)
    if hidden_states is not None:
        # fuse the V quant into a native-NVFP4 v_proj GEMM epilogue (no bf16 V)
        wnv, wsc = _get_vproj_packed(module)
        vnv, vsc, _ = fused_vproj_quant_v_keyaxis(
            hidden_states, wnv, wsc, hk=hk, d=d, block_n=_BLOCK_N
        )
    else:
        vnv, vsc, _ = quant_v_keyaxis(value_states, block_n=_BLOCK_N)

    flash_attention_packed = nvfp4_flash_attention_packed
    if _custom_op_enabled(module):
        from axolotl.kernels.attn_nvfp4_custom_op import (
            nvfp4_flash_attention_packed_custom_op,
        )

        flash_attention_packed = nvfp4_flash_attention_packed_custom_op

    out = flash_attention_packed(
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
    o_proj). In no-grad mode it routes through the fused-producer NVFP4 path. If
    ``train_backward`` was enabled at patch time, grad-enabled dense prefill uses
    the differentiable native-NVFP4 attention function.
    """
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        grad_enabled = torch.is_grad_enabled()
        if kwargs.get("output_attentions"):
            return orig_forward(
                self, hidden_states, position_embeddings, attention_mask,
                past_key_values, **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_len = input_shape[-1]

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

        if (
            grad_enabled
            and kind is not None
            and past_key_values is None
            and getattr(self, "_nvfp4_train_backward", False)
            and _layer_autograd_enabled(self)
        ):
            from axolotl.monkeypatch.attention.nvfp4_qwen35_layer_autograd import (
                qwen35_nvfp4_layer_attention,
                supports_qwen35_layer_autograd,
            )

            if supports_qwen35_layer_autograd(self):
                return (
                    qwen35_nvfp4_layer_attention(
                        self,
                        hidden_states,
                        position_embeddings,
                        causal=(kind == "causal"),
                    ),
                    None,
                )

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2, dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)

        cos, sin = position_embeddings

        if kind is not None:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                apply_rotary_pos_emb,
            )

            if grad_enabled:
                if (
                    not getattr(self, "_nvfp4_train_backward", False)
                    or past_key_values is not None
                ):
                    return orig_forward(
                        self, hidden_states, position_embeddings, attention_mask,
                        past_key_values, **kwargs,
                    )
                value_states = (
                    self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                )
                query_roped, key_roped = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
                attn_output = nvfp4_flash_attn_func(
                    query_roped,
                    key_roped,
                    value_states,
                    self.scaling,
                    causal=(kind == "causal"),
                    num_key_value_groups=query_states.shape[1] // key_states.shape[1],
                    stochastic_rounding=getattr(
                        self, "_nvfp4_stochastic_rounding", True
                    ),
                    save_backward_packs=getattr(
                        self, "_nvfp4_save_backward_packs", False
                    ),
                    backward_p_dv_stochastic_rounding=(
                        getattr(self, "_nvfp4_stochastic_rounding", True)
                        and not getattr(self, "_nvfp4_backward_dv_p_rtn", False)
                    ),
                    backward_dot_dv_stochastic_rounding=(
                        getattr(self, "_nvfp4_stochastic_rounding", True)
                        and not getattr(self, "_nvfp4_backward_dv_dot_rtn", False)
                    ),
                    backward_ds_dq_stochastic_rounding=(
                        getattr(self, "_nvfp4_stochastic_rounding", True)
                        and not getattr(self, "_nvfp4_backward_dq_ds_rtn", False)
                    ),
                    dkdv_scratch_bf16=getattr(
                        self, "_nvfp4_backward_dkdv_scratch_bf16", False
                    ),
                ).transpose(1, 2)
            else:
                # Write roped K/V to the cache so subsequent decode steps see prefill
                # context, then run NVFP4 on the same roped K/V.
                fuse_v = (
                    getattr(self, "_nvfp4_fuse_vproj", _FUSE_VPROJ)
                    and past_key_values is None
                    and _can_fuse_vproj(self)
                )
                value_states = None
                if not fuse_v:
                    value_states = (
                        self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    )
                if past_key_values is not None:
                    k_roped, _ = apply_rotary_pos_emb(
                        key_states, key_states, cos, sin
                    )
                    past_key_values.update(k_roped, value_states, self.layer_idx)

                attn_output = _nvfp4_attention(
                    self, query_states, key_states, value_states, cos, sin,
                    self.scaling, causal=(kind == "causal"),
                    hidden_states=hidden_states if fuse_v else None,
                )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            return self.o_proj(attn_output), None

        return orig_forward(
            self, hidden_states, position_embeddings, attention_mask,
            past_key_values, **kwargs,
        )

    return forward


def patch_qwen3_5_nvfp4_attention(
    model: nn.Module,
    fuse_vproj: bool = _FUSE_VPROJ,
    train_backward: bool = False,
    save_backward_packs: bool = False,
    backward_dv_p_rtn: bool = False,
    backward_dv_dot_rtn: bool = False,
    backward_dq_ds_rtn: bool = False,
    backward_dkdv_scratch_bf16: bool = False,
    compile_custom_op: bool = False,
    stochastic_rounding: bool = True,
    layer_autograd: bool = False,
) -> int:
    """Patch every Qwen3.5 FULL-attention layer's forward to use NVFP4 attention.

    Leaves linear-attention layers untouched. Returns the count of patched layers.
    Idempotent: re-patching a model is a no-op on already-patched modules.

    ``fuse_vproj``: run v_proj as a native-NVFP4 GEMM with a fused key-axis pack
    epilogue (no bf16 V / transpose / standalone quant). Faster V producer at a
    small parity cost (v_proj goes FP4); only active on the no-grad cache-free
    prefill path and only for plain ``nn.Linear`` v_proj modules.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    patched = 0
    seen_forward = None
    for module in model.modules():
        if isinstance(module, Qwen3_5Attention):
            if getattr(module, "_nvfp4_patched", False):
                module._nvfp4_fuse_vproj = fuse_vproj
                module._nvfp4_train_backward = train_backward
                module._nvfp4_save_backward_packs = save_backward_packs
                module._nvfp4_backward_dv_p_rtn = backward_dv_p_rtn
                module._nvfp4_backward_dv_dot_rtn = backward_dv_dot_rtn
                module._nvfp4_backward_dq_ds_rtn = backward_dq_ds_rtn
                module._nvfp4_backward_dkdv_scratch_bf16 = backward_dkdv_scratch_bf16
                module._nvfp4_compile_custom_op = compile_custom_op
                module._nvfp4_stochastic_rounding = stochastic_rounding
                module._nvfp4_layer_autograd = layer_autograd
                continue
            orig = type(module).forward
            if seen_forward is None:
                seen_forward = make_nvfp4_forward(orig)
            module.forward = seen_forward.__get__(module, type(module))
            module._nvfp4_patched = True
            module._nvfp4_fuse_vproj = fuse_vproj
            module._nvfp4_train_backward = train_backward
            module._nvfp4_save_backward_packs = save_backward_packs
            module._nvfp4_backward_dv_p_rtn = backward_dv_p_rtn
            module._nvfp4_backward_dv_dot_rtn = backward_dv_dot_rtn
            module._nvfp4_backward_dq_ds_rtn = backward_dq_ds_rtn
            module._nvfp4_backward_dkdv_scratch_bf16 = backward_dkdv_scratch_bf16
            module._nvfp4_compile_custom_op = compile_custom_op
            module._nvfp4_stochastic_rounding = stochastic_rounding
            module._nvfp4_layer_autograd = layer_autograd
            patched += 1
    LOG.info(
        "nvfp4 attention: patched %d Qwen3.5 full-attention layers "
        "(fuse_vproj=%s, train_backward=%s, save_backward_packs=%s, "
        "backward_dv_p_rtn=%s, backward_dv_dot_rtn=%s, "
        "backward_dq_ds_rtn=%s, backward_dkdv_scratch_bf16=%s, "
        "compile_custom_op=%s, layer_autograd=%s)",
        patched, fuse_vproj, train_backward, save_backward_packs,
        backward_dv_p_rtn, backward_dv_dot_rtn, backward_dq_ds_rtn,
        backward_dkdv_scratch_bf16, compile_custom_op, layer_autograd,
    )
    return patched
