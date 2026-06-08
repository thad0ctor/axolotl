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
_CUSTOM_OP_ENV = "AXOLOTL_NVFP4_QWEN35_ATTENTION_CUSTOM_OP"


def _custom_op_enabled(module: nn.Module) -> bool:
    requested = getattr(module, "_nvfp4_compile_custom_op", False) or os.environ.get(
        _CUSTOM_OP_ENV, ""
    ).lower() in {"1", "true", "yes"}
    return bool(requested) and torch.compiler.is_compiling()


_DENSE_KIND_ATTR = "_nvfp4_dense_kind"
_TRIL_CACHE: dict[tuple, torch.Tensor] = {}


def _causal_tril(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular keep mask, cached by (shape, device) to avoid rebuilding the
    O(S^2) tensor on every classification."""
    key = (q_len, kv_len, device)
    t = _TRIL_CACHE.get(key)
    if t is None:
        t = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool, device=device))
        _TRIL_CACHE[key] = t
    return t


def _classify_dense_mask(
    attention_mask: torch.Tensor, q_len: int, kv_len: int
) -> str | None:
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
        if bool((keep == _causal_tril(q_len, kv_len, keep.device)[None]).all()):
            return "causal"
    return None


def _mask_is_dense_causal_or_full(
    attention_mask: torch.Tensor | None, q_len: int, kv_len: int
) -> str | None:
    """Return 'causal' / 'full' if the mask is a dense causal/full mask, else None.

    None -> there is per-key padding (or a mask shape we can't densely represent),
    so the caller must fall back to the original forward.

    The dense 4D classification needs ``bool(...)`` reductions (each a device->host
    sync) plus an O(Z*S^2) full-mask compare. The mask tensor is shared across all
    decoder layers in a forward, so the result is cached on it (keyed by shape and
    ``_version``) — computed once per model-forward, not once per attention layer.
    """
    if attention_mask is None:
        return "causal" if q_len == kv_len else "full"
    if attention_mask.dim() != 4:
        return None
    ckey = (q_len, kv_len, attention_mask._version)
    cached = getattr(attention_mask, _DENSE_KIND_ATTR, None)
    if cached is not None and cached[0] == ckey:
        return cached[1]
    kind = _classify_dense_mask(attention_mask, q_len, kv_len)
    try:
        setattr(attention_mask, _DENSE_KIND_ATTR, (ckey, kind))
    except (AttributeError, RuntimeError):
        pass  # some tensor subclasses disallow setattr; correctness is unaffected
    return kind


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
    query_states: torch.Tensor,  # [Z, H, S, D]  post q_norm, PRE-RoPE
    key_states: torch.Tensor,  # [Z, Hk, S, D] post k_norm, PRE-RoPE
    value_states: torch.Tensor,  # [Z, Hk, Skv, D]
    cos: torch.Tensor,  # [Z, S, rotary_dim]
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

    if _custom_op_enabled(module):
        from axolotl.kernels.attn_nvfp4_custom_op import (
            nvfp4_flash_attention_packed_custom_op,
        )

        # The compile custom op keeps the [Z, H, S, D] schema (its registered op /
        # fake are layout-fixed), so the transpose stays on the compiled path.
        out = nvfp4_flash_attention_packed_custom_op(
            qnv,
            qsc,
            knv,
            ksc,
            vnv,
            vsc,
            z=z,
            h=h,
            hk=hk,
            s_q=s_q,
            s_kv=s_kv,
            d=d,
            scaling=scaling,
            out_dtype=query_states.dtype,
            causal=causal,
            block_n=_BLOCK_N,
        )  # [Z, H, S, D]
        return out.transpose(1, 2)  # [Z, S, H, D]

    # Eager path: the kernel writes the [Z, S, H, D] HF attn_output layout directly,
    # so the per-layer transpose(1,2)+contiguous copy at the caller is eliminated.
    return nvfp4_flash_attention_packed(
        qnv,
        qsc,
        knv,
        ksc,
        vnv,
        vsc,
        z=z,
        h=h,
        hk=hk,
        s_q=s_q,
        s_kv=s_kv,
        d=d,
        scaling=scaling,
        out_dtype=query_states.dtype,
        causal=causal,
        block_n=_BLOCK_N,
        out_layout="zshd",
    )  # [Z, S, H, D]


# ---------------------------------------------------------------------------
# Shared-activation-pack FP4 for the full-attention block's hidden-consuming
# projections. q_proj (gated, -> 2*head_dim) and k_proj BOTH read hidden_states
# and contract the same K=hidden, so hidden is NVFP4-packed ONCE and reused for
# both FP4 GEMMs (the cross-op lever already used by the MLP gate/up and the GDN
# in_proj). o_proj (post-attention activation) gets its own FP4 GEMM. All
# parity-affecting (q/k/o go FP4) -> opt-in, default OFF, plain-nn.Linear only.
# ---------------------------------------------------------------------------
def _attn_proj_ok(module: nn.Module) -> bool:
    from axolotl.monkeypatch.attention.nvfp4_linear_attn import _is_plain_linear

    if not (
        _is_plain_linear(module.q_proj)
        and _is_plain_linear(module.k_proj)
        and _is_plain_linear(module.o_proj)
    ):
        return False
    return (
        module.q_proj.in_features % 16 == 0
        and module.k_proj.in_features % 16 == 0
        and module.o_proj.in_features % 16 == 0
    )


def _nvfp4_qk_proj(module, hidden_states):
    """FP4 q_proj + k_proj sharing ONE NVFP4 pack of hidden_states."""
    from axolotl.kernels.attn_nvfp4_flash import _quant_nvfp4
    from axolotl.monkeypatch.attention.nvfp4_linear_attn import (
        _gemm_from_packed_act,
        _get_packed_weight,
    )

    dtype = hidden_states.dtype
    k = hidden_states.shape[-1]
    lead = hidden_states.shape[:-1]
    x2d = hidden_states.reshape(-1, k)
    m = x2d.shape[0]
    anv, asc = _quant_nvfp4(x2d.unsqueeze(0))
    anv, asc = anv[0], asc[0]
    q_wnv, q_wsc = _get_packed_weight(module, "_qproj_packed", module.q_proj)
    k_wnv, k_wsc = _get_packed_weight(module, "_kproj_packed", module.k_proj)
    q_outf, k_outf = module.q_proj.out_features, module.k_proj.out_features
    q = _gemm_from_packed_act(anv, asc, q_wnv, q_wsc, m, q_outf, k, dtype)
    kk = _gemm_from_packed_act(anv, asc, k_wnv, k_wsc, m, k_outf, k, dtype)
    if module.q_proj.bias is not None:
        q = q + module.q_proj.bias
    if module.k_proj.bias is not None:
        kk = kk + module.k_proj.bias
    return q.reshape(*lead, q_outf), kk.reshape(*lead, k_outf)


def _nvfp4_o_proj(module, attn_output):
    """FP4 o_proj (+bias)."""
    from axolotl.kernels.nvfp4_linear import nvfp4_linear
    from axolotl.monkeypatch.attention.nvfp4_linear_attn import _get_packed_weight

    o_wnv, o_wsc = _get_packed_weight(module, "_oproj_packed", module.o_proj)
    out = nvfp4_linear(attn_output, o_wnv, o_wsc, module.o_proj.out_features)
    if module.o_proj.bias is not None:
        out = out + module.o_proj.bias
    return out


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
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
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

        # Shared-pack FP4 q/k_proj: no-grad prefill, opt-in, plain-Linear only.
        use_fp4_proj = (
            not grad_enabled
            and kind is not None
            and getattr(self, "_nvfp4_fuse_attn_proj", False)
            and getattr(self, "_nvfp4_attn_proj_ok", False)
        )
        # q/k feed QK^T->softmax (error amplified through exp); o_proj is a direct
        # readout. Separable sub-gates let q/k stay bf16 while o_proj goes FP4 (the
        # near-lossless subset). Both default ON when the main flag is on.
        use_fp4_qk = use_fp4_proj and getattr(self, "_nvfp4_fp4_qk", True)
        use_fp4_o = use_fp4_proj and getattr(self, "_nvfp4_fp4_o", True)
        if use_fp4_qk:
            q_full, k_full = _nvfp4_qk_proj(self, hidden_states)
            query_states, gate = torch.chunk(
                q_full.view(*input_shape, -1, self.head_dim * 2),
                2,
                dim=-1,
            )
        else:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
                2,
                dim=-1,
            )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        if use_fp4_qk:
            key_states = self.k_norm(k_full.view(hidden_shape)).transpose(1, 2)
        else:
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
                        self,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values,
                        **kwargs,
                    )
                # Break the Inductor graph on BOTH sides of the FP4 attention block.
                # Fused with the FP4 plugin's quantized q/k/v/o_proj autograd, the
                # surrounding compiled backward miscompiles (param grads spike ~1e5
                # -> NaN). The op stays opaque (no InductorError, no bwd eager
                # fallback); the breaks just give it the eager boundary the
                # eager-fallback baseline already gets. Rest of the model still compiles.
                if _custom_op_enabled(self):
                    torch._dynamo.graph_break()
                value_states = (
                    self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                )
                query_roped, key_roped = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
                sr = getattr(self, "_nvfp4_stochastic_rounding", True)
                grad_sr = sr and not getattr(
                    self, "_nvfp4_backward_rtn_grad_packs", False
                )
                ng = query_states.shape[1] // key_states.shape[1]
                if _custom_op_enabled(self):
                    # Differentiable opaque custom op: Inductor compiles AROUND the
                    # whole native-NVFP4 attention (fwd + registered native-NVFP4
                    # bwd) instead of tracing tl.dot_scaled and silently falling the
                    # bwd subgraph back to eager. Same grads as nvfp4_flash_attn_func.
                    from axolotl.kernels.attn_nvfp4_custom_op import (
                        nvfp4_flash_attn_train_custom_op,
                    )

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
                        dkdv_scratch_bf16=getattr(
                            self, "_nvfp4_dkdv_scratch_bf16", False
                        ),
                        save_backward_packs=getattr(
                            self, "_nvfp4_save_backward_packs", False
                        ),
                        out_layout="zshd",
                    )
                    # trailing half of the isolating break (see the leading one above)
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
                        save_backward_packs=getattr(
                            self, "_nvfp4_save_backward_packs", False
                        ),
                        dkdv_scratch_bf16=getattr(
                            self, "_nvfp4_dkdv_scratch_bf16", False
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
                    k_roped, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
                    past_key_values.update(k_roped, value_states, self.layer_idx)

                attn_output = _nvfp4_attention(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    cos,
                    sin,
                    self.scaling,
                    causal=(kind == "causal"),
                    hidden_states=hidden_states if fuse_v else None,
                )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            if use_fp4_o:
                return _nvfp4_o_proj(self, attn_output), None
            return self.o_proj(attn_output), None

        return orig_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )

    return forward


def patch_qwen3_5_nvfp4_attention(
    model: nn.Module,
    fuse_vproj: bool = _FUSE_VPROJ,
    fuse_attn_proj: bool = False,
    train_backward: bool = False,
    backward_rtn_grad_packs: bool = False,
    save_backward_packs: bool = False,
    dkdv_scratch_bf16: bool = False,
    compile_custom_op: bool = False,
    stochastic_rounding: bool = True,
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

    # qwen3_5_moe uses its own full-attention class (Qwen3_5MoeAttention) for the
    # full_attention layers; its GatedDeltaNet/Vision attention are separate classes
    # and stay untouched. Same head_dim-256 gated softmax forward, so the wrapper fits.
    attn_classes: tuple = (Qwen3_5Attention,)
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeAttention,
        )

        attn_classes = attn_classes + (Qwen3_5MoeAttention,)
    except ImportError:
        pass

    patched = 0
    seen_forward = None
    for module in model.modules():
        if isinstance(module, attn_classes):
            if getattr(module, "_nvfp4_patched", False):
                module._nvfp4_fuse_vproj = fuse_vproj
                module._nvfp4_train_backward = train_backward
                module._nvfp4_backward_rtn_grad_packs = backward_rtn_grad_packs
                module._nvfp4_save_backward_packs = save_backward_packs
                module._nvfp4_dkdv_scratch_bf16 = dkdv_scratch_bf16
                module._nvfp4_compile_custom_op = compile_custom_op
                module._nvfp4_stochastic_rounding = stochastic_rounding
                module._nvfp4_fuse_attn_proj = fuse_attn_proj
                module._nvfp4_attn_proj_ok = (
                    _attn_proj_ok(module) if fuse_attn_proj else False
                )
                continue
            orig = type(module).forward
            if seen_forward is None:
                seen_forward = make_nvfp4_forward(orig)
            module.forward = seen_forward.__get__(module, type(module))
            module._nvfp4_patched = True
            module._nvfp4_fuse_vproj = fuse_vproj
            module._nvfp4_train_backward = train_backward
            module._nvfp4_backward_rtn_grad_packs = backward_rtn_grad_packs
            module._nvfp4_save_backward_packs = save_backward_packs
            module._nvfp4_dkdv_scratch_bf16 = dkdv_scratch_bf16
            module._nvfp4_compile_custom_op = compile_custom_op
            module._nvfp4_stochastic_rounding = stochastic_rounding
            module._nvfp4_fuse_attn_proj = fuse_attn_proj
            module._nvfp4_attn_proj_ok = (
                _attn_proj_ok(module) if fuse_attn_proj else False
            )
            patched += 1
    LOG.info(
        "nvfp4 attention: patched %d Qwen3.5 full-attention layers "
        "(fuse_vproj=%s, train_backward=%s, backward_rtn_grad_packs=%s, "
        "save_backward_packs=%s, dkdv_scratch_bf16=%s, compile_custom_op=%s)",
        patched,
        fuse_vproj,
        train_backward,
        backward_rtn_grad_packs,
        save_backward_packs,
        dkdv_scratch_bf16,
        compile_custom_op,
    )
    return patched
