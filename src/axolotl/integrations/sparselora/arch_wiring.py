"""Model-agnostic sparse wiring for SparseLoRA.

The vendored ``_vendor/sparselora/modules/llama.py`` hardcodes Llama's MLP and
attention. This module adds architecture-neutral replacements and an
auto-registration pass so the plugin works on any SwiGLU-MLP + standard-attention
transformer (Qwen2, Qwen3, Mistral, ...) without per-arch user config.

Two observations make this possible:

- The SwiGLU MLP forward (``down(silu(gate(x)) * up(x))``) is identical across
  these families, so the vendored ``SparseLlamaMLP`` wiring is reused verbatim.
- Attention differs only in three places that can be introspected per module:
  projection **bias** (Qwen2 q/k/v), **q_norm/k_norm** on the head dim (Qwen3),
  and a **sliding_window** passed to the attention interface (Qwen2/Qwen3/Mistral).
  :class:`SparseAttention` handles all three from attributes copied off the base
  module, falling back to Llama semantics when they are absent.

Nothing here edits vendored code: the bias-aware linear and generic attention
subclass the vendored classes, and registration goes through the public
``register_sparse_module`` extension point.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from ._vendor.sparselora.modules import (
    get_module_mapping,
    register_sparse_module,
)
from ._vendor.sparselora.modules.base import SparseModule
from ._vendor.sparselora.modules.linear import SparseLinear
from ._vendor.sparselora.modules.llama import SparseLlamaAttention, SparseLlamaMLP
from ._vendor.sparselora.modules.predictors import (
    AttentionPredictor,
    GQAAttentionPredictor,
)

_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
# Optional per-arch attention attributes carried onto the sparse module.
# ``attn_logit_softcapping`` (Gemma2) is forwarded to the attention interface;
# ``q_norm``/``k_norm`` (Qwen3/Gemma3) and ``sliding_window`` are applied inline.
_OPTIONAL_ATTN_ATTRS = ("q_norm", "k_norm", "sliding_window", "attn_logit_softcapping")


def is_swiglu_mlp(module: nn.Module) -> bool:
    return all(hasattr(module, proj) for proj in _MLP_PROJECTIONS)


def is_standard_attention(module: nn.Module) -> bool:
    return all(hasattr(module, proj) for proj in _ATTN_PROJECTIONS)


def is_silu_gated(module: nn.Module) -> bool:
    """Whether the MLP's activation is SiLU (the sparse FFN path hardcodes it).

    The vendored ``FFNPredictor`` and ``SparseLlamaMLP`` both use liger's
    ``silu_mul``; a SwiGLU MLP with a different gate (e.g. Gemma's
    ``gelu_pytorch_tanh``) would be silently miscomputed. Activation classes vary
    (``SiLUActivation`` vs ``nn.SiLU``), so probe numerically against ``F.silu``.
    """
    act = getattr(module, "act_fn", None)
    if act is None:
        return False
    try:
        probe = torch.linspace(-3.0, 3.0, 16)
        return torch.allclose(act(probe), F.silu(probe), atol=1e-6)
    except Exception:  # noqa: BLE001 - any non-elementwise activation is "not SiLU"
        return False


class SparseLinearBias(SparseLinear):
    """Bias-aware drop-in for the vendored :class:`SparseLinear`.

    Upstream ``SparseLinear`` ignores the base layer's bias entirely (Llama
    q/k/v/o and SwiGLU projections are bias-free, so it never mattered). Qwen2
    attention has q/k/v bias, so the sparse path must apply it — sliced to the
    selected output channels in output-sparse modes, full in input-sparse modes.
    With ``bias is None`` this is bit-identical to the vendored class.
    """

    def __init__(self, base: nn.Module, mode: str) -> None:
        super().__init__(base, mode)
        # Share (not clone) the base layer's trained, frozen bias.
        self.bias = base.bias

    def forward(
        self,
        x: torch.Tensor,
        sparse_indices: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        x = x.contiguous()
        if sparse_indices is None or self.mode is None:
            return F.linear(x, self.weight, self.bias)

        if self.mode == "in_gather":
            idx = (
                sparse_indices.unsqueeze(0)
                .unsqueeze(0)
                .expand(x.shape[0], x.shape[1], -1)
            )
            with torch.no_grad():
                x = torch.gather(x, 2, idx).contiguous()

        if self.mode.startswith("in"):
            # Input-sparse: output keeps the full out dim, so bias applies whole.
            w = self.weight[:, sparse_indices].contiguous()
            return F.linear(x, w, self.bias)

        # Output-sparse: keep only the selected rows of weight (and bias).
        w = self.weight[sparse_indices].contiguous()
        b = None if self.bias is None else self.bias[sparse_indices].contiguous()
        x = F.linear(x, w, b)

        if self.mode == "out_scatter" and x.shape[-1] != self.weight.shape[0]:
            with torch.no_grad():
                out = torch.zeros(
                    x.shape[0],
                    x.shape[1],
                    self.weight.shape[0],
                    dtype=x.dtype,
                    device=x.device,
                )
                idx = (
                    sparse_indices.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(x.shape[0], x.shape[1], -1)
                )
                x = out.scatter_add(2, idx, x).contiguous()

        return x


class SparseSwiGLUMLP(SparseLlamaMLP):
    """Architecture-neutral SwiGLU MLP (Qwen2/Qwen3/Mistral/Phi3/...).

    The vendored ``SparseLlamaMLP`` forward only touches ``gate_proj`` /
    ``up_proj`` / ``down_proj`` and ``silu_mul``, which is the shared SwiGLU
    contract; this alias makes that intent explicit and keeps the registry
    keyed by the architecture's own MLP class.
    """


class _GQAAttentionPredictorHD(GQAAttentionPredictor):
    """GQA predictor that allows ``q_proj`` out != ``hidden_size``.

    The vendored ``GQAAttentionPredictor`` sizes its ``q2`` buffer as
    ``(hidden_size, rank)``, assuming ``num_heads * head_dim == hidden_size``.
    Qwen3 decouples ``head_dim`` (e.g. 128) from ``hidden/num_heads``, so the q
    projection is wider than hidden and the buffer mismatches on load. Re-register
    ``q2`` at the true q dimension.
    """

    def __init__(self, hidden_size: int, kv_size: int, q_size: int, rank: int) -> None:
        super().__init__(hidden_size, kv_size, rank)
        if q_size != hidden_size:
            self.register_buffer("q2", torch.empty(q_size, rank))


class _AttentionPredictorHD(AttentionPredictor):
    """Non-GQA predictor without the ``hidden == out`` assumption.

    Same decoupled-``head_dim`` issue as the GQA case; size ``w2`` at the true
    projection out dim instead of ``hidden_size``.
    """

    def __init__(self, hidden_size: int, out_size: int, rank: int) -> None:
        nn.Module.__init__(self)
        self.register_buffer("w1", torch.empty(3, hidden_size, rank))
        self.register_buffer("w2", torch.empty(3, rank, out_size))


def _create_attn_predictor(base: nn.Module, rank: int, layer_name: str, cfg):
    """Like the vendored ``create_attn_predictor`` but tolerant of decoupled
    ``head_dim`` (q/k/v out dims sized from the factor tensors, not ``hidden``).
    """
    from ._vendor.sparselora.modules import svd

    device = base.q_proj.base_layer.weight.device
    dtype = svd._float_dtype(base.q_proj.base_layer.weight)
    tensors = svd._load_tensors(cfg, device, dtype)

    p = layer_name
    q = [tensors[f"{p}.q_proj.w1"], tensors[f"{p}.q_proj.w2"]]
    k = [tensors[f"{p}.k_proj.w1"], tensors[f"{p}.k_proj.w2"]]
    v = [tensors[f"{p}.v_proj.w1"], tensors[f"{p}.v_proj.w2"]]

    if q[1].shape != k[1].shape:  # GQA
        state_dict = {
            "w1": torch.stack([k[0], v[0]]),
            "w2": torch.stack([k[1], v[1]]),
            "q1": q[0].transpose(0, 1),
            "q2": q[1].transpose(0, 1),
        }
        pred: nn.Module = _GQAAttentionPredictorHD(
            q[0].shape[0], k[1].shape[-1], q[1].shape[-1], rank
        )
    else:
        state_dict = {
            "w1": torch.stack([q[0], k[0], v[0]]),
            "w2": torch.stack([q[1], k[1], v[1]]),
        }
        pred = _AttentionPredictorHD(q[0].shape[0], q[1].shape[-1], rank)

    pred.load_state_dict(state_dict)
    pred.eval()
    pred = torch.compile(pred)
    return pred.to(device=device, dtype=torch.bfloat16)


class SparseAttention(SparseLlamaAttention):
    """Architecture-neutral sparse attention.

    Extends the vendored Llama sparse attention with optional ``q_norm``/
    ``k_norm`` (Qwen3) applied on the head dim before the RoPE transpose, and a
    ``sliding_window`` forwarded to the attention interface (Qwen2/Qwen3/
    Mistral). Projection bias (Qwen2) is handled by :class:`SparseLinearBias`.
    When none are present this reproduces Llama attention.
    """

    def __init__(
        self, base: nn.Module, *, name: str, idx: int, sparsity: float = 0, cfg
    ) -> None:
        # Mirror SparseLlamaAttention.__init__ but build the predictor with the
        # decoupled-head_dim-tolerant creator (Qwen3 has q_out != hidden_size).
        SparseModule.__init__(self, base)
        self.sparsity = sparsity
        if self.sparsity > 0:
            self.pred = _create_attn_predictor(base, cfg.predictor_rank, name, cfg)
        for attr in _OPTIONAL_ATTN_ATTRS:
            setattr(self, attr, getattr(base, attr, None))
        # Use the base architecture's own eager attention as the fallback so
        # arch-specific terms (e.g. Gemma2's softcap) are applied when the model
        # runs eager; default to Llama's only if the module has none.
        import sys

        base_mod = sys.modules.get(type(base).__module__)
        self._eager_attention = getattr(
            base_mod, "eager_attention_forward", eager_attention_forward
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        sparse = self.enabled and self.sparsity > 0

        if sparse:
            q_i, k_i, v_i = self.pred.predict(hidden_states, self.sparsity)
            Q, K, V = self._proj_qkv(hidden_states, masks, q_i, k_i, v_i)
        else:
            Q, K, V = (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            )

        Q = Q.view(hidden_shape)
        K = K.view(hidden_shape)
        V = V.view(hidden_shape).transpose(1, 2)
        if self.q_norm is not None:
            Q = self.q_norm(Q)
        if self.k_norm is not None:
            K = self.k_norm(K)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)

        assert position_embeddings is not None
        cos, sin = position_embeddings
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            K, V = past_key_values.update(K, V, self.layer_idx, cache_kwargs)

        attention_fn = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, self._eager_attention
        )
        # Gemma2's attention interface applies softcap from this kwarg; for
        # architectures without it (None) the interface ignores it.
        if self.attn_logit_softcapping is not None:
            kwargs.setdefault("softcap", self.attn_logit_softcapping)
        attn_output, attn_weights = attention_fn(
            self,
            Q,
            K,
            V,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = (
            self._proj_o(attn_output, masks, v_i)
            if sparse
            else self.o_proj(attn_output)
        )
        return attn_output, attn_weights


def unsupported_reason(module: nn.Module) -> Optional[str]:
    """Why a target module can't be sparsified safely, or ``None``.

    The binding constraint for Gemma2/Gemma3 is the MLP gate: their SwiGLU uses
    ``gelu_pytorch_tanh``, but the vendored FFN predictor and sparse MLP kernel
    hardcode SiLU, so sparsifying that MLP would silently miscompute it. (Gemma2
    attention-logit softcapping *is* now reproduced — forwarded to the attention
    interface in :class:`SparseAttention` — so it is no longer a blocker.)
    """
    if is_swiglu_mlp(module) and not is_silu_gated(module):
        act = type(getattr(module, "act_fn", None)).__name__
        return (
            f"{type(module).__name__} uses a non-SiLU gated activation ({act}); "
            "SparseLoRA's FFN predictor and sparse MLP kernel hardcode SiLU, so "
            "this MLP cannot be sparsified faithfully. This architecture is not "
            "supported."
        )
    return None


def register_arch_wiring(model: nn.Module) -> dict[str, str]:
    """Auto-register sparse wiring for the loaded model's MLP/attention classes.

    Introspects every module: SwiGLU MLPs map to :class:`SparseSwiGLUMLP`,
    standard attention blocks to :class:`SparseAttention`. ``nn.Linear`` is
    (re)mapped to the bias-aware :class:`SparseLinearBias` (a safe superset of
    the vendored linear). Classes already registered — Llama, or anything a
    caller registered explicitly — are left untouched. Idempotent.

    Returns a ``{class_name: role}`` map of what was newly registered, for
    logging.
    """
    register_sparse_module(nn.Linear, SparseLinearBias)

    registered: dict[str, str] = {}
    mapping = get_module_mapping()
    for module in model.modules():
        cls = type(module)
        if cls in mapping:
            continue
        if is_swiglu_mlp(module):
            register_sparse_module(cls, SparseSwiGLUMLP)
            mapping[cls] = SparseSwiGLUMLP
            registered[cls.__name__] = "mlp"
        elif is_standard_attention(module):
            register_sparse_module(cls, SparseAttention)
            mapping[cls] = SparseAttention
            registered[cls.__name__] = "attention"
    return registered
