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
from ._vendor.sparselora.modules.linear import SparseLinear
from ._vendor.sparselora.modules.llama import SparseLlamaAttention, SparseLlamaMLP

_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
# Optional per-arch attention attributes carried onto the sparse module.
_OPTIONAL_ATTN_ATTRS = ("q_norm", "k_norm", "sliding_window")


def is_swiglu_mlp(module: nn.Module) -> bool:
    return all(hasattr(module, proj) for proj in _MLP_PROJECTIONS)


def is_standard_attention(module: nn.Module) -> bool:
    return all(hasattr(module, proj) for proj in _ATTN_PROJECTIONS)


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
        super().__init__(base, name=name, idx=idx, sparsity=sparsity, cfg=cfg)
        for attr in _OPTIONAL_ATTN_ATTRS:
            setattr(self, attr, getattr(base, attr, None))

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
            self.config._attn_implementation, eager_attention_forward
        )
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
    """Why a target attention module can't be sparsified safely, or ``None``.

    Catches semantics the generic wiring does not reproduce (e.g. Gemma2/Gemma3
    attention-logit softcapping), so the plugin can refuse with a clear message
    instead of silently dropping the term.
    """
    if is_standard_attention(module):
        softcap = getattr(
            getattr(module, "config", None), "attn_logit_softcapping", None
        )
        if softcap is not None:
            return (
                f"{type(module).__name__} uses attention-logit softcapping "
                f"({softcap}), which SparseLoRA's generic attention wiring does "
                "not apply. This architecture is not supported."
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
