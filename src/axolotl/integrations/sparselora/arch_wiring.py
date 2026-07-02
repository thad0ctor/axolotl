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

Phi3 fuses q/k/v into one ``qkv_proj`` and gate/up into one ``gate_up_proj``.
:class:`SparseFusedQKVAttention` / :class:`SparseFusedGateUpMLP` slice the fused
weight into its q/k/v (or gate/up) sub-blocks and project them in one call with a
combined index, selecting exactly the channels the separate-projection path
would — so they subclass the generic modules and reuse the same predictors.

The sparse modules here subclass the vendored classes and register through the
public ``register_sparse_module`` extension point. (The one vendored change the
fused path needs — teaching ``get_sparsity_mode`` that ``qkv_proj``/
``gate_up_proj`` are output-sparse — lives in ``_vendor`` and is documented in
``_vendor/PROVENANCE.md``.)
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
    FFNPredictor,
    GQAAttentionPredictor,
)
from .factors import (
    attention_output_projection_name,
    fused_intermediate_size,
    fused_qkv_projection_name,
    fused_qkv_sizes,
    non_gated_mlp_projection_names,
)

try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    _silu_mul = LigerSiLUMulFunction.apply
except Exception:  # noqa: BLE001 - liger is CUDA-only; the gelu path doesn't need it
    _silu_mul = None

_fused_rms_norm_rope: Any
try:
    from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

    _fused_rms_norm_rope = fused_rms_norm_rope
except Exception:  # noqa: BLE001 - Triton kernel; absent off-CUDA. Falls back to eager.
    _fused_rms_norm_rope = None

_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj")
_ATTN_OUTPUT_PROJECTIONS = ("o_proj", "proj", "out_proj", "projection_layer")
# Optional per-arch attention attributes carried onto the sparse module.
# ``attn_logit_softcapping`` (Gemma2) is forwarded to the attention interface;
# ``q_norm``/``k_norm`` (Qwen3/Gemma3) and ``sliding_window`` are applied inline.
_OPTIONAL_ATTN_ATTRS = ("q_norm", "k_norm", "sliding_window", "attn_logit_softcapping")


def _unwrap_linear(module: Any) -> Any:
    base = getattr(module, "base_layer", module)
    return getattr(base, "linear", base)


def _proj_weight(module: Any) -> Optional[torch.Tensor]:
    weight = getattr(_unwrap_linear(module), "weight", None)
    return weight if isinstance(weight, torch.Tensor) and weight.dim() == 2 else None


def _is_linear_proj(module: nn.Module, name: str) -> bool:
    """True if ``module.<name>`` is a 2-D linear projection (``nn.Linear`` or a
    LoRA/sparse/quantized wrapper of one), not a batched 3-D expert weight.

    MoE expert blocks (e.g. ``Qwen3_5MoeExperts``) expose ``gate_up_proj`` /
    ``down_proj`` as **batched 3-D Parameters** ``(num_experts, in, out)`` rather
    than Linears. They satisfy a bare ``hasattr`` check and would be misdetected
    as a fused SwiGLU MLP, so the structural detectors require a 2-D weight. The
    routed experts are then left dense (contextual sparsity over grouped experts
    is out of scope); the shared expert (a real 2-D SwiGLU) is still sparsified.
    """
    proj = getattr(module, name, None)
    if proj is None:
        return False
    return _proj_weight(proj) is not None


def is_swiglu_mlp(module: nn.Module) -> bool:
    return all(_is_linear_proj(module, proj) for proj in _MLP_PROJECTIONS)


def is_non_gated_mlp(module: nn.Module) -> bool:
    return non_gated_mlp_projection_names(module) is not None


def is_standard_attention(module: nn.Module) -> bool:
    return all(_is_linear_proj(module, proj) for proj in _ATTN_PROJECTIONS) and any(
        _is_linear_proj(module, proj) for proj in _ATTN_OUTPUT_PROJECTIONS
    )


def is_fused_gate_up_mlp(module: nn.Module) -> bool:
    """Phi3-style fused MLP: a single ``gate_up_proj`` Linear plus ``down_proj``
    (rather than separate ``gate_proj``/``up_proj``)."""
    return _is_linear_proj(module, "gate_up_proj") and _is_linear_proj(
        module, "down_proj"
    )


def is_fused_qkv_attention(module: nn.Module) -> bool:
    """Phi3-style fused attention: a single ``qkv_proj`` Linear plus ``o_proj``
    (rather than separate ``q_proj``/``k_proj``/``v_proj``)."""
    return fused_qkv_projection_name(module) is not None and (
        attention_output_projection_name(module) is not None
    )


def _proj_out_features(proj: Any) -> Optional[int]:
    base = _unwrap_linear(proj)
    n = getattr(base, "out_features", None)
    if n is not None:
        return int(n)
    w = _proj_weight(proj)
    return None if w is None else int(w.shape[0])


def _proj_in_features(proj: Any) -> Optional[int]:
    base = _unwrap_linear(proj)
    n = getattr(base, "in_features", None)
    if n is not None:
        return int(n)
    w = _proj_weight(proj)
    return None if w is None else int(w.shape[1])


def is_gated_attention(module: nn.Module) -> bool:
    """Qwen3-Next / Qwen3.5 style *gated* attention.

    ``q_proj`` emits ``[query | gate]`` interleaved per head (out =
    ``2 * num_heads * head_dim``), and the attention output is multiplied by
    ``sigmoid(gate)`` before ``o_proj``. Detected structurally: ``q_proj`` out is
    twice ``o_proj`` in (``o_proj`` in is the un-gated attention width
    ``num_heads * head_dim``). A plain attention has ``q_proj`` out == ``o_proj``
    in, so this never matches it.
    """
    if not is_standard_attention(module):
        return False
    q_out = _proj_out_features(getattr(module, "q_proj", None))
    o_in = _proj_in_features(getattr(module, "o_proj", None))
    return q_out is not None and o_in is not None and q_out == 2 * o_in


def gate_activation(module: nn.Module):
    """The MLP's gate activation, under either the ``act_fn`` (Llama/Qwen/Gemma)
    or ``activation_fn`` (Phi3) attribute name. None if neither is present."""
    return getattr(module, "act_fn", None) or getattr(module, "activation_fn", None)


def gate_kind(module: nn.Module) -> Optional[str]:
    """Classify the SwiGLU gate activation as ``"silu"``, ``"gelu_tanh"``, or None.

    The sparse FFN path needs the gate's elementwise activation for both channel
    selection (predictor) and compute. Activation classes vary (``SiLUActivation``
    vs ``nn.SiLU``, ``GELUTanh`` vs ``PytorchGELUTanh``), so probe numerically.
    Returns None for any other (unsupported) gate.
    """
    act = gate_activation(module)
    if act is None:
        return None
    try:
        probe = torch.linspace(-3.0, 3.0, 16)
        out = act(probe)
    except Exception:  # noqa: BLE001 - non-elementwise activation is unsupported
        return None
    if torch.allclose(out, F.silu(probe), atol=1e-6):
        return "silu"
    if torch.allclose(out, F.gelu(probe, approximate="tanh"), atol=1e-6):
        return "gelu_tanh"
    return None


def is_silu_gated(module: nn.Module) -> bool:
    return gate_kind(module) == "silu"


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
        self.bias = getattr(base, "bias", None)

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
            idx = _expand_last_dim_index(sparse_indices, x)
            with torch.no_grad():
                x = torch.gather(x, x.dim() - 1, idx).contiguous()

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
                    *x.shape[:-1],
                    self.weight.shape[0],
                    dtype=x.dtype,
                    device=x.device,
                )
                idx = _expand_last_dim_index(sparse_indices, out)
                x = out.scatter_add(out.dim() - 1, idx, x).contiguous()

        return x


def _expand_last_dim_index(indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    shape = (1,) * (x.dim() - 1) + (-1,)
    return indices.reshape(shape).expand(*x.shape[:-1], -1)


def _mask_matches(x: torch.Tensor, masks: Optional[torch.Tensor]) -> bool:
    return (
        isinstance(masks, torch.Tensor)
        and x.dim() >= 3
        and tuple(masks.shape) == tuple(x.shape[:-1])
    )


def _clipped_linear_forward(
    module: nn.Module, x: torch.Tensor, sparse_indices: Optional[torch.Tensor]
) -> torch.Tensor:
    if getattr(module, "use_clipped_linears", False):
        x = torch.clamp(x, module.input_min, module.input_max)
    linear = module.linear
    x = linear(x) if sparse_indices is None else linear(x, sparse_indices)
    if getattr(module, "use_clipped_linears", False):
        x = torch.clamp(x, module.output_min, module.output_max)
    return x


def _call_proj(
    proj: nn.Module, x: torch.Tensor, sparse_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if hasattr(proj, "linear"):
        return _clipped_linear_forward(proj, x, sparse_indices)
    return proj(x) if sparse_indices is None else proj(x, sparse_indices)


def _gelu_tanh_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.gelu(gate, approximate="tanh") * up


class _GELUFFNPredictor(FFNPredictor):
    """FFN channel predictor scoring with ``gelu_tanh`` instead of SiLU (Gemma).

    Identical to the vendored :class:`FFNPredictor` except the gated activation
    used to score intermediate channels matches the model's ``gelu_pytorch_tanh``
    gate, so the selection is faithful rather than a SiLU proxy.
    """

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:
        x = x.mean(dim=1, keepdim=True).view(1, -1, x.shape[-1]).expand(2, -1, -1)
        gate, up = torch.bmm(torch.bmm(x, self.w1), self.w2).unbind(0)
        scores = _gelu_tanh_mul(gate, up).norm(dim=0)
        k = int(scores.shape[-1] * (1 - sparsity))
        return scores.topk(k).indices.flatten()


class _NonGatedMLPPredictor(nn.Module):
    """Predicts intermediate channels for ``activation(fc1(x)) -> fc2`` MLPs."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, rank: int, activation
    ) -> None:
        super().__init__()
        self.activation = activation
        self.register_buffer("w1", torch.empty(hidden_size, rank))
        self.register_buffer("w2", torch.empty(rank, intermediate_size))

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:
        x = x.reshape(-1, x.shape[-1]).mean(dim=0, keepdim=True)
        scores = self.activation((x @ self.w1) @ self.w2).norm(dim=0)
        k = max(1, int(scores.shape[-1] * (1 - sparsity)))
        return scores.topk(k).indices.flatten()


def _build_mlp_predictor(tensors: dict, p: str, rank: int, device, kind: str):
    """Assemble an FFN predictor from the ``gate_proj``/``up_proj`` factor tensors.

    Shared by the separate (``gate_proj``/``up_proj``) and fused (sliced
    ``gate_up_proj``) paths — both write the same factor keys, so the predictor
    is identical; only where ``device``/``dtype`` come from differs.
    """
    w1 = torch.stack([tensors[f"{p}.gate_proj.w1"], tensors[f"{p}.up_proj.w1"]])
    w2 = torch.stack([tensors[f"{p}.gate_proj.w2"], tensors[f"{p}.up_proj.w2"]])

    cls = FFNPredictor if kind == "silu" else _GELUFFNPredictor
    pred = cls(w1.shape[1], w2.shape[2], rank)
    pred.load_state_dict({"w1": w1, "w2": w2})
    pred.eval()
    return pred.to(device=device, dtype=torch.bfloat16)


def _create_non_gated_mlp_predictor(base: nn.Module, rank: int, layer_name: str, cfg):
    from ._vendor.sparselora.modules import svd

    names = non_gated_mlp_projection_names(base)
    if names is None:
        raise ValueError(f"{type(base).__name__} is not a non-gated MLP")
    in_name, _ = names
    weight = _proj_weight(getattr(base, in_name))
    if weight is None:
        raise ValueError(f"{type(base).__name__}.{in_name} is not a 2-D projection")
    tensors = svd._load_tensors(cfg, weight.device, svd._float_dtype(weight))
    w1 = tensors[f"{layer_name}.{in_name}.w1"]
    w2 = tensors[f"{layer_name}.{in_name}.w2"]
    pred = _NonGatedMLPPredictor(
        w1.shape[0],
        w2.shape[1],
        rank,
        getattr(base, "activation_fn", getattr(base, "act_fn", F.gelu)),
    )
    pred.load_state_dict({"w1": w1, "w2": w2})
    pred.eval()
    return pred.to(device=weight.device, dtype=torch.bfloat16)


def _create_mlp_predictor(base: nn.Module, rank: int, layer_name: str, cfg, kind: str):
    """Like the vendored ``create_mlp_predictor`` but picks the predictor class by
    gate activation (SiLU vs gelu_tanh)."""
    from ._vendor.sparselora.modules import svd

    weight = _proj_weight(base.gate_proj)
    if weight is None:
        raise ValueError(f"{type(base).__name__}.gate_proj is not a 2-D projection")
    device = weight.device
    dtype = svd._float_dtype(weight)
    tensors = svd._load_tensors(cfg, device, dtype)
    return _build_mlp_predictor(tensors, layer_name, rank, device, kind)


def _create_fused_mlp_predictor(
    base: nn.Module, rank: int, layer_name: str, cfg, kind: str
):
    """MLP predictor for a fused ``gate_up_proj`` MLP (Phi3).

    The factor tensors already carry separate ``gate_proj``/``up_proj`` keys
    (``factors.compute_factor_tensors`` slices the fused weight), so only the
    ``device``/``dtype`` source — the fused projection's weight — changes.
    """
    from ._vendor.sparselora.modules import svd

    weight = _proj_weight(base.gate_up_proj)
    if weight is None:
        raise ValueError(f"{type(base).__name__}.gate_up_proj is not a 2-D projection")
    tensors = svd._load_tensors(cfg, weight.device, svd._float_dtype(weight))
    return _build_mlp_predictor(tensors, layer_name, rank, weight.device, kind)


class SparseSwiGLUMLP(SparseLlamaMLP):
    """Architecture-neutral SwiGLU MLP (Qwen2/Qwen3/Mistral/Gemma2/Gemma3/...).

    The vendored ``SparseLlamaMLP`` hardcodes liger ``silu_mul`` for both the
    predictor's channel selection and the gated compute. This subclass reads the
    base MLP's gate activation and uses the matching path: the fused liger SiLU
    kernel for SiLU models (Llama/Qwen/Mistral, unchanged), or ``gelu_tanh`` for
    Gemma. The token split/join and sparse projection machinery are reused as-is.
    """

    inherited_attributes = SparseLlamaMLP.inherited_attributes

    def __init__(
        self, base: nn.Module, *, name: str, idx: int, sparsity: float, cfg
    ) -> None:
        SparseModule.__init__(self, base)
        self.sparsity = sparsity
        self._gate_kind = self._require_gate_kind(base)
        # Resolve the gate via the helper so a module exposing it as
        # `activation_fn` (Phi3) rather than `act_fn` still has a callable for the
        # non-liger compute path.
        self.act_fn = gate_activation(base)
        self._init_extra(base)
        if self.sparsity > 0:
            self.pred = self._build_predictor(base, name, cfg)

    def _require_gate_kind(self, base: nn.Module) -> str:
        # The plugin's _validate refuses unsupported gates via unsupported_reason(),
        # but guard here too so direct register_arch_wiring() users don't silently
        # get a wrong SiLU approximation of another gated activation.
        kind = gate_kind(base)
        if kind is None:
            raise ValueError(
                f"{type(self).__name__}: unsupported gate activation on "
                f"{type(base).__name__}; only SiLU and gelu_tanh gated MLPs are "
                "supported."
            )
        return kind

    def _init_extra(self, base: nn.Module) -> None:
        """Subclass setup hook. No-op for the separate gate/up projection."""

    def _build_predictor(self, base: nn.Module, name: str, cfg):
        return _create_mlp_predictor(
            base, cfg.predictor_rank, name, cfg, self._gate_kind
        )

    def _gate_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Gated activation ``act(gate) * up`` using the matching kernel/path."""
        if self._gate_kind == "silu" and _silu_mul is not None:
            return _silu_mul(gate, up)
        return self.act_fn(gate) * up

    def _block(self, x: torch.Tensor, **kw: Any) -> torch.Tensor:
        sparse_indices = kw.get("sparse_indices")
        gate = _call_proj(self.gate_proj, x, sparse_indices)
        up = _call_proj(self.up_proj, x, sparse_indices)
        return _call_proj(self.down_proj, self._gate_mul(gate, up), sparse_indices)

    def forward(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not (self.enabled and self.sparsity > 0):
            return self._block(x)

        indices = self.pred.predict(x, self.sparsity)
        if not _mask_matches(x, masks):
            return self._block(x, sparse_indices=indices)

        sparse_x, dense_x = self.token_splits(x, masks)
        return self.token_join(
            sparse=self._block(sparse_x, sparse_indices=indices),
            dense=self._block(dense_x),
            masks=masks,
        )


class SparseFusedGateUpMLP(SparseSwiGLUMLP):
    """Sparse SwiGLU MLP for a fused ``gate_up_proj`` (Phi3).

    Phi3 fuses gate and up into one Linear whose output rows are ``[gate | up]``
    (``up_states.chunk(2, dim=-1)``). The predictor selects intermediate channels
    ``i``; kept channel ``i`` corresponds to row ``i`` (gate) and ``intermediate +
    i`` (up) of ``gate_up_proj`` and column ``i`` of ``down_proj``. The sparse path
    projects both halves in one fused call with the combined index
    ``[i, intermediate + i]``, splits, gates, then gathers ``down_proj`` at ``i``
    — selecting exactly the channels the separate-projection path would.
    """

    inherited_attributes = ["gate_up_proj", "down_proj"]

    def _init_extra(self, base: nn.Module) -> None:
        self.act_fn = gate_activation(base)
        self.intermediate_size = fused_intermediate_size(base)

    def _build_predictor(self, base: nn.Module, name: str, cfg):
        return _create_fused_mlp_predictor(
            base, cfg.predictor_rank, name, cfg, self._gate_kind
        )

    def _block(self, x: torch.Tensor, **kw: Any) -> torch.Tensor:
        idx = kw.get("sparse_indices")
        if idx is None:
            fused = _call_proj(self.gate_up_proj, x)
        else:
            # gate row i and up row (intermediate + i) for each kept channel i.
            combined = torch.cat([idx, idx + self.intermediate_size])
            fused = _call_proj(self.gate_up_proj, x, combined)
        # The fused output is always [gate_block | up_block] of equal halves,
        # whether the (bare) gate_up_proj returns the selected channels (mode
        # "out", width 2*len(idx)) or the full padded width (mode "out_scatter");
        # split by the actual width so both paths gate correctly.
        half = fused.shape[-1] // 2
        gate, up = fused[..., :half], fused[..., half : 2 * half]
        gated = self._gate_mul(gate, up)
        return _call_proj(self.down_proj, gated, idx)


class SparseNonGatedMLP(SparseModule):
    """Sparse ``activation(fc1(x)) -> fc2`` MLP used by ViT vision towers."""

    inherited_attributes: list[str] = []

    def __init__(
        self, base: nn.Module, *, name: str, idx: int, sparsity: float, cfg
    ) -> None:
        del idx
        SparseModule.__init__(self, base)
        names = non_gated_mlp_projection_names(base)
        if names is None:
            raise ValueError(f"{type(base).__name__} is not a non-gated MLP")
        self._in_proj_name, self._out_proj_name = names
        setattr(self, self._in_proj_name, getattr(base, self._in_proj_name))
        setattr(self, self._out_proj_name, getattr(base, self._out_proj_name))
        self.activation_fn = getattr(
            base, "activation_fn", getattr(base, "act_fn", F.gelu)
        )
        self.sparsity = sparsity
        if self.sparsity > 0:
            self.pred = _create_non_gated_mlp_predictor(
                base, cfg.predictor_rank, name, cfg
            )

    def _in_proj(self) -> nn.Module:
        return getattr(self, self._in_proj_name)

    def _out_proj(self) -> nn.Module:
        return getattr(self, self._out_proj_name)

    def _block(
        self, x: torch.Tensor, sparse_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden = _call_proj(self._in_proj(), x, sparse_indices)
        hidden = self.activation_fn(hidden)
        return _call_proj(self._out_proj(), hidden, sparse_indices)

    def forward(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not (self.enabled and self.sparsity > 0):
            return self._block(x)

        indices = self.pred.predict(x, self.sparsity)
        if not _mask_matches(x, masks):
            return self._block(x, sparse_indices=indices)

        sparse_x, dense_x = self.token_splits(x, masks)
        return self.token_join(
            sparse=self._block(sparse_x, sparse_indices=indices),
            dense=self._block(dense_x),
            masks=masks,
        )


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


def _build_attn_predictor(tensors: dict, p: str, rank: int, device):
    """Assemble a (GQA-aware, decoupled-head_dim-tolerant) attention predictor
    from the ``q/k/v_proj`` factor tensors.

    Shared by the separate (q/k/v_proj) and fused (sliced qkv_proj) paths: both
    write the same factor keys, so the predictor is identical and only the
    ``device``/``dtype`` source differs.
    """
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


def _create_attn_predictor(base: nn.Module, rank: int, layer_name: str, cfg):
    """Like the vendored ``create_attn_predictor`` but tolerant of decoupled
    ``head_dim`` (q/k/v out dims sized from the factor tensors, not ``hidden``).
    """
    from ._vendor.sparselora.modules import svd

    weight = _proj_weight(base.q_proj)
    if weight is None:
        raise ValueError(f"{type(base).__name__}.q_proj is not a 2-D projection")
    tensors = svd._load_tensors(cfg, weight.device, svd._float_dtype(weight))
    return _build_attn_predictor(tensors, layer_name, rank, weight.device)


def _create_fused_attn_predictor(base: nn.Module, rank: int, layer_name: str, cfg):
    """Attention predictor for a fused ``qkv_proj`` (Phi3).

    The factor tensors already carry separate ``q/k/v_proj`` keys (sliced from
    the fused weight), so only the ``device``/``dtype`` source — the fused
    projection's (possibly LoRA-wrapped) weight — changes.
    """
    from ._vendor.sparselora.modules import svd

    qkv_name = fused_qkv_projection_name(base)
    if qkv_name is None:
        raise ValueError(f"{type(base).__name__} has no fused qkv projection")
    qkv = getattr(base, qkv_name)
    weight = _proj_weight(qkv)
    if weight is None:
        raise ValueError(f"{type(base).__name__}.{qkv_name} is not a 2-D projection")
    tensors = svd._load_tensors(cfg, weight.device, svd._float_dtype(weight))
    return _build_attn_predictor(tensors, layer_name, rank, weight.device)


def _resolve_eager_attention(base: nn.Module):
    """The base architecture's own ``eager_attention_forward``, or Llama's.

    Using the base module's eager fn preserves arch-specific terms (e.g. Gemma2's
    softcap) when the model runs eager; Llama's is the fallback when absent.
    """
    import sys

    base_mod = sys.modules.get(type(base).__module__)
    return getattr(base_mod, "eager_attention_forward", eager_attention_forward)


def _partial_rotary_dim(config: Any, head_dim: int) -> Optional[int]:
    """RoPE rotary width for partial-rotary models (StableLM/Phi), else ``None``.

    ``partial_rotary_factor < 1.0`` rotates only the leading ``factor * head_dim``
    of each head; the rest passes through. ``None`` means full-head RoPE.
    """
    prf = getattr(config, "partial_rotary_factor", 1.0) if config is not None else 1.0
    return int(prf * head_dim) if prf < 1.0 else None


class SparseAttention(SparseLlamaAttention):
    """Architecture-neutral sparse attention.

    Extends the vendored Llama sparse attention with optional ``q_norm``/
    ``k_norm`` (Qwen3) applied on the head dim before the RoPE transpose, and a
    ``sliding_window`` forwarded to the attention interface (Qwen2/Qwen3/
    Mistral). Projection bias (Qwen2) is handled by :class:`SparseLinearBias`.
    When none are present this reproduces Llama attention.

    Subclasses customize three init hooks rather than reimplement ``__init__``:
    :meth:`_init_extra` (extra attributes), :meth:`_build_predictor` (the sparse
    predictor), and :meth:`_sliding_window_fallback` (window source).
    """

    def __init__(
        self, base: nn.Module, *, name: str, idx: int, sparsity: float = 0, cfg
    ) -> None:
        # Mirror SparseLlamaAttention.__init__ but build the predictor with the
        # decoupled-head_dim-tolerant creator (Qwen3 has q_out != hidden_size).
        SparseModule.__init__(self, base)
        config = getattr(base, "config", None)
        self._rotary_dim = _partial_rotary_dim(config, self.head_dim)
        self.sparsity = sparsity
        self._init_extra(base, config)
        if self.sparsity > 0:
            self.pred = self._build_predictor(base, name, cfg)
        for attr in _OPTIONAL_ATTN_ATTRS:
            setattr(self, attr, getattr(base, attr, None))
        if getattr(self, "sliding_window", None) is None:
            self.sliding_window = self._sliding_window_fallback(config)
        self._eager_attention = _resolve_eager_attention(base)

    def _init_extra(self, base: nn.Module, config: Any) -> None:
        """Subclass setup hook (e.g. fused qkv sizes). No-op for separate q/k/v."""

    def _build_predictor(self, base: nn.Module, name: str, cfg):
        return _create_attn_predictor(base, cfg.predictor_rank, name, cfg)

    def _sliding_window_fallback(self, config: Any) -> Optional[int]:
        """Sliding window when the module carries none. No fallback by default."""
        return None

    def _gate_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Hook to gate the attention output before ``o_proj``. Identity for
        standard attention; :class:`SparseGatedAttention` applies the gate."""
        return attn_output

    def _output_proj(self) -> nn.Module:
        return self.o_proj

    def _sparse_proj(self, proj, sparse_x, dense_x, indices, masks):
        return self.token_join(
            _call_proj(proj, sparse_x, indices), _call_proj(proj, dense_x), masks
        )

    def _proj_qkv(self, x, masks, q_i, k_i, v_i):
        if masks is None:
            return (
                _call_proj(self.q_proj, x, q_i),
                _call_proj(self.k_proj, x, k_i),
                _call_proj(self.v_proj, x, v_i),
            )
        sparse_x, dense_x = self.token_splits(x, masks)
        return (
            self._sparse_proj(self.q_proj, sparse_x, dense_x, q_i, masks),
            self._sparse_proj(self.k_proj, sparse_x, dense_x, k_i, masks),
            self._sparse_proj(self.v_proj, sparse_x, dense_x, v_i, masks),
        )

    def _proj_o(self, x, masks, v_i):
        proj = self._output_proj()
        if masks is None:
            return _call_proj(proj, x, v_i)
        sparse_x, dense_x = self.token_splits(x, masks)
        return self._sparse_proj(proj, sparse_x, dense_x, v_i, masks)

    def _qk_norm_rope(self, Q: torch.Tensor, K: torch.Tensor, position_embeddings):
        """Apply optional q/k-norm then RoPE to ``(B, T, H, D)`` Q/K, returning
        ``(B, H, T, D)``. Override point for a fused norm+RoPE kernel."""
        if self.q_norm is not None:
            Q = self.q_norm(Q)
        if self.k_norm is not None:
            K = self.k_norm(K)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        cos, sin = position_embeddings
        if self._rotary_dim is None:
            return apply_rotary_pos_emb(Q, K, cos, sin)
        # Rotate only the leading rotary_dim of each head; pass the rest through.
        r = self._rotary_dim
        q_rot, q_pass = Q[..., :r], Q[..., r:]
        k_rot, k_pass = K[..., :r], K[..., r:]
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    def _qkv(self, hidden_states: torch.Tensor, masks, sparse: bool):
        """Project to flat ``(Q, K, V, v_i)``; override point for fused qkv_proj.

        ``v_i`` (the selected value channels) is threaded back to ``o_proj``;
        ``None`` on the dense path.
        """
        if sparse:
            q_i, k_i, v_i = self.pred.predict(hidden_states, self.sparsity)
            Q, K, V = self._proj_qkv(hidden_states, masks, q_i, k_i, v_i)
            return Q, K, V, v_i
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
            None,
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
        masks = masks if _mask_matches(hidden_states, masks) else None

        Q, K, V, v_i = self._qkv(hidden_states, masks, sparse)

        Q = Q.view(hidden_shape)
        K = K.view(hidden_shape)
        V = V.view(hidden_shape).transpose(1, 2)
        assert position_embeddings is not None
        Q, K = self._qk_norm_rope(Q, K, position_embeddings)
        cos, sin = position_embeddings

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
        attn_output = self._gate_output(attn_output)
        attn_output = (
            self._proj_o(attn_output, masks, v_i)
            if sparse
            else _call_proj(self._output_proj(), attn_output)
        )
        return attn_output, attn_weights


class SparseFusedQKVAttention(SparseAttention):
    """Sparse attention for a fused ``qkv_proj`` (Phi3).

    Phi3 fuses q/k/v into one Linear whose output is
    ``[q (num_heads*head_dim) | k (num_kv*head_dim) | v (num_kv*head_dim)]``.
    The predictor selects q/k/v channels independently; the fused projection runs
    in one call with the combined index ``[q_i | q_size + k_i | q_size + kv_size +
    v_i]``, then the result is split back into Q/K/V. This selects exactly the
    rows of ``qkv_proj`` that the separate q/k/v_proj path would, so the rest of
    the attention computation (RoPE, the attention interface, ``o_proj``) is
    inherited from :class:`SparseAttention` unchanged. GQA (k/v narrower than q)
    is handled by the predictor and the per-sub-block offsets.
    """

    inherited_attributes = [
        "qkv_proj",
        "o_proj",
        "config",
        "layer_idx",
        "head_dim",
        "num_key_value_groups",
        "scaling",
        "attention_dropout",
        "is_causal",
    ]

    def _init_extra(self, base: nn.Module, config: Any) -> None:
        self._q_size, self._kv_size = fused_qkv_sizes(base)

    def _build_predictor(self, base: nn.Module, name: str, cfg):
        return _create_fused_attn_predictor(base, cfg.predictor_rank, name, cfg)

    def _sliding_window_fallback(self, config: Any) -> Optional[int]:
        # Phi3 reads sliding_window off config, not the module; mirror that.
        return getattr(config, "sliding_window", None)

    def _split_qkv(self, fused: torch.Tensor):
        q, kv = self._q_size, self._kv_size
        return (
            fused[..., :q],
            fused[..., q : q + kv],
            fused[..., q + kv : q + 2 * kv],
        )

    def _qkv(self, hidden_states: torch.Tensor, masks, sparse: bool):
        if not sparse:
            return (*self._split_qkv(_call_proj(self.qkv_proj, hidden_states)), None)

        q_i, k_i, v_i = self.pred.predict(hidden_states, self.sparsity)
        combined = torch.cat(
            [q_i, self._q_size + k_i, self._q_size + self._kv_size + v_i]
        )
        if masks is None:
            fused = _call_proj(self.qkv_proj, hidden_states, combined)
        else:
            sparse_x, dense_x = self.token_splits(hidden_states, masks)
            fused = self.token_join(
                _call_proj(self.qkv_proj, sparse_x, combined),
                _call_proj(self.qkv_proj, dense_x),
                masks,
            )
        return (*self._split_qkv(fused), v_i)


def _module_attr(base: nn.Module, name: str, default: Any = None) -> Any:
    import sys

    module = sys.modules.get(type(base).__module__)
    return getattr(module, name, default)


class SparseQwen3VLVisionAttention(SparseModule):
    """Sparse Qwen3-VL vision attention with fused ``qkv`` and output ``proj``."""

    inherited_attributes = [
        "qkv",
        "proj",
        "config",
        "dim",
        "num_heads",
        "head_dim",
        "num_key_value_groups",
        "scaling",
        "attention_dropout",
        "is_causal",
    ]

    def __init__(
        self, base: nn.Module, *, name: str, idx: int, sparsity: float = 0, cfg
    ) -> None:
        SparseModule.__init__(self, base)
        self.sparsity = sparsity
        self._q_size, self._kv_size = fused_qkv_sizes(base)
        self._eager_attention = _resolve_eager_attention(base)
        self._apply_rope = _module_attr(base, "apply_rotary_pos_emb_vision")
        self._is_flash_requested = _module_attr(
            base, "is_flash_attention_requested", lambda _config: False
        )
        if self.sparsity > 0:
            self.pred = _create_fused_attn_predictor(
                base, cfg.predictor_rank, name, cfg
            )

    def _split_qkv(
        self, fused: torch.Tensor, lengths: tuple[int, int, int] | None = None
    ):
        if fused.shape[-1] == self._q_size + 2 * self._kv_size:
            q, kv = self._q_size, self._kv_size
            return (
                fused[..., :q],
                fused[..., q : q + kv],
                fused[..., q + kv : q + 2 * kv],
            )
        if lengths is None:
            raise ValueError(
                "Sparse qkv output is not full-width and no split lengths were provided"
            )
        q, k, v = lengths
        return fused[..., :q], fused[..., q : q + k], fused[..., q + k : q + k + v]

    def _qkv(self, hidden_states: torch.Tensor, sparse: bool):
        if not sparse:
            return (*self._split_qkv(_call_proj(self.qkv, hidden_states)), None)
        q_i, k_i, v_i = self.pred.predict(hidden_states, self.sparsity)
        combined = torch.cat(
            [q_i, self._q_size + k_i, self._q_size + self._kv_size + v_i]
        )
        fused = _call_proj(self.qkv, hidden_states, combined)
        return (*self._split_qkv(fused, (len(q_i), len(k_i), len(v_i))), v_i)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del rotary_pos_emb, masks
        seq_length = hidden_states.shape[0]
        sparse = self.enabled and self.sparsity > 0
        query_states, key_states, value_states, v_i = self._qkv(hidden_states, sparse)
        query_states = query_states.reshape(seq_length, self.num_heads, -1)
        key_states = key_states.reshape(seq_length, self.num_heads, -1)
        value_states = value_states.reshape(seq_length, self.num_heads, -1)

        if position_embeddings is None:
            raise ValueError(
                "Sparse Qwen3-VL vision attention requires position_embeddings"
            )
        cos, sin = position_embeddings
        query_states, key_states = self._apply_rope(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, self._eager_attention
        )
        if self._is_flash_requested(self.config):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits, strict=False)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return _call_proj(self.proj, attn_output, v_i if sparse else None)


class SparseNoRoPEAttention(SparseAttention):
    """Sparse ViT attention with no rotary embedding.

    Covers SigLIP/Gemma3 and InternVL vision attention layouts. Their q/k/v
    projections are standard, but the output projection is named ``out_proj`` or
    ``projection_layer`` and q/k normalization, when present, happens on the flat
    hidden dimension before the head reshape.
    """

    inherited_attributes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "config",
        "head_dim",
        "is_causal",
    ]

    def _init_extra(self, base: nn.Module, config: Any) -> None:
        out_name = attention_output_projection_name(base)
        if out_name is None:
            raise ValueError(f"{type(base).__name__} has no output projection")
        self._output_proj_name = out_name
        setattr(self, out_name, getattr(base, out_name))
        self.num_heads = getattr(base, "num_heads", config.num_attention_heads)
        self.num_key_value_groups = 1
        self.scaling = getattr(
            base, "scaling", getattr(base, "scale", self.head_dim**-0.5)
        )
        self.attention_dropout = getattr(
            base, "attention_dropout", getattr(base, "dropout", 0.0)
        )
        self.projection_dropout = getattr(base, "projection_dropout", None)

    def _output_proj(self) -> nn.Module:
        return getattr(self, self._output_proj_name)

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
        del position_embeddings, past_key_values, cache_position
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        sparse = self.enabled and self.sparsity > 0
        masks = masks if _mask_matches(hidden_states, masks) else None

        Q, K, V, v_i = self._qkv(hidden_states, masks, sparse)
        if self.q_norm is not None:
            Q = self.q_norm(Q)
        if self.k_norm is not None:
            K = self.k_norm(K)
        Q = Q.view(hidden_shape).transpose(1, 2)
        K = K.view(hidden_shape).transpose(1, 2)
        V = V.view(hidden_shape).transpose(1, 2)

        attention_fn = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, self._eager_attention
        )
        attn_output, attn_weights = attention_fn(
            self,
            Q,
            K,
            V,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = (
            self._proj_o(attn_output, masks, v_i)
            if sparse
            else _call_proj(self._output_proj(), attn_output)
        )
        if self.projection_dropout is not None:
            attn_output = self.projection_dropout(attn_output)
        return attn_output, attn_weights


class SparseGemma4VisionAttention(SparseAttention):
    """Sparse Gemma4 vision attention with multidimensional RoPE and v-norm."""

    inherited_attributes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "config",
        "layer_idx",
        "head_dim",
        "num_key_value_groups",
        "scaling",
        "attention_dropout",
        "is_causal",
        "v_norm",
    ]

    def _init_extra(self, base: nn.Module, config: Any) -> None:
        self._apply_multidimensional_rope = _module_attr(
            base, "apply_multidimensional_rope"
        )
        if self._apply_multidimensional_rope is None:
            raise ValueError("Gemma4 vision attention rope helper is unavailable")

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
        del past_key_values, cache_position
        position_ids = kwargs.pop("position_ids", None)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        sparse = self.enabled and self.sparsity > 0
        masks = masks if _mask_matches(hidden_states, masks) else None

        Q, K, V, v_i = self._qkv(hidden_states, masks, sparse)
        if position_embeddings is None:
            raise ValueError(
                "Sparse Gemma4 vision attention requires position_embeddings"
            )
        cos, sin = position_embeddings
        Q = self.q_norm(Q.view(hidden_shape))
        Q = self._apply_multidimensional_rope(Q, cos, sin, position_ids).transpose(1, 2)
        K = self.k_norm(K.view(hidden_shape))
        K = self._apply_multidimensional_rope(K, cos, sin, position_ids).transpose(1, 2)
        V = self.v_norm(V.view(hidden_shape)).transpose(1, 2)

        attention_fn = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, self._eager_attention
        )
        attn_output, attn_weights = attention_fn(
            self,
            Q,
            K,
            V,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = (
            self._proj_o(attn_output, masks, v_i)
            if sparse
            else _call_proj(self._output_proj(), attn_output)
        )
        return attn_output, attn_weights


class SparseGatedAttention(SparseAttention):
    """Sparse attention for Qwen3-Next / Qwen3.5 *gated* attention.

    ``q_proj`` emits ``[query | gate]`` interleaved per head; the reference
    (``modeling_qwen3_5.py``) views it as ``(.., num_heads, 2*head_dim)``, chunks
    the last dim into ``query`` and ``gate``, runs attention on ``query``, then
    multiplies the attention output by ``sigmoid(gate)`` before ``o_proj``.

    Because the gate is half of ``q_proj``'s output, ``q_proj`` is computed
    **densely** (full channels, all tokens) so the gate stays intact — only
    ``k_proj``/``v_proj`` (and the input-sparse ``o_proj``) are channel/token
    sparsified. The per-head split, ``q_norm``, partial RoPE, sliding window and
    softcap all come from :class:`SparseAttention` unchanged; this class only
    splits the gate off ``q_proj`` and re-applies it via :meth:`_gate_output`.
    With sparsity disabled this is bit-exact to the reference forward.
    """

    _gate: Optional[torch.Tensor] = None
    # Opt-in (set by the plugin when cfg.fused_attn_kernel): fuse q/k-norm + RoPE
    # into one Triton kernel. Off by default (the separate path is bit-exact).
    _use_fused_qk_norm_rope: bool = False

    def _qk_norm_rope(self, Q: torch.Tensor, K: torch.Tensor, position_embeddings):
        # Fuse q/k RMSNorm + RoPE when enabled. Qwen3.5/3.6 RMSNorm scales by
        # (1 + weight) -> unit_offset=True. Restricted to full rotary (the kernel
        # supports partial via cos width, but only full is parity-verified here);
        # partial-rotary (Qwen3.6) and CPU/no-kernel fall back to the eager path.
        if not (
            self._use_fused_qk_norm_rope
            and _fused_rms_norm_rope is not None
            and Q.is_cuda
            and self.q_norm is not None
            and self.k_norm is not None
            and self._rotary_dim is None
        ):
            return super()._qk_norm_rope(Q, K, position_embeddings)
        cos, sin = position_embeddings
        eps = getattr(self.q_norm, "eps", None)
        if eps is None:
            eps = getattr(self.q_norm, "variance_epsilon", 1e-6)
        Q = _fused_rms_norm_rope(
            Q, self.q_norm.weight, cos, sin, eps=eps, unit_offset=True
        ).transpose(1, 2)
        K = _fused_rms_norm_rope(
            K, self.k_norm.weight, cos, sin, eps=eps, unit_offset=True
        ).transpose(1, 2)
        return Q, K

    def _split_query_gate(self, q_full: torch.Tensor):
        *prefix, _ = q_full.shape
        q4 = q_full.reshape(*prefix, -1, self.head_dim * 2)
        query, gate = torch.chunk(q4, 2, dim=-1)
        return query.reshape(*prefix, -1), gate.reshape(*prefix, -1)

    def _qkv(self, hidden_states: torch.Tensor, masks, sparse: bool):
        # q_proj carries the gate, so compute it densely and split; only k/v are
        # channel-sparsified. The gate is stashed for _gate_output (consumed once,
        # after the attention interface, before o_proj).
        query, gate = self._split_query_gate(self.q_proj(hidden_states))
        self._gate = gate
        if not sparse:
            return query, self.k_proj(hidden_states), self.v_proj(hidden_states), None

        _, k_i, v_i = self.pred.predict(hidden_states, self.sparsity)
        if masks is None:
            return (
                query,
                self.k_proj(hidden_states, k_i),
                self.v_proj(hidden_states, v_i),
                v_i,
            )
        sparse_x, dense_x = self.token_splits(hidden_states, masks)
        K = self._sparse_proj(self.k_proj, sparse_x, dense_x, k_i, masks)
        V = self._sparse_proj(self.v_proj, sparse_x, dense_x, v_i, masks)
        return query, K, V, v_i

    def _gate_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        gate = self._gate
        self._gate = None
        return attn_output * torch.sigmoid(gate)


def unsupported_reason(module: nn.Module) -> Optional[str]:
    """Why a target module can't be sparsified safely, or ``None``.

    The sparse MLP path supports SiLU (Llama/Qwen/Mistral) and ``gelu_tanh``
    (Gemma2/Gemma3) gates — both the predictor's channel selection and the gated
    compute follow the model's activation. Any other gated activation is refused
    rather than silently miscomputed. (Gemma2 attention-logit softcapping is
    reproduced in :class:`SparseAttention`, so it is not a blocker.) Fused Phi3
    MLPs (``gate_up_proj``) are checked the same way.
    """
    if (is_swiglu_mlp(module) or is_fused_gate_up_mlp(module)) and gate_kind(
        module
    ) is None:
        act = type(gate_activation(module)).__name__
        return (
            f"{type(module).__name__} uses an unsupported gated activation "
            f"({act}); SparseLoRA's sparse MLP supports only SiLU and gelu_tanh "
            "gates. This MLP cannot be sparsified faithfully."
        )
    return None


def is_qwen3_vl_vision_attention(module: nn.Module) -> bool:
    return type(module).__name__ == "Qwen3VLVisionAttention" and (
        fused_qkv_projection_name(module) == "qkv"
    )


def is_gemma4_vision_attention(module: nn.Module) -> bool:
    return type(module).__name__ == "Gemma4VisionAttention"


def is_no_rope_vision_attention(module: nn.Module) -> bool:
    if not is_standard_attention(module):
        return False
    name = type(module).__name__
    return name in {"SiglipAttention", "Siglip2Attention", "InternVLVisionAttention"}


def _register_projection_aliases() -> None:
    from ._vendor.sparselora.modules import registry

    registry._OUT_PROJS.update({"qkv", "fc1", "linear_fc1"})
    registry._IN_PROJS.update(
        {"proj", "out_proj", "projection_layer", "fc2", "linear_fc2"}
    )


def register_arch_wiring(model: nn.Module) -> dict[str, str]:
    """Auto-register sparse wiring for the loaded model's MLP/attention classes.

    Introspects every module: SwiGLU MLPs map to :class:`SparseSwiGLUMLP` and
    standard attention to :class:`SparseAttention`; the Phi3-style fused
    ``gate_up_proj`` / ``qkv_proj`` blocks map to :class:`SparseFusedGateUpMLP` /
    :class:`SparseFusedQKVAttention`. ``nn.Linear`` is (re)mapped to the
    bias-aware :class:`SparseLinearBias` (a safe superset of the vendored
    linear). Classes already registered — Llama, or anything a caller registered
    explicitly — are left untouched. Idempotent.

    Returns a ``{class_name: role}`` map of what was newly registered, for
    logging.
    """
    register_sparse_module(nn.Linear, SparseLinearBias)
    _register_projection_aliases()

    registered: dict[str, str] = {}
    mapping = get_module_mapping()
    for module in model.modules():
        cls = type(module)
        if cls in mapping:
            continue
        if is_qwen3_vl_vision_attention(module):
            register_sparse_module(cls, SparseQwen3VLVisionAttention)
            mapping[cls] = SparseQwen3VLVisionAttention
            registered[cls.__name__] = "vision_attention"
        elif is_gemma4_vision_attention(module):
            register_sparse_module(cls, SparseGemma4VisionAttention)
            mapping[cls] = SparseGemma4VisionAttention
            registered[cls.__name__] = "vision_attention"
        elif is_no_rope_vision_attention(module):
            register_sparse_module(cls, SparseNoRoPEAttention)
            mapping[cls] = SparseNoRoPEAttention
            registered[cls.__name__] = "vision_attention"
        elif is_swiglu_mlp(module):
            register_sparse_module(cls, SparseSwiGLUMLP)
            mapping[cls] = SparseSwiGLUMLP
            registered[cls.__name__] = "mlp"
        elif is_non_gated_mlp(module):
            register_sparse_module(cls, SparseNonGatedMLP)
            mapping[cls] = SparseNonGatedMLP
            registered[cls.__name__] = "mlp"
        elif is_fused_gate_up_mlp(module):
            register_sparse_module(cls, SparseFusedGateUpMLP)
            mapping[cls] = SparseFusedGateUpMLP
            registered[cls.__name__] = "mlp"
        elif is_gated_attention(module):
            register_sparse_module(cls, SparseGatedAttention)
            mapping[cls] = SparseGatedAttention
            registered[cls.__name__] = "attention"
        elif is_standard_attention(module):
            register_sparse_module(cls, SparseAttention)
            mapping[cls] = SparseAttention
            registered[cls.__name__] = "attention"
        elif is_fused_qkv_attention(module):
            register_sparse_module(cls, SparseFusedQKVAttention)
            mapping[cls] = SparseFusedQKVAttention
            registered[cls.__name__] = "attention"
    return registered
