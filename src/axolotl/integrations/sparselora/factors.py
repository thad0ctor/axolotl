"""Offline SVD predictor-factor computation for SparseLoRA.

Upstream z-lab/sparselora only *loads* predictor factors from a shipped
``model.safetensors`` (published for Llama-2/3 only). The offline computation
that produces those factors is not in the upstream repo. This module adds it so
predictors can be derived from *any* base model's weights, writing a
``model.safetensors`` in the exact key layout the vendored loader
(``_vendor/sparselora/modules/svd.py``) expects.

For a projection weight ``W`` of shape ``(out, in)``, the training-free rank-r
predictor factors satisfy ``w1 @ w2 ≈ W.T`` so that ``x @ w1 @ w2 ≈ x @ W.T``:

    W = U @ diag(S) @ Vh                      # thin SVD, W (out, in)
    w1 = Vh[:r].T * S[:r]   -> (in,  r)
    w2 = U[:, :r].T         -> (r,  out)

Only the *output-sparse* projections need predictors: ``gate_proj``/``up_proj``
for gated MLP, ``fc1``/``linear_fc1`` for non-gated MLP, and
``q_proj``/``k_proj``/``v_proj`` for attention. ``down_proj``/``fc2``/``o_proj``
are input-sparse and reuse the selected indices.
"""

from __future__ import annotations

import torch
from torch import nn

MLP_PROJECTIONS = ("gate_proj", "up_proj")
ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj")
ATTN_OUTPUT_PROJECTIONS = ("o_proj", "proj", "out_proj", "projection_layer")
NON_GATED_MLP_PROJECTIONS = (("fc1", "fc2"), ("linear_fc1", "linear_fc2"))

# Phi3-style fused projections: one Linear whose output rows concatenate the
# separate projections. ``gate_up_proj`` = [gate | up]; ``qkv_proj`` = [q | k | v].
FUSED_MLP_PROJECTION = "gate_up_proj"
FUSED_ATTN_PROJECTIONS = ("qkv_proj", "qkv")


def _unwrap_linear(module: nn.Module) -> nn.Module:
    base = getattr(module, "base_layer", module)
    return getattr(base, "linear", base)


def _linear_weight(module: nn.Module) -> torch.Tensor | None:
    base = _unwrap_linear(module)
    weight = getattr(base, "weight", None)
    return weight if isinstance(weight, torch.Tensor) and weight.dim() == 2 else None


def _linear_out_features(module: nn.Module) -> int:
    base = _unwrap_linear(module)
    out = getattr(base, "out_features", None)
    if out is not None:
        return int(out)
    weight = _linear_weight(module)
    if weight is None:
        raise ValueError(f"{type(module).__name__} is not a 2-D linear projection")
    return int(weight.shape[0])


def _linear_in_features(module: nn.Module) -> int:
    base = _unwrap_linear(module)
    in_features = getattr(base, "in_features", None)
    if in_features is not None:
        return int(in_features)
    weight = _linear_weight(module)
    if weight is None:
        raise ValueError(f"{type(module).__name__} is not a 2-D linear projection")
    return int(weight.shape[1])


def _has_linear_projection(module: nn.Module, name: str) -> bool:
    return hasattr(module, name) and _linear_weight(getattr(module, name)) is not None


def _first_projection_name(module: nn.Module, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if _has_linear_projection(module, name)), None)


def attention_output_projection_name(module: nn.Module) -> str | None:
    return _first_projection_name(module, ATTN_OUTPUT_PROJECTIONS)


def fused_qkv_projection_name(module: nn.Module) -> str | None:
    return _first_projection_name(module, FUSED_ATTN_PROJECTIONS)


def non_gated_mlp_projection_names(module: nn.Module) -> tuple[str, str] | None:
    for in_name, out_name in NON_GATED_MLP_PROJECTIONS:
        if _has_linear_projection(module, in_name) and _has_linear_projection(
            module, out_name
        ):
            return in_name, out_name
    return None


def linear_weight(module: nn.Module) -> torch.Tensor:
    """Return the logical ``(out, in)`` weight of a (possibly LoRA-wrapped) Linear.

    Dequantizes a bitsandbytes 4-bit (QLoRA) base so callers see the real 2D
    matrix: ``Params4bit`` stores packed data of shape ``(N, 1)``, not the
    weight the SVD predictor and reconstruction sweep need. 8-bit bases are
    rejected up front by the plugin and not handled here.
    """
    base = _unwrap_linear(module)
    weight = base.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:  # bitsandbytes Params4bit
        from bitsandbytes.functional import dequantize_4bit

        return dequantize_4bit(weight.data, quant_state)
    return weight


def is_mlp_module(module: nn.Module) -> bool:
    return all(_has_linear_projection(module, proj) for proj in MLP_PROJECTIONS)


def is_non_gated_mlp_module(module: nn.Module) -> bool:
    return non_gated_mlp_projection_names(module) is not None


def is_attn_module(module: nn.Module) -> bool:
    return all(_has_linear_projection(module, proj) for proj in ATTN_PROJECTIONS) and (
        attention_output_projection_name(module) is not None
    )


def is_fused_mlp_module(module: nn.Module) -> bool:
    """Phi3-style fused MLP: a single ``gate_up_proj`` Linear plus ``down_proj``."""
    return _has_linear_projection(
        module, FUSED_MLP_PROJECTION
    ) and _has_linear_projection(module, "down_proj")


def is_fused_attn_module(module: nn.Module) -> bool:
    """Phi3-style fused attention: a single ``qkv_proj`` Linear plus ``o_proj``."""
    return fused_qkv_projection_name(module) is not None and (
        attention_output_projection_name(module) is not None
    )


def fused_qkv_sizes(module: nn.Module) -> tuple[int, int]:
    """``(q_size, kv_size)`` output split of a fused ``qkv_proj``.

    ``qkv_proj`` out = ``[q (num_heads*head_dim) | k (num_kv*head_dim) | v (...)]``.
    Sizes come from the attention module's config/attrs, mirroring how the model
    itself slices the fused output.
    """
    proj_name = fused_qkv_projection_name(module)
    if proj_name is None:
        raise ValueError(f"{type(module).__name__} has no fused qkv projection")
    qkv_out = _linear_out_features(getattr(module, proj_name))
    config = getattr(module, "config", None)
    head_dim = getattr(module, "head_dim", None)
    if head_dim is None and config is not None:
        hidden = getattr(config, "hidden_size", getattr(module, "dim", None))
        heads = getattr(
            config, "num_attention_heads", getattr(module, "num_heads", None)
        )
        if hidden is not None and heads:
            head_dim = hidden // heads

    num_heads = getattr(module, "num_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(module, "num_key_value_heads", None)
    if num_kv_heads is None and config is not None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = num_heads

    if head_dim is not None and num_heads is not None and num_kv_heads is not None:
        q_size = int(num_heads) * int(head_dim)
        kv_size = int(num_kv_heads) * int(head_dim)
        if q_size + 2 * kv_size == qkv_out:
            return q_size, kv_size

    if qkv_out % 3 != 0:
        raise ValueError(
            f"{type(module).__name__}.{proj_name} out_features={qkv_out} cannot "
            "be split into q/k/v blocks"
        )
    return qkv_out // 3, qkv_out // 3


def fused_intermediate_size(module: nn.Module) -> int:
    """Per-gate intermediate width of a fused ``gate_up_proj`` ( = down_proj in)."""
    return _linear_in_features(module.down_proj)


def mlp_projection_weights(module: nn.Module) -> dict[str, torch.Tensor]:
    """Logical ``{gate_proj, up_proj, down_proj}`` weights, fused or separate.

    For a fused ``gate_up_proj`` the gate occupies the first ``intermediate`` rows
    and up the next ``intermediate`` (matching ``up_states.chunk(2, dim=-1)``).
    """
    if is_fused_mlp_module(module) and not is_mlp_module(module):
        gate_up = linear_weight(getattr(module, FUSED_MLP_PROJECTION))
        i = fused_intermediate_size(module)
        return {
            "gate_proj": gate_up[:i],
            "up_proj": gate_up[i : 2 * i],
            "down_proj": linear_weight(module.down_proj),
        }
    return {
        p: linear_weight(getattr(module, p))
        for p in ("gate_proj", "up_proj", "down_proj")
    }


def non_gated_mlp_projection_weights(module: nn.Module) -> dict[str, torch.Tensor]:
    """Logical ``{fc1|linear_fc1, fc2|linear_fc2}`` weights for GELU-style ViT MLPs."""
    names = non_gated_mlp_projection_names(module)
    if names is None:
        raise ValueError(f"{type(module).__name__} is not a non-gated MLP")
    in_name, out_name = names
    return {
        in_name: linear_weight(getattr(module, in_name)),
        out_name: linear_weight(getattr(module, out_name)),
    }


def attn_projection_weights(module: nn.Module) -> dict[str, torch.Tensor]:
    """Logical ``{q_proj, k_proj, v_proj}`` weights, fused or separate."""
    if is_fused_attn_module(module) and not is_attn_module(module):
        proj_name = fused_qkv_projection_name(module)
        if proj_name is None:
            raise ValueError(f"{type(module).__name__} has no fused qkv projection")
        qkv = linear_weight(getattr(module, proj_name))
        q_size, kv_size = fused_qkv_sizes(module)
        return {
            "q_proj": qkv[:q_size],
            "k_proj": qkv[q_size : q_size + kv_size],
            "v_proj": qkv[q_size + kv_size : q_size + 2 * kv_size],
        }
    return {p: linear_weight(getattr(module, p)) for p in ATTN_PROJECTIONS}


def output_sparse_weights(module: nn.Module) -> dict[str, torch.Tensor]:
    """``{proj: (out, in) weight}`` for the output-sparse projections that need a
    predictor: gate/up or fc1/linear_fc1 (MLP) or q/k/v (attention), fused or
    separate.

    ``down_proj``/``fc2``/``linear_fc2``/``o_proj`` are input-sparse and reuse
    the selected indices, so they get no predictor factors.
    """
    if is_mlp_module(module) or is_fused_mlp_module(module):
        w = mlp_projection_weights(module)
        return {"gate_proj": w["gate_proj"], "up_proj": w["up_proj"]}
    if is_non_gated_mlp_module(module):
        names = non_gated_mlp_projection_names(module)
        if names is None:
            raise ValueError(f"{type(module).__name__} is not a non-gated MLP")
        in_name, _ = names
        return {in_name: linear_weight(getattr(module, in_name))}
    if is_attn_module(module) or is_fused_attn_module(module):
        return attn_projection_weights(module)
    raise ValueError(
        "SparseLoRA target module is neither an MLP (gate_proj/up_proj, "
        "fc1/fc2, linear_fc1/linear_fc2, or fused gate_up_proj) nor an attention "
        f"(q_proj/k_proj/v_proj or fused qkv_proj) module: {type(module).__name__}"
    )


@torch.no_grad()
def svd_factor(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-``r`` factors ``(w1, w2)`` with ``w1 @ w2 ≈ weight.T``."""
    w = weight.detach().to(torch.float32)
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    r = min(rank, s.shape[0])
    w1 = (vh[:r].transpose(0, 1) * s[:r]).contiguous()  # (in, r)
    w2 = u[:, :r].transpose(0, 1).contiguous()  # (r, out)
    return w1.to(weight.dtype), w2.to(weight.dtype)


@torch.no_grad()
def compute_factor_tensors(
    model: nn.Module, layer_names: list[str], rank: int
) -> dict[str, torch.Tensor]:
    """Compute predictor factors for every target module, keyed for the loader.

    Keys are ``{module_name}.{proj}.w1`` / ``.w2`` (e.g.
    ``model.layers.3.mlp.gate_proj.w1``), matching the vendored
    ``create_mlp_predictor`` / ``create_attn_predictor`` lookups. Fused Phi3
    projections are sliced into their logical ``gate_proj``/``up_proj`` or
    ``q_proj``/``k_proj``/``v_proj`` sub-blocks first, so the keys (and the
    predictors that consume them) are identical to the separate-projection case.
    """
    modules = dict(model.named_modules())
    tensors: dict[str, torch.Tensor] = {}
    for name in layer_names:
        if name not in modules:
            raise KeyError(f"SparseLoRA target module not found in model: {name}")
        module = modules[name]
        for proj, weight in output_sparse_weights(module).items():
            w1, w2 = svd_factor(weight, rank)
            tensors[f"{name}.{proj}.w1"] = w1
            tensors[f"{name}.{proj}.w2"] = w2
    return tensors


def save_factors(tensors: dict[str, torch.Tensor], out_dir: str) -> str:
    """Write factors to ``{out_dir}/model.safetensors`` and return that path."""
    import os

    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.safetensors")
    # safetensors requires contiguous CPU tensors.
    save_file({k: v.cpu().contiguous() for k, v in tensors.items()}, path)
    return path
