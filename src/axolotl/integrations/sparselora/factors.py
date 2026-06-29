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
for MLP and ``q_proj``/``k_proj``/``v_proj`` for attention. ``down_proj`` and
``o_proj`` are input-sparse and reuse the selected indices.
"""

from __future__ import annotations

import torch
from torch import nn

MLP_PROJECTIONS = ("gate_proj", "up_proj")
ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj")


def linear_weight(module: nn.Module) -> torch.Tensor:
    """Return the underlying ``(out, in)`` weight of a (possibly LoRA-wrapped) Linear.

    SparseLoRA v1 requires a full-precision base: the vendored ``SparseLinear``
    channel-slices a dense weight and is only registered for ``nn.Linear``, so
    quantized bases (``Linear4bit``/``Linear8bitLt``) are rejected up front by
    the plugin rather than handled here.
    """
    base = getattr(module, "base_layer", module)
    return base.weight


def is_mlp_module(module: nn.Module) -> bool:
    return all(hasattr(module, proj) for proj in MLP_PROJECTIONS)


def is_attn_module(module: nn.Module) -> bool:
    return all(hasattr(module, proj) for proj in ATTN_PROJECTIONS)


def projections_for(module: nn.Module) -> tuple[str, ...]:
    if is_mlp_module(module):
        return MLP_PROJECTIONS
    if is_attn_module(module):
        return ATTN_PROJECTIONS
    raise ValueError(
        "SparseLoRA target module is neither an MLP (gate_proj/up_proj) nor an "
        f"attention (q_proj/k_proj/v_proj) module: {type(module).__name__}"
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
    ``create_mlp_predictor`` / ``create_attn_predictor`` lookups.
    """
    modules = dict(model.named_modules())
    tensors: dict[str, torch.Tensor] = {}
    for name in layer_names:
        if name not in modules:
            raise KeyError(f"SparseLoRA target module not found in model: {name}")
        module = modules[name]
        for proj in projections_for(module):
            weight = linear_weight(getattr(module, proj))
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
