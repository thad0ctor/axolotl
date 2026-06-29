"""GPU tests for applying SparseLoRA (predictors use liger/Triton -> CUDA only)."""

import tempfile

import pytest
import torch

from axolotl.integrations.sparselora._vendor.sparselora import (
    SparseLoRAConfig,
    apply_sparselora,
)
from axolotl.integrations.sparselora._vendor.sparselora.modules import SparseLinear
from axolotl.integrations.sparselora.calibration import discover_target_modules
from axolotl.integrations.sparselora.factors import compute_factor_tensors, save_factors

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SparseLoRA predictors require CUDA (liger/Triton)",
)


def test_dense_apply_is_logit_exact(tiny_lora_model):
    model = tiny_lora_model.to("cuda").eval()
    ids = torch.randint(0, 128, (2, 16), device="cuda")
    labels = ids.clone()
    labels[:, :8] = -100
    with torch.no_grad():
        ref = model(input_ids=ids, labels=labels).logits.clone()

    targets = discover_target_modules(model)
    layer_sparsity = {t: 0.0 for t in targets}  # all dense
    d = tempfile.mkdtemp()
    save_factors({}, d)  # dense schedule needs no factors
    apply_sparselora(
        model,
        SparseLoRAConfig(layer_sparsity=layer_sparsity, predictor_rank=8, path=d),
    )

    assert sum(isinstance(m, SparseLinear) for m in model.modules()) > 0
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels).logits
    assert (out - ref).abs().max().item() < 1e-4


def test_sparse_forward_backward_runs(tiny_lora_model):
    model = tiny_lora_model.to("cuda").to(torch.bfloat16)
    model.train()
    targets = discover_target_modules(model)
    factors = compute_factor_tensors(model, targets, rank=8)
    d = tempfile.mkdtemp()
    save_factors(factors, d)
    apply_sparselora(
        model,
        SparseLoRAConfig(
            layer_sparsity={t: 0.5 for t in targets}, predictor_rank=8, path=d
        ),
    )

    ids = torch.randint(0, 128, (2, 24), device="cuda")
    labels = ids.clone()
    labels[:, :10] = -100
    out = model(input_ids=ids, labels=labels)
    assert torch.isfinite(out.loss).item()
    out.loss.backward()
    grad_norm = sum(
        p.grad.float().norm().item()
        for n, p in model.named_parameters()
        if p.grad is not None and "lora_" in n
    )
    assert grad_norm > 0
