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


def test_mask_binding_does_not_nest_across_steps(tiny_lora_model):
    """The masks partial must replace, not nest, on each forward (no leak)."""
    import functools
    import tempfile

    from axolotl.integrations.sparselora._vendor.sparselora.modules import SparseModule

    model = tiny_lora_model.to("cuda").to(torch.bfloat16).train()
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

    ids = torch.randint(0, 128, (2, 16), device="cuda")
    labels = ids.clone()
    labels[:, :8] = -100
    for _ in range(4):  # multiple steps would nest partials under the old code
        model(input_ids=ids, labels=labels)

    sparse_mods = [m for m in model.modules() if isinstance(m, SparseModule)]
    assert sparse_mods
    for m in sparse_mods:
        assert isinstance(m.forward, functools.partial)
        # The wrapped target is the captured base method, never another partial.
        assert not isinstance(m.forward.func, functools.partial)
        assert m.forward.func is m._sparse_base_forward


def test_sparse_linear_4bit_dense_parity_and_sparse():
    """SparseLinear4bit dense path matches bnb Linear4bit; sparse path runs."""
    bnb = pytest.importorskip("bitsandbytes")
    from axolotl.integrations.sparselora.sparse_linear_4bit import SparseLinear4bit

    torch.manual_seed(0)
    out_f, in_f = 256, 128
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.02
    lin = bnb.nn.Linear4bit(
        in_f, out_f, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
    )
    lin.weight = bnb.nn.Params4bit(w.clone(), requires_grad=False, quant_type="nf4")
    lin = lin.cuda()

    x = torch.randn(4, 10, in_f, dtype=torch.bfloat16, device="cuda")
    ref = lin(x)

    # Dense path (no indices) must match bitsandbytes within bf16 tolerance.
    sparse = SparseLinear4bit(lin, mode="out")
    dense_out = sparse(x)
    assert dense_out.shape == ref.shape
    assert torch.allclose(dense_out, ref, rtol=0.05, atol=0.05)

    # Output-sparse path returns only the selected rows.
    idx = torch.arange(0, out_f, 2, device="cuda")
    sp = sparse(x, sparse_indices=idx)
    assert sp.shape == (4, 10, idx.numel())
    assert torch.isfinite(sp).all()


def test_sparse_linear_4bit_bias_passthrough():
    """A biased 4-bit projection (Qwen2 q/k/v) must apply its bias in every path."""
    bnb = pytest.importorskip("bitsandbytes")
    from axolotl.integrations.sparselora.sparse_linear_4bit import SparseLinear4bit

    torch.manual_seed(0)
    out_f, in_f = 256, 128
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.02
    bias = torch.randn(out_f, dtype=torch.bfloat16)
    lin = bnb.nn.Linear4bit(
        in_f, out_f, bias=True, compute_dtype=torch.bfloat16, quant_type="nf4"
    )
    lin.weight = bnb.nn.Params4bit(w.clone(), requires_grad=False, quant_type="nf4")
    lin.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
    lin = lin.cuda()

    x = torch.randn(4, 10, in_f, dtype=torch.bfloat16, device="cuda")
    ref = lin(x)

    sparse = SparseLinear4bit(lin, mode="out")
    # Dense path must match bnb (bias included).
    dense_out = sparse(x)
    assert torch.allclose(dense_out, ref, rtol=0.05, atol=0.05)

    # Output-sparse path slices the bias to the kept rows.
    idx = torch.arange(0, out_f, 2, device="cuda")
    sp = sparse(x, sparse_indices=idx)
    assert sp.shape == (4, 10, idx.numel())
    weight_dense = sparse._dense_weight(x.dtype)
    expected = torch.nn.functional.linear(x, weight_dense[idx], sparse.bias[idx])
    assert torch.allclose(sp, expected, rtol=0.02, atol=0.02)
