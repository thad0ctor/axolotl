"""Cross-architecture tests: model-agnostic sparse wiring (Qwen2/Qwen3/Mistral).

CPU tests cover target discovery, factor layout, auto-registration, validation,
and bias-aware linear correctness. GPU tests apply SparseLoRA and run a sparse
forward+backward (predictors use liger/Triton -> CUDA only).
"""

import tempfile

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.sparselora.arch_wiring import (
    SparseAttention,
    SparseLinearBias,
    SparseSwiGLUMLP,
    register_arch_wiring,
)
from axolotl.integrations.sparselora.calibration import discover_target_modules
from axolotl.integrations.sparselora.factors import compute_factor_tensors, save_factors
from axolotl.integrations.sparselora.plugin import SparseLoRAPlugin
from axolotl.utils.dict import DictDefault

from .conftest import make_lora_model

ARCHES = ["qwen2", "qwen3", "mistral"]


def _validate_cfg():
    return DictDefault(
        {
            "adapter": "lora",
            "sample_packing": False,
            "fsdp": None,
            "deepspeed": None,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "sparselora": {"predictor_rank": 8},
        }
    )


@pytest.mark.parametrize("arch", ARCHES)
class TestCrossArchCPU:
    def test_discover_finds_mlp_and_attn(self, arch):
        model = make_lora_model(arch)
        targets = discover_target_modules(model)
        assert sum(t.endswith("mlp") for t in targets) == 2
        assert sum(t.endswith("self_attn") for t in targets) == 2

    def test_factor_keys_and_shapes(self, arch):
        model = make_lora_model(arch)
        targets = discover_target_modules(model)
        factors = compute_factor_tensors(model, targets, rank=8)
        mlp = next(t for t in targets if t.endswith("mlp"))
        attn = next(t for t in targets if t.endswith("self_attn"))
        for key in (f"{mlp}.gate_proj.w1", f"{mlp}.up_proj.w2"):
            assert key in factors
        for key in (f"{attn}.q_proj.w1", f"{attn}.k_proj.w2", f"{attn}.v_proj.w1"):
            assert key in factors
        assert factors[f"{mlp}.gate_proj.w1"].shape[1] == 8
        assert factors[f"{mlp}.gate_proj.w2"].shape[0] == 8

    def test_auto_registration_maps_arch_classes(self, arch):
        model = make_lora_model(arch)
        registered = register_arch_wiring(model)
        # exactly one MLP class and one attention class for these single-stack models
        roles = sorted(registered.values())
        assert roles == ["attention", "mlp"]

        from axolotl.integrations.sparselora._vendor.sparselora.modules import (
            get_module_mapping,
        )

        mapping = get_module_mapping()
        modules = dict(model.named_modules())
        mlp = modules[
            next(t for t in discover_target_modules(model) if t.endswith("mlp"))
        ]
        attn = modules[
            next(t for t in discover_target_modules(model) if t.endswith("self_attn"))
        ]
        assert mapping[type(mlp)] is SparseSwiGLUMLP
        assert mapping[type(attn)] is SparseAttention

    def test_validate_accepts_attention_only_lora(self, arch):
        model = make_lora_model(arch)
        SparseLoRAPlugin()._validate(
            _validate_cfg(), model, discover_target_modules(model)
        )


def test_gemma2_softcapping_rejected():
    """Generic attention does not apply Gemma's softcap -> must refuse, not corrupt."""
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(0)
    model = get_peft_model(
        Gemma2ForCausalLM(
            Gemma2Config(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=64,
                attn_logit_softcapping=50.0,
            )
        ),
        LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )
    with pytest.raises(ValueError, match="softcapping"):
        SparseLoRAPlugin()._validate(
            _validate_cfg(), model, discover_target_modules(model)
        )


class TestSparseLinearBias:
    def _base(self, out_f=8, in_f=4, bias=True):
        torch.manual_seed(0)
        lin = torch.nn.Linear(in_f, out_f, bias=bias)
        return lin

    def test_dense_path_applies_bias(self):
        lin = self._base()
        sparse = SparseLinearBias(lin, mode="out")
        x = torch.randn(2, 3, 4)
        assert torch.allclose(sparse(x), F.linear(x, lin.weight, lin.bias), atol=1e-6)

    def test_output_sparse_slices_bias(self):
        lin = self._base(out_f=8, in_f=4)
        sparse = SparseLinearBias(lin, mode="out")
        x = torch.randn(2, 3, 4)
        idx = torch.tensor([0, 2, 5])
        out = sparse(x, sparse_indices=idx)
        ref = F.linear(x, lin.weight[idx], lin.bias[idx])
        assert out.shape == (2, 3, 3)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_none_bias_matches_vendored(self):
        from axolotl.integrations.sparselora._vendor.sparselora.modules.linear import (
            SparseLinear,
        )

        lin = self._base(bias=False)
        x = torch.randn(2, 3, 4)
        idx = torch.tensor([1, 3])
        bias_mod = SparseLinearBias(lin, mode="out")
        plain_mod = SparseLinear(lin, mode="out")
        assert torch.allclose(
            bias_mod(x, sparse_indices=idx), plain_mod(x, sparse_indices=idx), atol=1e-7
        )


# ---------------------------------------------------------------------------
# GPU apply tests
# ---------------------------------------------------------------------------

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SparseLoRA predictors require CUDA (liger/Triton)",
)


@cuda_only
@pytest.mark.parametrize("arch", ARCHES)
def test_sparse_apply_forward_backward(arch):
    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )
    from axolotl.integrations.sparselora._vendor.sparselora.modules import SparseModule

    model = make_lora_model(arch).to("cuda").to(torch.bfloat16).train()
    register_arch_wiring(model)
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

    sparse_mods = [m for m in model.modules() if isinstance(m, SparseModule)]
    assert any(isinstance(m, SparseAttention) for m in sparse_mods), "no sparse attn"
    assert any(isinstance(m, SparseSwiGLUMLP) for m in sparse_mods), "no sparse mlp"

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
