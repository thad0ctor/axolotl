"""Cross-architecture tests: model-agnostic sparse wiring (Qwen2/Qwen3/Mistral).

CPU tests cover target discovery, factor layout, auto-registration, validation,
and bias-aware linear correctness. GPU tests apply SparseLoRA and run a sparse
forward+backward (predictors use liger/Triton -> CUDA only).
"""

import tempfile

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from axolotl.integrations.sparselora.arch_wiring import (
    SparseAttention,
    SparseFusedGateUpMLP,
    SparseGemma4VisionAttention,
    SparseLinearBias,
    SparseNonGatedMLP,
    SparseNoRoPEAttention,
    SparseQwen3VLVisionAttention,
    SparseSwiGLUMLP,
    register_arch_wiring,
)
from axolotl.integrations.sparselora.calibration import discover_target_modules
from axolotl.integrations.sparselora.factors import compute_factor_tensors, save_factors
from axolotl.integrations.sparselora.plugin import SparseLoRAPlugin
from axolotl.utils.dict import DictDefault

from .conftest import make_lora_model

ARCHES = ["qwen2", "qwen3", "mistral", "gemma2", "gemma3"]


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


def _tiny_gemma2(**overrides):
    from transformers import Gemma2Config, Gemma2ForCausalLM

    kw = dict(
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
    kw.update(overrides)
    return Gemma2ForCausalLM(Gemma2Config(**kw))


def _gemma2_lora(**overrides):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = _tiny_gemma2(**overrides)
    base.config._attn_implementation = "eager"  # Gemma2 softcap path needs eager
    return get_peft_model(
        base,
        LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    ).eval()


def test_gemma2_supported_and_auto_registers():
    """Gemma2 (gelu MLP + softcap attn) is now supported, not refused."""
    pytest.importorskip("transformers")
    model = _gemma2_lora()
    registered = register_arch_wiring(model)
    assert registered.get("Gemma2MLP") == "mlp"
    assert registered.get("Gemma2Attention") == "attention"
    SparseLoRAPlugin()._validate(_validate_cfg(), model, discover_target_modules(model))


def test_gemma2_dense_apply_logit_exact():
    """Full Gemma2 dense-apply (gelu MLP + softcap attn) is logit-exact.

    With sparsity 0 the gated MLP must use gelu_tanh (not SiLU) and attention must
    apply softcap; any drop shows up as a logit divergence from the reference.
    """
    pytest.importorskip("transformers")
    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )

    model = _gemma2_lora(attn_logit_softcapping=30.0)
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        ref = model(input_ids=ids).logits.clone()

    register_arch_wiring(model)
    targets = discover_target_modules(model)
    d = tempfile.mkdtemp()
    save_factors({}, d)  # dense schedule needs no factors
    apply_sparselora(
        model,
        SparseLoRAConfig(
            layer_sparsity={t: 0.0 for t in targets}, predictor_rank=8, path=d
        ),
    )
    assert any(isinstance(m, SparseSwiGLUMLP) for m in model.modules())
    with torch.no_grad():
        out = model(input_ids=ids).logits
    assert torch.allclose(out, ref, atol=1e-4), (
        "gelu MLP or softcap dropped: dense-apply diverged from reference"
    )


def test_gate_kind_detection():
    """gate_kind classifies SiLU vs gelu_tanh numerically (class-agnostic)."""
    from axolotl.integrations.sparselora.arch_wiring import gate_kind, is_silu_gated

    qwen2 = make_lora_model("qwen2")
    mods = dict(qwen2.named_modules())
    mlp = mods[next(t for t in discover_target_modules(qwen2) if t.endswith("mlp"))]
    assert gate_kind(mlp) == "silu" and is_silu_gated(mlp)

    gemma_mlp = next(m for n, m in _tiny_gemma2().named_modules() if n.endswith("mlp"))
    assert gate_kind(gemma_mlp) == "gelu_tanh" and not is_silu_gated(gemma_mlp)


class _VisionAttentionOnly(nn.Module):
    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        block = nn.Module()
        block.attn = attn
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([block])

    def forward(self, x):
        return x


class _VisionMLPOnly(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        block = nn.Module()
        block.mlp = mlp
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([block])

    def forward(self, x):
        return x


def _tiny_qwen3_vl_vision_lora():
    from peft import LoraConfig, get_peft_model

    pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention

    attn = Qwen3VLVisionAttention(
        Qwen3VLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            depth=1,
            out_hidden_size=64,
        )
    )
    return get_peft_model(
        _VisionAttentionOnly(attn),
        LoraConfig(r=4, lora_alpha=8, target_modules=["qkv", "proj"]),
    )


def _tiny_gemma4_vision_lora():
    from peft import LoraConfig, get_peft_model

    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionAttention

    cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
    )
    return get_peft_model(
        _VisionAttentionOnly(Gemma4VisionAttention(cfg, layer_idx=0)),
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[
                "q_proj.linear",
                "k_proj.linear",
                "v_proj.linear",
                "o_proj.linear",
            ],
        ),
    )


def _tiny_siglip_vision_lora():
    from peft import LoraConfig, get_peft_model

    pytest.importorskip("transformers.models.siglip.modeling_siglip")
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipAttention

    attn = SiglipAttention(
        SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
        )
    )
    return get_peft_model(
        _VisionAttentionOnly(attn),
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        ),
    )


def _tiny_internvl_vision_lora():
    from peft import LoraConfig, get_peft_model

    pytest.importorskip("transformers.models.internvl.modeling_internvl")
    from transformers.models.internvl.configuration_internvl import InternVLVisionConfig
    from transformers.models.internvl.modeling_internvl import InternVLVisionAttention

    attn = InternVLVisionAttention(
        InternVLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            use_qk_norm=True,
        )
    )
    return get_peft_model(
        _VisionAttentionOnly(attn),
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "projection_layer"],
        ),
    )


def _tiny_qwen3_vl_vision_mlp_lora():
    from peft import LoraConfig, get_peft_model

    pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionMLP

    mlp = Qwen3VLVisionMLP(
        Qwen3VLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            depth=1,
            out_hidden_size=64,
        )
    )
    return get_peft_model(
        _VisionMLPOnly(mlp),
        LoraConfig(r=4, lora_alpha=8, target_modules=["linear_fc1", "linear_fc2"]),
    )


def _tiny_gemma4_vision_mlp_lora():
    from peft import LoraConfig, get_peft_model

    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionMLP

    cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
    )
    return get_peft_model(
        _VisionMLPOnly(Gemma4VisionMLP(cfg)),
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[
                "gate_proj.linear",
                "up_proj.linear",
                "down_proj.linear",
            ],
        ),
    )


def _tiny_siglip_vision_mlp_lora():
    from peft import LoraConfig, get_peft_model

    pytest.importorskip("transformers.models.siglip.modeling_siglip")
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipMLP

    mlp = SiglipMLP(
        SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
        )
    )
    return get_peft_model(
        _VisionMLPOnly(mlp),
        LoraConfig(r=4, lora_alpha=8, target_modules=["fc1", "fc2"]),
    )


def _tiny_internvl_vision_mlp_lora():
    from peft import LoraConfig, get_peft_model

    pytest.importorskip("transformers.models.internvl.modeling_internvl")
    from transformers.models.internvl.configuration_internvl import InternVLVisionConfig
    from transformers.models.internvl.modeling_internvl import InternVLVisionMLP

    mlp = InternVLVisionMLP(
        InternVLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
        )
    )
    return get_peft_model(
        _VisionMLPOnly(mlp),
        LoraConfig(r=4, lora_alpha=8, target_modules=["fc1", "fc2"]),
    )


@pytest.mark.parametrize(
    "builder,sparse_cls",
    [
        (_tiny_qwen3_vl_vision_lora, SparseQwen3VLVisionAttention),
        (_tiny_gemma4_vision_lora, SparseGemma4VisionAttention),
        (_tiny_siglip_vision_lora, SparseNoRoPEAttention),
        (_tiny_internvl_vision_lora, SparseNoRoPEAttention),
    ],
    ids=["qwen3vl", "gemma4", "siglip", "internvl"],
)
def test_vision_attention_discovery_registration_and_validation(builder, sparse_cls):
    model = builder()
    targets = discover_target_modules(model)
    assert targets == ["base_model.model.visual.blocks.0.attn"]

    factors = compute_factor_tensors(model, targets, rank=4)
    target = targets[0]
    assert f"{target}.q_proj.w1" in factors
    assert f"{target}.k_proj.w2" in factors
    assert f"{target}.v_proj.w1" in factors

    register_arch_wiring(model)
    from axolotl.integrations.sparselora._vendor.sparselora.modules import (
        get_module_mapping,
    )

    attn = dict(model.named_modules())[target]
    assert get_module_mapping()[type(attn)] is sparse_cls
    SparseLoRAPlugin()._validate(_validate_cfg(), model, targets)


@pytest.mark.parametrize(
    "builder,proj",
    [
        (_tiny_qwen3_vl_vision_mlp_lora, "linear_fc1"),
        (_tiny_siglip_vision_mlp_lora, "fc1"),
        (_tiny_internvl_vision_mlp_lora, "fc1"),
    ],
    ids=["qwen3vl", "siglip", "internvl"],
)
def test_vision_non_gated_mlp_discovery_registration_and_validation(builder, proj):
    model = builder()
    targets = discover_target_modules(model)
    assert targets == ["base_model.model.visual.blocks.0.mlp"]

    factors = compute_factor_tensors(model, targets, rank=4)
    target = targets[0]
    assert f"{target}.{proj}.w1" in factors
    assert f"{target}.{proj}.w2" in factors

    register_arch_wiring(model)
    from axolotl.integrations.sparselora._vendor.sparselora.modules import (
        get_module_mapping,
    )

    mlp = dict(model.named_modules())[target]
    assert get_module_mapping()[type(mlp)] is SparseNonGatedMLP
    SparseLoRAPlugin()._validate(_validate_cfg(), model, targets)


def test_gemma4_vision_gated_mlp_discovery_registration_and_validation():
    model = _tiny_gemma4_vision_mlp_lora()
    targets = discover_target_modules(model)
    assert targets == ["base_model.model.visual.blocks.0.mlp"]

    factors = compute_factor_tensors(model, targets, rank=4)
    target = targets[0]
    assert f"{target}.gate_proj.w1" in factors
    assert f"{target}.up_proj.w2" in factors

    register_arch_wiring(model)
    from axolotl.integrations.sparselora._vendor.sparselora.modules import (
        get_module_mapping,
    )

    mlp = dict(model.named_modules())[target]
    assert get_module_mapping()[type(mlp)] is SparseSwiGLUMLP
    SparseLoRAPlugin()._validate(_validate_cfg(), model, targets)


def test_sparse_linear_bias_supports_flat_vision_tokens():
    base = nn.Linear(8, 12, bias=True)
    sparse = SparseLinearBias(base, mode="out_scatter")
    x = torch.randn(5, 8)
    idx = torch.tensor([1, 4, 7])
    out = sparse(x, idx)
    expected = torch.zeros(5, 12)
    expected[:, idx] = F.linear(x, base.weight[idx], base.bias[idx])
    assert torch.allclose(out, expected)


def _apply_sparse_to_target(model, target):
    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )

    module = dict(model.named_modules())[target]
    config = getattr(module, "config", None)
    if config is not None:
        config._attn_implementation = "eager"
    register_arch_wiring(model)
    factors = compute_factor_tensors(model, [target], rank=4)
    with tempfile.TemporaryDirectory() as d:
        save_factors(factors, d)
        apply_sparselora(
            model,
            SparseLoRAConfig(
                layer_sparsity={target: 0.5},
                predictor_rank=4,
                path=d,
            ),
        )
    return dict(model.named_modules())[target]


def test_qwen3vl_vision_attention_sparse_apply_forward_cpu():
    model = _tiny_qwen3_vl_vision_lora().to(torch.bfloat16)
    target = discover_target_modules(model)[0]
    attn = _apply_sparse_to_target(model, target)

    assert isinstance(attn, SparseQwen3VLVisionAttention)
    x = torch.randn(6, 32, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)
    cos = torch.ones(6, 8, dtype=torch.bfloat16)
    sin = torch.zeros(6, 8, dtype=torch.bfloat16)
    out = attn(x, cu_seqlens=cu_seqlens, position_embeddings=(cos, sin))
    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()


def test_gemma4_vision_attention_sparse_apply_forward_cpu():
    model = _tiny_gemma4_vision_lora().to(torch.bfloat16)
    target = discover_target_modules(model)[0]
    attn = _apply_sparse_to_target(model, target)

    assert isinstance(attn, SparseGemma4VisionAttention)
    x = torch.randn(1, 6, 32, dtype=torch.bfloat16)
    cos = torch.ones(1, 6, 8, dtype=torch.bfloat16)
    sin = torch.zeros(1, 6, 8, dtype=torch.bfloat16)
    position_ids = torch.zeros(1, 6, 2, dtype=torch.long)
    out, weights = attn(x, position_embeddings=(cos, sin), position_ids=position_ids)
    assert out.shape == x.shape
    assert weights is None or torch.isfinite(weights.float()).all()
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize(
    "builder",
    [_tiny_siglip_vision_lora, _tiny_internvl_vision_lora],
    ids=["siglip", "internvl"],
)
def test_no_rope_vision_attention_sparse_apply_forward_cpu(builder):
    model = builder().to(torch.bfloat16)
    target = discover_target_modules(model)[0]
    attn = _apply_sparse_to_target(model, target)

    assert isinstance(attn, SparseNoRoPEAttention)
    x = torch.randn(1, 6, 32, dtype=torch.bfloat16)
    out, weights = attn(x)
    assert out.shape == x.shape
    assert weights is None or torch.isfinite(weights.float()).all()
    assert torch.isfinite(out.float()).all()


def test_qwen3vl_vision_mlp_sparse_apply_forward_cpu():
    model = _tiny_qwen3_vl_vision_mlp_lora().to(torch.bfloat16)
    target = discover_target_modules(model)[0]
    mlp = _apply_sparse_to_target(model, target)

    assert isinstance(mlp, SparseNonGatedMLP)
    x = torch.randn(6, 32, dtype=torch.bfloat16)
    out = mlp(x)
    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()


def test_gemma4_vision_gated_mlp_sparse_apply_forward_cpu():
    model = _tiny_gemma4_vision_mlp_lora().to(torch.bfloat16)
    target = discover_target_modules(model)[0]
    mlp = _apply_sparse_to_target(model, target)

    assert isinstance(mlp, SparseSwiGLUMLP)
    x = torch.randn(1, 6, 32, dtype=torch.bfloat16)
    out = mlp(x)
    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize(
    "builder",
    [_tiny_siglip_vision_mlp_lora, _tiny_internvl_vision_mlp_lora],
    ids=["siglip", "internvl"],
)
def test_vision_non_gated_mlp_sparse_apply_forward_cpu(builder):
    model = builder().to(torch.bfloat16)
    target = discover_target_modules(model)[0]
    mlp = _apply_sparse_to_target(model, target)

    assert isinstance(mlp, SparseNonGatedMLP)
    x = torch.randn(1, 6, 32, dtype=torch.bfloat16)
    out = mlp(x)
    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()


def test_gemma2_sparse_reconstruction_and_grads_cpu():
    """Gemma2's gelu path uses no liger, so the full sparse fwd+bwd runs on CPU.

    At 0.5 sparsity the logits must stay finite and a sane reconstruction of the
    dense model (not garbage), and LoRA params must get gradients.
    """
    pytest.importorskip("transformers")
    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )

    # predictors run in bf16 (matching bf16 training), so the model must too.
    model = _gemma2_lora(attn_logit_softcapping=30.0).to(torch.bfloat16)
    ids = torch.randint(0, 128, (2, 16))
    labels = ids.clone()
    labels[:, :8] = -100
    with torch.no_grad():
        ref = model(input_ids=ids).logits.clone()

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
    model.train()
    out = model(input_ids=ids, labels=labels)
    assert torch.isfinite(out.loss).item()
    rel = (out.logits - ref).float().norm() / ref.float().norm()
    assert rel < 1.0, f"sparse Gemma2 reconstruction implausibly far: rel={rel:.3f}"
    out.loss.backward()
    grad = sum(
        p.grad.float().norm().item()
        for n, p in model.named_parameters()
        if p.grad is not None and "lora_" in n
    )
    assert grad > 0


def test_sparseattention_applies_softcap():
    """SparseAttention forwards Gemma2's attn-logit softcap -> dense-apply parity.

    Sparsify only the attention blocks (MLP left as the original Gemma2MLP, which
    we don't support), at sparsity 0.0 so the dense branch runs. If softcap were
    dropped the logits would diverge from the unmodified reference.
    """
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model

    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )
    from axolotl.integrations.sparselora.arch_wiring import register_arch_wiring

    torch.manual_seed(0)
    base = _tiny_gemma2(attn_logit_softcapping=30.0)
    base.config._attn_implementation = "eager"  # softcap path needs eager
    model = get_peft_model(
        base,
        LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    ).eval()

    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        ref = model(input_ids=ids).logits.clone()

    register_arch_wiring(model)
    attn_targets = [
        t for t in discover_target_modules(model) if t.endswith("self_attn")
    ]
    apply_sparselora(
        model,
        SparseLoRAConfig(
            layer_sparsity={t: 0.0 for t in attn_targets}, predictor_rank=8, path="."
        ),
    )
    with torch.no_grad():
        out = model(input_ids=ids).logits
    assert torch.allclose(out, ref, atol=1e-4), (
        "softcap dropped: SparseAttention dense forward diverged from reference"
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


# ---------------------------------------------------------------------------
# Decoupled head_dim predictor fix (Qwen3: q_out != hidden_size)
# ---------------------------------------------------------------------------


def test_decoupled_head_dim_attn_predictor_loads():
    """Qwen3 sets head_dim != hidden/num_heads, so q_proj is wider than hidden.

    The vendored GQAAttentionPredictor sizes q2 as (hidden, rank) and crashes on
    load; the fixed creator sizes it at q_out. Predict runs on CPU (no liger).
    """
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from axolotl.integrations.sparselora.arch_wiring import _create_attn_predictor

    torch.manual_seed(0)
    # hidden 64, 4 q-heads, head_dim 32 -> q_out 128 != hidden 64; GQA (2 kv).
    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                max_position_embeddings=64,
            )
        ),
        LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    ).eval()

    targets = discover_target_modules(model)
    attn = next(t for t in targets if t.endswith("self_attn"))
    modules = dict(model.named_modules())
    # q_out (128) must differ from hidden (64), else this wouldn't exercise the fix.
    assert modules[attn].q_proj.base_layer.weight.shape[0] == 128

    factors = compute_factor_tensors(model, targets, rank=8)
    d = tempfile.mkdtemp()
    save_factors(factors, d)
    cfg = type("C", (), {"path": d, "predictor_rank": 8})()

    pred = _create_attn_predictor(modules[attn], 8, attn, cfg)  # must not raise
    x = torch.randn(1, 6, 64, dtype=torch.bfloat16)  # predictor is bf16
    q_i, k_i, v_i = pred.predict(x, 0.5)
    assert q_i.numel() > 0 and k_i.numel() > 0 and v_i.numel() > 0
    # q indices must address the full (wider) q dim, not hidden.
    assert int(q_i.max()) < 128


# ---------------------------------------------------------------------------
# Registry coverage across more SwiGLU + standard-attention families
# ---------------------------------------------------------------------------


def _tiny(config_cls, model_cls, **extra):
    import transformers as tf

    cfg = getattr(tf, config_cls)(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        **extra,
    )
    return getattr(tf, model_cls)(cfg)


@pytest.mark.parametrize(
    "config_cls,model_cls",
    [
        ("CohereConfig", "CohereForCausalLM"),
        ("StableLmConfig", "StableLmForCausalLM"),  # partial rotary, now supported
    ],
)
def test_registry_covers_silu_swiglu_families(config_cls, model_cls):
    """Cohere / StableLM are SiLU SwiGLU + standard attention -> auto-supported."""
    pytest.importorskip("transformers")
    from axolotl.integrations.sparselora.arch_wiring import (
        is_standard_attention,
        is_swiglu_mlp,
        unsupported_reason,
    )

    model = _tiny(config_cls, model_cls)
    registered = register_arch_wiring(model)
    assert sorted(registered.values()) == ["attention", "mlp"]
    for _, mod in model.named_modules():
        if is_swiglu_mlp(mod) or is_standard_attention(mod):
            assert unsupported_reason(mod) is None


def test_phi3_fused_projections_detected_and_registered():
    """Phi3 fuses q/k/v into qkv_proj and gate/up into gate_up_proj. These are not
    the separate-projection layout, but the fused detectors recognize them and the
    Phi3 sparse classes auto-register."""
    pytest.importorskip("transformers")
    from axolotl.integrations.sparselora._vendor.sparselora.modules import (
        get_module_mapping,
    )
    from axolotl.integrations.sparselora.arch_wiring import (
        SparseFusedGateUpMLP,
        SparseFusedQKVAttention,
        is_fused_gate_up_mlp,
        is_fused_qkv_attention,
        is_standard_attention,
        is_swiglu_mlp,
    )

    from .conftest import _tiny_phi3

    model = _tiny_phi3()
    attn = next(m for n, m in model.named_modules() if n.endswith("self_attn"))
    mlp = next(m for n, m in model.named_modules() if n.endswith("mlp"))
    assert not is_standard_attention(attn) and is_fused_qkv_attention(attn)
    assert not is_swiglu_mlp(mlp) and is_fused_gate_up_mlp(mlp)

    registered = register_arch_wiring(model)
    assert registered.get("Phi3Attention") == "attention"
    assert registered.get("Phi3MLP") == "mlp"
    mapping = get_module_mapping()
    assert mapping[type(attn)] is SparseFusedQKVAttention
    assert mapping[type(mlp)] is SparseFusedGateUpMLP


# ---------------------------------------------------------------------------
# Gated attention (Qwen3-Next / Qwen3.5): q_proj emits [query | gate] per head
# ---------------------------------------------------------------------------


def test_gated_attention_detected_and_registered():
    """Qwen3.5 full-attention layers are gated: q_proj out = 2 * o_proj in. They
    must register as SparseGatedAttention, and the interleaved linear-attention
    (GatedDeltaNet) layers must stay unregistered (no q/k/v/o to sparsify)."""
    pytest.importorskip("transformers")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers lacks Qwen3.5")
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from axolotl.integrations.sparselora._vendor.sparselora.modules import (
        get_module_mapping,
    )
    from axolotl.integrations.sparselora.arch_wiring import (
        SparseGatedAttention,
        is_gated_attention,
        is_standard_attention,
    )

    # 4 layers -> default hybrid pattern places a full_attention layer at idx 3.
    model = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,  # decoupled; gated q_out = 2*4*32 = 256, o_in = 4*32 = 128
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
        )
    ).eval()

    attn = next(m for n, m in model.named_modules() if n.endswith("self_attn"))
    assert is_standard_attention(attn) and is_gated_attention(attn)
    assert attn.q_proj.out_features == 2 * attn.o_proj.in_features

    registered = register_arch_wiring(model)
    assert get_module_mapping()[type(attn)] is SparseGatedAttention
    assert registered.get("Qwen3_5Attention") == "attention"
    # The linear-attention block has no q/k/v/o, so it is never wired as attention.
    assert "Qwen3_5GatedDeltaNet" not in registered


def test_plain_attention_is_not_gated():
    """A standard q_proj (out == o_proj in) must not be misdetected as gated."""
    from axolotl.integrations.sparselora.arch_wiring import is_gated_attention

    model = make_lora_model("qwen3")
    attn = next(m for n, m in model.named_modules() if n.endswith("self_attn"))
    assert not is_gated_attention(attn)


def _gated_attn_stub(head_dim, device="cpu"):
    """Bare SparseGatedAttention with just the attrs _qk_norm_rope reads."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    from axolotl.integrations.sparselora.arch_wiring import SparseGatedAttention

    obj = SparseGatedAttention.__new__(SparseGatedAttention)
    torch.nn.Module.__init__(obj)  # set up _modules so submodules can be assigned
    obj.q_norm = Qwen3_5RMSNorm(head_dim).to(device)
    obj.k_norm = Qwen3_5RMSNorm(head_dim).to(device)
    obj.q_norm.weight.data.normal_(0, 0.1)
    obj.k_norm.weight.data.normal_(0, 0.1)
    obj._rotary_dim = None
    return obj


def test_fused_qk_norm_rope_off_by_default_and_falls_back_off_cuda():
    """The fused kernel is opt-in (default off) and, when enabled off-CUDA, falls
    back to the eager path bit-identically (never crashes)."""
    from axolotl.integrations.sparselora.arch_wiring import SparseGatedAttention

    assert SparseGatedAttention._use_fused_qk_norm_rope is False

    obj = _gated_attn_stub(32)
    b, t, nh, nkv, hd = 1, 8, 4, 2, 32
    q = torch.randn(b, t, nh, hd)
    k = torch.randn(b, t, nkv, hd)
    ang = torch.randn(b, t, hd)
    pos = (ang.cos(), ang.sin())

    obj._use_fused_qk_norm_rope = False
    q_sep, k_sep = obj._qk_norm_rope(q, k, pos)
    obj._use_fused_qk_norm_rope = True  # on, but CPU -> must fall back identically
    q_fb, k_fb = obj._qk_norm_rope(q, k, pos)
    assert torch.equal(q_fb, q_sep) and torch.equal(k_fb, k_sep)


@cuda_only
def test_fused_qk_norm_rope_matches_separate_on_cuda():
    """On CUDA the fused q/k-norm+RoPE kernel matches the eager path within bf16
    fusion tolerance (algorithmically identical; verified ~1e-6 in fp32)."""
    from axolotl.integrations.sparselora.arch_wiring import _fused_rms_norm_rope

    if _fused_rms_norm_rope is None:
        pytest.skip("fused_rms_norm_rope kernel unavailable")

    torch.manual_seed(0)
    hd = 256
    obj = _gated_attn_stub(hd, device="cuda")
    obj.q_norm = obj.q_norm.to(torch.bfloat16)
    obj.k_norm = obj.k_norm.to(torch.bfloat16)
    b, t, nh, nkv = 1, 64, 4, 2
    q = torch.randn(b, t, nh, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, t, nkv, hd, device="cuda", dtype=torch.bfloat16)
    ang = torch.randn(b, t, hd, device="cuda", dtype=torch.bfloat16)
    pos = (ang.cos(), ang.sin())  # bounded, like real RoPE

    obj._use_fused_qk_norm_rope = False
    q_sep, k_sep = obj._qk_norm_rope(q, k, pos)
    obj._use_fused_qk_norm_rope = True
    q_fus, k_fus = obj._qk_norm_rope(q, k, pos)
    assert torch.allclose(q_fus.float(), q_sep.float(), atol=6e-2, rtol=1e-2)
    assert torch.allclose(k_fus.float(), k_sep.float(), atol=6e-2, rtol=1e-2)


def test_batched_moe_experts_not_detected_as_mlp():
    """A batched 3-D expert weight (MoE, e.g. Qwen3_5MoeExperts) exposes
    gate_up_proj/down_proj as Parameters, not Linears. It must NOT be detected as
    a fused SwiGLU MLP (that mis-wiring crashes at apply); only 2-D linear
    projections — including the real shared-expert SwiGLU — are sparsifiable."""
    from axolotl.integrations.sparselora.arch_wiring import (
        is_fused_gate_up_mlp,
        is_swiglu_mlp,
    )

    class BatchedExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.zeros(8, 16, 32))
            self.down_proj = torch.nn.Parameter(torch.zeros(8, 32, 16))

    assert not is_fused_gate_up_mlp(BatchedExperts())

    class RealFusedMLP(torch.nn.Module):  # Phi3-style: real 2-D Linears
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Linear(16, 32)
            self.down_proj = torch.nn.Linear(16, 16)

    assert is_fused_gate_up_mlp(RealFusedMLP())

    class SharedExpert(torch.nn.Module):  # MoE shared expert: real SwiGLU
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(16, 32)
            self.up_proj = torch.nn.Linear(16, 32)
            self.down_proj = torch.nn.Linear(32, 16)

    assert is_swiglu_mlp(SharedExpert())


def test_gated_split_and_gate_math():
    """SparseGatedAttention reproduces the reference per-head [query|gate] split
    and sigmoid gating bit-for-bit (this is the dense-apply parity guarantee)."""
    from axolotl.integrations.sparselora.arch_wiring import SparseGatedAttention

    obj = SparseGatedAttention.__new__(SparseGatedAttention)
    obj.head_dim = 8
    b, t, h, d = 2, 3, 4, 8
    q_full = torch.randn(b, t, h * d * 2)

    query, gate = obj._split_query_gate(q_full)
    # Reference: modeling_qwen3_5 views as (.., n_heads, 2*head_dim) then chunks.
    ref_q, ref_g = torch.chunk(q_full.view(b, t, -1, d * 2), 2, dim=-1)
    assert torch.equal(query, ref_q.reshape(b, t, -1))
    assert torch.equal(gate, ref_g.reshape(b, t, -1))

    obj._gate = gate
    attn_out = torch.randn(b, t, h * d)
    out = obj._gate_output(attn_out)
    assert torch.allclose(out, attn_out * torch.sigmoid(gate))
    assert obj._gate is None  # consumed once


@pytest.mark.parametrize("num_kv_heads", [2, 4], ids=["gqa", "mha"])
def test_phi3_factor_keys_and_validation(num_kv_heads):
    """Fused qkv_proj/gate_up_proj slice into the same q/k/v + gate/up factor keys
    as separate projections; _validate accepts [qkv_proj, o_proj] LoRA."""
    pytest.importorskip("transformers")
    from axolotl.integrations.sparselora.factors import (
        compute_factor_tensors,
        fused_qkv_sizes,
    )

    from .conftest import make_lora_model

    model = make_lora_model("phi3", num_kv_heads=num_kv_heads)
    targets = discover_target_modules(model)
    assert sum(t.endswith("mlp") for t in targets) == 2
    assert sum(t.endswith("self_attn") for t in targets) == 2

    factors = compute_factor_tensors(model, targets, rank=8)
    mlp = next(t for t in targets if t.endswith("mlp"))
    attn = next(t for t in targets if t.endswith("self_attn"))
    for key in (f"{mlp}.gate_proj.w1", f"{mlp}.up_proj.w2"):
        assert key in factors
    for key in (f"{attn}.q_proj.w1", f"{attn}.k_proj.w2", f"{attn}.v_proj.w1"):
        assert key in factors

    modules = dict(model.named_modules())
    q_size, kv_size = fused_qkv_sizes(modules[attn])
    # q/v factor out dims must match the sub-block sizes carved from qkv_proj.
    assert factors[f"{attn}.q_proj.w2"].shape[-1] == q_size
    assert factors[f"{attn}.v_proj.w2"].shape[-1] == kv_size
    # GQA: k/v sub-blocks narrower than q; MHA: equal.
    assert (kv_size < q_size) == (num_kv_heads < 4)

    SparseLoRAPlugin()._validate(_validate_cfg(), model, targets)


@cuda_only
def test_phi3_accepts_fused_mlp_lora_sparse_apply_forward_cuda():
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model

    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )

    from .conftest import _tiny_phi3

    torch.manual_seed(0)
    model = (
        get_peft_model(
            _tiny_phi3(),
            LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
            ),
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    register_arch_wiring(model)
    targets = discover_target_modules(model)
    SparseLoRAPlugin()._validate(_validate_cfg(), model, targets)

    target = next(t for t in targets if t.endswith("mlp"))
    factors = compute_factor_tensors(model, [target], rank=8)
    with tempfile.TemporaryDirectory() as d:
        save_factors(factors, d)
        apply_sparselora(
            model,
            SparseLoRAConfig(
                layer_sparsity={target: 0.5},
                predictor_rank=8,
                path=d,
            ),
        )

        mlp = dict(model.named_modules())[target]
        assert isinstance(mlp, SparseFusedGateUpMLP)
        x = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
        out = mlp(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


@cuda_only
@pytest.mark.parametrize("num_kv_heads", [2, 4], ids=["gqa", "mha"])
def test_phi3_dense_apply_logit_exact(num_kv_heads):
    """Dense-apply (sparsity 0) on Phi3 is logit-exact: the fused qkv_proj and
    gate_up_proj slicing must reproduce the model's own split bit-for-bit."""
    pytest.importorskip("transformers")
    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )
    from axolotl.integrations.sparselora.arch_wiring import (
        SparseFusedGateUpMLP,
        SparseFusedQKVAttention,
    )

    from .conftest import make_lora_model

    model = make_lora_model("phi3", num_kv_heads=num_kv_heads).to("cuda").eval()
    ids = torch.randint(0, 128, (2, 16), device="cuda")
    with torch.no_grad():
        ref = model(input_ids=ids).logits.clone()

    register_arch_wiring(model)
    targets = discover_target_modules(model)
    d = tempfile.mkdtemp()
    save_factors({}, d)  # dense schedule needs no factors
    apply_sparselora(
        model,
        SparseLoRAConfig(
            layer_sparsity={t: 0.0 for t in targets}, predictor_rank=8, path=d
        ),
    )
    assert any(isinstance(m, SparseFusedQKVAttention) for m in model.modules())
    assert any(isinstance(m, SparseFusedGateUpMLP) for m in model.modules())
    with torch.no_grad():
        out = model(input_ids=ids).logits
    assert torch.allclose(out, ref, atol=1e-4), (
        "fused qkv/gate_up slicing diverged from reference"
    )


@cuda_only
@pytest.mark.parametrize("num_kv_heads", [2, 4], ids=["gqa", "mha"])
def test_phi3_sparse_apply_forward_backward(num_kv_heads):
    """Phi3 at 0.5 sparsity: finite loss, a plausible reconstruction of the dense
    model, and non-zero LoRA gradients."""
    pytest.importorskip("transformers")
    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )
    from axolotl.integrations.sparselora.arch_wiring import (
        SparseFusedGateUpMLP,
        SparseFusedQKVAttention,
    )

    from .conftest import make_lora_model

    model = (
        make_lora_model("phi3", num_kv_heads=num_kv_heads)
        .to("cuda")
        .to(torch.bfloat16)
        .train()
    )
    ids = torch.randint(0, 128, (2, 24), device="cuda")
    labels = ids.clone()
    labels[:, :10] = -100
    with torch.no_grad():
        ref = model(input_ids=ids).logits.clone()

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
    assert any(isinstance(m, SparseFusedQKVAttention) for m in model.modules())
    assert any(isinstance(m, SparseFusedGateUpMLP) for m in model.modules())

    out = model(input_ids=ids, labels=labels)
    assert torch.isfinite(out.loss).item()
    rel = (out.logits - ref).float().norm() / ref.float().norm()
    assert rel < 1.0, f"sparse Phi3 reconstruction implausibly far: rel={rel:.3f}"
    out.loss.backward()
    grad_norm = sum(
        p.grad.float().norm().item()
        for n, p in model.named_parameters()
        if p.grad is not None and "lora_" in n
    )
    assert grad_norm > 0


def _stablelm_lora():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = _tiny("StableLmConfig", "StableLmForCausalLM", partial_rotary_factor=0.25)
    return get_peft_model(
        base,
        LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    ).eval()


def test_stablelm_partial_rotary_supported():
    """StableLM uses partial rotary; SparseAttention splits RoPE to the rotary dim
    (rest passes through) instead of refusing. It registers, isn't refused, and
    the sparse attention computes the right rotary_dim."""
    pytest.importorskip("transformers")
    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )
    from axolotl.integrations.sparselora.arch_wiring import unsupported_reason

    model = _stablelm_lora()
    attn = next(m for n, m in model.named_modules() if n.endswith("self_attn"))
    head_dim = attn.config.hidden_size // attn.config.num_attention_heads
    assert attn.config.partial_rotary_factor < 1.0  # genuinely partial-rotary
    assert unsupported_reason(attn) is None  # no longer refused
    register_arch_wiring(model)
    targets = discover_target_modules(model)
    d = tempfile.mkdtemp()
    save_factors({}, d)
    apply_sparselora(  # sparsity 0 -> no predictor/forward, CPU-safe
        model,
        SparseLoRAConfig(
            layer_sparsity={t: 0.0 for t in targets}, predictor_rank=8, path=d
        ),
    )
    sa = next(m for m in model.modules() if isinstance(m, SparseAttention))
    assert sa._rotary_dim == int(0.25 * head_dim)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SiLU MLP predictor needs CUDA"
)
def test_stablelm_partial_rotary_logit_exact():
    """Dense-apply on StableLM reproduces the model's own attention (split RoPE)."""
    pytest.importorskip("transformers")
    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )

    model = _stablelm_lora().to("cuda").to(torch.bfloat16)
    ids = torch.randint(0, 256, (2, 16), device="cuda")
    with torch.no_grad():
        ref = model(input_ids=ids).logits.clone()
    register_arch_wiring(model)
    targets = discover_target_modules(model)
    d = tempfile.mkdtemp()
    save_factors({}, d)
    apply_sparselora(
        model,
        SparseLoRAConfig(
            layer_sparsity={t: 0.0 for t in targets}, predictor_rank=8, path=d
        ),
    )
    with torch.no_grad():
        out = model(input_ids=ids).logits
    assert torch.allclose(out, ref, atol=5e-3), "partial RoPE diverged from reference"


def test_gemma3_text_fully_supported():
    """Gemma3-text (gelu MLP, no softcap) is fully supported: both MLP and
    attention auto-register and neither is refused."""
    pytest.importorskip("transformers")
    from axolotl.integrations.sparselora.arch_wiring import (
        gate_kind,
        is_standard_attention,
        is_swiglu_mlp,
        unsupported_reason,
    )

    model = _tiny("Gemma3TextConfig", "Gemma3ForCausalLM", head_dim=16)
    registered = register_arch_wiring(model)
    assert sorted(registered.values()) == ["attention", "mlp"]
    attn = next(m for n, m in model.named_modules() if is_standard_attention(m))
    mlp = next(m for n, m in model.named_modules() if is_swiglu_mlp(m))
    assert gate_kind(mlp) == "gelu_tanh"
    assert unsupported_reason(attn) is None
    assert unsupported_reason(mlp) is None
