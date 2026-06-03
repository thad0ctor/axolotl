"""Schema, gate, and (GPU-gated) end-to-end tests for nvfp4_training."""

import os

import pytest

import axolotl.utils.nvfp4_training as nvfp4_mod
from axolotl.utils.schemas.config import (
    AxolotlConfigWCapabilities,
    AxolotlInputConfig,
)

BASE = {
    "base_model": "Qwen/Qwen2.5-0.5B",
    "datasets": [{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
    "learning_rate": 1e-5,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "num_epochs": 1,
}

# Capability gates live on AxolotlConfigWCapabilities; supply the two capability
# blocks so the gate validator runs without touching real hardware.
CAPS = {
    "capabilities": {"bf16": True, "fp8": True, "n_gpu": 1, "compute_capability": "sm_120"},
    "env_capabilities": {"torch_version": "2.8.0"},
}


def _supported(monkeypatch, ok, reason=""):
    monkeypatch.setattr(nvfp4_mod, "nvfp4_supported", lambda: (ok, reason))


def test_schema_accepts_valid_nvfp4_config(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={
            "enabled": True,
            "stochastic_rounding": True,
            "hadamard": True,
            "exclude_modules": ["lm_head", "embed_tokens"],
            "skip_first_n_blocks": 1,
            "skip_last_n_blocks": 1,
        },
    )
    assert cfg.nvfp4_training.enabled is True
    assert cfg.nvfp4_training.skip_first_n_blocks == 1


def test_schema_backend_defaults_native_and_accepts_te(monkeypatch):
    _supported(monkeypatch, True)
    assert (
        AxolotlInputConfig(**BASE, nvfp4_training={"enabled": True}).nvfp4_training.backend
        == "native"
    )
    cfg = AxolotlInputConfig(**BASE, nvfp4_training={"enabled": True, "backend": "te"})
    assert cfg.nvfp4_training.backend == "te"


def test_schema_accepts_fp8_lm_head_eval_knobs(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={
            "enabled": True,
            "fp8_lm_head": True,
            "fp8_lm_head_granularity": "rowwise",
        },
    )
    assert cfg.nvfp4_training.fp8_lm_head is True
    assert cfg.nvfp4_training.fp8_lm_head_granularity == "rowwise"


def test_schema_accepts_qwen3_5_native_switches(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        model_config_type="qwen3_5",
        nvfp4_training={
            "enabled": True,
            "qwen3_5_native_attention": True,
            "qwen3_5_native_attention_backward": True,
            "qwen3_5_native_attention_backward_rtn_grad_packs": True,
            "qwen3_5_native_attention_save_backward_packs": True,
            "qwen3_5_fuse_vproj": True,
            "qwen3_5_native_linear_attn": True,
            "qwen3_5_native_mlp": True,
        },
    )
    assert cfg.nvfp4_training.qwen3_5_native_attention is True
    assert cfg.nvfp4_training.qwen3_5_native_attention_backward is True
    assert cfg.nvfp4_training.qwen3_5_native_attention_backward_rtn_grad_packs is True
    assert cfg.nvfp4_training.qwen3_5_native_attention_save_backward_packs is True


def test_gate_refuses_qwen3_5_switch_on_other_model(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="model_config_type"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen2",
            nvfp4_training={
                "enabled": True,
                "qwen3_5_native_attention": True,
            },
        )


def test_gate_refuses_unsupported_hardware(monkeypatch):
    _supported(monkeypatch, False, "no Blackwell here")
    with pytest.raises(ValueError, match="no Blackwell here"):
        AxolotlConfigWCapabilities(
            **BASE, **CAPS, nvfp4_training={"enabled": True}
        )


def test_gate_allows_lora(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE, **CAPS, adapter="lora", nvfp4_training={"enabled": True}
    )
    assert cfg.adapter == "lora"
    assert cfg.nvfp4_training.quantize_base is False


def test_gate_allows_lora_quantize_base(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        adapter="lora",
        nvfp4_training={"enabled": True, "quantize_base": True},
    )
    assert cfg.nvfp4_training.quantize_base is True


def test_gate_refuses_unsupported_adapter(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="lora, qlora"):
        AxolotlConfigWCapabilities(
            **BASE, **CAPS, adapter="llama-adapter", nvfp4_training={"enabled": True}
        )


def test_gate_refuses_load_in_4bit_with_quantize_base(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="load_in_4bit"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            adapter="lora",
            load_in_4bit=True,
            nvfp4_training={"enabled": True, "quantize_base": True},
        )


def test_gate_refuses_deepspeed(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="DeepSpeed"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            deepspeed="deepspeed_configs/zero3.json",
            nvfp4_training={"enabled": True},
        )


def test_gate_refuses_fp16(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="does not support fp16"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            fp16=True,
            bf16=False,
            nvfp4_training={"enabled": True},
        )


def test_disabled_nvfp4_skips_gate(monkeypatch):
    _supported(monkeypatch, False, "should not be raised")
    cfg = AxolotlConfigWCapabilities(
        **BASE, **CAPS, nvfp4_training={"enabled": False}
    )
    assert cfg.nvfp4_training.enabled is False


def test_gate_refuses_qwen3_5_backward_without_attention(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="qwen3_5_native_attention"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            nvfp4_training={
                "enabled": True,
                "qwen3_5_native_attention_backward": True,
            },
        )


def test_gate_refuses_qwen3_5_rtn_without_backward(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="qwen3_5_native_attention_backward"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "qwen3_5_native_attention": True,
                "qwen3_5_native_attention_backward_rtn_grad_packs": True,
            },
        )


def test_gate_refuses_qwen3_5_saved_packs_without_backward(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="qwen3_5_native_attention_backward"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "qwen3_5_native_attention": True,
                "qwen3_5_native_attention_save_backward_packs": True,
            },
        )


def _tiny_lora_model():
    """A 2-layer toy model wrapped with a PEFT LoRA adapter (CPU-friendly)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.v_proj = nn.Linear(32, 32, bias=False)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

        def forward(self, x):
            for blk in self.layers:
                x = blk.v_proj(blk.q_proj(x))
            return x

    base = Net()
    lcfg = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], lora_alpha=8)
    return get_peft_model(base, lcfg)


def _patch_manager(cfg_dict):
    from axolotl.loaders.patch_manager import PatchManager
    from axolotl.utils.dict import DictDefault

    return PatchManager(DictDefault(cfg_dict), model_config=DictDefault({}))


def test_apply_qwen3_5_native_attention_forwards_saved_pack_flag(monkeypatch):
    _supported(monkeypatch, True)
    from axolotl.monkeypatch.attention import nvfp4_flash_attn

    captured = {}

    def fake_patch(_model, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(
        nvfp4_flash_attn, "patch_qwen3_5_nvfp4_attention", fake_patch
    )
    pm = _patch_manager(
        {
            "model_config_type": "qwen3_5",
            "nvfp4_training": {
                "enabled": True,
                "stochastic_rounding": True,
                "qwen3_5_native_attention": True,
                "qwen3_5_native_attention_backward": True,
                "qwen3_5_native_attention_save_backward_packs": True,
            },
        }
    )

    pm._apply_qwen3_5_native_nvfp4_patches(object())

    assert captured["save_backward_packs"] is True
    assert captured["train_backward"] is True


def test_apply_selects_lora_compute_mode(monkeypatch):
    """adapter=lora default -> base_layer becomes an FP4 compute base (the
    pre-quantized FP4-compute base, the recommended default)."""
    _supported(monkeypatch, True)
    from peft.tuners.lora import Linear as LoraLinear

    from axolotl.utils.nvfp4_training import (
        NVFP4ComputeBaseLinear,
        NVFP4FastComputeBaseLinear,
        NVFP4FrozenBaseLinear,
        NVFP4FastFrozenBaseLinear,
    )

    model = _tiny_lora_model()
    pm = _patch_manager(
        {"adapter": "lora", "nvfp4_training": {"enabled": True}}
    )
    pm._apply_nvfp4_training(model)

    bases = [
        m.base_layer for m in model.modules() if isinstance(m, LoraLinear)
    ]
    assert bases and all(
        isinstance(b, (NVFP4ComputeBaseLinear, NVFP4FastComputeBaseLinear))
        for b in bases
    )
    assert not any(
        isinstance(b, (NVFP4FrozenBaseLinear, NVFP4FastFrozenBaseLinear))
        for b in bases
    )


def test_apply_selects_hp_mode_when_requested(monkeypatch):
    """base_mode: hp -> NVFP4Linear (HP frozen base, re-quantized each step)."""
    _supported(monkeypatch, True)
    from peft.tuners.lora import Linear as LoraLinear

    from axolotl.utils.nvfp4_training import NVFP4Linear

    model = _tiny_lora_model()
    pm = _patch_manager(
        {"adapter": "lora", "nvfp4_training": {"enabled": True, "base_mode": "hp"}}
    )
    pm._apply_nvfp4_training(model)
    bases = [m.base_layer for m in model.modules() if isinstance(m, LoraLinear)]
    assert bases and all(isinstance(b, NVFP4Linear) for b in bases)
    assert all(not b.weight.requires_grad for b in bases)


def test_apply_selects_fft_mode_when_no_adapter(monkeypatch):
    """No adapter -> raw nn.Linear swapped to NVFP4Linear (full fine-tune)."""
    _supported(monkeypatch, True)
    import torch
    from torch import nn

    from axolotl.utils.nvfp4_training import NVFP4Linear

    model = nn.Sequential(nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False))
    pm = _patch_manager({"nvfp4_training": {"enabled": True}})
    pm._apply_nvfp4_training(model)
    assert any(isinstance(m, NVFP4Linear) for m in model.modules())


def test_apply_honors_block_exclusions(monkeypatch):
    """skip_last_n_blocks keeps the tail block's base_layer un-swapped."""
    _supported(monkeypatch, True)
    from peft.tuners.lora import Linear as LoraLinear

    from axolotl.utils.nvfp4_training import (
        NVFP4ComputeBaseLinear,
        NVFP4FastComputeBaseLinear,
    )

    model = _tiny_lora_model()
    pm = _patch_manager(
        {
            "adapter": "lora",
            "nvfp4_training": {"enabled": True, "skip_last_n_blocks": 1},
        }
    )
    pm._apply_nvfp4_training(model)

    swapped = {
        name: isinstance(
            m.base_layer, (NVFP4ComputeBaseLinear, NVFP4FastComputeBaseLinear)
        )
        for name, m in model.named_modules()
        if isinstance(m, LoraLinear)
    }
    # layer 0 swapped, layer 1 (last block) excluded
    assert any("layers.0." in n and v for n, v in swapped.items())
    assert all(not v for n, v in swapped.items() if "layers.1." in n)


_ok, _ = nvfp4_mod.nvfp4_supported()


@pytest.mark.skipif(not _ok, reason="requires Blackwell FP4 tensor cores")
def test_e2e_swap_and_train_step(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    from axolotl.utils.nvfp4_training import (
        NVFP4Linear,
        NVFP4Recipe,
        convert_to_nvfp4_training,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "axolotl-ai-co/tiny-qwen2-129m", torch_dtype=torch.bfloat16
    ).cuda()
    count = convert_to_nvfp4_training(model, NVFP4Recipe())
    assert count > 0
    assert any(isinstance(m, NVFP4Linear) for m in model.modules())

    ids = torch.randint(0, model.config.vocab_size, (1, 64), device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss).item()


@pytest.mark.skipif(not _ok, reason="requires Blackwell FP4 tensor cores")
@pytest.mark.parametrize("quantize_base", [False, True])
def test_e2e_lora_swap_and_train_step(quantize_base):
    """Both adapter modes: right NVFP4 class appears under PEFT and a step is finite."""
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import Linear as LoraLinear
    from transformers import AutoModelForCausalLM

    from axolotl.utils.nvfp4_training import (
        NVFP4FrozenBaseLinear,
        NVFP4FastFrozenBaseLinear,
        NVFP4FastComputeBaseLinear,
        NVFP4Linear,
        NVFP4Recipe,
        convert_lora_base_to_nvfp4,
    )

    base = AutoModelForCausalLM.from_pretrained(
        "axolotl-ai-co/tiny-qwen2-129m", torch_dtype=torch.bfloat16
    ).cuda()
    model = get_peft_model(
        base, LoraConfig(r=8, target_modules=["q_proj", "v_proj"], lora_alpha=16)
    )
    count = convert_lora_base_to_nvfp4(
        model, NVFP4Recipe(), quantized_storage=quantize_base
    )
    assert count > 0

    expected = (
        (NVFP4FrozenBaseLinear, NVFP4FastFrozenBaseLinear)
        if quantize_base
        else (NVFP4Linear, NVFP4FastComputeBaseLinear)
    )
    bases = [m.base_layer for m in model.modules() if isinstance(m, LoraLinear)]
    assert any(isinstance(b, expected) for b in bases)

    ids = torch.randint(0, base.config.vocab_size, (1, 64), device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss).item()
