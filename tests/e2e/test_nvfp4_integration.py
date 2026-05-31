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


def test_gate_refuses_unsupported_hardware(monkeypatch):
    _supported(monkeypatch, False, "no Blackwell here")
    with pytest.raises(ValueError, match="no Blackwell here"):
        AxolotlConfigWCapabilities(
            **BASE, **CAPS, nvfp4_training={"enabled": True}
        )


def test_gate_refuses_adapter(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="full-fine-tune only"):
        AxolotlConfigWCapabilities(
            **BASE, **CAPS, adapter="lora", nvfp4_training={"enabled": True}
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


def test_disabled_nvfp4_skips_gate(monkeypatch):
    _supported(monkeypatch, False, "should not be raised")
    cfg = AxolotlConfigWCapabilities(
        **BASE, **CAPS, nvfp4_training={"enabled": False}
    )
    assert cfg.nvfp4_training.enabled is False


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
