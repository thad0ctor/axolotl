"""Unit tests for translating Axolotl config into MegaTrain config."""

import pytest
import torch

from axolotl.integrations.megatrain._vendor.infinity.config.training import (
    CPUMasterConfig,
)
from axolotl.integrations.megatrain.trainer import build_megatrain_config
from axolotl.utils.dict import DictDefault


def _config(**overrides):
    config = DictDefault(
        base_model="HuggingFaceTB/SmolLM2-135M",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        micro_batch_size=2,
        gradient_accumulation_steps=4,
        sequence_len=512,
        megatrain_checkpoint_interval=3,
        megatrain_num_grad_slabs=8,
        learning_rate=2e-4,
        weight_decay=0.1,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        max_grad_norm=0.5,
        max_steps=20,
        seed=123,
    )
    config.update(overrides)
    return config


def test_build_megatrain_config_maps_core_fields(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    config = build_megatrain_config(_config())

    assert isinstance(config, CPUMasterConfig)
    assert config.model_name == "HuggingFaceTB/SmolLM2-135M"
    assert config.dtype is torch.bfloat16
    assert config.attn_implementation == "sdpa"
    assert config.batch_size == 2
    assert config.max_seq_len == 512
    assert config.checkpoint_interval == 3
    assert config.num_grad_slabs == 8
    assert config.dataset_path == "__axolotl__"
    assert config.device == 2
    assert config.devices == [2]
    assert config.world_size == 1


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"attn_implementation": "flash_attention_2"}, "flash_attention_2"),
        ({"attn_implementation": "sdpa"}, "sdpa"),
        ({"attn_implementation": None, "flash_attention": True}, "flash_attention_2"),
        ({"attn_implementation": None, "sdp_attention": True}, "sdpa"),
        ({"attn_implementation": None}, "eager"),
    ],
)
def test_build_megatrain_config_maps_attention(monkeypatch, overrides, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    config = build_megatrain_config(_config(**overrides))

    assert config.attn_implementation == expected
