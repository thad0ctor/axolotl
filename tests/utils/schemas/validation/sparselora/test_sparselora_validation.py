"""Validation tests for the SparseLoRA integration (plugin args merge)."""

import pytest

from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

PLUGIN = "axolotl.integrations.sparselora.SparseLoRAPlugin"


def _cfg(min_base_cfg, **sparselora):
    return min_base_cfg | DictDefault(
        {
            "adapter": "lora",
            "plugins": [PLUGIN],
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "sample_packing": False,
            "sparselora": sparselora or {"target_sparsity": 0.9},
        }
    )


def test_sparselora_block_round_trips(min_base_cfg):
    cfg = _cfg(min_base_cfg, target_sparsity=0.85, calibration={"method": "proxy"})
    prepare_plugins(cfg)
    validated = validate_config(cfg)

    assert validated.sparselora.enabled is True
    assert validated.sparselora.target_sparsity == 0.85
    assert validated.sparselora.calibration.method.value == "proxy"


def test_method_none_requires_layer_sparsity(min_base_cfg):
    cfg = _cfg(min_base_cfg, calibration={"method": "none"})
    prepare_plugins(cfg)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_method_none_with_layer_sparsity_ok(min_base_cfg):
    cfg = _cfg(
        min_base_cfg,
        calibration={"method": "none"},
        layer_sparsity={"model.layers.0.mlp": 0.5},
    )
    prepare_plugins(cfg)
    validated = validate_config(cfg)
    assert validated.sparselora.layer_sparsity["model.layers.0.mlp"] == 0.5


def test_unknown_sparselora_key_rejected(min_base_cfg):
    # A typo in the `sparselora` block must error, not be silently ignored.
    cfg = _cfg(min_base_cfg, target_sparsity=0.9, target_sparsiti=0.9)
    prepare_plugins(cfg)
    with pytest.raises(ValueError, match=r"target_sparsiti"):
        validate_config(cfg)


def test_unknown_calibration_key_rejected(min_base_cfg):
    cfg = _cfg(min_base_cfg, calibration={"method": "proxy", "num_sample": 64})
    prepare_plugins(cfg)
    with pytest.raises(ValueError, match=r"num_sample"):
        validate_config(cfg)
