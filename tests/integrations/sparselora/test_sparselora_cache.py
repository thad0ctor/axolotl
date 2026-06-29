"""CPU tests for the calibration cache (key stability + round-trip)."""

import torch

from axolotl.integrations.sparselora import cache as cache_mod
from axolotl.utils.dict import DictDefault


def _cfg(tmp_path, target_sparsity=0.9):
    return DictDefault(
        {
            "base_model": "meta-llama/Llama-3.1-8B",
            "adapter": "lora",
            "sequence_len": 2048,
            "output_dir": str(tmp_path),
            "datasets": [{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
            "sparselora": {
                "target_sparsity": target_sparsity,
                "predictor_rank": 8,
                "layer_sparsity": None,
                "cache_dir": None,
                "share_calibration": False,
                "calibration": {
                    "method": "faithful",
                    "num_samples": 128,
                    "batch_size": 1,
                    "warmup_steps": 50,
                    "loss_budget": 0.01,
                },
            },
        }
    )


def test_key_is_stable_and_input_sensitive(tmp_path):
    a = cache_mod.compute_cache_key(_cfg(tmp_path))
    b = cache_mod.compute_cache_key(_cfg(tmp_path))
    c = cache_mod.compute_cache_key(_cfg(tmp_path, target_sparsity=0.5))
    assert a == b
    assert a != c


def test_default_cache_root_under_output_dir(tmp_path):
    cfg = _cfg(tmp_path)
    root = cache_mod.resolve_cache_root(cfg)
    assert root.endswith("sparselora_calibration")
    assert str(tmp_path) in root


def test_save_then_load_round_trip(tmp_path):
    cfg = _cfg(tmp_path)
    key = cache_mod.compute_cache_key(cfg)
    layer_sparsity = {"model.layers.0.mlp": 0.5, "model.layers.0.self_attn": 0.0}
    factors = {"model.layers.0.mlp.gate_proj.w1": torch.randn(8, 4)}

    assert cache_mod.load_cached(cfg, key) is None  # cold
    cache_mod.save_cached(
        cfg, key, layer_sparsity, {"base_model": cfg.base_model}, factors
    )

    loaded = cache_mod.load_cached(cfg, key)
    assert loaded is not None
    assert loaded["layer_sparsity"] == layer_sparsity
    assert loaded["factors_path"].endswith("model.safetensors")
    assert loaded["meta"]["base_model"] == cfg.base_model
