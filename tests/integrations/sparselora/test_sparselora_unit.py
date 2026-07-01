"""CPU unit tests: SVD factors, calibration, allocation, cache, args validation."""

import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from axolotl.integrations.sparselora import (
    CalibrationMethod,
    SparseLoRAArgs,
    SparseLoRASettings,
)
from axolotl.integrations.sparselora.calibration import (
    SPARSITY_GRID,
    allocate_structural,
    discover_target_modules,
    load_preset_schedule,
    refine_with_sensitivity,
    resolve_attn_sparsity,
    run_sensitivity,
)
from axolotl.integrations.sparselora.factors import (
    compute_factor_tensors,
    svd_factor,
)


def _multi_layer_llama(num_hidden_layers: int = 8):
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )


class TestSVDFactors:
    def test_full_rank_reconstruction_is_exact(self):
        torch.manual_seed(0)
        weight = torch.randn(48, 32)  # (out, in); min dim 32
        w1, w2 = svd_factor(weight, rank=32)
        assert w1.shape == (32, 32) and w2.shape == (32, 48)
        err = ((w1 @ w2) - weight.T).norm() / weight.T.norm()
        assert err < 1e-4

    def test_truncation_error_is_monotone_in_rank(self):
        torch.manual_seed(0)
        weight = torch.randn(48, 32)
        prev = float("inf")
        for r in (4, 8, 16, 32):
            w1, w2 = svd_factor(weight, rank=r)
            err = (((w1 @ w2) - weight.T).norm() / weight.T.norm()).item()
            assert err <= prev + 1e-6
            prev = err

    def test_factor_keys_match_loader_layout(self, tiny_lora_model):
        targets = discover_target_modules(tiny_lora_model)
        factors = compute_factor_tensors(tiny_lora_model, targets, rank=8)
        mlp = next(t for t in targets if t.endswith("mlp"))
        attn = next(t for t in targets if t.endswith("self_attn"))
        for key in (f"{mlp}.gate_proj.w1", f"{mlp}.up_proj.w2"):
            assert key in factors
        for key in (f"{attn}.q_proj.w1", f"{attn}.k_proj.w2", f"{attn}.v_proj.w1"):
            assert key in factors
        # w1 is (in, rank); w2 is (rank, out)
        assert factors[f"{mlp}.gate_proj.w1"].shape[1] == 8
        assert factors[f"{mlp}.gate_proj.w2"].shape[0] == 8


class TestCalibration:
    def test_discover_finds_mlp_and_attn(self, tiny_lora_model):
        targets = discover_target_modules(tiny_lora_model)
        assert sum(t.endswith("mlp") for t in targets) == 2
        assert sum(t.endswith("self_attn") for t in targets) == 2

    def test_sensitivity_grows_with_sparsity(self, tiny_lora_model, calib_batches):
        targets = discover_target_modules(tiny_lora_model)
        factors = compute_factor_tensors(tiny_lora_model, targets, rank=8)
        loader = DataLoader(calib_batches, batch_size=None, collate_fn=lambda b: b)
        errors = run_sensitivity(
            tiny_lora_model, targets, factors, loader, torch.device("cpu")
        )
        # Reconstruction error must be non-negative and higher at the high-sparsity
        # end than the low-sparsity end (the property allocation relies on). Strict
        # step-wise monotonicity is not guaranteed through the SiLU nonlinearity.
        for grid in errors.values():
            assert all(grid[s] >= 0.0 for s in SPARSITY_GRID)
            assert grid[SPARSITY_GRID[-1]] >= grid[SPARSITY_GRID[0]] - 1e-4

    def test_resolve_attn_sparsity(self):
        # Default caps attention below the MLP level (z-lab: 0.75 attn vs 0.99 MLP).
        assert resolve_attn_sparsity(0.99, None) == 0.75
        assert resolve_attn_sparsity(0.5, None) == 0.5
        assert resolve_attn_sparsity(0.9, 0.3) == 0.3

    def test_structural_is_nonempty_and_zlab_shaped(self):
        # The bug being fixed: the default schedule was empty (0/N sparse). The
        # structural profile must sparsify a deep band while keeping shallow +
        # final layers dense, MLP more aggressively than attention.
        model = _multi_layer_llama(8)
        targets = discover_target_modules(model)
        sched = allocate_structural(
            model,
            targets,
            target_sparsity=0.9,
            attn_sparsity=0.75,
            dense_prefix=0.1,
            attn_dense_prefix=0.45,
        )
        assert sched, "structural schedule must never be empty for a normal model"

        def idx(name):
            return int(name.split(".layers.")[1].split(".")[0])

        mlp = {idx(n): v for n, v in sched.items() if n.endswith("mlp")}
        attn = {idx(n): v for n, v in sched.items() if n.endswith("self_attn")}
        assert set(mlp.values()) == {0.9} and set(attn.values()) == {0.75}
        # Shallow (layer 0) and final (layer 7) layers stay dense in both groups.
        assert 0 not in mlp and 7 not in mlp
        assert 0 not in attn and 7 not in attn
        # Deep layers are sparsified; attention starts deeper than the MLP.
        assert 6 in mlp and 6 in attn
        assert min(attn) > min(mlp)

    def test_structural_keeps_final_layer_dense(self):
        model = _multi_layer_llama(4)
        targets = discover_target_modules(model)
        sched = allocate_structural(
            model, targets, 0.9, 0.75, dense_prefix=0.0, attn_dense_prefix=0.0
        )
        # Even with no dense prefix, the final layer is never sparsified.
        assert not any(n.endswith(".layers.3.mlp") for n in sched)
        assert not any(n.endswith(".layers.3.self_attn") for n in sched)

    def test_refine_demotes_most_sensitive_band_layer(self):
        model = _multi_layer_llama(8)
        targets = discover_target_modules(model)
        base = allocate_structural(model, targets, 0.9, 0.75, 0.1, 0.45)
        mlp_band = sorted(
            (n for n in base if n.endswith("mlp")),
            key=lambda n: int(n.split(".layers.")[1].split(".")[0]),
        )
        # Give the deepest MLP band layer a huge reconstruction error; refine
        # (demote_frac 0.25 -> one per group) must drop exactly it.
        worst = mlp_band[-1]
        errors = {n: {0.9: 0.1} for n in base}
        errors[worst] = {0.9: 5.0}
        refined = refine_with_sensitivity(model, base, errors, demote_frac=0.25)
        assert worst not in refined
        assert all(n in refined for n in mlp_band[:-1])

    def test_refine_noop_when_frac_zero(self):
        model = _multi_layer_llama(8)
        targets = discover_target_modules(model)
        base = allocate_structural(model, targets, 0.9, 0.75, 0.1, 0.45)
        assert refine_with_sensitivity(model, base, {}, demote_frac=0.0) == base

    def test_preset_loads_local_config_and_maps_keys(self, tmp_path):
        import json

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "predictor_rank": 8,
                    "modes": {
                        "o2": {
                            "layer_sparsity": {
                                "model.layers.0.mlp": 0.0,  # dense entries skipped
                                "model.layers.1.mlp": 0.99,
                                "model.layers.1.self_attn": 0.75,
                            }
                        }
                    },
                }
            )
        )
        targets = [
            "base_model.model.model.layers.0.mlp",
            "base_model.model.model.layers.1.mlp",
            "base_model.model.model.layers.1.self_attn",
        ]
        sched = load_preset_schedule(str(tmp_path), "o2", targets)
        assert sched == {
            "base_model.model.model.layers.1.mlp": 0.99,
            "base_model.model.model.layers.1.self_attn": 0.75,
        }

    def test_preset_all_unmatched_raises(self, tmp_path):
        import json

        (tmp_path / "config.json").write_text(
            json.dumps(
                {"modes": {"o2": {"layer_sparsity": {"model.layers.9.mlp": 0.9}}}}
            )
        )
        with pytest.raises(ValueError, match="matched no sparsifiable"):
            load_preset_schedule(str(tmp_path), "o2", ["model.layers.0.mlp"])


class TestArgsValidation:
    def test_defaults(self):
        s = SparseLoRASettings()
        assert s.enabled is True
        assert s.calibration.method == CalibrationMethod.STRUCTURAL
        assert 0 < s.target_sparsity < 1

    def test_preset_method_requires_preset(self):
        with pytest.raises(ValidationError, match="preset requires"):
            SparseLoRASettings(calibration={"method": "preset"})
        ok = SparseLoRASettings(
            calibration={"method": "preset"},
            preset="z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA",
        )
        assert ok.preset and ok.preset_mode == "o2"

    def test_attn_sparsity_range_checked(self):
        with pytest.raises(ValidationError):
            SparseLoRASettings(attn_sparsity=1.2)
        assert SparseLoRASettings(attn_sparsity=0.5).attn_sparsity == 0.5

    def test_method_none_requires_layer_sparsity(self):
        with pytest.raises(ValidationError):
            SparseLoRASettings(calibration={"method": "none"})
        ok = SparseLoRASettings(
            calibration={"method": "none"},
            layer_sparsity={"model.layers.0.mlp": 0.5},
        )
        assert ok.layer_sparsity["model.layers.0.mlp"] == 0.5

    def test_end_step_must_exceed_start_step(self):
        with pytest.raises(ValidationError):
            SparseLoRASettings(start_step=0.5, end_step=0.2)

    def test_layer_sparsity_range_checked(self):
        with pytest.raises(ValidationError):
            SparseLoRASettings(
                calibration={"method": "none"},
                layer_sparsity={"m": 1.5},
            )

    def test_args_wrapper_exposes_block(self):
        args = SparseLoRAArgs(sparselora={"target_sparsity": 0.8})
        assert args.sparselora.target_sparsity == 0.8
