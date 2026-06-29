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
    allocate_schedule,
    discover_target_modules,
    run_sensitivity,
)
from axolotl.integrations.sparselora.factors import (
    compute_factor_tensors,
    svd_factor,
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

    def test_allocate_respects_budget_and_targets(self):
        # Two layers: one low-error (sparsifiable), one high-error (stays dense).
        errors = {
            "a": {s: 0.001 for s in SPARSITY_GRID},
            "b": {s: 0.9 for s in SPARSITY_GRID},
        }
        sched = allocate_schedule(errors, target_sparsity=0.4, loss_budget=0.01)
        assert "b" not in sched  # over budget at every sparsity -> dense
        assert sched.get("a", 0.0) > 0.0

    def test_allocate_empty_when_all_over_budget(self):
        errors = {"a": {s: 0.5 for s in SPARSITY_GRID}}
        assert allocate_schedule(errors, 0.9, loss_budget=0.01) == {}


class TestArgsValidation:
    def test_defaults(self):
        s = SparseLoRASettings()
        assert s.enabled is True
        assert s.calibration.method == CalibrationMethod.FAITHFUL
        assert 0 < s.target_sparsity < 1

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
