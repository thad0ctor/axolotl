"""Config args for SparseLoRA contextual-sparsity fine-tuning."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class CalibrationMethod(str, Enum):
    """How the per-layer sparsity schedule is determined before training."""

    FAITHFUL = "faithful"  # short dense warm-up reference, then per-layer sweep
    PROXY = "proxy"  # forward-only block-reconstruction sweep, no warm-up
    NONE = "none"  # use the explicit `layer_sparsity` map verbatim


class CalibrationConfig(BaseModel):
    """Calibration settings. Calibration runs on a slice of the *same* dataset."""

    method: CalibrationMethod = Field(
        default=CalibrationMethod.FAITHFUL,
        description=(
            "Schedule source: `faithful` (default, paper-style sensitivity sweep "
            "after a short dense warm-up), `proxy` (forward-only sweep, no warm-up, "
            "fast), or `none` (use the explicit `layer_sparsity` override)."
        ),
    )
    num_samples: int = Field(
        default=128,
        gt=0,
        description="Number of examples drawn from the configured dataset for calibration.",
    )
    batch_size: int = Field(
        default=1, gt=0, description="Micro-batch size for the calibration passes."
    )
    warmup_steps: int = Field(
        default=50,
        ge=0,
        description="`faithful` only: dense LoRA reference-training steps before the sweep.",
    )
    loss_budget: float = Field(
        default=0.01,
        gt=0,
        description=(
            "Max per-layer relative reconstruction error tolerated when raising a "
            "layer's sparsity during the sweep."
        ),
    )


class SparseLoRASettings(BaseModel):
    """Nested SparseLoRA configuration available under the `sparselora` key."""

    enabled: bool = Field(
        default=True, description="Master switch for the SparseLoRA plugin."
    )
    target_sparsity: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description="Overall base-path sparsity the calibrated schedule aims to reach.",
    )
    predictor_rank: int = Field(
        default=8, gt=0, description="Rank of the training-free SVD sparsity predictor."
    )
    start_step: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Enable sparsity after this fraction of training (<=1) or absolute step "
            "(>1). The first slice stays dense for gradient stability."
        ),
    )
    end_step: float = Field(
        default=1.0,
        ge=0.0,
        description="Disable sparsity after this fraction of training (<=1) or absolute step (>1).",
    )
    layer_sparsity: dict[str, float] | None = Field(
        default=None,
        description=(
            "Explicit per-module sparsity map keyed by full module path "
            "(e.g. `model.layers.3.mlp`). Required when calibration.method is `none`; "
            "otherwise ignored (the schedule is calibrated)."
        ),
    )
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    cache_dir: str | None = Field(
        default=None,
        description=(
            "Directory for cached calibration artifacts (schedule + SVD factors). "
            "Defaults to `{output_dir}/sparselora_calibration`. The resolved path is "
            "logged at INFO on every run."
        ),
    )
    share_calibration: bool = Field(
        default=False,
        description=(
            "Opt-in: package the calibrated schedule (model id + sparsity params "
            "only, never dataset content) for optional upstream submission. Off by "
            "default; no data leaves the machine unless this is true."
        ),
    )

    @model_validator(mode="after")
    def _check_steps_and_override(self) -> "SparseLoRASettings":
        if self.end_step <= self.start_step:
            raise ValueError(
                f"sparselora.end_step ({self.end_step}) must be greater than "
                f"start_step ({self.start_step})."
            )
        if (
            self.calibration.method == CalibrationMethod.NONE
            and not self.layer_sparsity
        ):
            raise ValueError(
                "sparselora.calibration.method: none requires an explicit "
                "`sparselora.layer_sparsity` map."
            )
        if self.layer_sparsity:
            for name, val in self.layer_sparsity.items():
                if not 0.0 <= val < 1.0:
                    raise ValueError(
                        f"sparselora.layer_sparsity['{name}'] must be in [0, 1); got {val}."
                    )
        return self


class SparseLoRAArgs(BaseModel):
    """Plugin entry that exposes the nested `sparselora` block to the core config."""

    sparselora: SparseLoRASettings | None = Field(
        default=None,
        description=(
            "SparseLoRA contextual-sparsity configuration. Register "
            "`axolotl.integrations.sparselora.SparseLoRAPlugin` to enable this block."
        ),
    )
