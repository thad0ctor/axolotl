"""Config args for SparseLoRA contextual-sparsity fine-tuning."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CalibrationMethod(str, Enum):
    """How the per-layer sparsity schedule is determined before training."""

    STRUCTURAL = (
        "structural"  # z-lab-shaped positional profile (default, no forward pass)
    )
    PRESET = "preset"  # load a published z-lab schedule (from_pretrained mode=o1/o2)
    FAITHFUL = "faithful"  # structural profile refined by a warm-up sensitivity sweep
    PROXY = "proxy"  # structural profile refined by a forward-only sensitivity sweep
    NONE = "none"  # use the explicit `layer_sparsity` map verbatim


class CalibrationConfig(BaseModel):
    """Calibration settings. Calibration runs on a slice of the *same* dataset."""

    # Reject typos in the `sparselora.calibration` block instead of dropping them.
    model_config = ConfigDict(extra="forbid")

    method: CalibrationMethod = Field(
        default=CalibrationMethod.STRUCTURAL,
        description=(
            "Schedule source: `structural` (default) applies z-lab's empirical "
            "profile -- dense shallow + final layers, aggressive deep MLP, milder "
            "attention -- from the model's layer structure alone (no forward pass); "
            "`preset` loads a published z-lab schedule via `preset`/`preset_mode`; "
            "`faithful`/`proxy` start from the structural profile and use a "
            "sensitivity sweep to demote the most sensitive band layers back to "
            "dense (`faithful` adds a short dense LoRA warm-up first); `none` uses "
            "the explicit `layer_sparsity` override."
        ),
    )
    num_samples: int = Field(
        default=128,
        gt=0,
        description="`faithful`/`proxy`: examples drawn from the dataset for the sweep.",
    )
    batch_size: int = Field(
        default=1, gt=0, description="Micro-batch size for the calibration passes."
    )
    warmup_steps: int = Field(
        default=50,
        ge=0,
        description="`faithful` only: dense LoRA reference-training steps before the sweep.",
    )
    dense_prefix: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "MLP: keep this many leading (shallow) layers dense -- a fraction of "
            "depth when <=1, else an absolute layer count. Shallow layers are the "
            "least amenable to sparsification (z-lab)."
        ),
    )
    attn_dense_prefix: float = Field(
        default=0.45,
        ge=0.0,
        description=(
            "Attention: keep this many leading layers dense (fraction when <=1). "
            "z-lab sparsifies attention only in deeper layers, so this is larger "
            "than `dense_prefix`."
        ),
    )
    sensitivity_demote: float = Field(
        default=0.25,
        ge=0.0,
        lt=1.0,
        description=(
            "`faithful`/`proxy` only: fraction of each group's sparsifiable band "
            "(the most sensitive layers by the reconstruction sweep) demoted back "
            "to dense."
        ),
    )
    loss_budget: float = Field(
        default=0.01,
        gt=0,
        description=(
            "Deprecated and ignored. Absolute per-block reconstruction error was "
            "found not to track downstream task sensitivity (z-lab's published "
            "schedules run MLP layers at 0.97-0.99 sparsity, whose block "
            "reconstruction error is ~0.9); it is retained only for config/cache "
            "compatibility."
        ),
    )


class SparseLoRASettings(BaseModel):
    """Nested SparseLoRA configuration available under the `sparselora` key."""

    # Reject typos in the `sparselora` block instead of silently dropping them.
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True, description="Master switch for the SparseLoRA plugin."
    )
    target_sparsity: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description=(
            "Base-path sparsity applied to MLP layers inside the sparsifiable band "
            "(z-lab's published schedules use 0.97-0.99 here)."
        ),
    )
    attn_sparsity: float | None = Field(
        default=None,
        description=(
            "Base-path sparsity for attention layers in the band. Defaults to "
            "min(target_sparsity, 0.75) -- attention tolerates far less sparsity "
            "than the MLP (z-lab o2 uses 0.75 attention vs 0.99 MLP)."
        ),
    )
    preset: str | None = Field(
        default=None,
        description=(
            "`preset` method: a z-lab-format SparseLoRA repo id or local dir whose "
            "`config.json` holds published `modes` (e.g. "
            "'z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA'). This is z-lab's "
            "downstream-validated path for the Llama models they calibrated."
        ),
    )
    preset_mode: str = Field(
        default="o2",
        description="`preset` method: which published mode to load (`o1` conservative, `o2` aggressive).",
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
        if self.calibration.method == CalibrationMethod.PRESET and not self.preset:
            raise ValueError(
                "sparselora.calibration.method: preset requires `sparselora.preset` "
                "(a z-lab-format SparseLoRA repo id or local dir, e.g. "
                "'z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA')."
            )
        if self.attn_sparsity is not None and not 0.0 <= self.attn_sparsity < 1.0:
            raise ValueError(
                f"sparselora.attn_sparsity must be in [0, 1); got {self.attn_sparsity}."
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
