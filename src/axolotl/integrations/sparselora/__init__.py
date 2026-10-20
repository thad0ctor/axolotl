"""SparseLoRA contextual-sparsity integration for Axolotl.

Vendors z-lab/sparselora (see ``_vendor/PROVENANCE.md``) and adds self
calibration so the per-layer sparsity schedule and SVD predictor factors are
derived from the user's own model and dataset rather than z-lab's Llama-only
presets.
"""

from .args import (
    CalibrationConfig,
    CalibrationMethod,
    SparseLoRAArgs,
    SparseLoRASettings,
)
from .plugin import SparseLoRAPlugin

__all__ = [
    "CalibrationConfig",
    "CalibrationMethod",
    "SparseLoRAArgs",
    "SparseLoRASettings",
    "SparseLoRAPlugin",
]
