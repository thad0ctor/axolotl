"""Shared fixtures for ProTrain plugin tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def gpu_device() -> int:
    """Resolve the GPU ordinal tests should use.

    Honors ``CUDA_VISIBLE_DEVICES`` when set — the first listed device maps to
    logical ordinal 0 under PyTorch's device masking. Falls back to 0.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",")[0].strip()
        if first.isdigit():
            return 0  # logical ordinal under CUDA_VISIBLE_DEVICES masking
    return 0


@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Deterministic seed for every test in this package."""
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
