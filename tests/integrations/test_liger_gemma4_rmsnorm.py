"""Regression tests for the Liger Gemma4 RMSNorm deepcopy-safety fix.

When ``liger_rms_norm`` is enabled and any Gemma 4 module containing nested
``Gemma4RMSNorm`` layers (e.g. ``vision_tower``) is listed in
``lora_modules_to_save``, PEFT deep-copies the module to create a trainable
clone. ``copy.deepcopy`` dispatches through ``copyreg.__newobj__`` which
invokes ``cls.__new__(cls)`` with no positional arguments, so
``_LigerGemma4RMSNorm.__new__`` / ``__init__`` must accept the no-arg call
shape. These tests lock that invariant in so a future refactor that makes
``dim`` required again fails fast.
"""

import copy

import pytest
import torch
import torch.nn as nn

pytest.importorskip("liger_kernel")
modeling_gemma4 = pytest.importorskip(
    "transformers.models.gemma4.modeling_gemma4",
    reason="Gemma 4 not available in installed transformers",
)

from axolotl.integrations.liger.plugin import LigerPlugin  # noqa: E402
from axolotl.utils.dict import DictDefault  # noqa: E402


@pytest.fixture
def liger_gemma4_rmsnorm_cls():
    """Run ``LigerPlugin.pre_model_load`` for Gemma 4 with ``liger_rms_norm``
    enabled so that ``modeling_gemma4.Gemma4RMSNorm`` points at the plugin's
    closure-scoped ``_LigerGemma4RMSNorm``. Restore the original on teardown.
    """
    original = modeling_gemma4.Gemma4RMSNorm

    cfg = DictDefault(
        {
            "model_config_type": "gemma4",
            "liger_rms_norm": True,
        }
    )
    LigerPlugin().pre_model_load(cfg)
    installed = modeling_gemma4.Gemma4RMSNorm
    assert installed is not original, "plugin did not install Liger RMSNorm"
    try:
        yield installed
    finally:
        modeling_gemma4.Gemma4RMSNorm = original


def test_new_accepts_no_args(liger_gemma4_rmsnorm_cls):
    """``cls.__new__(cls)`` with no ``dim`` is the pickle/deepcopy call shape."""
    instance = liger_gemma4_rmsnorm_cls.__new__(liger_gemma4_rmsnorm_cls)
    assert isinstance(instance, liger_gemma4_rmsnorm_cls)


def test_init_no_args_is_noop(liger_gemma4_rmsnorm_cls):
    """``__init__`` must short-circuit when ``dim`` is missing, otherwise
    ``LigerRMSNorm.__init__`` raises on required ``dim``."""
    instance = liger_gemma4_rmsnorm_cls.__new__(liger_gemma4_rmsnorm_cls)
    nn.Module.__init__(instance)
    instance.__init__()


def test_deepcopy_bare_instance(liger_gemma4_rmsnorm_cls):
    """Directly reproduces the crash fixed in 000260ed: PEFT's
    ``ModulesToSaveWrapper.update`` calls ``copy.deepcopy(original_module)``,
    which recurses into every contained RMSNorm."""
    original = liger_gemma4_rmsnorm_cls(64)
    clone = copy.deepcopy(original)
    assert clone is not original
    assert isinstance(clone, liger_gemma4_rmsnorm_cls)


def test_deepcopy_preserves_state(liger_gemma4_rmsnorm_cls):
    """Deepcopy's ``__setstate__`` must restore the ``weight`` tensor."""
    original = liger_gemma4_rmsnorm_cls(32)
    with torch.no_grad():
        original.weight.fill_(0.42)
    clone = copy.deepcopy(original)
    assert torch.equal(original.weight, clone.weight)
    assert clone.weight.data_ptr() != original.weight.data_ptr()


def test_normal_construction_unchanged(liger_gemma4_rmsnorm_cls):
    """Keyword / positional construction with ``dim`` must still produce a
    fully-initialized module — the ``dim=None`` default is only for the
    pickle path."""
    instance = liger_gemma4_rmsnorm_cls(dim=16, eps=1e-6, with_scale=True)
    assert hasattr(instance, "weight")
    assert instance.weight.shape == (16,)


def test_with_scale_false_delegates_to_original(liger_gemma4_rmsnorm_cls):
    """``with_scale=False`` is how Gemma 4 opts out of learnable scale; this
    path must still return the stock HF class, not the Liger subclass."""
    _OrigGemma4RMSNorm = type(liger_gemma4_rmsnorm_cls(16, with_scale=False))
    assert _OrigGemma4RMSNorm is not liger_gemma4_rmsnorm_cls
    assert _OrigGemma4RMSNorm.__name__ == "Gemma4RMSNorm"


def test_deepcopy_of_containing_module(liger_gemma4_rmsnorm_cls):
    """Real PEFT failure mode: ``ModulesToSaveWrapper`` deepcopies the
    outer module (vision_tower), which recurses through every submodule —
    every nested RMSNorm must survive the round-trip."""

    class VisionTowerLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = liger_gemma4_rmsnorm_cls(16)
            self.proj = nn.Linear(16, 16)

    tower = VisionTowerLike()
    tower_clone = copy.deepcopy(tower)
    assert isinstance(tower_clone.norm, liger_gemma4_rmsnorm_cls)
    assert tower_clone.norm is not tower.norm
    assert tower_clone.proj.weight.data_ptr() != tower.proj.weight.data_ptr()
