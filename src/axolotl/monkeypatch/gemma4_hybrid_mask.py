"""Hybrid attention mask fix for Gemma 4 (standard and unified).

Gemma 4's full-attention (global) layers have ``head_dim=512`` which
exceeds FA2's supported size, so ``patch_manager._apply_gemma_hybrid_attention``
forces SDPA per-global-layer while leaving sliding layers on FA2. But
the model's forward builds the ``causal_mask_mapping`` dict from the
**model-level** config, so its ``full_attention`` entry is a 2D FA2 mask
that the SDPA layers then crash on with::

    RuntimeError: The expanded size of the tensor (S) must match the existing
    size (B) at non-singleton dimension 2. Target sizes: [B, H, S, S]. Tensor
    sizes: [B, S]

...whenever a batch has padding (mbs>1) or seq len exceeds ~7k.

We fix the symptom by wrapping the mask-builder symbols in the model's
*module namespace* (NOT the originals in ``masking_utils``). Each wrapper
forces ``_attn_implementation="sdpa"`` on a shallow-copied config before
calling through, so the ``full_attention`` mask comes out 4D/SDPA-compatible.
``create_sliding_window_causal_mask`` is left alone, so sliding-window
layers keep receiving FA2-format masks.

``gemma4_unified`` reproduces the same mixed sliding/global architecture
(``global_head_dim=512``) in its own ``modeling_gemma4_unified`` namespace,
so both namespaces are patched when present.

Two forward paths exist, both must be covered:

* ``Gemma4TextModel.forward`` calls ``create_causal_mask`` directly.
* ``Gemma4Model.forward`` (multimodal) branches: the vision branch goes
  through ``create_causal_mask_mapping`` (covered transitively via
  module-level ``create_causal_mask``); the non-vision branch calls
  ``create_masks_for_generate``, whose dispatch lives in
  ``masking_utils`` and is NOT reached by the ``create_causal_mask``
  rebind — so it must be wrapped explicitly.

Idempotent. Install once per process before any Gemma 4 forward pass.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any, Callable

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PATCH_APPLIED = False

# MagicMock auto-spawns attributes, so ``hasattr(fn, "_axolotl_original")``
# alone would falsely identify mocks as wrappers. Marker fixes that.
_WRAPPER_MARKER = "_axolotl_gemma4_hybrid_mask_wrapper"


def _is_axolotl_wrapper(fn: Any) -> bool:
    return callable(fn) and getattr(fn, _WRAPPER_MARKER, False) is True


def _build_create_causal_mask_wrapper(original: Callable) -> Callable:
    def hybrid_create_causal_mask(config: Any, *args: Any, **kwargs: Any):
        sdpa_config = copy.copy(config)
        sdpa_config._attn_implementation = "sdpa"
        return original(sdpa_config, *args, **kwargs)

    hybrid_create_causal_mask._axolotl_original = original  # type: ignore[attr-defined]
    setattr(hybrid_create_causal_mask, _WRAPPER_MARKER, True)
    return hybrid_create_causal_mask


def _build_create_masks_for_generate_wrapper(original: Callable) -> Callable:
    """Wrap ``create_masks_for_generate`` for the multimodal forward path.

    For Gemma 4 the original returns a dict keyed by layer pattern, each
    entry built from a single config — so an FA2 config produces a 2D
    ``full_attention`` mask the SDPA global layers crash on. We call the
    original twice (FA2 then SDPA) and keep ``full_attention`` from the
    SDPA result, everything else from the FA2 result. The doubled mask
    allocation (~128 MB at S=8192, B=2) is dwarfed by attention cost in
    the failure-mode batch shapes; if it ever matters, swap the second
    call for a direct ``LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING["full_attention"]``
    invocation. Non-dict returns pass through (no hybrid layer types).
    """

    def hybrid_create_masks_for_generate(config: Any, *args: Any, **kwargs: Any):
        original_result = original(config, *args, **kwargs)

        if not isinstance(original_result, dict):
            return original_result
        if "full_attention" not in original_result:
            return original_result

        sdpa_config = copy.copy(config)
        sdpa_config._attn_implementation = "sdpa"
        sdpa_result = original(sdpa_config, *args, **kwargs)

        if not isinstance(sdpa_result, dict) or "full_attention" not in sdpa_result:
            # SDPA call took a different branch; bail out rather than guess.
            return original_result

        merged = dict(original_result)
        merged["full_attention"] = sdpa_result["full_attention"]
        return merged

    hybrid_create_masks_for_generate._axolotl_original = original  # type: ignore[attr-defined]
    setattr(hybrid_create_masks_for_generate, _WRAPPER_MARKER, True)
    return hybrid_create_masks_for_generate


# Symbols missing on older transformers versions are skipped at install.
_WRAPPED_SYMBOLS: dict[str, Callable[[Callable], Callable]] = {
    "create_causal_mask": _build_create_causal_mask_wrapper,
    "create_masks_for_generate": _build_create_masks_for_generate_wrapper,
}

# Each Gemma 4 variant fully redefines ``create_causal_mask`` in its own module
# namespace (gemma4_unified does NOT modular-import from gemma4), so both must be
# patched independently.
_TARGET_MODULES = (
    "transformers.models.gemma4.modeling_gemma4",
    "transformers.models.gemma4_unified.modeling_gemma4_unified",
)


def _patch_module(module: Any) -> tuple[list[str], list[str]]:
    """Wrap every known mask-builder symbol present in one module namespace.

    Returns ``(installed, covered)`` where ``installed`` is the symbols newly
    wrapped this call and ``covered`` is every symbol left in a wrapped state
    (newly wrapped *or* already our wrapper from a prior call). A symbol that is
    already our wrapper is left untouched — this keeps re-entry idempotent even
    if ``_PATCH_APPLIED`` was reset externally, without re-wrapping a wrapper.
    Missing symbols (older transformers) are skipped.
    """
    installed: list[str] = []
    covered: list[str] = []
    for symbol_name, build_wrapper in _WRAPPED_SYMBOLS.items():
        current = getattr(module, symbol_name, None)
        if current is None:
            continue
        if _is_axolotl_wrapper(current):
            covered.append(symbol_name)
            continue
        setattr(module, symbol_name, build_wrapper(current))
        installed.append(symbol_name)
        covered.append(symbol_name)
    return installed, covered


def patch_gemma4_hybrid_mask() -> bool:
    """Install the Gemma 4 hybrid-attention mask fix across all variants.

    Returns ``True`` if at least one namespace was patched, ``False`` if none
    of the target modules could be imported (e.g. transformers version predates
    Gemma 4) — in which case nothing is done and the caller can continue
    unaffected.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    patched_any = False
    for module_path in _TARGET_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            LOG.debug(
                "gemma4_hybrid_mask: %s not importable, skipping. This is fine "
                "for non-Gemma4 training.",
                module_path,
            )
            continue
        installed, covered = _patch_module(module)
        if covered:
            patched_any = True
        if installed:
            LOG.info(
                "gemma4_hybrid_mask: patched %s symbols [%s] to force "
                "SDPA-format masks for full-attention layers",
                module_path,
                ", ".join(installed),
            )

    if patched_any:
        _PATCH_APPLIED = True
    else:
        LOG.warning(
            "gemma4_hybrid_mask: no expected mask-builder symbols found on any "
            "target module (looked for: %s). Transformers API may have changed; "
            "the hybrid attention bug fix is not active.",
            ", ".join(_WRAPPED_SYMBOLS),
        )
    return patched_any


def unpatch_gemma4_hybrid_mask() -> None:
    """Restore the original mask-builder symbols in every namespace. Tests."""
    global _PATCH_APPLIED
    if not _PATCH_APPLIED:
        return
    for module_path in _TARGET_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        for symbol_name in _WRAPPED_SYMBOLS:
            current = getattr(module, symbol_name, None)
            # Only restore if a downstream patch hasn't replaced our wrapper.
            if _is_axolotl_wrapper(current):
                setattr(module, symbol_name, current._axolotl_original)
    _PATCH_APPLIED = False
