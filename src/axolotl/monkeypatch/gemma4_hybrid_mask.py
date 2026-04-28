"""Hybrid attention mask fix for Gemma 4.

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

We rebind the mask-builder symbols in ``modeling_gemma4``'s namespace
(NOT the originals in ``masking_utils``) to force SDPA on a shallow
config copy, so the ``full_attention`` mask comes out 4D.

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
from typing import Any, Callable

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PATCH_STATE: dict[str, Callable] = {}

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


def patch_gemma4_hybrid_mask() -> bool:
    """Install the Gemma 4 hybrid-attention mask fix.

    Returns True on install (or no-op if already installed), False if the
    target module is missing.
    """
    if _PATCH_STATE:
        return True

    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        LOG.debug(
            "gemma4_hybrid_mask: transformers.models.gemma4 not importable, "
            "skipping. This is fine for non-Gemma4 training."
        )
        return False

    installed: list[str] = []
    skipped: list[str] = []
    for symbol_name, build_wrapper in _WRAPPED_SYMBOLS.items():
        current = getattr(modeling_gemma4, symbol_name, None)
        if current is None:
            skipped.append(symbol_name)
            continue
        if _is_axolotl_wrapper(current):
            # Wrapper bound but state empty — fixture cleared state without
            # restoring bindings. Recover original via marker and warn.
            LOG.warning(
                "gemma4_hybrid_mask: %s is already wrapped but _PATCH_STATE "
                "was empty. Recovering original via wrapper marker.",
                symbol_name,
            )
            _PATCH_STATE[symbol_name] = current._axolotl_original  # type: ignore[attr-defined]
            continue
        wrapper = build_wrapper(current)
        setattr(modeling_gemma4, symbol_name, wrapper)
        _PATCH_STATE[symbol_name] = current
        installed.append(symbol_name)

    if not _PATCH_STATE:
        LOG.warning(
            "gemma4_hybrid_mask: no expected mask-builder symbols found on "
            "modeling_gemma4 (looked for: %s). Transformers API may have "
            "changed; the hybrid attention bug fix is not active.",
            ", ".join(_WRAPPED_SYMBOLS),
        )
        return False

    if installed:
        LOG.info(
            "gemma4_hybrid_mask: patched modeling_gemma4 symbols [%s] to "
            "force SDPA-format masks for full-attention layers",
            ", ".join(installed),
        )
    if skipped:
        LOG.debug(
            "gemma4_hybrid_mask: skipped symbols missing on this "
            "transformers version: %s",
            ", ".join(skipped),
        )
    return True


def unpatch_gemma4_hybrid_mask() -> None:
    """Restore the original mask-builder symbols. Useful for tests."""
    if not _PATCH_STATE:
        return
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        _PATCH_STATE.clear()
        return

    for symbol_name, original in list(_PATCH_STATE.items()):
        current = getattr(modeling_gemma4, symbol_name, None)
        # Only restore if a downstream patch hasn't replaced our wrapper.
        if _is_axolotl_wrapper(current):
            setattr(modeling_gemma4, symbol_name, original)
    _PATCH_STATE.clear()
