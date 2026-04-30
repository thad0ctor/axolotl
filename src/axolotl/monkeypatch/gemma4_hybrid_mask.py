"""Hybrid attention mask fix for Gemma 4.

Gemma 4 has full-attention (global) layers with ``head_dim=512`` which
exceeds flash-attention-2's supported size. Axolotl's hybrid-attention
patch in ``patch_manager._apply_gemma_hybrid_attention`` works around
this by forcing ``_attn_implementation="sdpa"`` on each global layer's
``self_attn.config``, leaving sliding-window layers on FA2.

The per-layer config override alone is insufficient, however: the model's
forward builds a ``causal_mask_mapping`` dict using the **model-level**
config and passes the mapped mask to each decoder layer. With FA2 still
set at the model level, the ``full_attention`` entry in that mapping is
a 2D mask (FA2 format), but SDPA needs a 4D mask. The global layers then
fail with::

    RuntimeError: The expanded size of the tensor (S) must match the existing
    size (B) at non-singleton dimension 2. Target sizes: [B, H, S, S]. Tensor
    sizes: [B, S]

...whenever a batch contains any padding tokens (mbs>1 with variable-length
samples, or sequence lengths past roughly 7k tokens).

This module fixes the symptom by monkey-patching the mask-builder symbols
in ``transformers.models.gemma4.modeling_gemma4``'s module namespace —
NOT the originals in ``masking_utils``. The wrappers force
``_attn_implementation="sdpa"`` on a shallow-copied config before calling
through, so the ``full_attention`` mask built inside the model's forward
is always 4D/SDPA-compatible. ``create_sliding_window_causal_mask`` is
left alone, so sliding-window layers continue to receive FA2-format masks.

Two distinct forward paths exist and both must be covered:

* ``Gemma4TextModel.forward`` (text-only ``Gemma4ForCausalLM``) calls
  ``create_causal_mask`` directly. Wrapping that symbol covers this path.

* ``Gemma4Model.forward`` (multimodal ``Gemma4ForConditionalGeneration``)
  branches on ``config.use_bidirectional_attention == "vision"``. The
  ``"vision"`` branch calls ``create_causal_mask_mapping`` (defined inside
  ``modeling_gemma4`` itself, which internally calls module-level
  ``create_causal_mask`` — covered transitively). The other branch calls
  ``create_masks_for_generate`` (imported from ``masking_utils``), whose
  internal dispatch happens in ``masking_utils``' namespace and is NOT
  reached by the ``create_causal_mask`` rebind. That second branch must
  be wrapped explicitly.

The patch is idempotent. Install once per process, before any Gemma 4
forward pass runs.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Source of truth: maps modeling_gemma4 symbol name -> original (unwrapped)
# callable. Empty when no patch is active. Test fixtures may reach in and
# clear this dict to reset patch state between tests.
_PATCH_STATE: dict[str, Callable] = {}


# Sentinel marker on every wrapper produced by this module. We cannot
# rely on ``hasattr(fn, "_axolotl_original")`` alone because MagicMock
# auto-spawns any attribute access, which would falsely identify a test
# mock as an already-installed wrapper.
_WRAPPER_MARKER = "_axolotl_gemma4_hybrid_mask_wrapper"


def _is_axolotl_wrapper(fn: Any) -> bool:
    """Detect a wrapper installed by this module."""
    return callable(fn) and getattr(fn, _WRAPPER_MARKER, False) is True


def _build_create_causal_mask_wrapper(original: Callable) -> Callable:
    """Wrap ``create_causal_mask`` to force SDPA format.

    The global layers were patched to SDPA by
    ``_apply_gemma_hybrid_attention``, so their mask must be 4D. The
    original dispatches on ``config._attn_implementation``; we shadow
    that with a local override.
    """

    def hybrid_create_causal_mask(config: Any, *args: Any, **kwargs: Any):
        sdpa_config = copy.copy(config)
        sdpa_config._attn_implementation = "sdpa"
        return original(sdpa_config, *args, **kwargs)

    hybrid_create_causal_mask._axolotl_original = original  # type: ignore[attr-defined]
    setattr(hybrid_create_causal_mask, _WRAPPER_MARKER, True)
    return hybrid_create_causal_mask


def _build_create_masks_for_generate_wrapper(original: Callable) -> Callable:
    """Wrap ``create_masks_for_generate`` for the multimodal forward path.

    For Gemma 4 (``config.layer_types`` set), the original returns a dict
    keyed by layer pattern. Each entry is built by dispatching on
    ``config._attn_implementation``, so a single FA2 config produces a
    2D mask for every key — including ``full_attention``, which the
    SDPA-promoted global layers then crash on.

    Strategy: call the original twice — once with the caller's config,
    once with an SDPA-shadowed copy. From the SDPA result take the
    ``full_attention`` entry (forces 4D shape for SDPA layers); from the
    original result take everything else (sliding/chunked layers stay in
    FA2 format).

    Cost note: mask construction at long context is non-trivial — an SDPA
    ``full_attention`` mask is ``[B, 1, S, S]`` boolean, which at S=8192,
    B=2 is roughly 128 MB of transient allocation per forward step. The
    two-call approach allocates that twice (once for the FA2 result we
    discard, once for the SDPA result we keep). For the failure mode
    being patched (mbs>1 + padding) attention itself is far more expensive
    than mask construction, so the doubled cost is acceptable. If
    profiling later flags this path, replace the second call with direct
    invocation of ``LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING["full_attention"]``
    using the SDPA-shadowed config — same result, half the allocation.

    Non-dict returns (configs without ``layer_types``) are passed through
    untouched: those models don't have hybrid layer types, so the bug
    doesn't apply.
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
            # Original took a different branch under SDPA; bail out rather
            # than guess. The unpatched output is still usable for
            # non-hybrid configs.
            return original_result

        merged = dict(original_result)
        merged["full_attention"] = sdpa_result["full_attention"]
        return merged

    hybrid_create_masks_for_generate._axolotl_original = original  # type: ignore[attr-defined]
    setattr(hybrid_create_masks_for_generate, _WRAPPER_MARKER, True)
    return hybrid_create_masks_for_generate


# Maps modeling_gemma4 symbol → wrapper factory. Symbols missing on older
# transformers versions are skipped at install time.
_WRAPPED_SYMBOLS: dict[str, Callable[[Callable], Callable]] = {
    "create_causal_mask": _build_create_causal_mask_wrapper,
    "create_masks_for_generate": _build_create_masks_for_generate_wrapper,
}


def patch_gemma4_hybrid_mask() -> bool:
    """Install the Gemma 4 hybrid-attention mask fix.

    Returns ``True`` if the patch was installed (or was already installed),
    ``False`` if the target module could not be imported (e.g. transformers
    version predates Gemma 4) — in which case nothing is done and the
    caller can continue unaffected.
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
            # Already wrapped by an earlier install whose state we lost
            # (e.g. test fixture cleared _PATCH_STATE without restoring the
            # bindings). Recover the original via the wrapper's marker so
            # unpatch still works, but warn — this indicates buggy state
            # management upstream of us.
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
        # Only restore if the current binding is still our wrapper. If
        # something downstream replaced it, leave it alone.
        if _is_axolotl_wrapper(current):
            setattr(modeling_gemma4, symbol_name, original)
    _PATCH_STATE.clear()
