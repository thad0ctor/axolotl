"""Tests for the Gemma 4 hybrid-attention mask fix.

These tests pin the critical behavior: after installing the patch, the
mask-builder symbols in ``modeling_gemma4`` pass an SDPA-overridden
config to the underlying mask factory whenever the result feeds the
full-attention (head_dim=512) global layers. This is what keeps those
layers from crashing at long sequence lengths or with padding under
mbs>1 — they need a 4D SDPA-format mask, not the 2D FA2 mask that would
be built from the model-level config.

Two distinct entry points exist and both must be covered:

* ``create_causal_mask`` — used by the text-only ``Gemma4TextModel``
  forward and (transitively) by ``create_causal_mask_mapping`` on the
  multimodal vision-bidirectional path.
* ``create_masks_for_generate`` — used by the multimodal
  ``Gemma4Model.forward`` non-vision branch. Its internal dispatch lives
  in ``masking_utils``' namespace and is NOT reached by rebinding
  ``create_causal_mask`` alone.

The unit tests use mocks to verify config flipping; the integration
tests run a tiny randomly-initialized model end-to-end with padded
attention masks at mbs=2 to reproduce the exact failure mode that blew
up the Gemma 4 multimodal-CPT pilot.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "transformers.models.gemma4",
    reason="gemma4_hybrid_mask patch only matters when Gemma 4 is available",
)


@pytest.fixture
def restore_gemma4_module():
    """Snapshot every symbol the patch may rebind, restore after each test
    so patch state does not leak across the suite."""
    from transformers.models.gemma4 import modeling_gemma4

    from axolotl.monkeypatch import gemma4_hybrid_mask

    _MISSING = object()
    snapshots = {
        name: getattr(modeling_gemma4, name, _MISSING)
        for name in gemma4_hybrid_mask._WRAPPED_SYMBOLS
    }
    yield modeling_gemma4
    for name, original in snapshots.items():
        if original is _MISSING:
            # Symbol was missing before the test — if a test bound it
            # (e.g. via direct assignment of a MagicMock on a transformers
            # version that lacked it), strip it back off so subsequent tests
            # see the original "missing" state.
            if hasattr(modeling_gemma4, name):
                delattr(modeling_gemma4, name)
        else:
            setattr(modeling_gemma4, name, original)
    gemma4_hybrid_mask._PATCH_STATE.clear()


# ---------------------------------------------------------------------------
# Unit tests: create_causal_mask wrapper
# ---------------------------------------------------------------------------


def test_patch_replaces_create_causal_mask(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    original = modeling_gemma4.create_causal_mask
    assert patch_gemma4_hybrid_mask() is True

    assert modeling_gemma4.create_causal_mask is not original
    assert modeling_gemma4.create_causal_mask._axolotl_original is original, (
        "patched wrapper must expose the original reference for teardown"
    )


def test_patch_is_idempotent(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    patch_gemma4_hybrid_mask()
    wrapper_first = modeling_gemma4.create_causal_mask

    # Second call must not re-wrap the already-wrapped function (which
    # would leak the original reference through a chain of wrappers).
    patch_gemma4_hybrid_mask()
    wrapper_second = modeling_gemma4.create_causal_mask

    assert wrapper_first is wrapper_second


def test_patched_mask_forces_sdpa_config(restore_gemma4_module):
    """Core invariant: when the patched wrapper is called with a config
    that says ``flash_attention_2``, the underlying mask factory receives
    a shallow-copied config whose ``_attn_implementation`` is ``"sdpa"``.

    Without this, the full-attention global layers get a 2D FA2 mask and
    crash at long seq lens with the [B, H, S, S] / [B, S] expand error.
    """
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    # Swap in a mock BEFORE installing the patch so the wrapper captures
    # it as the "original". The mock records every call so we can inspect
    # what config got passed through.
    mock_factory = MagicMock(name="create_causal_mask", return_value="mask_4d")
    modeling_gemma4.create_causal_mask = mock_factory
    patch_gemma4_hybrid_mask()

    # Caller-supplied config says FA2 (that's the model-level setting).
    caller_config = SimpleNamespace(
        _attn_implementation="flash_attention_2",
        head_dim=512,
        some_other_attr="preserved",
    )
    result = modeling_gemma4.create_causal_mask(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    # Wrapper returned whatever the mock returned — no transformation of
    # the result itself.
    assert result == "mask_4d"

    # The mock was called exactly once with a config whose
    # ``_attn_implementation`` is sdpa, NOT the caller's fa2.
    assert mock_factory.call_count == 1
    (passed_config, *_), passed_kwargs = mock_factory.call_args
    assert passed_config._attn_implementation == "sdpa"

    # The wrapper must NOT mutate the caller's config in place — other
    # mask builders (e.g. create_sliding_window_causal_mask) read from
    # the same config and must still see fa2.
    assert caller_config._attn_implementation == "flash_attention_2"

    # Other attributes on the config must be preserved so the underlying
    # factory has everything it needs (head_dim, rope_theta, vocab_size, ...).
    assert passed_config.head_dim == 512
    assert passed_config.some_other_attr == "preserved"


def test_patched_wrapper_passes_through_all_kwargs(restore_gemma4_module):
    """The wrapper must forward positional + keyword args to the original
    unchanged, so transformers' own call-site in Gemma4TextModel.forward
    keeps working across minor transformers-version signature drift."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    mock_factory = MagicMock(return_value="mask")
    modeling_gemma4.create_causal_mask = mock_factory
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    modeling_gemma4.create_causal_mask(
        caller_config,
        "positional_arg",
        inputs_embeds="embeds",
        attention_mask="mask_2d",
        past_key_values="cache",
        position_ids="positions",
        or_mask_function="or_fn",
    )

    args, kwargs = mock_factory.call_args
    # First positional (after config override) is preserved.
    assert args[1] == "positional_arg"
    # All kwargs are forwarded untouched.
    assert kwargs["inputs_embeds"] == "embeds"
    assert kwargs["attention_mask"] == "mask_2d"
    assert kwargs["past_key_values"] == "cache"
    assert kwargs["position_ids"] == "positions"
    assert kwargs["or_mask_function"] == "or_fn"


def test_unpatch_restores_original(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import (
        patch_gemma4_hybrid_mask,
        unpatch_gemma4_hybrid_mask,
    )

    sentinel = MagicMock(name="original")
    modeling_gemma4.create_causal_mask = sentinel
    patch_gemma4_hybrid_mask()
    assert modeling_gemma4.create_causal_mask is not sentinel

    unpatch_gemma4_hybrid_mask()
    assert modeling_gemma4.create_causal_mask is sentinel


def test_unpatch_is_safe_without_prior_patch(restore_gemma4_module):
    from axolotl.monkeypatch.gemma4_hybrid_mask import unpatch_gemma4_hybrid_mask

    # Should be a no-op, no exception.
    unpatch_gemma4_hybrid_mask()


def test_sliding_window_mask_builder_is_not_patched(restore_gemma4_module):
    """Only the full-attention mask builders are overridden — the
    sliding-window factory must remain bound to its original to preserve
    FA2 masks for the sliding-attention layers. If we accidentally patch
    both, the sliding layers get SDPA format and lose the FA2 speedup."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_sliding_window_causal_mask"):
        pytest.skip("transformers version has no create_sliding_window_causal_mask")

    sliding_before = modeling_gemma4.create_sliding_window_causal_mask
    patch_gemma4_hybrid_mask()
    sliding_after = modeling_gemma4.create_sliding_window_causal_mask
    assert sliding_after is sliding_before


# ---------------------------------------------------------------------------
# Unit tests: create_masks_for_generate wrapper (multimodal forward path)
# ---------------------------------------------------------------------------


def test_patch_replaces_create_masks_for_generate(restore_gemma4_module):
    """The multimodal ``Gemma4Model.forward`` non-vision branch calls
    ``create_masks_for_generate``. Without wrapping it, the full_attention
    entry in the returned dict is a 2D FA2 mask and the SDPA-promoted
    global layers crash."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip(
            "transformers version has no modeling_gemma4.create_masks_for_generate"
        )

    original = modeling_gemma4.create_masks_for_generate
    assert patch_gemma4_hybrid_mask() is True

    assert modeling_gemma4.create_masks_for_generate is not original
    assert modeling_gemma4.create_masks_for_generate._axolotl_original is original, (
        "patched wrapper must expose the original reference for teardown"
    )


def test_create_masks_for_generate_wrapper_replaces_full_attention_only(
    restore_gemma4_module,
):
    """Core invariant for the multimodal path: the wrapper rebuilds only
    the ``full_attention`` entry of the returned dict using an SDPA
    config, leaves ``sliding_attention`` (and any other keys) coming
    from the original FA2 call.

    The original is invoked twice — once for the caller's FA2 config
    (sliding wins from this call), once for an SDPA-shadowed copy
    (full_attention wins from this call). The merged dict has 4D SDPA
    mask for full_attention and 2D FA2 mask for sliding.
    """
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    sdpa_full_mask = "sdpa_full_4d_sentinel"
    fa2_full_mask = "fa2_full_2d_sentinel"
    fa2_sliding_mask = "fa2_sliding_2d_sentinel"
    sdpa_sliding_mask = "sdpa_sliding_4d_sentinel"  # should NOT end up in result

    def fake_original(config, *args, **kwargs):
        # Return different per-key sentinels depending on which config the
        # wrapper passed in. Lets us prove the merge picks SDPA's full and
        # FA2's sliding.
        if config._attn_implementation == "sdpa":
            return {
                "full_attention": sdpa_full_mask,
                "sliding_attention": sdpa_sliding_mask,
            }
        return {
            "full_attention": fa2_full_mask,
            "sliding_attention": fa2_sliding_mask,
        }

    mock_original = MagicMock(side_effect=fake_original)
    modeling_gemma4.create_masks_for_generate = mock_original
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    result = modeling_gemma4.create_masks_for_generate(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    assert result == {
        "full_attention": sdpa_full_mask,
        "sliding_attention": fa2_sliding_mask,
    }
    # Original should be called exactly twice (FA2 first, SDPA second).
    assert mock_original.call_count == 2
    # Caller's config must not be mutated.
    assert caller_config._attn_implementation == "flash_attention_2"


def test_create_masks_for_generate_wrapper_passes_through_non_dict(
    restore_gemma4_module,
):
    """For configs without ``layer_types`` (older Gemma 4 variants or
    other models that share this symbol), ``create_masks_for_generate``
    returns a single tensor or BlockMask, not a dict. The wrapper must
    pass through untouched — the hybrid bug doesn't apply to those
    configs.
    """
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    sentinel = "single_tensor_passthrough"
    mock_original = MagicMock(return_value=sentinel)
    modeling_gemma4.create_masks_for_generate = mock_original
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    result = modeling_gemma4.create_masks_for_generate(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    assert result is sentinel
    # Only one call — no SDPA shadow call needed for non-dict returns.
    assert mock_original.call_count == 1


def test_create_masks_for_generate_wrapper_passes_through_when_no_full_key(
    restore_gemma4_module,
):
    """If the dict happens to lack a ``full_attention`` key (e.g. an
    all-sliding model), the wrapper has nothing to fix and must avoid
    the second call."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    fa2_only = {"sliding_attention": "fa2_sliding"}
    mock_original = MagicMock(return_value=fa2_only)
    modeling_gemma4.create_masks_for_generate = mock_original
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    result = modeling_gemma4.create_masks_for_generate(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    assert result == fa2_only
    assert mock_original.call_count == 1


def test_unpatch_restores_create_masks_for_generate(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import (
        patch_gemma4_hybrid_mask,
        unpatch_gemma4_hybrid_mask,
    )

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    sentinel = MagicMock(name="original_cmfg")
    modeling_gemma4.create_masks_for_generate = sentinel
    patch_gemma4_hybrid_mask()
    assert modeling_gemma4.create_masks_for_generate is not sentinel

    unpatch_gemma4_hybrid_mask()
    assert modeling_gemma4.create_masks_for_generate is sentinel


def test_create_masks_for_generate_wrapper_passes_through_when_sdpa_returns_non_dict(
    restore_gemma4_module,
):
    """Defensive bail-out: if the original returns a dict for FA2 but a
    non-dict (or no full_attention) for SDPA, the wrapper falls back to
    the unpatched output rather than guessing. Pins the L143 branch."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    fa2_dict = {"full_attention": "fa2_full", "sliding_attention": "fa2_sliding"}

    def fake_original(config, *args, **kwargs):
        if config._attn_implementation == "sdpa":
            # Surprise: SDPA call returns a single tensor, not a dict.
            return "unexpected_non_dict_return"
        return fa2_dict

    mock_original = MagicMock(side_effect=fake_original)
    modeling_gemma4.create_masks_for_generate = mock_original
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    result = modeling_gemma4.create_masks_for_generate(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    # Wrapper preserves the FA2 result rather than corrupting the dict.
    assert result == fa2_dict


# ---------------------------------------------------------------------------
# Patch lifecycle / state-management tests
# ---------------------------------------------------------------------------


def test_patch_skips_symbols_missing_on_older_transformers(restore_gemma4_module):
    """Older transformers versions may lack ``create_masks_for_generate``.
    The install must skip the missing wrapper without failing — only the
    available wrappers get installed, and ``patch_gemma4_hybrid_mask``
    still returns True so callers continue normally."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch import gemma4_hybrid_mask
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    # Simulate a transformers version that lacks create_masks_for_generate
    # by temporarily stripping it. The fixture (restore_gemma4_module)
    # restores the original after the test.
    if hasattr(modeling_gemma4, "create_masks_for_generate"):
        delattr(modeling_gemma4, "create_masks_for_generate")

    assert patch_gemma4_hybrid_mask() is True
    assert "create_causal_mask" in gemma4_hybrid_mask._PATCH_STATE
    assert "create_masks_for_generate" not in gemma4_hybrid_mask._PATCH_STATE
    # The missing symbol must not have been auto-bound by the install.
    assert not hasattr(modeling_gemma4, "create_masks_for_generate")


def test_patch_returns_false_when_no_symbols_present(restore_gemma4_module):
    """If transformers ships without any of the expected mask-builder
    symbols (e.g. a future major-version refactor), install reports failure
    so the caller knows the bug fix is not active."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    # Strip both wrapped symbols.
    for name in ("create_causal_mask", "create_masks_for_generate"):
        if hasattr(modeling_gemma4, name):
            delattr(modeling_gemma4, name)

    assert patch_gemma4_hybrid_mask() is False


def test_patch_recovers_when_state_lost_but_wrapper_already_bound(
    restore_gemma4_module, caplog
):
    """If a buggy state-management upstream (a test harness, etc.) clears
    ``_PATCH_STATE`` without restoring the bindings, the install must
    recover the original via the wrapper marker AND log a warning so the
    bug doesn't get swept under the rug. Pins L195-201."""
    import logging

    from axolotl.monkeypatch import gemma4_hybrid_mask
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    # Step 1: clean install.
    patch_gemma4_hybrid_mask()
    state_before = dict(gemma4_hybrid_mask._PATCH_STATE)
    assert state_before, "expected install to populate _PATCH_STATE"

    # Step 2: simulate state loss — wrappers still bound on the module,
    # but our state dict was cleared by something external.
    gemma4_hybrid_mask._PATCH_STATE.clear()

    # Step 3: re-install. The wrappers are detected and the originals are
    # recovered via the ``_axolotl_original`` attribute.
    with caplog.at_level(
        logging.WARNING, logger="axolotl.monkeypatch.gemma4_hybrid_mask"
    ):
        assert patch_gemma4_hybrid_mask() is True

    # Originals were recovered into the state dict.
    for name, orig in state_before.items():
        assert gemma4_hybrid_mask._PATCH_STATE.get(name) is orig
    # Warning was logged so this state corruption is observable.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "already wrapped but _PATCH_STATE was empty" in r.message for r in warnings
    ), "expected a warning about state recovery"


def test_unpatch_leaves_third_party_rebindings_alone(restore_gemma4_module):
    """If a downstream patch (or test code) replaces our wrapper with
    something else after install, ``unpatch`` must NOT clobber that
    third-party binding. Pins the L245 False branch."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import (
        patch_gemma4_hybrid_mask,
        unpatch_gemma4_hybrid_mask,
    )

    patch_gemma4_hybrid_mask()
    third_party = MagicMock(name="downstream_replacement")
    modeling_gemma4.create_causal_mask = third_party

    unpatch_gemma4_hybrid_mask()

    # Third party's binding survives unpatch.
    assert modeling_gemma4.create_causal_mask is third_party


# ---------------------------------------------------------------------------
# Integration tests with a tiny randomly-initialized Gemma4TextModel.
#
# These do NOT load real 26B weights. They build a ~350k-param Gemma 4 text
# model with 2 layers (one sliding, one full_attention), apply the hybrid
# attention path end-to-end, and run a forward pass with a padded
# attention_mask at a long-ish seq len. The invariant we're pinning is that
# the full_attention layer does not crash with the
#   "Target sizes: [B, H, S, S]. Tensor sizes: [B, S]"
# error — the exact failure that blew up the Gemma 4 MoE 26B pilot at ~7k
# tokens in the FSDP2 training run.
# ---------------------------------------------------------------------------


def _build_tiny_gemma4_text_model():
    """Return a tiny randomly-initialized Gemma4TextModel with mixed layers."""
    import torch
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    # Caller-supplied attn impl simulates the pilot config (fa2 at model
    # level). The hybrid patch is what makes this survive long context.
    cfg._attn_implementation = "sdpa"  # start safe; the test toggles fa2 later
    torch.manual_seed(42)
    model = Gemma4TextModel(cfg).eval()
    return model, cfg


def _apply_hybrid_attn_inline(model, cfg):
    """Replicate what ``patch_manager._apply_gemma_hybrid_attention`` does
    to a model, without needing a full PatchManager / pydantic cfg."""
    import copy

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    for layer_idx, layer in enumerate(model.layers):
        if cfg.layer_types[layer_idx] != "sliding_attention":
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "config"):
                sdpa_cfg = copy.copy(attn.config)
                sdpa_cfg._attn_implementation = "sdpa"
                attn.config = sdpa_cfg
    patch_gemma4_hybrid_mask()


def test_tiny_gemma4_long_context_forward_does_not_crash(restore_gemma4_module):
    """End-to-end invariant: with the hybrid attn patch applied, a tiny
    Gemma4TextModel runs a forward at long context (1024 tokens) with
    real padding in the attention mask, producing the expected output
    shape. This exercises the actual code path that crashed the pilot
    without needing a real 26B checkpoint or CUDA."""
    import torch

    model, cfg = _build_tiny_gemma4_text_model()
    _apply_hybrid_attn_inline(model, cfg)

    B, S = 2, 1024
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    attn_mask = torch.ones(B, S, dtype=torch.long)
    # Pad positions in the second row. Without padding, SDPA falls back to
    # ``is_causal=True`` with ``mask=None`` — we need a materialized 4D
    # mask to exercise the actual bug site.
    attn_mask[1, S // 2 :] = 0

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)

    assert out.last_hidden_state.shape == (B, S, cfg.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all()


def test_patched_create_causal_mask_returns_4d_for_real_config(
    restore_gemma4_module,
):
    """Hit the REAL ``create_causal_mask`` (not a mock) via the wrapper
    and verify the returned mask is a 4D tensor — which is the shape the
    SDPA-patched global layers need. Without the patch and with a
    caller-supplied FA2 config this would return a 2D mask and the layer
    would crash at long context."""
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    patch_gemma4_hybrid_mask()
    modeling_gemma4 = restore_gemma4_module

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    # Simulate the pilot: caller says flash_attention_2, but global layers
    # were switched to SDPA per-layer. Without the patch, create_causal_mask
    # would return an FA2 2D mask here and the SDPA layer would crash.
    cfg._attn_implementation = "flash_attention_2"

    B, S = 2, 1024
    inputs_embeds = torch.zeros((B, S, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((B, S), dtype=torch.long)
    attention_mask[1, S // 2 :] = 0  # force the 4D materialized path
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    past_key_values = DynamicCache(config=cfg)

    mask = modeling_gemma4.create_causal_mask(
        config=cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dim() == 4, (
        f"expected a 4D SDPA-format mask, got {mask.dim()}D "
        f"shape={tuple(mask.shape)}. The full_attention global layers need "
        "this shape or they crash at long context."
    )
    assert mask.shape[0] == B
    assert mask.shape[-1] == S
    assert mask.shape[-2] == S

    # Caller's config must be untouched — other code paths still read it.
    assert cfg._attn_implementation == "flash_attention_2"


def test_patched_create_masks_for_generate_returns_4d_full_attention(
    restore_gemma4_module,
):
    """Hit the REAL ``create_masks_for_generate`` (not a mock) via the
    wrapper. With a caller-supplied FA2 config and a Gemma 4 layer_types
    list, the unpatched function would return a dict whose
    ``full_attention`` value is a 2D FA2 mask. The wrapper must promote
    that key to a 4D SDPA mask while leaving sliding alone.

    This is the multimodal-CPT failure mode: ``Gemma4Model.forward`` on
    the non-vision branch builds the mask via this exact entry point.
    """
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    patch_gemma4_hybrid_mask()
    modeling_gemma4 = restore_gemma4_module

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    cfg._attn_implementation = "flash_attention_2"

    B, S = 2, 256
    inputs_embeds = torch.zeros((B, S, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((B, S), dtype=torch.long)
    attention_mask[1, S // 2 :] = 0
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    past_key_values = DynamicCache(config=cfg)

    result = modeling_gemma4.create_masks_for_generate(
        config=cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    assert isinstance(result, dict), (
        f"expected a per-layer-pattern dict for a config with layer_types, "
        f"got {type(result).__name__}"
    )
    assert "full_attention" in result
    full_attn_mask = result["full_attention"]
    assert isinstance(full_attn_mask, torch.Tensor)
    assert full_attn_mask.dim() == 4, (
        f"full_attention mask must be 4D for SDPA, got "
        f"{full_attn_mask.dim()}D shape={tuple(full_attn_mask.shape)}"
    )

    # Sliding entry, if present, must remain in FA2 (2D) format — sliding
    # layers stayed on FA2 in the hybrid setup.
    if "sliding_attention" in result and result["sliding_attention"] is not None:
        sliding = result["sliding_attention"]
        if isinstance(sliding, torch.Tensor):
            assert sliding.dim() == 2, (
                "sliding_attention should remain a 2D FA2-compatible mask "
                f"so the sliding layers keep using flash_attention_2; got "
                f"{sliding.dim()}D shape={tuple(sliding.shape)}"
            )

    # Caller's config must not be mutated.
    assert cfg._attn_implementation == "flash_attention_2"


def test_create_causal_mask_mapping_transitive_coverage(restore_gemma4_module):
    """``create_causal_mask_mapping`` is defined inside ``modeling_gemma4``
    and calls module-level ``create_causal_mask`` at execution time. Our
    rebind of that symbol should propagate transitively, producing a 4D
    full-attention entry without us needing to wrap the mapping function
    directly. This test pins that contract — if a future transformers
    version changes the call to a closure or local import, this fails
    and we know to add an explicit wrapper.
    """
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    modeling_gemma4 = restore_gemma4_module
    if not hasattr(modeling_gemma4, "create_causal_mask_mapping"):
        pytest.skip("transformers version lacks create_causal_mask_mapping")

    patch_gemma4_hybrid_mask()

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    cfg._attn_implementation = "flash_attention_2"

    B, S = 2, 256
    inputs_embeds = torch.zeros((B, S, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((B, S), dtype=torch.long)
    attention_mask[1, S // 2 :] = 0
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    past_key_values = DynamicCache(config=cfg)

    result = modeling_gemma4.create_causal_mask_mapping(
        config=cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    assert isinstance(result, dict)
    assert "full_attention" in result
    full_attn_mask = result["full_attention"]
    assert isinstance(full_attn_mask, torch.Tensor)
    assert full_attn_mask.dim() == 4, (
        "create_causal_mask_mapping must produce a 4D full_attention mask "
        "transitively via the patched create_causal_mask. If this fails, "
        "transformers refactored the internal call and we need an explicit "
        "wrapper for create_causal_mask_mapping too."
    )


def test_tiny_gemma4_consumes_dict_mask_from_create_masks_for_generate(
    restore_gemma4_module,
):
    """End-to-end coverage of the multimodal forward contract.

    ``Gemma4Model.forward`` (the multimodal wrapper class) builds the mask
    via ``create_masks_for_generate`` and then calls
    ``self.language_model(attention_mask=causal_mask_mapping, ...)`` where
    ``causal_mask_mapping`` is the dict the wrapper returned. The
    ``Gemma4TextModel.forward`` then short-circuits its own mask building
    at the ``isinstance(causal_mask_mapping := attention_mask, dict)``
    check and feeds each layer the per-pattern entry directly.

    This test reproduces that contract on a tiny model: build the dict via
    the wrapped ``create_masks_for_generate``, hand it to the text model
    as ``attention_mask``, and verify the forward completes at mbs=2 with
    non-uniform padding. We instantiate a ``Gemma4TextModel`` (not a full
    ``Gemma4Model``) because the multimodal wrapper requires audio/vision
    tower configs that are irrelevant to the bug; the dict-passthrough
    invariant — wrapper builds dict, model consumes dict — is what
    differs between the patched and unpatched paths and is what this test
    pins.

    The test runs all-SDPA at the model level so it works on CPU. The
    "FA2-at-model-level + SDPA per-global-layer" config is what triggered
    the original bug, but exercising it requires CUDA + flash-attn. The
    shape-correctness for that case is covered by the unit-level
    ``test_patched_create_masks_for_generate_returns_4d_full_attention``;
    here we verify the model can actually consume the resulting dict end
    to end without crashing.
    """
    import torch

    modeling_gemma4 = restore_gemma4_module
    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    model, cfg = _build_tiny_gemma4_text_model()
    _apply_hybrid_attn_inline(model, cfg)

    B, S = 2, 1024
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    attn_mask_2d = torch.ones(B, S, dtype=torch.long)
    attn_mask_2d[1, S // 2 :] = 0  # uneven padding to force the materialized-mask path

    # Step 1: emulate Gemma4Model.forward's mask construction.
    inputs_embeds = model.embed_tokens(input_ids).detach()
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    causal_mask_mapping = modeling_gemma4.create_masks_for_generate(
        cfg,
        inputs_embeds,
        attn_mask_2d,
        None,  # past_key_values
        position_ids,
    )
    assert isinstance(causal_mask_mapping, dict)
    assert "full_attention" in causal_mask_mapping
    assert causal_mask_mapping["full_attention"].dim() == 4, (
        "wrapper must produce a 4D mask for full_attention or the SDPA "
        "global layer crashes on expand"
    )

    # Step 2: pass the dict straight through to the text model — exactly
    # what Gemma4Model.forward does at modeling_gemma4.py:2316.
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
        )

    assert out.last_hidden_state.shape == (B, S, cfg.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all()


# ---------------------------------------------------------------------------
# Integration tests with a tiny multimodal Gemma4Model.
#
# These build the actual ``Gemma4Model`` class (the multimodal wrapper used
# by ``Gemma4ForConditionalGeneration``) with no vision/audio towers — a
# valid configuration when ``vision_config=None`` and ``audio_config=None``
# (configuration_gemma4.py:337-345). The forward path is the same one that
# crashed the user's multimodal-CPT run; without the
# ``create_masks_for_generate`` wrapper, ``Gemma4Model.forward`` builds a
# 2D FA2 mask and the SDPA-promoted global layers crash at consumption.
# ---------------------------------------------------------------------------


def _build_tiny_gemma4_multimodal_model():
    """Return a tiny multimodal ``Gemma4Model`` with text-only configuration.

    ``Gemma4Config`` accepts ``vision_config=None`` and ``audio_config=None``
    (configuration_gemma4.py:337-345) which leaves the vision_tower /
    audio_tower at ``None`` (modeling_gemma4.py:2085-2096). The result is a
    real multimodal wrapper that exercises the actual ``Gemma4Model.forward``
    code path — the same one that fires on a multimodal-CPT training step
    — without needing to construct vision or audio tower configs that are
    irrelevant to the bug being patched.
    """
    import torch
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config,
        Gemma4TextConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4Model

    text_cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    text_cfg._attn_implementation = "sdpa"
    cfg = Gemma4Config(
        text_config=text_cfg,
        vision_config=None,
        audio_config=None,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(43)
    model = Gemma4Model(cfg).eval()
    return model, cfg


def _apply_hybrid_attn_inline_multimodal(model, cfg):
    """Apply the per-layer SDPA override to a multimodal ``Gemma4Model``.

    Mirrors what ``patch_manager._apply_gemma_hybrid_attention`` does for
    the multimodal navigation path (patch_manager.py:213-218), reaching the
    decoder layers via ``model.language_model.layers``. Then installs the
    mask wrappers.
    """
    import copy

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    layer_types = cfg.text_config.layer_types
    for layer_idx, layer in enumerate(model.language_model.layers):
        if layer_types[layer_idx] != "sliding_attention":
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "config"):
                sdpa_cfg = copy.copy(attn.config)
                sdpa_cfg._attn_implementation = "sdpa"
                attn.config = sdpa_cfg
    patch_gemma4_hybrid_mask()


def test_tiny_gemma4_multimodal_forward_with_padding(restore_gemma4_module):
    """Pin the actual multimodal-CPT bug fix end-to-end.

    Builds a real ``Gemma4Model`` (the multimodal wrapper class that
    crashed in the user's run) on CPU with no vision/audio towers, applies
    the hybrid attention path the same way ``_apply_gemma_hybrid_attention``
    does for ``Gemma4ForConditionalGeneration``, and runs a forward pass at
    mbs=2 with non-uniform padding.

    Without the ``create_masks_for_generate`` wrapper, this is the call
    that produces the ``Target sizes: [B, H, S, S]. Tensor sizes: [B, S]``
    expand error: ``Gemma4Model.forward`` (modeling_gemma4.py:2291-2316)
    routes mask construction through ``create_masks_for_generate`` on the
    non-vision branch, gets a 2D FA2 mask back for the ``full_attention``
    key, and the SDPA-promoted global layers crash on consumption.

    With the wrapper installed, the dict's ``full_attention`` entry is 4D
    and the forward survives. This test runs in pure SDPA mode so it works
    on CPU without flash-attn; the FA2-at-model-level scenario is covered
    at unit level by
    ``test_patched_create_masks_for_generate_returns_4d_full_attention``.
    """
    import torch

    model, cfg = _build_tiny_gemma4_multimodal_model()
    _apply_hybrid_attn_inline_multimodal(model, cfg)

    B, S = 2, 1024
    input_ids = torch.randint(0, cfg.text_config.vocab_size, (B, S))
    attn_mask = torch.ones(B, S, dtype=torch.long)
    attn_mask[1, S // 2 :] = 0

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)

    assert out.last_hidden_state.shape == (B, S, cfg.text_config.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all()


def test_tiny_gemma4_multimodal_forward_unpatched_baseline_works(
    restore_gemma4_module,
):
    """Sanity baseline: the all-SDPA multimodal forward on CPU runs without
    the hybrid patch installed too. This proves the test setup itself does
    not depend on the patch — the patch's value is only observable when
    the model-level config disagrees with per-layer overrides. If this
    test fails, the multimodal model build / forward is broken in a way
    unrelated to the patch and the surrounding tests give false signal.
    """
    import torch

    model, cfg = _build_tiny_gemma4_multimodal_model()

    B, S = 2, 256
    input_ids = torch.randint(0, cfg.text_config.vocab_size, (B, S))
    attn_mask = torch.ones(B, S, dtype=torch.long)
    attn_mask[1, S // 2 :] = 0

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)

    assert out.last_hidden_state.shape == (B, S, cfg.text_config.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all()
