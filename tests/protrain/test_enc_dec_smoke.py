"""T5 encoder-decoder E2E smoke test for ProTrain — Item 9 cell B.

Item 8's ``batch_factory`` adds a ``seq2seq_lm`` factory and is covered
by ``test_batch_factory.py`` for shape contracts and CPU-only
forward+backward, but no test drives a real encoder-decoder model
end-to-end through ``protrain_model_wrapper``. The encoder-decoder
block discovery (``block.layout_rules.discover_blocks``) has never been
tested against a model with two transformer trees (encoder + decoder).

**Real finding (documented gap, not a test fudge):**

``discover_blocks`` does NOT support T5-family encoder-decoder models
on this branch. The function searches a fixed list of dotted paths
(``transformer.h``, ``model.layers``, ``transformer.layers``,
``base_model.layers``, ``base_model.model.model.layers``,
``base_model.model.transformer.h``) and falls back to a heuristic that
flags an ``nn.ModuleList`` whose children expose either an
``attention`` or ``self_attn`` direct attribute.

T5's structure violates both checks:

1. **Dotted paths.** T5 stores its transformer blocks at
   ``encoder.block`` and ``decoder.block`` — neither path is in
   ``_KNOWN_BLOCK_PATHS``, and even if one were, the discovery
   contract is "return the first matching ModuleList" so a single
   call cannot return both encoder and decoder blocks.
2. **Attention heuristic.** ``T5Block`` does not have ``attention``
   or ``self_attn`` as a direct attribute. Its sub-modules live
   inside a nested ``T5Block.layer`` ``nn.ModuleList`` whose elements
   are ``T5LayerSelfAttention`` / ``T5LayerCrossAttention`` /
   ``T5LayerFF``. ``_looks_like_block`` does not look one level
   deeper, so the heuristic also misses.

Net result: ``discover_blocks(t5_model)`` raises ``RuntimeError``,
which means ``protrain_model_wrapper`` cannot wrap a T5 model on the
current branch. Adding T5 support requires either expanding
``_KNOWN_BLOCK_PATHS`` to include ``encoder.block`` /
``decoder.block`` AND extending the discovery contract to return
multiple block trees, or expanding ``_looks_like_block`` to recognise
T5Block-style nested layer ModuleLists. Both are out of scope for the
v1 validation matrix add — the test below skips loudly and the seq2seq
LM factory's CPU-only forward+backward in ``test_batch_factory.py``
remains the only enc-dec coverage in v1.

This file ships the skip rather than excising the test so the gap is
discoverable in the test runner output (``SKIPPED [reason]``) rather
than buried in design notes.
"""

from __future__ import annotations

import math

import pytest


def _build_tiny_t5():
    """Construct a fresh-init tiny T5 — same shape as in test_batch_factory.

    Module-local helper so the skip path below can still import its
    way to the model when the discover_blocks check is being exercised.
    """
    from transformers import T5Config, T5ForConditionalGeneration

    cfg = T5Config(
        d_model=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=256,
        d_kv=32,
        vocab_size=128,
        decoder_start_token_id=0,
        pad_token_id=0,
    )
    return cfg, T5ForConditionalGeneration(cfg)


def test_protrain_enc_dec_smoke_t5() -> None:
    """T5-small enc-dec smoke: wrap + 3 iters; document discover_blocks gap.

    Two-stage acceptance:

    1. Pre-flight check: confirm ``discover_blocks`` rejects the T5
       model, which is what causes ``protrain_model_wrapper`` to fail
       on encoder-decoder topologies. If that check ever starts
       PASSING (i.e. discover_blocks gains T5 support), this test will
       skip with a different reason and the developer should remove
       the skip and let the real wrap path exercise.
    2. End-to-end (only if step 1 succeeds): wrap with ProTrain Mode-A,
       run 3 forward+backward+step iters on a fixed batch, assert
       finite loss + chunk discovery accepted both block trees.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain enc-dec smoke requires CUDA.")

    from axolotl.integrations.protrain.block.layout_rules import discover_blocks
    from axolotl.integrations.protrain.profiler.batch_factory import (
        TASK_SEQ2SEQ_LM,
        build_batch,
        detect_task_type,
    )

    cfg, model = _build_tiny_t5()

    # batch_factory must already classify this as seq2seq — that's the
    # part Item 8 covers and we re-assert it here so this test fails
    # loudly if a future refactor breaks task detection on T5.
    assert detect_task_type(model) == TASK_SEQ2SEQ_LM, (
        "T5ForConditionalGeneration must be detected as seq2seq_lm — "
        "the batch_factory path depends on it."
    )

    # Pre-flight: try discover_blocks on the bare T5 model. The
    # expected outcome on this branch is RuntimeError — documenting
    # the gap (see module docstring). If the call ever succeeds,
    # branch into the wrap path to keep the test useful as the gap
    # closes.
    discover_failure: str | None = None
    try:
        blocks = discover_blocks(model)
    except RuntimeError as exc:  # noqa: BLE001
        discover_failure = str(exc)
        blocks = None

    if discover_failure is not None:
        # Sanity: the encoder + decoder blocks really are present on
        # the model — the gap is in discover_blocks, not in the model.
        assert hasattr(model, "encoder") and hasattr(model, "decoder"), (
            "T5 model unexpectedly missing encoder/decoder; test fixture "
            "may be wrong"
        )
        assert len(model.encoder.block) > 0 and len(model.decoder.block) > 0, (
            "T5 model has empty encoder.block or decoder.block — "
            "fixture build is wrong"
        )

        # Also exercise the seq2seq batch_factory path on CPU so this
        # test contributes positive coverage even when the wrap path
        # is unsupported. Mirrors the assertions in
        # test_batch_factory but on this exact model — the v1 fast
        # lane only ever sees the GPT-2 / BERT shapes there.
        batch = build_batch(model, batch_size=2, seq_len=8, device="cpu")
        assert set(batch.keys()) >= {"input_ids", "labels"}
        assert batch["labels"].shape == (2, 8)
        out = model(**batch)
        assert out.loss is not None
        assert torch.isfinite(out.loss).item()
        out.loss.backward()

        pytest.skip(
            "T5 enc-dec block discovery: discover_blocks rejects T5 — "
            "encoder.block/decoder.block dotted paths are not in "
            "_KNOWN_BLOCK_PATHS, and T5Block's attention modules sit "
            "one level deep inside T5Block.layer (a nested ModuleList) "
            "so the attention/self_attn heuristic also misses. Adding "
            "T5 support requires extending discover_blocks to return "
            "multiple block trees AND recognising T5Block-style nested "
            "layer ModuleLists. CPU-only batch_factory + bare-model "
            "forward+backward exercised above. "
            f"Underlying error: {discover_failure}"
        )

    # ---- discover_blocks accepted T5 (future state) --------------------
    # If we reach here the gap has closed and discover_blocks returned
    # a non-empty list of T5Block-or-equivalent modules. Drive the
    # full ProTrain wrap + 3 iters.
    assert blocks is not None and len(blocks) > 0, (
        "discover_blocks returned an empty list for T5 — protocol "
        "violation: it should raise RuntimeError on no match."
    )

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    cfg.use_cache = False
    device = torch.device("cuda:0")
    model = model.to(device).to(dtype=torch.bfloat16)

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
    )
    bs, seq = 2, 16
    wrapped = protrain_model_wrapper(
        model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
        capacity_bytes=20 * (1 << 30),
        force_all_persistent=True,
    )
    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)

    vocab = int(getattr(cfg, "vocab_size", 128))
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (bs, seq), device=device, dtype=torch.long)
    attention_mask = torch.ones((bs, seq), device=device, dtype=torch.long)
    labels = torch.randint(0, vocab, (bs, seq), device=device, dtype=torch.long)

    losses: list[float] = []
    for i in range(3):
        out = wrapped.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss_value = float(out.loss.detach())
        assert math.isfinite(loss_value), (
            f"iter {i}: non-finite loss {loss_value}"
        )
        out.loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss_value)

    print(f"\nProTrain enc-dec smoke (T5-tiny): losses={losses}")
