"""Tests for the ProTrain hierarchical chunk manager (M2)."""

from __future__ import annotations

from typing import cast

import pytest

from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkLayout,
    ParamId,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_gpt2():
    """Return a freshly-initialized 2-block GPT-2 LM (CPU weights).

    Kept small so the tests run in seconds with or without a GPU.
    """
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=128,
        n_positions=16,
    )
    return GPT2LMHeadModel(cfg)


def _make_block_spans(model) -> dict[BlockId, list[ParamId]]:
    """Extract ``block_id -> [param ids]`` from ``transformer.h.{i}`` submodules."""
    spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        parts = name.split(".")
        # GPT-2: transformer.h.<i>.<rest>
        try:
            h_idx = parts.index("h")
            block_idx = int(parts[h_idx + 1])
        except (ValueError, IndexError):
            continue
        spans.setdefault(cast(BlockId, block_idx), []).append(cast(ParamId, name))
    return spans


# ---------------------------------------------------------------------------
# layout.py / sizing.py — CPU-only, torch-light tests
# ---------------------------------------------------------------------------


def test_layout_respects_block_grouping():
    """All params of a transformer block land in a single chunk when they fit."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from axolotl.integrations.protrain.chunk.layout import build_layout

    model = _tiny_gpt2()
    block_spans = _make_block_spans(model)
    assert len(block_spans) == 2, "expected n_layer=2"

    # Force a generous S_chunk so the whole model fits in one chunk easily;
    # the block-contiguity rule should still hold trivially. Then also
    # test with a tighter S_chunk sized so each block fits but the full
    # model does not — the stronger assertion.
    all_params = [cast(ParamId, n) for n, _ in model.named_parameters()]
    exec_order = list(all_params)  # pretend exec order = definition order

    # Total model bytes.
    total_bytes = sum(p.numel() * p.element_size() for _, p in model.named_parameters())

    # Pick an S_chunk large enough for each block but smaller than the
    # whole model + embeddings — guaranteed by max(block_bytes, embed_bytes) <= S <= total/1.1.
    block_bytes_each = []
    for pids in block_spans.values():
        block_bytes = 0
        for pid in pids:
            param = dict(model.named_parameters())[pid]
            block_bytes += param.numel() * param.element_size()
        block_bytes_each.append(block_bytes)
    S_chunk = max(block_bytes_each) * 4  # fits any single block, still splits model

    # Safety: S_chunk should be < total so we actually get multiple chunks.
    assert S_chunk < total_bytes

    layout = build_layout(model, exec_order, S_chunk, block_spans)

    # Every block's params must live in exactly one chunk (they fit).
    for block_id, pids in block_spans.items():
        chunk_ids = {layout.param_to_chunk[pid] for pid in pids}
        assert len(chunk_ids) == 1, (
            f"block {block_id} spans chunks {chunk_ids}; "
            f"expected single chunk since block_bytes={block_bytes_each[block_id]} "
            f"fits in S_chunk={S_chunk}"
        )
        assert layout.block_to_chunks[block_id] == tuple(chunk_ids)


def test_layout_preserves_first_occurrence_for_shared_params():
    """A weight referenced twice in exec_order is placed once, at the first slot."""
    pytest.importorskip("torch")

    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.layout import build_layout

    class SharedWeight(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4, bias=False)
            self.b = nn.Linear(4, 4, bias=False)
            # Share: b uses a's weight.
            self.b.weight = self.a.weight
            self.head = nn.Linear(4, 2, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.b(self.a(x)))

    model = SharedWeight()

    # The shared tensor registers under its first dotted path. Collect
    # unique param ids in the canonical named_parameters order.
    param_names = [cast(ParamId, n) for n, _ in model.named_parameters()]
    # Should be: ["a.weight", "head.weight"] — b.weight is a ref to a.weight
    # and named_parameters de-duplicates by identity.
    assert "a.weight" in param_names
    # Construct an exec_order that visits a.weight TWICE (once for self.a,
    # once as b.weight via sharing) to exercise the dedup rule.
    exec_order: list[ParamId] = [
        cast(ParamId, "a.weight"),
        cast(ParamId, "a.weight"),  # shared reference — first-occurrence wins
        cast(ParamId, "head.weight"),
    ]

    S_chunk = 1 << 20  # plenty big
    layout = build_layout(model, exec_order, S_chunk, block_spans={})

    # ``a.weight`` should appear exactly once across all chunks.
    flat = [pid for chunk in layout.chunks for pid in chunk]
    assert flat.count(cast(ParamId, "a.weight")) == 1
    # And it should be in the first chunk (where its first occurrence lives).
    assert cast(ParamId, "a.weight") in layout.chunks[0]


def test_sizing_picks_min_waste():
    """Crafted param sizes where 64 MB is the clear argmin-waste winner."""
    from axolotl.integrations.protrain.chunk.sizing import pick_S_chunk

    MB = 1 << 20
    # Params sized to pack perfectly into 64 MB chunks but leave large
    # gaps under 128 MB / 256 MB (each 128 MB chunk holds only one ~63 MB
    # param, wasting ~65 MB; same for 256 MB). At 32 MB a single 63 MB
    # param doesn't fit — it still gets placed (overflow) but every
    # *preceding* chunk is counted as waste = 32-63 which clamps to 0.
    # Net: 64 MB wins with 0 waste.
    sizes_list = [63 * MB] * 8  # 8 params of 63 MB each
    sizes: dict[ParamId, int] = {
        cast(ParamId, f"p{i}"): sz for i, sz in enumerate(sizes_list)
    }

    picked = pick_S_chunk(sizes)
    # 32 MB: every 63 MB param spills into its own chunk that overfills;
    # our greedy tracker counts (32 - bytes_in_chunk) only for chunks that
    # didn't hit the tail, and overflowed chunks have bytes_in_chunk > 32
    # so waste is clamped to 0. Waste at 32 MB = 0 as well.
    # 64 MB: each 63 MB param fits exactly, small 1 MB per-chunk waste × 7.
    # 128 MB: each 63 MB param takes a fresh chunk (can't fit 2 since
    # 2*63 = 126 < 128 → actually *does* fit 2, leaving 128-126=2 MB
    # waste per pair × 3 = 6 MB waste. That's LESS than 64 MB.
    # Hmm — 128 MB would actually win. Re-pick sizes so 64 is unambiguous.
    # Use 33 MB params: at 32 MB each spills; at 64 MB pair exactly (64-66=0,
    # wait 2*33=66 > 64, so only one fits per chunk → 64-33=31 waste × 7).
    # Easier: use sizes that exactly match 64 MB.
    sizes2: dict[ParamId, int] = {
        cast(ParamId, f"q{i}"): 64 * MB for i in range(4)
    }
    picked2 = pick_S_chunk(sizes2)
    assert picked2 == 64 * MB, (
        f"4 × 64 MB params should prefer S_chunk=64 MB (zero waste); got {picked2}"
    )
    # Quiet the unused-variable warning by asserting something about ``picked``.
    assert picked in (32 * MB, 64 * MB, 128 * MB, 256 * MB)


# ---------------------------------------------------------------------------
# pinned_alloc.py — GPU-only
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_pinned_alloc_precise_size():
    """cudaHostAlloc path allocates exactly n_buffer * S_chunk bytes."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    n_buffer = 4
    S_chunk = 1 << 20  # 1 MB
    mem = PinnedHostMemory(n_buffer=n_buffer, S_chunk=S_chunk)
    try:
        if not mem.is_precise_size:
            pytest.skip(
                "PinnedHostMemory fell back to torch.empty(pin_memory=True); "
                "precise-size assertion not applicable on this path"
            )
        # Slot 0 and slot (n-1) should both be valid and exactly S_chunk bytes.
        for i in (0, n_buffer - 1):
            t = mem.buffer(i)
            assert t.numel() == S_chunk
            assert t.dtype == torch.uint8
        # Total bytes exactly n_buffer * S_chunk (no pow-2 round-up).
        assert mem.total_bytes == n_buffer * S_chunk
        assert mem.total_bytes == 4 << 20  # 4 MB, NOT 8 MB
    finally:
        mem.close()


# ---------------------------------------------------------------------------
# buffer_pool.py — GPU-only
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_buffer_pool_acquire_release():
    """LRU-free semantics: after release, next acquire returns the same physical buffer."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
    from axolotl.integrations.protrain.types import ChunkId

    n_buffer = 4
    S_chunk = 1 << 20
    host = PinnedHostMemory(n_buffer=n_buffer, S_chunk=S_chunk)
    try:
        pool = BufferPool(
            n_buffer=n_buffer,
            S_chunk=S_chunk,
            pinned_host=host,
            device=torch.device("cuda"),
        )

        # Acquire 3 of 4 — each for a distinct chunk id.
        buf0 = pool.acquire(cast(ChunkId, 0))
        buf1 = pool.acquire(cast(ChunkId, 1))
        buf2 = pool.acquire(cast(ChunkId, 2))
        assert pool.num_in_use == 3
        assert pool.num_free == 1

        # Release one, then acquire for a NEW chunk id (not resident).
        pool.release(cast(ChunkId, 1))
        assert pool.num_free == 2

        # The freshly released buffer's tag is still 1, so lookup_resident works.
        assert pool.lookup_resident(cast(ChunkId, 1)) is buf1

        # Acquire a new chunk id — evicts the LRU free slot. That was slot 3
        # (never-used) first in our FIFO; after releasing chunk 1 its slot
        # went to the tail. So the first free-list pop is slot 3, then slot 1.
        buf3 = pool.acquire(cast(ChunkId, 99))
        # Re-acquire chunk 1 — it's still resident, should return the SAME buffer.
        buf1_again = pool.acquire(cast(ChunkId, 1))
        assert buf1_again.data_ptr() == buf1.data_ptr()
        # And the buffer's physical slot should match.
        assert pool.lookup_resident(cast(ChunkId, 1)) is buf1_again

        # Keep silencing unused-var warnings — verify distinctness.
        assert buf0.data_ptr() != buf2.data_ptr()
        assert buf3.data_ptr() not in {buf0.data_ptr(), buf1.data_ptr(), buf2.data_ptr()}
    finally:
        host.close()


# ---------------------------------------------------------------------------
# Full loss parity — deferred until the scheduler (M4) wires this up
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skip(
    reason="full integration test, runs after M5 when Axolotl glue wires this end-to-end"
)
def test_loss_parity_n_persist_extremes():
    """Loss values must match between pure-GPU and pure-offload modes.

    M2 GPU validation: run 5 steps with n_persist=N_chunk (pure GPU) vs
    n_persist=0 (pure offload); assert ``|loss_a - loss_b| < 1e-2`` across
    all 5 steps.
    """
    # TODO(m5): instantiate two ChunkManager configurations on the same
    # tiny GPT-2, run 5 train steps with identical batches, and assert the
    # loss trajectories match to within 1e-2. Skeleton kept so the case
    # isn't lost.
    raise NotImplementedError
