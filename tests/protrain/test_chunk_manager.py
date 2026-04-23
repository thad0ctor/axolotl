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

    # Pick an S_chunk large enough for each block (and every single param)
    # but smaller than the whole model so we actually get multiple chunks.
    # For the tiny GPT-2 here each block is ~200 KB and total is ~437 KB,
    # so S_chunk just above max(block_bytes) guarantees the block fits in
    # one chunk while forcing at least two chunks overall.
    block_bytes_each = []
    named = dict(model.named_parameters())
    for pids in block_spans.values():
        block_bytes = 0
        for pid in pids:
            param = named[pid]
            block_bytes += param.numel() * param.element_size()
        block_bytes_each.append(block_bytes)
    max_param_bytes = max(p.numel() * p.element_size() for p in named.values())
    # Ensure S_chunk fits the largest single param and any single block, with
    # a modest safety margin, yet is strictly less than ``total_bytes``.
    S_chunk = max(max(block_bytes_each), max_param_bytes) + 1024

    # Safety: S_chunk should be < total so we actually get multiple chunks.
    assert S_chunk < total_bytes, (
        f"test setup: S_chunk={S_chunk} must be < total_bytes={total_bytes} "
        "to exercise multi-chunk layout"
    )

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
    """Grid-search chooses the minimum-waste candidate, tie-breaking to the larger S.

    The algorithm (Appendix B.1) simulates greedy-fit chunking for each
    candidate in {32, 64, 128, 256} MB and picks the S_chunk that minimizes
    the sum of ``S_chunk - bytes_used`` across every *non-tail* chunk.
    Overfilled chunks (a single param larger than S) contribute zero waste
    because the clamp ``max(0, S - bytes)`` floors negatives to zero. Ties
    are broken by picking the *larger* candidate — fewer chunks ⇒ fewer
    scheduler iterations.
    """
    from axolotl.integrations.protrain.chunk.sizing import pick_S_chunk

    MB = 1 << 20

    # Case A — oversized-param regime. 8 × 63 MB params: at S=32 every param
    # overflows its chunk (63 > 32) so waste clamps to 0, which becomes the
    # global minimum. At S=64 each 63 MB param sits alone in a chunk leaving
    # 1 MB of trailing slack × 7 preceding chunks = 7 MB of waste. At S=128
    # pairs fit (2*63=126 ≤ 128) → 4 chunks, 3 preceding × 2 MB = 6 MB
    # waste. At S=256 quadruples fit → 2 chunks, 1 preceding × 4 MB = 4 MB.
    # So S=32 (waste 0) strictly wins; S=256 is the runner-up.
    sizes_a: dict[ParamId, int] = {
        cast(ParamId, f"p{i}"): 63 * MB for i in range(8)
    }
    picked_a = pick_S_chunk(sizes_a)
    assert picked_a == 32 * MB, (
        f"overflow-clamp scenario: expected S=32 MB (waste=0); got {picked_a}"
    )

    # Case B — exact-fit regime with an all-tied waste profile. 4 × 64 MB
    # params: at S=32 each overflows (waste=0); at S=64 each fills a chunk
    # exactly (all preceding chunks have waste=0); at S=128 pairs fit
    # exactly (waste=0); at S=256 all four fit in a single chunk (waste=0
    # since tail slack is excluded). Every candidate ties at 0 waste, so
    # the tie-break rule ("prefer larger S_chunk") selects 256 MB.
    sizes_b: dict[ParamId, int] = {
        cast(ParamId, f"q{i}"): 64 * MB for i in range(4)
    }
    picked_b = pick_S_chunk(sizes_b)
    assert picked_b == 256 * MB, (
        f"tie-at-zero-waste scenario: expected S=256 MB via tie-break; got {picked_b}"
    )

    # Case C — mid-grid winner. Construct a layout where S=128 MB is
    # strictly minimum-waste. Use 3 × 100 MB params: at S=32 each overflows
    # (waste=0 via clamp); at S=64 each overflows (100 > 64, waste=0); at
    # S=128 each fills one chunk leaving 28 MB preceding-slack × 2 chunks =
    # 56 MB; at S=256 pairs fit (200 ≤ 256) so [200][100] — waste =
    # 256-200 = 56 MB preceding. Ties between 32/64 at 0 and between 128/
    # 256 at 56; the zero-waste bucket wins, and within it S=64 beats S=32
    # by tie-break. So the *overall* pick is S=64 MB.
    sizes_c: dict[ParamId, int] = {
        cast(ParamId, f"r{i}"): 100 * MB for i in range(3)
    }
    picked_c = pick_S_chunk(sizes_c)
    assert picked_c == 64 * MB, (
        f"mixed-waste scenario: expected S=64 MB (waste=0, larger of the "
        f"two zero-waste candidates); got {picked_c}"
    )

    # Sanity — every pick is drawn from the documented grid.
    for picked in (picked_a, picked_b, picked_c):
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
