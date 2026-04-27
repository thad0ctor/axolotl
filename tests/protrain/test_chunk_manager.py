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


def test_param_exec_order_follows_trace_op_stream_not_declaration_order():
    """Exec order is derived from ``trace.op_order`` (§3.1.1), not param declaration.

    Build a 2-block model that *registers* its blocks in one order
    (``b`` then ``a``) but *executes* them in the opposite order
    (``a`` then ``b``) on the forward pass. The trace-driven helper
    must emit ``a``'s param before ``b``'s, so the gather pattern lines
    up with the actual op stream rather than the storage order.
    """
    pytest.importorskip("torch")

    import torch
    from torch import nn

    from axolotl.integrations.protrain.api.model_wrapper import (
        _param_exec_order,
    )
    from axolotl.integrations.protrain.types import OpId, OpRecord

    class FlippedOrder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Registration order: b first, then a — opposite to forward order.
            self.b = nn.Linear(4, 4, bias=False)
            self.a = nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Execution order: a first, then b.
            return self.b(self.a(x))

    model = FlippedOrder()

    # Sanity: declaration order really is (b, a).
    declared = [n for n, _ in model.named_parameters()]
    assert declared == ["b.weight", "a.weight"], (
        f"test setup invariant broken: declared order is {declared}; "
        "expected ['b.weight', 'a.weight'] so a trace-driven order can "
        "differ from declaration order"
    )

    # Synthesize a minimal trace whose op_order reflects forward order.
    # build_layout doesn't care about non-module-path fields, but we
    # still construct a valid OpRecord for each step.
    def _op(op_id: int, mod_path: str) -> OpRecord:
        return OpRecord(
            op_id=cast(OpId, op_id),
            module_path=mod_path,
            qualified_name="aten::linear",
            shape_signature=((1, 4),),
            block_id=None,
            is_forward=True,
        )

    class FakeTrace:
        op_order = (_op(0, "a"), _op(1, "b"))

    # _param_exec_order ignores block_spans (block grouping happens in
    # build_layout); pass an empty mapping to avoid invoking
    # discover_blocks on this non-transformer toy model.
    exec_order = _param_exec_order(model, {}, FakeTrace())

    assert exec_order == [
        cast(ParamId, "a.weight"),
        cast(ParamId, "b.weight"),
    ], (
        f"trace-driven exec order should be (a, b) — the forward order — "
        f"got {exec_order}"
    )

    # And the layout chunks must reflect the same order.
    from axolotl.integrations.protrain.chunk.layout import build_layout

    layout = build_layout(model, exec_order, S_chunk=1 << 20, block_spans={})
    flat = [pid for chunk in layout.chunks for pid in chunk]
    a_idx = flat.index(cast(ParamId, "a.weight"))
    b_idx = flat.index(cast(ParamId, "b.weight"))
    assert a_idx < b_idx, (
        f"layout still walks declaration order: a@{a_idx} b@{b_idx}; "
        "expected a before b to match forward op stream"
    )


def test_param_exec_order_dedups_weight_tied_params():
    """A tied weight visited twice in the trace keeps only the first slot."""
    pytest.importorskip("torch")

    import torch
    from torch import nn

    from axolotl.integrations.protrain.api.model_wrapper import (
        _param_exec_order,
    )
    from axolotl.integrations.protrain.types import OpId, OpRecord

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(4, 4, bias=False)
            self.second = nn.Linear(4, 4, bias=False)
            self.second.weight = self.first.weight  # tie

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.second(self.first(x))

    model = Tied()

    def _op(op_id: int, mod_path: str) -> OpRecord:
        return OpRecord(
            op_id=cast(OpId, op_id),
            module_path=mod_path,
            qualified_name="aten::linear",
            shape_signature=((1, 4),),
            block_id=None,
            is_forward=True,
        )

    class FakeTrace:
        # second uses the SAME tensor as first; the second op should not
        # introduce a duplicate slot.
        op_order = (_op(0, "first"), _op(1, "second"))

    exec_order = _param_exec_order(model, {}, FakeTrace())

    # named_parameters dedups by tensor identity, exposing the tied
    # weight under its first registered name (``first.weight``).
    assert exec_order.count(cast(ParamId, "first.weight")) == 1
    assert cast(ParamId, "second.weight") not in exec_order


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
def test_loss_parity_n_persist_extremes():
    """Loss values must match between pure-GPU and pure-offload modes.

    End-to-end correctness check that ProTrain's chunk-offload paths do
    not perturb training math. Run 5 steps of a tiny GPT-2 twice with
    identical seeds and batches:

    * Config A: ``n_persist = N_chunk`` (every chunk stays on GPU; no
      offload, no prefetch).
    * Config B: ``n_persist = 0`` (pure offload; every chunk H2D/D2H-
      transits the PCIe bus each iteration).

    The per-step loss trajectories must match to fp16-noise tolerance
    (``|loss_a[i] - loss_b[i]| < 5e-2``) — optimizer math is the same in
    both cases; only the physical residency of params differs.
    """
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    device = torch.device("cuda")
    gpt2_cfg = GPT2Config(
        n_layer=2, n_head=2, n_embd=64, vocab_size=128, n_positions=16
    )

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(device),
        gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        gpu_count=1,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        has_nvlink=False,
    )

    bs, seq = 1, 8
    # Shared batches — generated once so both configs see the same data.
    torch.manual_seed(123)
    batches = [
        {
            "input_ids": torch.randint(
                0, gpt2_cfg.vocab_size, (bs, seq), device=device, dtype=torch.long
            ),
        }
        for _ in range(5)
    ]
    for b in batches:
        b["labels"] = b["input_ids"].clone()

    def _run_config(n_persist_mode: str) -> list[float]:
        """Run 5 steps and return per-step losses."""
        import gc

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        # Deterministic init.
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = GPT2LMHeadModel(gpt2_cfg).to(device)

        if n_persist_mode == "all":
            # force_all_persistent synthesizes n_persist=N_chunk, which is
            # the "pure GPU" config we want here. It also enables CKPT on
            # every block — we don't want that for the math-parity test
            # because CKPT's recompute can swing fp32 activations by a ulp
            # and we need <5e-2 tolerance. Use explicit override instead.
            probe = protrain_model_wrapper(
                model,
                model_config=gpt2_cfg,
                hardware_profile=hw,
                batch_size=bs,
                seq_len=seq,
                capacity_bytes=2 * (1 << 30),
                force_all_persistent=True,
            )
            n_chunk = probe.chunk_manager.layout.N_chunk
            # Tear down and rebuild without CKPT.
            for h in probe._hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            del probe
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            model = GPT2LMHeadModel(gpt2_cfg).to(device)
            wrapped = protrain_model_wrapper(
                model,
                model_config=gpt2_cfg,
                hardware_profile=hw,
                batch_size=bs,
                seq_len=seq,
                capacity_bytes=2 * (1 << 30),
                n_persist_override=n_chunk,
                n_buffer_override=max(1, n_chunk),
                n_swap_override=0,
                n_checkpoint_override=0,
            )
        elif n_persist_mode == "none":
            # Full offload — need N_chunk. Probe first.
            probe = protrain_model_wrapper(
                model,
                model_config=gpt2_cfg,
                hardware_profile=hw,
                batch_size=bs,
                seq_len=seq,
                capacity_bytes=2 * (1 << 30),
                force_all_persistent=True,
            )
            n_chunk = probe.chunk_manager.layout.N_chunk
            for h in probe._hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            del probe
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            model = GPT2LMHeadModel(gpt2_cfg).to(device)
            # n_persist=0, still no CKPT so the math matches A exactly.
            wrapped = protrain_model_wrapper(
                model,
                model_config=gpt2_cfg,
                hardware_profile=hw,
                batch_size=bs,
                seq_len=seq,
                capacity_bytes=2 * (1 << 30),
                n_persist_override=0,
                n_buffer_override=max(2, n_chunk),
                n_swap_override=0,
                n_checkpoint_override=0,
            )
        else:
            raise AssertionError(f"unknown mode {n_persist_mode!r}")

        optim = protrain_optimizer_wrapper(wrapped, lr=1e-4)

        losses: list[float] = []
        for batch in batches:
            out = wrapped.module(**batch)
            out.loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(float(out.loss.detach()))

        # Teardown.
        for h in wrapped._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        del wrapped, model, optim
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        return losses

    losses_all = _run_config("all")
    losses_none = _run_config("none")

    print(f"\nloss trajectory (n_persist=N_chunk):  {losses_all}")
    print(f"loss trajectory (n_persist=0):        {losses_none}")

    assert len(losses_all) == len(losses_none) == 5
    for i, (a, b) in enumerate(zip(losses_all, losses_none)):
        assert abs(a - b) < 5e-2, (
            f"loss divergence at step {i}: n_persist=N_chunk->{a:.6f} "
            f"vs n_persist=0->{b:.6f} (|Δ|={abs(a-b):.6f})"
        )
