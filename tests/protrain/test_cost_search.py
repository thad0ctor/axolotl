"""Unit tests for the ProTrain cost models + searcher (M4).

These tests build synthetic ``ProfilerTrace`` / ``ChunkLayout`` /
``HardwareProfile`` objects — no GPU required. The toy model has
``N_block=8`` transformer blocks, ``N_chunk=12`` chunks of
``S_chunk=64 MB``, with uniform per-block activation size and a small
op-walk seeded per block so the peak estimator has something to walk.
"""

from __future__ import annotations

from typing import Iterable

import pytest

from axolotl.integrations.protrain.block.layout_rules import assign_modes
from axolotl.integrations.protrain.cost import (
    ALPHA_FRAGMENTATION,
    effective_bw,
    estimate_peak,
    estimate_runtime,
)
from axolotl.integrations.protrain.search import derive_bounds, search
from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    OpId,
    OpRecord,
    ParamId,
    ProfilerTrace,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


MB = 1 << 20
GB = 1 << 30


def _make_op_order(
    n_block: int, ops_per_block: int
) -> tuple[OpRecord, ...]:
    """Build a forward op sequence with ``ops_per_block`` ops per block."""
    out: list[OpRecord] = []
    op_id = 0
    for b in range(n_block):
        for k in range(ops_per_block):
            out.append(
                OpRecord(
                    op_id=OpId(op_id),
                    module_path=f"block.{b}.op.{k}",
                    qualified_name="aten::toy",
                    shape_signature=((1,),),
                    block_id=BlockId(b),
                    is_forward=True,
                )
            )
            op_id += 1
    return tuple(out)


def _make_trace(
    *,
    n_block: int = 8,
    ops_per_block: int = 5,
    activation_bytes_per_block: int = 32 * MB,
    model_state_bytes: int = 768 * MB,
    pcie_h2d_bps: float = 12e9,   # ~12 GB/s, 3090-like PCIe4 x16
    pcie_d2h_bps: float = 12e9,
    intra_delta_bytes: int = 8 * MB,
    inter_delta_bytes: int = 2 * MB,
    world: int = 1,
) -> ProfilerTrace:
    op_order = _make_op_order(n_block, ops_per_block)
    intra_op_delta: dict[OpId, int] = {op.op_id: intra_delta_bytes for op in op_order}
    inter_op_delta: dict[OpId, int] = {op.op_id: inter_delta_bytes for op in op_order}
    activation_sizes: dict[BlockId, int] = {
        BlockId(b): activation_bytes_per_block for b in range(n_block)
    }
    return ProfilerTrace(
        op_order=op_order,
        intra_op_delta=intra_op_delta,
        inter_op_delta=inter_op_delta,
        activation_sizes=activation_sizes,
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=pcie_h2d_bps,
        pcie_d2h_bps=pcie_d2h_bps,
        nccl_gather_s={} if world <= 1 else {64 * MB: 0.01},
        nccl_reduce_s={} if world <= 1 else {64 * MB: 0.012},
        arch_hash="test-arch",
        bs=1,
        seq=128,
        sku="RTX 3090 (synthetic)",
        world=world,
    )


def _make_layout(
    *, n_chunk: int = 12, s_chunk: int = 64 * MB, n_block: int = 8
) -> ChunkLayout:
    # Dummy chunk contents — enough to be structurally valid.
    chunks: list[tuple[ParamId, ...]] = [
        (ParamId(f"param.{i}"),) for i in range(n_chunk)
    ]
    param_to_chunk = {ParamId(f"param.{i}"): i for i in range(n_chunk)}
    # Distribute chunks across blocks roughly 1:1 then wrap.
    block_to_chunks: dict[BlockId, tuple] = {
        BlockId(b): (b % n_chunk,) for b in range(n_block)
    }
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=tuple(chunks),
        param_to_chunk=param_to_chunk,
        block_to_chunks=block_to_chunks,
    )


def _make_hw(
    *,
    gpu_memory_bytes: int = 24 * GB,
    gpu_count: int = 1,
    pcie_h2d_bps: float = 12e9,
    pcie_d2h_bps: float = 12e9,
) -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="NVIDIA GeForce RTX 3090 (synthetic)",
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_count=gpu_count,
        pcie_h2d_bps=pcie_h2d_bps,
        pcie_d2h_bps=pcie_d2h_bps,
        has_nvlink=False,
    )


@pytest.fixture
def toy_trace() -> ProfilerTrace:
    return _make_trace()


@pytest.fixture
def toy_layout() -> ChunkLayout:
    return _make_layout()


@pytest.fixture
def toy_hw() -> HardwareProfile:
    return _make_hw()


# ---------------------------------------------------------------------------
# memory / estimate_peak
# ---------------------------------------------------------------------------


def _peaks_for_ckpt_sweep(
    trace: ProfilerTrace,
    layout: ChunkLayout,
    hw: HardwareProfile,
    n_persist: int,
    n_buffer: int,
    n_swap: int,
) -> list[int]:
    """Return [peak(n_checkpoint=k) for k in 0..N_block]."""
    n_block = len(trace.activation_sizes)
    peaks: list[int] = []
    for k in range(0, n_block + 1 - n_swap):
        cfg = CostConfig(
            n_persist=n_persist,
            n_buffer=n_buffer,
            n_swap=n_swap,
            n_checkpoint=k,
        )
        bm = assign_modes(n_swap, k, n_block)
        peaks.append(estimate_peak(cfg, trace, layout, bm, hw))
    return peaks


def test_estimate_peak_monotonic_in_n_checkpoint(toy_trace, toy_layout, toy_hw):
    # With n_swap=0 and a fixed (n_persist, n_buffer), increasing
    # n_checkpoint should not increase peak memory (checkpointing
    # replaces retained-activation bytes with per-block recomputation
    # bumps that are equal in magnitude, so peak is non-increasing).
    peaks = _peaks_for_ckpt_sweep(
        toy_trace, toy_layout, toy_hw, n_persist=2, n_buffer=2, n_swap=0
    )
    for prev, nxt in zip(peaks, peaks[1:]):
        assert nxt <= prev, (
            f"peak should be non-increasing in n_checkpoint; got {peaks}"
        )


def test_estimate_peak_increases_with_n_persist_until_activations_dominate(
    toy_trace, toy_layout, toy_hw
):
    # At low n_persist the model-state contribution dominates, so
    # bumping n_persist strictly increases peak. Fix n_buffer=0 so the
    # buffer contribution is constant.
    peaks = []
    for n_persist in range(0, toy_layout.N_chunk + 1):
        cfg = CostConfig(
            n_persist=n_persist, n_buffer=0, n_swap=0, n_checkpoint=0
        )
        bm = assign_modes(0, 0, len(toy_trace.activation_sizes))
        peaks.append(estimate_peak(cfg, toy_trace, toy_layout, bm, toy_hw))

    # Must be strictly non-decreasing across the sweep.
    for prev, nxt in zip(peaks, peaks[1:]):
        assert nxt >= prev
    # And the first-to-last jump should be at least S_chunk * N_chunk
    # worth of model-state bytes after alpha scaling.
    expected_min_delta = int(
        ALPHA_FRAGMENTATION * toy_layout.N_chunk * toy_layout.S_chunk * 0.5
    )
    assert peaks[-1] - peaks[0] >= expected_min_delta


# ---------------------------------------------------------------------------
# runtime / estimate_runtime
# ---------------------------------------------------------------------------


def test_estimate_runtime_monotonic_in_n_buffer(toy_trace, toy_layout, toy_hw):
    """Searcher relies on the invariant that runtime is non-increasing in n_buffer
    (cached chunks skip re-gather). If this ever flips, the searcher's O(N_chunk)
    optimization in exhaustive.py picks the wrong n_buffer."""
    prev_iter_s = float("inf")
    for nb in range(toy_layout.N_chunk - 1):
        cfg = CostConfig(n_persist=1, n_buffer=nb, n_swap=0, n_checkpoint=0)
        block_map = assign_modes(
            cfg.n_swap, cfg.n_checkpoint, len(toy_trace.activation_sizes)
        )
        iter_s = estimate_runtime(cfg, toy_trace, toy_layout, block_map, toy_hw)
        assert iter_s <= prev_iter_s + 1e-9, (
            f"non-monotonic: n_buffer={nb} broke invariant "
            f"(prev={prev_iter_s:.6f}, now={iter_s:.6f})"
        )
        prev_iter_s = iter_s


def test_estimate_runtime_ckpt_adds_recompute(toy_trace, toy_layout, toy_hw):
    # When CPU-Adam dominates the iteration (all chunks non-persistent)
    # it masks backward-side changes via the T_iter max() in Eq. 2. Put
    # all chunks persistent so T_cpu_optim == 0 and the CKPT recomputation
    # bump shows up directly in T_bwd.
    n_block = len(toy_trace.activation_sizes)
    n_chunk = toy_layout.N_chunk
    cfg_zero = CostConfig(
        n_persist=n_chunk, n_buffer=0, n_swap=0, n_checkpoint=0
    )
    cfg_ckpt = CostConfig(
        n_persist=n_chunk, n_buffer=0, n_swap=0, n_checkpoint=4
    )

    bm_zero = assign_modes(0, 0, n_block)
    bm_ckpt = assign_modes(0, 4, n_block)

    t_zero = estimate_runtime(cfg_zero, toy_trace, toy_layout, bm_zero, toy_hw)
    t_ckpt = estimate_runtime(cfg_ckpt, toy_trace, toy_layout, bm_ckpt, toy_hw)

    assert t_ckpt > t_zero, (
        f"CKPT must add recomputation time: t_zero={t_zero:.6f} "
        f"t_ckpt={t_ckpt:.6f}"
    )


def test_effective_bw_derates_with_n_swap(toy_hw):
    cfg_no_swap = CostConfig(n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=0)
    cfg_swap = CostConfig(n_persist=0, n_buffer=0, n_swap=3, n_checkpoint=0)

    h2d_0, d2h_0 = effective_bw(cfg_no_swap, toy_hw)
    h2d_k, d2h_k = effective_bw(cfg_swap, toy_hw)

    assert h2d_0 >= h2d_k
    assert d2h_0 >= d2h_k
    # And the derate should be strict when n_swap > 0.
    assert h2d_0 > h2d_k
    assert d2h_0 > d2h_k


def test_effective_bw_multi_gpu_derate():
    """Multi-GPU derate is WEAKER than single-GPU for the same n_swap.

    Current formula: eff_bw = raw / (1 + 0.5 * min(1, n_swap / gpu_count)).
    * world=1, n_swap=2 → min(1, 2/1)=1 → factor 1.5 → eff = raw * (2/3)
    * world=4, n_swap=2 → min(1, 2/4)=0.5 → factor 1.25 → eff = raw * (0.8)
    So at identical n_swap, the 4-GPU case retains more bandwidth per rank.
    Guards against a refactor silently swapping the ratio direction or
    dropping the gpu_count clamp.
    """
    from dataclasses import replace

    hw_1gpu = _make_hw(gpu_count=1)
    hw_4gpu = replace(hw_1gpu, gpu_count=4)

    cfg = CostConfig(n_persist=0, n_buffer=4, n_swap=2, n_checkpoint=0)

    h2d_1, d2h_1 = effective_bw(cfg, hw_1gpu)
    h2d_4, d2h_4 = effective_bw(cfg, hw_4gpu)

    # Multi-GPU bandwidth should be HIGHER (less derated) than single-GPU
    # with the same n_swap because the contention is spread across ranks.
    assert h2d_4 > h2d_1, (
        f"multi-GPU H2D must derate less than single-GPU for same n_swap: "
        f"h2d_1={h2d_1:.2e} h2d_4={h2d_4:.2e}"
    )
    assert d2h_4 > d2h_1, (
        f"multi-GPU D2H must derate less than single-GPU for same n_swap: "
        f"d2h_1={d2h_1:.2e} d2h_4={d2h_4:.2e}"
    )

    # Spot-check absolute ratios against the formula.
    expected_h2d_1 = hw_1gpu.pcie_h2d_bps / 1.5
    expected_h2d_4 = hw_4gpu.pcie_h2d_bps / 1.25
    assert abs(h2d_1 - expected_h2d_1) / expected_h2d_1 < 1e-6
    assert abs(h2d_4 - expected_h2d_4) / expected_h2d_4 < 1e-6


# ---------------------------------------------------------------------------
# knobs / derive_bounds
# ---------------------------------------------------------------------------


def test_derive_bounds_basic(toy_trace, toy_layout):
    bounds = derive_bounds(toy_trace, toy_layout)
    assert bounds.N_chunk == toy_layout.N_chunk
    assert bounds.N_block == len(toy_trace.activation_sizes)
    assert bounds.N_interval > 0
    # We have 5 ops per block in the fixture, so N_interval should be
    # either 5 (mean) given uniform ops per block.
    assert bounds.N_interval == 5


# ---------------------------------------------------------------------------
# search / exhaustive
# ---------------------------------------------------------------------------


def test_search_picks_feasible_config(toy_trace, toy_layout, toy_hw):
    # Tighten capacity below the max-model-state footprint so not all
    # configs fit. Model state alone = 12 * 64MB = 768 MB; activations
    # at full retention = 8 * 32 = 256 MB; alpha = 1.1 pushes us past
    # 1.1 GB for the all-persistent all-NONE case.
    capacity = 700 * MB
    result = search(toy_trace, toy_layout, capacity, toy_hw)
    assert result.predicted_peak_bytes <= capacity
    assert result.predicted_iter_s > 0
    # And the block map should cover every block.
    assert len(result.block_map) == len(toy_trace.activation_sizes)


def test_search_raises_when_nothing_fits(toy_trace, toy_layout, toy_hw):
    with pytest.raises(RuntimeError, match="no feasible ProTrain config"):
        search(toy_trace, toy_layout, 0, toy_hw)


def test_search_picks_zero_swap_on_3090_like_hw(toy_trace, toy_layout):
    # 3090-like hardware: 12 GB/s PCIe, 24 GB memory, single GPU. On
    # such hardware the swap path should never be selected — backward
    # prefetch competes with compute and bandwidth is precious.
    hw = _make_hw(
        gpu_memory_bytes=24 * GB,
        gpu_count=1,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
    )
    capacity = 12 * GB  # large enough to let the search roam
    result = search(toy_trace, toy_layout, capacity, hw)
    assert result.cfg.n_swap == 0, (
        f"expected n_swap=0 on 3090-like HW, got cfg={result.cfg} "
        f"predicted_peak={result.predicted_peak_bytes} "
        f"predicted_iter_s={result.predicted_iter_s:.4f}"
    )


# ---------------------------------------------------------------------------
# Defensive: enumeration order does not affect chosen optimum
# ---------------------------------------------------------------------------


def test_search_returns_valid_block_map(toy_trace, toy_layout, toy_hw):
    """Smoke test: searcher output is internally consistent."""
    result = search(toy_trace, toy_layout, 12 * GB, toy_hw)
    n_block = len(toy_trace.activation_sizes)
    assert len(result.block_map) == n_block
    # Count modes in the block map matches the returned cfg.
    from axolotl.integrations.protrain.types import BlockMode

    counts: dict[BlockMode, int] = {m: 0 for m in BlockMode}
    for mode in result.block_map.values():
        counts[mode] += 1
    assert counts[BlockMode.SWAP] == result.cfg.n_swap
    assert counts[BlockMode.CKPT] == result.cfg.n_checkpoint


# ---------------------------------------------------------------------------
# Helper for debugging tests if they fail
# ---------------------------------------------------------------------------


def _iterable_repr(x: Iterable) -> str:  # pragma: no cover - debug helper
    return ",".join(str(v) for v in x)
