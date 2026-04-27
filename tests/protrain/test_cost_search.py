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
    estimate_cpu_footprint,
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
    op_latency_s: float = 0.0002,   # 200 µs per forward op; toy but >0
    hook_scale_ratio: float = 1.0,   # steady/hooked forward wall ratio; 1.0 = no-op
) -> ProfilerTrace:
    op_order = _make_op_order(n_block, ops_per_block)
    intra_op_delta: dict[OpId, int] = {op.op_id: intra_delta_bytes for op in op_order}
    inter_op_delta: dict[OpId, int] = {op.op_id: inter_delta_bytes for op in op_order}
    activation_sizes: dict[BlockId, int] = {
        BlockId(b): activation_bytes_per_block for b in range(n_block)
    }
    # Populated op_latencies so the cost model exercises the measured-compute
    # path rather than the activation-bytes fallback. Uniform per-op timing
    # keeps the synthetic invariants (monotonicity in n_buffer, CKPT-adds-
    # recompute, etc.) easy to reason about.
    op_latencies: dict[OpId, float] = {op.op_id: op_latency_s for op in op_order}
    # Hooked/steady forward wall-time fields (TRACE_VERSION=4). Default 1:1
    # ratio so the cost model's scale factor is identity and existing
    # invariants still hold. Individual tests can pass a non-default
    # ratio to exercise the scale path.
    hooked_sum = sum(op_latencies.values())
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
        op_latencies=op_latencies,
        hooked_fwd_wall_s=hooked_sum,
        steady_fwd_wall_s=hooked_sum * hook_scale_ratio,
        steady_bwd_wall_s=0.0,
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
    zero3_shard: bool = False,
) -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="NVIDIA GeForce RTX 3090 (synthetic)",
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_count=gpu_count,
        pcie_h2d_bps=pcie_h2d_bps,
        pcie_d2h_bps=pcie_d2h_bps,
        has_nvlink=False,
        zero3_shard=zero3_shard,
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


def test_estimate_peak_uses_per_block_caps(toy_layout, toy_hw):
    """``steady_fwd_block_peak_bytes`` caps the op-walk raw_peak for ANY config.

    Build a trace with an absurdly large synthetic intra_op_delta so the
    op-walk would compute a huge raw_peak absent the measured cap. Populate
    ``steady_fwd_block_peak_bytes`` with a modest per-block peak; the cap
    must pull raw_peak down to ``forward_max_block_peak + ckpt_recomp_bump``
    regardless of n_checkpoint/n_swap.

    Contrast: the v5 ``steady_fwd_peak_bytes`` cap only fires when
    n_checkpoint==0 && n_swap==0, so a config with n_checkpoint>0 would
    see the full (huge) op-walk peak. With per-block data the cap
    tightens fractional-NONE configs too.
    """
    n_block = 8
    # Raw op-walk raw_peak: uniform intra_delta of 1 GB per op.
    # Op-walk raw_peak >> 1 GB. Set per-block measured peaks to 512 MB —
    # the cap must pull raw_peak to ~512 MB + max(activation CKPT bump).
    huge_intra = 1 * GB
    activation_bytes_per_block = 64 * MB
    trace = _make_trace(
        n_block=n_block,
        ops_per_block=5,
        activation_bytes_per_block=activation_bytes_per_block,
        intra_delta_bytes=huge_intra,
    )
    per_block_peak = 512 * MB
    # Rebuild with block-peak dict populated — ProfilerTrace is frozen,
    # so construct a fresh one copying all fields from the base trace.
    from dataclasses import replace

    trace = replace(
        trace,
        steady_fwd_block_peak_bytes={
            BlockId(b): per_block_peak for b in range(n_block)
        },
    )

    # All-NONE config: ckpt_recomp_bump = 0, cap = per_block_peak.
    cfg_all_none = CostConfig(
        n_persist=4, n_buffer=2, n_swap=0, n_checkpoint=0
    )
    bm_all_none = assign_modes(0, 0, n_block)
    peak_all_none = estimate_peak(
        cfg_all_none, trace, toy_layout, bm_all_none, toy_hw
    )
    # Scaled cap = ALPHA_FRAGMENTATION * per_block_peak; op-walk would
    # otherwise be > 1 GB * alpha. The cap should pin peak near the
    # scaled per_block_peak value.
    assert peak_all_none <= int(ALPHA_FRAGMENTATION * per_block_peak) + 1, (
        f"all-NONE peak {peak_all_none/1e6:.1f}MB should be capped at "
        f"~{ALPHA_FRAGMENTATION * per_block_peak / 1e6:.1f}MB"
    )

    # Fractional-NONE config: 3 blocks CKPT. ckpt_recomp_bump =
    # max activation across CKPT blocks = activation_bytes_per_block.
    cfg_mixed = CostConfig(
        n_persist=4, n_buffer=2, n_swap=0, n_checkpoint=3
    )
    bm_mixed = assign_modes(0, 3, n_block)
    peak_mixed = estimate_peak(
        cfg_mixed, trace, toy_layout, bm_mixed, toy_hw
    )
    expected_cap = int(
        ALPHA_FRAGMENTATION * (per_block_peak + activation_bytes_per_block)
    )
    # 1% slack for ALPHA_FRAGMENTATION * int() rounding.
    assert peak_mixed <= expected_cap + 1, (
        f"mixed-CKPT peak {peak_mixed/1e6:.1f}MB should be capped at "
        f"~{expected_cap/1e6:.1f}MB (forward_max_block + max_ckpt_activation)"
    )
    # Without per-block cap the op-walk raw_peak would dwarf this
    # (intra_delta=1GB per op). Sanity check: the capped value is well
    # below 1 GB * alpha.
    assert peak_mixed < int(ALPHA_FRAGMENTATION * huge_intra), (
        "per-block cap should pull peak well below the raw op-walk "
        "estimate; got {peak_mixed/1e9:.3f}GB"
    )


def test_estimate_peak_per_block_cap_respects_under_predict_floor(toy_layout, toy_hw):
    """Per-block cap must not under-predict when the op-walk is tighter.

    If the op-walk's raw_peak is ALREADY smaller than
    ``forward_max_block_peak + ckpt_recomp_bump``, the cap is a no-op.
    Verify that a trace with tiny intra_deltas and a large per-block
    measurement yields the op-walk's value, not the inflated measurement.
    """
    n_block = 8
    trace = _make_trace(
        n_block=n_block,
        ops_per_block=3,
        activation_bytes_per_block=4 * MB,
        intra_delta_bytes=1 * MB,
        inter_delta_bytes=256 * 1024,
    )
    from dataclasses import replace

    trace = replace(
        trace,
        steady_fwd_block_peak_bytes={
            BlockId(b): 10 * GB for b in range(n_block)
        },
    )
    cfg = CostConfig(n_persist=4, n_buffer=2, n_swap=0, n_checkpoint=0)
    bm = assign_modes(0, 0, n_block)
    peak = estimate_peak(cfg, trace, toy_layout, bm, toy_hw)
    # The per-block cap is 10 GB+; the op-walk gives a much smaller
    # peak (<< 1 GB). The cap must NOT raise raw_peak — only lower it.
    assert peak < int(ALPHA_FRAGMENTATION * 1 * GB), (
        f"peak {peak/1e9:.3f}GB should track the tight op-walk, not the "
        "10 GB per-block measurement"
    )


# ---------------------------------------------------------------------------
# memory / estimate_cpu_footprint (M7 follow-up: ZeRO-3 awareness)
# ---------------------------------------------------------------------------


def test_estimate_cpu_footprint_scales_with_world_size():
    """Per-rank pinned CPU footprint divides by ``gpu_count`` under sharding.

    The replicated path (``zero3_shard=False``) has every rank hold a
    full copy of every non-persistent chunk on CPU. The ZeRO-3
    sharded path (``zero3_shard=True``) partitions each chunk's bytes
    across ranks so each rank holds only ``chunk_bytes/world_size``
    pinned bytes per chunk. This test locks in the arithmetic that
    future searcher CPU-budget filters (if added) rely on.

    Toy layout: N_chunk=12, S_chunk=128MB. With n_persist=4 the
    non-persistent set is 8 chunks * 128MB = 1 GB.
    """
    n_chunk = 12
    s_chunk = 128 * MB
    n_persist = 4
    cfg = CostConfig(
        n_persist=n_persist, n_buffer=2, n_swap=0, n_checkpoint=0
    )
    layout = _make_layout(n_chunk=n_chunk, s_chunk=s_chunk, n_block=8)

    expected_total = (n_chunk - n_persist) * s_chunk  # 1 GB

    hw_single = _make_hw(gpu_count=1, zero3_shard=False)
    footprint_single = estimate_cpu_footprint(cfg, layout, hw_single)
    assert footprint_single == expected_total, (
        f"single-GPU / no-shard footprint should be the full "
        f"non-persistent total ({expected_total}B), got {footprint_single}B"
    )

    hw_4gpu_ddp = _make_hw(gpu_count=4, zero3_shard=False)
    footprint_4gpu_ddp = estimate_cpu_footprint(cfg, layout, hw_4gpu_ddp)
    assert footprint_4gpu_ddp == expected_total, (
        f"4-GPU without shard (DDP mode) still replicates full chunks "
        f"per rank — expected {expected_total}B, got {footprint_4gpu_ddp}B"
    )

    hw_4gpu_shard = _make_hw(gpu_count=4, zero3_shard=True)
    footprint_4gpu_shard = estimate_cpu_footprint(cfg, layout, hw_4gpu_shard)
    # Ceiling division so the trailing rank's shard pad counts: for
    # 1 GB / 4 = 256 MB exactly, no rounding.
    expected_sharded = expected_total // 4
    assert footprint_4gpu_shard == expected_sharded, (
        f"4-GPU sharded footprint should be total/world_size = "
        f"{expected_sharded}B, got {footprint_4gpu_shard}B"
    )

    # Sanity ratio: sharded is exactly 1/world_size of replicated at
    # this chunk-size / world_size alignment.
    assert footprint_single == 4 * footprint_4gpu_shard
    assert footprint_4gpu_ddp > footprint_4gpu_shard


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


def test_estimate_runtime_falls_back_when_adam_bps_zero(toy_trace, toy_layout):
    """HardwareProfile with ``cpu_adam_bytes_per_sec=0.0`` must trigger the
    fallback path in ``estimate_runtime`` (and likewise for GPU Adam). The
    output must be a finite positive number; the fallback constants live in
    ``cost/runtime.py`` as ``_CPU_ADAM_FALLBACK`` / ``_GPU_ADAM_FALLBACK``.
    """
    hw_no_adam = _make_hw()  # defaults: cpu_adam=0.0, gpu_adam=0.0
    cfg = CostConfig(n_persist=2, n_buffer=2, n_swap=0, n_checkpoint=0)
    block_map = assign_modes(0, 0, len(toy_trace.activation_sizes))

    t = estimate_runtime(cfg, toy_trace, toy_layout, block_map, hw_no_adam)

    assert t > 0.0
    import math

    assert math.isfinite(t)


def test_estimate_runtime_uses_measured_adam_when_provided(toy_trace, toy_layout):
    """A 10x larger ``cpu_adam_bytes_per_sec`` on the HardwareProfile must
    translate to a ~10x smaller CPU-optim contribution in the runtime
    estimate.

    Picks a CPU-Adam-dominated config (all chunks non-persistent) so
    ``t_cpu_optim`` shows up on the critical path via the ``max()`` in
    Eq. 2. The ratio-assertion avoids needing to know the other terms
    exactly — we only care that the Adam rate IS the knob controlling
    the CPU-optim contribution.
    """
    from dataclasses import replace

    n_block = len(toy_trace.activation_sizes)
    # Force CPU-Adam onto the critical path: n_persist=0 moves all chunks
    # to the CPU-Adam branch, n_checkpoint=0 keeps t_bwd small so
    # t_cpu_optim > t_bwd + t_gpu_optim.
    cfg = CostConfig(n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=0)
    block_map = assign_modes(0, 0, n_block)

    hw_slow = _make_hw()
    hw_slow = replace(hw_slow, cpu_adam_bytes_per_sec=1e9)  # 1 GB/s
    hw_fast = replace(hw_slow, cpu_adam_bytes_per_sec=1e10)  # 10 GB/s

    t_slow = estimate_runtime(cfg, toy_trace, toy_layout, block_map, hw_slow)
    t_fast = estimate_runtime(cfg, toy_trace, toy_layout, block_map, hw_fast)

    # The CPU-Adam contribution scales inversely with the rate. Since
    # this config puts CPU-Adam on the critical path (see docstring), the
    # iteration time drop should approach 10x on the CPU-optim term.
    # Other terms (t_fwd forward-only) are small and identical between
    # runs, so the total ratio is ~10 but loosely so; assert >5 as a
    # robust sanity threshold.
    assert t_fast < t_slow
    # Compute the t_cpu_optim contribution alone: for the same config,
    # everything except the Adam term is constant. Use the difference:
    delta_slow_vs_fast = t_slow - t_fast
    # Reconstruct the implicit t_cpu_optim term from the rate change:
    # t_cpu_optim_slow = X / 1e9; t_cpu_optim_fast = X / 1e10;
    # their difference = 0.9 * X / 1e9 = 0.9 * t_cpu_optim_slow.
    # So delta_slow_vs_fast == 0.9 * t_cpu_optim_slow — this means the
    # ratio delta/t_slow should be close to 0.9 when CPU-optim
    # dominates. Allow a generous 0.5 floor to tolerate non-dominating
    # configs without masking regressions.
    assert delta_slow_vs_fast / t_slow > 0.5, (
        f"10x faster CPU Adam barely moved the needle: "
        f"t_slow={t_slow:.6f} t_fast={t_fast:.6f}"
    )


def test_bwd_compute_time_uses_phase2_chunked_measurement_when_present():
    """Phase-2 path (TRACE_VERSION 10) takes precedence over the v8 unwrapped ratio.

    A trace with both ``steady_bwd_chunked_wall_s`` and the legacy
    ``steady_bwd_wall_s`` populated must use the chunked field. The
    return value is the BASE backward (recompute subtracted), so the
    caller's per-cfg recompute term still adds the right amount on top.
    """
    from dataclasses import replace

    from axolotl.integrations.protrain.cost.runtime import (
        _bwd_compute_time_from_trace,
    )

    base_trace = _make_trace()
    # Numbers picked so the translation is hand-verifiable:
    # measurement = 1.20s, bootstrap had 4 CKPT'd blocks, per-block
    # recompute = 0.05s -> phase2_recompute = 0.20s -> base = 1.00s.
    trace = replace(
        base_trace,
        steady_bwd_wall_s=2.50,  # would give a 1.0× clamp via path 2
        steady_bwd_chunked_wall_s=1.20,
        phase2_n_checkpoint=4,
        phase2_per_block_recompute_s=0.05,
    )
    base = _bwd_compute_time_from_trace(trace, t_fwd_total=2.50)
    assert base == pytest.approx(1.00, abs=1e-9), (
        f"phase-2 base should be measured - bootstrap_recompute = "
        f"1.20 - 4*0.05 = 1.00, got {base}"
    )


def test_bwd_compute_time_phase2_clamped_to_non_negative():
    """If the measurement is shorter than bootstrap recompute (degenerate case),
    the base is clamped to 0 — the caller's per-cfg recompute then provides
    the entire backward time. Real measurements should never trigger this,
    but we guard against arithmetic surprises.
    """
    from dataclasses import replace

    from axolotl.integrations.protrain.cost.runtime import (
        _bwd_compute_time_from_trace,
    )

    base_trace = _make_trace()
    # Bootstrap recompute = 4 * 0.5 = 2.0s but measurement = 1.0s.
    trace = replace(
        base_trace,
        steady_bwd_chunked_wall_s=1.0,
        phase2_n_checkpoint=4,
        phase2_per_block_recompute_s=0.5,
    )
    base = _bwd_compute_time_from_trace(trace, t_fwd_total=2.50)
    assert base == 0.0, f"expected clamp to 0, got {base}"


def test_bwd_compute_time_falls_back_when_phase2_not_populated():
    """When phase-2 fields are 0 (pre-v10 cache or skipped phase-2), use v8 path."""
    from dataclasses import replace

    from axolotl.integrations.protrain.cost.runtime import (
        _bwd_compute_time_from_trace,
    )

    base_trace = _make_trace()

    # v8-style trace: legacy steady_bwd_wall_s populated, phase-2 fields 0.
    trace_v8 = replace(
        base_trace,
        steady_bwd_wall_s=1.5,
        steady_fwd_wall_s=1.0,  # ratio = 1.5
        # phase-2 fields all default 0.0 / 0
    )
    bwd_v8 = _bwd_compute_time_from_trace(trace_v8, t_fwd_total=2.0)
    assert bwd_v8 == pytest.approx(2.0 * 1.5, abs=1e-9), (
        f"v8 path should return t_fwd * measured_ratio = 3.0, got {bwd_v8}"
    )

    # Pure heuristic: nothing measured at all -> 2x canonical (assuming
    # trainable_param_fraction defaults to 0 which goes to else branch).
    trace_h = replace(
        base_trace,
        steady_bwd_wall_s=0.0,
        steady_fwd_wall_s=0.0,
    )
    bwd_h = _bwd_compute_time_from_trace(trace_h, t_fwd_total=2.0)
    assert bwd_h == pytest.approx(2.0 * 2.0, abs=1e-9), (
        f"heuristic path should return t_fwd * 2.0 = 4.0, got {bwd_h}"
    )


def test_fwd_compute_time_uses_phase2_chunked_fwd_when_present():
    """``_fwd_compute_time_from_trace`` overrides the total with the chunked
    forward measurement when populated (TRACE_VERSION ≥ 11).

    Mirrors the precedence pattern in
    :func:`_bwd_compute_time_from_trace`: the phase-2 chunked
    measurement takes precedence over the per-op-derived total. The
    per-block distribution stays at the per-op-derived shape — used
    for CKPT recompute accounting in ``estimate_runtime``.
    """
    from dataclasses import replace

    from axolotl.integrations.protrain.cost.runtime import (
        _fwd_compute_time_from_trace,
    )

    base_trace = _make_trace()
    per_op_sum = 8 * 5 * 0.0002

    # Without chunked fwd populated — total = per-op sum.
    trace_no = replace(base_trace, steady_fwd_chunked_wall_s=0.0)
    total_no, per_block_no, used_no = _fwd_compute_time_from_trace(trace_no)
    assert used_no is True
    assert total_no == pytest.approx(per_op_sum, abs=1e-9), (
        f"v10 fallback should return per-op sum {per_op_sum}, got {total_no}"
    )

    # With chunked fwd populated — total = chunked wall.
    chunked_fwd = 0.30
    trace_with = replace(base_trace, steady_fwd_chunked_wall_s=chunked_fwd)
    total_with, per_block_with, used_with = _fwd_compute_time_from_trace(
        trace_with
    )
    assert used_with is True
    assert total_with == pytest.approx(chunked_fwd, abs=1e-9), (
        f"phase-2 fwd path should return chunked wall {chunked_fwd}, "
        f"got {total_with}"
    )
    # Per-block stays at per-op-derived shape — does NOT rescale.
    for bid in per_block_no:
        assert per_block_with[bid] == pytest.approx(per_block_no[bid], rel=1e-6), (
            f"per-block must stay per-op-derived for block {bid}: "
            f"with={per_block_with[bid]} no={per_block_no[bid]}"
        )


def test_estimate_runtime_uses_phase2_chunked_fwd_measurement():
    """End-to-end: ``estimate_runtime`` substitutes ``steady_fwd_chunked_wall_s``
    for the per-chunk-roofline t_fwd assembly.

    With phase-2 fwd populated, t_fwd should equal the measured
    chunked wall (plus SKU scale + any swap transfer) — NOT the
    per-chunk max(compute, comm) sum. The bootstrap-then-search
    pipeline depends on this for the cost model to predict close to
    actual on the bootstrap config.
    """
    from dataclasses import replace

    from axolotl.integrations.protrain.cost.runtime import estimate_runtime

    base_trace = _make_trace()
    n_block = len(base_trace.activation_sizes)
    chunked_fwd = 0.20
    trace = replace(
        base_trace,
        steady_fwd_chunked_wall_s=chunked_fwd,
        # Set chunked bwd too so the bwd path is also on the phase-2
        # branch (otherwise its fallback paths depend on
        # steady_fwd_wall_s and would mask the forward signal).
        steady_bwd_chunked_wall_s=0.30,
        phase2_n_checkpoint=n_block,
        phase2_per_block_recompute_s=8 * 5 * 0.0002 / n_block,
    )
    layout = _make_layout()
    hw = _make_hw()
    n_chunk = layout.N_chunk

    cfg_high_persist = CostConfig(
        n_persist=n_chunk, n_buffer=0, n_swap=0, n_checkpoint=0
    )
    bm = assign_modes(0, 0, n_block)

    t_with = estimate_runtime(cfg_high_persist, trace, layout, bm, hw)

    # Synthesize a trace WITHOUT the chunked fwd; the per-chunk-roofline
    # forward path fires instead. Under cfg_high_persist (all
    # persistent, no comm), that path collapses to per-op-sum × hook
    # scale = 8 * 5 * 0.0002 = 0.008s. With phase-2 forward, t_fwd
    # = chunked_fwd (0.20s). So the t_iter delta should be
    # chunked_fwd - per_op_sum ≈ 0.192s (forward is the only
    # phase-2-affected term in this all-NONE config).
    trace_no_fwd = replace(trace, steady_fwd_chunked_wall_s=0.0)
    t_without = estimate_runtime(
        cfg_high_persist, trace_no_fwd, layout, bm, hw
    )
    delta = t_with - t_without
    expected_delta = 0.20 - 8 * 5 * 0.0002  # ~0.192
    assert delta == pytest.approx(expected_delta, abs=1e-3), (
        f"chunked-fwd override should increase t_fwd by ~{expected_delta:.4f}, "
        f"got delta={delta:.4f} (t_with={t_with:.4f} t_without={t_without:.4f})"
    )


def test_estimate_runtime_phase2_translation_changes_with_n_checkpoint():
    """End-to-end: with phase-2 populated, increasing n_checkpoint adds recompute.

    The translation is the whole point of D1b. A trace whose phase-2
    measurement was taken under all-CKPT bootstrap should yield bigger
    backward times for configs with more CKPT blocks (the addition is
    via the caller's per_block_compute walk, NOT via the measurement
    itself).
    """
    from dataclasses import replace

    from axolotl.integrations.protrain.cost.runtime import estimate_runtime

    base_trace = _make_trace()
    n_block = len(base_trace.activation_sizes)
    # Bootstrap was n_checkpoint=N_block (all CKPT). Per-block recompute
    # at 0.001s — small enough that the translation doesn't dominate
    # but big enough to be visible after the n_block multiplier.
    trace = replace(
        base_trace,
        steady_bwd_chunked_wall_s=0.5,
        phase2_n_checkpoint=n_block,
        phase2_per_block_recompute_s=0.001,
    )
    layout = _make_layout()
    hw = _make_hw()
    n_chunk = layout.N_chunk

    # All-persistent so CPU-Adam doesn't mask backward changes.
    cfg_zero = CostConfig(
        n_persist=n_chunk, n_buffer=0, n_swap=0, n_checkpoint=0
    )
    cfg_full_ckpt = CostConfig(
        n_persist=n_chunk, n_buffer=0, n_swap=0, n_checkpoint=n_block
    )
    bm_zero = assign_modes(0, 0, n_block)
    bm_full = assign_modes(0, n_block, n_block)

    t_zero = estimate_runtime(cfg_zero, trace, layout, bm_zero, hw)
    t_full = estimate_runtime(cfg_full_ckpt, trace, layout, bm_full, hw)

    # The all-CKPT config must add per-block recompute on top of the
    # base; the all-NONE config must not. The DELTA proves the
    # translation is wired up.
    assert t_full > t_zero, (
        f"phase-2 translation broken: t_full={t_full:.6f} <= t_zero={t_zero:.6f}; "
        "all-CKPT should be more expensive than all-NONE because the "
        "caller's per-cfg recompute term adds time on top of the base"
    )


def test_estimate_runtime_phase2_bwd_bypasses_chunk_comm_but_keeps_recompute():
    """Phase-2 backward consumes translated measured wall directly.

    Changing n_persist/n_buffer changes the analytical backward comm assembly,
    but must not change t_bwd when the phase-2 chunked backward measurement is
    populated. Candidate CKPT recompute should still be added on top of the
    translated base.
    """
    from dataclasses import replace

    base_trace = _make_trace(world=2)
    n_block = len(base_trace.activation_sizes)
    per_op_sum = 8 * 5 * 0.0002
    trace = replace(
        base_trace,
        model_state_bytes=0,
        steady_fwd_chunked_wall_s=0.05,
        steady_bwd_chunked_wall_s=0.020,
        phase2_n_checkpoint=n_block,
        phase2_per_block_recompute_s=0.0005,
    )
    layout = _make_layout()
    hw = _make_hw(gpu_count=2)
    n_chunk = layout.N_chunk
    bm_none = assign_modes(0, 0, n_block)

    cfg_uncached = CostConfig(
        n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=0
    )
    cfg_cached = CostConfig(
        n_persist=0, n_buffer=n_chunk, n_swap=0, n_checkpoint=0
    )
    cfg_persistent = CostConfig(
        n_persist=n_chunk, n_buffer=0, n_swap=0, n_checkpoint=0
    )

    t_uncached = estimate_runtime(cfg_uncached, trace, layout, bm_none, hw)
    t_cached = estimate_runtime(cfg_cached, trace, layout, bm_none, hw)
    t_persistent = estimate_runtime(cfg_persistent, trace, layout, bm_none, hw)

    assert t_cached == pytest.approx(t_uncached, abs=1e-9)
    assert t_persistent == pytest.approx(t_uncached, abs=1e-9)

    cfg_ckpt = CostConfig(
        n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=n_block
    )
    bm_ckpt = assign_modes(0, n_block, n_block)
    t_ckpt = estimate_runtime(cfg_ckpt, trace, layout, bm_ckpt, hw)

    assert t_ckpt - t_uncached == pytest.approx(per_op_sum, abs=1e-9)


def test_estimate_runtime_per_sku_compute_scale(toy_trace, toy_layout):
    """SKU compute-rate calibration scales forward compute proportionally.

    Trace captured on a faster SKU (higher TFLOPS) replayed on a slower SKU
    (lower TFLOPS) → the cost model must scale forward-time UP by the ratio.
    Picks an all-persistent config so forward compute is on the critical
    path with no comm dominance, making the scale visible end-to-end.
    """
    from dataclasses import replace

    n_block = len(toy_trace.activation_sizes)
    n_chunk = toy_layout.N_chunk
    cfg = CostConfig(n_persist=n_chunk, n_buffer=0, n_swap=0, n_checkpoint=0)
    block_map = assign_modes(0, 0, n_block)

    # Trace says "I was captured on a 60 TFLOPS card."
    fast_trace = replace(toy_trace, compute_rate_tflops=60.0)

    # Live SKU is 60 TFLOPS — same card. Scale = 1.0.
    hw_same = _make_hw()
    hw_same = replace(hw_same, gpu_compute_tflops=60.0)
    t_same = estimate_runtime(cfg, fast_trace, toy_layout, block_map, hw_same)

    # Live SKU is 30 TFLOPS — half the speed. Scale = 60/30 = 2.0; forward
    # compute should roughly double.
    hw_slow = _make_hw()
    hw_slow = replace(hw_slow, gpu_compute_tflops=30.0)
    t_slow = estimate_runtime(cfg, fast_trace, toy_layout, block_map, hw_slow)

    # The forward term should grow by ~2x; total iter time ratio should be
    # >1.4 (allowing for non-fwd terms diluting the signal). When backward
    # is roughly proportional to forward (default 2x ratio), total scales
    # ~ proportionally, so >1.4 is a robust threshold.
    assert t_slow > t_same * 1.4, (
        f"per-SKU calibration didn't scale t_iter: t_same={t_same:.6f} "
        f"t_slow={t_slow:.6f} (expected >1.4x)"
    )


def test_estimate_runtime_sku_scale_identity_when_unmeasured(toy_trace, toy_layout, toy_hw):
    """0.0 on either side of the SKU ratio falls back to identity scale."""
    from dataclasses import replace

    cfg = CostConfig(n_persist=2, n_buffer=2, n_swap=0, n_checkpoint=0)
    block_map = assign_modes(0, 0, len(toy_trace.activation_sizes))

    # Both unmeasured → identity scale → unchanged result.
    t_baseline = estimate_runtime(cfg, toy_trace, toy_layout, block_map, toy_hw)

    # Trace measured but live not measured → still identity (HW info missing).
    trace_with = replace(toy_trace, compute_rate_tflops=60.0)
    t_trace_only = estimate_runtime(cfg, trace_with, toy_layout, block_map, toy_hw)
    assert abs(t_trace_only - t_baseline) < 1e-9, (
        f"identity scale violated when only trace had a measurement: "
        f"baseline={t_baseline:.6f} with={t_trace_only:.6f}"
    )

    # Live measured but trace not → also identity.
    hw_with = replace(toy_hw, gpu_compute_tflops=60.0)
    t_hw_only = estimate_runtime(cfg, toy_trace, toy_layout, block_map, hw_with)
    assert abs(t_hw_only - t_baseline) < 1e-9, (
        f"identity scale violated when only hw had a measurement: "
        f"baseline={t_baseline:.6f} with={t_hw_only:.6f}"
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


def test_search_cpu_capacity_filter_excludes_high_offload_configs(
    toy_trace, toy_layout, toy_hw
):
    """CPU feasibility filter must drop configs whose CPU footprint exceeds the budget.

    Toy layout: N_chunk=12, S_chunk=64MB → CPU footprint =
    ``(12 - n_persist) * S_chunk`` per rank under the replicated
    (``zero3_shard=False``) path.

    Setup: a tight GPU capacity forces the unfiltered searcher to pick
    a CPU-heavy cfg (the lowest n_persist that still clears the GPU
    gate is also the highest n_persist the runtime model can pick,
    because the runtime favours fewer CPU-resident chunks). With a
    LOOSE CPU budget (>= baseline footprint) the same cfg is picked.
    With a TIGHT CPU budget (< baseline footprint) the searcher must
    either pick a different cfg or raise — and on this synthetic
    fixture every higher-n_persist alternative is GPU-infeasible, so
    the filter exposes the no-fit case. That last branch is covered
    by ``test_search_raises_cpu_pressure_specific_message_when_no_cfg_fits_both``;
    here we assert (a) loose-budget = baseline pick, (b) tighter-but-
    still-feasible budget = baseline still picked, (c) budget below
    baseline footprint excludes baseline (verified via the picked
    cfg's footprint).
    """
    capacity = 600 * MB
    # Sanity: unfiltered pick has non-zero CPU footprint on this fixture.
    baseline = search(toy_trace, toy_layout, capacity, toy_hw)
    baseline_cpu = (
        toy_layout.N_chunk - baseline.cfg.n_persist
    ) * toy_layout.S_chunk
    assert baseline_cpu > 0, (
        f"fixture sanity: baseline must offload >0B to CPU for the "
        f"filter to have anything to reject; got cfg={baseline.cfg}"
    )

    # (a) Loose CPU budget (matches baseline footprint) -> same pick.
    loose = search(
        toy_trace,
        toy_layout,
        capacity,
        toy_hw,
        cpu_capacity_bytes=baseline_cpu,
    )
    assert loose.cfg == baseline.cfg, (
        f"CPU budget == baseline footprint should not change the pick; "
        f"baseline={baseline.cfg} loose={loose.cfg}"
    )

    # (b) CPU budget strictly above baseline footprint -> same pick.
    above = search(
        toy_trace,
        toy_layout,
        capacity,
        toy_hw,
        cpu_capacity_bytes=baseline_cpu + 10 * MB,
    )
    assert above.cfg == baseline.cfg

    # (c) CPU budget BELOW baseline footprint -> baseline excluded.
    # On this fixture every n_persist >= baseline.n_persist that would
    # reduce CPU footprint is GPU-infeasible at capacity=600MB, so the
    # search must raise — covered by the dedicated CPU-pressure test
    # below. Here we just assert the boundary: at exactly
    # ``baseline_cpu - 1`` the search no longer admits the baseline cfg.
    with pytest.raises(RuntimeError, match=r"no ProTrain config fits in"):
        search(
            toy_trace,
            toy_layout,
            capacity,
            toy_hw,
            cpu_capacity_bytes=baseline_cpu - 1,
        )


def test_search_cpu_capacity_none_matches_pre_filter_behaviour(
    toy_trace, toy_layout, toy_hw
):
    """Backward-compat: ``cpu_capacity_bytes=None`` -> identical pick.

    The pre-filter signature ``search(trace, layout, capacity, hw)`` and
    the new signature ``search(..., cpu_capacity_bytes=None)`` must
    produce byte-identical SearchResults. Same cfg, same block_map,
    same predicted peak, same predicted iter_s.
    """
    capacity = 12 * GB
    pre_filter = search(toy_trace, toy_layout, capacity, toy_hw)
    explicit_none = search(
        toy_trace, toy_layout, capacity, toy_hw, cpu_capacity_bytes=None
    )
    assert pre_filter.cfg == explicit_none.cfg
    assert pre_filter.block_map == explicit_none.block_map
    assert pre_filter.predicted_peak_bytes == explicit_none.predicted_peak_bytes
    assert pre_filter.predicted_iter_s == explicit_none.predicted_iter_s


def test_search_raises_cpu_pressure_specific_message_when_no_cfg_fits_both(
    toy_trace, toy_layout, toy_hw
):
    """When at least one cfg clears the GPU gate but every one busts the
    CPU envelope, the failure message must explicitly cite the host RAM
    budget so the user knows to scale up RAM, not GPU memory.
    """
    capacity = 12 * GB  # roomy GPU — many configs clear the GPU gate
    # Tight CPU budget: 0 bytes means only the all-persistent
    # (n_persist=N_chunk → 0 non-persistent chunks on CPU) cfg could
    # fit. But the toy layout's _min_n_buffer_for at n_persist=N_chunk
    # is 0, so n_persist=N_chunk is itself feasible only if the
    # GPU capacity admits the full model-state. We block that by
    # picking a CPU budget that's strictly less than ``S_chunk`` —
    # so even a single non-persistent chunk on CPU busts it — AND
    # combine with a GPU capacity that prevents fully-on-GPU
    # configs from clearing the GPU gate.
    #
    # Calibration: the all-persistent cfg's GPU peak ~= alpha *
    # (N_chunk * S_chunk + activations + intra/inter). With
    # 768 MB of model state alone, capping GPU at 600 MB ensures
    # the all-persistent cfg fails the GPU gate, while leaving
    # some room for partially-offloaded cfgs to clear it. CPU
    # budget = 1 byte then makes them all bust the CPU gate.
    tight_capacity = 600 * MB
    with pytest.raises(RuntimeError, match=r"no ProTrain config fits in"):
        search(
            toy_trace,
            toy_layout,
            tight_capacity,
            toy_hw,
            cpu_capacity_bytes=1,
        )


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
