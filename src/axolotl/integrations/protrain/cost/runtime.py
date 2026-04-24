"""Runtime (wall-clock) cost estimator for the ProTrain searcher (§3.3, App A.1).

Implements Eqs. 2-7 from the paper:

    T_iter    = T_fwd + max(T_bwd + T_gpu_optim, T_cpu_optim)
    T_fwd     = sum_chunks  max(T_compute_chunk, T_comm_chunk)     [Eq. 2-3]
    T_bwd     = sum_chunks  max(T_compute_chunk + T_recomp_chunk,
                                T_comm_chunk)                      [Eq. 4-5]
    T_gpu_opt = sum_{persistent chunks} T_step(chunk)              [Eq. 6]
    T_cpu_opt = sum_{non-persistent chunks} T_step(chunk)          [Eq. 7]

Key accounting rules (summary §3.3, paper §3.3.1):

- Persistent chunks contribute no prefetch/gather cost (they never leave
  GPU).
- Buffer-cached chunks skip re-gather in backward — modeled by halving
  their backward communication term.
- CPU-Adam overlaps GPU backward; only exposed if ``T_cpu_optim`` exceeds
  ``T_bwd + T_gpu_optim``.
- CKPT blocks add a recomputation-compute term to backward.
- SWAP blocks add CPU<->GPU activation transfer on both sides.
- For single-rank (``world == 1``) the NCCL gather/reduce terms are 0
  because there are no collectives.

The estimator is a pure function of the frozen dataclass inputs; it does
not allocate tensors or touch CUDA.
"""

from __future__ import annotations

from axolotl.integrations.protrain.cost.bandwidth import effective_bw
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ProfilerTrace,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# FALLBACK compute throughput proxy — only used when the ProfilerTrace has no
# ``op_latencies`` (e.g. a trace recorded on CPU, or a stale cached trace from
# before TRACE_VERSION=2). When measured per-op latencies ARE available, the
# cost model consumes them directly and this constant is not read.
_COMPUTE_BYTES_PER_SEC: float = 3.0e11  # ~300 GB/s, rough 3090 effective

# Fallback CPU-Adam step throughput (bytes of optim-state processed per
# second). The cost model prefers the MEASURED rate from
# ``HardwareProfile.cpu_adam_bytes_per_sec`` (populated by
# ``profiler/hw_bench.measure_cpu_adam``); this constant is only consumed
# when the measurement returned 0.0 (e.g. DeepSpeedCPUAdam failed to
# compile, common on dev rigs with CUDA toolchain mismatches).
# DeepSpeedCPUAdam benches around 1-2 GB/s per step on a decent Xeon/
# Threadripper; the "20 B/param" accounting in hw_bench pushes the
# measured throughput a bit higher — 8 GB/s is a reasonable middle-of-
# the-road prior that avoids under- or over-predicting catastrophically.
_CPU_ADAM_FALLBACK: float = 8.0e9

# Fallback GPU FusedAdam throughput, same semantics as ``_CPU_ADAM_FALLBACK``.
# GPU Adam is HBM-bandwidth-bound on 3090s; 500 GB/s is a mid-range prior
# that matches the 3090's sustained HBM BW.
_GPU_ADAM_FALLBACK: float = 5.0e11

# Backward-vs-forward compute ratio when the trace has forward latencies but
# no per-block backward split. The synthetic ``<backward>`` op records a
# single aggregate latency; using that directly is more accurate than the
# heuristic factor, and the code below prefers it when present.
_BWD_FWD_COMPUTE_RATIO: float = 2.0


def _compute_time(activation_bytes: int) -> float:
    """Rough compute time proxy — used only as a fallback for traces that
    carry no measured ``op_latencies`` (see ``_fwd_compute_time_from_trace``).
    """
    return activation_bytes / _COMPUTE_BYTES_PER_SEC


def _block_compute_time(trace: ProfilerTrace, block_id: BlockId) -> float:
    """Wall-clock forward compute for one block from profiler measurements.

    Sums the measured op latencies for all forward ops whose ``block_id``
    matches. Returns 0.0 for blocks that have no measured ops (e.g. non-
    block ops like embedding) — the caller is responsible for handling
    that case with a fallback.
    """
    total_s = 0.0
    for op in trace.op_order:
        if op.block_id != block_id or not op.is_forward:
            continue
        total_s += trace.op_latencies.get(op.op_id, 0.0)
    return total_s


def _fwd_compute_time_from_trace(trace: ProfilerTrace) -> tuple[float, dict[BlockId, float], bool]:
    """Return (total_fwd_compute_s, per_block_compute_s, used_measured).

    Behavior:
    - If the trace carries ``op_latencies`` AND the measured total is not
      larger than the activation-size roofline by more than 2x (which
      indicates the measurement was inflated by cold-start + pre/post-hook
      overhead that the roofline prices out), return the measured
      per-block compute.
    - If measured totals are inflated (common for 7B+ on a single-iter
      profile where JIT + hook dispatch adds multiple seconds of Python
      overhead), fall back to the measured-total rescaled so the
      aggregate matches the roofline budget — this keeps the per-block
      shape from the measurement while bounding absolute magnitude to
      a physically plausible range.
    - If the trace has no measured latencies, use the activation-size
      roofline proxy and return ``used_measured=False`` so the caller
      can log a warning.
    """
    per_block: dict[BlockId, float] = {}
    total = 0.0
    # Always compute the roofline reference; cheap, and used as a sanity cap.
    roofline_per_block: dict[BlockId, float] = {}
    roofline_total = 0.0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        t = _compute_time(act_sz)
        roofline_per_block[bid] = t
        roofline_total += t

    if trace.op_latencies:
        for op in trace.op_order:
            if not op.is_forward or op.block_id is None:
                continue
            lat = trace.op_latencies.get(op.op_id)
            if lat is None:
                continue
            per_block[op.block_id] = per_block.get(op.block_id, 0.0) + lat
            total += lat
        for bid_raw in trace.activation_sizes:
            bid = BlockId(int(bid_raw))
            per_block.setdefault(bid, 0.0)

        if total > 0.0:
            # Cap absolute magnitude at the roofline budget. Single-iter
            # profiling on 7B+ inflates measurements ~8x due to cold kernels
            # and hook dispatch; without the cap the searcher reorders
            # toward offload-everything configs that are worse in reality.
            # Preserve the measurement's per-block SHAPE by scaling uniformly.
            if roofline_total > 0.0 and total > 2.0 * roofline_total:
                scale = roofline_total / total
                per_block = {bid: v * scale for bid, v in per_block.items()}
                total = roofline_total
            return total, per_block, True

    # Fallback: pure roofline. No measurements available (empty op_latencies).
    return roofline_total, roofline_per_block, False


def _bwd_compute_time_from_trace(trace: ProfilerTrace, t_fwd_total: float) -> float:
    """Return the aggregate backward compute time in seconds.

    The profiler's pre/post-forward hooks inflate the measured aggregate
    ``<backward>`` latency by a large factor on transformer-sized models
    (autograd holds the hook-saved tensors, and cpu-side hook dispatch
    during the forward materializes extra intermediates that make the
    backward pass artificially slow on the profile iteration). Using that
    measurement directly steers the searcher toward n_persist=0 configs
    because it inflates ``T_bwd`` uniformly across all configs without
    shifting their ranking.

    For this reason we prefer ``t_fwd_total * _BWD_FWD_COMPUTE_RATIO`` as
    the aggregate backward estimate — the 2x ratio is the canonical
    transformer-block backward/forward rule and is free of hook bias.
    The measured ``<backward>`` latency is retained in ``trace.op_latencies``
    for future calibration (e.g. a non-hook warmup pass).
    """
    return t_fwd_total * _BWD_FWD_COMPUTE_RATIO


def _comm_time_chunk(
    S_chunk: int,
    eff_h2d: float,
    eff_d2h: float,
    nccl_gather_s: float,
    *,
    is_backward: bool,
    buffer_cached: bool,
) -> float:
    """Return the communication time for a single non-persistent chunk.

    Per-chunk cost = NCCL gather (for the shard) + PCIe H2D (CPU->GPU)
    in forward, + PCIe D2H (grad reduce-offload) in backward. Buffer-
    cached chunks skip the backward re-gather.
    """
    # NCCL gather contribution is size-dependent; the trace keys
    # ``nccl_gather_s`` by payload bytes. We pre-selected the right
    # entry in the caller.
    collective = nccl_gather_s

    bw = eff_h2d if not is_backward else eff_d2h
    if bw <= 0:
        # Defensive: avoid division by zero on a pathological profile.
        pcie = 0.0
    else:
        pcie = S_chunk / bw

    if is_backward and buffer_cached:
        # The buffer still has the chunk — no re-gather, just the
        # reduce-offload on the D2H side.
        return pcie
    return collective + pcie


def _pick_nccl(nccl_table: dict, payload_bytes: int) -> float:
    """Look up the nearest payload size in an NCCL latency table.

    ``nccl_table`` is ``{payload_bytes -> seconds}``. If empty, return
    0.0 — single-rank / no-collective case.
    """
    if not nccl_table:
        return 0.0
    # Nearest-size lookup in log space would be fancier; cheapest
    # correct thing is pick the entry whose key is closest.
    best = min(nccl_table.keys(), key=lambda k: abs(int(k) - payload_bytes))
    return float(nccl_table[best])


def estimate_runtime(
    cfg: CostConfig,
    trace: ProfilerTrace,
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    hw: HardwareProfile,
) -> float:
    """Estimate wall-clock iteration time in seconds.

    See module docstring for the equations and accounting rules.
    """
    eff_h2d, eff_d2h = effective_bw(cfg, hw)

    # ----- Per-chunk comm / compute decomposition -----------------------
    n_persist = max(0, min(cfg.n_persist, layout.N_chunk))
    n_buffer = max(0, min(cfg.n_buffer, layout.N_chunk - n_persist))
    n_nonpersist = max(0, layout.N_chunk - n_persist)

    # NCCL table lookup at chunk-payload size. Single-rank -> world==1
    # and the tables should be empty (or contain zero times), yielding
    # 0s here.
    if hw.gpu_count <= 1 or trace.world <= 1:
        nccl_gather = 0.0
        nccl_reduce = 0.0
    else:
        nccl_gather = _pick_nccl(trace.nccl_gather_s, layout.S_chunk)
        nccl_reduce = _pick_nccl(trace.nccl_reduce_s, layout.S_chunk)

    # Non-persistent chunks: forward has gather + H2D.
    t_fwd_comm_per_chunk = _comm_time_chunk(
        layout.S_chunk,
        eff_h2d,
        eff_d2h,
        nccl_gather,
        is_backward=False,
        buffer_cached=False,
    )
    # Backward: buffer-cached chunks (up to n_buffer of them) skip re-
    # gather; the rest pay the full round-trip with reduce-offload.
    t_bwd_comm_per_chunk_cached = _comm_time_chunk(
        layout.S_chunk,
        eff_h2d,
        eff_d2h,
        nccl_reduce,
        is_backward=True,
        buffer_cached=True,
    )
    t_bwd_comm_per_chunk_uncached = _comm_time_chunk(
        layout.S_chunk,
        eff_h2d,
        eff_d2h,
        nccl_reduce,
        is_backward=True,
        buffer_cached=False,
    )

    # ----- Forward compute ---------------------------------------------
    # Forward per-block compute is the SUM of measured op latencies for that
    # block when the profiler recorded them; otherwise the activation-size
    # roofline proxy. SWAP blocks add activation H2D/D2H on top of compute.
    n_block = len(trace.activation_sizes)
    t_fwd_compute_total, per_block_compute, used_measured = _fwd_compute_time_from_trace(
        trace
    )
    if not used_measured:
        LOG.warning(
            "ProTrain: using approximate compute-rate proxy; re-run profiler "
            "for measured latencies"
        )
    t_fwd_swap_transfer = 0.0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.SWAP:
            # Offload activation CPU-side during forward.
            if eff_d2h > 0:
                t_fwd_swap_transfer += act_sz / eff_d2h

    # Per-chunk forward roofline: max(compute per chunk, comm per chunk).
    # Distribute the per-block compute evenly across non-persistent
    # chunks (persistent chunks are counted in compute but have no
    # comm). This is the chunk-level roofline the paper describes.
    if layout.N_chunk > 0:
        t_fwd_compute_per_chunk = t_fwd_compute_total / layout.N_chunk
    else:
        t_fwd_compute_per_chunk = 0.0

    t_fwd_persistent_chunks = n_persist * t_fwd_compute_per_chunk
    t_fwd_nonpersistent_chunks = n_nonpersist * max(
        t_fwd_compute_per_chunk, t_fwd_comm_per_chunk
    )
    t_fwd = (
        t_fwd_persistent_chunks
        + t_fwd_nonpersistent_chunks
        + t_fwd_swap_transfer
    )

    # ----- Backward compute --------------------------------------------
    # Baseline backward: either the measured aggregate <backward> latency
    # from the profiler (preferred) or t_fwd * _BWD_FWD_COMPUTE_RATIO. On
    # top of that, CKPT blocks pay one extra forward per CKPT block (their
    # per-block compute time), and SWAP blocks add the activation prefetch.
    t_bwd_compute_base = _bwd_compute_time_from_trace(trace, t_fwd_compute_total)
    t_bwd_recompute = 0.0
    t_bwd_swap_prefetch = 0.0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            # Recompute the block's forward to restore activations. Use the
            # measured per-block compute when available; fall back to the
            # activation-size proxy for blocks the profiler didn't cover.
            t_block = per_block_compute.get(bid, 0.0)
            if t_block <= 0.0:
                t_block = _compute_time(act_sz)
            t_bwd_recompute += t_block
        elif mode is BlockMode.SWAP:
            if eff_h2d > 0:
                t_bwd_swap_prefetch += act_sz / eff_h2d

    t_bwd_compute_total = t_bwd_compute_base + t_bwd_recompute
    if layout.N_chunk > 0:
        t_bwd_compute_per_chunk = t_bwd_compute_total / layout.N_chunk
    else:
        t_bwd_compute_per_chunk = 0.0

    # Split non-persistent chunks into buffer-cached vs. uncached.
    # Buffer-cached chunks carry forward their GPU residency; up to
    # n_buffer of them skip the re-gather in backward.
    n_cached = min(n_buffer, n_nonpersist)
    n_uncached = n_nonpersist - n_cached

    t_bwd_persistent_chunks = n_persist * t_bwd_compute_per_chunk
    t_bwd_cached_chunks = n_cached * max(
        t_bwd_compute_per_chunk, t_bwd_comm_per_chunk_cached
    )
    t_bwd_uncached_chunks = n_uncached * max(
        t_bwd_compute_per_chunk, t_bwd_comm_per_chunk_uncached
    )
    t_bwd = (
        t_bwd_persistent_chunks
        + t_bwd_cached_chunks
        + t_bwd_uncached_chunks
        + t_bwd_swap_prefetch
    )

    # ----- Optimizer step ----------------------------------------------
    # Model-state bytes per chunk = model_state_bytes / N_chunk.
    if layout.N_chunk > 0:
        ms_per_chunk = trace.model_state_bytes / layout.N_chunk
    else:
        ms_per_chunk = 0.0

    # Prefer the profiler-measured Adam rates on ``HardwareProfile``; fall
    # back to the hardcoded priors when the microbenchmarks returned 0.0
    # (e.g. DeepSpeedCPUAdam compile failure). Log at WARN exactly once
    # per estimate_runtime call so repeated search invocations don't spam.
    if hw.cpu_adam_bytes_per_sec > 0.0:
        cpu_adam_bps = hw.cpu_adam_bytes_per_sec
    else:
        LOG.warning(
            "estimate_runtime: cpu_adam_bytes_per_sec unavailable; using "
            "fallback %.2e (re-run profiler for a calibrated rate)",
            _CPU_ADAM_FALLBACK,
        )
        cpu_adam_bps = _CPU_ADAM_FALLBACK

    if hw.gpu_adam_bytes_per_sec > 0.0:
        gpu_adam_bps = hw.gpu_adam_bytes_per_sec
    else:
        LOG.warning(
            "estimate_runtime: gpu_adam_bytes_per_sec unavailable; using "
            "fallback %.2e (re-run profiler for a calibrated rate)",
            _GPU_ADAM_FALLBACK,
        )
        gpu_adam_bps = _GPU_ADAM_FALLBACK

    t_gpu_optim = n_persist * ms_per_chunk / gpu_adam_bps
    t_cpu_optim = n_nonpersist * ms_per_chunk / cpu_adam_bps

    # Eq. 2: T_iter = T_fwd + max(T_bwd + T_gpu_optim, T_cpu_optim)
    t_iter = t_fwd + max(t_bwd + t_gpu_optim, t_cpu_optim)

    LOG.debug(
        "estimate_runtime: cfg=%s t_fwd=%.4fs t_bwd=%.4fs t_gpu_opt=%.4fs "
        "t_cpu_opt=%.4fs -> t_iter=%.4fs",
        cfg,
        t_fwd,
        t_bwd,
        t_gpu_optim,
        t_cpu_optim,
        t_iter,
    )
    # Silence unused n_block — kept for debug/extension symmetry.
    _ = n_block
    return t_iter


__all__ = ["estimate_runtime"]
