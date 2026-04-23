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

# GPU compute throughput is embedded implicitly in the profiled op-walk:
# the paper derives per-chunk compute time from the summed op latencies
# inside that chunk. Since our ProfilerTrace does not currently carry
# per-op latency, we treat activation size as a proxy for compute work,
# scaled by this factor (bytes of activation per second of GPU compute).
# This is a load-bearing approximation: M6 should replace it once the
# profiler records per-op timing. Until then the cost model produces
# relative orderings that are correct for the knob-comparison use case
# — absolute iteration time will drift from measurement.
_COMPUTE_BYTES_PER_SEC: float = 3.0e11  # ~300 GB/s, rough 3090 effective

# CPU-Adam step throughput (bytes of optim-state processed per second).
# DeepSpeedCPUAdam benches around 1-2 GB/s per step on a decent Xeon/
# Threadripper. Conservative.
_CPU_ADAM_BYTES_PER_SEC: float = 1.5e9

# GPU FusedAdam throughput. Limited by HBM bandwidth, not FLOPs.
_GPU_ADAM_BYTES_PER_SEC: float = 5.0e11


def _compute_time(activation_bytes: int) -> float:
    """Rough compute time proxy — see module constants."""
    return activation_bytes / _COMPUTE_BYTES_PER_SEC


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
    # Forward per-block compute approximated from activation size. SWAP
    # blocks add activation H2D/D2H on top of their compute.
    n_block = len(trace.activation_sizes)
    t_fwd_compute_total = 0.0
    t_fwd_swap_transfer = 0.0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        t_block_compute = _compute_time(act_sz)
        t_fwd_compute_total += t_block_compute
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
    # Backward compute == forward compute (standard assumption) plus
    # recomputation for each CKPT block plus SWAP prefetch.
    t_bwd_compute_base = t_fwd_compute_total  # same workload going back
    t_bwd_recompute = 0.0
    t_bwd_swap_prefetch = 0.0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            # Recompute the block's forward to restore activations.
            t_bwd_recompute += _compute_time(act_sz)
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
    t_gpu_optim = n_persist * ms_per_chunk / _GPU_ADAM_BYTES_PER_SEC
    t_cpu_optim = n_nonpersist * ms_per_chunk / _CPU_ADAM_BYTES_PER_SEC

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
