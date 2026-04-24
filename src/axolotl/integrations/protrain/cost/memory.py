"""Peak-memory reconstruction for the ProTrain searcher (§3.3, App A.2).

Implements Eqs. 8-10 — an operator-by-operator walk of the forward pass
that tracks live tensors, adds the profiled intra- and inter-op deltas,
and accounts for the per-block activation strategy (NONE / CKPT / SWAP).
Applies Eq. 11 — the ``alpha`` fragmentation factor — as a final
multiplicative over-estimate so the searcher conservatively prunes.

Design contract (see DESIGN.md §Design Decisions):

- ``ALPHA_FRAGMENTATION = 1.10`` matches the paper's "up to 10%
  overestimate on best-selected configurations" claim.
- SWAP blocks do not contribute to the op-walk peak: the paper argues
  swap-in "only fires when memory is available", so activation swapping
  is assumed to trade runtime for zero steady-state peak.
- Gradient checkpointing bumps the peak at the *first* op of each CKPT
  block — this is when recomputation materializes the block's
  activations before the backward pass consumes them.
- ZeRO-3 sharding (``HardwareProfile.zero3_shard=True``) does NOT
  reduce the GPU peak: each rank's gather issues
  ``all_gather_into_tensor`` to reconstruct the full chunk on GPU
  before forward/backward compute, so the buffer-pool residency term
  is identical to the replicated path. Sharding only changes the
  per-rank pinned CPU footprint — see :func:`estimate_cpu_footprint`.
"""

from __future__ import annotations

from collections import defaultdict

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


#: Eq. 11 fragmentation factor — applied as a final multiplier on the
#: raw op-walk peak. Treated as a module-level constant so tests can
#: import it explicitly for sanity checks.
#: Matches the paper's "up to 10% overestimate on best-selected
#: configurations" claim. Previously bumped to 1.20 as an empirical
#: band-aid for backward-peak underprediction; with the M4.5 runtime
#: gaps now closed (per-param grad offload, init-time chunk offload,
#: the BUG-1-4 fixes in ``chunk/manager.py``) the op-walk matches
#: measured peaks tightly enough to restore the paper value — see
#: DESIGN.md §Design Decisions point 1.
ALPHA_FRAGMENTATION: float = 1.10


def _group_ops_by_block(trace: ProfilerTrace) -> dict[BlockId, list[int]]:
    """Return ``{block_id -> [op_positions]}`` for forward ops only.

    ``op_positions`` are indices into ``trace.op_order``; ops that do
    not belong to any block (e.g. embedding, final LM head) are skipped.
    """
    grouped: dict[BlockId, list[int]] = defaultdict(list)
    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            continue
        if op.block_id is None:
            continue
        grouped[op.block_id].append(i)
    return grouped


def estimate_cpu_footprint(
    cfg: CostConfig,
    layout: ChunkLayout,
    hw: HardwareProfile,
) -> int:
    """Per-rank pinned CPU bytes held by non-persistent chunks.

    The non-persistent chunks live on CPU in pinned memory. Under the
    replicated (pre-M7) path every rank holds a FULL copy of each
    non-persistent chunk, so the per-rank footprint is
    ``(N_chunk - n_persist) * S_chunk``. Under the M7 ZeRO-3 sharded
    path each rank holds only ``ceil(chunk_bytes / world_size)`` per
    chunk, so the per-rank footprint divides by ``gpu_count``.

    This accounting is **orthogonal to** :func:`estimate_peak`, which
    models GPU memory: the gather materializes the full chunk on GPU
    via ``all_gather_into_tensor`` regardless of sharding, so GPU peak
    is unchanged by ``zero3_shard``. The real savings from sharding
    appear here (CPU bytes/rank) and in the reduce bandwidth
    (reduce_scatter vs. per-param all_reduce).

    Parameters
    ----------
    cfg:
        Candidate knob configuration. Only ``n_persist`` is consumed —
        ``n_buffer``/``n_swap``/``n_checkpoint`` do not change pinned
        CPU footprint.
    layout:
        Chunk layout. ``S_chunk`` and ``N_chunk`` are read directly.
    hw:
        Hardware profile. Reads ``gpu_count`` and ``zero3_shard``.

    Returns
    -------
    int
        Per-rank pinned CPU bytes. Rounded up via ceiling division so
        the returned value is a conservative upper bound on actual
        shard allocations (shard sizes themselves are rounded up to a
        dtype-aligned boundary by ``ChunkManager.materialize_offload``;
        the arithmetic here tracks the same ceiling).
    """
    non_persist = max(0, layout.N_chunk - cfg.n_persist)
    total_bytes = non_persist * layout.S_chunk
    # Under sharding each rank holds 1/gpu_count of each chunk. Ceiling
    # division so small chunks don't underreport for the trailing rank.
    per_rank_divisor = hw.gpu_count if hw.zero3_shard else 1
    per_rank_divisor = max(1, per_rank_divisor)
    return (total_bytes + per_rank_divisor - 1) // per_rank_divisor


def estimate_peak(
    cfg: CostConfig,
    trace: ProfilerTrace,
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    hw: HardwareProfile,  # noqa: ARG001 - accepted for API symmetry with runtime
) -> int:
    """Estimate steady-state peak GPU memory in bytes.

    Walks ``trace.op_order`` in forward order. At each op the candidate
    peak is:

        model_state_present
        + activations_live_at_op
        + intra_op_delta[op]
        + inter_op_delta[op_prev -> op]

    Then scaled by ``ALPHA_FRAGMENTATION``. See module docstring for the
    SWAP / CKPT accounting rules.

    Parameters
    ----------
    cfg:
        Candidate knob configuration. Only ``n_persist`` and
        ``n_buffer`` are consumed directly here; ``n_swap`` and
        ``n_checkpoint`` show up via ``block_map``.
    trace:
        Output of the M1 profiler. Provides op order, intra/inter deltas,
        per-block activation sizes.
    layout:
        Chunk layout (``S_chunk``, ``N_chunk``).
    block_map:
        Per-block mode assignment (output of ``assign_modes``).
    hw:
        Hardware profile — currently unused, accepted for API symmetry
        with ``estimate_runtime`` so the searcher can call both with the
        same argument pack.

    Returns
    -------
    int
        Peak bytes, rounded via ``int(alpha * raw_peak)``.
    """
    # --- Static model-state footprint ----------------------------------
    # Persistent chunks are always on GPU. Non-persistent chunks only
    # occupy GPU memory through the buffer pool, so their GPU residency
    # is ``n_buffer * S_chunk`` not ``(N_chunk - n_persist) * S_chunk``.
    # Clamp n_persist/n_buffer into [0, N_chunk] defensively — the
    # searcher should never violate these, but other callers may.
    n_persist = max(0, min(cfg.n_persist, layout.N_chunk))
    n_buffer = max(0, min(cfg.n_buffer, layout.N_chunk - n_persist))
    model_state_present = (n_persist + n_buffer) * layout.S_chunk

    # --- Per-block activation policy -----------------------------------
    # NONE / CKPT / SWAP blocks contribute differently to the live set:
    #   NONE: full activation bytes retained from fwd to bwd.
    #   CKPT: 0 bytes retained; bumps peak at first op of this block.
    #   SWAP: 0 bytes retained in steady state (see module docstring).
    n_block = len(trace.activation_sizes)
    forward_ops_by_block = _group_ops_by_block(trace)

    # Resolve "first op index" for each CKPT block; used to schedule the
    # checkpoint recomputation bump. If the block has no ops (degenerate
    # test input) the bump lands at op index -1 and is ignored below.
    ckpt_bump_op: dict[int, int] = {}
    for block_id, op_idxs in forward_ops_by_block.items():
        if not op_idxs:
            continue
        mode = block_map.get(block_id, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            ckpt_bump_op[op_idxs[0]] = int(block_id)

    # Retained-activation contribution from NONE blocks — constant across
    # the op-walk (these activations are live from their first op
    # through the end of forward).
    retained_none_bytes = 0
    for block_id_raw, act_sz in trace.activation_sizes.items():
        # ``activation_sizes`` is typed ``dict[BlockId, int]`` but
        # pickled maps may use int keys; normalize.
        bid = BlockId(int(block_id_raw))
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE:
            retained_none_bytes += act_sz
        # CKPT: only live during its recomputation window -> handled
        #       by the per-op bump below.
        # SWAP: live only during the block's forward compute; assumed
        #       to overlap free GPU memory (§3.3).

    # --- Op walk -------------------------------------------------------
    raw_peak = 0
    # Track activations that are "live as of op i". We build this
    # incrementally so ops inside a NONE block see that block's
    # activation bytes accumulate progressively (safer upper bound even
    # though the end-of-fwd sum already accounts for all of it). The
    # simplest correct accounting is:
    #
    #   live_at_op = retained_none_bytes_accumulated_up_to_block(op)
    #              + ckpt_bump_if_this_op_triggers
    #
    # We pre-compute the cumulative "NONE activations active by this
    # point in forward" by walking blocks in order.

    # Map op index -> cumulative NONE-activation bytes active at or
    # before this op. Blocks without a position in forward_ops_by_block
    # contribute no ordering, so we sort blocks by their first forward
    # op index.
    block_first_op = {
        bid: ops[0] for bid, ops in forward_ops_by_block.items() if ops
    }
    blocks_in_fwd_order = sorted(block_first_op.items(), key=lambda kv: kv[1])

    cumulative_none: list[tuple[int, int]] = []  # (first_op_idx, cumulative_bytes)
    running = 0
    for bid, first_idx in blocks_in_fwd_order:
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE:
            running += trace.activation_sizes.get(bid, 0)
        cumulative_none.append((first_idx, running))

    def _none_live_at(op_idx: int) -> int:
        """Cumulative NONE-block activation bytes at or before op_idx."""
        # Linear scan is fine; cumulative_none has at most N_block
        # entries (8-256 in realistic workloads).
        live = 0
        for first_idx, cum in cumulative_none:
            if first_idx <= op_idx:
                live = cum
            else:
                break
        return live

    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            # Backward-only ops are out of scope for the forward
            # op-walk. Eq. 8-10 explicitly walk forward ops.
            continue

        intra = trace.intra_op_delta.get(op.op_id, 0)
        inter = trace.inter_op_delta.get(op.op_id, 0)
        live_none = _none_live_at(i)

        # CKPT bump: when we hit the first op of a CKPT block, the
        # recomputation materializes that block's activations *in
        # addition to* any retained activations. This models the peak
        # during the backward-driven recomp window that lines up with
        # this op's forward-equivalent workload.
        ckpt_extra = 0
        if i in ckpt_bump_op:
            ckpt_extra = trace.activation_sizes.get(
                BlockId(ckpt_bump_op[i]), 0
            )

        candidate = (
            model_state_present
            + live_none
            + ckpt_extra
            + intra
            + inter
        )
        if candidate > raw_peak:
            raw_peak = candidate

    # If the trace has no forward ops (degenerate test input) fall back
    # to a static estimate. This keeps the function total.
    if raw_peak == 0:
        raw_peak = model_state_present + retained_none_bytes

    # Ground-truth forward cap from the profiler's hook-less steady pass.
    #
    # Per-block cap (TRACE_VERSION>=6): lightweight block-level hooks during
    # the steady forward record each block's peak bytes. The MAX across
    # those per-block peaks is a strict upper bound on the forward peak
    # regardless of which blocks are NONE/CKPT/SWAP — CKPT and SWAP blocks
    # free their activations before the next block runs, so a mixed
    # configuration's forward peak can never exceed the per-block max
    # observed under the all-NONE profile. CKPT blocks do add a
    # recomputation peak during BACKWARD (one block's activations
    # rematerialized at a time, serially), which isn't captured during
    # this forward-only measurement — add the max single-CKPT-block
    # activation bytes on top.
    #
    # This supersedes the v5 aggregate-only cap (which only applied when
    # n_checkpoint==0 && n_swap==0, making it a no-op for the 7B LoRA
    # test where the searcher picks n_checkpoint≈9). With per-block data
    # the cap tightens ALL configs, including fractional-NONE.
    #
    # Fallback order:
    #   1. Per-block dict populated (v6+) -> use forward_max_block + ckpt_bump
    #   2. Aggregate-only populated (v5, or v6 when discover_blocks failed)
    #      AND all-NONE cfg -> use aggregate
    #   3. Neither -> preserve op-walk raw_peak
    if trace.steady_fwd_block_peak_bytes:
        forward_max_block_peak = max(trace.steady_fwd_block_peak_bytes.values())
        # Max single-CKPT-block activation bytes. Backward replays CKPT
        # blocks one at a time, so the bump is per-block not summed.
        # (This mirrors the op-walk's ckpt_extra, which adds a single
        # block's activation at the first op of each CKPT block and
        # takes the max across op positions.)
        ckpt_recomp_bump = 0
        for bid_raw, act_sz in trace.activation_sizes.items():
            bid = BlockId(int(bid_raw))
            if block_map.get(bid, BlockMode.NONE) is BlockMode.CKPT:
                if act_sz > ckpt_recomp_bump:
                    ckpt_recomp_bump = act_sz
        measured_cap = forward_max_block_peak + ckpt_recomp_bump
        if raw_peak > measured_cap:
            raw_peak = measured_cap
    elif (
        trace.steady_fwd_peak_bytes > 0
        and cfg.n_checkpoint == 0
        and cfg.n_swap == 0
        and raw_peak > trace.steady_fwd_peak_bytes
    ):
        raw_peak = trace.steady_fwd_peak_bytes

    scaled = int(ALPHA_FRAGMENTATION * raw_peak)
    LOG.debug(
        "estimate_peak: n_persist=%d n_buffer=%d n_swap=%d n_ckpt=%d raw=%dB alpha=%.2f -> %dB",
        cfg.n_persist,
        cfg.n_buffer,
        cfg.n_swap,
        cfg.n_checkpoint,
        raw_peak,
        ALPHA_FRAGMENTATION,
        scaled,
    )
    # Silence the unused-var warning when trace has no forward ops.
    _ = n_block
    return scaled


__all__ = ["estimate_peak", "estimate_cpu_footprint", "ALPHA_FRAGMENTATION"]
