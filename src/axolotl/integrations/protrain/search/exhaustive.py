"""Exhaustive 4-knob search for ProTrain (§3.3).

Algorithm:

1. Derive ``Bounds`` from ``(trace, layout)``.
2. Enumerate ``(n_persist, n_buffer, n_swap, n_checkpoint)`` within
   bounds, subject to:

   - ``n_persist + n_buffer <= N_chunk``
   - ``n_swap + n_checkpoint <= N_block``
   - ``n_swap <= min(N_block - n_checkpoint, N_interval)``

3. For each candidate, compute ``block_map = assign_modes(...)``.
4. Evaluate ``estimate_peak``; drop candidates above ``capacity_bytes``.
5. Among survivors, evaluate ``estimate_runtime`` and pick argmin.
6. Raise ``RuntimeError`` if no candidate fits.

The search space is tiny (~10^4 at most on realistic models) — no
pruning cleverness is needed for correctness. We do sort candidates
by a cheap static peak estimate so early OOMs filter out large chunks
of the space without the full op-walk.
"""

from __future__ import annotations

from typing import Iterator

from collections import defaultdict

from axolotl.integrations.protrain.block.layout_rules import assign_modes
from axolotl.integrations.protrain.cost.memory import estimate_peak  # noqa: F401 - re-exported for test back-compat
from axolotl.integrations.protrain.cost.runtime import estimate_runtime
from axolotl.integrations.protrain.search.knobs import derive_bounds
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    Bounds,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ProfilerTrace,
    SearchResult,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _iter_candidates(bounds: Bounds) -> Iterator[CostConfig]:
    """Enumerate feasible ``CostConfig`` tuples within ``bounds``."""
    n_chunk = bounds.N_chunk
    n_block = bounds.N_block
    n_interval = bounds.N_interval

    for n_ckpt in range(0, n_block + 1):
        # n_swap bounded by (a) blocks remaining after ckpt, (b) N_interval.
        max_swap = min(n_block - n_ckpt, n_interval)
        for n_swap in range(0, max_swap + 1):
            for n_persist in range(0, n_chunk + 1):
                # n_buffer fills the remainder of chunk budget.
                max_buffer = n_chunk - n_persist
                for n_buffer in range(0, max_buffer + 1):
                    yield CostConfig(
                        n_persist=n_persist,
                        n_buffer=n_buffer,
                        n_swap=n_swap,
                        n_checkpoint=n_ckpt,
                    )


def _block_map_peak_contribution(
    block_map: BlockStrategyMap, trace: ProfilerTrace
) -> int:
    """Compute the block-map-dependent part of the raw peak.

    Matches the op-walk inside :func:`estimate_peak` but returns only
    the terms that do not depend on ``(n_persist, n_buffer)``:

        F(block_map) = max over forward ops i of
            (live_none_at(i) + ckpt_extra_at(i) + intra[i] + inter[i])

    The returned value is the pre-alpha raw contribution; the caller
    multiplies the full ``model_state_present + F`` sum by
    ``ALPHA_FRAGMENTATION`` and ``int()``-casts to match
    ``estimate_peak`` exactly.
    """
    # Group forward ops by block.
    forward_ops_by_block: dict[BlockId, list[int]] = defaultdict(list)
    for i, op in enumerate(trace.op_order):
        if op.is_forward and op.block_id is not None:
            forward_ops_by_block[op.block_id].append(i)

    # Identify CKPT bump ops.
    ckpt_bump_op: dict[int, int] = {}
    for block_id, op_idxs in forward_ops_by_block.items():
        if not op_idxs:
            continue
        if block_map.get(block_id, BlockMode.NONE) is BlockMode.CKPT:
            ckpt_bump_op[op_idxs[0]] = int(block_id)

    # Cumulative NONE-block activation bytes at each forward-op index.
    block_first_op = {
        bid: ops[0] for bid, ops in forward_ops_by_block.items() if ops
    }
    blocks_in_fwd_order = sorted(block_first_op.items(), key=lambda kv: kv[1])
    cumulative_none: list[tuple[int, int]] = []  # (first_op_idx, cumulative)
    running = 0
    for bid, first_idx in blocks_in_fwd_order:
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE:
            running += trace.activation_sizes.get(bid, 0)
        cumulative_none.append((first_idx, running))

    def _none_live_at(op_idx: int) -> int:
        live = 0
        for first_idx, cum in cumulative_none:
            if first_idx <= op_idx:
                live = cum
            else:
                break
        return live

    best = 0
    have_any_forward = False
    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            continue
        have_any_forward = True
        intra = trace.intra_op_delta.get(op.op_id, 0)
        inter = trace.inter_op_delta.get(op.op_id, 0)
        live_none = _none_live_at(i)
        ckpt_extra = 0
        if i in ckpt_bump_op:
            ckpt_extra = trace.activation_sizes.get(
                BlockId(ckpt_bump_op[i]), 0
            )
        candidate = live_none + ckpt_extra + intra + inter
        if candidate > best:
            best = candidate

    if not have_any_forward:
        # Degenerate trace: fall back to the NONE retained-activation
        # total so the caller's peak is at least ``model_state_present +
        # retained``.
        total_none = 0
        for bid_raw, act_sz in trace.activation_sizes.items():
            bid = BlockId(int(bid_raw))
            if block_map.get(bid, BlockMode.NONE) is BlockMode.NONE:
                total_none += act_sz
        return total_none

    return best


def _quick_peak_proxy(
    cfg: CostConfig, trace: ProfilerTrace, layout: ChunkLayout
) -> int:
    """Cheap ordering key for memory-ascending enumeration.

    Not used for correctness — the full ``estimate_peak`` is always
    called. Used only to sort candidates so we walk small-peak configs
    first, which tightens log output when we report "evaluated N
    feasible".
    """
    model_state = (cfg.n_persist + cfg.n_buffer) * layout.S_chunk
    avg_act = (
        sum(trace.activation_sizes.values()) / max(1, len(trace.activation_sizes))
    )
    # CKPT and SWAP both reduce retained activations.
    retained_blocks = (
        len(trace.activation_sizes) - cfg.n_checkpoint - cfg.n_swap
    )
    retained_bytes = int(max(0, retained_blocks) * avg_act)
    return model_state + retained_bytes


def search(
    trace: ProfilerTrace,
    layout: ChunkLayout,
    capacity_bytes: int,
    hw: HardwareProfile,
) -> SearchResult:
    """Return the minimum-runtime ``SearchResult`` fitting under
    ``capacity_bytes``.

    Raises
    ------
    RuntimeError
        If no candidate has ``predicted_peak_bytes <= capacity_bytes``.

    Notes
    -----
    Correctness is equivalent to the naive 4-loop enumeration that
    calls ``estimate_peak`` and ``estimate_runtime`` inside the inner
    (n_persist, n_buffer) iteration. We exploit two structural
    invariants to avoid quadratic op-walks across the full search
    space:

    1. ``estimate_peak``'s raw peak decomposes as
       ``(n_persist + n_buffer) * S_chunk + F(block_map)``. The
       block-map-dependent term ``F`` is independent of
       ``(n_persist, n_buffer)`` so we compute it once per
       ``(n_swap, n_ckpt)`` pair (O(N_swap*N_ckpt*N_op)).
    2. ``estimate_runtime`` is a closed-form function of the config,
       evaluated only for configs that already clear the capacity
       gate — keeping the inner loop purely arithmetic.

    For a 7B-class model this cuts the search from ~50 billion op-walk
    iterations down to ~1 million, without changing the selected
    ``(cfg, block_map)``.
    """
    bounds = derive_bounds(trace, layout)

    n_total = 0
    n_feasible = 0
    best_iter_s: float = float("inf")
    best_cfg: CostConfig | None = None
    best_block_map: BlockStrategyMap | None = None
    best_peak: int = 0

    # Pre-compute block-map-dependent terms once per (n_swap, n_ckpt).
    # ``F(block_map)`` is the raw-peak contribution excluding the
    # ``(n_persist + n_buffer) * S_chunk`` term, pre-alpha.
    from axolotl.integrations.protrain.cost.memory import ALPHA_FRAGMENTATION

    alpha = ALPHA_FRAGMENTATION
    s_chunk = layout.S_chunk

    for n_ckpt in range(0, bounds.N_block + 1):
        max_swap = min(bounds.N_block - n_ckpt, bounds.N_interval)
        for n_swap in range(0, max_swap + 1):
            block_map = assign_modes(n_swap, n_ckpt, bounds.N_block)
            # F_bm: max over forward ops of
            #   live_none + ckpt_extra + intra + inter
            f_bm = _block_map_peak_contribution(block_map, trace)

            # For a fixed (n_ckpt, n_swap) sweep n_persist. The optimal
            # n_buffer at each n_persist is the maximum feasible value
            # in [0, N_chunk - n_persist]: ``estimate_runtime``'s
            # n_buffer dependence enters only through ``n_cached =
            # min(n_buffer, n_nonpersist)`` inside the backward
            # communication term, and
            # ``max(compute, comm_cached) <= max(compute, comm_uncached)``
            # because cached chunks skip the re-gather. So moving a
            # chunk from uncached to cached never increases ``t_iter``;
            # the argmin is reached by maximising n_buffer within
            # capacity. That collapses the inner (n_persist, n_buffer)
            # loop from O(N_chunk^2) to O(N_chunk), which is the
            # difference between finishing in ~1s and ~10min on 7B
            # configurations where ``N_chunk`` lands in the hundreds.
            #
            # Peak bound on (n_persist + n_buffer):
            #   int(alpha * (sum * S_chunk + F_bm)) <= capacity
            #   => sum <= floor((capacity/alpha - F_bm) / S_chunk)
            if alpha > 0 and s_chunk > 0:
                max_sum = int((capacity_bytes / alpha - f_bm) / s_chunk)
            else:
                max_sum = bounds.N_chunk
            max_sum = max(0, min(max_sum, bounds.N_chunk))

            for n_persist in range(0, bounds.N_chunk + 1):
                # Max feasible n_buffer at this n_persist.
                max_buffer = min(bounds.N_chunk - n_persist, max_sum - n_persist)
                if max_buffer < 0:
                    # n_persist alone exceeds the capacity budget — any
                    # larger n_persist will too; stop scanning.
                    break

                # Optimum n_buffer is the max feasible (see rationale
                # above). Also evaluate n_buffer=0 as a sanity boundary
                # — in the degenerate case where cached and uncached
                # times are identical the two are equivalent, but we
                # pay the arithmetic anyway so the tie-breaker is
                # deterministic.
                for n_buffer in {max_buffer, 0}:
                    n_total += 1
                    model_state_present = (n_persist + n_buffer) * s_chunk
                    raw_peak = model_state_present + f_bm
                    predicted_peak = (
                        int(alpha * raw_peak) if raw_peak > 0 else 0
                    )
                    if predicted_peak > capacity_bytes:
                        continue
                    n_feasible += 1
                    cfg = CostConfig(
                        n_persist=n_persist,
                        n_buffer=n_buffer,
                        n_swap=n_swap,
                        n_checkpoint=n_ckpt,
                    )
                    predicted_iter_s = estimate_runtime(
                        cfg, trace, layout, block_map, hw
                    )
                    if predicted_iter_s < best_iter_s:
                        best_iter_s = predicted_iter_s
                        best_cfg = cfg
                        best_block_map = block_map
                        best_peak = predicted_peak

    if best_cfg is None or best_block_map is None:
        raise RuntimeError(
            "no feasible ProTrain config under capacity_bytes="
            f"{capacity_bytes} (evaluated {n_total} configs)"
        )

    LOG.info(
        "ProTrain search: evaluated %d configs, %d feasible, picked %s "
        "predicted=%dMB %.3fs",
        n_total,
        n_feasible,
        best_cfg,
        best_peak // (1 << 20),
        best_iter_s,
    )
    return SearchResult(
        cfg=best_cfg,
        block_map=best_block_map,
        predicted_peak_bytes=best_peak,
        predicted_iter_s=best_iter_s,
    )


__all__ = ["search"]
