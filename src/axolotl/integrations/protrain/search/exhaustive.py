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

from axolotl.integrations.protrain.block.layout_rules import assign_modes
from axolotl.integrations.protrain.cost.memory import estimate_peak
from axolotl.integrations.protrain.cost.runtime import estimate_runtime
from axolotl.integrations.protrain.search.knobs import derive_bounds
from axolotl.integrations.protrain.types import (
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
    """
    bounds = derive_bounds(trace, layout)

    # Enumerate, sort by cheap proxy, then evaluate full peak.
    candidates = list(_iter_candidates(bounds))
    candidates.sort(key=lambda c: _quick_peak_proxy(c, trace, layout))

    n_total = len(candidates)
    n_feasible = 0
    best_iter_s: float = float("inf")
    best_cfg: CostConfig | None = None
    best_block_map: BlockStrategyMap | None = None
    best_peak: int = 0

    for cfg in candidates:
        block_map = assign_modes(cfg.n_swap, cfg.n_checkpoint, bounds.N_block)
        predicted_peak = estimate_peak(cfg, trace, layout, block_map, hw)
        if predicted_peak > capacity_bytes:
            continue

        n_feasible += 1
        predicted_iter_s = estimate_runtime(cfg, trace, layout, block_map, hw)
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
