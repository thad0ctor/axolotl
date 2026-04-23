"""S_chunk grid search over the {32, 64, 128, 256} MB grid (Appendix B.1).

We simulate the layout for each candidate and pick the candidate that
minimizes fragmentation waste — summed ``S_chunk - bytes_used`` across
non-full chunks. The full simulation is identical to ``build_layout`` but
without needing a model handle: the input is a ``{ParamId -> bytes}`` map.
"""

from __future__ import annotations

from typing import Mapping

from axolotl.integrations.protrain.types import ParamId
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Paper-specified grid; also duplicated in DESIGN.md §Design Decisions.
DEFAULT_GRID: tuple[int, ...] = (32 << 20, 64 << 20, 128 << 20, 256 << 20)


def _simulate_waste(sizes_in_order: list[int], S_chunk: int) -> int:
    """Return total fragmentation waste for a greedy-fit layout.

    Mirrors the non-block-grouped ``build_layout`` inner loop: open a fresh
    chunk once the next param wouldn't fit. The last chunk's trailing slack
    is *not* counted as waste — it's just the natural tail and the caller
    can't recover bytes by picking a different ``S_chunk``. Every earlier
    chunk contributes ``S_chunk - bytes_used``.
    """
    if S_chunk <= 0:
        raise ValueError(f"S_chunk must be positive, got {S_chunk}")

    chunk_bytes: list[int] = [0]
    for sz in sizes_in_order:
        cur = chunk_bytes[-1]
        if cur > 0 and cur + sz > S_chunk:
            chunk_bytes.append(0)
        chunk_bytes[-1] += sz

    if len(chunk_bytes) <= 1:
        return 0
    # Exclude the tail chunk from waste accounting — its slack is inherent.
    return sum(max(0, S_chunk - b) for b in chunk_bytes[:-1])


def pick_S_chunk(
    model_state_bytes_per_param: Mapping[ParamId, int],
    candidates: tuple[int, ...] = DEFAULT_GRID,
) -> int:
    """Pick the ``S_chunk`` from ``candidates`` minimizing fragmentation waste.

    Ties are broken by picking the *larger* candidate — fewer chunks means
    less scheduler overhead and larger individual H2D transfers, both of
    which are strictly preferable at equal waste (App B.1 motivation).
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    # Dict iteration order is insertion order (Python 3.7+), which matches
    # the caller's intended layout order. If the caller wants exec-order
    # simulation, they should pass an exec-ordered dict.
    sizes_in_order = list(model_state_bytes_per_param.values())

    best_S = candidates[0]
    best_waste = _simulate_waste(sizes_in_order, best_S)
    for S in candidates[1:]:
        waste = _simulate_waste(sizes_in_order, S)
        if waste < best_waste or (waste == best_waste and S > best_S):
            best_S = S
            best_waste = waste

    LOG.debug(
        "pick_S_chunk: selected %d bytes (waste=%d) from grid %s",
        best_S,
        best_waste,
        candidates,
    )
    return best_S


__all__ = ["pick_S_chunk", "DEFAULT_GRID"]
