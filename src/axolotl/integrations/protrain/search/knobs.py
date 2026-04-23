"""Bound derivation for the ProTrain 4-knob search (§3.3).

The searcher enumerates ``(n_persist, n_buffer, n_swap, n_checkpoint)``
within the ``Bounds`` returned here:

- ``N_chunk`` — upper bound on ``n_persist`` and ``n_buffer`` (they sum
  to at most ``N_chunk`` since they partition chunks).
- ``N_block`` — upper bound on ``n_swap + n_checkpoint``.
- ``N_interval`` — forward-pass ops per block, used to cap ``n_swap`` by
  how much compute is available to hide prefetch behind.

``Bounds`` is frozen and owned by ``types.py``; do not redefine.
"""

from __future__ import annotations

from collections import Counter

from axolotl.integrations.protrain.types import (
    Bounds,
    ChunkLayout,
    ProfilerTrace,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def derive_bounds(trace: ProfilerTrace, layout: ChunkLayout) -> Bounds:
    """Derive the upper bounds on the 4 knobs.

    Parameters
    ----------
    trace:
        Profiler output. ``op_order`` is scanned to compute
        ``N_interval``; ``activation_sizes`` gives ``N_block``.
    layout:
        Chunk layout. ``N_chunk`` is lifted directly.

    Returns
    -------
    Bounds
        ``Bounds(N_chunk, N_block, N_interval)``.
    """
    n_chunk = int(layout.N_chunk)
    n_block = int(len(trace.activation_sizes))

    # ``N_interval`` is the number of forward ops per block. If
    # activation_sizes is empty (degenerate test input) use 1 to keep
    # downstream arithmetic total.
    if n_block <= 0:
        n_interval = 1
    else:
        per_block: Counter[int] = Counter()
        for op in trace.op_order:
            if op.is_forward and op.block_id is not None:
                per_block[int(op.block_id)] += 1
        if per_block:
            # Average ops per block; round down so bounds stay
            # conservative. Taking the mean (not the min) avoids
            # punishing blocks that happen to contain a single hot op.
            n_interval = max(1, sum(per_block.values()) // len(per_block))
        else:
            # No op has a block_id — fall back to the flat ratio.
            forward_op_count = sum(1 for op in trace.op_order if op.is_forward)
            n_interval = max(1, forward_op_count // max(1, n_block))

    LOG.debug(
        "derive_bounds: N_chunk=%d N_block=%d N_interval=%d",
        n_chunk,
        n_block,
        n_interval,
    )
    return Bounds(N_chunk=n_chunk, N_block=n_block, N_interval=n_interval)


__all__ = ["derive_bounds"]
