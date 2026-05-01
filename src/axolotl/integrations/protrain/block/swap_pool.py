"""Pinned-RAM activation pool for the SWAP block path (§3.1.2).

The SWAP wrapper offloads each forward block's output activation to
pinned host memory, then prefetches it back during backward. To make
the D2H copy non-blocking and to give PyTorch a stable pointer to copy
into, we pre-allocate one large pinned host region and hand out fixed-
size slots from it.

This pool is independent of the chunk-buffer pool: the chunk pool
holds parameter slabs (sized to ``S_chunk``), the activation pool
holds activations (sized to ``max_activation_bytes`` per slot). The
two pools never share a slot and are sized independently from the
searcher's decision (``n_swap`` and ``prefetch_depth``).

Lifecycle
---------
Constructed by ``protrain_model_wrapper`` once it knows
``result.cfg.n_swap > 0``. A single :class:`PinnedHostMemory` backs
the entire pool; slots are uint8 narrow views into that region.
Tensors are hashed into slots via :meth:`acquire`; the consumer must
call :meth:`release` (typically inside autograd backward) to return
the slot to the free list. The pool is closed at scheduler tear-down
or ``WrappedModel`` GC, releasing the pinned region.

Sizing
------
``slot_bytes`` is the worst-case activation bytes per SWAP block (the
maximum across the searcher's chosen swap-band of blocks). ``n_slot``
is ``n_swap * prefetch_depth``: each SWAP block needs ``prefetch_depth``
slots in flight (one for the activation in CPU residency, plus one for
each pre-fetched H2D buffer the scheduler stages). For ``option 2A``
(minimum-viable single-block lookahead) ``prefetch_depth = 2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


class ActivationSwapPool:
    """Fixed-size pinned-host slot pool for SWAP-block activations.

    Parameters
    ----------
    n_swap:
        Number of SWAP blocks the searcher selected. Must be ``>= 1``;
        callers should not construct a pool when ``n_swap == 0``.
    slot_bytes:
        Worst-case activation bytes per SWAP block, in bytes. The pool
        sizes every slot to exactly this value so any SWAP block's
        activation fits any slot.
    prefetch_depth:
        How many slots per SWAP block to keep in flight. ``2`` is the
        minimum-viable single-block lookahead (one slot holds the
        currently-resident CPU copy, one slot is being H2D-fetched for
        the next block in backward). ``1`` collapses to fully-serial
        SWAP — only useful for unit tests.

    Notes
    -----
    The pool is **stream-agnostic** — copies onto/from slots happen on
    the SWAP wrapper's chosen stream (typically the scheduler's
    ``_swap_stream``). Slot ownership is tracked by Python-side ID
    only; CUDA never sees the pool's free-list state. Callers MUST
    synchronize the swap stream with their consumer before
    ``release`` reuses the slot for a fresh acquire — otherwise the
    in-flight D2H/H2D may race against the next acquire's writes.
    """

    def __init__(
        self, n_swap: int, slot_bytes: int, prefetch_depth: int = 2
    ) -> None:
        if n_swap < 1:
            raise ValueError(f"n_swap must be >= 1, got {n_swap}")
        if slot_bytes <= 0:
            raise ValueError(f"slot_bytes must be positive, got {slot_bytes}")
        if prefetch_depth < 1:
            raise ValueError(
                f"prefetch_depth must be >= 1, got {prefetch_depth}"
            )

        self.n_swap = int(n_swap)
        self.slot_bytes = int(slot_bytes)
        self.prefetch_depth = int(prefetch_depth)
        self.n_slot = self.n_swap * self.prefetch_depth

        # Backing pinned-host region (split into ``n_slot`` equal slots).
        self._pinned = PinnedHostMemory(
            n_buffer=self.n_slot, S_chunk=self.slot_bytes
        )
        self._closed = False
        # Free-list of available slot indices. We use a plain list as a
        # LIFO stack — locality of reuse is irrelevant for pinned host
        # memory (no allocator state to amortize), and a list is
        # cheaper than a deque for the small N_slot we work with
        # (typically <= 16).
        self._free: list[int] = list(range(self.n_slot))
        self._inflight: int = 0

        LOG.debug(
            "ActivationSwapPool: n_swap=%d slot_bytes=%d prefetch_depth=%d "
            "n_slot=%d total_bytes=%d precise=%s",
            self.n_swap,
            self.slot_bytes,
            self.prefetch_depth,
            self.n_slot,
            self.n_slot * self.slot_bytes,
            self._pinned.is_precise_size,
        )

    def acquire(self) -> tuple[int, "torch.Tensor"]:
        """Reserve a slot; return ``(slot_id, pinned_uint8_view)``.

        The returned tensor is a 1-D ``uint8`` view of length
        ``slot_bytes`` over the pinned region. Callers reshape it to
        their target dtype with ``.view(dtype).reshape(shape)`` after
        copying via ``.copy_(src, non_blocking=True)`` on the swap stream.
        """
        if self._closed:
            raise RuntimeError("ActivationSwapPool is closed")
        if not self._free:
            raise RuntimeError(
                f"ActivationSwapPool exhausted (n_slot={self.n_slot}, "
                f"in-flight={self._inflight}); increase prefetch_depth or "
                "verify the SWAP wrapper releases slots after backward."
            )
        slot_id = self._free.pop()
        self._inflight += 1
        return slot_id, self._pinned.buffer(slot_id)

    def release(self, slot_id: int) -> None:
        """Return ``slot_id`` to the free list. Idempotent on bad ids.

        The caller is responsible for ensuring no in-flight CUDA
        operation references this slot before calling — the pool does
        NOT issue stream syncs.
        """
        if self._closed:
            return
        if not 0 <= slot_id < self.n_slot:
            LOG.warning(
                "ActivationSwapPool.release: slot_id %d out of range [0, %d); ignored",
                slot_id,
                self.n_slot,
            )
            return
        if slot_id in self._free:
            # Defensive: double-release. Log loudly because this likely
            # signals a swap-wrapper bug (e.g. backward executed twice
            # because of a retain_graph=True replay).
            LOG.warning(
                "ActivationSwapPool.release: slot %d already free; double-release",
                slot_id,
            )
            return
        self._free.append(slot_id)
        self._inflight -= 1

    @property
    def total_bytes(self) -> int:
        """Total pinned-host bytes held by the pool."""
        return self.n_slot * self.slot_bytes

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def inflight_count(self) -> int:
        return self._inflight

    def close(self) -> None:
        """Free the pinned region. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._pinned.close()
        self._free.clear()
        self._inflight = 0

    def __del__(self) -> None:  # noqa: D401
        try:
            self.close()
        except Exception:  # noqa: BLE001 — destructor must not throw
            pass


__all__ = ["ActivationSwapPool"]
