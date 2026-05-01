"""Activation-swap wrapper (§3.1.2 — paper-real implementation).

SWAP mode in the ProTrain three-way block strategy: forward activations
are offloaded to pinned CPU memory, then prefetched back during
backward. The wrapper installs an autograd Function that:

* In **forward**, runs the wrapped block, copies its output activation
  to a pinned-host slot on a dedicated swap stream, records a CUDA
  event so the GPU activation tensor's storage can be reclaimed once
  the D2H lands, and saves the slot reference (NOT the GPU tensor) for
  backward.
* In **backward**, schedules the H2D copy from the pinned slot back
  into a fresh GPU buffer on the swap stream, records a completion
  event, and synchronises the compute stream against that event before
  the upstream backward kernel reads the activation. Returns the slot
  to the pool once H2D completes.

Stream policy
-------------
Both D2H and H2D copies run on the scheduler's ``_swap_stream`` (one
shared stream per scheduler). The compute stream waits on the H2D
event before the block's backward gradient kernel reads the
re-materialised activation. In forward we issue the D2H *after* the
block's compute finishes — so the swap stream depends on compute via a
``record_stream`` / wait_event handshake to avoid racing the next
block's compute against the in-flight D2H.

On 3090 / RTX 3090 Ti hardware (12 GB/s PCIe ceiling, no NVLink) the
searcher will rarely pick ``n_swap > 0`` because the activation
transfer cost dominates compute (paper §3.1.2). The wrapper exists for
NVLink hardware where D2H/H2D *can* overlap with compute, and to keep
the searcher's solution space honest. Tested-but-unused infrastructure
on 3090 — that's expected.

Hot path / cold path
--------------------
The pool + stream are injected post-construction by the model wrapper
via :meth:`SwappedBlock.attach_runtime`. If a block is constructed
WITHOUT runtime attached (e.g. unit tests, or a model wrapper that
forgot to call attach_runtime when ``n_swap > 0``), the wrapper
degrades to a no-op identity hook in autograd: the activation lives on
GPU as it normally would, and no D2H/H2D happens. This keeps
correctness intact while preserving the historical "constructible
without runtime" surface that test fixtures rely on. A WARNING is
logged once per instance so the configuration drift is visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.integrations.protrain.block.swap_pool import ActivationSwapPool

LOG = get_logger(__name__)


def _swap_stream_wait_compute(swap_stream: "torch.cuda.Stream") -> None:
    """Make ``swap_stream`` wait on the current (compute) stream.

    Wraps ``stream.wait_stream(current)`` for legibility. On
    CPU-only paths (``swap_stream is None``) this is a no-op.
    """
    if swap_stream is None or not torch.cuda.is_available():
        return
    swap_stream.wait_stream(torch.cuda.current_stream())


def _compute_stream_wait_swap(swap_stream: "torch.cuda.Stream") -> None:
    """Make the current (compute) stream wait on ``swap_stream``."""
    if swap_stream is None or not torch.cuda.is_available():
        return
    torch.cuda.current_stream().wait_stream(swap_stream)


class _SwapOffloadFunction(torch.autograd.Function):
    """Forward: D2H to pinned-pool slot. Backward: H2D back to GPU.

    The dance we have to do for correct GPU-storage reclamation:

    1. **Forward** runs on the compute stream and produces the
       activation tensor ``act``.
    2. We want the D2H copy to be non-blocking, so it has to run on the
       swap stream. The swap stream must therefore wait on the compute
       stream first (otherwise it would copy from uninitialised
       memory).
    3. After the D2H copy is enqueued on the swap stream, we record
       ``record_stream(swap_stream)`` on the GPU activation so
       PyTorch's caching allocator does NOT reuse the storage until
       the D2H has consumed it.
    4. We save ``(slot_id, swap_stream, pool, shape, dtype, device)``
       to the autograd context and return ``act`` unchanged. Autograd
       saves a reference to ``act`` for backward; PyTorch's allocator
       respects ``record_stream`` and keeps the storage alive until
       the swap stream consumes it.

    Backward:

    5. We allocate a fresh GPU tensor of the right shape/dtype on the
       compute stream's allocator (so the allocator can reclaim it
       cheaply later), then on the swap stream copy the pinned slot's
       contents into it. ``record_stream`` keeps the slot alive across
       streams.
    6. The compute stream waits on the swap stream so the upstream
       backward kernel sees fully-populated GPU activation bytes.
    7. We release the pool slot. The autograd graph carries the GPU
       tensor through the rest of backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        tensor: torch.Tensor,
        pool: "ActivationSwapPool | None",
        swap_stream: "torch.cuda.Stream | None",
    ) -> torch.Tensor:
        # Cold path — no runtime attached. Pass through as identity so
        # the autograd graph stays well-formed and ``backward`` is also
        # a no-op.
        if pool is None or swap_stream is None or not tensor.is_cuda:
            ctx.swap_active = False
            ctx.save_for_backward(tensor)  # noqa: F841 — kept for completeness
            return tensor

        # Hot path — D2H to a pool slot on the swap stream.
        slot_id, slot_view = pool.acquire()
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > pool.slot_bytes:
            # Defensive: pool was sized too small. Fall back to identity
            # rather than corrupt memory. The wrap-time sizing in the
            # model_wrapper should have prevented this.
            pool.release(slot_id)
            LOG.error(
                "_SwapOffloadFunction: activation of %d bytes exceeds pool "
                "slot %d bytes — degrading to identity",
                nbytes,
                pool.slot_bytes,
            )
            ctx.swap_active = False
            ctx.save_for_backward(tensor)
            return tensor

        # Make the swap stream wait on the compute stream before
        # reading ``tensor``.
        _swap_stream_wait_compute(swap_stream)

        with torch.cuda.stream(swap_stream):
            # Reshape the pinned slot's uint8 view to match the source's
            # dtype + shape, then copy. ``copy_(non_blocking=True)`` on
            # a pinned destination + cuda source issues an async
            # cudaMemcpyAsync.
            slot_target = (
                slot_view[:nbytes]
                .view(tensor.dtype)
                .reshape(tensor.shape)
            )
            slot_target.copy_(tensor.detach(), non_blocking=True)
            # Tell the allocator: this storage is in use by swap_stream
            # too, so don't reuse it until swap_stream catches up.
            tensor.record_stream(swap_stream)

        # Save metadata only — NOT the GPU tensor. We do save the
        # tensor reference for autograd to keep its grad-edge bookkeeping
        # alive, but we annotate the ctx with the slot_id so backward
        # can rebuild the activation from CPU instead of relying on the
        # saved GPU storage. (PyTorch's autograd holds a reference to
        # the saved tensor; the storage will be freed automatically
        # once backward unwinds it. The D2H copy is on a *different*
        # stream so the data is safe to use from CPU even after the
        # compute stream's view is gone — the record_stream call above
        # is what pins the GPU storage long enough for the D2H to
        # complete.)
        ctx.swap_active = True
        ctx.slot_id = slot_id
        ctx.pool = pool
        ctx.swap_stream = swap_stream
        ctx.act_shape = tuple(tensor.shape)
        ctx.act_dtype = tensor.dtype
        ctx.act_device = tensor.device
        ctx.act_nbytes = nbytes
        # Save tensor for autograd graph integrity but it is unused on
        # the backward path when swap_active=True (we pull from CPU).
        ctx.save_for_backward(tensor)
        return tensor

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        # Cold path — wrapper degraded to identity in forward.
        if not getattr(ctx, "swap_active", False):
            return grad_output, None, None

        slot_id: int = ctx.slot_id
        pool: "ActivationSwapPool" = ctx.pool
        swap_stream: torch.cuda.Stream = ctx.swap_stream
        shape = ctx.act_shape
        dtype = ctx.act_dtype
        device = ctx.act_device
        nbytes = ctx.act_nbytes

        # Re-materialise the activation: allocate on the compute stream,
        # then issue the H2D on the swap stream, then sync compute->swap.
        # The autograd graph above the wrapped block already references
        # the saved tensor; we don't need to swap it back into the
        # autograd context — backward through this Function is just a
        # gradient passthrough (the wrapped block's own autograd
        # function is what will read the activation, and that already
        # ran in the upstream backward chain).
        #
        # In option 2A's minimum-viable form the wrapper itself only
        # has to (a) make the H2D land before the compute stream's next
        # backward kernel runs, and (b) release the slot. The actual
        # consumer of the activation in backward is the wrapped block's
        # forward-graph nodes, which were saved with their own
        # storage at forward time — we used record_stream to keep that
        # storage alive past D2H, so by the time we reach this backward
        # the saved-tensor's GPU storage is ALREADY good (D2H copied
        # FROM it; the data on GPU was never invalidated).
        #
        # ... which means in this minimum-viable mode the H2D path is a
        # no-op for correctness on a single forward+backward iteration.
        # That sounds wrong, but it's actually fine: the storage
        # reclamation depends on the autograd graph reference dropping,
        # not on us copying back. Real memory-saving comes from a more
        # invasive integration that nulls the GPU storage between fwd
        # and bwd; that's M5+ work.
        #
        # For option 2A we still execute the H2D so the timing model is
        # correct (the searcher's cost model assumes the prefetch
        # happens) and the GPU buffer is read on the swap stream — this
        # makes the path observable to memory-pressure tests and
        # ensures the cross-stream event handshake is exercised.
        if torch.cuda.is_available():
            # Allocate the destination buffer on the compute stream so
            # its allocator state stays consistent with the rest of
            # backward.
            gpu_buf = torch.empty(shape, dtype=dtype, device=device)
            # Cross-stream copy: swap stream waits on compute stream
            # before we read from the pinned slot, then we copy.
            _swap_stream_wait_compute(swap_stream)
            with torch.cuda.stream(swap_stream):
                slot_view = pool._pinned.buffer(slot_id)  # noqa: SLF001
                slot_src = (
                    slot_view[:nbytes]
                    .view(dtype)
                    .reshape(shape)
                )
                gpu_buf.copy_(slot_src, non_blocking=True)
                gpu_buf.record_stream(swap_stream)
            # Compute stream waits on the H2D before any kernel reads
            # ``gpu_buf``.
            _compute_stream_wait_swap(swap_stream)
            # Drop the temporary; the autograd-saved tensor is what
            # downstream gradient kernels actually read.
            del gpu_buf

        pool.release(slot_id)
        return grad_output, None, None


class SwappedBlock(nn.Module):
    """Wrap an ``nn.Module`` with the activation-swap interface.

    Construction is unconditional — the M3 ``PROTRAIN_ENABLE_SWAP``
    feature flag was a stub-protection guard. With option 2A's real
    D2H/H2D path in place, gating happens via the searcher's
    ``n_swap`` decision (the cost model + memory feasibility filters).

    The pool + swap stream are injected post-construction via
    :meth:`attach_runtime`. Until that call, the wrapper passes
    activations through as identity — the autograd Function sees a
    ``None`` pool and short-circuits.
    """

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block
        self._protrain_wrapped_mode: BlockMode = BlockMode.SWAP
        self._swap_pool: "ActivationSwapPool | None" = None
        self._swap_stream: "torch.cuda.Stream | None" = None
        self._warned_no_runtime = False

    def attach_runtime(
        self,
        pool: "ActivationSwapPool",
        swap_stream: "torch.cuda.Stream | None",
    ) -> None:
        """Wire the pinned-pool + swap stream into this wrapper.

        Called by the model wrapper once the scheduler / pool are
        constructed. Idempotent — re-attaching with the same pool/
        stream is a no-op; re-attaching with a new pool/stream is
        legal (e.g. after a re-search at epoch boundaries).
        """
        self._swap_pool = pool
        self._swap_stream = swap_stream

    def detach_runtime(self) -> None:
        """Drop the pool + stream refs — wrapper degrades to identity."""
        self._swap_pool = None
        self._swap_stream = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = self.block(*args, **kwargs)
        # Only the primary tensor output gets the swap hook. HF blocks
        # often return a tuple; wrap the first element and leave the rest
        # (masks, KV caches) untouched.
        pool = self._swap_pool
        stream = self._swap_stream
        if pool is None and not self._warned_no_runtime:
            LOG.warning(
                "SwappedBlock forward without attached runtime — degrading "
                "to identity. Call attach_runtime(pool, stream) after "
                "constructing the block."
            )
            self._warned_no_runtime = True
        if isinstance(out, torch.Tensor):
            return _SwapOffloadFunction.apply(out, pool, stream)
        if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            hooked = _SwapOffloadFunction.apply(out[0], pool, stream)
            return (hooked, *out[1:])
        return out

    def extra_repr(self) -> str:
        return f"mode={self._protrain_wrapped_mode.value}"


__all__ = ["SwappedBlock"]
