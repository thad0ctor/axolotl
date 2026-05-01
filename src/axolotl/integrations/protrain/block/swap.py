"""Activation-swap wrapper (§3.1.2 — paper-real implementation, M5+).

SWAP mode in the ProTrain three-way block strategy: forward activations
are offloaded to pinned CPU memory, then prefetched back during
backward. The wrapper installs a
:func:`torch.autograd.graph.saved_tensors_hooks` context around the
block's forward so **every** saved tensor (residuals, attention QKV/
scores, FFN intermediates) is D2H'd to a pinned CPU pool and H2D'd
back on backward — not just the block's output tensor.

This is the M5+ upgrade over option-2A. Option-2A only swapped the
block's output tensor via a custom autograd Function; the GPU
activation stayed pinned by autograd because ``ctx.save_for_backward``
keeps a CUDA reference. With ``saved_tensors_hooks`` the saved-tensor
references handed to autograd are CPU-only handles, so the GPU storage
is reclaimed when the local Python frame drops its last GPU reference
to the activation. The result: actual GPU memory is freed between
forward and backward, not just shuffled.

Stream policy
-------------
Both D2H and H2D copies run on the scheduler's ``_swap_stream`` (one
shared stream per scheduler). The compute stream waits on the swap
stream's H2D event before the upstream backward kernel reads the
re-materialised activation. In forward the swap stream waits on the
compute stream before reading the GPU tensor we are offloading.

Hot path / cold path
--------------------
The pool + stream are injected post-construction by the model wrapper
via :meth:`SwappedBlock.attach_runtime`. If a block is constructed
WITHOUT runtime attached (e.g. unit tests, or a model wrapper that
forgot to call attach_runtime when ``n_swap > 0``), the wrapper
degrades to a no-op identity hook in autograd: the activations live on
GPU as they normally would, and no D2H/H2D happens. This keeps
correctness intact while preserving the historical "constructible
without runtime" surface that test fixtures rely on. A WARNING is
logged once per instance so the configuration drift is visible.

Tunable: ``SIZE_THRESHOLD_BYTES``
---------------------------------
Saved tensors smaller than this byte threshold pass through as-is
(kept on GPU). Small tensors don't recover much memory and the
pinned-slot bookkeeping + PCIe round trip cost dominates. The default
1 MiB is chosen to cover scalar-ish saved tensors (LayerNorm gamma/
beta, softmax masks, attention biases) while still capturing the big
ones (residual stream ``(batch, seq, hidden)`` and attention scores
``(batch, heads, seq, seq)``). Override per-test via the constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.integrations.protrain.block.swap_pool import ActivationSwapPool

LOG = get_logger(__name__)


#: Saved tensors smaller than this many bytes are kept on GPU (not
#: swapped). 1 MiB is the default; tests may override by reassigning
#: this module attribute. See the module docstring for derivation.
SIZE_THRESHOLD_BYTES: int = 1 << 20  # 1 MiB


def _swap_stream_wait_compute(swap_stream: "torch.cuda.Stream") -> None:
    """Make ``swap_stream`` wait on the current (compute) stream."""
    if swap_stream is None or not torch.cuda.is_available():
        return
    swap_stream.wait_stream(torch.cuda.current_stream())


def _compute_stream_wait_swap(swap_stream: "torch.cuda.Stream") -> None:
    """Make the current (compute) stream wait on ``swap_stream``."""
    if swap_stream is None or not torch.cuda.is_available():
        return
    torch.cuda.current_stream().wait_stream(swap_stream)


@dataclass
class _CPUHandle:
    """CPU-resident handle returned by ``pack_to_pool``.

    Holds the pool slot id + the metadata needed to reconstruct the
    GPU tensor in ``unpack_from_pool``. Because the handle does NOT
    reference the GPU tensor, autograd's saved-tensor table no longer
    pins GPU storage — that is the whole point of the M5+ rewrite.
    """

    pool: "ActivationSwapPool"
    swap_stream: "torch.cuda.Stream"
    slot_id: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    nbytes: int
    requires_grad: bool


class _PassThrough:
    """Sentinel for tensors that bypass swapping (too small / not on GPU).

    We wrap the original tensor so the pack/unpack pair is symmetrical
    and ``unpack_from_pool`` can dispatch on type rather than checking
    ``isinstance(handle, torch.Tensor)`` which would conflict with the
    "saved tensor IS a tensor" idiom on the cold path.
    """

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


def _make_pack_unpack(
    pool: "ActivationSwapPool",
    swap_stream: "torch.cuda.Stream",
    size_threshold: int,
):
    """Build the (pack, unpack) hook pair bound to ``pool``/``swap_stream``.

    A factory rather than a class so the hooks are plain closures —
    ``saved_tensors_hooks`` accepts any pair of callables and the
    closure form keeps the per-block state minimal.
    """

    def pack_to_pool(t: torch.Tensor):
        # Cold path — non-CUDA tensor or below the swap threshold.
        # Returning a ``_PassThrough`` keeps the saved-tensor reference
        # cheap (no slot acquisition) without changing the autograd
        # contract.
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            return _PassThrough(t)
        nbytes = t.numel() * t.element_size()
        if nbytes < size_threshold:
            return _PassThrough(t)
        if nbytes > pool.slot_bytes:
            # Defensive: tensor exceeds slot size. Keep on GPU rather
            # than corrupt memory. The wrap-time sizing in the model
            # wrapper should have prevented this; log and pass through.
            LOG.error(
                "_swap pack: tensor of %d bytes exceeds pool slot "
                "%d bytes — keeping on GPU",
                nbytes,
                pool.slot_bytes,
            )
            return _PassThrough(t)
        # Pool may be exhausted under pathological scheduling. Fall
        # back to identity rather than raising — autograd will simply
        # keep this tensor on GPU.
        try:
            slot_id, slot_view = pool.acquire()
        except RuntimeError:
            LOG.warning(
                "_swap pack: pool exhausted (n_slot=%d, in-flight=%d); "
                "keeping tensor on GPU",
                pool.n_slot,
                pool.inflight_count,
            )
            return _PassThrough(t)

        # Make the swap stream wait on the compute stream before
        # reading ``t``.
        _swap_stream_wait_compute(swap_stream)
        with torch.cuda.stream(swap_stream):
            slot_target = (
                slot_view[:nbytes].view(t.dtype).reshape(t.shape)
            )
            slot_target.copy_(t.detach(), non_blocking=True)
            # Tell the allocator: this storage is in use by swap_stream
            # too, so don't reuse it until swap_stream catches up.
            t.record_stream(swap_stream)

        return _CPUHandle(
            pool=pool,
            swap_stream=swap_stream,
            slot_id=slot_id,
            shape=tuple(t.shape),
            dtype=t.dtype,
            device=t.device,
            nbytes=nbytes,
            requires_grad=t.requires_grad,
        )

    def unpack_from_pool(handle):
        # Cold-path passthrough — return the original tensor unchanged.
        if isinstance(handle, _PassThrough):
            return handle.tensor

        if not isinstance(handle, _CPUHandle):
            # Defensive: PyTorch internals may pass other types through
            # the unpack hook (e.g. None for retained_grad sentinels).
            return handle

        # H2D from pinned slot to a fresh GPU buffer.
        # ``record_stream`` keeps the slot alive across streams; the
        # compute stream waits on the H2D event before any kernel reads
        # ``gpu_buf``.
        gpu_buf = torch.empty(
            handle.shape, dtype=handle.dtype, device=handle.device
        )
        _swap_stream_wait_compute(handle.swap_stream)
        with torch.cuda.stream(handle.swap_stream):
            slot_view = handle.pool._pinned.buffer(handle.slot_id)  # noqa: SLF001
            slot_src = (
                slot_view[: handle.nbytes]
                .view(handle.dtype)
                .reshape(handle.shape)
            )
            gpu_buf.copy_(slot_src, non_blocking=True)
            gpu_buf.record_stream(handle.swap_stream)
        _compute_stream_wait_swap(handle.swap_stream)

        # Return the slot to the pool. The H2D copy reads from the
        # pinned slot on swap_stream; record_stream above keeps the
        # slot's lifetime past the H2D's consumption. Subsequent
        # ``acquire()`` callers must still respect the swap stream's
        # completion before writing — the pool itself does no syncing,
        # so callers MUST wait on ``swap_stream`` before re-using the
        # slot for a new D2H. Inside the same step backward is purely
        # consumer-side, so this is safe.
        handle.pool.release(handle.slot_id)

        # Restore requires_grad flag if the original tensor had one.
        # Saved tensors that participated in autograd should preserve
        # their grad-fn linkage; ``empty()`` returns a leaf, but the
        # consumer of an unpacked saved-tensor reads it as data only
        # (no grad flows backward through the saved tensor itself —
        # that's a property of save_for_backward semantics).
        if handle.requires_grad:
            gpu_buf.requires_grad_(True)
        return gpu_buf

    return pack_to_pool, unpack_from_pool


class SwappedBlock(nn.Module):
    """Wrap an ``nn.Module`` so its saved tensors are swapped to pinned CPU.

    Construction is unconditional. Gating happens via the searcher's
    ``n_swap`` decision (the cost model + memory feasibility filters).

    The pool + swap stream are injected post-construction via
    :meth:`attach_runtime`. Until that call, the wrapper passes the
    block forward through unchanged — no saved_tensors_hooks context
    is installed, so saved tensors live on GPU as they normally would.
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

        Idempotent — re-attaching with the same pool/stream is a no-op;
        re-attaching with a new pool/stream is legal (e.g. after a
        re-search at epoch boundaries).
        """
        self._swap_pool = pool
        self._swap_stream = swap_stream

    def detach_runtime(self) -> None:
        """Drop the pool + stream refs — wrapper degrades to identity."""
        self._swap_pool = None
        self._swap_stream = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pool = self._swap_pool
        stream = self._swap_stream

        # Cold path — no runtime attached. Run the block plain.
        if pool is None or stream is None or not torch.cuda.is_available():
            if pool is None and not self._warned_no_runtime:
                LOG.warning(
                    "SwappedBlock forward without attached runtime — "
                    "degrading to identity. Call attach_runtime(pool, "
                    "stream) after constructing the block."
                )
                self._warned_no_runtime = True
            return self.block(*args, **kwargs)

        # Hot path — install saved_tensors_hooks for the duration of
        # the wrapped block's forward. Every saved tensor created
        # inside this context goes through ``pack_to_pool``; backward
        # restores them via ``unpack_from_pool``.
        pack, unpack = _make_pack_unpack(pool, stream, SIZE_THRESHOLD_BYTES)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = self.block(*args, **kwargs)
        return out

    def extra_repr(self) -> str:
        return f"mode={self._protrain_wrapped_mode.value}"


__all__ = ["SwappedBlock", "SIZE_THRESHOLD_BYTES"]
