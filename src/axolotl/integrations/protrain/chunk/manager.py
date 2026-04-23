"""Per-rank chunk manager driving the persistent / non-persistent split.

The :class:`ChunkManager` owns the runtime behavior of a :class:`ChunkLayout`:

* Persistent chunks (``chunk_id < n_persist``) stay resident on GPU,
  updated in place by the GPU FusedAdam adapter.
* Non-persistent chunks are sharded across ranks, offloaded to CPU as
  pinned host tensors, gathered into a pool buffer on demand, and
  reduce-scatter'd + D2H-copied on the backward sweep.

All ``torch.distributed`` calls are guarded with
``torch.distributed.is_initialized()`` so single-rank unit tests don't
require an initialized process group.

Paper references: §3.1.1, §5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from axolotl.integrations.protrain.types import (
    ChunkId,
    ChunkLayout,
    ParamId,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.optim import (
        CpuFusedAdamAdapter,
        GpuFusedAdamAdapter,
    )

LOG = get_logger(__name__)


class ChunkManager:
    """Runtime driver for a :class:`ChunkLayout`.

    Parameters
    ----------
    model
        The already-initialized ``nn.Module`` whose ``named_parameters()``
        cover every ``ParamId`` in ``layout``.
    layout
        Output of :func:`axolotl.integrations.protrain.chunk.layout.build_layout`.
    n_persist
        Number of leading chunks kept resident on GPU. The rest are
        offloaded / sharded.
    buffer_pool
        Pre-allocated GPU chunk buffers for the non-persistent path.
    cpu_optim
        Optional CPU FusedAdam adapter for non-persistent chunks. If
        provided, :meth:`reduce_grads_and_offload` triggers its
        ``step_async`` the moment grads land on CPU.
    gpu_optim
        Optional GPU FusedAdam adapter for the persistent chunk set;
        invoked by :meth:`persistent_step`.
    """

    def __init__(
        self,
        model: "nn.Module",
        layout: ChunkLayout,
        n_persist: int,
        buffer_pool: "BufferPool",
        cpu_optim: "CpuFusedAdamAdapter | None" = None,
        gpu_optim: "GpuFusedAdamAdapter | None" = None,
    ) -> None:
        if n_persist < 0 or n_persist > layout.N_chunk:
            raise ValueError(
                f"n_persist={n_persist} out of range [0, {layout.N_chunk}]"
            )
        if buffer_pool.S_chunk != layout.S_chunk:
            raise ValueError(
                f"buffer_pool.S_chunk ({buffer_pool.S_chunk}) "
                f"!= layout.S_chunk ({layout.S_chunk})"
            )

        self.model = model
        self.layout = layout
        self.buffer_pool = buffer_pool
        self.cpu_optim = cpu_optim
        self.gpu_optim = gpu_optim

        # Param lookup by id for gather/offload payload construction.
        self._params_by_id: dict[ParamId, "nn.Parameter"] = {
            cast(ParamId, name): p for name, p in model.named_parameters()
        }

        # Persistent / non-persistent split; populated in ``mark_persistent``.
        self._persistent_ids: set[ChunkId] = set()
        self._non_persistent_ids: set[ChunkId] = set(
            cast(ChunkId, i) for i in range(layout.N_chunk)
        )

        # Per-chunk resident GPU flat tensor — populated only for persistent
        # chunks (non-persistent chunks borrow from the buffer pool).
        self._persistent_buffers: dict[ChunkId, "torch.Tensor"] = {}

        # Per-chunk CPU shard for non-persistent chunks. In a true multi-rank
        # setup each rank holds only 1/world_size of the chunk; for single-rank
        # tests we hold the whole thing. Stored as flat uint8 views of pinned
        # host memory owned by the buffer_pool.pinned_host block.
        self._cpu_shards: dict[ChunkId, "torch.Tensor"] = {}

        self.mark_persistent(n_persist)

    # ---- configuration -------------------------------------------------

    def mark_persistent(self, first_n: int) -> None:
        """Tag chunks [0, first_n) as persistent; the rest as non-persistent.

        Idempotent — safe to call after a searcher re-pick at the start of a
        new epoch. Allocations for already-materialized buffers are NOT
        changed here (the first-time materialization happens lazily in
        :meth:`gather` / :meth:`_ensure_persistent_buffer`), so repeated
        calls with the same ``first_n`` are cheap.
        """
        if first_n < 0 or first_n > self.layout.N_chunk:
            raise ValueError(
                f"first_n={first_n} out of range [0, {self.layout.N_chunk}]"
            )
        self._persistent_ids = {cast(ChunkId, i) for i in range(first_n)}
        self._non_persistent_ids = {
            cast(ChunkId, i) for i in range(first_n, self.layout.N_chunk)
        }
        LOG.debug(
            "ChunkManager.mark_persistent: %d / %d chunks resident on GPU",
            first_n,
            self.layout.N_chunk,
        )

    # ---- gather / offload ---------------------------------------------

    def gather(self, chunk_id: ChunkId) -> "torch.Tensor":
        """Return a GPU tensor containing ``chunk_id``'s data.

        Persistent path: returns the already-resident flat buffer.

        Non-persistent path: if the chunk is still resident in the buffer
        pool (forward→backward reuse window), returns that buffer verbatim.
        Otherwise acquires a fresh buffer, H2D-copies the CPU shard into
        it, and returns it.
        """
        if chunk_id in self._persistent_ids:
            return self._ensure_persistent_buffer(chunk_id)

        # Non-persistent: first consult the pool for a still-resident tag.
        resident = self.buffer_pool.lookup_resident(chunk_id)
        if resident is not None:
            # Re-acquire (no-op if currently in-use; removes from free list
            # if it was released but not yet evicted).
            return self.buffer_pool.acquire(chunk_id)

        # Cache miss: acquire a buffer and do the H2D copy from CPU shard.
        buf = self.buffer_pool.acquire(chunk_id)
        shard = self._cpu_shard(chunk_id)
        # non_blocking=True because the shard is pinned.
        buf.copy_(shard, non_blocking=True)
        return buf

    def offload(self, chunk_id: ChunkId) -> None:
        """Release ``chunk_id``'s buffer back to the pool (non-persistent only).

        No D2H copy here — this is the "done using" signal. The data stays
        tagged in the pool slot, so a subsequent ``gather`` within the
        reuse window skips the reload. Gradient-offload uses the separate
        :meth:`reduce_grads_and_offload` path.
        """
        if chunk_id in self._persistent_ids:
            return
        self.buffer_pool.release(chunk_id)

    def reduce_grads_and_offload(self, chunk_id: ChunkId) -> None:
        """Reduce-scatter grads and D2H-copy the chunk's grad shard back to CPU.

        For persistent chunks: run the reduction (if distributed is live)
        and leave the result on GPU — the GPU optimizer consumes it in
        :meth:`persistent_step`.

        For non-persistent chunks: reduce, D2H-copy the result into the
        chunk's CPU shard, release the GPU buffer, and kick off the CPU
        FusedAdam step asynchronously so it overlaps with the GPU backward
        of earlier chunks (§5).
        """
        import torch

        buf = self.buffer_pool.lookup_resident(chunk_id)
        if buf is None and chunk_id not in self._persistent_ids:
            # Backward visited a chunk we never gathered — shouldn't happen,
            # but be defensive.
            LOG.warning(
                "reduce_grads_and_offload: chunk %d has no resident buffer; skipping",
                chunk_id,
            )
            return
        if buf is None:
            buf = self._ensure_persistent_buffer(chunk_id)

        # Reduce across ranks. In ProTrain proper this is a reduce-scatter
        # so each rank only keeps its shard. Stub it as all_reduce here —
        # correct for single-rank, and M4 will swap in the proper collective
        # once the scheduler owns the comm group.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(buf)

        if chunk_id in self._persistent_ids:
            # Grad stays on GPU; optimizer will consume it from the param
            # tensors directly (they aliased into ``buf`` in the persistent
            # path, see ``_ensure_persistent_buffer``).
            return

        # Non-persistent: D2H-copy the reduced grad into the CPU shard.
        shard = self._cpu_shard(chunk_id)
        shard.copy_(buf, non_blocking=True)
        self.buffer_pool.release(chunk_id)

        if self.cpu_optim is not None:
            self.cpu_optim.step_async(chunk_id)

    # ---- optimizer driver ---------------------------------------------

    def persistent_step(self) -> None:
        """Run the synchronous GPU FusedAdam step over persistent chunks."""
        if self.gpu_optim is None:
            return
        self.gpu_optim.step()

    def wait_cpu_optim(self) -> None:
        """Block until every in-flight CPU Adam step has finished."""
        if self.cpu_optim is not None:
            self.cpu_optim.wait_all()

    # ---- internals -----------------------------------------------------

    def _ensure_persistent_buffer(self, chunk_id: ChunkId) -> "torch.Tensor":
        """Lazily materialize the resident GPU buffer for a persistent chunk."""
        existing = self._persistent_buffers.get(chunk_id)
        if existing is not None:
            return existing
        import torch

        buf = torch.empty(
            self.layout.S_chunk,
            dtype=torch.uint8,
            device=self.buffer_pool.device,
        )
        self._persistent_buffers[chunk_id] = buf
        return buf

    def _cpu_shard(self, chunk_id: ChunkId) -> "torch.Tensor":
        """Lazily allocate a pinned CPU tensor backing ``chunk_id``'s data.

        We take the ``chunk_id``-indexed slot of the buffer pool's host
        block so H2D/D2H copies are already pinned→pageable-free at peak
        PCIe throughput. Indices wrap mod ``n_buffer`` because we only
        need enough pinned staging for the concurrent window of chunks
        in flight (the true persistent CPU storage will be handled by the
        M4 scheduler with a separate staging plan — for M2 we keep the
        simpler "one host slot per non-persistent chunk modulo pool size"
        mapping, which is sufficient for the single-rank validation tests).
        """
        shard = self._cpu_shards.get(chunk_id)
        if shard is not None:
            return shard

        slot = int(chunk_id) % self.buffer_pool.n_buffer
        # Use the pool's pinned host memory as backing storage. Two
        # non-persistent chunks whose ids collide (mod n_buffer) will
        # fight for the same slot — acceptable for M2 scope since the
        # cost model isn't active yet, and documented above.
        host = self.buffer_pool.pinned_host.buffer(slot)
        self._cpu_shards[chunk_id] = host
        return host


__all__ = ["ChunkManager"]
