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

M4.5 runtime-primitives additions
---------------------------------

:meth:`materialize_offload` physically moves every non-persistent chunk's
param data from GPU to pinned CPU memory and replaces the GPU storage
with an empty placeholder tensor — this is what closes the paper's
"non-persistent chunks live on CPU" promise end-to-end (Gap 1). The
method is idempotent and must be called exactly once after the chunk
manager is constructed but before the first :meth:`gather` / any
forward pass. :func:`protrain_model_wrapper` drives this from step 4.5
of its construction sequence.

:meth:`_offload_grad` — per-parameter post-accumulate grad hook installed
on every trainable non-persistent param by :meth:`materialize_offload`
(Gap 2). Fires the instant PyTorch autograd accumulates a grad, copies
it to a pinned CPU grad shard, nulls ``param.grad`` on GPU, and — once
every param in the chunk has contributed — enqueues the async CPU
FusedAdam step. This is what keeps GPU grad pressure ≈ zero for
non-persistent chunks during backward, matching ZeRO-Offload's invariant.

Paper references: §3.1.1, §5; ZeRO-Offload's per-param hook pattern.
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


class _CpuParamSlot:
    """Per-parameter bookkeeping for a non-persistent chunk.

    Holds the pinned CPU tensor containing the fp16 (or whatever dtype)
    parameter data, the original shape, dtype, and byte offset inside
    the chunk's flat byte buffer — everything :meth:`ChunkManager.gather`
    needs to rebind ``param.data`` to a GPU view after the H2D copy.
    """

    __slots__ = (
        "param_id",
        "cpu_data",
        "cpu_grad",
        "shape",
        "dtype",
        "byte_offset",
        "numel",
        "element_size",
    )

    def __init__(
        self,
        param_id: ParamId,
        cpu_data: "torch.Tensor",
        cpu_grad: "torch.Tensor | None",
        shape: "torch.Size",
        dtype: "torch.dtype",
        byte_offset: int,
        numel: int,
        element_size: int,
    ) -> None:
        self.param_id = param_id
        self.cpu_data = cpu_data
        self.cpu_grad = cpu_grad
        self.shape = shape
        self.dtype = dtype
        self.byte_offset = byte_offset
        self.numel = numel
        self.element_size = element_size


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
    device
        The CUDA device where non-persistent chunks land when gathered.
        Defaults to ``buffer_pool.device``.
    """

    def __init__(
        self,
        model: "nn.Module",
        layout: ChunkLayout,
        n_persist: int,
        buffer_pool: "BufferPool",
        cpu_optim: "CpuFusedAdamAdapter | None" = None,
        gpu_optim: "GpuFusedAdamAdapter | None" = None,
        device: "torch.device | str | None" = None,
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

        import torch

        self.model = model
        self.layout = layout
        self.buffer_pool = buffer_pool
        self.cpu_optim = cpu_optim
        self.gpu_optim = gpu_optim
        self.device = torch.device(
            device if device is not None else buffer_pool.device
        )

        # When True, :meth:`reduce_grads_and_offload` and the per-param
        # grad-offload hook skip their internal ``dist.all_reduce`` calls
        # and trust an outer layer (typically ``DistributedDataParallel``
        # wrapped over the protrain'd module) to own cross-rank grad
        # sync. Toggled by ``protrain_model_wrapper`` at compose-time —
        # see the Multi-GPU section of ``DESIGN.md``.
        self.skip_internal_grad_reduce: bool = False

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

        # Per-chunk CPU slots: materialize_offload populates this dict
        # mapping chunk_id -> list[_CpuParamSlot] ordered as the params
        # appear in ``layout.chunks[chunk_id]``.
        self._cpu_slots: dict[ChunkId, list[_CpuParamSlot]] = {}

        # Empty GPU sentinel (one per dtype) — reused for all param.data
        # "placeholders" after offload so we don't allocate a fresh 0-byte
        # tensor per param (cheap but not free).
        self._empty_by_dtype: dict["torch.dtype", "torch.Tensor"] = {}

        # Per-chunk grad-drain counter: decremented by _offload_grad for
        # every trainable param in the chunk; when it hits zero we kick
        # off the async CPU Adam step (Gap 2).
        self._grad_remaining: dict[ChunkId, int] = {}
        # How many trainable params a chunk started with, used to reset
        # _grad_remaining at the top of every backward pass (we clone this
        # dict on demand).
        self._grad_initial: dict[ChunkId, int] = {}

        # Hook handles stored so ``uninstall`` / ``__del__`` can remove
        # them deterministically and we don't leak closures over ``self``.
        self._grad_hook_handles: list[object] = []

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

    # ---- M4.5: init-time chunk offload + per-param grad hooks ----------

    def materialize_offload(self) -> int:
        """Physically move non-persistent chunks' params to pinned CPU memory.

        For every non-persistent chunk:

        1. Sum the total byte footprint of its params (variable — a chunk
           is at most ``S_chunk`` bytes but may be smaller, e.g. the
           trailing chunk).
        2. Allocate one pinned CPU tensor of that size (uint8 flat), then
           partition it into per-param byte slots.
        3. For each param: copy ``param.data`` (GPU) into its CPU slot,
           then replace ``param.data`` with an empty GPU placeholder.
        4. For each *trainable* (``requires_grad=True``) param: allocate
           a pinned CPU grad shard of the same shape+dtype and register
           a ``register_post_accumulate_grad_hook`` that drains the grad
           to CPU on the fly (Gap 2).

        Returns
        -------
        int
            Bytes freed on the GPU by the offload. Sum of
            ``param.numel() * param.element_size()`` across every
            offloaded param.

        Idempotent: a second call is a no-op (detected via
        ``self._cpu_slots`` already being populated).
        """
        if self._cpu_slots:
            LOG.debug(
                "ChunkManager.materialize_offload: already materialized "
                "(%d chunks), no-op", len(self._cpu_slots)
            )
            return 0

        import torch

        freed = 0
        for cid_int in sorted(self._non_persistent_ids):
            cid = cast(ChunkId, cid_int)
            param_ids = self.layout.chunks[int(cid)]
            if not param_ids:
                continue

            # --- Step 1: compute the chunk's actual byte footprint ------
            chunk_bytes = 0
            per_param_bytes: list[int] = []
            for pid in param_ids:
                param = self._params_by_id.get(pid)
                if param is None:
                    per_param_bytes.append(0)
                    continue
                nbytes = int(param.numel()) * int(param.element_size())
                per_param_bytes.append(nbytes)
                chunk_bytes += nbytes

            if chunk_bytes == 0:
                continue

            # --- Step 2: one pinned CPU allocation per chunk ------------
            # We allocate fresh pinned memory rather than reusing the
            # buffer_pool's pinned host region (that was sized to
            # ``n_buffer * S_chunk`` for staging, not persistent storage —
            # collisions mod n_buffer would corrupt data). Sizing is
            # precise: ``chunk_bytes`` bytes exactly.
            cpu_bytes = torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True)

            # --- Step 3: copy + rebind param.data -----------------------
            slots: list[_CpuParamSlot] = []
            offset = 0
            trainable_count = 0
            for pid, nbytes in zip(param_ids, per_param_bytes):
                param = self._params_by_id.get(pid)
                if param is None or nbytes == 0:
                    continue

                orig_data = param.data
                dtype = orig_data.dtype
                shape = orig_data.shape
                numel = orig_data.numel()
                element_size = orig_data.element_size()

                # Slice of the pinned buffer for this param, reinterpret as
                # the param's dtype, reshape to original shape. The copy is
                # pinned→pageable with a GPU→CPU D2H.
                cpu_view = cpu_bytes.narrow(0, offset, nbytes)
                cpu_param = cpu_view.view(dtype).view(shape)
                cpu_param.copy_(orig_data)

                # Release GPU storage by rebinding .data to an empty
                # placeholder of the same dtype.
                param.data = self._empty_placeholder(dtype)

                # Optional: pinned CPU grad buffer for trainable params.
                cpu_grad: "torch.Tensor | None" = None
                if param.requires_grad:
                    trainable_count += 1
                    cpu_grad = torch.zeros(
                        shape, dtype=dtype, pin_memory=True
                    )

                slots.append(
                    _CpuParamSlot(
                        param_id=pid,
                        cpu_data=cpu_param,
                        cpu_grad=cpu_grad,
                        shape=shape,
                        dtype=dtype,
                        byte_offset=offset,
                        numel=numel,
                        element_size=element_size,
                    )
                )
                offset += nbytes
                freed += nbytes

            self._cpu_slots[cid] = slots
            self._grad_initial[cid] = trainable_count
            self._grad_remaining[cid] = trainable_count

            # --- Step 4: per-param grad hooks for trainable params -----
            for slot in slots:
                param = self._params_by_id[slot.param_id]
                if not param.requires_grad or slot.cpu_grad is None:
                    continue
                handle = param.register_post_accumulate_grad_hook(
                    self._make_grad_offload_hook(cid, slot)
                )
                self._grad_hook_handles.append(handle)

        LOG.info(
            "ChunkManager.materialize_offload: offloaded %d non-persistent "
            "chunks to pinned CPU memory, freed %.3f GB on GPU",
            len(self._cpu_slots),
            freed / 1e9,
        )
        return freed

    def _empty_placeholder(self, dtype: "torch.dtype") -> "torch.Tensor":
        """Return a zero-element GPU tensor of ``dtype`` (cached per dtype)."""
        import torch

        existing = self._empty_by_dtype.get(dtype)
        if existing is not None:
            return existing
        t = torch.empty(0, device=self.device, dtype=dtype)
        self._empty_by_dtype[dtype] = t
        return t

    def _make_grad_offload_hook(self, chunk_id: ChunkId, slot: _CpuParamSlot):
        """Build a post-accumulate grad hook for one trainable non-persistent param.

        Captures ``chunk_id`` + ``slot`` by closure. On fire:

        1. Copy ``param.grad`` into the pinned CPU grad shard.
        2. Null out ``param.grad`` to free GPU storage immediately.
        3. Decrement the chunk's grad counter; if zero, enqueue the
           async CPU Adam step so it overlaps with the remaining GPU
           backward compute (§5).
        """
        cm = self
        # Keep a strong ref to the slot so the param lifetime isn't what
        # keeps it alive.
        captured_slot = slot
        captured_cid = chunk_id

        def _hook(param: "nn.Parameter") -> None:
            if param.grad is None:
                return
            # Multi-rank data-parallel path: reduce the GPU grad across
            # ranks (AVG = sum / world_size) BEFORE draining to the CPU
            # shard. Guarded on world_size > 1 AND ``skip_internal_grad_reduce``
            # being False — the M6 DDP-composed stack sets the flag to
            # True so DDP's own bucketed allreduce handles this sync
            # and we don't do a second per-param reduce here. In a bare
            # non-DDP distributed run the flag is False and this is the
            # sole grad-sync point.
            import torch.distributed as _dist
            if (
                _dist.is_available()
                and _dist.is_initialized()
                and _dist.get_world_size() > 1
                and not cm.skip_internal_grad_reduce
            ):
                _dist.all_reduce(param.grad, op=_dist.ReduceOp.AVG)
            # copy_ supports cross-device; non_blocking=True is safe
            # because the destination is pinned host memory.
            captured_slot.cpu_grad.copy_(param.grad, non_blocking=True)  # type: ignore[union-attr]
            # Null the grad so PyTorch frees the GPU storage right away —
            # this is the whole point of the per-param hook.
            param.grad = None

            remaining = cm._grad_remaining.get(captured_cid, 0) - 1
            cm._grad_remaining[captured_cid] = remaining
            if remaining == 0:
                # All of the chunk's trainable params are drained. If a
                # CPU FusedAdam adapter is attached, install the CPU
                # shards onto the param objects and kick off the async
                # step — the adapter was built against the GPU param
                # refs but consumes grads from our CPU shards, so we
                # temporarily repoint ``.data`` and ``.grad`` for it.
                #
                # When ``cpu_optim is None`` (no DeepSpeedCPUAdam — e.g.
                # the system toolchain's CUDA version mismatches torch's
                # build), we deliberately skip the repoint: leaving
                # ``param.grad`` as None and ``param.data`` as the empty
                # GPU placeholder keeps every ``nn.Parameter`` device-
                # consistent across iterations. Without this guard,
                # iter 0's hook would leave 56 trainable LoRA params
                # pointing at CPU storage and iter 1's backward would
                # trip the "expected same device" check when autograd
                # accumulates the new GPU grad onto the stale CPU grad.
                if cm.cpu_optim is not None:
                    cm._ensure_cpu_grads_attached(captured_cid)
                    cm.cpu_optim.step_async(captured_cid)
                # Reset the counter now so the next backward fires again.
                cm._grad_remaining[captured_cid] = cm._grad_initial.get(
                    captured_cid, 0
                )

        return _hook

    def _ensure_cpu_grads_attached(self, chunk_id: ChunkId) -> None:
        """Prepare the non-persistent chunk for its CPU Adam step.

        The CPU FusedAdam adapter was built over the GPU ``nn.Parameter``
        objects (see ``protrain_optimizer_wrapper``). For the CPU step to
        consume the drained grads, we temporarily:

        * Point each param's ``.data`` at its CPU shard (so Adam updates
          the CPU master in place).
        * Point each param's ``.grad`` at its CPU grad shard.

        This matches DeepSpeed's CPU-offload pattern where the optimizer
        holds param references but those references are repointed at CPU
        storage for the step's duration. ``gather`` will re-point ``.data``
        back at the GPU buffer after the step (the CPU shard's updated
        bytes flow back via the gather's H2D copy).
        """
        slots = self._cpu_slots.get(chunk_id, [])
        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            # Swap .data to point at the CPU master so the CPU Adam kernel
            # has somewhere to read/write. This is a view of pinned memory;
            # no allocation.
            param.data = slot.cpu_data
            param.grad = slot.cpu_grad

    # ---- gather / offload ---------------------------------------------

    def gather(self, chunk_id: ChunkId) -> None:
        """Make ``chunk_id``'s params GPU-resident.

        Persistent chunks: no-op — they were never offloaded.

        Non-persistent chunks: acquire a GPU buffer from the pool,
        copy the chunk's CPU bytes into it (skipping the copy if the
        chunk is already resident-tagged in the pool), and rebind every
        param's ``.data`` to a GPU view. After this call the chunk's
        params are fully usable by forward/backward compute on GPU.

        Unlike the M2 stub signature, this method no longer returns the
        tensor — the side effect is the ``param.data`` rebind, and the
        raw buffer is owned by the pool.
        """
        if chunk_id in self._persistent_ids:
            return

        if chunk_id not in self._cpu_slots:
            # materialize_offload wasn't called, or this chunk had no
            # params — nothing to do.
            return

        # Consult the pool for a still-resident tag (forward→backward
        # reuse window).
        resident = self.buffer_pool.lookup_resident(chunk_id)
        if resident is not None:
            # Re-acquire (removes from free list if present; no-op if
            # already in-use). We still re-bind param.data in case a
            # previous offload nulled it out.
            buf = self.buffer_pool.acquire(chunk_id)
            self._rebind_params_to_buffer(chunk_id, buf, needs_copy=False)
            return

        # Cache miss: acquire a fresh buffer and H2D-copy.
        buf = self.buffer_pool.acquire(chunk_id)
        self._rebind_params_to_buffer(chunk_id, buf, needs_copy=True)

    def _rebind_params_to_buffer(
        self,
        chunk_id: ChunkId,
        buf: "torch.Tensor",
        needs_copy: bool,
    ) -> None:
        """Copy CPU shards into ``buf`` (if needed) and rebind each param's data.

        ``buf`` is the pool-owned GPU uint8 tensor of length ``S_chunk``.
        For each param slot we slice off ``slot.byte_offset .. +slot.nbytes``,
        reinterpret it as the param's dtype, reshape to the param's shape,
        and assign to ``param.data``.
        """
        slots = self._cpu_slots.get(chunk_id, [])
        if not slots:
            return

        if needs_copy:
            # One large H2D per chunk is faster than per-param — the CPU
            # shards are already laid out contiguously by
            # materialize_offload, so we copy the whole flat byte region
            # in a single call.
            total_bytes = sum(
                slot.numel * slot.element_size for slot in slots
            )
            # Grab the chunk's pinned CPU byte view (all slots share the
            # same parent storage).
            first_cpu = slots[0].cpu_data
            # Reconstruct the flat uint8 view of the parent pinned
            # allocation: the cpu_data was built from a narrow on a
            # uint8 tensor, so .untyped_storage() gives us back the flat
            # bytes without breaking pinning.
            # Simpler: copy per-slot. These copies are pipelined on the
            # same H2D engine and the total bytes moved is identical.
            buf_view = buf.narrow(0, 0, total_bytes)
            offset = 0
            for slot in slots:
                nbytes = slot.numel * slot.element_size
                dst_bytes = buf_view.narrow(0, offset, nbytes)
                # view into CPU as uint8 for a byte-exact copy.
                src_bytes = slot.cpu_data.view(slot.dtype)  # already that dtype
                # Copy as the native dtype — same number of bytes moved,
                # but avoids dtype mismatch in the copy_ call.
                dst_typed = dst_bytes.view(slot.dtype).view(slot.shape)
                dst_typed.copy_(slot.cpu_data, non_blocking=True)
                offset += nbytes
                # ignore unused
                _ = src_bytes

        # Rebind .data unconditionally — even on the no-copy path, a
        # previous offload() nulled out param.data, and re-acquiring from
        # the pool keeps the GPU bytes but requires re-pointing the
        # param at them.
        offset = 0
        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            nbytes = slot.numel * slot.element_size
            # Slice the chunk buffer at this param's byte offset and view
            # as (dtype, shape).
            byte_view = buf.narrow(0, offset, nbytes)
            typed = byte_view.view(slot.dtype).view(slot.shape)
            param.data = typed
            offset += nbytes

    def offload(self, chunk_id: ChunkId) -> None:
        """Release ``chunk_id``'s GPU storage (non-persistent only).

        Null out every param.data back to the empty sentinel, then return
        the buffer to the pool. The pool keeps the resident tag (so a
        backward-pass gather within the reuse window can skip the H2D
        re-copy) — but the param-level bindings are severed here so
        nothing tries to read stale GPU bytes after the pool reassigns
        the slot to a different chunk.
        """
        if chunk_id in self._persistent_ids:
            return
        slots = self._cpu_slots.get(chunk_id, [])
        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            param.data = self._empty_placeholder(slot.dtype)
        self.buffer_pool.release(chunk_id)

    def reduce_grads_and_offload(self, chunk_id: ChunkId) -> None:
        """Reduce-scatter grads and D2H-copy the chunk's grad shard back to CPU.

        Persistent chunks: run the reduction (if distributed is live)
        and leave the result on GPU — the GPU optimizer consumes it in
        :meth:`persistent_step`.

        Non-persistent chunks: the per-param post-accumulate-grad hooks
        installed by :meth:`materialize_offload` already drained each
        param's grad to CPU and kicked off the async CPU FusedAdam step
        at the moment the last param's grad landed (§5, ZeRO-Offload).
        All that's left for the block-granularity scheduler to do is
        release the chunk's buffer — the grad work is already in flight.
        """
        import torch

        if chunk_id in self._persistent_ids:
            # Persistent chunks keep their grads GPU-resident for the
            # FusedAdam step.
            #
            # Distributed grad-sync policy. When another layer above
            # ProTrain owns the cross-rank reduction (the M6 stack wraps
            # the protrain'd module in ``DistributedDataParallel``, which
            # fires its own bucketed allreduce via autograd hooks),
            # this in-manager all_reduce would be a redundant second
            # sync — and a costly one on pure-PCIe 3090 pairs because
            # it runs per-param without bucketing. ``self.skip_internal_grad_reduce``
            # (set by the wrapper when it detects DDP composition) tells
            # us to leave the grads alone.
            #
            # In the non-DDP distributed path (e.g. a bare ZeRO-3 run)
            # the flag is False and we do the reduction per-param with
            # AVG semantics — correct, if slower than a bucketed path.
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
                and not self.skip_internal_grad_reduce
            ):
                for pid in self.layout.chunks[int(chunk_id)]:
                    param = self._params_by_id.get(pid)
                    if param is not None and param.grad is not None:
                        torch.distributed.all_reduce(
                            param.grad, op=torch.distributed.ReduceOp.AVG
                        )
            return

        # Non-persistent: grad offload is owned by _offload_grad (per-param
        # hooks). The block-granularity scheduler here releases the chunk
        # buffer AND nulls the param.data placeholder so the GPU storage
        # is fully freed and the params are in a clean state for the
        # next gather. (Calling ``self.offload`` rather than a raw pool
        # release — the param.data null-out is what matters for peak.)
        self.offload(chunk_id)

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

    def wait_cpu_optim_all(self) -> None:
        """Alias of :meth:`wait_cpu_optim` for the public optim wrapper."""
        self.wait_cpu_optim()

    # ---- cleanup -------------------------------------------------------

    def uninstall(self) -> None:
        """Remove every registered per-param grad hook. Idempotent."""
        for handle in self._grad_hook_handles:
            try:
                handle.remove()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — best-effort
                LOG.debug("ChunkManager.uninstall: hook remove failed: %s", exc)
        self._grad_hook_handles.clear()

    def __del__(self) -> None:  # noqa: D401
        try:
            self.uninstall()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass

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
        """Legacy accessor — returns the first param's CPU shard for ``chunk_id``.

        Only kept for backwards compatibility with M2-era tests. The M4.5
        semantics are the per-param ``_CpuParamSlot`` list in
        ``self._cpu_slots``.
        """
        slots = self._cpu_slots.get(chunk_id)
        if not slots:
            # Fall back to the M2 pool-slot semantics for chunks that
            # were never materialize_offload'd (e.g. bare unit tests).
            slot = int(chunk_id) % self.buffer_pool.n_buffer
            return self.buffer_pool.pinned_host.buffer(slot)
        return slots[0].cpu_data


__all__ = ["ChunkManager"]
