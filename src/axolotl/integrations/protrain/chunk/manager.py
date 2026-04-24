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

M7: true ZeRO-3 chunk sharding
------------------------------

When ``zero3_shard=True`` is set on construction (driven automatically
by ``protrain_model_wrapper`` when ``world_size > 1`` AND no outer DDP
wrapper is detected), every non-persistent chunk's bytes are partitioned
across ranks on CPU: each rank keeps only ``ceil(chunk_bytes / world_size)``
pinned bytes — the ``rank``-th slice of the full chunk's byte layout.

* :meth:`gather` in sharded mode H2D-uploads this rank's CPU shard then
  issues ``torch.distributed.all_gather_into_tensor`` to reconstruct the
  full chunk into the pool buffer — every rank gets a bit-identical full
  copy for forward / backward compute.
* :meth:`reduce_grads_and_offload` for non-persistent chunks in sharded
  mode flattens the chunk's GPU grads into a contiguous buffer, issues
  ``torch.distributed.reduce_scatter_tensor(op=AVG)`` so each rank
  receives only its slice of the reduced-average grad, then D2H-copies
  the slice to the rank's pinned CPU grad shard and kicks the CPU
  FusedAdam step against the shard (CPU Adam is built over a single
  shard-flat ``nn.Parameter`` — see ``materialize_offload``).

The sharded path assumes a homogeneous-dtype chunk (all params share
``element_size``) and an element-size-aligned shard boundary; both hold
for the typical fp16/bf16 transformer-block payload. The shard size is
padded up so ``shard_bytes * world_size`` ≥ the chunk's actual byte
footprint and the final rank's shard may contain trailing zeros (the
boundary is a byte offset, not a param boundary — params straddling the
boundary are partitioned across two ranks' shards and reassembled on
gather by ``all_gather``).

Persistent chunks are FULLY REPLICATED even in sharded mode — they're
small, live on GPU, and the FusedAdam step runs locally on each rank.
The persistent branch of :meth:`reduce_grads_and_offload` still uses
per-param ``all_reduce(op=AVG)`` when ``zero3_shard=True`` (unchanged
from the non-sharded path).

Paper references: §1 (parallelism foundation), §2A (chunks), §5
(low-level overlaps).
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

    In the ZeRO-3 sharded path (``zero3_shard=True``) each param's
    ``cpu_data`` / ``cpu_grad`` may be ``None`` when the param lies
    outside this rank's shard range — the bytes live on a peer rank
    and will be reconstructed on ``gather`` via ``all_gather``. The
    ``byte_offset`` / ``numel`` / ``element_size`` fields are
    authoritative regardless; they describe the full-chunk layout
    shared by every rank.
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
        cpu_data: "torch.Tensor | None",
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


class _ChunkShardState:
    """Per-chunk ZeRO-3 shard bookkeeping (populated when ``zero3_shard=True``).

    For each non-persistent chunk we keep:

    * ``cpu_shard_bytes`` — a pinned ``uint8`` tensor of exactly
      ``shard_bytes`` bytes holding THIS RANK's slice of the full
      chunk's byte layout. The slice covers the byte range
      ``[rank * shard_bytes, (rank + 1) * shard_bytes)`` of the logical
      full chunk (truncated by ``chunk_bytes`` for the trailing rank).
    * ``cpu_shard_grad_bytes`` — a same-sized pinned ``uint8`` tensor
      holding the ``reduce_scatter``'d grad slice once backward drains.
    * ``chunk_bytes`` — the total byte footprint of the full chunk
      (including alignment padding; matches the pre-M7 single-rank
      cpu buffer size).
    * ``shard_bytes`` — ``ceil(chunk_bytes / world_size)`` padded up to
      a multiple of the dominant element size so shard boundaries land
      on clean fp16/bf16/fp32 element alignments (avoids an unaligned
      ``.view(dtype)`` after ``all_gather`` reconstructs the full
      chunk). ``shard_bytes * world_size >= chunk_bytes``.
    * ``primary_dtype`` / ``primary_element_size`` — the dominant dtype
      of params in this chunk. When the chunk is homogeneous (all
      params share one dtype) this is that dtype; when mixed we fall
      back to ``torch.uint8`` and forgo the single-param CPU-Adam
      shortcut (the chunk is kept fully-replicated in that case — see
      ``materialize_offload``'s shard-feasibility check).
    * ``shard_param`` — a single ``nn.Parameter`` whose ``.data`` views
      ``cpu_shard_bytes`` reinterpreted as the primary dtype. This is
      the param DeepSpeedCPUAdam is built over for the sharded path:
      one flat param per chunk instead of one per original weight,
      because each rank only owns a SLICE of the chunk's bytes and
      those slices generally don't align to original-param boundaries.
      The CPU Adam step updates ``shard_param.data`` in place; the
      next ``gather`` re-uploads the updated shard + re-runs
      ``all_gather`` to propagate the changes to every rank.
    """

    __slots__ = (
        "cpu_shard_bytes",
        "cpu_shard_grad_bytes",
        "chunk_bytes",
        "shard_bytes",
        "primary_dtype",
        "primary_element_size",
        "shard_param",
    )

    def __init__(
        self,
        cpu_shard_bytes: "torch.Tensor",
        cpu_shard_grad_bytes: "torch.Tensor",
        chunk_bytes: int,
        shard_bytes: int,
        primary_dtype: "torch.dtype",
        primary_element_size: int,
        shard_param: "torch.Tensor",
    ) -> None:
        self.cpu_shard_bytes = cpu_shard_bytes
        self.cpu_shard_grad_bytes = cpu_shard_grad_bytes
        self.chunk_bytes = chunk_bytes
        self.shard_bytes = shard_bytes
        self.primary_dtype = primary_dtype
        self.primary_element_size = primary_element_size
        self.shard_param = shard_param


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
    world_size, rank
        Collective-comms context, defaulting to ``1`` / ``0`` for the
        single-rank unit-test path. When ``world_size > 1`` and
        ``zero3_shard=True``, non-persistent chunks are partitioned
        across ranks on CPU and ``gather``/``reduce_grads_and_offload``
        become ``all_gather_into_tensor`` / ``reduce_scatter_tensor``
        respectively (M7 true ZeRO-3 path).
    zero3_shard
        When True, activate the sharded non-persistent-chunk path
        described in the module docstring. When False (the default), the
        manager behaves identically to the M4.5 / M6 snapshot: every
        rank holds a full copy of each non-persistent chunk on CPU and
        cross-rank grad sync uses per-param ``all_reduce(op=AVG)``
        (ZeRO-2-ish, composes cleanly under an outer DDP wrapper).
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
        world_size: int = 1,
        rank: int = 0,
        zero3_shard: bool = False,
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

        # ZeRO-3 sharding context. ``world_size`` and ``rank`` default
        # to the single-rank case; when either is > default AND
        # ``zero3_shard`` is True, :meth:`materialize_offload` creates
        # per-rank CPU shards and :meth:`gather` /
        # :meth:`reduce_grads_and_offload` take the collectives path.
        self.world_size: int = int(max(1, world_size))
        self.rank: int = int(max(0, rank))
        if self.rank >= self.world_size:
            raise ValueError(
                f"rank={self.rank} out of range for world_size={self.world_size}"
            )
        # Sharding is only physically active when BOTH the flag is set
        # and we have peers to talk to. With ``world_size == 1`` a
        # "sharded" chunk would be the full chunk (a rank of 1 talking
        # to itself) — degrading cleanly to the ZeRO-2-style replication
        # path keeps the unit tests for zero3_shard=True viable on
        # single-GPU hosts.
        self.zero3_shard: bool = bool(zero3_shard) and self.world_size > 1

        # When True, :meth:`reduce_grads_and_offload` and the per-param
        # grad-offload hook skip their internal ``dist.all_reduce`` calls
        # and trust an outer layer (typically ``DistributedDataParallel``
        # wrapped over the protrain'd module) to own cross-rank grad
        # sync. Toggled by ``protrain_model_wrapper`` at compose-time —
        # see the Multi-GPU section of ``DESIGN.md``. Mutually exclusive
        # with ``zero3_shard=True``: the sharded path is the grad-sync
        # point in its own right (reduce_scatter), so an outer DDP
        # wouldn't compose anyway.
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

        # Per-chunk sharded state (ZeRO-3 path). Populated by
        # :meth:`materialize_offload` only when ``self.zero3_shard`` is
        # True and the chunk qualifies for sharding (homogeneous dtype).
        # Unset entries signal the chunk falls back to the replicated
        # path even in sharded mode.
        self._chunk_shards: dict[ChunkId, _ChunkShardState] = {}

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
            # BUG 2 FIX: each param's byte_offset must be aligned to its
            # element_size, otherwise ``byte_view.view(dtype)`` raises
            # ``RuntimeError: offset is not aligned``. This bites when a
            # chunk contains a mix of 2-byte (fp16/bf16) and 4-byte
            # (fp32) params — e.g. Llama's fp16 attention weights sitting
            # next to fp32 RMSNorm scales — because the running offset
            # lands on an odd multiple of 2 when an fp16 param precedes
            # an fp32 one. We pad each param's starting offset up to a
            # multiple of its element_size before laying it down; this
            # guarantees alignment for any dtype mix up to 8 bytes
            # (fp64). The padding bytes stay zero (we allocated with
            # ``torch.empty`` so technically uninitialized, but no code
            # ever reads a padding region — the only readers are the
            # per-param typed views and the per-param H2D copy which
            # only touches ``nbytes``).
            element_sizes: list[int] = []
            per_param_bytes: list[int] = []
            for pid in param_ids:
                param = self._params_by_id.get(pid)
                if param is None:
                    element_sizes.append(0)
                    per_param_bytes.append(0)
                    continue
                nbytes = int(param.numel()) * int(param.element_size())
                per_param_bytes.append(nbytes)
                element_sizes.append(int(param.element_size()))

            # Running-offset computation with per-param alignment, so
            # the actual chunk allocation size accounts for any padding
            # gaps.
            aligned_offsets: list[int] = []
            offset = 0
            for nbytes, esz in zip(per_param_bytes, element_sizes):
                if nbytes == 0 or esz == 0:
                    aligned_offsets.append(offset)
                    continue
                # Round offset up to the next multiple of esz.
                offset = ((offset + esz - 1) // esz) * esz
                aligned_offsets.append(offset)
                offset += nbytes
            chunk_bytes = offset

            if chunk_bytes == 0:
                continue

            # --- Step 1b: decide whether to shard this chunk ------------
            # Sharding is only viable if we're running with
            # ``zero3_shard=True`` AND the chunk's params share a single
            # element size (so the shard boundary can be aligned). For
            # mixed-dtype chunks (e.g. a trailing chunk holding both
            # fp16 weights and fp32 RMSNorm scales) we fall back to the
            # replicated path even when zero3_shard is on — this is
            # rare enough on Llama-style models that the memory gain is
            # negligible, and the alternative (padding each param to
            # max_element_size) wastes more memory than sharding saves.
            unique_esizes = {
                esz for esz in element_sizes if esz > 0
            }
            unique_dtypes = {
                self._params_by_id[pid].data.dtype
                for pid, nbytes in zip(param_ids, per_param_bytes)
                if nbytes > 0 and self._params_by_id.get(pid) is not None
            }
            chunk_is_shardable = (
                self.zero3_shard
                and len(unique_esizes) == 1
                and len(unique_dtypes) == 1
            )

            # --- Step 2: one pinned CPU allocation per chunk ------------
            # We allocate fresh pinned memory rather than reusing the
            # buffer_pool's pinned host region (that was sized to
            # ``n_buffer * S_chunk`` for staging, not persistent storage —
            # collisions mod n_buffer would corrupt data). Sizing is
            # precise: ``chunk_bytes`` bytes exactly (including any
            # per-param alignment padding).
            #
            # In the sharded path this full-chunk buffer is allocated
            # ONLY to perform the initial H2D→shard partition; after
            # the per-rank shard is populated it is released. Each rank
            # permanently holds only ``shard_bytes`` of pinned CPU
            # storage per chunk.
            if chunk_is_shardable:
                primary_esize = next(iter(unique_esizes))
                primary_dtype = next(iter(unique_dtypes))
                # Pad chunk_bytes up so (chunk_bytes_padded / world_size)
                # is both integral and a multiple of primary_esize.
                # ``lcm(world_size, primary_esize)`` is the smallest
                # padded size that satisfies both. For fp16
                # (primary_esize=2) and world_size=4, the total pads up
                # to a multiple of 4 bytes; shard_bytes is a multiple
                # of 2 (fp16-aligned), as required by ``.view(dtype)``
                # after ``all_gather`` reassembles the chunk.
                import math as _math
                pad_unit = (primary_esize * self.world_size) // _math.gcd(
                    primary_esize, self.world_size
                )
                chunk_bytes_padded = (
                    (chunk_bytes + pad_unit - 1) // pad_unit
                ) * pad_unit
                shard_bytes = chunk_bytes_padded // self.world_size
            else:
                chunk_bytes_padded = chunk_bytes
                shard_bytes = 0
                primary_esize = 0
                primary_dtype = None  # type: ignore[assignment]

            # Full-chunk buffer (transient in sharded mode, permanent
            # otherwise).
            cpu_bytes = torch.empty(
                chunk_bytes_padded, dtype=torch.uint8, pin_memory=True
            )

            # --- Step 3: copy + rebind param.data -----------------------
            slots: list[_CpuParamSlot] = []
            trainable_count = 0
            for pid, nbytes, off in zip(
                param_ids, per_param_bytes, aligned_offsets
            ):
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
                cpu_view = cpu_bytes.narrow(0, off, nbytes)
                cpu_param = cpu_view.view(dtype).view(shape)
                cpu_param.copy_(orig_data)

                # Release GPU storage by rebinding .data to an empty
                # placeholder of the same dtype.
                param.data = self._empty_placeholder(dtype)

                # Optional: pinned CPU grad buffer for trainable params.
                # In the sharded path we do NOT allocate a per-param
                # grad tensor — the shard-level grad buffer
                # (``cpu_shard_grad_bytes``) covers every param's
                # contribution to this rank's slice. Keeping
                # ``cpu_grad=None`` for sharded slots disables the
                # per-param-hook D2H in :meth:`_make_grad_offload_hook`
                # (see the hook body's sharded-mode short-circuit).
                cpu_grad: "torch.Tensor | None" = None
                if param.requires_grad:
                    trainable_count += 1
                    if not chunk_is_shardable:
                        cpu_grad = torch.zeros(
                            shape, dtype=dtype, pin_memory=True
                        )

                # For sharded chunks ``slot.cpu_data`` points into the
                # full-chunk transient buffer — but that buffer is
                # about to be released. Set cpu_data=None on sharded
                # slots; the only consumer (the H2D copy inside
                # ``_rebind_params_to_buffer`` on the replicated path)
                # never runs for sharded chunks (gather handles bytes
                # through all_gather, not per-slot H2D).
                slot_cpu_data: "torch.Tensor | None" = None
                if not chunk_is_shardable:
                    slot_cpu_data = cpu_param

                slots.append(
                    _CpuParamSlot(
                        param_id=pid,
                        cpu_data=slot_cpu_data,
                        cpu_grad=cpu_grad,
                        shape=shape,
                        dtype=dtype,
                        byte_offset=off,
                        numel=numel,
                        element_size=element_size,
                    )
                )
                freed += nbytes

            self._cpu_slots[cid] = slots
            self._grad_initial[cid] = trainable_count
            self._grad_remaining[cid] = trainable_count

            # --- Step 3b: partition the full chunk bytes into this rank's shard
            # Only applies to shardable chunks. After this block the
            # full-chunk ``cpu_bytes`` tensor is no longer referenced
            # (Python GC will reclaim it).
            if chunk_is_shardable:
                # Pad the full-chunk buffer up to chunk_bytes_padded by
                # leaving any trailing bytes zero-initialized. The
                # ``torch.empty`` above did NOT zero, so explicitly zero
                # the tail so peer ranks with trailing slices don't hold
                # uninitialized bytes that would then propagate through
                # all_gather on the first gather (correctness doesn't
                # depend on this since the initial gather overwrites
                # with the trained values anyway — but a zero-init makes
                # the first-iter param.data deterministic).
                if chunk_bytes_padded > chunk_bytes:
                    cpu_bytes.narrow(
                        0, chunk_bytes, chunk_bytes_padded - chunk_bytes
                    ).zero_()
                # This rank's byte slice of the padded full chunk.
                my_off = self.rank * shard_bytes
                my_end = my_off + shard_bytes
                cpu_shard_bytes = torch.empty(
                    shard_bytes, dtype=torch.uint8, pin_memory=True
                )
                cpu_shard_bytes.copy_(
                    cpu_bytes.narrow(0, my_off, shard_bytes)
                )
                cpu_shard_grad_bytes = torch.zeros(
                    shard_bytes, dtype=torch.uint8, pin_memory=True
                )
                # Shard-level nn.Parameter — the CPU Adam's view of this
                # rank's slice. Build it against the pinned bytes
                # reinterpreted as primary_dtype so DeepSpeedCPUAdam's
                # element-wise updates land on the right storage.
                from torch import nn as _nn
                shard_numel = shard_bytes // primary_esize
                shard_view = cpu_shard_bytes.view(primary_dtype).view(
                    shard_numel
                )
                shard_param = _nn.Parameter(shard_view, requires_grad=True)
                # Pin its grad at a view of the pinned grad bytes so
                # the CPU Adam reads the right storage without a copy.
                shard_grad_view = cpu_shard_grad_bytes.view(
                    primary_dtype
                ).view(shard_numel)
                shard_param.grad = shard_grad_view

                self._chunk_shards[cid] = _ChunkShardState(
                    cpu_shard_bytes=cpu_shard_bytes,
                    cpu_shard_grad_bytes=cpu_shard_grad_bytes,
                    chunk_bytes=chunk_bytes_padded,
                    shard_bytes=shard_bytes,
                    primary_dtype=primary_dtype,
                    primary_element_size=primary_esize,
                    shard_param=shard_param,
                )

            # --- Step 4: per-param grad hooks for trainable params -----
            # In sharded mode the hook still fires per-param — we need
            # the counter decrement so :meth:`reduce_grads_and_offload`
            # can tell when every param in the chunk has an accumulated
            # grad. The hook body takes a different fast-path for
            # sharded slots (see :meth:`_make_grad_offload_hook`).
            for slot in slots:
                param = self._params_by_id[slot.param_id]
                if not param.requires_grad:
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

            # ---- M7 sharded fast-path ----------------------------------
            # When this chunk has a shard state, the per-param hook does
            # NOT:
            #   * all_reduce the grad (done at chunk level via reduce_scatter)
            #   * copy the grad to CPU (reduce_scatter drains to CPU)
            #   * kick CPU Adam (deferred to reduce_grads_and_offload)
            #   * null the grad (it needs to live on GPU until the
            #     chunk-level reduce_scatter collects every param's grad)
            # We still decrement the chunk counter so the block-level
            # scheduler knows backward-for-this-chunk is done.
            shard_state_local = cm._chunk_shards.get(captured_cid)
            if shard_state_local is not None:
                remaining = cm._grad_remaining.get(captured_cid, 0) - 1
                cm._grad_remaining[captured_cid] = remaining
                return

            # ---- Replicated (non-sharded) path: original M4.5 logic ----
            # Multi-rank data-parallel path: reduce the GPU grad across
            # ranks (AVG = sum / world_size) BEFORE draining to the CPU
            # shard. Guarded on world_size > 1 AND ``skip_internal_grad_reduce``
            # being False — the M6 DDP-composed stack sets the flag to
            # True so DDP's own bucketed allreduce handles this sync
            # and we don't do a second per-param reduce here. In a bare
            # non-DDP distributed run the flag is False and this is the
            # sole grad-sync point.
            import torch as _torch
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
            # BUG 1 FIX: record a CUDA event on the current stream the
            # moment the async D2H is dispatched. The CPU Adam worker
            # thread will synchronize on this event before reading the
            # pinned grad shard — without the wait, the worker can race
            # the D2H and read uninitialized/partial bytes the moment
            # the ThreadPoolExecutor pops its queue (DeepSpeedCPUAdam
            # holds no implicit CUDA-side ordering). Recording the event
            # here (after copy_) captures the D2H completion exactly;
            # the event itself is cheap to record.
            d2h_event = None
            if param.grad.is_cuda:
                d2h_event = _torch.cuda.Event(blocking=True)
                d2h_event.record()
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
                    # BUG 4 FIX: after the worker thread runs
                    # ``optim.step()`` the CPU shards hold the updated
                    # weights, but ``param.data`` still points at the
                    # CPU tensor (we repointed it in
                    # _ensure_cpu_grads_attached). Install a post_step
                    # callback that repoints ``param.data`` back to the
                    # GPU empty placeholder so any intermediate code
                    # reading ``.data`` between iters (clip_grad_norm_,
                    # checkpoint save, Trainer metric hooks) sees a
                    # zero-element GPU tensor — matching the invariant
                    # the rest of the runtime relies on. The CPU master
                    # weights are still held by ``slot.cpu_data`` so
                    # the next gather() flows the updated values back
                    # to GPU via its H2D copy.
                    cm.cpu_optim.step_async(
                        captured_cid,
                        d2h_event=d2h_event,
                        post_step=cm._make_post_cpu_step_repoint(captured_cid),
                    )
                # Reset the counter now so the next backward fires again.
                cm._grad_remaining[captured_cid] = cm._grad_initial.get(
                    captured_cid, 0
                )

        return _hook

    def _make_post_cpu_step_repoint(self, chunk_id: ChunkId):
        """Build the after-step callback that repoints ``.data`` back to GPU.

        BUG 4 FIX: between the end of iter N's optimizer step and the
        start of iter N+1's gather, ``param.data`` must be a GPU tensor
        (zero-element is fine — it's the same empty-placeholder used
        elsewhere in the runtime). If we leave it pointing at the CPU
        master shard, any caller between iters (clip_grad_norm_, Trainer
        logging, checkpoint save) sees a CPU tensor where a GPU tensor
        was expected. The CPU shard continues to hold the post-step
        weights; the next :meth:`gather` H2D-copies them into the GPU
        buffer.
        """
        cm = self
        captured_cid = chunk_id

        def _repoint() -> None:
            slots = cm._cpu_slots.get(captured_cid, [])
            for slot in slots:
                param = cm._params_by_id.get(slot.param_id)
                if param is None:
                    continue
                param.data = cm._empty_placeholder(slot.dtype)
                # Also clear grad: we've consumed it in the CPU step,
                # and leaving param.grad pointing at the CPU grad shard
                # means iter N+1's autograd would accumulate new GPU
                # grad onto a CPU tensor → "expected same device" fail.
                param.grad = None

        return _repoint

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

        Non-persistent chunks (replicated path): acquire a GPU buffer
        from the pool, copy the chunk's CPU bytes into it (skipping the
        copy if the chunk is already resident-tagged in the pool), and
        rebind every param's ``.data`` to a GPU view.

        Non-persistent chunks (sharded path, ``zero3_shard=True`` AND
        chunk has a shard state): each rank H2D-uploads its
        ``shard_bytes`` CPU shard into a slice of the pool buffer, then
        issues ``torch.distributed.all_gather_into_tensor`` to fill the
        full-chunk buffer from every rank's contribution. After the
        collective the buffer holds the full chunk on every rank, and
        params are rebound exactly as in the replicated path.

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

        shard_state = self._chunk_shards.get(chunk_id)

        # Consult the pool for a still-resident tag (forward→backward
        # reuse window). The all_gather path skips this re-use: the
        # collective cost is < re-running all_gather's worth of data
        # motion, but the correctness invariant (every rank sees the
        # SAME full chunk) requires the full chunk to be present —
        # which is what ``lookup_resident`` guarantees when it returns
        # a non-None buffer. The shard state's presence doesn't change
        # the cache-hit semantics; only the cache-miss path diverges.
        resident = self.buffer_pool.lookup_resident(chunk_id)
        if resident is not None:
            buf = self.buffer_pool.acquire(chunk_id)
            self._rebind_params_to_buffer(chunk_id, buf, needs_copy=False)
            return

        # Cache miss.
        buf = self.buffer_pool.acquire(chunk_id)
        if shard_state is not None:
            self._gather_sharded(chunk_id, buf, shard_state)
            self._rebind_params_to_buffer(chunk_id, buf, needs_copy=False)
            return

        # Replicated path: per-slot H2D copies directly into the buffer.
        self._rebind_params_to_buffer(chunk_id, buf, needs_copy=True)

    def _gather_sharded(
        self,
        chunk_id: ChunkId,
        buf: "torch.Tensor",
        shard_state: "_ChunkShardState",
    ) -> None:
        """ZeRO-3 all_gather path: reconstruct the full chunk on GPU.

        Uses ``torch.distributed.all_gather_into_tensor`` (new in
        torch 2.1+; confirmed present on this codebase's torch 2.10).
        The gather layout is rank-contiguous: rank ``r``'s bytes
        occupy ``[r * shard_bytes, (r + 1) * shard_bytes)`` of the
        gathered full-chunk buffer, mirroring the partition applied
        at ``materialize_offload`` time.
        """
        import torch
        import torch.distributed as dist

        shard_bytes = shard_state.shard_bytes
        full_bytes = shard_state.chunk_bytes  # padded
        # We write the all_gather output directly into the pool buffer
        # (truncated to ``full_bytes`` — the pool buffer is S_chunk
        # wide which may be > full_bytes for non-final chunks, but the
        # collective only writes the prefix).
        #
        # H2D the local shard into pinned-free GPU staging. For
        # correctness all_gather_into_tensor requires the input to live
        # on the same device as the output (the GPU buffer) and the
        # dtypes to match. We allocate a staging tensor on the same
        # device as ``buf``.
        gather_out = buf.narrow(0, 0, full_bytes)
        my_shard_gpu = torch.empty(
            shard_bytes, dtype=torch.uint8, device=buf.device
        )
        my_shard_gpu.copy_(shard_state.cpu_shard_bytes, non_blocking=True)
        dist.all_gather_into_tensor(gather_out, my_shard_gpu)

    def _rebind_params_to_buffer(
        self,
        chunk_id: ChunkId,
        buf: "torch.Tensor",
        needs_copy: bool,
    ) -> None:
        """Copy CPU shards into ``buf`` (if needed) and rebind each param's data.

        ``buf`` is the pool-owned GPU uint8 tensor of length ``S_chunk``.
        For each param slot we slice off
        ``slot.byte_offset .. +slot.numel*slot.element_size``, reinterpret
        it as the param's dtype, reshape to the param's shape, and
        assign to ``param.data``. ``slot.byte_offset`` already includes
        any per-param alignment padding applied by
        :meth:`materialize_offload` (BUG 2 fix), so the GPU buffer layout
        mirrors the pinned CPU layout exactly.
        """
        slots = self._cpu_slots.get(chunk_id, [])
        if not slots:
            return

        if needs_copy:
            for slot in slots:
                nbytes = slot.numel * slot.element_size
                # Slice the buffer at this param's recorded
                # (alignment-padded) byte offset — same offset used for
                # the pinned CPU layout in materialize_offload — and view
                # as the param's dtype+shape for an element-typed copy.
                dst_bytes = buf.narrow(0, slot.byte_offset, nbytes)
                dst_typed = dst_bytes.view(slot.dtype).view(slot.shape)
                dst_typed.copy_(slot.cpu_data, non_blocking=True)

        # Rebind .data unconditionally — even on the no-copy path, a
        # previous offload() nulled out param.data, and re-acquiring from
        # the pool keeps the GPU bytes but requires re-pointing the
        # param at them.
        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            nbytes = slot.numel * slot.element_size
            # Slice the chunk buffer at this param's byte offset (with
            # alignment padding already baked in) and view as
            # (dtype, shape).
            byte_view = buf.narrow(0, slot.byte_offset, nbytes)
            typed = byte_view.view(slot.dtype).view(slot.shape)
            param.data = typed

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

        # ---- Non-persistent sharded path -------------------------------
        shard_state = self._chunk_shards.get(chunk_id)
        if shard_state is not None:
            self._reduce_scatter_and_offload_shard(chunk_id, shard_state)
            self.offload(chunk_id)
            return

        # Non-persistent, replicated: grad offload is owned by
        # _offload_grad (per-param hooks). The block-granularity
        # scheduler here releases the chunk buffer AND nulls the
        # param.data placeholder so the GPU storage is fully freed and
        # the params are in a clean state for the next gather.
        self.offload(chunk_id)

    def _reduce_scatter_and_offload_shard(
        self, chunk_id: ChunkId, shard_state: "_ChunkShardState"
    ) -> None:
        """Sharded path: reduce_scatter chunk grads, D2H shard, kick CPU Adam.

        Precondition: every trainable param in the chunk has a GPU grad
        (backward drained the chunk). Postcondition: every GPU grad is
        nulled, this rank's CPU shard grad holds its slice of the
        ``AVG``-reduced cross-rank grad, and the CPU Adam step for
        this chunk has been submitted to the async worker.
        """
        import torch
        import torch.distributed as dist

        shard_bytes = shard_state.shard_bytes
        chunk_bytes = shard_state.chunk_bytes
        primary_dtype = shard_state.primary_dtype
        primary_esize = shard_state.primary_element_size

        slots = self._cpu_slots.get(chunk_id, [])
        if not slots:
            return

        # Device from the first live param.grad (all params in a chunk
        # share a device by construction).
        device = self.device
        for slot in slots:
            p = self._params_by_id.get(slot.param_id)
            if p is not None and p.grad is not None:
                device = p.grad.device
                break

        # Flatten every param's grad bytes into a full-chunk buffer at
        # the recorded byte offsets — same layout the all_gather output
        # occupies. Trailing pad bytes stay zero.
        grad_flat_bytes = torch.zeros(
            chunk_bytes, dtype=torch.uint8, device=device
        )
        any_grad = False
        for slot in slots:
            p = self._params_by_id.get(slot.param_id)
            if p is None or p.grad is None:
                continue
            any_grad = True
            nbytes = slot.numel * slot.element_size
            dst_bytes = grad_flat_bytes.narrow(0, slot.byte_offset, nbytes)
            dst_typed = dst_bytes.view(slot.dtype).view(slot.shape)
            dst_typed.copy_(p.grad)
            # Null the GPU grad now that we've captured its bytes.
            p.grad = None

        if not any_grad:
            return

        # reduce_scatter_tensor requires matching typed views on input
        # (full chunk) and output (this rank's shard). Reinterpret the
        # byte buffer as the primary dtype.
        shard_numel = shard_bytes // primary_esize
        full_numel = chunk_bytes // primary_esize
        grad_flat_typed = grad_flat_bytes.view(primary_dtype).view(full_numel)
        my_shard_grad_gpu = torch.empty(
            shard_numel, dtype=primary_dtype, device=device
        )
        dist.reduce_scatter_tensor(
            my_shard_grad_gpu, grad_flat_typed, op=dist.ReduceOp.AVG
        )

        # D2H the rank's grad slice to the pinned shard grad. The
        # shard_param.grad was pinned to a view over
        # cpu_shard_grad_bytes at materialize_offload time; copying
        # into it is what makes the CPU Adam see the fresh grad.
        d2h_event = None
        if my_shard_grad_gpu.is_cuda:
            shard_state.shard_param.grad.copy_(  # type: ignore[union-attr]
                my_shard_grad_gpu, non_blocking=True
            )
            d2h_event = torch.cuda.Event(blocking=True)
            d2h_event.record()
        else:
            shard_state.shard_param.grad.copy_(my_shard_grad_gpu)  # type: ignore[union-attr]

        # Reset the hook counter so the next backward's per-param
        # decrements land correctly.
        self._grad_remaining[chunk_id] = self._grad_initial.get(chunk_id, 0)

        # Kick async CPU Adam for this chunk's shard. The adapter's
        # per-chunk optim was built over shard_state.shard_param, so
        # step_async updates only this rank's slice.
        if self.cpu_optim is not None:
            self.cpu_optim.step_async(
                chunk_id, d2h_event=d2h_event, post_step=None
            )

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

    # ---- introspection for tests --------------------------------------

    def sharded_chunk_ids(self) -> list[ChunkId]:
        """Return the list of chunks currently held in ZeRO-3 sharded form.

        Useful for test assertions: a non-empty list confirms the
        ``zero3_shard`` path engaged at ``materialize_offload`` time.
        """
        return sorted(self._chunk_shards.keys())

    def shard_bytes_for(self, chunk_id: ChunkId) -> int:
        """Return this rank's ``shard_bytes`` for ``chunk_id``.

        Returns 0 when the chunk is not sharded (persistent or dropped
        out of the sharded path due to mixed-dtype).
        """
        s = self._chunk_shards.get(chunk_id)
        return 0 if s is None else s.shard_bytes

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
        ``self._cpu_slots``; the M7 sharded semantics are the shard
        state in ``self._chunk_shards``.
        """
        slots = self._cpu_slots.get(chunk_id)
        if not slots:
            # Fall back to the M2 pool-slot semantics for chunks that
            # were never materialize_offload'd (e.g. bare unit tests).
            slot = int(chunk_id) % self.buffer_pool.n_buffer
            return self.buffer_pool.pinned_host.buffer(slot)
        if slots[0].cpu_data is None:
            # Sharded slot — return the shard bytes reinterpreted as the
            # primary dtype as a best-effort legacy answer.
            shard = self._chunk_shards.get(chunk_id)
            if shard is not None:
                return shard.cpu_shard_bytes.view(shard.primary_dtype)
        return slots[0].cpu_data  # type: ignore[return-value]


__all__ = ["ChunkManager"]
