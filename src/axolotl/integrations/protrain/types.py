"""Shared data types for the ProTrain memory manager.

Pure data shapes only — no runtime logic, no torch tensors allocated at import
time. Every downstream subpackage (profiler, chunk, block, cost, search,
runtime, api) depends on this module. Keeping it allocation-light lets the
subpackages develop in parallel against a stable contract.

Paper references: MLSys 2026, arXiv 2406.08334 (§3.1–3.3, Appendix A–B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, NewType

if TYPE_CHECKING:
    from torch import nn


# ---------------------------------------------------------------------------
# Identifier aliases
# ---------------------------------------------------------------------------

# Dotted path from `model.named_parameters()`, e.g. "layers.0.attn.q_proj.weight".
# Stable across pickling, debuggable, and what all profiler/chunk modules key on.
ParamId = NewType("ParamId", str)

# Monotonic op index during the profiler's single-iteration trace.
OpId = NewType("OpId", int)

# Transformer block index, 0 .. N_block-1.
BlockId = NewType("BlockId", int)

# Chunk index, 0 .. N_chunk-1.
ChunkId = NewType("ChunkId", int)


# ---------------------------------------------------------------------------
# Block modes (§3.1.2)
# ---------------------------------------------------------------------------


class BlockMode(str, Enum):
    """Activation strategy selected per transformer block."""

    NONE = "none"   # keep activations on GPU, no checkpoint, no swap
    CKPT = "ckpt"   # drop + recompute in backward
    SWAP = "swap"   # offload to CPU in forward, prefetch in backward (feature-flagged)


# Per-block mode selection, output of `block.layout_rules.assign_modes`.
BlockStrategyMap = dict[BlockId, BlockMode]


# ---------------------------------------------------------------------------
# Profiler inputs + outputs (§3.2, App A.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpRecord:
    """One op captured during the profiler trace."""

    op_id: OpId
    module_path: str                                  # dotted nn.Module path owning this op
    qualified_name: str                               # e.g. "aten::addmm", "prim::Constant"
    shape_signature: tuple[tuple[int, ...], ...]     # input tensor shapes
    block_id: BlockId | None                          # transformer block, if inside one
    is_forward: bool                                  # True for fwd, False for bwd


@dataclass(frozen=True)
class ProfilerConfig:
    """Arguments to `profiler.trace.run_trace`."""

    batch_size: int
    seq_len: int
    device: str                                       # e.g. "cuda:2"
    include_backward: bool = True
    on_demand: bool = True                            # OnDemandTensorMgr for models > single-GPU
    # Distributed world size. ``None`` (default) means "auto-detect" — the
    # tracer probes ``torch.distributed.get_world_size()`` if a process
    # group is initialized and falls back to 1 otherwise. Pass an explicit
    # int to force a specific size (sanity-checked against the live group
    # by ``measure_nccl``).
    world_size: int | None = None


@dataclass(frozen=True)
class ProfilerTrace:
    """Serializable single-iteration trace. Cache key: (arch_hash, bs, seq, sku, world).

    Re-profile triggers: any change to model arch, batch_size * seq_len, GPU SKU or
    count, PCIe/NVLink topology (§7).
    """

    # Operator trace
    op_order: tuple[OpRecord, ...]
    intra_op_delta: dict[OpId, int]                   # bytes; peak_during_op - allocated_before_op
    inter_op_delta: dict[OpId, int]                   # bytes; peak_between_hooks - allocated_prev_end

    # Per-block summaries
    activation_sizes: dict[BlockId, int]              # retained-activation bytes per block

    # Model-state constants (constant across the run given the model + dtype config)
    model_state_bytes: int                            # fp16 params + grads + fp32 master + momentums

    # Hardware microbenchmarks (§3.2 hardware profiling)
    pcie_h2d_bps: float
    pcie_d2h_bps: float
    nccl_gather_s: dict[int, float]                   # keyed by payload size in bytes
    nccl_reduce_s: dict[int, float]

    # Cache key components
    arch_hash: str                                    # deterministic hash of model architecture
    bs: int
    seq: int
    sku: str                                          # torch.cuda.get_device_name() result
    world: int                                        # world_size at profile time

    # Per-op wall-clock latencies (seconds), measured via torch.cuda.Event during
    # the same single-iteration trace. Keys match ``op_order[i].op_id``. Populated
    # for forward ops and for the synthetic ``<backward>`` op that stands in for
    # the aggregate backward pass. Consumed by ``cost/runtime.py`` to replace the
    # activation-bytes compute-rate proxy with measured per-block compute time.
    # Optional: traces predating this field deserialize with an empty dict, in
    # which case ``cost/runtime.py`` falls back to the roofline proxy and logs a
    # warning. New in TRACE_VERSION=2 (see profiler/cache.py).
    op_latencies: dict[OpId, float] = field(default_factory=dict)

    # Measured CPU / GPU Adam throughput (bytes/sec) from the hw_bench
    # microbenchmarks. Replaces the hardcoded ``_CPU_ADAM_BYTES_PER_SEC``
    # / ``_GPU_ADAM_BYTES_PER_SEC`` priors in ``cost/runtime.py``. 0.0
    # means "unavailable" — the cost model falls back to a hardcoded
    # prior and logs a warning. New in TRACE_VERSION=3.
    cpu_adam_bytes_per_sec: float = 0.0
    gpu_adam_bytes_per_sec: float = 0.0

    # Hook-dispatch calibration fields — new in TRACE_VERSION=4.
    #
    # The profiler installs pre/post forward hooks on every ``nn.Module`` to
    # record per-op memory deltas + latencies. On transformer-sized models
    # (~1000 leaf modules) the hook dispatch alone inflates measured forward
    # wall time ~2.5x over a steady-state (hook-less) forward. The cost
    # model consumes this ratio to scale the hooked per-op latencies down
    # to a realistic prior:
    #
    #   scale = steady_fwd_wall_s / hooked_fwd_wall_s
    #   t_fwd_calibrated = sum(per_block_latencies) * scale
    #
    # ``hooked_fwd_wall_s`` is the total wall-clock of the hooked forward
    # (measured via a ``torch.cuda.Event`` pair around the full forward
    # pass, NOT summed from per-op latencies — that sum misses inter-op
    # Python overhead).
    #
    # ``steady_fwd_wall_s`` is the same forward measured BEFORE hooks are
    # installed, on the same warm model + batch, with a pair of un-hooked
    # warmup passes first so allocator state is representative.
    #
    # ``steady_bwd_wall_s`` is the hook-less backward wall-clock, captured
    # on a separately-timed un-hooked backward (optional; 0.0 means
    # "unavailable" — the cost model falls back to ``bwd_fwd_ratio`` of
    # the scaled forward).
    #
    # Traces loaded from cache that predate v4 have 0.0 defaults here; the
    # cost model detects the 0.0 and falls back to the unscaled per-op
    # sum (identity scale factor), preserving backward compatibility until
    # the cache is refreshed.
    hooked_fwd_wall_s: float = 0.0
    steady_fwd_wall_s: float = 0.0
    steady_bwd_wall_s: float = 0.0
    # ``steady_fwd_peak_bytes`` is ``torch.cuda.max_memory_allocated()``
    # captured across the hook-less steady forward pass. Used by the
    # memory cost model as a ground-truth floor on the forward
    # contribution — eliminates the search's "retained-NONE-activations"
    # over-estimate when a hot-iter measurement is available. 0 means
    # unavailable (pre-v5 cached traces, or CUDA unavailable at profile
    # time).
    steady_fwd_peak_bytes: int = 0

    # Per-block peak bytes captured during the hook-less steady forward.
    # Lightweight forward pre/post hooks installed ONLY at block level (tens
    # of blocks, not the ~1000 leaves the main profiling path targets) call
    # ``torch.cuda.reset_peak_memory_stats`` before each block and read
    # ``torch.cuda.max_memory_allocated`` after. Keys are transformer block
    # indices discovered via ``discover_blocks``; values are per-block peak
    # bytes observed during that block's forward.
    #
    # The memory cost model consumes ``max(steady_fwd_block_peak_bytes.values())``
    # as a ground-truth upper bound on the FORWARD peak for any NONE/CKPT/SWAP
    # mix — unlike ``steady_fwd_peak_bytes`` (which is an aggregate only valid
    # for all-NONE configs), the per-block max bounds any fractional-NONE
    # config too: CKPT/SWAP blocks free their activations before the next
    # block runs, so the forward peak across a mixed configuration cannot
    # exceed the max per-block peak observed during the all-NONE profile.
    # Backward CKPT recomputation bumps are added on top because they occur
    # during backward and weren't measured here.
    #
    # Empty dict means unavailable (pre-v6 cached traces, or CUDA unavailable
    # at profile time). New in TRACE_VERSION=6.
    steady_fwd_block_peak_bytes: dict[BlockId, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunk layout (§3.1.1, App B.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkLayout:
    """Per-rank chunk assignment plus intra-chunk ordering. Output of M2 layout pass."""

    S_chunk: int                                      # bytes per chunk
    N_chunk: int                                      # total chunks
    chunks: tuple[tuple[ParamId, ...], ...]           # exec-order within each chunk
    param_to_chunk: dict[ParamId, ChunkId]
    block_to_chunks: dict[BlockId, tuple[ChunkId, ...]]


# ---------------------------------------------------------------------------
# Cost / search (§3.3, App A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostConfig:
    """The four tunable knobs (§3.3 table)."""

    n_persist: int                                    # chunks pinned on GPU
    n_buffer: int                                     # pre-allocated chunk buffers
    n_swap: int                                       # blocks using activation swap
    n_checkpoint: int                                 # blocks using gradient checkpointing


@dataclass(frozen=True)
class Bounds:
    """Upper bounds on the four knobs, derived from trace + layout."""

    N_chunk: int
    N_block: int
    N_interval: int                                   # swap-interval bound in compute units


@dataclass(frozen=True)
class SearchResult:
    """Output of `search.exhaustive.search`."""

    cfg: CostConfig
    block_map: BlockStrategyMap
    predicted_peak_bytes: int
    predicted_iter_s: float


# ---------------------------------------------------------------------------
# Hardware profile (§3.2, §7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """Static hardware description consumed by the searcher.

    ProTrain is RTX 3090 / 3090 Ti scoped for this workstream — treat the two
    SKUs as equivalent when picking the target pool.

    The ``zero3_shard`` flag is plumbed from ``protrain_model_wrapper`` (which
    decides sharding on/off via the same auto-detect logic documented in
    ``DESIGN.md §Multi-GPU``) through to ``cost/memory.estimate_cpu_footprint``
    so per-rank CPU-pressure accounting reflects ZeRO-3 partitioning. It does
    NOT change the GPU peak estimate — the gather materializes the full chunk
    on GPU regardless of sharding — so ``estimate_peak`` ignores this field.
    """

    gpu_sku: str
    gpu_memory_bytes: int
    gpu_count: int                                    # world size for this run
    pcie_h2d_bps: float
    pcie_d2h_bps: float
    has_nvlink: bool                                  # informational; we never use NVLink paths
    zero3_shard: bool = False                         # True when M7 chunk-sharding is active
    # Measured Adam throughput (bytes/sec). 0.0 means "unavailable" —
    # ``cost/runtime.estimate_runtime`` falls back to a hardcoded prior in
    # that case. Populated by
    # :func:`axolotl.integrations.protrain.profiler.hw_bench.measure_cpu_adam`
    # and ``measure_gpu_adam`` after :func:`run_trace` completes, then
    # plumbed into the HardwareProfile the searcher consumes. New in
    # TRACE_VERSION=3 (see profiler/cache.py).
    cpu_adam_bytes_per_sec: float = 0.0
    gpu_adam_bytes_per_sec: float = 0.0


# ---------------------------------------------------------------------------
# Wrapped model handle (api/)
# ---------------------------------------------------------------------------


@dataclass
class WrappedModel:
    """Opaque handle returned by `protrain_model_wrapper`.

    Owns: ChunkManager, BlockStrategyMap (via search_result), installed hooks, the
    chosen SearchResult, and the Scheduler. Mutable because it holds runtime state
    (hook handles, buffer pool). Concrete internal types are `object` here to keep
    this module pure data — see `chunk.manager`, `runtime.scheduler`, etc.
    """

    module: "nn.Module"                               # the original model, with hooks installed
    search_result: SearchResult
    chunk_manager: object = None
    scheduler: object = None
    _hook_handles: list[object] = field(default_factory=list)


__all__ = [
    "ParamId",
    "OpId",
    "BlockId",
    "ChunkId",
    "BlockMode",
    "BlockStrategyMap",
    "OpRecord",
    "ProfilerConfig",
    "ProfilerTrace",
    "ChunkLayout",
    "CostConfig",
    "Bounds",
    "SearchResult",
    "HardwareProfile",
    "WrappedModel",
]
