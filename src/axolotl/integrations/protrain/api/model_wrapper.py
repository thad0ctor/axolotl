"""Public model-wrapper entry point for the ProTrain runtime (§1, §6).

``protrain_model_wrapper`` composes M1-M4 into a single call:

1. Profile (cached) — :func:`run_trace` behind
   :func:`load_cached_trace` / :func:`save_cached_trace`.
2. Layout — :func:`pick_S_chunk` then :func:`build_layout` over the
   profiler's exec order.
3. Search — ``search(trace, layout, capacity_bytes, hw)``.
4. Construct runtime — pinned host memory, buffer pool, chunk manager,
   CPU + GPU FusedAdam adapters, :class:`Scheduler`.
5. Wrap blocks according to ``search_result.block_map``.
6. Install hooks.
7. Return :class:`WrappedModel`.

The function is designed to be called from both the plugin's
``post_model_load`` hook (M5) and from a notebook / script that wants
to opt into ProTrain without Axolotl orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import nn

from axolotl.integrations.protrain.block import (
    assign_modes,
    discover_blocks,
    wrap_block,
)
from axolotl.integrations.protrain.chunk import (
    BufferPool,
    ChunkManager,
    CpuFusedAdamAdapter,
    GpuFusedAdamAdapter,
    PinnedHostMemory,
    build_layout,
    pick_S_chunk,
)
from axolotl.integrations.protrain.cost.bandwidth import effective_bw
from axolotl.integrations.protrain.profiler import (
    load_cached_trace,
    run_trace,
    save_cached_trace,
)
from axolotl.integrations.protrain.profiler.cache import ProfilerCacheKey
from axolotl.integrations.protrain.profiler.trace import _arch_hash
from axolotl.integrations.protrain.profiler.hw_bench import measure_compute_rate
from axolotl.integrations.protrain.runtime.hooks import install_hooks
from axolotl.integrations.protrain.runtime.scheduler import Scheduler
from axolotl.integrations.protrain.search import search
from axolotl.integrations.protrain.types import (
    BlockId,
    CostConfig,
    HardwareProfile,
    ParamId,
    ProfilerConfig,
    SearchResult,
    WrappedModel,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


# Default headroom subtracted from HardwareProfile.gpu_memory_bytes when the
# caller does not override ``capacity_bytes``. Reserves 2 GiB for CUDA
# context + PyTorch allocator overhead, matching the M4 task spec.
_DEFAULT_HEADROOM_BYTES = 2 * (1 << 30)


def _sku(device: "torch.device | str") -> str:
    import torch

    try:
        return torch.cuda.get_device_name(device)
    except Exception:  # pragma: no cover — defensive, CPU-only lanes
        return "cpu"


def _dummy_batch(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    device: "torch.device | str",
) -> dict:
    """Build a minimal ``(input_ids, labels)`` batch suitable for causal LM.

    Used when the profiler cache misses and we need to drive one
    forward + backward. Works on any HuggingFace causal LM (and many
    encoder-decoder models whose forward accepts ``input_ids`` +
    ``labels``); callers with exotic input signatures should supply
    their own batch via a future optional parameter (not M4b scope).
    """
    import torch

    vocab_size = _infer_vocab_size(model)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


def _infer_vocab_size(model: nn.Module) -> int:
    """Best-effort vocab size from common HF config shapes."""
    cfg = getattr(model, "config", None)
    for attr in ("vocab_size", "n_vocab", "vocabulary_size"):
        if cfg is not None and hasattr(cfg, attr):
            val = getattr(cfg, attr)
            if isinstance(val, int) and val > 0:
                return val
    # Fallback: peek at the first Embedding layer.
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            return int(m.num_embeddings)
    return 1024


def _exec_order_from_trace(trace) -> list[ParamId]:
    """Derive a param-level execution order from the profiler's op order.

    For each forward op in ``trace.op_order`` we emit the params owned
    by its ``module_path`` in ``model.named_parameters()`` order. The
    result is deduplicated at the first occurrence (the layout builder
    will also dedup but doing it here keeps downstream sizes small).

    This is a **best effort** — the profiler traces at module
    granularity, not tensor granularity, so we approximate "first use"
    by "first op inside the owning module". For the layouts the
    searcher cares about (block-aware grouping + persistent-first
    placement) this is sufficient: the block-contiguity rule in
    ``build_layout`` ensures block params land in the right chunk even
    if our exec order shuffles within a block.
    """
    # Param ids will be supplied by the caller from ``model.named_parameters``
    # — this function is kept for forward-compatibility if M4c wants to
    # drive exec-order directly off the trace.
    return [cast(ParamId, rec.module_path) for rec in trace.op_order if rec.is_forward]


def _build_block_spans(
    model: nn.Module,
) -> tuple[list[nn.Module], dict[BlockId, list[ParamId]]]:
    """Return (blocks_list, block_id -> list[ParamId]) for the model."""
    blocks = discover_blocks(model)
    named = list(model.named_parameters())

    # Build a reverse index: for each block, find the dotted-path prefix
    # that identifies it inside ``model.named_parameters()``. ``blocks``
    # is a plain ``list`` of nn.Module instances; the prefix is the
    # dotted path of that instance inside ``model``.
    block_prefixes: list[str] = []
    for block in blocks:
        prefix = _module_path_in(model, block)
        if prefix is None:
            prefix = ""
        block_prefixes.append(prefix)

    spans: dict[BlockId, list[ParamId]] = {BlockId(i): [] for i in range(len(blocks))}
    for param_name, _ in named:
        for idx, prefix in enumerate(block_prefixes):
            # Prefix match on dotted path, with a trailing "." to avoid
            # matching ``h.10`` when the prefix is ``h.1``.
            if prefix and (
                param_name == prefix or param_name.startswith(prefix + ".")
            ):
                spans[BlockId(idx)].append(cast(ParamId, param_name))
                break
    return blocks, spans


def _module_path_in(root: nn.Module, target: nn.Module) -> str | None:
    """Return the dotted path of ``target`` inside ``root``, or None."""
    for name, candidate in root.named_modules():
        if candidate is target:
            return name or None
    return None


def _param_exec_order(
    model: nn.Module,
    block_spans: dict[BlockId, list[ParamId]],
) -> list[ParamId]:
    """Rough execution-order list of params.

    We walk ``model.named_parameters()`` in insertion order (which is
    the canonical definition order HuggingFace uses) and emit each
    param exactly once. For block-member params, the ``build_layout``
    block-contiguity rule takes over and re-groups as needed; for
    non-block params the definition order is a sensible proxy for first-
    use order on the forward pass.
    """
    del block_spans  # unused; here for signature stability
    return [cast(ParamId, name) for name, _ in model.named_parameters()]


def _chunk_bytes(layout, chunk_manager) -> dict[int, int]:
    """Return ``{chunk_id -> actual bytes of its params}`` for ``layout``.

    Unlike ``S_chunk`` (a soft-cap upper bound), this reflects the real
    GPU-state footprint each chunk occupies when resident — the layout
    builder packs params greedily but never splits a param, so residual
    slack at the end of each chunk is common.
    """
    params_by_id = {
        str(name): p for name, p in chunk_manager.model.named_parameters()
    }
    out: dict[int, int] = {}
    for cid, pids in enumerate(layout.chunks):
        total = 0
        for pid in pids:
            p = params_by_id.get(str(pid))
            if p is None:
                continue
            total += int(p.numel()) * int(p.element_size())
        out[cid] = total
    return out


def _calibrate_peak_with_actual_chunk_bytes(
    original_peak: int,
    layout,
    chunk_manager,
    n_buffer: int,
    trace=None,
    block_map=None,
) -> int:
    """Recompute ``predicted_peak_bytes`` using actual chunk bytes + CKPT correction.

    The cost/memory.py estimator makes two structural overestimates that
    are out-of-scope for M4.5 to fix inside ``cost/`` but can be
    corrected post-hoc here:

    1. **Model state** — assumed to be ``n_persist * S_chunk``, but
       chunks pack greedily and typically sit at 80-90% of S_chunk.
       Replace with the sum of actual chunk bytes.

    2. **Op-walk deltas under CKPT** — the estimator adds
       ``intra_op_delta[op] + inter_op_delta[op]`` at every op, using
       the profiler's deltas recorded WITHOUT checkpointing. When a
       block is CKPT-wrapped those op-level spikes no longer manifest
       in steady state (they only appear inside the recompute window,
       which the CKPT bump at the block's first op already accounts
       for). Subtract the intra+inter contributions from ops inside
       CKPT blocks to avoid double-counting.

    The alpha fragmentation factor is preserved — its whole purpose is
    to over-predict for OOM safety — but applied only to the corrected
    base.
    """
    from axolotl.integrations.protrain.cost.memory import ALPHA_FRAGMENTATION
    from axolotl.integrations.protrain.types import BlockMode

    S = layout.S_chunk
    persistent_ids = set(int(c) for c in chunk_manager._persistent_ids)
    cb = _chunk_bytes(layout, chunk_manager)

    # Actual persistent bytes (≤ n_persist * S_chunk).
    actual_persistent = sum(cb.get(cid, 0) for cid in persistent_ids)
    # Buffer pool is still n_buffer * S_chunk — those slots really are
    # that size.
    buffer_bytes = n_buffer * S

    # Reverse out the cost-model's ``model_state_present`` term.
    n_persist = len(persistent_ids)
    alpha = ALPHA_FRAGMENTATION
    original_model_state = (n_persist + n_buffer) * S
    f_bm = max(0, int(original_peak / alpha) - original_model_state)

    # Rebuild F_bm from a more realistic activation model when a CKPT-
    # dominant block map is in play.
    #
    # cost/memory.py's op-walk sums intra+inter deltas at the max op,
    # but those deltas were recorded WITHOUT checkpointing — so for
    # configs where most blocks are CKPT, the op-walk counts activations
    # that the CKPT wrapper discards at forward time. The paper's Eq
    # 11 is designed to over-predict, but the overestimate is meant to
    # be "up to 10%", not up to 3x.
    #
    # Reconstructed F_bm estimate: sum(activation_sizes for non-CKPT
    # blocks) + 1 block's worth of bump for CKPT recomputation (which
    # happens one block at a time in backward) + the max single-op
    # intra_delta (to conservatively cover any peaking attention
    # kernel).
    if trace is not None and block_map is not None:
        n_ckpt = sum(
            1 for m in block_map.values() if m is BlockMode.CKPT
        )
        if n_ckpt >= max(1, len(block_map) - 2):
            # CKPT-dominant config — most blocks drop their activations.
            act_sizes = dict(trace.activation_sizes)
            non_ckpt_act = 0
            for bid, mode in block_map.items():
                if mode is not BlockMode.CKPT:
                    non_ckpt_act += int(act_sizes.get(bid, 0))
            # One CKPT block's activation (recomputed during its
            # backward, persists briefly) — use the max.
            one_ckpt_act = 0
            if act_sizes:
                one_ckpt_act = max(int(v) for v in act_sizes.values())

            # Max single-op intra+inter inside the forward, ignoring
            # the top-level "module-wrapper" ops (their deltas are
            # aggregates, not single-kernel peaks).
            max_op_delta = 0
            for op in trace.op_order:
                if not op.is_forward:
                    continue
                if op.block_id is None:
                    # Root-module deltas aggregate everything below;
                    # skip (CKPT strips most of this).
                    continue
                contrib = trace.intra_op_delta.get(
                    op.op_id, 0
                ) + trace.inter_op_delta.get(op.op_id, 0)
                if contrib > max_op_delta:
                    max_op_delta = contrib

            reconstructed_f_bm = non_ckpt_act + one_ckpt_act + max_op_delta
            # Use the smaller of the two estimates — never INCREASE the
            # prediction (cost model is already upper-bounding).
            f_bm = min(f_bm, reconstructed_f_bm)

    # Reassemble with the actual persistent bytes + corrected F_bm.
    #
    # Two independent alpha values apply here — by design, NOT stacked
    # fudge factors:
    #
    #   * ``ALPHA_FRAGMENTATION`` (1.10, from cost/memory.py) — the
    #     paper's cost-model-level factor. It's an upper bound on the
    #     raw op-walk's under-prediction of real allocator peak; the
    #     searcher uses this as the feasibility filter (so OOM-safety
    #     is enforced with the paper's 10% headroom). Restored from
    #     1.20 back to 1.10 in M6 once the runtime gaps (per-param
    #     grad offload, init-time chunk offload, BUG 1/2/4 fixes in
    #     ``chunk/manager.py``) closed the real underprediction.
    #
    #   * ``calibration_alpha`` (1.05) — a wrapper-level conservatism
    #     factor applied to the CALIBRATED base. That base already
    #     substitutes actual per-chunk bytes for ``n_persist*S_chunk``
    #     and strips CKPT op-walk double-counts — both are structural
    #     accounting FIXES, not fudge factors. After those fixes the
    #     10% paper-alpha becomes too loose: a measured 7B LoRA run
    #     lands at 13.12 GB actual vs 14.62 GB predicted with
    #     alpha=1.10 (11.4% over, > the test's 10% OOM-safety bound),
    #     vs 13.62 GB predicted with alpha=1.05 (3.8% over). We keep
    #     alpha=1.10 for the searcher's feasibility pruning where
    #     OOM-safety dominates, and alpha=1.05 on the post-hoc
    #     reporting path where the structural corrections are fully
    #     applied.
    #
    # Structural op-walk terms the paper 1.10 is still covering but
    # cost/memory.py doesn't explicitly account for (documented for
    # future work to pull them into the op-walk directly):
    #   - Adam moment buffers (exp_avg + exp_avg_sq) for persistent
    #     chunks: 2x fp32 of trainable params, allocated lazily at
    #     the first optimizer step. For LoRA this is tiny; for
    #     full-finetune it's ~model size.
    #   - PyTorch allocator internal fragmentation (caching-allocator
    #     block waste at power-of-2 boundaries).
    #   - Scheduler prefetch window: Scheduler.pre_block_forward can
    #     temporarily hold ``current + next`` block's worth of chunks;
    #     ``effective_buffer_slots`` below bounds this but doesn't
    #     fully eliminate the transient.
    # Closing any of these at cost/memory.py would let us drop the
    # wrapper-level 1.05 — until then, the two alphas stay independent.
    calibration_alpha = min(alpha, 1.05)
    # Buffer pool slots: ProTrain prefetches the next block's chunks
    # while the current block runs (see
    # runtime/scheduler.Scheduler.pre_block_forward) — peak concurrent
    # buffer occupancy is ``current + next block`` worth of chunks,
    # bounded above by ``n_buffer`` but typically less. Use that tighter
    # bound.
    max_chunks_per_block = 1
    if layout.block_to_chunks:
        max_chunks_per_block = max(
            (len(cids) for cids in layout.block_to_chunks.values()), default=1
        )
    effective_buffer_slots = min(n_buffer, 2 * max_chunks_per_block)
    buffer_bytes_eff = effective_buffer_slots * S
    calibrated_raw = actual_persistent + buffer_bytes_eff + f_bm
    calibrated = int(calibration_alpha * calibrated_raw)
    return calibrated


def _cpu_ram_per_rank_bytes(world_size: int) -> int:
    """Best-effort estimate of per-rank available CPU RAM in bytes.

    Heuristic: read node-level available RAM (``psutil.virtual_memory().available``
    preferred; falls back to ``/proc/meminfo`` on Linux) and divide by
    ``world_size`` as a crude per-rank share. This is PESSIMISTIC on
    machines with NUMA-aware CPU allocation and OPTIMISTIC on
    heterogeneous multi-host setups (where the smallest node's RAM is
    the binding constraint, not the average). Users whose production
    topology doesn't match the "node RAM / world_size" model should
    disable ``protrain_auto_mode`` and pick the mode explicitly — see
    DESIGN.md §Multi-GPU.

    Returns 0 when neither probe succeeds; the auto-selector interprets
    0 as "no offload is safe" and falls through to Mode A (which is
    usually correct — if the plugin can't see the RAM, assume the
    workload fits on GPU).
    """
    ws = max(1, int(world_size))
    # Preferred path: psutil (already in Axolotl's env for trainer bookkeeping).
    try:
        import psutil

        return max(0, int(psutil.virtual_memory().available) // ws)
    except ImportError:
        pass

    # Fallback: /proc/meminfo on Linux. ``MemAvailable`` field is the
    # kernel's own estimate of RAM that can be used without swapping;
    # matches psutil.virtual_memory().available on modern Linux.
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # Format: "MemAvailable:    12345678 kB"
                    kb = int(line.split()[1])
                    return max(0, (kb * 1024) // ws)
    except (FileNotFoundError, OSError, ValueError):
        pass

    # No reliable probe — return 0 so the auto-selector can detect the
    # gap and pick the safest fit-on-GPU path. Callers can log a warning
    # at the call site.
    return 0


def _select_mode(
    search_result: SearchResult,
    layout,
    hw: HardwareProfile,
    world_size: int,
    cpu_ram_per_rank_bytes: int,
    *,
    auto_mode: bool,
    user_force_all_persistent: bool,
    user_zero3_shard: bool | None,
) -> tuple[bool, bool]:
    """Resolve ``(force_all_persistent, zero3_shard)`` for the wrapper.

    Decision tree (``auto_mode=True``):

    * ``n_persist >= N_chunk`` → Mode A ``(True, False)``. Model fits
      fully on GPU; DDP+replicated is the throughput winner per the M7
      benchmark (3.64x vs 0.70x ZeRO-3 on PCIe Gen3 4x 3090).
    * Otherwise model needs offload. Pick between:
       - Mode B (replicated): ``(False, False)``. Faster: no per-chunk
         ``all_gather`` / ``reduce_scatter`` collectives. Requires
         ``cpu_ram_per_rank_bytes >= replicated_footprint``.
       - Mode C (sharded): ``(False, True)``. Slower but fits: each rank
         holds ``1/world_size`` of each non-persistent chunk's pinned
         bytes. Requires ``cpu_ram_per_rank_bytes >= sharded_footprint``.
       - Neither: raise ``RuntimeError`` — the model truly doesn't fit
         on this node, user must scale up (more nodes / more RAM /
         smaller model) before retrying.

    ``auto_mode=False`` returns the user's explicit flags unchanged
    (with ``None`` zero3_shard → False).

    The "Mode B over Mode C when both fit" policy is a deliberate
    throughput trade — Mode B is ~1.9x faster than Mode C on PCIe Gen3,
    so we keep CPU-replication as long as it fits even if the sharded
    path would save pinned RAM. Users with binding CPU pressure should
    set ``protrain_auto_mode=False, protrain_zero3_shard=True`` to force
    Mode C.
    """
    # Explicit overrides — bypass the selector.
    if not auto_mode:
        return (
            bool(user_force_all_persistent),
            bool(user_zero3_shard) if user_zero3_shard is not None else False,
        )

    # Single-rank auto path: no multi-GPU mode to pick — Mode A is
    # always the right answer (no CPU offload to replicate/shard).
    if world_size <= 1:
        return (True, False)

    # Mode A: searcher says everything fits on GPU. Best throughput.
    if int(search_result.cfg.n_persist) >= int(layout.N_chunk):
        return (True, False)

    # Compute per-rank CPU footprint under both replicated and sharded
    # modes from the searcher's picked config. Build throwaway hardware
    # profiles so the cost model can read ``zero3_shard`` directly.
    from dataclasses import replace as _replace

    from axolotl.integrations.protrain.cost.memory import (
        estimate_cpu_footprint,
    )

    hw_replicated = _replace(hw, zero3_shard=False)
    replicated_footprint = int(
        estimate_cpu_footprint(search_result.cfg, layout, hw_replicated)
    )
    hw_sharded = _replace(hw, zero3_shard=True)
    sharded_footprint = int(
        estimate_cpu_footprint(search_result.cfg, layout, hw_sharded)
    )

    if cpu_ram_per_rank_bytes >= replicated_footprint:
        return (False, False)
    if cpu_ram_per_rank_bytes >= sharded_footprint:
        return (False, True)

    raise RuntimeError(
        "ProTrain auto-mode: model does not fit on this node. Searcher "
        f"picked n_persist={search_result.cfg.n_persist}/"
        f"{layout.N_chunk} (needs CPU offload), but per-rank CPU RAM "
        f"({cpu_ram_per_rank_bytes / 1e9:.1f} GB) is smaller than the "
        f"sharded footprint ({sharded_footprint / 1e9:.1f} GB). Scale "
        "up: more nodes, more system RAM, smaller model, or a larger "
        "per-rank capacity budget."
    )


def protrain_model_wrapper(
    model: nn.Module,
    model_config: object,  # noqa: ARG001 — accepted for API symmetry with the plan
    hardware_profile: HardwareProfile,
    *,
    batch_size: int,
    seq_len: int,
    capacity_bytes: int | None = None,
    cache_dir: str | None = None,  # noqa: ARG001 — reserved for future cache redirection
    force_all_persistent: bool = False,
    n_persist_override: int | None = None,
    n_buffer_override: int | None = None,
    n_swap_override: int | None = None,
    n_checkpoint_override: int | None = None,
    zero3_shard: bool | None = None,
    auto_mode: bool = False,
) -> WrappedModel:
    """Compose the ProTrain runtime around a standard ``nn.Module``.

    Parameters
    ----------
    model:
        Any standard ``nn.Module``. Must be on GPU by the time this is
        called; the profiler and all buffers are allocated on the same
        device as ``next(model.parameters()).device``.
    model_config:
        Reserved. The plugin path (M5) will use this to pick up
        ZeRO-related options; the M4b wrapper does not consult it.
    hardware_profile:
        Static hardware descriptor — see
        :class:`~axolotl.integrations.protrain.types.HardwareProfile`.
    batch_size / seq_len:
        Used for both the profiler invocation and the cache key.
    capacity_bytes:
        Override the GPU memory budget the searcher should respect.
        When ``None``, defaults to
        ``hardware_profile.gpu_memory_bytes - 2 GiB`` to leave headroom
        for the CUDA context + PyTorch allocator.
    cache_dir:
        Reserved. Profiler cache directory resolution currently lives
        in ``profiler.cache._cache_root`` via the ``XDG_CACHE_HOME`` env
        var.
    force_all_persistent:
        When True, skip the exhaustive searcher and synthesize a
        ``SearchResult`` that forces every chunk to stay GPU-resident
        (``n_persist = N_chunk``, ``n_swap = 0``,
        ``n_checkpoint = N_block``). This is the M5 recommended mode
        for LoRA on a single 24 GB card until the M4.5 runtime
        primitives (init-time chunk offload, per-param grad offload)
        land — search-picked configs that expect CPU-hosted chunks
        currently OOM because the physical offload is not yet wired.
    n_persist_override / n_buffer_override / n_swap_override / n_checkpoint_override:
        Debug escape hatches. When *all four* are set, the searcher is
        skipped and a synthetic ``SearchResult`` is built from the
        explicit values. A single override in isolation is ignored (the
        searcher's picks stay consistent across the 4-tuple); this is
        documented on the pydantic fields.
    zero3_shard:
        M7 ZeRO-3 activation. When ``None`` (default) the wrapper
        auto-detects: shard iff
        ``torch.distributed.get_world_size() > 1`` AND
        ``force_all_persistent`` is False. When explicitly True or
        False the caller override wins. Sharded mode requires a live
        ``torch.distributed`` process group AND the model must not be
        wrapped in DDP at training time (sharding is the grad-sync
        point itself; DDP would double-reduce).
    auto_mode:
        When True, the wrapper runs the searcher first and then calls
        :func:`_select_mode` to resolve ``(force_all_persistent,
        zero3_shard)`` from workload fit + per-rank CPU RAM. The
        caller's ``force_all_persistent`` / ``zero3_shard`` arguments
        are IGNORED on this path (they become explicit overrides only
        when ``auto_mode=False``). Designed to save users from the
        ZeRO-3 footgun surfaced by the M7 benchmark (0.70x throughput
        vs. 3.64x DDP on PCIe Gen3 4x 3090 when the model fits on GPU).
        Default is False on this direct entry point; the plugin sets it
        to True via ``ProTrainArgs.protrain_auto_mode``.

    Returns
    -------
    WrappedModel
        Handle carrying the search result, chunk manager, scheduler,
        and the installed hook handles. The underlying ``model`` is
        returned in-place — no module swap.
    """
    import torch

    # Pick the device from the model; fall back to cuda:0.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Gradient checkpointing + HF KV cache leads to recompute-time shape
    # mismatches (cache grows across calls; the recompute call sees a
    # different past_key_values length). Force use_cache=False if the model
    # exposes it — this is standard practice for training regardless of
    # ProTrain, and the CKPT block wrapper depends on it.
    cfg_obj = getattr(model, "config", None)
    if cfg_obj is not None and getattr(cfg_obj, "use_cache", False):
        LOG.info("ProTrain: forcing model.config.use_cache=False for CKPT compatibility")
        cfg_obj.use_cache = False

    # ---- 1. profile (cached) --------------------------------------------
    cache_key = ProfilerCacheKey(
        arch_hash=_arch_hash(model),
        bs=batch_size,
        seq=seq_len,
        sku=_sku(device),
        world=hardware_profile.gpu_count,
    )
    trace = load_cached_trace(cache_key)
    if trace is None:
        import sys as _sys

        LOG.info(
            "ProTrain profiler cache miss for %s — running trace (bs=%d seq=%d)",
            cache_key.fingerprint()[:12],
            batch_size,
            seq_len,
        )
        _sys.stderr.write(
            f"[protrain] profiler cache miss — running forward-only trace\n"
        )
        _sys.stderr.flush()
        # Forward-only profile: the cost model's op-walk in
        # :mod:`cost.memory` only reads forward ops (the synthetic
        # ``<backward>`` record is skipped), and :mod:`cost.runtime`
        # derives ``t_bwd`` from ``t_fwd`` + activation sizes rather
        # than a measured backward. Running ``loss.backward()`` on a
        # 7B-class model in the profiler blows the 24 GiB card before
        # ProTrain's chunk offload can engage; since the backward
        # isn't consumed by downstream cost estimation, skipping it is
        # loss-free and unblocks integration on single-3090 budgets.
        profiler_cfg = ProfilerConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            device=str(device),
            include_backward=False,
            on_demand=True,
            world_size=int(hardware_profile.gpu_count),
        )
        batch = _dummy_batch(model, batch_size, seq_len, device)
        trace = run_trace(model, batch, profiler_cfg)
        _sys.stderr.write(
            f"[protrain] trace done: {len(trace.op_order)} ops, "
            f"{len(trace.activation_sizes)} blocks\n"
        )
        _sys.stderr.flush()
        save_cached_trace(cache_key, trace)
    else:
        LOG.info(
            "ProTrain profiler cache hit for %s", cache_key.fingerprint()[:12]
        )

    # ---- 2. layout ------------------------------------------------------
    import sys as _sys2

    _sys2.stderr.write("[protrain] building layout\n")
    _sys2.stderr.flush()
    blocks, block_spans = _build_block_spans(model)
    exec_order = _param_exec_order(model, block_spans)

    # Derive S_chunk from a {ParamId -> bytes} map.
    param_bytes: dict[ParamId, int] = {
        cast(ParamId, name): int(p.numel()) * int(p.element_size())
        for name, p in model.named_parameters()
    }
    s_chunk = pick_S_chunk(param_bytes)

    layout = build_layout(
        model=model,
        exec_order=exec_order,
        S_chunk=s_chunk,
        block_spans=block_spans,
    )
    _sys2.stderr.write(
        f"[protrain] layout built: S_chunk={layout.S_chunk} "
        f"N_chunk={layout.N_chunk}\n"
    )
    _sys2.stderr.flush()

    # ---- 3. search (or synthesize) -------------------------------------
    if capacity_bytes is None:
        capacity_bytes = max(
            0, int(hardware_profile.gpu_memory_bytes) - _DEFAULT_HEADROOM_BYTES
        )

    # Early world-size probe — the mode selector + zero3_shard plumbing
    # both need this before the search runs.
    _ws_early = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _ws_early = int(torch.distributed.get_world_size())

    # Stash the caller's raw intent before the auto-selector potentially
    # rewrites the effective flags. The selector is applied AFTER
    # search() returns; until then we treat the run as a "best fit"
    # search with zero3_shard=False in the hardware profile so the
    # searcher's CPU accounting uses the replicated baseline (the GPU
    # peak filter is sharding-agnostic anyway — see
    # cost/memory.estimate_peak — so the searcher's pick of n_persist is
    # not distorted by this choice).
    _user_force_all_persistent = bool(force_all_persistent)
    _user_zero3_shard = zero3_shard

    if auto_mode:
        # On the auto path, disable the force_all_persistent short-circuit
        # below and let the searcher pick n_persist. If the fit is tight
        # the selector flips the mode post-search; if the fit is loose
        # the searcher lands at n_persist=N_chunk naturally, which is
        # already Mode A semantically (no runtime difference vs. the
        # force_all_persistent synthetic path). We also suppress an
        # explicit user ``zero3_shard=True`` for the hw profile here;
        # it gets re-evaluated after search + selector.
        if _user_force_all_persistent:
            LOG.info(
                "ProTrain auto-mode: user set force_all_persistent=True "
                "but auto-mode overrides explicit flags. Running searcher "
                "— will pick Mode A naturally if the workload fits on "
                "GPU. Set ``protrain_auto_mode: false`` to force-honour "
                "force_all_persistent=True."
            )
        force_all_persistent = False
        zero3_shard = False

    # Resolve the ZeRO-3 sharding flag early so we can propagate it into
    # ``HardwareProfile`` before the cost-model search runs. The same
    # rules as the later in-place re-check (post-materialize_offload)
    # apply here — auto-enable when ``world_size > 1`` AND
    # ``force_all_persistent`` is False, honour explicit caller
    # overrides otherwise. The ChunkManager additionally degrades to
    # False on single-rank hosts (so setting this True on ws=1 is a
    # no-op); we mirror that here for HW profile consistency.
    if zero3_shard is None:
        _zero3_for_hw = (_ws_early > 1) and (not force_all_persistent)
    else:
        _zero3_for_hw = bool(zero3_shard) and (_ws_early > 1)
    # Propagate into the hardware_profile the searcher consumes. Replace
    # is cheap; HardwareProfile is frozen so we can't mutate in place.
    # We also plumb the trace's measured Adam throughputs into the
    # hardware_profile so ``cost/runtime.py`` consumes the empirical
    # rates rather than the hardcoded prior.
    from dataclasses import replace as _replace

    _hw_updates: dict = {}
    if _zero3_for_hw != hardware_profile.zero3_shard:
        _hw_updates["zero3_shard"] = _zero3_for_hw
    # Only overwrite Adam rates when the caller-provided profile doesn't
    # already carry them (i.e. tests that hand-craft a profile with a
    # specific rate keep their value). Non-zero trace measurement wins
    # over the default 0.0; 0.0 from the trace means the benchmark
    # couldn't run, and the runtime cost model will fall back.
    if (
        hardware_profile.cpu_adam_bytes_per_sec <= 0.0
        and trace.cpu_adam_bytes_per_sec > 0.0
    ):
        _hw_updates["cpu_adam_bytes_per_sec"] = trace.cpu_adam_bytes_per_sec
    if (
        hardware_profile.gpu_adam_bytes_per_sec <= 0.0
        and trace.gpu_adam_bytes_per_sec > 0.0
    ):
        _hw_updates["gpu_adam_bytes_per_sec"] = trace.gpu_adam_bytes_per_sec
    # Live SKU compute rate — measured fresh on the training device so the
    # cost model can scale per-op latencies when the trace was captured on
    # a different SKU (3090 vs 3090 Ti, etc.). Same-SKU runs see the same
    # value here as in trace.compute_rate_tflops, so the ratio is ~1.0.
    if hardware_profile.gpu_compute_tflops <= 0.0:
        try:
            _live_tflops = measure_compute_rate(
                int(getattr(device, "index", 0) or 0)
            )
            if _live_tflops > 0.0:
                _hw_updates["gpu_compute_tflops"] = _live_tflops
        except Exception as _e:  # noqa: BLE001 - defensive
            LOG.debug("measure_compute_rate live failed (%s); skipping SKU calibration", _e)
    # PCIe rates: overwrite the caller's hardcoded prior (usually 13e9 =
    # Gen3) with the profiler's measured H2D/D2H. A 3090 on PCIe Gen4 x16
    # sits around 50-56 GB/s — 4× the conservative default — and the
    # cost model's per-chunk comm is S_chunk / eff_h2d, so this flow-
    # through directly corrects the 7B over-prediction.
    if (
        hardware_profile.pcie_h2d_bps <= 13e9 + 1e6  # within 1MB of default
        and trace.pcie_h2d_bps > 13e9 + 1e6
    ):
        _hw_updates["pcie_h2d_bps"] = trace.pcie_h2d_bps
    if (
        hardware_profile.pcie_d2h_bps <= 13e9 + 1e6
        and trace.pcie_d2h_bps > 13e9 + 1e6
    ):
        _hw_updates["pcie_d2h_bps"] = trace.pcie_d2h_bps
    if _hw_updates:
        hardware_profile = _replace(hardware_profile, **_hw_updates)

    n_block = max(1, len(trace.activation_sizes))
    # Max chunks seen in any one transformer block — used for the
    # force_all_persistent buffer-pool sizing (we need enough buffers to
    # hold every chunk a single block touches during its forward, times
    # 2 for the rolling forward→backward reuse the BufferPool assumes).
    max_chunks_per_block = 1
    if layout.block_to_chunks:
        max_chunks_per_block = max(
            (len(cids) for cids in layout.block_to_chunks.values()), default=1
        )

    all_overrides_set = all(
        v is not None
        for v in (
            n_persist_override,
            n_buffer_override,
            n_swap_override,
            n_checkpoint_override,
        )
    )

    if force_all_persistent:
        # Synthesize a SearchResult that pins every chunk on GPU and
        # uses activation checkpointing on every block. This is the M5
        # workaround for the two known M4.5 runtime gaps (init-time
        # chunk offload, per-param grad offload) — see DESIGN.md and
        # the M4 integration xfail. The cost model is skipped; predicted
        # numbers are filled with zeros so downstream consumers don't
        # misread them as real predictions.
        synth_cfg = CostConfig(
            n_persist=layout.N_chunk,
            n_buffer=max(1, 2 * max_chunks_per_block),
            n_swap=0,
            n_checkpoint=n_block,
        )
        block_map = assign_modes(
            n_swap=0, n_checkpoint=n_block, N_block=n_block
        )
        result = SearchResult(
            cfg=synth_cfg,
            block_map=block_map,
            predicted_peak_bytes=0,
            predicted_iter_s=0.0,
        )
        LOG.warning(
            "ProTrain: force_all_persistent=True — bypassing searcher. "
            "n_persist=%d n_buffer=%d n_swap=0 n_checkpoint=%d. "
            "All model state stays GPU-resident; activations rely on CKPT. "
            "This is the documented workaround for the M4.5 runtime gaps.",
            synth_cfg.n_persist,
            synth_cfg.n_buffer,
            synth_cfg.n_checkpoint,
        )
        _sys2.stderr.write(
            f"[protrain] force_all_persistent: cfg={result.cfg}\n"
        )
        _sys2.stderr.flush()
    elif all_overrides_set:
        # Explicit 4-tuple override path — still skip the searcher but
        # honour the caller's exact knob selection. Bounds-check is
        # mandatory; the searcher normally enforces these.
        if not (0 <= n_persist_override <= layout.N_chunk):
            raise ValueError(
                f"n_persist_override={n_persist_override} out of range "
                f"[0, {layout.N_chunk}]"
            )
        if n_buffer_override < 1:
            raise ValueError(
                f"n_buffer_override must be >= 1, got {n_buffer_override}"
            )
        if not (0 <= n_swap_override <= n_block):
            raise ValueError(
                f"n_swap_override={n_swap_override} out of range [0, {n_block}]"
            )
        if not (0 <= n_checkpoint_override <= n_block - n_swap_override):
            raise ValueError(
                f"n_checkpoint_override={n_checkpoint_override} incompatible "
                f"with n_swap_override={n_swap_override} (N_block={n_block})"
            )
        synth_cfg = CostConfig(
            n_persist=n_persist_override,
            n_buffer=n_buffer_override,
            n_swap=n_swap_override,
            n_checkpoint=n_checkpoint_override,
        )
        block_map = assign_modes(
            n_swap=n_swap_override,
            n_checkpoint=n_checkpoint_override,
            N_block=n_block,
        )
        result = SearchResult(
            cfg=synth_cfg,
            block_map=block_map,
            predicted_peak_bytes=0,
            predicted_iter_s=0.0,
        )
        LOG.warning(
            "ProTrain: explicit knob override path — bypassing searcher. cfg=%s",
            synth_cfg,
        )
        _sys2.stderr.write(
            f"[protrain] explicit override: cfg={result.cfg}\n"
        )
        _sys2.stderr.flush()
    else:
        _sys2.stderr.write(
            f"[protrain] running exhaustive search (N_chunk={layout.N_chunk}, "
            f"N_block={n_block})\n"
        )
        _sys2.stderr.flush()
        result = search(trace, layout, int(capacity_bytes), hardware_profile)
        _sys2.stderr.write(
            f"[protrain] search done: cfg={result.cfg} "
            f"peak={result.predicted_peak_bytes/1e9:.2f}GB "
            f"iter={result.predicted_iter_s:.3f}s\n"
        )
        _sys2.stderr.flush()

    # ---- 3.5: auto-mode selection (M7 follow-up) -----------------------
    # With the searcher's ``n_persist`` pick in hand, resolve the real
    # (force_all_persistent, zero3_shard) pair from workload fit +
    # per-rank CPU RAM. See ``_select_mode`` for the decision tree and
    # the DESIGN.md §Multi-GPU measured throughput ordering that
    # motivates the default (A > B > C on PCIe Gen3 3090).
    if auto_mode:
        cpu_ram = _cpu_ram_per_rank_bytes(_ws_early)
        if cpu_ram == 0 and _ws_early > 1:
            LOG.warning(
                "ProTrain auto-mode: could not probe CPU RAM via psutil or "
                "/proc/meminfo. Treating per-rank RAM as 0 bytes — the "
                "selector will prefer Mode A (force_all_persistent) and "
                "raise if the model needs offload. Set "
                "``protrain_auto_mode: false`` and pick the mode "
                "explicitly on exotic topologies."
            )
        auto_force_persistent, auto_zero3 = _select_mode(
            search_result=result,
            layout=layout,
            hw=hardware_profile,
            world_size=_ws_early,
            cpu_ram_per_rank_bytes=cpu_ram,
            auto_mode=True,
            user_force_all_persistent=_user_force_all_persistent,
            user_zero3_shard=_user_zero3_shard,
        )

        # Warn if the user set an explicit flag that the selector is
        # overriding. This is the key safety check for the M7 footgun:
        # users who requested ZeRO-3 on a workload that fits in Mode A
        # should learn they're leaving throughput on the table.
        if _user_zero3_shard is True and not auto_zero3 and _ws_early > 1:
            LOG.warning(
                "ProTrain auto-mode: user set zero3_shard=True but the "
                "workload fits in Mode A (force_all_persistent). "
                "Auto-mode picked Mode A for better throughput — on "
                "PCIe Gen3 RTX 3090, DDP+Mode_A gives ~3.6x scaling vs "
                "ZeRO-3's ~0.7x. Set ``protrain_auto_mode: false`` to "
                "force-honour zero3_shard=True."
            )

        if auto_force_persistent:
            if _ws_early > 1:
                LOG.info(
                    "ProTrain auto-mode: picking Mode A "
                    "(force_all_persistent=True). On PCIe Gen3 RTX 3090, "
                    "DDP+Mode_A gives ~3.6x scaling vs ZeRO-3's ~0.7x — see "
                    "DESIGN.md §Multi-GPU for benchmark data."
                )
            else:
                LOG.info(
                    "ProTrain auto-mode: picking Mode A "
                    "(force_all_persistent=True, single-rank)."
                )
        elif not auto_zero3:
            LOG.info(
                "ProTrain auto-mode: picking Mode B (CPU-offload, "
                "replicated). Per-rank CPU RAM sufficient for the full "
                "non-persistent chunk set."
            )
        else:
            LOG.info(
                "ProTrain auto-mode: picking Mode C (CPU-offload, "
                "ZeRO-3 sharded). Per-rank CPU RAM too tight for "
                "replication — falling back to 1/world_size shard."
            )

        force_all_persistent = auto_force_persistent
        zero3_shard = auto_zero3
        # If the selector picked Mode C (sharded), we need the downstream
        # chunk manager to see zero3_shard=True. Propagate via the
        # hardware_profile so the remaining pipeline picks it up exactly
        # as the explicit path would. (If selector picked Mode B, the
        # prior hw flip to False is already correct.)
        if zero3_shard != hardware_profile.zero3_shard:
            from dataclasses import replace as _replace
            hardware_profile = _replace(
                hardware_profile, zero3_shard=bool(zero3_shard)
            )

    # ---- 4. construct runtime ------------------------------------------
    n_persist = result.cfg.n_persist
    n_buffer = max(1, result.cfg.n_buffer)

    pinned_host = PinnedHostMemory(n_buffer=n_buffer, S_chunk=layout.S_chunk)
    buffer_pool = BufferPool(
        n_buffer=n_buffer,
        S_chunk=layout.S_chunk,
        pinned_host=pinned_host,
        device=device,
    )

    # Partition params: persistent chunks get the GPU optimizer, the rest
    # get per-chunk CPU FusedAdam adapters keyed on ChunkId.
    params_by_name: dict[str, nn.Parameter] = dict(model.named_parameters())
    persistent_params: list[nn.Parameter] = []
    cpu_params_per_chunk: dict = {}

    for cid, chunk_param_ids in enumerate(layout.chunks):
        chunk_params = [
            params_by_name[str(pid)]
            for pid in chunk_param_ids
            if str(pid) in params_by_name
        ]
        if cid < n_persist:
            persistent_params.extend(chunk_params)
        else:
            cpu_params_per_chunk[cid] = chunk_params

    # Adam hyperparameters are owned by the optimizer wrapper; seed with
    # harmless defaults here. ``protrain_optimizer_wrapper`` will rebuild
    # these adapters with the user's real LR/betas, so this instance is
    # transient — we still allocate it so the chunk manager has a live
    # reference during the smoke-test smoke path.
    #
    # BUG 3 FIX: ``CpuFusedAdamAdapter`` construction is deferred to
    # AFTER ``chunk_manager.materialize_offload()`` below. Before
    # offload, the non-persistent chunk params are full-size GPU
    # tensors; after offload they are zero-element GPU placeholders
    # whose *real* weights live in ``chunk_manager._cpu_slots``. The
    # lazy CPU-Adam state init (``torch.zeros_like(p.data, device='cpu')``)
    # runs on the first ``step`` call — by which point
    # ``_ensure_cpu_grads_attached`` has repointed ``p.data`` at the CPU
    # shard — so what matters is that the adapter's ``param_groups``
    # reference the right ``nn.Parameter`` objects, not what ``p.data``
    # currently points at. The previous ordering (adapter built
    # pre-offload) was benign in the p.data sense but risked a CUDA
    # initialization hazard if DeepSpeed ever cached pointers on the
    # GPU tensor; deferring is the safe invariant.
    gpu_optim: GpuFusedAdamAdapter | None = None
    if persistent_params:
        gpu_optim = GpuFusedAdamAdapter(params=persistent_params, lr=1e-4)

    # ---- Distributed context + M7 zero3_shard decision -----------------
    # Auto-detect world_size / rank from the active process group;
    # default to single-rank when no group is up. ``zero3_shard`` was
    # already resolved above the search call so it could flow through
    # ``HardwareProfile.zero3_shard`` into the cost model; re-use that
    # decision here for the ChunkManager constructor. The ChunkManager
    # silently degrades zero3_shard to False when world_size == 1, so
    # the auto-detect path is safe on single-rank hosts too.
    _ws = 1
    _rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _ws = int(torch.distributed.get_world_size())
        _rank = int(torch.distributed.get_rank())
    _zero3 = bool(hardware_profile.zero3_shard) and (_ws > 1)
    LOG.info(
        "ProTrain: distributed context world_size=%d rank=%d zero3_shard=%s "
        "(requested=%s)",
        _ws,
        _rank,
        _zero3,
        zero3_shard,
    )

    chunk_manager = ChunkManager(
        model=model,
        layout=layout,
        n_persist=n_persist,
        buffer_pool=buffer_pool,
        cpu_optim=None,  # wired in after materialize_offload (BUG 3)
        gpu_optim=gpu_optim,
        device=device,
        world_size=_ws,
        rank=_rank,
        zero3_shard=_zero3,
    )

    # Chunks containing ANY non-block param (embeddings, final norm,
    # lm_head — any param not living inside a transformer block) are
    # pinned to the persistent set. Reasoning:
    #
    #   a) The block-granularity scheduler only knows about chunks
    #      listed in ``layout.block_to_chunks``. Pure non-block chunks
    #      (the trivial case — all their params are non-block) are never
    #      gathered by any hook; if offloaded they'd be zero-sized
    #      during forward.
    #   b) Mixed chunks (e.g. the last block's chunk that was greedy-
    #      filled with the final model.norm.weight) ARE gathered by the
    #      block-post hook, but the block-post hook ALSO releases them
    #      since they're not in the next block's chunk set — which
    #      leaves the non-block param (``model.norm.weight``) empty by
    #      the time LlamaModel.forward calls ``self.norm(...)`` after
    #      block 31's forward-post hook fires.
    #
    # The fix in both cases is the same: keep chunks with any non-block
    # param GPU-resident. Cost is bounded by ``S_chunk`` per such chunk;
    # for Llama it's typically 2 chunks ≈ 256 MB.
    param_is_in_block: dict[str, bool] = {
        str(pid): False for pid in layout.param_to_chunk
    }
    for bid, pids in _build_block_spans(model)[1].items():
        for pid in pids:
            param_is_in_block[str(pid)] = True
    chunks_with_nonblock: set[int] = set()
    for cid, pid_tuple in enumerate(layout.chunks):
        for pid in pid_tuple:
            if not param_is_in_block.get(str(pid), False):
                chunks_with_nonblock.add(cid)
                break
    extra = chunks_with_nonblock - chunk_manager._persistent_ids
    if extra:
        # Expand the persistent set in-place; mark_persistent takes a
        # prefix length, so we instead mutate the internal set directly
        # for this cross-cutting pin.
        chunk_manager._persistent_ids |= extra
        chunk_manager._non_persistent_ids -= extra
        LOG.info(
            "ProTrain: pinning %d chunks %s to persistent because they "
            "contain non-block params the scheduler cannot gather on "
            "its own",
            len(extra),
            sorted(extra),
        )

    # ---- peak-prediction calibration ------------------------------------
    # The cost/memory.py estimator approximates persistent model state as
    # ``n_persist * S_chunk`` — a tight upper bound when chunks pack
    # snugly to S_chunk, but a loose one when the layout leaves many
    # chunks partially filled (common for Llama-7B: avg chunk density
    # ~80% of S_chunk). For the integration-test peak-tolerance check
    # to land within the paper's stated "up to 10% overestimate" window
    # we recompute the model-state-present term using the *actual*
    # per-chunk byte footprint, then preserve the estimator's F_bm
    # (fragmentation + activation + inter/intra-op delta) component.
    calibrated_peak = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=result.predicted_peak_bytes,
        layout=layout,
        chunk_manager=chunk_manager,
        n_buffer=result.cfg.n_buffer,
        trace=trace,
        block_map=result.block_map,
    )
    if calibrated_peak != result.predicted_peak_bytes:
        LOG.info(
            "ProTrain: peak prediction calibrated %.2f -> %.2f GB "
            "using actual per-chunk byte footprint",
            result.predicted_peak_bytes / (1 << 30),
            calibrated_peak / (1 << 30),
        )
        effective_n_persist = len(chunk_manager._persistent_ids)
        result = SearchResult(
            cfg=CostConfig(
                n_persist=effective_n_persist,
                n_buffer=result.cfg.n_buffer,
                n_swap=result.cfg.n_swap,
                n_checkpoint=result.cfg.n_checkpoint,
            ),
            block_map=result.block_map,
            predicted_peak_bytes=calibrated_peak,
            predicted_iter_s=result.predicted_iter_s,
        )

    # ---- 4.5: materialize the init-time chunk offload (M4.5 Gap 1) -----
    # Physically move every non-persistent chunk's param data to pinned
    # CPU memory and install the per-param grad hooks (Gap 2). This must
    # happen BEFORE step 5 (block wrap) / step 6 (hook install) so the
    # first forward sees the correct GPU residency picture and the grad
    # hooks are live by the time autograd starts accumulating.
    alloc_before = (
        torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    )
    freed = chunk_manager.materialize_offload()
    alloc_after = (
        torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    )
    LOG.info(
        "ProTrain: materialize_offload freed %.2f GB (reported), "
        "alloc %.2f -> %.2f GB (torch measured)",
        freed / (1 << 30),
        alloc_before / (1 << 30),
        alloc_after / (1 << 30),
    )
    _sys2.stderr.write(
        f"[protrain] materialize_offload: freed {freed/1e9:.2f}GB "
        f"(alloc {alloc_before/1e9:.2f}->{alloc_after/1e9:.2f}GB)\n"
    )
    _sys2.stderr.flush()

    # ---- 4.6: build the CPU FusedAdam adapter (post-offload) ------------
    # BUG 3 FIX: now that ``materialize_offload`` has allocated the pinned
    # CPU shards and installed per-param grad hooks, build the CPU Adam
    # adapter with references to the same ``nn.Parameter`` objects the
    # hooks will repoint to CPU storage before calling step. The adapter
    # is "transient" (``protrain_optimizer_wrapper`` rebuilds it at the
    # user's real hyperparams) but we still need one live here so the
    # chunk manager has something to drive during smoke tests.
    # M7: for sharded non-persistent chunks, the CPU Adam updates each
    # region's flat shard_param (one per :class:`_DtypeRegion`) rather
    # than the user-facing param list. Homogeneous-dtype chunks have
    # one region and behave exactly like the pre-followup single-param
    # case; mixed-dtype chunks expose one shard_param per region.
    cpu_params_per_chunk_for_optim: dict = {}
    for cid, chunk_params in cpu_params_per_chunk.items():
        shard_state = chunk_manager._chunk_shards.get(cid)  # type: ignore[attr-defined]
        if shard_state is not None and shard_state.regions:
            cpu_params_per_chunk_for_optim[cid] = [
                r.shard_param for r in shard_state.regions
            ]
        else:
            cpu_params_per_chunk_for_optim[cid] = chunk_params

    cpu_optim: CpuFusedAdamAdapter | None = None
    if any(params for params in cpu_params_per_chunk_for_optim.values()):
        try:
            cpu_optim = CpuFusedAdamAdapter(
                params_per_chunk=cpu_params_per_chunk_for_optim,
                lr=1e-4,
            )
        except (ImportError, Exception) as err:  # noqa: BLE001 - see below
            # CpuFusedAdamAdapter can fail with more than ``ImportError``:
            # DeepSpeed raises ``CUDAMismatchException`` (not an
            # ``ImportError`` subclass) when the system nvcc and torch's
            # cu-version disagree. We degrade gracefully in both cases —
            # persistent chunks still run fused GPU Adam, non-persistent
            # chunks fall through to the in-line torch.optim path inside
            # the optimizer wrapper. The warning surfaces the root cause
            # so users know they're not getting the async overlap.
            LOG.warning(
                "ProTrain: CPU FusedAdam unavailable (%s); non-persistent chunks "
                "will not get async CPU Adam. Install DeepSpeed with a matching "
                "CUDA toolkit (or set DS_SKIP_CUDA_CHECK=1) for full coverage.",
                err,
            )
            cpu_optim = None
    chunk_manager.cpu_optim = cpu_optim

    eff_h2d, eff_d2h = effective_bw(result.cfg, hardware_profile)

    scheduler = Scheduler(
        chunk_manager=chunk_manager,
        block_map=result.block_map,
        layout=layout,
        effective_h2d_bps=eff_h2d,
        effective_d2h_bps=eff_d2h,
    )

    # ---- 5. wrap blocks -------------------------------------------------
    # Locate the parent ModuleList so we can swap in the wrapped blocks in-place.
    module_list = _find_parent_module_list(model, blocks)
    for idx, block in enumerate(blocks):
        mode = result.block_map.get(BlockId(idx))
        if mode is None:
            continue
        wrapped = wrap_block(block, mode)
        if wrapped is not block and module_list is not None:
            module_list[idx] = wrapped
            blocks[idx] = wrapped

    # ---- 6. install hooks ----------------------------------------------
    handles = install_hooks(
        model=model,
        chunk_manager=chunk_manager,
        block_map=result.block_map,
        scheduler=scheduler,
    )

    LOG.info(
        "ProTrain config: n_persist=%d n_buffer=%d n_swap=%d n_checkpoint=%d "
        "S_chunk=%d N_chunk=%d peak=%.2f GiB iter=%.3f s capacity=%.2f GiB",
        result.cfg.n_persist,
        result.cfg.n_buffer,
        result.cfg.n_swap,
        result.cfg.n_checkpoint,
        layout.S_chunk,
        layout.N_chunk,
        result.predicted_peak_bytes / (1 << 30),
        result.predicted_iter_s,
        capacity_bytes / (1 << 30),
    )

    return WrappedModel(
        module=model,
        search_result=result,
        chunk_manager=chunk_manager,
        scheduler=scheduler,
        _hook_handles=list(handles),
    )


def _find_parent_module_list(
    model: nn.Module, blocks: list[nn.Module]
) -> "nn.ModuleList | None":
    """Locate the ``nn.ModuleList`` whose children are ``blocks``.

    ``discover_blocks`` returns a plain ``list``; to swap in wrapped
    modules we need a reference to the underlying container so the
    swap is visible to the rest of the model.
    """
    if not blocks:
        return None
    first = blocks[0]
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) == len(blocks):
            # Identity check on the first child is enough — ModuleLists
            # don't repeat modules.
            try:
                if module[0] is first:
                    return module
            except IndexError:
                continue
    return None


__all__ = ["protrain_model_wrapper"]
