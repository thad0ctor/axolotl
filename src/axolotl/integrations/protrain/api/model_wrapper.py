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

import hashlib
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
from axolotl.integrations.protrain.runtime.hooks import install_hooks
from axolotl.integrations.protrain.runtime.scheduler import Scheduler
from axolotl.integrations.protrain.search import search
from axolotl.integrations.protrain.types import (
    BlockId,
    HardwareProfile,
    ParamId,
    ProfilerConfig,
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


def _arch_hash(model: nn.Module) -> str:
    """Deterministic hash of the model architecture for the cache key.

    Mirrors the profiler's internal hash so the cache key is stable
    across processes that only see the module (no trace) — the plugin
    (M5) will call this before invoking the profiler.
    """
    parts: list[str] = [type(model).__name__]
    for name, p in model.named_parameters():
        parts.append(f"{name}:{tuple(p.shape)}:{p.dtype}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


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


def protrain_model_wrapper(
    model: nn.Module,
    model_config: object,  # noqa: ARG001 — accepted for API symmetry with the plan
    hardware_profile: HardwareProfile,
    *,
    batch_size: int,
    seq_len: int,
    capacity_bytes: int | None = None,
    cache_dir: str | None = None,  # noqa: ARG001 — reserved for future cache redirection
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

    # ---- 3. search ------------------------------------------------------
    if capacity_bytes is None:
        capacity_bytes = max(
            0, int(hardware_profile.gpu_memory_bytes) - _DEFAULT_HEADROOM_BYTES
        )
    _sys2.stderr.write(
        f"[protrain] running exhaustive search (N_chunk={layout.N_chunk}, "
        f"N_block={len(trace.activation_sizes)})\n"
    )
    _sys2.stderr.flush()
    result = search(trace, layout, int(capacity_bytes), hardware_profile)
    _sys2.stderr.write(
        f"[protrain] search done: cfg={result.cfg} "
        f"peak={result.predicted_peak_bytes/1e9:.2f}GB "
        f"iter={result.predicted_iter_s:.3f}s\n"
    )
    _sys2.stderr.flush()

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
    gpu_optim: GpuFusedAdamAdapter | None = None
    cpu_optim: CpuFusedAdamAdapter | None = None
    if persistent_params:
        gpu_optim = GpuFusedAdamAdapter(params=persistent_params, lr=1e-4)
    if any(params for params in cpu_params_per_chunk.values()):
        try:
            cpu_optim = CpuFusedAdamAdapter(
                params_per_chunk=cpu_params_per_chunk,
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

    chunk_manager = ChunkManager(
        model=model,
        layout=layout,
        n_persist=n_persist,
        buffer_pool=buffer_pool,
        cpu_optim=cpu_optim,
        gpu_optim=gpu_optim,
    )

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
