"""Single-iteration forward/backward trace driver for the ProTrain profiler.

Walks every ``nn.Module`` leaf with pre/post forward hooks, attaches a
tensor-level backward hook to the loss output, and records the intra/inter-op
memory deltas that ``torch.profiler`` misses (§3.2, App A.2).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from axolotl.utils.logging import get_logger

from axolotl.integrations.protrain.types import (
    BlockId,
    OpId,
    OpRecord,
    ProfilerConfig,
    ProfilerTrace,
)

from axolotl.integrations.protrain.profiler.hw_bench import (
    measure_compute_rate,
    measure_cpu_adam,
    measure_gpu_adam,
    measure_nccl,
    measure_pcie,
)
from axolotl.integrations.protrain.profiler.memory_deltas import (
    MemoryDeltaTracker,
    inter_op_delta,
    intra_op_delta,
)
from axolotl.integrations.protrain.profiler.on_demand import OnDemandTensorMgr

if TYPE_CHECKING:
    import torch
    from torch import nn

LOG = get_logger(__name__)


# Bytes per fp32 master + two Adam momentums. Assumes mixed-precision Adam
# (the training regime ProTrain targets): fp16 params+grads are 2+2 B/param,
# fp32 master is 4 B, m and v are 4 B each => 16 B additional per param.
# Callers can override via ``ProfilerConfig`` extensions or by patching
# ``optim_state_bytes_per_param`` below (kept as a module-level knob so M4
# can plug in a real ZeRO-3 sharding calculation without reshaping the API).
DEFAULT_OPTIM_STATE_BYTES_PER_PARAM = 16
DEFAULT_PARAM_GRAD_BYTES_PER_PARAM = 4  # fp16 param + fp16 grad

# Fraction of total GPU memory above which the profiler auto-engages
# on-demand mode (param offload + saved-for-backward CPU spill). At 60%, a
# 24 GB card auto-engages once params exceed ~14.4 GB — i.e. 13B-class fp16
# models and up. Below the threshold the profiler stays on the fast path
# so the cost model's calibration (captured against fast-path traces)
# remains valid. Exposed as a module-level constant so tests can monkey-
# patch it down to force on-demand engagement on small models.
ON_DEMAND_PARAM_BYTES_FRACTION: float = 0.60


@dataclass
class _OpFrame:
    """Mutable per-op bookkeeping used only while a forward hook pair is live."""

    op_id: OpId
    module_path: str
    qualified_name: str
    shape_signature: tuple[tuple[int, ...], ...]
    block_id: BlockId | None
    is_forward: bool
    allocated_before: int
    prev_end_before: int
    # Pair of torch.cuda.Events recorded at pre-/post-forward. ``elapsed_time``
    # is read lazily after the final ``torch.cuda.synchronize`` at the end of
    # ``run_trace`` so the hook path does not stall on a per-op sync.
    # Typed as ``object`` here to keep this module import-light (torch is a
    # TYPE_CHECKING-only import at the top of the file).
    pre_event: object = None
    post_event: object = None


def _infer_block_id(module_path: str) -> BlockId | None:
    """Extract a transformer-block index from a dotted module path, if present.

    Heuristic: look for an ``...h.<i>...`` (GPT-2), ``layers.<i>``, or
    ``transformer.blocks.<i>`` fragment. Good enough for the M1 contract;
    M2's ChunkLayout supplies the authoritative block->module map.
    """
    parts = module_path.split(".")
    for prev, cur in zip(parts, parts[1:]):
        if prev in {"h", "layers", "blocks", "block", "layer"} and cur.isdigit():
            return BlockId(int(cur))
    return None


def _shape_sig(inputs: Any) -> tuple[tuple[int, ...], ...]:
    """Best-effort input-shape signature. Non-tensor inputs become ``()``."""
    out: list[tuple[int, ...]] = []
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)
    for arg in inputs:
        shape = getattr(arg, "shape", None)
        if shape is not None:
            try:
                out.append(tuple(int(d) for d in shape))
            except TypeError:
                out.append(())
        else:
            out.append(())
    return tuple(out)


def _count_model_state_bytes(
    model: "nn.Module",
    *,
    param_grad_bytes_per_param: int = DEFAULT_PARAM_GRAD_BYTES_PER_PARAM,
    optim_state_bytes_per_param: int = DEFAULT_OPTIM_STATE_BYTES_PER_PARAM,
) -> int:
    """Constant-size model-state footprint: params + grads + optimizer states."""
    n = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    return int(n) * (param_grad_bytes_per_param + optim_state_bytes_per_param)


def _arch_hash(model: "nn.Module") -> str:
    """Deterministic hash of the model architecture for the cache key."""
    parts: list[str] = [type(model).__name__]
    for name, p in model.named_parameters():
        parts.append(f"{name}:{tuple(p.shape)}:{p.dtype}")
    for name, b in model.named_buffers():
        parts.append(f"B:{name}:{tuple(b.shape)}:{b.dtype}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _sku(device: "torch.device | str") -> str:
    import torch

    try:
        return torch.cuda.get_device_name(device)
    except Exception:  # pragma: no cover - defensive
        return "cpu"


def run_trace(
    model: "nn.Module",
    batch: dict,
    cfg: ProfilerConfig,
    *,
    param_grad_bytes_per_param: int = DEFAULT_PARAM_GRAD_BYTES_PER_PARAM,
    optim_state_bytes_per_param: int = DEFAULT_OPTIM_STATE_BYTES_PER_PARAM,
) -> ProfilerTrace:
    """Run a single forward (+optional backward) pass and record memory deltas.

    Args:
        model: any standard ``nn.Module``. Must be on ``cfg.device``.
        batch: kwargs dict passed to ``model(**batch)``. The output must expose
            a ``.loss`` scalar or be a tensor we can call ``.sum().backward()``
            on, if ``cfg.include_backward`` is True.
        cfg: profiler configuration — see ``types.ProfilerConfig``.
        param_grad_bytes_per_param: override the fp16 param+grad assumption.
        optim_state_bytes_per_param: override the Adam (fp32 master + m + v)
            assumption.

    Returns:
        A fully-populated ``ProfilerTrace``.
    """
    import torch

    device = torch.device(cfg.device)
    cuda_available_for_bench = (
        device.type == "cuda" and torch.cuda.is_available()
    )

    # Run the Adam microbenchmarks BEFORE installing the memory-delta
    # tracker. The benchmarks allocate a ~100-200 MB synthetic param
    # + optimizer state that is cleaned up before return, but the
    # caching allocator retains some of it as reserved-but-free. By
    # folding that into the ``tracker.mark_end`` baseline below, we
    # avoid perturbing the intra/inter-op delta accounting that the
    # cost model consumes for peak reconstruction.
    try:
        cpu_adam_bps = measure_cpu_adam()
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("measure_cpu_adam failed (%s); recording 0.0", exc)
        cpu_adam_bps = 0.0
    try:
        dev_idx_for_bench = device.index if device.index is not None else 0
        gpu_adam_bps = (
            measure_gpu_adam(dev_idx_for_bench) if cuda_available_for_bench else 0.0
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("measure_gpu_adam failed (%s); recording 0.0", exc)
        gpu_adam_bps = 0.0

    # Sync after benches — but do NOT call empty_cache() here. Doing so
    # would release reserved-but-free blocks that the caching allocator
    # would later need to reallocate during the traced forward+backward,
    # inflating the traced pass's peak memory vs. the post-trace
    # "ground truth" run (which the reconstructed-peak test compares
    # against). Letting the allocator reuse the reserved pool keeps
    # the first-iter peak representative.
    if cuda_available_for_bench:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    tracker = MemoryDeltaTracker(device)
    # Seed the tracker's baseline with the CURRENT allocated bytes so the
    # first op's inter-op delta measures only the transient allocated
    # *between* profiler entry and first hook fire — not the model weights
    # already resident when the profiler started. Without this, the first
    # op's inter-op delta captures the entire baseline (e.g. 13 GiB for
    # Llama-7B), which F_bm in cost/memory.py then double-counts against
    # the model_state_present term.
    tracker.mark_end(tracker.snapshot().allocated_bytes)

    # --- per-op accumulators -------------------------------------------
    op_records: list[OpRecord] = []
    intra_deltas: dict[OpId, int] = {}
    inter_deltas: dict[OpId, int] = {}
    activation_sizes: dict[BlockId, int] = {}

    # Eager-record / lazy-read cuda.Event pairs per op. Populated by the
    # post-forward hook after recording the "post" event; resolved into
    # ``op_latencies`` (seconds) after ``torch.cuda.synchronize()`` so that
    # ``Event.elapsed_time`` reads never stall the hook path.
    pending_events: list[tuple[OpId, object, object]] = []

    # Stack of in-flight _OpFrames keyed by the calling module id. Submodules
    # fire pre-hooks before their parent's post-hook; a dict keyed on id()
    # matches that LIFO nesting without needing a real stack type.
    live_frames: dict[int, _OpFrame] = {}

    next_op_id = 0

    cuda_available = device.type == "cuda" and torch.cuda.is_available()

    def _module_path(m: "nn.Module") -> str:
        """Dotted path of ``m`` inside ``model`` (root -> '')."""
        for name, candidate in model.named_modules():
            if candidate is m:
                return name or type(m).__name__
        return type(m).__name__  # unreachable in practice

    def _pre_forward(module: "nn.Module", inputs):
        nonlocal next_op_id
        op_id = OpId(next_op_id)
        next_op_id += 1
        tracker.reset()
        snap = tracker.snapshot()
        path = _module_path(module)
        pre_event = None
        if cuda_available:
            pre_event = torch.cuda.Event(enable_timing=True)
            pre_event.record()
        live_frames[id(module)] = _OpFrame(
            op_id=op_id,
            module_path=path,
            qualified_name=type(module).__name__,
            shape_signature=_shape_sig(inputs),
            block_id=_infer_block_id(path),
            is_forward=True,
            allocated_before=snap.allocated_bytes,
            prev_end_before=tracker.last_end_bytes,
            pre_event=pre_event,
        )

    def _post_forward(module: "nn.Module", inputs, output):
        frame = live_frames.pop(id(module), None)
        if frame is None:
            return
        snap = tracker.snapshot()
        intra = intra_op_delta(frame.allocated_before, snap.peak_allocated_bytes)
        inter = inter_op_delta(frame.prev_end_before, snap.peak_allocated_bytes)
        tracker.mark_end(snap.allocated_bytes)

        if cuda_available and frame.pre_event is not None:
            post_event = torch.cuda.Event(enable_timing=True)
            post_event.record()
            pending_events.append((frame.op_id, frame.pre_event, post_event))

        op_records.append(
            OpRecord(
                op_id=frame.op_id,
                module_path=frame.module_path,
                qualified_name=frame.qualified_name,
                shape_signature=frame.shape_signature,
                block_id=frame.block_id,
                is_forward=True,
            )
        )
        intra_deltas[frame.op_id] = intra
        inter_deltas[frame.op_id] = inter

        # Retained-activation approximation: bytes of the output tensor(s).
        # The authoritative per-block activation footprint is reconstructed
        # in M4; this gives the M1 peak estimator something non-zero to work
        # with when a block_id is inferrable.
        if frame.block_id is not None:
            out_bytes = _output_bytes(output)
            activation_sizes[frame.block_id] = activation_sizes.get(
                frame.block_id, 0
            ) + out_bytes

    def _output_bytes(output: Any) -> int:
        total = 0
        stack: list[Any] = [output]
        while stack:
            item = stack.pop()
            if isinstance(item, torch.Tensor):
                total += item.numel() * item.element_size()
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
            elif isinstance(item, dict):
                stack.extend(item.values())
        return total

    # --- decide on-demand engagement up front --------------------------
    # The decision must happen before warmups + steady-state, because for
    # 13B+ models the very first un-offloaded forward will OOM. When on-
    # demand is engaged we SKIP warmups and steady-state — those passes
    # depend on running a normal full-forward without offload, which is
    # exactly what doesn't fit. The cost model falls back to defaults
    # (identity scale, default bwd_fwd ratio) for traces marked on-demand.
    engage_on_demand = False
    if cfg.on_demand and cuda_available:
        try:
            gpu_total = int(
                torch.cuda.get_device_properties(device).total_memory
            )
            param_bytes = sum(
                p.numel() * p.element_size()
                for p in model.parameters()
            )
            if param_bytes > ON_DEMAND_PARAM_BYTES_FRACTION * gpu_total:
                engage_on_demand = True
                LOG.info(
                    "Profiler engaging on-demand mode: params=%.2f GB exceed "
                    "%.0f%% of %.2f GB device memory; offloading params + "
                    "saved-for-backward tensors to CPU between modules.",
                    param_bytes / 1e9,
                    ON_DEMAND_PARAM_BYTES_FRACTION * 100,
                    gpu_total / 1e9,
                )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug(
                "On-demand size check failed (%s); falling back to fast path",
                exc,
            )

    # --- warmup passes (no hooks) to JIT-compile kernels ---------------
    # Without warmup, the ``op_latencies`` captured in the traced pass
    # below measure COLD-start kernel times (JIT compile + allocator
    # warm-up), which can be 10x higher than steady-state. Running a
    # couple of un-timed forward+backward passes first brings kernels
    # into the cache so the traced pass reflects steady-state per-op
    # cost. Two warmups land comfortably inside the 3-6s profiling
    # budget §3.2 quotes for 7-20B models and closes most of the
    # cold-vs-warm gap (the second hot iter is ~2x faster than the
    # first, diminishing-returns after).
    N_WARMUP = 0 if engage_on_demand else 2
    if cuda_available and N_WARMUP > 0:
        for _i in range(N_WARMUP):
            try:
                torch.cuda.synchronize(device)
                warm_out = model(**batch)
                if cfg.include_backward:
                    warm_loss = _extract_loss(warm_out)
                    warm_loss.backward()
                    model.zero_grad(set_to_none=True)
                del warm_out
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover - defensive
                LOG.debug("profiler warmup pass failed (%s); continuing cold", exc)
                break

    # --- steady-state (hook-less) wall-time measurement ---------------
    # Captured BEFORE hooks are installed. The scalar ratio
    # ``steady_fwd_wall_s / hooked_fwd_wall_s`` is the calibration factor
    # the cost model applies to strip hook dispatch overhead out of the
    # hooked per-op latencies (~2.5x inflation on ~1000-leaf transformer
    # models). See ``ProfilerTrace.hooked_fwd_wall_s`` docstring for the
    # full rationale.
    #
    # During this pass we ALSO install a lightweight pair of pre/post
    # forward hooks on each TRANSFORMER BLOCK (not every leaf) to capture
    # per-block peak bytes. The hooks only call
    # ``torch.cuda.reset_peak_memory_stats`` + ``torch.cuda.max_memory_allocated``
    # (two allocator reads, ~tens of µs each). Since we only instrument
    # at block granularity (tens of blocks, not ~1000 leaves), hook
    # dispatch cost here is negligible relative to the block compute
    # itself — unlike the per-leaf hooks used later for the full trace,
    # which inflate wall time ~8x on 7B Llama. The per-block peaks are
    # consumed by the memory cost model as a ground-truth upper bound
    # on the forward peak for any NONE/CKPT/SWAP mix.
    steady_fwd_wall_s = 0.0
    steady_bwd_wall_s = 0.0
    steady_fwd_peak_bytes = 0
    steady_fwd_block_peak_bytes: dict[BlockId, int] = {}
    # Skip steady-state when on-demand engaged — running full-forward
    # without offload is exactly what we can't do for these models. Cost
    # model falls back to identity scale + default bwd/fwd ratio.
    if cuda_available and not engage_on_demand:
        # Discover transformer blocks for per-block peak instrumentation.
        # If discovery fails (non-standard model shape), skip per-block
        # capture — the aggregate ``steady_fwd_peak_bytes`` below still
        # fires and preserves backward compat with the v5 cap path.
        block_handles: list[Any] = []
        try:
            from axolotl.integrations.protrain.block.layout_rules import (
                discover_blocks,
            )

            blocks = discover_blocks(model)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug(
                "profiler: discover_blocks failed (%s); skipping per-block "
                "peak capture, aggregate cap only", exc
            )
            blocks = []

        def _make_pre(_dev):
            def _pre(_mod, _inputs):
                torch.cuda.reset_peak_memory_stats(_dev)
            return _pre

        def _make_post(bid, _dev):
            def _post(_mod, _inputs, _output):
                steady_fwd_block_peak_bytes[bid] = int(
                    torch.cuda.max_memory_allocated(_dev)
                )
            return _post

        for idx, block in enumerate(blocks):
            bid = BlockId(idx)
            block_handles.append(
                block.register_forward_pre_hook(_make_pre(device))
            )
            block_handles.append(
                block.register_forward_hook(_make_post(bid, device))
            )

        # Multi-iter hot-loop measurement. A single forward still carries
        # allocator-settle cost that a real steady-state training loop
        # wouldn't pay. Run N=4 un-hooked iters and take the median of
        # iters 2-3 as the steady value; iter 0/1 soak up any residual
        # warmup. Per-block peak bytes take the max across all measured
        # iters to capture the true high-water mark.
        # Best-effort steady backward: runs inside the same loop (after
        # each forward) IFF the trace config allows it. Backward on a
        # 7B-class model without chunking engaged will OOM, so guard
        # with try/except per-iter and fall back to 0.0 on any failure
        # (cost model then uses the default bwd_fwd ratio).
        N_STEADY_ITERS = 4
        N_STEADY_WARMUP = 2
        fwd_iter_s: list[float] = []
        bwd_iter_s: list[float] = []
        try:
            for i in range(N_STEADY_ITERS):
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                pre_sf = torch.cuda.Event(enable_timing=True)
                post_sf = torch.cuda.Event(enable_timing=True)
                pre_sf.record()
                steady_out = model(**batch)
                post_sf.record()
                torch.cuda.synchronize(device)
                fwd_iter_s.append(pre_sf.elapsed_time(post_sf) / 1000.0)
                # High-water mark across all iters
                steady_fwd_peak_bytes = max(
                    steady_fwd_peak_bytes,
                    int(torch.cuda.max_memory_allocated(device)),
                )

                if cfg.include_backward:
                    try:
                        steady_loss = _extract_loss(steady_out)
                        torch.cuda.synchronize(device)
                        pre_sb = torch.cuda.Event(enable_timing=True)
                        post_sb = torch.cuda.Event(enable_timing=True)
                        pre_sb.record()
                        steady_loss.backward()
                        post_sb.record()
                        torch.cuda.synchronize(device)
                        bwd_iter_s.append(
                            pre_sb.elapsed_time(post_sb) / 1000.0
                        )
                        model.zero_grad(set_to_none=True)
                    except Exception as bwd_exc:  # pragma: no cover
                        LOG.debug(
                            "profiler steady backward iter %d failed (%s); "
                            "cost model falls back to bwd_fwd ratio", i, bwd_exc
                        )
                        bwd_iter_s.clear()  # drop partial measurements
                        # Don't raise — continue forward timing
                del steady_out
                torch.cuda.synchronize(device)

            # Steady value = median of iters [N_STEADY_WARMUP:]. With
            # N=4 warmup=2 this is the median of the last 2.
            import statistics
            steady_slice = fwd_iter_s[N_STEADY_WARMUP:]
            if steady_slice:
                steady_fwd_wall_s = statistics.median(steady_slice)
            bwd_slice = bwd_iter_s[N_STEADY_WARMUP:] if bwd_iter_s else []
            if bwd_slice:
                steady_bwd_wall_s = statistics.median(bwd_slice)
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug(
                "profiler hook-less steady-state measurement failed (%s); "
                "cost model will fall back to identity scale", exc
            )
            steady_fwd_wall_s = 0.0
            steady_bwd_wall_s = 0.0
            steady_fwd_peak_bytes = 0
            steady_fwd_block_peak_bytes = {}
        finally:
            for h in block_handles:
                h.remove()

    # --- install hooks on every nn.Module (leaves + composites) --------
    handles: list[Any] = []
    for sub in model.modules():
        handles.append(sub.register_forward_pre_hook(_pre_forward))
        handles.append(sub.register_forward_hook(_post_forward))

    model_state_bytes = _count_model_state_bytes(
        model,
        param_grad_bytes_per_param=param_grad_bytes_per_param,
        optim_state_bytes_per_param=optim_state_bytes_per_param,
    )

    # --- on-demand wrapper for the traced forward ----------------------
    # The engage decision was made up-front (before warmups). Wrapper
    # honours that — fast path stays a no-op context manager.
    on_demand_mgr = OnDemandTensorMgr(
        device=device, disabled=not engage_on_demand, model=model
    )

    # Record total wall-clock of the HOOKED forward pass. Event-timed so
    # hook dispatch gaps (Python overhead between ops) are included — the
    # sum of per-op ``op_latencies`` would miss those gaps and understate
    # the hook penalty. Paired with ``steady_fwd_wall_s`` above, this is
    # what the cost model's scale factor consumes.
    hooked_fwd_wall_s = 0.0
    hooked_fwd_pre_event = None
    hooked_fwd_post_event = None

    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with on_demand_mgr:
            if cuda_available:
                hooked_fwd_pre_event = torch.cuda.Event(enable_timing=True)
                hooked_fwd_pre_event.record()
            output = model(**batch)
            if cuda_available and hooked_fwd_pre_event is not None:
                hooked_fwd_post_event = torch.cuda.Event(enable_timing=True)
                hooked_fwd_post_event.record()

            if cfg.include_backward:
                loss = _extract_loss(output)
                # Record a synthetic backward op id so intra/inter maps carry
                # a "backward total" entry — matches the paper's op_order being
                # fwd ops then bwd ops.
                next_op_id_local = next_op_id
                bwd_op_id = OpId(next_op_id_local)
                next_op_id = next_op_id_local + 1
                tracker.reset()
                before = tracker.snapshot()
                prev_end = tracker.last_end_bytes
                bwd_pre_event = None
                if cuda_available:
                    bwd_pre_event = torch.cuda.Event(enable_timing=True)
                    bwd_pre_event.record()
                loss.backward()
                if cuda_available and bwd_pre_event is not None:
                    bwd_post_event = torch.cuda.Event(enable_timing=True)
                    bwd_post_event.record()
                    pending_events.append((bwd_op_id, bwd_pre_event, bwd_post_event))
                snap = tracker.snapshot()
                intra_deltas[bwd_op_id] = intra_op_delta(
                    before.allocated_bytes, snap.peak_allocated_bytes
                )
                inter_deltas[bwd_op_id] = inter_op_delta(
                    prev_end, snap.peak_allocated_bytes
                )
                tracker.mark_end(snap.allocated_bytes)
                op_records.append(
                    OpRecord(
                        op_id=bwd_op_id,
                        module_path="<backward>",
                        qualified_name="<backward>",
                        shape_signature=(),
                        block_id=None,
                        is_forward=False,
                    )
                )
        torch.cuda.synchronize(device)
    finally:
        for h in handles:
            h.remove()

    # --- resolve pending events into op_latencies (seconds) -------------
    # Eager-record / lazy-read: all Events were recorded during the hook
    # path; ``elapsed_time`` is only valid after both events complete,
    # which the sync above guarantees. Reading now avoids per-op stalls.
    op_latencies: dict[OpId, float] = {}
    if cuda_available:
        for op_id, pre_ev, post_ev in pending_events:
            try:
                elapsed_ms = pre_ev.elapsed_time(post_ev)
            except Exception as exc:  # pragma: no cover - defensive
                LOG.debug("Event.elapsed_time failed for op %s: %s", op_id, exc)
                continue
            # Guard negative / absurd readings from clock skew.
            if elapsed_ms < 0:
                continue
            op_latencies[op_id] = elapsed_ms / 1000.0

        # Resolve the whole-forward hooked wall time from the pair of
        # events wrapping the hooked forward call (see above). Must
        # happen after the ``torch.cuda.synchronize`` that ends the
        # traced iter so both events are complete.
        if hooked_fwd_pre_event is not None and hooked_fwd_post_event is not None:
            try:
                hooked_fwd_wall_s = (
                    hooked_fwd_pre_event.elapsed_time(hooked_fwd_post_event)
                    / 1000.0
                )
            except Exception as exc:  # pragma: no cover - defensive
                LOG.debug("hooked forward Event.elapsed_time failed: %s", exc)
                hooked_fwd_wall_s = 0.0

    # --- hardware microbenchmarks --------------------------------------
    # PCIe is measured here (post-trace) rather than pre-trace because the
    # copy engines are unaffected by the earlier Adam microbenchmarks and
    # running PCIe post-trace matches the pre-v3 measurement ordering.
    try:
        dev_idx = device.index if device.index is not None else 0
        pcie_h2d_bps, pcie_d2h_bps = measure_pcie(dev_idx)
    except Exception as exc:  # pragma: no cover - defensive, GPU-only
        LOG.warning("measure_pcie failed (%s); recording zeros", exc)
        pcie_h2d_bps = pcie_d2h_bps = 0.0

    # Adam microbenchmark results (cpu_adam_bps, gpu_adam_bps) were
    # populated above, BEFORE the tracker baseline was captured, so
    # their allocator footprint does not perturb op-delta accounting.

    # Trainable-param fraction. LoRA training has ~0.1% trainable; the cost
    # model uses this to pick a tighter bwd/fwd-ratio fallback (LoRA backward
    # is ~1× forward, vs the 2× canonical full-finetune ratio).
    try:
        n_trainable = sum(
            int(p.numel()) for p in model.parameters() if p.requires_grad
        )
        n_total = sum(int(p.numel()) for p in model.parameters())
        trainable_param_fraction = (
            n_trainable / n_total if n_total > 0 else 0.0
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.debug("trainable_param_fraction probe failed (%s)", exc)
        trainable_param_fraction = 0.0

    # Per-SKU compute rate, captured on the trace SKU so cross-SKU replays
    # can scale per-op latencies. Same-SKU runs see ratio ≈ 1.0 and the
    # calibration is a no-op. Recorded post-PCIe so allocator state is settled.
    try:
        dev_idx_for_compute = device.index if device.index is not None else 0
        compute_rate_tflops = (
            measure_compute_rate(dev_idx_for_compute) if cuda_available else 0.0
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning(
            "measure_compute_rate failed (%s); recording 0.0 — cost model "
            "will skip SKU calibration", exc,
        )
        compute_rate_tflops = 0.0

    # Resolve world size: prefer cfg.world_size, fall back to the live
    # torch.distributed group, default to 1.
    resolved_world = cfg.world_size
    if resolved_world is None:
        try:
            import torch.distributed as _dist
            resolved_world = (
                _dist.get_world_size() if _dist.is_initialized() else 1
            )
        except Exception:  # noqa: BLE001 - defensive
            resolved_world = 1

    try:
        gather_table, reduce_table = measure_nccl(world_size=resolved_world)
    except Exception as exc:  # pragma: no cover - distributed-only paths
        LOG.warning(
            "measure_nccl failed (%s); recording empty collective tables. "
            "Cost model's communication term will degrade to 0.", exc,
        )
        gather_table, reduce_table = ({}, {})

    return ProfilerTrace(
        op_order=tuple(op_records),
        intra_op_delta=intra_deltas,
        inter_op_delta=inter_deltas,
        activation_sizes=activation_sizes,
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=pcie_h2d_bps,
        pcie_d2h_bps=pcie_d2h_bps,
        nccl_gather_s=gather_table,
        nccl_reduce_s=reduce_table,
        arch_hash=_arch_hash(model),
        bs=cfg.batch_size,
        seq=cfg.seq_len,
        sku=_sku(device),
        world=resolved_world,
        op_latencies=op_latencies,
        cpu_adam_bytes_per_sec=cpu_adam_bps,
        gpu_adam_bytes_per_sec=gpu_adam_bps,
        hooked_fwd_wall_s=hooked_fwd_wall_s,
        steady_fwd_wall_s=steady_fwd_wall_s,
        steady_bwd_wall_s=steady_bwd_wall_s,
        steady_fwd_peak_bytes=steady_fwd_peak_bytes,
        steady_fwd_block_peak_bytes=steady_fwd_block_peak_bytes,
        compute_rate_tflops=compute_rate_tflops,
        trainable_param_fraction=trainable_param_fraction,
    )


def _extract_loss(output: Any) -> "torch.Tensor":
    """Pull a scalar loss out of a HuggingFace-style output or raw tensor."""
    import torch

    loss = getattr(output, "loss", None)
    if isinstance(loss, torch.Tensor):
        return loss
    if isinstance(output, dict) and isinstance(output.get("loss"), torch.Tensor):
        return output["loss"]
    if isinstance(output, torch.Tensor):
        return output.sum()
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() == 0:
                return item
        # fall back to summing the first tensor we can find
        for item in output:
            if isinstance(item, torch.Tensor):
                return item.sum()
    raise TypeError(f"run_trace: unable to extract a loss from output of type {type(output)}")


__all__ = ["run_trace"]
