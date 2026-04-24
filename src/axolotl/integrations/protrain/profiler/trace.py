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
    N_WARMUP = 2
    if cuda_available:
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

    # --- execute the single iteration under the on-demand wrapper ------
    on_demand_mgr = OnDemandTensorMgr(device=device, disabled=not cfg.on_demand)
    # For M1 the wrapper is a no-op fast path; replay mode is M4.
    on_demand_mgr.disabled = True  # M1 override: full fwd+bwd always.

    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with on_demand_mgr:
            output = model(**batch)

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

    # --- hardware microbenchmarks --------------------------------------
    try:
        dev_idx = device.index if device.index is not None else 0
        pcie_h2d_bps, pcie_d2h_bps = measure_pcie(dev_idx)
    except Exception as exc:  # pragma: no cover - defensive, GPU-only
        LOG.warning("measure_pcie failed (%s); recording zeros", exc)
        pcie_h2d_bps = pcie_d2h_bps = 0.0

    nccl_table = measure_nccl(world_size=1)  # M1 is single-rank.

    return ProfilerTrace(
        op_order=tuple(op_records),
        intra_op_delta=intra_deltas,
        inter_op_delta=inter_deltas,
        activation_sizes=activation_sizes,
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=pcie_h2d_bps,
        pcie_d2h_bps=pcie_d2h_bps,
        nccl_gather_s=nccl_table,
        nccl_reduce_s=nccl_table,
        arch_hash=_arch_hash(model),
        bs=cfg.batch_size,
        seq=cfg.seq_len,
        sku=_sku(device),
        world=1,
        op_latencies=op_latencies,
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
