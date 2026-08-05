# Vendored from MegaTrain: https://github.com/DLYuanGod/MegaTrain
# Revision: 7f5c9597e5b20bb618932c77c922e8eac4a11c4d (Apache-2.0)
# Modified by Axolotl; see _vendor/PROVENANCE.md for the list of changes.

"""Worker process for multiprocessing data parallelism.

Each worker process owns one GPU and runs forward/backward independently.
Workers read shared-memory weight buffers for H2D and write gradients to
shared-memory accumulators with a lock.
"""

import logging
import threading
import traceback
import queue
import torch
import torch.nn as nn

from axolotl.integrations.megatrain._vendor.infinity.model.cpu_master import (
    _GPUContext, _accumulate_parameter_grads_from_slab, _copy_module_for_compute,
    _copy_parameter_grads_to_slab, _dedup_parameters, _prepare_attention_mask,
    _preserve_attn_implementation, _replay_cuda_rng_state,
    FLASH_CE_AVAILABLE,
)
if FLASH_CE_AVAILABLE:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss

from axolotl.integrations.megatrain._vendor.infinity.model.mp_state import WorkerCommandType, WorkerResult

logger = logging.getLogger(__name__)


def _create_worker_gpu_context(rank, device_id, shared_state):
    """Create a _GPUContext for this worker's GPU.

    Similar to CPUMasterModel._create_gpu_context but reads from
    shared_state instead of self.
    """
    config = shared_state.config
    device = torch.device(f"cuda:{device_id}")
    ctx = _GPUContext(rank, device)

    logger.info(f"Worker {rank}: creating GPU context on {device}...")

    # Double-buffered GPU flat buffers (no CPU pinned staging needed —
    # we copy directly from shared-memory flats to GPU)
    ctx.gpu_flat_buffers = [
        torch.empty(shared_state.max_layer_numel, dtype=config.dtype, device=device),
        torch.empty(shared_state.max_layer_numel, dtype=config.dtype, device=device)
    ]
    # Unused but kept for API compat with _unflatten_to_layer
    ctx.cpu_flat_buffers = [None, None]

    # GPU layer templates (per structure group, double buffered)
    ctx.gpu_layer_templates = {}
    for gid, group in shared_state.layer_groups.items():
        representative_idx = group['indices'][0]
        templates = []
        for _ in range(2):
            template = _copy_module_for_compute(
                shared_state.cpu_layers[representative_idx], device, config.dtype
            )
            _preserve_attn_implementation(template, shared_state._model_config)
            for p in template.parameters():
                p.requires_grad_(False)
            templates.append(template)
        ctx.gpu_layer_templates[gid] = templates

    # GPU modules
    ctx.emb_gpu = _copy_module_for_compute(shared_state.embedding, device, config.dtype)
    ctx.norm_gpu = _copy_module_for_compute(shared_state.norm, device, config.dtype)
    ctx.lm_head_gpu = _copy_module_for_compute(shared_state.lm_head, device, config.dtype)

    if shared_state.tied_lm_head and hasattr(ctx.lm_head_gpu, "weight"):
        ctx.lm_head_gpu.weight = ctx.emb_gpu.weight

    ctx.rotary_gpu = _copy_module_for_compute(
        shared_state.rotary_emb, device, config.dtype
    )

    # CUDA streams
    ctx.compute_stream = torch.cuda.current_stream(device=device)
    ctx.weight_stream = torch.cuda.Stream(device=device)
    ctx.grad_stream = torch.cuda.Stream(device=device)

    # Synchronization events
    ctx.weight_ready_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
    ctx.h2d_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
    ctx.backward_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
    ctx.buffer_busy_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
    ctx.buffer_free_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
    ctx.template_free_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
    ctx.param_sync_event = torch.cuda.Event(enable_timing=False)
    ctx.loss_backward_done = torch.cuda.Event(enable_timing=False)
    ctx.embedding_backward_done = torch.cuda.Event(enable_timing=False)

    # K-slab gradient pool (per-worker, pinned CPU memory)
    ctx.layer_grad_slabs = [
        torch.empty(shared_state.max_layer_numel, dtype=config.dtype, device='cpu', pin_memory=True)
        for _ in range(config.num_grad_slabs)
    ]
    ctx.layer_slab_events = [
        torch.cuda.Event(enable_timing=False) for _ in range(config.num_grad_slabs)
    ]
    ctx.layer_slab_free_list = queue.Queue()
    for i in range(config.num_grad_slabs):
        ctx.layer_slab_free_list.put(i)

    ctx.head_grad_slab = torch.empty(shared_state.head_total_numel, dtype=config.dtype, device='cpu', pin_memory=True)
    ctx.head_slab_event = torch.cuda.Event(enable_timing=False)
    ctx.head_slab_free = threading.Event()
    ctx.head_slab_free.set()

    ctx.embed_grad_slab = torch.empty(shared_state.embed_total_numel, dtype=config.dtype, device='cpu', pin_memory=True)
    ctx.embed_slab_event = torch.cuda.Event(enable_timing=False)
    ctx.embed_slab_free = threading.Event()
    ctx.embed_slab_free.set()

    # Flash CE loss
    if FLASH_CE_AVAILABLE:
        ctx.ce_loss = FlashCrossEntropyLoss(inplace_backward=True, ignore_index=-100, reduction='none')
    else:
        ctx.ce_loss = None

    # Initialize events
    current_stream = torch.cuda.current_stream(device)
    for i in range(2):
        ctx.buffer_free_events[i].record(current_stream)
        ctx.template_free_events[i].record(current_stream)
        ctx.h2d_done_events[i].record(current_stream)
    ctx.param_sync_event.record(current_stream)
    current_stream.synchronize()

    return ctx


def _worker_grad_fn(grad_queue, worker_stop, shared_state, ctx, worker_error):
    """Per-worker gradient accumulation thread.

    Same as CPUMasterModel._grad_worker but writes to shared-memory
    p_cpu.grad tensors with a lock for cross-process safety.
    """
    while not worker_stop.is_set():
        try:
            task = grad_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        slab_type, slab_idx, cpu_params, shapes, numels, has_grads = task

        try:
            if slab_type == 'layer':
                event = ctx.layer_slab_events[slab_idx]
                slab_flat = ctx.layer_grad_slabs[slab_idx]
            elif slab_type == 'head':
                event = ctx.head_slab_event
                slab_flat = ctx.head_grad_slab
            else:
                event = ctx.embed_slab_event
                slab_flat = ctx.embed_grad_slab

            event.synchronize()

            # Accumulate to shared-memory p_cpu.grad with lock
            with shared_state.grad_lock:
                _accumulate_parameter_grads_from_slab(
                    cpu_params, shapes, numels, has_grads, slab_flat
                )
        except Exception as exc:  # noqa: BLE001
            # Keep draining: an un-acked task would block the command loop forever
            # on grad_queue.join().
            if worker_error.get('error') is None:
                worker_error['error'] = traceback.format_exc()
            logger.exception("MegaTrain worker gradient thread failed")
            del exc
        finally:
            if slab_type == 'layer':
                ctx.layer_slab_free_list.put(slab_idx)
            elif slab_type == 'head':
                ctx.head_slab_free.set()
            else:
                ctx.embed_slab_free.set()

            grad_queue.task_done()


def _worker_load_layer_to_buffer_async(layer_idx, buffer_idx, ctx, shared_state):
    """Load layer weights from shared-memory flat to GPU buffer.

    Direct shared-memory → GPU copy (non_blocking). Not as fast as
    pinned → GPU DMA, but avoids the need for pinned shared memory.
    """
    ctx.h2d_done_events[buffer_idx].synchronize()
    ctx.weight_stream.wait_event(ctx.buffer_free_events[buffer_idx])

    shared_flat = shared_state.layer_shared_flats[layer_idx]
    layer_numel = shared_state.layer_numels[layer_idx]

    with torch.cuda.stream(ctx.weight_stream):
        ctx.gpu_flat_buffers[buffer_idx][:layer_numel].copy_(
            shared_flat, non_blocking=True
        )
        ctx.weight_ready_events[buffer_idx].record(ctx.weight_stream)
        ctx.h2d_done_events[buffer_idx].record(ctx.weight_stream)


def _worker_sync_gpu_modules(ctx, shared_state):
    """Refresh worker's GPU modules from shared CPU master params."""
    for p_gpu, p_cpu in zip(
        ctx.emb_gpu.parameters(), shared_state.embedding.parameters(), strict=True
    ):
        p_gpu.data.copy_(p_cpu.data, non_blocking=True)

    if ctx.norm_gpu:
        for p_gpu, p_cpu in zip(
            ctx.norm_gpu.parameters(), shared_state.norm.parameters(), strict=True
        ):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)

    if not shared_state.tied_lm_head:
        for p_gpu, p_cpu in zip(
            ctx.lm_head_gpu.parameters(), shared_state.lm_head.parameters(), strict=True
        ):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)

    if ctx.rotary_gpu:
        for p_gpu, p_cpu in zip(
            ctx.rotary_gpu.parameters(), shared_state.rotary_emb.parameters(), strict=True
        ):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)

    ctx.param_sync_event.record(torch.cuda.current_stream(ctx.device))


def gpu_worker_fn(rank, shared_state):
    """Main entry point for a worker process (spawned via mp.spawn).

    Creates a GPU context, starts a gradient worker thread, and enters
    a command loop receiving work from the main process.
    """
    device_id = shared_state.device_ids[rank]
    torch.cuda.set_device(device_id)

    # Re-attach shared-memory grad tensors lost during spawn pickling
    shared_state.reattach_grads()

    # Register shared-memory flats with CUDA for pinned DMA speed
    shared_state.register_shared_flats_as_pinned()

    logger.info(f"Worker {rank}: starting on cuda:{device_id}")

    ctx = _create_worker_gpu_context(rank, device_id, shared_state)

    # Per-process grad worker thread
    grad_queue = queue.Queue()
    worker_stop = threading.Event()
    worker_error = {'error': None}
    grad_thread = threading.Thread(
        target=_worker_grad_fn,
        args=(grad_queue, worker_stop, shared_state, ctx, worker_error),
        daemon=True,
    )
    grad_thread.start()

    cmd_q = shared_state.cmd_queues[rank]
    result_q = shared_state.result_queues[rank]

    while True:
        cmd = cmd_q.get()

        if cmd.type == WorkerCommandType.SHUTDOWN:
            logger.info(f"Worker {rank}: shutting down")
            break

        # Every branch must reply exactly once: an unanswered command leaves the
        # parent blocked on this rank's result queue.
        try:
            if cmd.type == WorkerCommandType.SYNC_WEIGHTS:
                _worker_sync_gpu_modules(ctx, shared_state)
                result = WorkerResult()

            elif cmd.type == WorkerCommandType.FORWARD_BACKWARD:
                try:
                    result = _run_forward_backward(
                        rank, ctx, shared_state, grad_queue, cmd)
                finally:
                    # Must drain even when the step raised: a task still in flight
                    # would keep writing into shared gradients after the parent has
                    # been told the step failed, and would be counted next step.
                    grad_queue.join()
                if worker_error['error'] is not None:
                    result = WorkerResult(error=worker_error['error'])

            elif cmd.type == WorkerCommandType.RELEASE_GPU:
                _worker_release_gpu(ctx)
                result = WorkerResult()

            else:
                result = WorkerResult(
                    error=f"unknown worker command {cmd.type!r}"
                )
        except Exception:  # noqa: BLE001
            logger.exception(f"Worker {rank}: command {cmd.type} failed")
            result = WorkerResult(error=traceback.format_exc())

        # Always clear, so a fault is never reported against a later command.
        worker_error['error'] = None
        result_q.put(result)

    worker_stop.set()
    grad_thread.join(timeout=5.0)
    logger.info(f"Worker {rank}: exited")


def _worker_build_layer_kwargs(mask, cache_position, position_ids,
                               position_embeddings, shared_state):
    """Per-layer forward kwargs; `mask` may be a dict keyed by layer type."""
    model_config = getattr(
        shared_state._model_config, "text_config", shared_state._model_config
    )
    num_layers = len(shared_state.cpu_layers)
    per_layer = []
    for layer_idx in range(num_layers):
        layer_mask = mask
        if isinstance(mask, dict):
            layer_mask = mask[model_config.layer_types[layer_idx]]
        kwargs = {
            'attention_mask': layer_mask,
            'use_cache': False,
            'output_attentions': False,
        }
        if shared_state.layer_accepts_cache_position and cache_position is not None:
            kwargs['cache_position'] = cache_position
        if shared_state.layer_accepts_position_embeddings and position_embeddings is not None:
            kwargs['position_embeddings'] = position_embeddings
        if shared_state.layer_accepts_position_ids and position_ids is not None:
            kwargs['position_ids'] = position_ids
        per_layer.append(kwargs)
    return per_layer


def _run_forward_backward(rank, ctx, shared_state, grad_queue, cmd):
    """Run forward + backward on this worker's GPU.

    Adapted from CPUMasterModel._forward_and_backward_single_gpu.
    Uses shared_state for weight loading and grad_queue for gradient collection.
    """
    input_ids = cmd.input_ids
    attention_mask = cmd.attention_mask
    labels = cmd.labels
    global_valid_tokens = cmd.global_valid_tokens
    pixel_values = cmd.pixel_values

    if pixel_values is not None or cmd.vision_kwargs:
        raise NotImplementedError(
            "MegaTrain's multi-GPU worker does not implement vision inputs."
        )

    B, T = input_ids.shape

    ctx.compute_stream.wait_event(ctx.param_sync_event)

    start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    start.record(ctx.compute_stream)

    # === FORWARD ===
    with torch.no_grad():
        input_ids_gpu = input_ids.to(ctx.device)
        hidden = ctx.emb_gpu(input_ids_gpu)
        del input_ids_gpu

    cache_position = torch.arange(T, device=ctx.device)
    position_ids = torch.arange(T, device=ctx.device).unsqueeze(0).expand(B, -1)

    position_embeddings = None
    if ctx.rotary_gpu and shared_state.layer_accepts_position_embeddings:
        if shared_state.is_vlm:
            pos_3d = torch.arange(T, device=ctx.device).unsqueeze(0).unsqueeze(0).expand(3, B, -1)
            dummy = torch.empty((1, 1, T, shared_state.head_dim), device=ctx.device, dtype=torch.float32)
            cos, sin = ctx.rotary_gpu(dummy, pos_3d)
            position_embeddings = (cos.to(shared_state.config.dtype), sin.to(shared_state.config.dtype))
            del dummy, pos_3d
        else:
            dummy = torch.empty((1, 1, T, shared_state.head_dim), device=ctx.device, dtype=torch.float32)
            cos, sin = ctx.rotary_gpu(dummy, position_ids[:1])
            position_embeddings = (cos.to(shared_state.config.dtype), sin.to(shared_state.config.dtype))
            del dummy

    # Same mask construction as the single-GPU path: a hand-rolled causal|padding
    # mask would drop sliding-window attention entirely for sdpa and eager.
    mask = _prepare_attention_mask(
        shared_state._model_config,
        attention_mask.to(ctx.device) if attention_mask is not None else None,
        hidden,
        position_ids,
    )
    layer_kwargs = _worker_build_layer_kwargs(
        mask, cache_position, position_ids, position_embeddings, shared_state
    )

    checkpoints = {}
    layer_rng_states = {}
    num_layers = len(shared_state.cpu_layers)
    checkpoint_interval = shared_state.config.checkpoint_interval

    with torch.no_grad():
        _worker_load_layer_to_buffer_async(0, 0, ctx, shared_state)
        ctx.weight_stream.synchronize()
        _worker_unflatten_to_layer(0, 0, ctx, shared_state)

        for i in range(num_layers):
            buffer_idx = i % 2
            next_buffer_idx = (i + 1) % 2

            if i % checkpoint_interval == 0:
                checkpoints[i] = hidden.detach()

            if i + 1 < num_layers:
                _worker_load_layer_to_buffer_async(i + 1, next_buffer_idx, ctx, shared_state)

            ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

            with torch.cuda.stream(ctx.compute_stream):
                _worker_unflatten_to_layer(i, buffer_idx, ctx, shared_state)
                ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                group_id = shared_state.layer_to_group[i]
                gpu_layer = ctx.gpu_layer_templates[group_id][buffer_idx]
                layer_rng_states[i] = torch.cuda.get_rng_state(ctx.device)
                out = gpu_layer(hidden, **layer_kwargs[i])
                hidden = out[0] if isinstance(out, tuple) else out

    checkpoints[num_layers] = hidden.detach()

    fwd_end.record(ctx.compute_stream)

    # === LOSS + BACKWARD ===
    labels_gpu = labels.to(ctx.device)
    V = shared_state.vocab_size
    chunk_size = 128

    hidden_before_norm = checkpoints[num_layers].requires_grad_(True)
    if ctx.norm_gpu:
        hidden_after_norm = ctx.norm_gpu(hidden_before_norm)
    else:
        hidden_after_norm = hidden_before_norm

    total_loss = torch.zeros((), device=ctx.device, dtype=torch.float32)
    total_valid_tokens = int(labels_gpu[:, 1:].ne(-100).sum().item())

    if total_valid_tokens == 0:
        return WorkerResult(loss_val=0.0, total_tokens=B * T, valid_tokens=0)

    denom = global_valid_tokens if global_valid_tokens > 0 else total_valid_tokens

    # Mirror the single-GPU path: backward each chunk immediately to bound retained
    # logits memory, and sum the BF16 head/norm grads in FP32.
    head_grad_params = _dedup_parameters(
        list(ctx.lm_head_gpu.parameters())
        + (list(ctx.norm_gpu.parameters()) if ctx.norm_gpu else [])
    )
    accumulate_head_grads_in_fp32 = (
        shared_state.config.fp32_head_grad and (T - 1) > chunk_size
    )
    head_grad_accum = []
    head_grad_seen = []
    if accumulate_head_grads_in_fp32:
        for parameter in head_grad_params:
            accumulator = torch.zeros_like(parameter, dtype=torch.float32)
            seen = parameter.grad is not None
            if seen:
                accumulator.add_(parameter.grad.float())
                parameter.grad = None
            head_grad_accum.append(accumulator)
            head_grad_seen.append(seen)

    for t_start in range(0, T - 1, chunk_size):
        t_end = min(t_start + chunk_size, T - 1)
        h = hidden_after_norm[:, t_start:t_end, :]
        y = labels_gpu[:, t_start+1:t_end+1]
        logits = ctx.lm_head_gpu(h)
        flat_y = y.reshape(-1)
        flat_logits = logits.reshape(-1, V)

        if ctx.ce_loss is not None:
            per_tok = ctx.ce_loss(flat_logits, flat_y)
            valid = (flat_y != -100)
            loss_chunk = per_tok[valid].sum()
        else:
            loss_chunk = nn.functional.cross_entropy(
                flat_logits.float(), flat_y, ignore_index=-100, reduction='sum'
            )

        total_loss.add_(loss_chunk.detach())
        (loss_chunk / denom).backward(retain_graph=t_end < T - 1)
        del logits, loss_chunk

        if accumulate_head_grads_in_fp32:
            for index, parameter in enumerate(head_grad_params):
                if parameter.grad is not None:
                    head_grad_accum[index].add_(parameter.grad.float())
                    head_grad_seen[index] = True
                    parameter.grad = None

    if accumulate_head_grads_in_fp32:
        for index, parameter in enumerate(head_grad_params):
            # Leave `grad=None` for parameters that never received one, so the slab
            # presence bitmap keeps skipping them. Drop each accumulator as it is
            # consumed; the head one is large.
            if head_grad_seen[index]:
                parameter.grad = head_grad_accum[index].to(parameter.dtype)
            head_grad_accum[index] = None
        head_grad_accum.clear()

    loss_val = (total_loss / total_valid_tokens).item()

    if not torch.isfinite(torch.tensor(loss_val)):
        logger.error(f"Worker {rank}: Loss is {loss_val}!")

    ctx.loss_backward_done.record(ctx.compute_stream)

    grad_hidden = hidden_before_norm.grad.detach()

    # Collect lm_head/norm grads
    if not ctx.head_slab_free.wait(timeout=30.0):
        raise RuntimeError("head slab wait timeout")
    ctx.head_slab_free.clear()
    slab_flat = ctx.head_grad_slab

    with torch.cuda.stream(ctx.grad_stream):
        ctx.grad_stream.wait_event(ctx.loss_backward_done)
        offset = 0
        has_grads = []
        if not shared_state.tied_lm_head:
            offset, head_has_grads = _copy_parameter_grads_to_slab(
                ctx.lm_head_gpu.parameters(), slab_flat, offset, ctx.grad_stream
            )
            has_grads.extend(head_has_grads)
        if ctx.norm_gpu:
            offset, norm_has_grads = _copy_parameter_grads_to_slab(
                ctx.norm_gpu.parameters(), slab_flat, offset, ctx.grad_stream
            )
            has_grads.extend(norm_has_grads)
        ctx.head_slab_event.record(ctx.grad_stream)

    cpu_params = []
    if not shared_state.tied_lm_head:
        cpu_params.extend(shared_state.lm_head.parameters())
    if shared_state.norm:
        cpu_params.extend(shared_state.norm.parameters())
    shapes = [p.shape for p in cpu_params]
    numels = [p.numel() for p in cpu_params]
    grad_queue.put(('head', None, cpu_params, shapes, numels, has_grads))

    del labels_gpu, hidden_after_norm, hidden_before_norm, total_loss

    # Backward through layers
    num_blocks = (num_layers + checkpoint_interval - 1) // checkpoint_interval

    for block_idx in range(num_blocks - 1, -1, -1):
        block_start = block_idx * checkpoint_interval
        block_end = min((block_idx + 1) * checkpoint_interval, num_layers)

        current_checkpoint = checkpoints[block_start]

        recompute_cache = {}
        hidden_recompute = current_checkpoint

        with torch.no_grad():
            for j in range(block_start, block_end):
                buffer_idx = j % 2
                _worker_load_layer_to_buffer_async(j, buffer_idx, ctx, shared_state)
                ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

                with torch.cuda.stream(ctx.compute_stream):
                    _worker_unflatten_to_layer(j, buffer_idx, ctx, shared_state)
                    ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                    group_id = shared_state.layer_to_group[j]
                    gpu_layer = ctx.gpu_layer_templates[group_id][buffer_idx]
                    with _replay_cuda_rng_state(ctx.device, layer_rng_states[j]):
                        out = gpu_layer(hidden_recompute, **layer_kwargs[j])
                    hidden_recompute = out[0] if isinstance(out, tuple) else out

                recompute_cache[j] = hidden_recompute.detach()
                del out

        for i in range(block_end - 1, block_start - 1, -1):
            buffer_idx = i % 2

            if i == block_start:
                layer_input = current_checkpoint.detach().requires_grad_(True)
            else:
                layer_input = recompute_cache[i - 1].requires_grad_(True)

            _worker_load_layer_to_buffer_async(i, buffer_idx, ctx, shared_state)
            ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

            with torch.cuda.stream(ctx.compute_stream):
                _worker_unflatten_to_layer(i, buffer_idx, ctx, shared_state)
                ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                group_id = shared_state.layer_to_group[i]
                gpu_layer = ctx.gpu_layer_templates[group_id][buffer_idx]

                for p in gpu_layer.parameters():
                    p.requires_grad_(True)

                with _replay_cuda_rng_state(ctx.device, layer_rng_states[i]):
                    out = gpu_layer(layer_input, **layer_kwargs[i])
                layer_output = out[0] if isinstance(out, tuple) else out

                grads = torch.autograd.grad(
                    outputs=layer_output,
                    inputs=(layer_input, *gpu_layer.parameters()),
                    grad_outputs=grad_hidden,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_hidden = grads[0].detach()
                param_grads = grads[1:]

                for p, g in zip(gpu_layer.parameters(), param_grads):
                    p.grad = g

                for p in gpu_layer.parameters():
                    p.requires_grad_(False)

                ctx.backward_done_events[buffer_idx].record(ctx.compute_stream)

            _worker_collect_layer_grads_async(i, buffer_idx, ctx, shared_state, grad_queue)

            if i in recompute_cache:
                del recompute_cache[i]
            del layer_input, layer_output, out

        recompute_cache.clear()

    # === BACKWARD THROUGH EMBEDDING ===
    input_ids_gpu = input_ids.to(ctx.device)
    emb_out = ctx.emb_gpu(input_ids_gpu)
    emb_out.backward(grad_hidden)
    ctx.embedding_backward_done.record(ctx.compute_stream)

    if not ctx.embed_slab_free.wait(timeout=30.0):
        raise RuntimeError("embed slab wait timeout")
    ctx.embed_slab_free.clear()
    slab_flat = ctx.embed_grad_slab

    with torch.cuda.stream(ctx.grad_stream):
        ctx.grad_stream.wait_event(ctx.embedding_backward_done)
        _, has_grads = _copy_parameter_grads_to_slab(
            ctx.emb_gpu.parameters(), slab_flat, stream=ctx.grad_stream
        )
        ctx.embed_slab_event.record(ctx.grad_stream)

    cpu_params = list(shared_state.embedding.parameters())
    shapes = [p.shape for p in cpu_params]
    numels = [p.numel() for p in cpu_params]
    grad_queue.put(('embed', None, cpu_params, shapes, numels, has_grads))

    del input_ids_gpu, emb_out
    del mask, cache_position, position_ids, position_embeddings, grad_hidden
    checkpoints.clear()

    bwd_end.record(ctx.compute_stream)
    torch.cuda.synchronize(ctx.device)
    fwd_time = start.elapsed_time(fwd_end) / 1000.0
    bwd_time = fwd_end.elapsed_time(bwd_end) / 1000.0

    return WorkerResult(
        loss_val=loss_val,
        total_tokens=B * T,
        timing={'forward': fwd_time, 'backward': bwd_time, 'total': fwd_time + bwd_time},
        valid_tokens=total_valid_tokens,
    )


def _worker_unflatten_to_layer(layer_idx, buffer_idx, ctx, shared_state):
    """Unflatten GPU buffer to the appropriate layer template parameters."""
    flat = ctx.gpu_flat_buffers[buffer_idx]
    group_id = shared_state.layer_to_group[layer_idx]
    gpu_layer = ctx.gpu_layer_templates[group_id][buffer_idx]

    ctx.compute_stream.wait_event(ctx.template_free_events[buffer_idx])

    offset = 0
    for p in gpu_layer.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view(p.shape))
        offset += numel

    ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)


def _worker_collect_layer_grads_async(layer_idx, buffer_idx, ctx, shared_state, grad_queue):
    """Collect GPU buffer grads to CPU using K-slab pool."""
    slab_idx = ctx.layer_slab_free_list.get()
    slab_flat = ctx.layer_grad_slabs[slab_idx]

    ctx.grad_stream.wait_event(ctx.backward_done_events[buffer_idx])

    group_id = shared_state.layer_to_group[layer_idx]
    gpu_layer = ctx.gpu_layer_templates[group_id][buffer_idx]

    with torch.cuda.stream(ctx.grad_stream):
        _, has_grads = _copy_parameter_grads_to_slab(
            gpu_layer.parameters(), slab_flat, stream=ctx.grad_stream
        )

        ctx.layer_slab_events[slab_idx].record(ctx.grad_stream)
        ctx.template_free_events[buffer_idx].record(ctx.grad_stream)

    grad_queue.put((
        'layer',
        slab_idx,
        shared_state.layer_cpu_params[layer_idx],
        shared_state.layer_param_shapes[layer_idx],
        shared_state.layer_param_numel[layer_idx],
        has_grads,
    ))




def _worker_release_gpu(ctx):
    """Release GPU buffers to free VRAM (for SGLang colocation)."""
    import gc

    # Release flat buffers
    if ctx.gpu_flat_buffers:
        for i in range(len(ctx.gpu_flat_buffers)):
            ctx.gpu_flat_buffers[i] = None
        ctx.gpu_flat_buffers = []

    # Release layer templates
    if ctx.gpu_layer_templates:
        for gid in list(ctx.gpu_layer_templates.keys()):
            for template in ctx.gpu_layer_templates[gid]:
                del template
        ctx.gpu_layer_templates = {}

    # Release GPU modules
    if ctx.emb_gpu is not None:
        del ctx.emb_gpu
        ctx.emb_gpu = None
    if ctx.norm_gpu is not None:
        del ctx.norm_gpu
        ctx.norm_gpu = None
    if ctx.lm_head_gpu is not None:
        del ctx.lm_head_gpu
        ctx.lm_head_gpu = None
    if ctx.rotary_gpu is not None:
        del ctx.rotary_gpu
        ctx.rotary_gpu = None

    # Release grad slabs
    if ctx.layer_grad_slabs:
        ctx.layer_grad_slabs = []
    if ctx.head_grad_slab is not None:
        ctx.head_grad_slab = None
    if ctx.embed_grad_slab is not None:
        ctx.embed_grad_slab = None

    gc.collect()
    torch.cuda.empty_cache()
