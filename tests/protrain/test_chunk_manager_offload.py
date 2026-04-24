"""Tests for the M4.5 chunk-manager offload primitives.

Covers :meth:`ChunkManager.materialize_offload` and the per-param
post-accumulate-grad hooks — the two runtime gaps closed in M4.5. Every
test here runs on GPU (``@pytest.mark.gpu``); there's no meaningful CPU
equivalent because the offload semantics are defined in terms of
``torch.cuda.memory_allocated`` dropping.
"""

from __future__ import annotations

from typing import cast

import pytest

from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    ParamId,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model(hidden: int = 64, n_layers: int = 4):
    """A tiny 4-layer "transformer-ish" model.

    Each layer is one Linear — enough to give the layout builder N_block=4
    and 4 separable param groups. We use nn.ModuleList so the block
    discovery logic in layout.py picks it up as the transformer stack.
    """
    import torch
    from torch import nn

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(hidden, hidden, bias=False)
            self.h = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)]
            )
            self.head = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.embed(x)
            for layer in self.h:
                x = layer(x)
            return self.head(x)

    torch.manual_seed(0)
    return TinyTransformer()


def _build_layout_for(model, S_chunk: int):
    """Build a ChunkLayout where each ``h.{i}`` linear is its own chunk."""
    from axolotl.integrations.protrain.chunk.layout import build_layout

    # Block spans: each h.i is a block. embed and head are unaffiliated.
    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        if name.startswith("h."):
            idx = int(name.split(".")[1])
            block_spans.setdefault(cast(BlockId, idx), []).append(
                cast(ParamId, name)
            )

    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    return build_layout(model, exec_order, S_chunk, block_spans)


def _build_chunk_manager(
    model, n_persist: int, S_chunk: int, n_buffer: int | None = None
):
    """Assemble a :class:`ChunkManager` from scratch for offload tests."""
    import torch

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    layout = _build_layout_for(model, S_chunk)
    if n_buffer is None:
        n_buffer = max(2, min(4, layout.N_chunk - n_persist))
    host = PinnedHostMemory(n_buffer=n_buffer, S_chunk=layout.S_chunk)
    pool = BufferPool(
        n_buffer=n_buffer,
        S_chunk=layout.S_chunk,
        pinned_host=host,
        device=torch.device("cuda"),
    )
    mgr = ChunkManager(
        model=model,
        layout=layout,
        n_persist=n_persist,
        buffer_pool=pool,
        cpu_optim=None,
        gpu_optim=None,
        device=torch.device("cuda"),
    )
    return mgr, layout, pool, host


# ---------------------------------------------------------------------------
# Test 1: materialize_offload releases GPU memory
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_materialize_offload_frees_gpu_memory() -> None:
    """Non-persistent chunks' param bytes should leave the GPU after offload."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    # Tiny 4-layer model, one chunk per layer when S_chunk is sized so
    # each layer exactly fills a chunk. hidden=64, fp32 -> 64*64*4 = 16 KB
    # per layer. Set S_chunk at 32 KB so each block lands in its own chunk.
    hidden = 64
    n_layers = 4
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")

    # Per-layer weight bytes: 64 * 64 * 4 = 16 KB. Pick S_chunk above that
    # per-param size, but below two-params-worth so each block gets its
    # own chunk.
    per_layer_bytes = hidden * hidden * 4
    S_chunk = per_layer_bytes + 4096  # 16 KB + 4 KB headroom

    mgr, layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)
    # Expect N_chunk >= n_layers + 1 (+1 for embed / head grouping).
    n_non_persist = layout.N_chunk - 1
    assert n_non_persist >= 2, (
        f"test setup: expected >=2 non-persistent chunks, got {n_non_persist} "
        f"(N_chunk={layout.N_chunk})"
    )

    # Record baseline GPU memory before offload.
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    freed = mgr.materialize_offload()

    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()

    # Expect at least (n_non_persist) * per_layer_bytes to be freed —
    # the non-persistent chunks' params are now on pinned CPU memory.
    # We tolerate some slack because embed / head may land in the
    # persistent chunk and not count toward the saved bytes.
    expected_min_freed = (n_non_persist - 1) * per_layer_bytes
    delta = before - after
    assert delta >= expected_min_freed, (
        f"expected >= {expected_min_freed} bytes freed, got {delta} "
        f"(before={before}, after={after}, reported_freed={freed})"
    )
    assert freed >= expected_min_freed, (
        f"materialize_offload reported freed={freed}, expected "
        f">= {expected_min_freed}"
    )

    # Cleanup.
    mgr.uninstall()
    host.close()
    # Silence unused-var warnings — pool is referenced by mgr.
    del pool


# ---------------------------------------------------------------------------
# Test 2: gather / offload rebinds param.data correctly
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_gather_rebinds_param_data() -> None:
    """After gather() the param.data is a non-empty GPU view; offload() empties it."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096

    mgr, layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)
    mgr.materialize_offload()

    # Pick any non-persistent chunk id and confirm its params are empty.
    non_persist = sorted(mgr._non_persistent_ids)
    assert non_persist, "need at least one non-persistent chunk for this test"
    cid = non_persist[0]
    param_ids = layout.chunks[int(cid)]

    # Before gather: every non-persistent param has an empty .data tensor.
    for pid in param_ids:
        param = dict(model.named_parameters())[str(pid)]
        assert param.data.numel() == 0, (
            f"param {pid} not offloaded: .data.numel()={param.data.numel()}"
        )

    # Gather and check the params are now GPU-resident with the right shape.
    mgr.gather(cid)
    for pid in param_ids:
        param = dict(model.named_parameters())[str(pid)]
        assert param.data.numel() > 0, (
            f"param {pid} still empty after gather: {param.data.shape}"
        )
        assert param.data.device.type == "cuda", (
            f"param {pid} not on cuda after gather: {param.data.device}"
        )
        # Shape must match the original.
        assert tuple(param.data.shape) == (hidden, hidden), (
            f"param {pid} has wrong shape after gather: {param.data.shape}"
        )

    # Offload again — params should return to the empty placeholder.
    mgr.offload(cid)
    for pid in param_ids:
        param = dict(model.named_parameters())[str(pid)]
        assert param.data.numel() == 0, (
            f"param {pid} not emptied after offload: .data.numel()={param.data.numel()}"
        )

    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Test 2b: materialize_offload under mixed-dtype chunks (BUG 2 regression)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_materialize_offload_mixed_dtype() -> None:
    """Chunks holding a mix of fp16 + fp32 params must not hit ``view`` alignment.

    Before the fix (BUG 2), a chunk containing fp16 Linear weights
    followed by fp32 LayerNorm scales tripped
    ``RuntimeError: offset is not aligned``: the per-param byte offset
    landed on an odd multiple of 2 after the first fp16 param, and
    ``byte_view.view(torch.float32)`` rejected the unaligned view.

    The fix pads each slot's starting offset up to a multiple of the
    param's ``element_size``. This test builds a mixed-dtype module,
    forces everything into a single non-persistent chunk, and verifies
    materialize + gather both succeed and that ``param.data.dtype`` is
    preserved across the round trip.
    """
    pytest.importorskip("torch")
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    class MixedDtype(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # fp16 Linear + fp32 LayerNorm — the exact pattern Llama
            # emits inside each transformer block when attention
            # weights are fp16 but RMSNorm scales stay fp32. Put them
            # inside a ModuleList so layout.build_layout picks them up
            # as a single "block".
            attn = nn.Linear(32, 32, bias=False).half()
            # An fp32 tensor deliberately ordered AFTER the fp16 one
            # so the running byte offset lands at an odd 2-byte
            # boundary (32*32*2=2048 bytes — actually aligned, but
            # add an odd number of fp16 bytes to force misalignment).
            extra_fp16 = nn.Linear(1, 32, bias=False).half()  # 64 bytes, /=2
            norm = nn.LayerNorm(32).float()  # fp32 weight+bias
            layer = nn.Module()
            layer.attn = attn  # type: ignore[attr-defined]
            layer.extra = extra_fp16  # type: ignore[attr-defined]
            layer.norm = norm  # type: ignore[attr-defined]

            def fwd(x: torch.Tensor) -> torch.Tensor:
                y = layer.attn(x.half())
                y = layer.norm(y.float())
                return y

            layer.forward = fwd  # type: ignore[assignment]
            self.h = nn.ModuleList([layer])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.h[0](x)

    torch.manual_seed(0)
    model = MixedDtype().to("cuda")

    # Large enough S_chunk so the whole ModuleList lands in one chunk.
    S_chunk = 1 << 16  # 64 KB — fits everything
    mgr, layout, pool, host = _build_chunk_manager(
        model, n_persist=0, S_chunk=S_chunk, n_buffer=2
    )

    # Sanity: before the fix, this raised RuntimeError inside
    # ``byte_view.view(torch.float32)``.
    freed = mgr.materialize_offload()
    assert freed > 0, "expected some bytes freed from mixed-dtype chunk"

    # After offload, each param.data should be the empty GPU placeholder
    # with the ORIGINAL dtype preserved.
    expected_dtypes = {
        "h.0.attn.weight": torch.float16,
        "h.0.extra.weight": torch.float16,
        "h.0.norm.weight": torch.float32,
        "h.0.norm.bias": torch.float32,
    }
    for name, param in model.named_parameters():
        assert param.data.dtype == expected_dtypes[name], (
            f"{name} dtype {param.data.dtype} != expected "
            f"{expected_dtypes[name]} after offload"
        )
        assert param.data.numel() == 0, (
            f"{name} still has non-empty .data after offload: {param.data.shape}"
        )

    # Gather every non-persistent chunk and verify dtype+shape survive
    # the round trip without alignment errors.
    for cid_int in sorted(mgr._non_persistent_ids):
        cid = cast(ChunkId, cid_int)
        mgr.gather(cid)

    for name, param in model.named_parameters():
        assert param.data.dtype == expected_dtypes[name], (
            f"{name} dtype changed after gather: {param.data.dtype}"
        )
        assert param.data.device.type == "cuda", (
            f"{name} landed on {param.data.device} after gather"
        )
        assert param.data.numel() > 0, (
            f"{name} still empty after gather"
        )

    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Test 2c: param.data returns to empty-GPU placeholder between iterations (BUG 4)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_param_data_empty_between_iters() -> None:
    """After CPU Adam step, ``param.data`` must be a zero-element GPU tensor.

    BUG 4: before the fix, ``_ensure_cpu_grads_attached`` repointed
    ``param.data`` at the CPU shard for the CPU Adam step and nothing
    repointed it back. Between end-of-iter and start-of-next-iter,
    ``param.data`` was a CPU tensor — any intermediate code reading
    ``.data`` (``clip_grad_norm_``, Trainer metric hooks, checkpoint
    save) saw CPU where GPU was expected.

    The fix registers a ``post_step`` callback on ``step_async`` that
    repoints ``.data`` back to ``_empty_placeholder(dtype)`` after the
    CPU Adam step resolves. This test runs a full fwd+bwd+step cycle
    and asserts post-step that every non-persistent param has
    ``param.data.numel() == 0`` AND ``param.data.device.type == "cuda"``.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")
    # DeepSpeedCPUAdam compiles a CUDA extension lazily — import
    # success doesn't imply it can build. Probe cheaply so the test
    # gracefully skips in envs where nvcc↔torch CUDA versions
    # disagree (the runtime path handles the missing adapter; this
    # test just isolates BUG 4's repointing semantics).
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        _probe = DeepSpeedCPUAdam(
            [torch.nn.Parameter(torch.zeros(1))], lr=1e-4
        )
        del _probe
    except Exception:  # noqa: BLE001
        pytest.skip("DeepSpeedCPUAdam unavailable — BUG 4 path requires CPU optim")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    S_chunk = hidden * hidden * 4 + 4096

    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    layout_probe = _build_layout_for(model, S_chunk)
    n_non_persist = layout_probe.N_chunk - 1
    mgr, layout, pool, host = _build_chunk_manager(
        model, n_persist=1, S_chunk=S_chunk, n_buffer=n_non_persist
    )
    mgr.materialize_offload()

    # Build a CPU Adam adapter so the BUG 4 repoint callback fires.
    from axolotl.integrations.protrain.chunk.optim import CpuFusedAdamAdapter

    cpu_params_per_chunk: dict = {}
    for cid_int in sorted(mgr._non_persistent_ids):
        params = [
            dict(model.named_parameters())[str(pid)]
            for pid in layout.chunks[int(cid_int)]
            if str(pid) in dict(model.named_parameters())
        ]
        if params:
            cpu_params_per_chunk[cid_int] = params

    cpu_optim = CpuFusedAdamAdapter(
        params_per_chunk=cpu_params_per_chunk, lr=1e-4
    )
    mgr.cpu_optim = cpu_optim

    # Drive one fwd+bwd+step cycle. Gather everything manually (no
    # scheduler in this bare test).
    for cid_int in range(layout.N_chunk):
        mgr.gather(cast(ChunkId, cid_int))

    x = torch.randn(2, hidden, device="cuda")
    y = model(x)
    loss = y.sum()
    loss.backward()

    # The per-param hooks fired step_async on the CPU optim. Block
    # until every future has resolved — the post_step callback runs
    # inside that wait, so after this line param.data MUST be the
    # empty GPU placeholder.
    mgr.wait_cpu_optim_all()

    for cid_int in sorted(mgr._non_persistent_ids):
        cid = cast(ChunkId, cid_int)
        slots = mgr._cpu_slots.get(cid, [])
        for slot in slots:
            param = dict(model.named_parameters())[str(slot.param_id)]
            if not param.requires_grad:
                continue
            assert param.data.numel() == 0, (
                f"non-persistent param {slot.param_id}.data non-empty "
                f"between iters: shape={param.data.shape} "
                f"device={param.data.device}"
            )
            assert param.data.device.type == "cuda", (
                f"non-persistent param {slot.param_id}.data on "
                f"{param.data.device} between iters (BUG 4 regression)"
            )

    cpu_optim.shutdown()
    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Test 3: per-param grad hooks fire and drain to CPU shards
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_grad_offload_hook_fires() -> None:
    """After backward, the CPU grad shards hold the correct grad values.

    We compare against a reference run of the same model WITHOUT ProTrain
    wrapping — both runs should produce identical grads on identical
    inputs, with the ProTrain run's grads landing on the CPU shards
    instead of ``param.grad``.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    S_chunk = hidden * hidden * 4 + 4096

    # ---- Reference run: plain PyTorch -----------------------------------
    torch.manual_seed(7)
    ref_model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    x = torch.randn(2, hidden, device="cuda")
    y_ref = ref_model(x)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    ref_grads = {
        name: p.grad.detach().clone().cpu()
        for name, p in ref_model.named_parameters()
    }

    # ---- ProTrain-wrapped run ------------------------------------------
    torch.manual_seed(7)  # same init → same params
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    # n_buffer large enough to gather every non-persistent chunk at once —
    # the scheduler normally rotates through a smaller pool, but this
    # test runs without the scheduler and needs every param resident
    # simultaneously for the forward pass to succeed.
    layout_probe = _build_layout_for(model, S_chunk)
    n_non_persist = layout_probe.N_chunk - 1
    mgr, layout, pool, host = _build_chunk_manager(
        model, n_persist=1, S_chunk=S_chunk, n_buffer=n_non_persist
    )
    mgr.materialize_offload()

    # Gather all non-persistent chunks so the forward has GPU-resident
    # params. Without the scheduler pumping this (it's not installed in
    # this bare-metal test), we drive it manually.
    for cid_int in range(layout.N_chunk):
        mgr.gather(cast(ChunkId, cid_int))

    # Forward / backward with the SAME input as the reference.
    y = model(x)
    loss = y.sum()
    loss.backward()

    # The per-param hook should have offloaded every non-persistent
    # param's .grad to the pinned-CPU shard. After the last param in a
    # chunk fires its hook, :meth:`_ensure_cpu_grads_attached` repoints
    # ``param.grad`` at the CPU shard so the optimizer adapter can consume
    # it — so ``param.grad`` is either None (draining in progress) or a
    # CPU tensor (fully drained), but NEVER a GPU tensor.
    for cid_int in sorted(mgr._non_persistent_ids):
        cid = cast(ChunkId, cid_int)
        slots = mgr._cpu_slots.get(cid, [])
        for slot in slots:
            param = dict(model.named_parameters())[str(slot.param_id)]
            if not param.requires_grad:
                continue
            # Hook should have drained the GPU grad. ``param.grad`` is
            # either None or a CPU tensor; it must NOT be a GPU tensor.
            if param.grad is not None:
                assert param.grad.device.type == "cpu", (
                    f"non-persistent param {slot.param_id} still has a GPU "
                    f".grad of shape {param.grad.shape}; hook did not "
                    "drain to CPU"
                )
            # The CPU grad shard must match the reference grad.
            ref = ref_grads[str(slot.param_id)]
            got = slot.cpu_grad
            assert got is not None, (
                f"slot {slot.param_id}: cpu_grad shard was not allocated"
            )
            assert torch.allclose(ref, got.cpu().float(), atol=1e-4, rtol=1e-4), (
                f"CPU grad for {slot.param_id} diverged from reference: "
                f"max abs diff = {(ref - got.cpu().float()).abs().max().item()}"
            )

    # Persistent-chunk params keep their GPU grads (not hook-drained).
    for cid_int in sorted(mgr._persistent_ids):
        cid = cast(ChunkId, cid_int)
        for pid in layout.chunks[int(cid)]:
            param = dict(model.named_parameters())[str(pid)]
            if not param.requires_grad:
                continue
            assert param.grad is not None, (
                f"persistent param {pid} unexpectedly had grad drained"
            )
            ref = ref_grads[str(pid)]
            assert torch.allclose(
                ref, param.grad.cpu().float(), atol=1e-4, rtol=1e-4
            ), (
                f"persistent-chunk grad for {pid} diverged from reference"
            )

    mgr.uninstall()
    host.close()
    del pool
