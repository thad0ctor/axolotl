"""Distributed-path coverage for :meth:`ChunkManager.reduce_grads_and_offload`.

The M6 multi-GPU test (``test_multi_gpu_7b.py``) sets
``skip_internal_grad_reduce=True`` because it composes the protrain'd
module inside ``DistributedDataParallel`` — DDP's bucketed allreduce
owns cross-rank grad sync there. That means the M6 test NEVER
exercises:

* The per-param ``all_reduce`` branch inside
  :meth:`ChunkManager._make_grad_offload_hook._hook` (non-persistent
  chunks).
* The persistent-chunk ``all_reduce`` branch inside
  :meth:`ChunkManager.reduce_grads_and_offload` (manager.py:644-655).

This module fills that gap using a tiny 2-rank gloo cluster — gloo on
CPU is sufficient for correctness coverage of the reduction math, and
it's the only backend we can reasonably run inside a pytest ``mp.spawn``
without requiring NCCL + multiple GPUs reserved for the test.
"""

from __future__ import annotations

import os
from typing import cast

import pytest

from axolotl.integrations.protrain.types import BlockId, ChunkId, ParamId


# ---------------------------------------------------------------------------
# Helpers (must be top-level so ``mp.spawn`` can pickle them)
# ---------------------------------------------------------------------------


def _tiny_cpu_model():
    """A two-param module: a single Linear, used to exercise a 2-param chunk.

    CPU-only on purpose — the gloo backend does not use CUDA, and this
    keeps the spawned subprocesses free of any GPU resource requirement.
    """
    import torch
    from torch import nn

    torch.manual_seed(0)
    layer = nn.Linear(4, 4, bias=True)
    # Bundle in a ModuleList so ``discover_blocks`` picks it up cleanly.
    model = nn.Module()
    model.h = nn.ModuleList([layer])  # type: ignore[attr-defined]
    return model


def _build_chunk_manager_cpu(model, n_persist: int):
    """Assemble a :class:`ChunkManager` with a CPU-device buffer pool.

    The pool's device is set to CPU so the manager can function
    end-to-end without CUDA. The offload / gather path still exercises
    the same byte-level operations the GPU path does; only the physical
    copy engine is different.
    """
    import torch

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    # Treat the single Linear as block 0.
    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        block_spans.setdefault(cast(BlockId, 0), []).append(cast(ParamId, name))

    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    # S_chunk large enough to land all params in ONE chunk so the test
    # exercises a 2-param reduction cleanly.
    S_chunk = 1 << 14  # 16 KB
    layout = build_layout(model, exec_order, S_chunk, block_spans)
    # BufferPool pins its host region; pinning on a CPU-only test host
    # still works because pin_memory is a property of host memory, not
    # of an active CUDA context. But if no CUDA is reachable at all,
    # PyTorch quietly falls back to pageable. For the distributed test
    # we don't need pinning.
    host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
    pool = BufferPool(
        n_buffer=1,
        S_chunk=layout.S_chunk,
        pinned_host=host,
        device=torch.device("cpu"),
    )
    mgr = ChunkManager(
        model=model,
        layout=layout,
        n_persist=n_persist,
        buffer_pool=pool,
        cpu_optim=None,
        gpu_optim=None,
        device=torch.device("cpu"),
    )
    return mgr, layout, pool, host


def _worker_reduce_grads_and_offload(rank: int, world_size: int, tmpdir: str) -> None:
    """Child process body for the gloo test.

    Plants rank-specific grads on every param — rank ``r`` writes
    ``r`` into every element — then exercises the distributed path and
    asserts each CPU grad shard holds the cross-rank MEAN (which is
    ``(0 + 1 + ... + (W-1)) / W``).

    The persistent path exercises :meth:`reduce_grads_and_offload`'s
    ``all_reduce(op=AVG)`` branch; to also cover the non-persistent
    per-param-hook reduce branch we run the manager with
    ``n_persist == 0`` and fire the grad hooks by invoking backward.
    Each of the two param types gets its own assertion.
    """
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous",
        rank=rank,
        world_size=world_size,
    )

    try:
        # ---- Path A: NON-persistent chunk — per-param grad hook -----
        # n_persist = 0 so the sole chunk is non-persistent and runs
        # through the materialize_offload / _offload_grad hook path.
        torch.manual_seed(0)
        model_a = _tiny_cpu_model()
        mgr_a, layout_a, pool_a, host_a = _build_chunk_manager_cpu(
            model_a, n_persist=0
        )
        mgr_a.materialize_offload()

        # Gather the chunk so param.data is GPU-... er, CPU-buffer-
        # resident with the right shape, then plant rank-specific grads.
        for cid_int in range(layout_a.N_chunk):
            mgr_a.gather(cast(ChunkId, cid_int))

        expected_mean = sum(range(world_size)) / float(world_size)

        # Drive backward: each rank emits a loss whose grad is a
        # constant ``rank`` across every param element. We assemble
        # this by hand rather than via loss.backward() so we don't
        # depend on the model's forward matching shape on CPU:
        # manually set param.grad then call the hook.
        for name, p in model_a.named_parameters():
            p.grad = torch.full_like(p.data, float(rank))
            # Fire the post-accumulate hook manually — in real
            # training PyTorch fires it at the end of backward. For
            # the test, we want explicit control over when the
            # all_reduce happens.
            # find the hook: we stored the handles, but each hook is a
            # closure over a slot. Simplest path: re-register by
            # iterating mgr._cpu_slots and call the hook directly.

        # Walk the slots and invoke the hooks directly.
        for cid_int in sorted(mgr_a._non_persistent_ids):
            cid = cast(ChunkId, cid_int)
            slots = mgr_a._cpu_slots.get(cid, [])
            for slot in slots:
                param = dict(model_a.named_parameters())[str(slot.param_id)]
                if not param.requires_grad:
                    continue
                # Re-build and fire the same hook the manager would
                # have registered (the manager kept the handles; we
                # just don't have a clean "run me" entry point that
                # doesn't also go through autograd). This path is
                # what installs all_reduce + cpu_grad.copy_ +
                # param.grad = None.
                hook = mgr_a._make_grad_offload_hook(cid, slot)
                hook(param)

        # Every CPU grad shard must now hold the cross-rank MEAN.
        for cid_int in sorted(mgr_a._non_persistent_ids):
            cid = cast(ChunkId, cid_int)
            slots = mgr_a._cpu_slots.get(cid, [])
            for slot in slots:
                assert slot.cpu_grad is not None, (
                    f"rank {rank}: slot {slot.param_id} has no cpu_grad"
                )
                obs = slot.cpu_grad.detach().cpu().float()
                assert torch.allclose(
                    obs,
                    torch.full_like(obs, float(expected_mean)),
                    atol=1e-5,
                    rtol=1e-5,
                ), (
                    f"rank {rank}: non-persistent CPU grad shard for "
                    f"{slot.param_id} should be uniform {expected_mean}, "
                    f"got min={obs.min().item()} max={obs.max().item()}"
                )

        mgr_a.uninstall()
        host_a.close()
        del pool_a

        # ---- Path B: PERSISTENT chunk — manager.py:644 branch -------
        # n_persist = N_chunk so every chunk stays resident and
        # reduce_grads_and_offload takes the persistent-chunk branch
        # (the per-param all_reduce(AVG) at manager.py:644-655).
        torch.manual_seed(0)
        model_b = _tiny_cpu_model()
        mgr_b, layout_b, pool_b, host_b = _build_chunk_manager_cpu(
            model_b, n_persist=1
        )
        # Force every chunk persistent — the helper built the manager
        # with ``n_persist=1`` but if the layout produced >1 chunk we
        # need to expand. This model's 2 params fit in one chunk.
        assert layout_b.N_chunk == 1, (
            f"test setup expects a single-chunk layout, got "
            f"N_chunk={layout_b.N_chunk}"
        )

        # Plant rank-specific grads directly on the param objects.
        for name, p in model_b.named_parameters():
            p.grad = torch.full_like(p.data, float(rank))

        for cid_int in sorted(mgr_b._persistent_ids):
            cid = cast(ChunkId, cid_int)
            mgr_b.reduce_grads_and_offload(cid)

        # After the AVG all_reduce, every persistent-chunk param.grad
        # should be ``expected_mean`` across all elements.
        for name, p in model_b.named_parameters():
            assert p.grad is not None, (
                f"rank {rank}: persistent param {name} grad cleared"
            )
            obs = p.grad.detach().cpu().float()
            assert torch.allclose(
                obs,
                torch.full_like(obs, float(expected_mean)),
                atol=1e-5,
                rtol=1e-5,
            ), (
                f"rank {rank}: persistent param {name} grad should be "
                f"uniform {expected_mean}, got min={obs.min().item()} "
                f"max={obs.max().item()}"
            )

        mgr_b.uninstall()
        host_b.close()
        del pool_b

    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu  # carries @mark.gpu because the wider test suite pairs
# "slow" with "gpu" for the integration lane; the test itself uses gloo
# (CPU-only) but we want it to run in the same slot as the other
# distributed-composition tests.
def test_reduce_grads_and_offload_distributed(tmp_path) -> None:
    """2-rank gloo test covering the per-rank grad-reduce paths.

    Both the persistent branch of
    :meth:`ChunkManager.reduce_grads_and_offload` and the non-persistent
    per-param-hook ``all_reduce`` branch of
    :meth:`ChunkManager._make_grad_offload_hook` should produce the
    cross-rank MEAN when run under a 2-rank gloo process group. We
    plant rank 0's grads as 0.0 and rank 1's grads as 1.0, then check
    every CPU grad shard on every rank reads 0.5 after reduction.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    # Each rank writes a rendezvous file under tmpdir; the gloo init
    # method points at the same file so the subprocesses can find
    # each other without depending on a free TCP port.
    mp.spawn(
        _worker_reduce_grads_and_offload,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )


# ---------------------------------------------------------------------------
# M7 sharded-path coverage (gloo, CPU-only, 2-rank)
# ---------------------------------------------------------------------------


def _worker_zero3_sharded_roundtrip(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """2-rank gloo test: gather → fake backward → reduce_scatter → step.

    Builds a :class:`ChunkManager` with ``zero3_shard=True`` on a CPU
    device (gloo backend does not need CUDA). Exercises the full
    sharded round-trip:

    1. ``materialize_offload()`` partitions the chunk's bytes across
       ranks. Each rank only holds ``shard_bytes`` of the full chunk.
    2. ``gather()`` runs ``all_gather_into_tensor`` to reconstruct the
       full chunk on each rank's pool buffer. Verify the reconstructed
       bytes match the original param data across ranks.
    3. Plant rank-specific grads, call ``reduce_grads_and_offload()``.
       The reduce_scatter output on rank ``r`` must equal the mean
       grad in rank ``r``'s slice of the full chunk.

    The test skips if gloo doesn't support the needed collectives on
    the installed torch version.
    """
    import os as _os
    import torch
    import torch.distributed as dist

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import (
        PinnedHostMemory,
    )
    from axolotl.integrations.protrain.types import BlockId, ChunkId, ParamId

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29545")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-zero3",
        rank=rank,
        world_size=world_size,
    )

    try:
        # Tiny model: one fp16 Linear layer — 4-in, 4-out + bias,
        # enough to stress the byte-slicing logic.
        torch.manual_seed(0)  # SAME seed on every rank — fresh-init
        # bytes are identical across ranks before training.
        from torch import nn
        layer = nn.Linear(4, 4, bias=True).half()
        model = nn.Module()
        model.h = nn.ModuleList([layer])  # type: ignore[attr-defined]

        # Layout: single chunk holding both params.
        block_spans: dict = {}
        for name, _p in model.named_parameters():
            block_spans.setdefault(BlockId(0), []).append(ParamId(name))  # type: ignore[index]
        exec_order = [ParamId(n) for n, _ in model.named_parameters()]
        S_chunk = 1 << 14
        layout = build_layout(model, exec_order, S_chunk, block_spans)

        host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
        pool = BufferPool(
            n_buffer=1,
            S_chunk=layout.S_chunk,
            pinned_host=host,
            device=torch.device("cpu"),
        )

        # Snapshot the original param bytes BEFORE materialize_offload
        # so we can compare the gathered output against the truth.
        pre_data = {
            str(name): p.detach().clone().cpu()
            for name, p in model.named_parameters()
        }

        # zero3_shard=True + world_size=2 should activate the sharded
        # path on the single chunk.
        mgr = ChunkManager(
            model=model,
            layout=layout,
            n_persist=0,
            buffer_pool=pool,
            cpu_optim=None,
            gpu_optim=None,
            device=torch.device("cpu"),
            world_size=world_size,
            rank=rank,
            zero3_shard=True,
        )
        try:
            mgr.materialize_offload()
        except RuntimeError as exc:
            # gloo + older torch may not support all_gather_into_tensor
            # on CPU tensors; if construction itself works but we can't
            # exercise the sharded collective, skip.
            if "gloo" in str(exc).lower():
                _os.makedirs(tmpdir, exist_ok=True)
                with open(_os.path.join(tmpdir, f"rank{rank}.skip"), "w") as f:
                    f.write(f"gloo-unsupported: {exc}\n")
                return
            raise

        # (1) Invariant: chunk 0 is sharded.
        assert mgr.sharded_chunk_ids() == [ChunkId(0)], (
            f"rank {rank}: expected chunk 0 to be sharded, got "
            f"{mgr.sharded_chunk_ids()}"
        )
        my_shard_bytes = mgr.shard_bytes_for(ChunkId(0))
        assert my_shard_bytes > 0, (
            f"rank {rank}: shard_bytes is 0 — sharding not engaged"
        )

        # (2) Gather should reconstruct identical full chunks on every
        # rank. We verify this by reading back the gathered param.data
        # bytes and comparing against the pre-offload snapshot.
        try:
            mgr.gather(ChunkId(0))
        except RuntimeError as exc:
            if "not implemented" in str(exc).lower() or "nccl" in str(exc).lower():
                # gloo doesn't support all_gather_into_tensor on this
                # build — skip the round-trip test body but let the
                # materialize_offload/sharding invariant above stand.
                with open(_os.path.join(tmpdir, f"rank{rank}.skip"), "w") as f:
                    f.write(f"gloo-collective-unsupported: {exc}\n")
                return
            raise

        for name, p in model.named_parameters():
            snap = pre_data[str(name)]
            # param.data after gather is a view into the pool buffer;
            # bytes should match the original.
            assert torch.allclose(p.data.cpu().float(), snap.float(), atol=0.0), (
                f"rank {rank}: after sharded gather, param '{name}' does "
                f"not match pre-offload snapshot"
            )

        # (3) Plant rank-specific grads on every param, call
        # reduce_grads_and_offload, verify the shard grad holds the
        # MEAN across ranks (AVG reduction).
        for _n, p in model.named_parameters():
            p.grad = torch.full_like(p.data, float(rank))

        mgr.reduce_grads_and_offload(ChunkId(0))

        # The rank's CPU shard grad, reinterpreted as the region's
        # dtype (fp16 for this homogeneous chunk), should be uniformly
        # (0 + 1 + ... + W-1) / W. Homogeneous chunks produce a single
        # :class:`_DtypeRegion` carrying the whole chunk.
        expected_mean = sum(range(world_size)) / float(world_size)
        shard_state = mgr._chunk_shards[ChunkId(0)]
        assert len(shard_state.regions) == 1, (
            f"rank {rank}: homogeneous chunk should produce one dtype "
            f"region, got {len(shard_state.regions)}"
        )
        region0 = shard_state.regions[0]
        obs = region0.shard_param.grad.detach().cpu().float()  # type: ignore[union-attr]
        assert torch.allclose(
            obs,
            torch.full_like(obs, float(expected_mean)),
            atol=1e-3,
            rtol=1e-3,
        ), (
            f"rank {rank}: sharded reduce_scatter grad should be "
            f"uniform {expected_mean}, got min={obs.min().item()} "
            f"max={obs.max().item()}"
        )

        mgr.uninstall()
        host.close()

    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.gpu  # paired with the other distributed tests' marks
def test_zero3_sharded_roundtrip_2rank(tmp_path) -> None:
    """2-rank gloo test for the M7 ZeRO-3 sharded round-trip.

    Each rank (a) holds only its shard on CPU after materialize_offload,
    (b) reconstructs the full chunk via all_gather on gather, and
    (c) receives its slice of the AVG-reduced grad via reduce_scatter
    on reduce_grads_and_offload.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_zero3_sharded_roundtrip,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    # If any rank wrote a ``.skip`` file due to unsupported collectives,
    # downgrade to a skip rather than a fail.
    skip_files = list(tmp_path.glob("rank*.skip"))
    if skip_files:
        reasons = [f.read_text().strip() for f in skip_files]
        pytest.skip(f"gloo does not support required collective(s): {reasons}")


# ---------------------------------------------------------------------------
# M7 follow-up: mixed-dtype sharded round-trip (gloo, CPU-only, 2-rank)
# ---------------------------------------------------------------------------


def _worker_zero3_sharded_roundtrip_mixed_dtype(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """2-rank gloo test: sharded round-trip over a fp16 + fp32 chunk.

    Builds a model with ``nn.Linear(16, 16, dtype=fp16)`` followed by
    ``nn.LayerNorm(16, dtype=fp32)``, packs both into one chunk, and
    drives the sharded gather/reduce_scatter path. The dtype-regions
    machinery should produce 2 regions (one fp16, one fp32); each
    region gets its own collective. After gather every param
    reconstructs bit-exactly; after reduce_scatter each rank's
    region-level shard grad is the cross-rank AVG of the planted
    grads.
    """
    import os as _os

    import torch
    import torch.distributed as dist

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import (
        PinnedHostMemory,
    )
    from axolotl.integrations.protrain.types import BlockId, ChunkId, ParamId

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29547")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-zero3-mixed",
        rank=rank,
        world_size=world_size,
    )

    try:
        torch.manual_seed(0)  # SAME seed on every rank — fresh-init
        # bytes are identical before training.
        from torch import nn

        # fp16 Linear + fp32 LayerNorm in one module, packed into a
        # single chunk. Sizes chosen so both region kinds carry
        # non-trivial byte counts: Linear = 16*16+16 = 272 params *
        # 2 bytes = 544 B; LayerNorm = 16+16 = 32 params * 4 bytes =
        # 128 B.
        class _MixedLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(16, 16, bias=True).to(torch.float16)
                self.norm = nn.LayerNorm(16).to(torch.float32)

        layer = _MixedLayer()
        model = nn.Module()
        model.h = nn.ModuleList([layer])  # type: ignore[attr-defined]

        block_spans: dict = {}
        for name, _p in model.named_parameters():
            block_spans.setdefault(BlockId(0), []).append(ParamId(name))  # type: ignore[index]
        exec_order = [ParamId(n) for n, _ in model.named_parameters()]
        S_chunk = 1 << 14
        layout = build_layout(model, exec_order, S_chunk, block_spans)

        host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
        pool = BufferPool(
            n_buffer=1,
            S_chunk=layout.S_chunk,
            pinned_host=host,
            device=torch.device("cpu"),
        )

        pre_data = {
            str(name): p.detach().clone().cpu()
            for name, p in model.named_parameters()
        }

        mgr = ChunkManager(
            model=model,
            layout=layout,
            n_persist=0,
            buffer_pool=pool,
            cpu_optim=None,
            gpu_optim=None,
            device=torch.device("cpu"),
            world_size=world_size,
            rank=rank,
            zero3_shard=True,
        )

        try:
            mgr.materialize_offload()
        except RuntimeError as exc:
            if "gloo" in str(exc).lower():
                _os.makedirs(tmpdir, exist_ok=True)
                with open(
                    _os.path.join(tmpdir, f"rank{rank}.skip"), "w"
                ) as f:
                    f.write(f"gloo-unsupported: {exc}\n")
                return
            raise

        # (1) Mixed-dtype chunk must actually shard — no silent
        # fall-back to replicated. Post-followup ``materialize_offload``
        # produces a shard state with 2 regions (fp16 + fp32).
        assert mgr.sharded_chunk_ids() == [ChunkId(0)], (
            f"rank {rank}: mixed-dtype chunk should engage sharded path"
        )
        shard_state = mgr._chunk_shards[ChunkId(0)]
        # Expect two regions: fp16 (Linear) and fp32 (LayerNorm). Order
        # follows named_parameters() insertion order — Linear first,
        # then LayerNorm.
        assert len(shard_state.regions) == 2, (
            f"rank {rank}: expected 2 dtype regions (fp16 + fp32), "
            f"got {len(shard_state.regions)}"
        )
        dtypes_seen = {r.dtype for r in shard_state.regions}
        assert dtypes_seen == {torch.float16, torch.float32}, (
            f"rank {rank}: unexpected region dtypes: {dtypes_seen}"
        )

        # (2) Gather should reconstruct every param bit-exactly on
        # every rank. Because materialize_offload ran the initial
        # shard copy from full-chunk CPU bytes, and all ranks started
        # from identical weights, a successful all_gather produces
        # identical full chunks on every rank.
        try:
            mgr.gather(ChunkId(0))
        except RuntimeError as exc:
            if "not implemented" in str(exc).lower() or "nccl" in str(exc).lower():
                with open(
                    _os.path.join(tmpdir, f"rank{rank}.skip"), "w"
                ) as f:
                    f.write(f"gloo-collective-unsupported: {exc}\n")
                return
            raise

        for name, p in model.named_parameters():
            snap = pre_data[str(name)]
            # Compare element-wise without dtype coercion loss: both
            # sides share the param's original dtype.
            assert p.data.dtype == snap.dtype, (
                f"rank {rank}: dtype mismatch after gather for "
                f"{name}: {p.data.dtype} vs {snap.dtype}"
            )
            assert torch.equal(p.data.cpu(), snap), (
                f"rank {rank}: after mixed-dtype sharded gather, param "
                f"'{name}' does not match pre-offload snapshot"
            )

        # (3) Plant rank-specific grads on every param, call
        # reduce_grads_and_offload, verify each region's CPU shard grad
        # holds the AVG across ranks.
        for _n, p in model.named_parameters():
            p.grad = torch.full_like(p.data, float(rank))

        mgr.reduce_grads_and_offload(ChunkId(0))

        expected_mean = sum(range(world_size)) / float(world_size)
        for region in shard_state.regions:
            obs = region.shard_param.grad.detach().cpu().float()  # type: ignore[union-attr]
            assert torch.allclose(
                obs,
                torch.full_like(obs, float(expected_mean)),
                atol=1e-3,
                rtol=1e-3,
            ), (
                f"rank {rank}: region (dtype={region.dtype}) shard grad "
                f"should be uniform {expected_mean}, got "
                f"min={obs.min().item()} max={obs.max().item()}"
            )

        mgr.uninstall()
        host.close()

    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.gpu
def test_zero3_sharded_roundtrip_mixed_dtype_2rank(tmp_path) -> None:
    """M7-followup mixed-dtype variant of the 2-rank sharded round-trip.

    Covers the dtype-region machinery that replaced the pre-followup
    "fall back to replicated when dtypes are mixed" path.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_zero3_sharded_roundtrip_mixed_dtype,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    skip_files = list(tmp_path.glob("rank*.skip"))
    if skip_files:
        reasons = [f.read_text().strip() for f in skip_files]
        pytest.skip(f"gloo does not support required collective(s): {reasons}")
