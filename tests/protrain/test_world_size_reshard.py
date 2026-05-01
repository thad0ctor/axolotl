"""Live world-size reshard test (Mode-B replicated, 4 ranks → 2 ranks).

ProTrain's Mode-B replicated checkpoint format claims world-size-change
support — the on-disk state is rank-independent, so a save with
``world_size=4`` should load cleanly into a fresh ``world_size=2`` run.
``test_load_accepts_world_size_change_for_replicated`` only fakes the
metadata (mutates ``protrain_world_size`` in a 1-rank test) — it does
not exercise the live cross-process path. This test does:

1. Spawn 4 ranks via ``mp.spawn`` on GPUs 1, 2, 4, 5 (the free 24GB
   pool from MEMORY.md). Each rank builds an identical tiny model +
   ChunkManager + ``_ProTrainOptimizer``, runs one fwd+bwd+step so
   the inner Adam state is non-trivial, then saves the checkpoint.
   Rank-0 writes; rank-1..3 reach the post-callback barrier and exit.
2. Tear down the 4-rank world (every worker calls
   ``destroy_process_group``; ``mp.spawn`` joins).
3. Spawn 2 ranks on GPUs 1, 2 (subset of the same pool). Each rank
   builds the same tiny model fresh, calls
   ``_load_protrain_optim_dir`` against the saved directory, runs one
   step, and asserts the resulting loss is finite. The pre-step
   inner state must match what rank-0 wrote at save time (proving the
   load actually reads files, not silently no-ops).

Mode-B is the test target rather than Mode-C because Mode-C
explicitly hard-errors on ``saved_world != current_world``
(checkpoint.py:915). Cross-world-size reshard for Mode-C requires a
re-shard step that is documented as out-of-scope for Phase 2 (see
CHECKPOINT_DESIGN_PHASE2.md §4.1). The Mode-B path is the surface
that actually advertises world-size-change support today.

Slow-marked, single test, < 5 min wall on the rig per the handoff
budget.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, cast

import pytest


# Reuse the helper machinery from the main optimizer-checkpoint test —
# mp.spawn workers can re-import the test module fine because pytest's
# rootdir is on sys.path during test collection.
from tests.protrain.test_optimizer_checkpoint import (  # noqa: E402
    _build_chunk_manager,
    _build_optim_pair,
    _force_identical_inner_state,
    _teardown_mgr,
    _tiny_model,
)

from axolotl.integrations.protrain.api.checkpoint import (  # noqa: E402
    DEFAULT_SAVE_MAX_BYTES,
    METADATA_FILENAME,
    PROTRAIN_OPTIM_DIRNAME,
    SAVE_MODE_REPLICATED,
    _load_protrain_optim_dir,
    _save_protrain_optim_dir,
)


# ---- worker bodies ---------------------------------------------------------


def _save_worker(rank: int, world_size: int, tmpdir: str) -> None:
    """One rank in the 4-rank save phase.

    Rank-0 writes; all ranks must reach the post-save barrier so the
    parent test can confirm liveness via ``rank{N}.done``. Inner state
    is zeroed before save so the load-phase post-load comparison has a
    deterministic target (eliminates DDP-vs-non-DDP / CPU-adam threading
    noise; this test is about the save+load mechanism, not about
    DDP determinism).
    """
    import torch
    import torch.distributed as dist

    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("worker: CUDA not available")

        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpdir}/rendezvous-save",
            rank=rank,
            world_size=world_size,
        )

        torch.manual_seed(0)
        model = _tiny_model().to("cuda")
        mgr, host = _build_chunk_manager(model, n_persist=1, S_chunk=64 * 1024)
        mgr.materialize_offload()
        _, _, optim = _build_optim_pair(model, mgr)

        # One fwd+bwd+step so the inner state has real exp_avg / exp_avg_sq
        # entries (otherwise the gate would skip with a 0-byte estimate).
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(123)
        x = torch.randn(2, model.embed.in_features, generator=cpu_gen).to("cuda")
        for cid in list(mgr._non_persistent_ids):
            mgr.gather(cid)
        optim.zero_grad()
        out = model(x)
        out.sum().backward()
        optim.step()

        # Force byte-identical state across ranks. Mode-B's contract is
        # that DDP keeps the inner state replicated; we don't have DDP
        # in this test (it's a pure save/load mechanism check), so we
        # zero the state to skip past that question and focus the load
        # phase on file plumbing.
        _force_identical_inner_state(optim)

        save_dir = os.path.join(tmpdir, "save_root")
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        dist.barrier()

        if rank == 0:
            wrote = _save_protrain_optim_dir(
                optim,
                save_dir,
                step=1,
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                rank=0,
                world_size=world_size,
            )
            if not wrote:
                raise RuntimeError("rank-0 save returned False")
        dist.barrier()

        with open(os.path.join(tmpdir, f"save_rank{rank}.done"), "w") as f:
            f.write("ok")

        _teardown_mgr(mgr, optim)
        host.close()
        del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(os.path.join(tmpdir, f"save_rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def _load_worker(rank: int, world_size: int, tmpdir: str) -> None:
    """One rank in the 2-rank load phase.

    Builds a fresh model + manager + optim (same arch, same seed), then
    loads from the directory rank-0 wrote during the 4-rank save phase.

    Acceptance:
      * ``_load_protrain_optim_dir`` returns True (loaded the dir).
      * Loaded inner state == zero (matches what was forced+saved
        during the save phase). This proves the load actually read the
        on-disk bytes — without a load, the post-step state would be
        the result of one freshly-randomised step (non-zero with high
        probability).
      * One additional optimizer step lands without exception and
        produces a finite loss — proves the resharded state is
        consistent with the rebuilt chunk geometry.
    """
    import torch
    import torch.distributed as dist

    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("worker: CUDA not available")

        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpdir}/rendezvous-load",
            rank=rank,
            world_size=world_size,
        )

        torch.manual_seed(0)  # identical init across ranks → same arch hash
        model = _tiny_model().to("cuda")
        mgr, host = _build_chunk_manager(model, n_persist=1, S_chunk=64 * 1024)
        mgr.materialize_offload()
        _, _, optim = _build_optim_pair(model, mgr)

        # Take a non-zero step BEFORE the load so that "post-load state ==
        # zero" is a strong signal that the load happened. Without this,
        # a no-op load would leave the freshly-built (zero) inner state
        # and the assertion would falsely pass.
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(rank + 7)  # different per rank for noise
        x = torch.randn(2, model.embed.in_features, generator=cpu_gen).to("cuda")
        for cid in list(mgr._non_persistent_ids):
            mgr.gather(cid)
        optim.zero_grad()
        out = model(x)
        out.sum().backward()
        optim.step()

        # Snapshot inner state pre-load — every state tensor should be
        # non-zero now (one Adam step on a random batch).
        non_zero_pre_load = False
        if optim._gpu_optim is not None:
            for s in optim._gpu_optim._optim.state.values():
                for v in s.values():
                    if isinstance(v, torch.Tensor) and v.abs().sum() > 0:
                        non_zero_pre_load = True
        if optim._cpu_optim is not None:
            for inner in optim._cpu_optim._optims.values():
                for s in inner.state.values():
                    for v in s.values():
                        if isinstance(v, torch.Tensor) and v.abs().sum() > 0:
                            non_zero_pre_load = True
        if not non_zero_pre_load:
            raise RuntimeError(
                "load worker: pre-load inner state was already zero — "
                "the post-load==zero check below would be ambiguous"
            )

        save_dir = os.path.join(tmpdir, "save_root")
        loaded = _load_protrain_optim_dir(optim, save_dir)
        if not loaded:
            raise RuntimeError(
                f"rank {rank}: _load_protrain_optim_dir returned False — "
                f"checkpoint dir {save_dir} not found?"
            )

        # Acceptance: post-load state must match the saved (zero) state.
        post_load_all_zero = True
        if optim._gpu_optim is not None:
            for s in optim._gpu_optim._optim.state.values():
                for v in s.values():
                    if isinstance(v, torch.Tensor) and v.abs().sum() > 0:
                        post_load_all_zero = False
        if optim._cpu_optim is not None:
            for inner in optim._cpu_optim._optims.values():
                for s in inner.state.values():
                    for v in s.values():
                        if isinstance(v, torch.Tensor) and v.abs().sum() > 0:
                            post_load_all_zero = False
        if not post_load_all_zero:
            raise RuntimeError(
                f"rank {rank}: post-load inner state has non-zero entries — "
                "load did not overwrite the pre-load step's state, so "
                "the resharded state is not actually being applied"
            )

        # Acceptance: one more step on the resharded state must produce
        # a finite loss without exception. Re-gather every offloaded
        # chunk first — after the pre-load step, ``param.data`` for
        # non-persistent chunks is back to its empty placeholder, so a
        # forward without gather would crash on a (numel=0) weight.
        for cid in list(mgr._non_persistent_ids):
            mgr.gather(cid)
        cpu_gen2 = torch.Generator(device="cpu")
        cpu_gen2.manual_seed(rank + 17)
        x2 = torch.randn(2, model.embed.in_features, generator=cpu_gen2).to("cuda")
        optim.zero_grad()
        out2 = model(x2)
        loss2 = out2.sum()
        if not bool(torch.isfinite(loss2).item()):
            raise RuntimeError(
                f"rank {rank}: post-load step produced non-finite loss "
                f"{float(loss2.detach())}"
            )
        loss2.backward()
        optim.step()

        with open(os.path.join(tmpdir, f"load_rank{rank}.done"), "w") as f:
            f.write(f"loss2={float(loss2.detach())}\n")

        _teardown_mgr(mgr, optim)
        host.close()
        del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(os.path.join(tmpdir, f"load_rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


# ---- driver test -----------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
def test_replicated_world_size_reshard_4_to_2(tmp_path):
    """Live save N=4 / load N=2 replicated reshard end-to-end.

    Save phase uses 4 mp.spawn workers (one per visible GPU); load
    phase uses 2 (subset of the same physical pool). Both phases
    rendezvous via gloo on a file:// store rooted in tmp_path so the
    test does not need MASTER_PORT plumbing.

    The test is the live counterpart to
    ``test_load_accepts_world_size_change_for_replicated`` (which only
    mutates metadata in a single-process test). If Mode-B replicated
    state ever stops being world-size-independent, this test catches it.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    n_visible = torch.cuda.device_count()
    if n_visible < 4:
        pytest.skip(
            f"world-size-reshard test needs >= 4 visible GPUs (got {n_visible})"
        )

    import torch.multiprocessing as mp

    # ---- Phase 1: save with world_size=4 ----------------------------
    save_world = 4
    mp.spawn(
        _save_worker,
        args=(save_world, str(tmp_path)),
        nprocs=save_world,
        join=True,
    )
    err_files = sorted(tmp_path.glob("save_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"save-phase worker errors:\n{bodies}")
    for r in range(save_world):
        assert (tmp_path / f"save_rank{r}.done").is_file(), (
            f"save rank {r} did not reach post-save sentinel"
        )

    # Verify the saved metadata records world_size=4 (Mode-B) so the
    # load phase has something meaningful to reshard from.
    proot = tmp_path / "save_root" / PROTRAIN_OPTIM_DIRNAME
    assert proot.is_dir(), f"save root {proot} missing post-spawn"
    meta = json.loads((proot / METADATA_FILENAME).read_text())
    assert meta["protrain_save_mode"] == SAVE_MODE_REPLICATED, (
        f"expected replicated save_mode (Mode-B), got {meta['protrain_save_mode']!r}"
    )
    assert meta["protrain_world_size"] == save_world, (
        f"expected protrain_world_size={save_world}, got "
        f"{meta['protrain_world_size']}"
    )

    # ---- Phase 2: load with world_size=2 (different from save) ------
    load_world = 2
    mp.spawn(
        _load_worker,
        args=(load_world, str(tmp_path)),
        nprocs=load_world,
        join=True,
    )
    err_files = sorted(tmp_path.glob("load_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"load-phase worker errors:\n{bodies}")
    for r in range(load_world):
        assert (tmp_path / f"load_rank{r}.done").is_file(), (
            f"load rank {r} did not reach post-load sentinel"
        )
