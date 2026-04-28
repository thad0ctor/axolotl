"""Optimizer-state checkpoint/resume for the ProTrain runtime.

Implements Phase 1 (CHECKPOINT_DESIGN.md) and Phase 2 Mode-B
(CHECKPOINT_DESIGN_PHASE2.md). Save runs through
``ProTrainOptimizerCheckpointCallback.on_save`` after HF writes its
standard checkpoint files; load runs through a monkey-patched
``trainer._load_optimizer_and_scheduler`` (HF has no
``on_load_checkpoint`` callback, and ``on_train_begin`` fires after
the load slot, so the patch is the only correct hook).

On disk under ``{checkpoint_dir}/protrain_optim/``:

* ``metadata.json``        — schema version, layout signature,
                             effective persistent_ids set, world_size,
                             zero3_shard, save_mode, saving_rank,
                             hyperparam snapshot, step.
* ``gpu_optim.pt``         — ``torch.save`` of the persistent inner
                             optimizer's ``state_dict`` (absent if no
                             chunks are persistent).
* ``cpu_optim/chunk_N.pt`` — one file per non-persistent chunk; each
                             holds the inner DeepSpeedCPUAdam's
                             ``state_dict``. Bounds peak save-time RAM
                             to one chunk's worth of state.

Mode-B (DDP-replicated) writes only on rank-0 — every rank has the
same state by DDP's grad-allreduce contract. Mode-C (ZeRO-3 sharded)
is not yet implemented; the dispatcher raises ``NotImplementedError``
for that path.

Hard validation on load: zero3_shard, layout signature, save_mode,
and effective persistent_ids set must all match the current run. World
size is allowed to differ between save and load in Mode-B (replicated
state is shape-independent of world_size). All ``torch.load`` calls
pin ``map_location='cpu'`` to defeat HF Trainer's hostile
``map_location=device`` default for CPU-offloaded adam state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import TYPE_CHECKING, Any

import torch

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers.trainer_callback import (
        TrainerCallback,
        TrainerControl,
        TrainerState,
    )
    from transformers.training_args import TrainingArguments

LOG = get_logger(__name__)

PROTRAIN_OPTIM_DIRNAME = "protrain_optim"
METADATA_FILENAME = "metadata.json"
GPU_OPTIM_FILENAME = "gpu_optim.pt"
CPU_OPTIM_DIRNAME = "cpu_optim"
CHUNK_FILE_RE = re.compile(r"^chunk_(\d+)\.pt$")
SCHEMA_FORMAT_VERSION = 2
SAVE_MODE_REPLICATED = "replicated"
SAVE_MODE_SHARDED = "sharded"
DEFAULT_SAVE_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB; mirrors args.py default


# ---------------------------------------------------------------------------
# Distributed helpers — no-op on single-rank
# ---------------------------------------------------------------------------


def _dist_is_active() -> bool:
    return bool(
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )


def _broadcast_object_list_or_noop(obj_list: list, src: int = 0) -> None:
    """Broadcast a list of picklable objects from ``src`` to every rank.

    No-op when ``torch.distributed`` is not initialized — preserves
    Phase 1 single-rank behavior. ``obj_list`` is mutated in place to
    match ``src``'s contents.
    """
    if not _dist_is_active():
        return
    torch.distributed.broadcast_object_list(obj_list, src=src)


def _barrier_or_noop() -> None:
    """``dist.barrier()`` if dist is active; else no-op."""
    if not _dist_is_active():
        return
    torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def _effective_persistent_ids(chunk_manager: Any) -> list[int]:
    """Sorted list of persistent ChunkIds — the post-non-block-pin set."""
    return sorted(int(cid) for cid in chunk_manager._persistent_ids)


def _layout_signature(
    chunk_manager: Any, world_size: int, zero3_shard: bool
) -> str:
    """SHA-256 over the load-bearing layout fields.

    The signature catches model/architecture drift between save and
    load: a checkpoint built against one chunk geometry must not be
    quietly loaded against a different geometry. Inputs include the
    full per-chunk param-name ordering, S_chunk, N_chunk, the
    effective persistent set, world_size, and zero3_shard.
    """
    layout = chunk_manager.layout
    fingerprint = {
        "S_chunk": int(layout.S_chunk),
        "N_chunk": int(layout.N_chunk),
        "chunks": [list(map(str, c)) for c in layout.chunks],
        "persistent_ids": _effective_persistent_ids(chunk_manager),
        "world_size": int(world_size),
        "zero3_shard": bool(zero3_shard),
    }
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _estimate_optim_state_bytes(optim: Any) -> int:
    """Estimated bytes for the optimizer's persisted Adam state.

    Walks each INNER adapter's ``state`` dict (``_gpu_optim._optim`` and
    every entry in ``_cpu_optim._optims``) and sums tensor bytes —
    counting exactly what gets pickled to disk modulo Python object
    overhead.

    Walking the user-facing ``optim.param_groups`` is wrong here:
    after :meth:`ChunkManager.materialize_offload` runs, every
    offloaded param's ``.data`` is replaced with an empty placeholder
    (manager.py:706 / :1494), so ``p.numel()`` returns 0 between
    training steps and the estimate misses every offloaded chunk's
    optimizer state. For 7B full-FT that's the difference between a
    silent 84 GB write and a correct gate trip.

    Pre-first-step the inner state dicts are empty and this returns 0
    — that's correct: there is no state to save yet, so any save would
    produce small placeholder files that can pass the gate.
    """
    import torch

    total = 0

    def _add_inner(inner_optim: Any) -> None:
        nonlocal total
        for state in getattr(inner_optim, "state", {}).values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    total += int(v.numel()) * int(v.element_size())

    gpu_optim = getattr(optim, "_gpu_optim", None)
    if gpu_optim is not None:
        inner = getattr(gpu_optim, "_optim", None)
        if inner is not None:
            _add_inner(inner)

    cpu_optim = getattr(optim, "_cpu_optim", None)
    if cpu_optim is not None:
        for inner in getattr(cpu_optim, "_optims", {}).values():
            _add_inner(inner)

    return total


def _hyperparam_snapshot(optim: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in optim.param_groups:
        out.append(
            {
                k: v
                for k, v in group.items()
                if k in ("lr", "betas", "eps", "weight_decay")
            }
        )
    return out


def _is_raw_protrain_optimizer(optim: Any) -> bool:
    """Duck-type for the raw _ProTrainOptimizer (avoids a circular import)."""
    return (
        hasattr(optim, "_gpu_optim")
        and hasattr(optim, "_cpu_optim")
        and hasattr(optim, "_chunk_manager")
    )


def _unwrap_protrain_optim(optim: Any) -> Any:
    """Return the raw _ProTrainOptimizer or None.

    HF Trainer + Accelerate wrap ``trainer.optimizer`` with
    ``AcceleratedOptimizer`` after Accelerate's ``prepare`` runs, and
    every callback fired post-prepare receives the wrapped form (see
    accelerate/optimizer.py: AcceleratedOptimizer stores the raw
    optimizer at ``.optimizer``). Without this unwrap, the callback's
    duck-type check fails on the wrapper and the save silently no-ops
    in real Trainer runs.
    """
    if optim is None:
        return None
    if _is_raw_protrain_optimizer(optim):
        return optim
    inner = getattr(optim, "optimizer", None)
    if inner is not None and _is_raw_protrain_optimizer(inner):
        return inner
    return None


def _is_protrain_optimizer(optim: Any) -> bool:
    """Truthy iff ``optim`` is (or wraps) a _ProTrainOptimizer."""
    return _unwrap_protrain_optim(optim) is not None


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def _hash_state_dict(sd: dict) -> bytes:
    """Recursively hash a state_dict-like nested structure deterministically.

    pickle.dumps is NOT cross-process-deterministic for torch tensors:
    the pickle stream embeds Python-level metadata (storage offsets,
    type-class object IDs in some torch builds) that can drift between
    the two mp.spawn workers' independent CUDA contexts even when the
    tensor *values* are identical. We instead walk the nested dict and
    feed only the load-bearing bytes (tensor element bytes, scalar
    values, sorted dict keys) into the hash.
    """
    h = hashlib.sha256()

    def _emit(obj: Any) -> None:
        if isinstance(obj, dict):
            h.update(b"dict:")
            for k in sorted(obj, key=repr):
                h.update(repr(k).encode("utf-8"))
                h.update(b"=")
                _emit(obj[k])
                h.update(b";")
        elif isinstance(obj, (list, tuple)):
            h.update(b"seq:")
            for item in obj:
                _emit(item)
                h.update(b",")
        elif isinstance(obj, torch.Tensor):
            t = obj.detach().contiguous().cpu()
            h.update(b"t:")
            h.update(str(t.dtype).encode("utf-8"))
            h.update(b":")
            h.update(repr(tuple(t.shape)).encode("utf-8"))
            h.update(b":")
            # numpy() avoids pickle stream non-determinism; the raw
            # tensor bytes are what we actually care about.
            h.update(t.numpy().tobytes())
        else:
            # Scalar: int, float, bool, str, None, etc. repr() is
            # stable across processes.
            h.update(repr(obj).encode("utf-8"))

    _emit(sd)
    return h.digest()


def _hash_inner_state_dicts(optim: Any) -> str:
    """SHA-256 over the rank's inner optimizer state dicts.

    Used by the optional Mode-B cross-rank verify path (§2.4 of the
    Phase 2 design). Walks the same inner adapters the save path
    serializes (``_gpu_optim._optim`` and every entry in
    ``_cpu_optim._optims``) and folds each state_dict's structural
    bytes into the hash via :func:`_hash_state_dict`.
    """
    h = hashlib.sha256()
    if optim._gpu_optim is not None:
        h.update(b"gpu:")
        h.update(_hash_state_dict(optim._gpu_optim._optim.state_dict()))
    if optim._cpu_optim is not None:
        for cid in sorted(optim._cpu_optim._optims):
            h.update(f"cpu:{int(cid)}:".encode("utf-8"))
            h.update(
                _hash_state_dict(optim._cpu_optim._optims[cid].state_dict())
            )
    return h.hexdigest()


def _verify_replicated_state_across_ranks(optim: Any, *, world_size: int) -> None:
    """Cross-rank state-equality check for Mode-B (opt-in, single shot).

    Each rank computes a SHA-256 over its inner state, all_gather_object
    the hashes, and raises if any rank disagrees with rank-0. Cheap
    insurance against the corner case where DDP determinism fails
    (numerical drift, manual override, etc.) and rank-0's saved state
    would not represent the cluster.
    """
    if world_size <= 1 or not _dist_is_active():
        return
    local_hash = _hash_inner_state_dicts(optim)
    gathered: list[str] = [""] * world_size
    torch.distributed.all_gather_object(gathered, local_hash)
    rank0 = gathered[0]
    diverged = [
        (r, h) for r, h in enumerate(gathered) if h != rank0
    ]
    if diverged:
        raise RuntimeError(
            "ProTrain optimizer save: Mode-B precondition violated — "
            "optimizer state diverges across ranks. Refusing to save "
            "(rank-0's state would not represent the cluster). "
            f"rank-0 hash={rank0!r}, divergent ranks: {diverged!r}"
        )


def _save_protrain_optim_dir(
    optim: Any,
    output_dir: str,
    *,
    step: int,
    save_max_bytes: int,
    rank: int = 0,
    world_size: int | None = None,
) -> bool:
    """Write the protrain_optim/ subdirectory. Returns True iff written.

    Mode-B (DDP-replicated) is the supported multi-rank flow. When
    ``world_size > 1`` and ``zero3_shard == False``, only rank-0
    actually writes; other ranks return True (the save was performed
    cluster-wide via rank-0). Mode-C (sharded) raises
    ``NotImplementedError`` — that lands in a follow-up.

    Returns False (with a WARN) when the size estimate exceeds
    ``save_max_bytes``. The user opts in to large saves by raising
    that threshold via ``protrain_optim_save_max_bytes``. The HF-side
    optimizer.pt is independent — the plugin's ``save_only_model``
    knob controls that.

    Raises RuntimeError on zero3_shard=True (Mode-C save is not yet
    implemented).

    ``rank`` and ``world_size`` are the HF Trainer's view (typically
    ``args.process_index`` / ``args.world_size``). ``world_size=None``
    falls back to ``_current_world_size`` for backward compatibility
    with Phase-1 callers.
    """
    chunk_manager = optim._chunk_manager
    if world_size is None:
        world_size = _current_world_size()
    zero3_shard = bool(getattr(chunk_manager, "zero3_shard", False))

    if zero3_shard:
        raise NotImplementedError(
            "ProTrain optimizer save: Mode-C sharded save/load is "
            "Phase 2-second; lands in protrain-optim-checkpoint-phase2-"
            "mode-c. Disable via protrain_save_optimizer_state=False."
        )

    estimate = _estimate_optim_state_bytes(optim)
    if estimate > save_max_bytes:
        LOG.warning(
            "ProTrain optimizer save: estimated %d bytes (~%.2f GiB) exceeds "
            "protrain_optim_save_max_bytes=%d (~%.2f GiB) — skipping save. "
            "Raise protrain_optim_save_max_bytes to opt in to larger saves.",
            estimate,
            estimate / 1024**3,
            save_max_bytes,
            save_max_bytes / 1024**3,
        )
        return False

    # Drain any in-flight async CPU Adam futures so we snapshot a
    # consistent post-step state, not a half-applied one. Every rank
    # drains its own queue; the rank-0-only-write contract is below.
    chunk_manager.wait_cpu_optim_all()

    if rank != 0:
        # Mode-B: only rank-0 writes. Other ranks just return True so
        # the caller knows the save was performed cluster-wide.
        return True

    target = os.path.join(output_dir, PROTRAIN_OPTIM_DIRNAME)
    os.makedirs(target, exist_ok=True)

    metadata = {
        "format_version": SCHEMA_FORMAT_VERSION,
        "protrain_layout_signature": _layout_signature(
            chunk_manager, world_size, zero3_shard
        ),
        "protrain_persistent_ids": _effective_persistent_ids(chunk_manager),
        "protrain_n_buffer": int(getattr(chunk_manager, "n_buffer", 0)),
        "protrain_world_size": int(world_size),
        "protrain_zero3_shard": zero3_shard,
        "protrain_save_mode": SAVE_MODE_REPLICATED,
        "saving_rank": int(rank),
        "param_groups_meta": _hyperparam_snapshot(optim),
        "saved_at_step": int(step),
        "torch_version": str(torch.__version__),
        "estimated_optim_state_bytes": int(estimate),
    }
    with open(os.path.join(target, METADATA_FILENAME), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if optim._gpu_optim is not None:
        torch.save(
            optim._gpu_optim._optim.state_dict(),
            os.path.join(target, GPU_OPTIM_FILENAME),
        )

    if optim._cpu_optim is not None and optim._cpu_optim._optims:
        cpu_dir = os.path.join(target, CPU_OPTIM_DIRNAME)
        os.makedirs(cpu_dir, exist_ok=True)
        for cid, inner in optim._cpu_optim._optims.items():
            torch.save(
                inner.state_dict(),
                os.path.join(cpu_dir, f"chunk_{int(cid)}.pt"),
            )

    LOG.info(
        "ProTrain optimizer save: wrote %s (estimate=%d bytes, "
        "persistent=%d chunks, cpu_chunks=%d, step=%d, "
        "world_size=%d, save_mode=%s)",
        target,
        estimate,
        len(metadata["protrain_persistent_ids"]),
        len(optim._cpu_optim._optims) if optim._cpu_optim is not None else 0,
        step,
        world_size,
        SAVE_MODE_REPLICATED,
    )
    return True


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _load_protrain_optim_dir(optim: Any, checkpoint_dir: str) -> bool:
    """Load a previously saved protrain_optim/ subdirectory in-place.

    Returns True iff the directory existed and was loaded (or False if
    the checkpoint dir simply has no ProTrain shard, which is the
    normal "first run / opt-out" case).

    Raises RuntimeError on any mismatch the saved metadata flags
    against the current run (zero3_shard, save_mode, layout signature,
    persistent_ids set, missing per-chunk file).

    World-size mismatch policy (CHECKPOINT_DESIGN_PHASE2.md §4.1
    Option B): Mode-B replicated saves are tolerated across world_size
    changes — the on-disk state is rank-independent. Mode-C sharded
    saves require identical world_size (and Mode-C resume itself is
    not yet implemented).

    Forward compatibility: ``format_version=1`` saves are read as
    Mode-B replicated with ``saving_rank=0`` and ``world_size=1``
    (CHECKPOINT_DESIGN_PHASE2.md §5).

    All torch.load calls use map_location='cpu'. Inner load_state_dict
    handles device placement per-tensor (GPU adam → GPU, CPU adam →
    CPU), which is correct because the inner state_dicts already hold
    the right device tags.
    """
    target = os.path.join(checkpoint_dir, PROTRAIN_OPTIM_DIRNAME)
    if not os.path.isdir(target):
        return False

    meta_path = os.path.join(target, METADATA_FILENAME)
    if not os.path.isfile(meta_path):
        raise RuntimeError(
            f"ProTrain optimizer load: {target!r} exists but lacks "
            f"{METADATA_FILENAME}. Refusing to load partial checkpoint."
        )
    with open(meta_path) as f:
        metadata = json.load(f)

    fmt = int(metadata.get("format_version", 0))
    if fmt == 1:
        # Forward compat: v1 saves predate the save_mode / saving_rank
        # fields. They're known to be single-rank non-ZeRO replicated
        # by Phase 1's hard guard.
        metadata.setdefault("protrain_save_mode", SAVE_MODE_REPLICATED)
        metadata.setdefault("saving_rank", 0)
        metadata.setdefault("protrain_world_size", 1)
    elif fmt == SCHEMA_FORMAT_VERSION:
        if "protrain_save_mode" not in metadata:
            raise RuntimeError(
                "ProTrain optimizer load: v2 metadata missing required "
                "field 'protrain_save_mode'. Refusing to load."
            )
        if "saving_rank" not in metadata:
            raise RuntimeError(
                "ProTrain optimizer load: v2 metadata missing required "
                "field 'saving_rank'. Refusing to load."
            )
    else:
        raise RuntimeError(
            f"ProTrain optimizer load: unknown format_version={fmt} "
            f"(this build expects {SCHEMA_FORMAT_VERSION}). Refusing to load."
        )

    chunk_manager = optim._chunk_manager
    current_world = _current_world_size()
    current_zero3 = bool(getattr(chunk_manager, "zero3_shard", False))
    saved_world = int(metadata["protrain_world_size"])
    saved_zero3 = bool(metadata["protrain_zero3_shard"])
    saved_mode = str(metadata["protrain_save_mode"])
    current_mode = (
        SAVE_MODE_SHARDED if current_zero3 else SAVE_MODE_REPLICATED
    )

    if saved_mode not in (SAVE_MODE_REPLICATED, SAVE_MODE_SHARDED):
        raise RuntimeError(
            f"ProTrain optimizer load: unknown protrain_save_mode="
            f"{saved_mode!r}. Refusing to load."
        )

    # Save-mode mismatch (§4.2). Hard error in either direction.
    if saved_mode != current_mode:
        raise RuntimeError(
            f"ProTrain optimizer load: save_mode mismatch — "
            f"saved={saved_mode!r} current={current_mode!r}. "
            "Replicated state cannot be loaded into a sharded run, and "
            "sharded state cannot be loaded into a replicated run; the "
            "on-disk shape doesn't match what the current run needs."
        )

    if saved_zero3 != current_zero3:
        raise RuntimeError(
            f"ProTrain optimizer load: zero3_shard mismatch — saved={saved_zero3} "
            f"current={current_zero3}."
        )

    if current_zero3:
        # We never reach here with a replicated save (the save_mode
        # mismatch above would have fired). A sharded save into the
        # current sharded run hits this guard until Mode-C lands.
        raise NotImplementedError(
            "ProTrain optimizer load: Mode-C sharded resume is not yet "
            "implemented; lands in protrain-optim-checkpoint-phase2-"
            "mode-c. Disable via protrain_save_optimizer_state=False."
        )

    # Mode-B replicated load (current scope). World-size differences
    # are tolerated per Option B — replicated state is shape-
    # independent of world_size.
    if saved_world != current_world:
        LOG.info(
            "ProTrain optimizer load: replicated checkpoint saved with "
            "world_size=%d loading into world_size=%d. Replicated state "
            "is rank-independent, so this is supported.",
            saved_world,
            current_world,
        )

    # Layout signature embeds world_size, so a world_size delta would
    # naively trip the signature check. Recompute the saved signature's
    # would-be value at the CURRENT world_size for the comparison —
    # the only legitimately load-bearing layout fields here are chunk
    # geometry + persistent_ids + zero3_shard.
    saved_sig = metadata["protrain_layout_signature"]
    expected_sig = _layout_signature(
        chunk_manager, saved_world, saved_zero3
    )
    if saved_sig != expected_sig:
        raise RuntimeError(
            "ProTrain optimizer load: layout signature mismatch.\n"
            f"  saved   = {saved_sig}\n"
            f"  current = {expected_sig}\n"
            "The model architecture, S_chunk, persistent_ids, world_size, or "
            "zero3_shard differs between save and load. Resume is unsafe."
        )

    saved_pids = list(metadata["protrain_persistent_ids"])
    current_pids = _effective_persistent_ids(chunk_manager)
    if saved_pids != current_pids:
        raise RuntimeError(
            "ProTrain optimizer load: persistent_ids set mismatch.\n"
            f"  saved   = {saved_pids}\n"
            f"  current = {current_pids}\n"
            "The search picked a different partition. Pin the saved set via "
            "protrain_n_persist_override (and related overrides) to resume."
        )

    # GPU optim: load if both saved file and current optim slot exist.
    gpu_path = os.path.join(target, GPU_OPTIM_FILENAME)
    if os.path.isfile(gpu_path):
        if optim._gpu_optim is None:
            raise RuntimeError(
                "ProTrain optimizer load: gpu_optim.pt present on disk but "
                "current optimizer has no persistent (GPU) inner — partition "
                "mismatch slipped past the layout-signature check."
            )
        loaded = torch.load(gpu_path, map_location="cpu", weights_only=False)
        optim._gpu_optim._optim.load_state_dict(loaded)
    elif optim._gpu_optim is not None:
        raise RuntimeError(
            "ProTrain optimizer load: current optimizer has a persistent "
            "(GPU) inner but gpu_optim.pt is absent on disk."
        )

    # CPU optim: walk saved chunk files; require an exact match against the
    # current set of non-persistent chunk IDs.
    cpu_dir = os.path.join(target, CPU_OPTIM_DIRNAME)
    saved_chunks: dict[int, str] = {}
    if os.path.isdir(cpu_dir):
        for name in os.listdir(cpu_dir):
            m = CHUNK_FILE_RE.match(name)
            if m is None:
                raise RuntimeError(
                    f"ProTrain optimizer load: unexpected file {name!r} in "
                    f"{cpu_dir!r} — refusing to load."
                )
            saved_chunks[int(m.group(1))] = os.path.join(cpu_dir, name)

    current_cpu_ids = (
        set(int(cid) for cid in optim._cpu_optim._optims)
        if optim._cpu_optim is not None
        else set()
    )
    saved_cpu_ids = set(saved_chunks)
    if saved_cpu_ids != current_cpu_ids:
        missing_on_disk = current_cpu_ids - saved_cpu_ids
        extra_on_disk = saved_cpu_ids - current_cpu_ids
        raise RuntimeError(
            "ProTrain optimizer load: CPU chunk set mismatch — "
            f"missing on disk: {sorted(missing_on_disk)}, "
            f"extra on disk: {sorted(extra_on_disk)}."
        )

    if optim._cpu_optim is not None:
        for cid, inner in optim._cpu_optim._optims.items():
            loaded = torch.load(
                saved_chunks[int(cid)], map_location="cpu", weights_only=False
            )
            inner.load_state_dict(loaded)
            # ``torch.optim.Optimizer.load_state_dict`` auto-casts every
            # state tensor to the device of the matching param. After
            # ``ChunkManager.materialize_offload`` runs, the user-facing
            # params held by the inner CPU adam have empty GPU
            # placeholders for ``.data`` — so torch silently moves the
            # loaded ``exp_avg`` / ``exp_avg_sq`` tensors to CUDA. The
            # DeepSpeedCPUAdam C++ kernel then segfaults on the next
            # step trying to write through a GPU pointer. Force the
            # inner CPU adam state back to CPU after the cast.
            for state in inner.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                        state[k] = v.cpu()

    # Hyperparam drift: warn but accept. JSON serialization turns
    # ``betas`` tuples into lists; normalize before comparing so
    # round-tripped data doesn't trigger a spurious warning.
    def _normalize_hp(hp: dict[str, Any]) -> dict[str, Any]:
        return {
            k: (tuple(v) if isinstance(v, list) else v)
            for k, v in hp.items()
        }

    saved_hp = metadata.get("param_groups_meta", [])
    current_hp = _hyperparam_snapshot(optim)
    for i, (s, c) in enumerate(zip(saved_hp, current_hp)):
        if _normalize_hp(s) != _normalize_hp(c):
            LOG.warning(
                "ProTrain optimizer load: param_groups[%d] hyperparams drifted "
                "between save and load — saved=%s current=%s. Continuing.",
                i,
                s,
                c,
            )

    LOG.info(
        "ProTrain optimizer load: restored from %s (saved_at_step=%d, "
        "persistent=%d chunks, cpu_chunks=%d)",
        target,
        int(metadata.get("saved_at_step", -1)),
        len(saved_pids),
        len(saved_chunks),
    )
    return True


# ---------------------------------------------------------------------------
# Public callback (save side)
# ---------------------------------------------------------------------------


def _make_callback_class():
    """Lazy-imported callback class — keeps ``transformers`` out of the
    module-import path so unit tests that don't need HF can stay light."""
    from transformers.trainer_callback import TrainerCallback

    class ProTrainOptimizerCheckpointCallback(TrainerCallback):
        """``on_save``: write protrain_optim/ beside HF's checkpoint dir.

        Reads the optimizer off ``kwargs['optimizer']`` (HF passes it in
        on every callback). Routes the save through
        ``_save_protrain_optim_dir``, which enforces the gating + scope
        checks. Failures are loud (raise) — silently producing an
        unloadable checkpoint is worse than crashing on save.

        HF's ``on_save`` fires on every rank
        (``_maybe_log_save_evaluate`` calls ``callback_handler.on_save``
        unconditionally). For Mode-B the callback orchestrates a rank-0-
        only write with cross-rank coordination:

        * Every rank drains ``wait_cpu_optim_all`` (CPU adam must be
          quiescent before any rank snapshots).
        * Rank-0 computes the size-gate decision; the decision is
          broadcast so all ranks act consistently (no partial saves).
        * Optional opt-in: on the FIRST save of each run, every rank
          hashes its inner state and ``all_gather_object``-s the hashes
          to verify Mode-B's replication invariant. Skipped on
          subsequent saves to keep per-save overhead low.
        * Rank-0 writes; other ranks no-op.
        * ``dist.barrier()`` at exit so callers see a complete dir.
        """

        def __init__(
            self,
            *,
            save_max_bytes: int,
            verify_replicated: bool = False,
        ) -> None:
            self._save_max_bytes = save_max_bytes
            self._verify_replicated = bool(verify_replicated)
            # Track whether the cross-rank verify already fired for
            # this run; we only do it on the first save (cheap insurance
            # at run start, but per-save would be expensive).
            self._verify_replicated_done = False

        def on_save(
            self,
            args: "TrainingArguments",
            state: "TrainerState",
            control: "TrainerControl",
            **kwargs: Any,
        ) -> "TrainerControl":
            # Trainer.optimizer is wrapped by AcceleratedOptimizer after
            # prepare runs; the callback receives the wrapped form. Unwrap
            # before the duck-type guard.
            raw = _unwrap_protrain_optim(kwargs.get("optimizer"))
            if raw is None:
                return control

            rank = int(getattr(args, "process_index", 0))
            world_size = int(getattr(args, "world_size", 1))
            chunk_manager = raw._chunk_manager
            zero3_shard = bool(getattr(chunk_manager, "zero3_shard", False))

            checkpoint_dir = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            # Only rank-0 sees the HF-created checkpoint dir on multi-
            # rank runs (`should_save` gates HF's mkdir). The other
            # ranks must still drain their CPU adam and participate in
            # the broadcast / barrier so the cross-rank protocol stays
            # in sync — but if rank-0 itself doesn't see the dir, that's
            # the legitimate "skip" case.
            if rank == 0 and not os.path.isdir(checkpoint_dir):
                LOG.warning(
                    "ProTrainOptimizerCheckpointCallback.on_save: expected "
                    "checkpoint dir %s does not exist on rank-0; skipping "
                    "ProTrain shard.",
                    checkpoint_dir,
                )
                # Still broadcast the skip so non-rank-0 ranks bail in
                # lockstep.
                skip_decision = [True]
                _broadcast_object_list_or_noop(skip_decision, src=0)
                _barrier_or_noop()
                return control

            # ---------- 1. Drain CPU adam on every rank ----------
            chunk_manager.wait_cpu_optim_all()

            # Mode-C save is not yet implemented; raise loudly here so
            # the failure points at the right follow-up PR. Every rank
            # raises in lockstep — no risk of a partial Mode-C save.
            if zero3_shard:
                raise NotImplementedError(
                    "Mode-C sharded save/load is Phase 2-second; lands "
                    "in protrain-optim-checkpoint-phase2-mode-c. Disable "
                    "via protrain_save_optimizer_state=False."
                )

            # ---------- 2. Estimate-gate broadcast ----------
            # Rank-0 decides; all ranks act on rank-0's decision. The
            # broadcast is a no-op on single-rank runs.
            if rank == 0:
                estimate = _estimate_optim_state_bytes(raw)
                skip = estimate > self._save_max_bytes
                if skip:
                    LOG.warning(
                        "ProTrain optimizer save: estimated %d bytes "
                        "(~%.2f GiB) exceeds protrain_optim_save_max_bytes="
                        "%d (~%.2f GiB) — skipping save (decision "
                        "broadcast to %d ranks).",
                        estimate,
                        estimate / 1024**3,
                        self._save_max_bytes,
                        self._save_max_bytes / 1024**3,
                        world_size,
                    )
            else:
                skip = False  # placeholder, will be overwritten by broadcast
            skip_decision = [skip]
            _broadcast_object_list_or_noop(skip_decision, src=0)
            if skip_decision[0]:
                _barrier_or_noop()
                return control

            # ---------- 3. Cross-rank verify (opt-in, once per run) ----------
            if (
                self._verify_replicated
                and not self._verify_replicated_done
                and world_size > 1
            ):
                _verify_replicated_state_across_ranks(
                    raw, world_size=world_size
                )
                self._verify_replicated_done = True

            # ---------- 4. Mode-B rank-0-only write ----------
            _save_protrain_optim_dir(
                raw,
                checkpoint_dir,
                step=int(state.global_step),
                save_max_bytes=self._save_max_bytes,
                rank=rank,
                world_size=world_size,
            )

            # ---------- 5. Barrier so downstream code sees the dir ----------
            _barrier_or_noop()
            return control

    return ProTrainOptimizerCheckpointCallback


def make_checkpoint_callback(
    *,
    save_max_bytes: int,
    verify_replicated: bool = False,
) -> "TrainerCallback":
    cls = _make_callback_class()
    return cls(
        save_max_bytes=save_max_bytes,
        verify_replicated=verify_replicated,
    )


# ---------------------------------------------------------------------------
# Load monkey-patch
# ---------------------------------------------------------------------------


def install_load_hook(trainer: Any, optim: Any) -> None:
    """Wrap ``trainer._load_optimizer_and_scheduler`` to also load ProTrain.

    HF's TrainerCallback API has no ``on_load_checkpoint``;
    ``on_train_begin`` fires AFTER the load slot. This patch is the
    only correct lifecycle position. Symmetric with the existing
    optim.state_dict / optim.load_state_dict monkey-patches in
    plugin.py: the no-op patches stay (they coexist with Accelerate's
    prepare round-trip), and this load hook handles real resume via a
    completely separate path.

    The closed-over ``optim`` is captured at install time (in
    ``post_trainer_create``, BEFORE Accelerate.prepare wraps the
    optimizer), so it's already raw. We unwrap defensively in case
    the caller hands in a wrapper.
    """
    raw = _unwrap_protrain_optim(optim)
    if raw is None:
        # Caller passed something that isn't a ProTrain optimizer —
        # silently no-op rather than installing a hook that would
        # never fire.
        return

    original = trainer._load_optimizer_and_scheduler

    def _patched(checkpoint: str | None) -> None:
        original(checkpoint)
        if checkpoint is None:
            return
        try:
            _load_protrain_optim_dir(raw, checkpoint)
        except Exception:
            LOG.exception(
                "ProTrain optimizer load failed from %s — re-raising. "
                "If you intended to discard the saved state, set "
                "protrain_save_optimizer_state=False and remove the "
                "protrain_optim/ subdirectory from the checkpoint.",
                checkpoint,
            )
            raise
        # Defensive barrier: every rank loaded its own copy of the
        # files; the barrier just ensures the cluster moves past the
        # load slot in lockstep before training resumes. Cheap on
        # single-rank (no-op).
        _barrier_or_noop()

    trainer._load_optimizer_and_scheduler = _patched  # type: ignore[method-assign]


__all__ = [
    "PROTRAIN_OPTIM_DIRNAME",
    "SCHEMA_FORMAT_VERSION",
    "SAVE_MODE_REPLICATED",
    "SAVE_MODE_SHARDED",
    "DEFAULT_SAVE_MAX_BYTES",
    "make_checkpoint_callback",
    "install_load_hook",
    # Internals exposed for unit tests:
    "_save_protrain_optim_dir",
    "_load_protrain_optim_dir",
    "_layout_signature",
    "_effective_persistent_ids",
    "_estimate_optim_state_bytes",
    "_is_protrain_optimizer",
    "_is_raw_protrain_optimizer",
    "_unwrap_protrain_optim",
    "_hash_inner_state_dicts",
    "_verify_replicated_state_across_ranks",
    "_broadcast_object_list_or_noop",
    "_barrier_or_noop",
]
