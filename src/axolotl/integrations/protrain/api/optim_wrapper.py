"""Public optimizer-wrapper for the ProTrain runtime (§1, §5).

``protrain_optimizer_wrapper`` returns a :class:`torch.optim.Optimizer`
subclass that proxies ``step`` / ``zero_grad`` through the persistent
(GPU FusedAdam) and non-persistent (CPU FusedAdam, async) adapters
already instantiated by :func:`protrain_model_wrapper`.

Semantics:

* ``step()`` — synchronously runs the GPU step for persistent chunks,
  then blocks on every outstanding CPU Adam future so the non-persistent
  chunk updates have landed in their CPU shards before control returns.
* ``zero_grad()`` — zeros grads on both adapters.
* ``state_dict`` / ``load_state_dict`` — explicitly raise
  ``NotImplementedError``. Optimizer-state checkpointing is M5/M6
  scope; the M4b contract is to keep the method names resolvable so
  HuggingFace Trainer does not blow up if it touches the optimizer
  during init.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from axolotl.integrations.protrain.chunk import (
    CpuFusedAdamAdapter,
    GpuFusedAdamAdapter,
)
from axolotl.integrations.protrain.types import ChunkId, WrappedModel
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn

LOG = get_logger(__name__)


class _ProTrainOptimizer(torch.optim.Optimizer):
    """``torch.optim.Optimizer`` facade over the ProTrain adapter pair.

    We inherit from ``torch.optim.Optimizer`` primarily for interface
    compatibility with HuggingFace Trainer (which calls
    ``isinstance(optim, torch.optim.Optimizer)``); the actual update
    math is delegated to the two adapters.
    """

    def __init__(
        self,
        gpu_optim: GpuFusedAdamAdapter | None,
        cpu_optim: CpuFusedAdamAdapter | None,
        params: list["nn.Parameter"],
        defaults: dict[str, Any],
        chunk_manager: Any,
    ) -> None:
        # ``torch.optim.Optimizer.__init__`` requires at least one non-empty
        # parameter group. We pass the full param list so ``optim.param_groups``
        # reflects the real set — schedulers iterating over it still see
        # every tuneable param. The base class uses these only for
        # ``load_state_dict`` bookkeeping; the actual updates are routed
        # through the adapters in ``step``.
        if not params:
            # An empty-param optimizer is nonsensical — but during some smoke
            # tests every chunk can end up persistent and cpu_optim can be
            # None; we still need ``Optimizer`` super-init to succeed. Seed
            # with a dummy zero tensor in that case (torch rejects an empty
            # param group).
            raise ValueError(
                "_ProTrainOptimizer: model has no tunable parameters; "
                "nothing to optimize."
            )
        super().__init__(params, defaults)
        self._gpu_optim = gpu_optim
        self._cpu_optim = cpu_optim
        self._chunk_manager = chunk_manager

    # ---- step / zero_grad ----------------------------------------------

    def step(self, closure: Any = None) -> Any:  # noqa: ARG002 — HF convention
        """Drive both adapters then block on in-flight CPU futures.

        Persistent chunks: run the GPU step synchronously.
        Non-persistent chunks: per-param post-accumulate-grad hooks
        (installed by :meth:`ChunkManager.materialize_offload`) already
        kicked off the CPU FusedAdam step the instant each chunk's last
        grad landed on CPU. Here we just wait on every outstanding
        future so the next forward sees the updated CPU master params.
        """
        if self._gpu_optim is not None:
            self._gpu_optim.step()
        # Drain every in-flight CPU Adam future (M4.5 Gap 2: per-param
        # grad offload enqueued these from the grad hooks).
        self._chunk_manager.wait_cpu_optim_all()

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        if self._gpu_optim is not None:
            self._gpu_optim.zero_grad(set_to_none=set_to_none)
        if self._cpu_optim is not None:
            self._cpu_optim.zero_grad(set_to_none=set_to_none)
        # Also zero any param grads that weren't routed through either
        # adapter (e.g. buffers that slipped through the chunk layout) so
        # the next iteration starts clean.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    # ---- checkpointing: deliberately unimplemented for M4 ---------------

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        raise NotImplementedError(
            "ProTrain optimizer checkpointing is M5/M6 work; "
            "disable optimizer-state saving for now."
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        raise NotImplementedError(
            "ProTrain optimizer checkpointing is M5/M6 work; "
            "disable optimizer-state loading for now."
        )


def protrain_optimizer_wrapper(
    wrapped: WrappedModel,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Rebuild the GPU/CPU FusedAdam adapters at user-specified hyperparams.

    ``protrain_model_wrapper`` instantiates transient adapters with
    placeholder hyperparams so the chunk manager has something to drive
    during bring-up. This function rebuilds them with the real
    ``lr`` / ``betas`` / ``eps`` / ``weight_decay``, then swaps them
    into the chunk manager in-place so the scheduler's async
    ``reduce_grads_and_offload`` path continues to pump the right
    optimizer.
    """
    chunk_manager = wrapped.chunk_manager
    layout = chunk_manager.layout  # type: ignore[union-attr]
    n_persist = len(chunk_manager._persistent_ids)  # type: ignore[union-attr]

    # Partition params the same way ``protrain_model_wrapper`` did —
    # persistent chunks go to GPU FusedAdam, the rest to per-chunk
    # CPU FusedAdam adapters.
    module = wrapped.module
    params_by_name = dict(module.named_parameters())

    persistent_params: list["nn.Parameter"] = []
    cpu_params_per_chunk: dict[ChunkId, list["nn.Parameter"]] = {}

    for cid, chunk_param_ids in enumerate(layout.chunks):
        chunk_params = [
            params_by_name[str(pid)]
            for pid in chunk_param_ids
            if str(pid) in params_by_name
        ]
        if cid < n_persist:
            persistent_params.extend(chunk_params)
        else:
            cpu_params_per_chunk[ChunkId(cid)] = chunk_params

    gpu_optim: GpuFusedAdamAdapter | None = None
    cpu_optim: CpuFusedAdamAdapter | None = None
    if persistent_params:
        gpu_optim = GpuFusedAdamAdapter(
            params=persistent_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    # M7: for sharded non-persistent chunks the CPU Adam updates the
    # chunk's flat shard_param (one per rank slice) rather than the
    # user-facing per-param list.
    cpu_params_per_chunk_for_optim: dict[ChunkId, list["nn.Parameter"]] = {}
    for cid, chunk_params in cpu_params_per_chunk.items():
        shard_state = chunk_manager._chunk_shards.get(cid)  # type: ignore[attr-defined]
        if shard_state is not None:
            cpu_params_per_chunk_for_optim[cid] = [shard_state.shard_param]
        else:
            cpu_params_per_chunk_for_optim[cid] = chunk_params

    if any(params for params in cpu_params_per_chunk_for_optim.values()):
        try:
            cpu_optim = CpuFusedAdamAdapter(
                params_per_chunk=cpu_params_per_chunk_for_optim,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        except (ImportError, Exception) as err:  # noqa: BLE001 - see below
            # See ``protrain_model_wrapper``: DeepSpeed's CUDA-version
            # mismatch is a ``CUDAMismatchException`` that bypasses
            # ``ImportError``. Fall back to the inline GPU optimizer
            # path for non-persistent chunks.
            LOG.warning(
                "protrain_optimizer_wrapper: CPU FusedAdam unavailable (%s); "
                "non-persistent chunks will be stepped inline on the GPU optimizer. "
                "Install DeepSpeed for the async-overlap path.",
                err,
            )
            cpu_optim = None

    # Swap the freshly-built adapters into the chunk manager so the
    # scheduler's post_block_backward -> reduce_grads_and_offload ->
    # cpu_optim.step_async chain uses them.
    chunk_manager.cpu_optim = cpu_optim  # type: ignore[union-attr]
    chunk_manager.gpu_optim = gpu_optim  # type: ignore[union-attr]

    # Build the flat param list for the Optimizer base class.
    all_params: list["nn.Parameter"] = list(persistent_params)
    for params in cpu_params_per_chunk.values():
        all_params.extend(params)
    # Dedupe while preserving order — shared weights may appear twice.
    seen: set[int] = set()
    unique_params: list["nn.Parameter"] = []
    for p in all_params:
        if id(p) in seen:
            continue
        seen.add(id(p))
        unique_params.append(p)

    defaults: dict[str, Any] = dict(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    return _ProTrainOptimizer(
        gpu_optim=gpu_optim,
        cpu_optim=cpu_optim,
        params=unique_params,
        defaults=defaults,
        chunk_manager=chunk_manager,
    )


__all__ = ["protrain_optimizer_wrapper"]
