"""Fused-Adam adapters for persistent (GPU) and non-persistent (CPU) chunks.

Two classes with a similar shape:

* :class:`CpuFusedAdamAdapter` wraps ``deepspeed.ops.adam.DeepSpeedCPUAdam``
  and adds a ``step_async(chunk_id)`` path so the CPU optimizer step for
  chunk ``c`` can launch the instant that chunk's grads have been
  reduce-offloaded — overlapping with GPU backward for later chunks (§5).
* :class:`GpuFusedAdamAdapter` wraps Apex ``FusedAdam`` (or falls back to
  ``torch.optim.AdamW`` with a warning) for the persistent-resident subset.

Async semantics: we use a single-worker ``ThreadPoolExecutor``. DeepSpeed's
CPU Adam kernel releases the GIL inside its compiled op, so "async" here
means "run overlapped with the GPU kernels the main Python thread is
launching", not parallel across chunks. Serializing through one worker also
sidesteps the CPU Adam op's internal state sharing between chunks of the
same optimizer instance.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Iterable

from axolotl.integrations.protrain.types import ChunkId
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# CPU FusedAdam — non-persistent chunks
# ---------------------------------------------------------------------------


class CpuFusedAdamAdapter:
    """Per-chunk CPU FusedAdam driver for the non-persistent chunk set.

    We construct one underlying ``DeepSpeedCPUAdam`` instance per chunk.
    That matches the design where each non-persistent chunk's params live
    on CPU (sharded), their gradients are reduced and D2H-copied back to
    the same shard, and the CPU step consumes them in place. Keeping the
    instances separate per chunk means :meth:`step_async` can target
    exactly one chunk's param group without touching the others.
    """

    def __init__(
        self,
        params_per_chunk: dict[ChunkId, list["nn.Parameter"]],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError(
                "CpuFusedAdamAdapter requires DeepSpeed's CPU Adam kernel — "
                "install via `pip install axolotl[deepspeed]`."
            ) from err

        self._DeepSpeedCPUAdam = DeepSpeedCPUAdam
        self._params_per_chunk = dict(params_per_chunk)
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        # One DeepSpeedCPUAdam per chunk — cheap; shares no state.
        self._optims: dict[ChunkId, Any] = {}
        for cid, params in self._params_per_chunk.items():
            if not params:
                continue
            self._optims[cid] = DeepSpeedCPUAdam(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )

        # Single-worker executor — see module docstring for rationale.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="protrain-cpu-adam"
        )
        self._pending: dict[ChunkId, Future[None]] = {}

    # ---- step interface -------------------------------------------------

    def step_async(self, chunk_id: ChunkId) -> "Future[None]":
        """Submit the CPU Adam step for ``chunk_id`` to the worker thread.

        Idempotent with :meth:`wait`: if a prior step is still pending for
        the same chunk, we wait for it first so we never run two steps
        concurrently against the same param shard.
        """
        prev = self._pending.get(chunk_id)
        if prev is not None and not prev.done():
            prev.result()  # propagate any exception
        optim = self._optims.get(chunk_id)
        if optim is None:
            # No params belonging to this chunk live on CPU (e.g. a fully
            # persistent layout). Return an already-completed future.
            fut: Future[None] = Future()
            fut.set_result(None)
            self._pending[chunk_id] = fut
            return fut

        fut = self._executor.submit(optim.step)
        self._pending[chunk_id] = fut
        return fut

    def wait(self, chunk_id: ChunkId) -> None:
        """Block until ``step_async(chunk_id)``'s worker has finished."""
        fut = self._pending.get(chunk_id)
        if fut is None:
            return
        fut.result()  # re-raises worker exceptions on the caller's thread

    def wait_all(self) -> None:
        """Block until every in-flight chunk step has finished."""
        for fut in list(self._pending.values()):
            fut.result()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients across every chunk's params."""
        for optim in self._optims.values():
            optim.zero_grad(set_to_none=set_to_none)

    # ---- lifecycle ------------------------------------------------------

    def shutdown(self) -> None:
        """Tear down the worker pool. Call explicitly before process exit."""
        self.wait_all()
        self._executor.shutdown(wait=True)

    def __del__(self) -> None:  # noqa: D401
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass


# ---------------------------------------------------------------------------
# GPU FusedAdam — persistent chunks
# ---------------------------------------------------------------------------


class GpuFusedAdamAdapter:
    """Synchronous fused GPU Adam for the persistent chunk set.

    Prefers ``apex.optimizers.FusedAdam`` (paper-cited backend). Falls back
    to stock ``torch.optim.AdamW`` with a warning when Apex is unavailable
    — the cost model will be off in that case (AdamW is a distinct update
    rule, not just a different kernel) but training stays correct.
    """

    def __init__(
        self,
        params: Iterable["nn.Parameter"],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        param_list = [p for p in params if p is not None]

        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        optim = self._build_optim(param_list)
        self._optim = optim

    def _build_optim(self, params: list["nn.Parameter"]) -> Any:
        try:
            from apex.optimizers import FusedAdam  # type: ignore[import-not-found]

            return FusedAdam(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        except ImportError:
            LOG.warning(
                "apex.optimizers.FusedAdam unavailable; falling back to "
                "torch.optim.AdamW for the persistent-chunk optimizer. "
                "Install Apex for the paper-configured fused kernel."
            )

        import torch

        return torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

    # ---- step interface -------------------------------------------------

    def step(self) -> None:
        """Synchronous fused GPU Adam step over persistent-chunk params."""
        self._optim.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optim.zero_grad(set_to_none=set_to_none)

    @property
    def underlying(self) -> Any:
        """The wrapped optimizer instance (useful for LR schedulers)."""
        return self._optim


__all__ = ["CpuFusedAdamAdapter", "GpuFusedAdamAdapter"]
