"""Block-granularity forward/backward hooks for the ProTrain runtime.

``install_hooks`` attaches four hooks per transformer block:

* forward-pre hook -> :meth:`Scheduler.pre_block_forward`
* forward-post hook -> :meth:`Scheduler.post_block_forward`
* backward-pre hook -> :meth:`Scheduler.pre_block_backward`
* backward-post hook -> :meth:`Scheduler.post_block_backward`

The hooks operate at **block** granularity only — op-level hooks are
the profiler's job (M1). This module's contract is to wire the already-
wrapped blocks (see :mod:`axolotl.integrations.protrain.block.dispatcher`)
into the scheduler's prefetch / release / reduce-offload machine.

Ordering note: ``protrain_model_wrapper`` wraps every block *before*
installing these hooks, so the hooks attach to the post-wrap modules
(``CheckpointedBlock`` / ``SwappedBlock`` / identity). The wrapper
idempotency guarantee means a re-search at epoch boundaries can
uninstall + re-wrap + re-install without any hook-level bookkeeping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import nn

from axolotl.integrations.protrain.block.layout_rules import discover_blocks
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockStrategyMap,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

    from axolotl.integrations.protrain.chunk import ChunkManager
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler

LOG = get_logger(__name__)


class _RecomputePreHookHandle:
    """Small removable handle for CheckpointedBlock recompute callbacks."""

    def __init__(self, module: nn.Module) -> None:
        self._module: nn.Module | None = module

    def remove(self) -> None:
        module = self._module
        if module is not None and hasattr(module, "set_recompute_pre_hook"):
            module.set_recompute_pre_hook(None)
        self._module = None


def _make_forward_pre_hook(scheduler: "Scheduler", block_id: BlockId):
    def _hook(module: nn.Module, inputs):  # noqa: ARG001 — signature required
        scheduler.pre_block_forward(block_id)
        return None  # allow default arg flow

    return _hook


def _make_forward_post_hook(scheduler: "Scheduler", block_id: BlockId):
    def _hook(module: nn.Module, inputs, output):  # noqa: ARG001
        scheduler.post_block_forward(block_id)
        return None

    return _hook


def _make_backward_pre_hook(scheduler: "Scheduler", block_id: BlockId):
    def _hook(module: nn.Module, grad_output):  # noqa: ARG001
        scheduler.pre_block_backward(block_id)
        return None

    return _hook


def _make_backward_post_hook(scheduler: "Scheduler", block_id: BlockId):
    def _hook(module: nn.Module, grad_input, grad_output):  # noqa: ARG001
        scheduler.post_block_backward(block_id)
        return None

    return _hook


def install_hooks(
    model: nn.Module,
    chunk_manager: "ChunkManager",  # noqa: ARG001 — reserved for future use
    block_map: BlockStrategyMap,  # noqa: ARG001 — scheduler already owns this
    scheduler: "Scheduler",
) -> list["RemovableHandle"]:
    """Attach the four-per-block scheduler hooks.

    The ``chunk_manager`` and ``block_map`` parameters are accepted for
    API symmetry with the design doc but are not consulted directly —
    the scheduler already holds references to both. Keeping them in the
    signature lets the plugin (M5) compose ``install_hooks`` without
    reaching into the ``Scheduler``'s private state.

    Parameters
    ----------
    model:
        The user model, post-block-wrapping. ``discover_blocks`` runs
        against this to locate the transformer-block ModuleList.
    chunk_manager:
        Runtime chunk driver. Reserved.
    block_map:
        Per-block activation mode. Reserved.
    scheduler:
        The :class:`Scheduler` instance that owns the prefetch stream
        and the per-block entry points.

    Returns
    -------
    list[RemovableHandle]
        One ``RemovableHandle`` per installed hook — pass to
        :func:`uninstall_hooks` to restore the model to its pre-install
        state.
    """
    blocks = discover_blocks(model)

    handles: list["RemovableHandle"] = []
    for idx, block in enumerate(blocks):
        block_id = cast(BlockId, idx)

        handles.append(
            block.register_forward_pre_hook(_make_forward_pre_hook(scheduler, block_id))
        )
        handles.append(
            block.register_forward_hook(_make_forward_post_hook(scheduler, block_id))
        )
        # ``register_full_backward_pre_hook`` exists on nn.Module from
        # PyTorch >= 2.0. We use the "full" variant so the hook observes
        # grads to the entire block, not just the last parameter.
        handles.append(
            block.register_full_backward_pre_hook(
                _make_backward_pre_hook(scheduler, block_id)
            )
        )
        handles.append(
            block.register_full_backward_hook(
                _make_backward_post_hook(scheduler, block_id)
            )
        )
        if hasattr(block, "set_recompute_pre_hook"):
            block.set_recompute_pre_hook(
                lambda block_id=block_id: scheduler.ensure_block_resident(block_id)
            )
            handles.append(_RecomputePreHookHandle(block))  # type: ignore[arg-type]

    LOG.debug(
        "install_hooks: attached %d handles across %d transformer blocks",
        len(handles),
        len(blocks),
    )
    return handles


def uninstall_hooks(handles: list["RemovableHandle"]) -> None:
    """Remove every handle produced by :func:`install_hooks`.

    Safe to call multiple times — ``RemovableHandle.remove`` is
    idempotent in modern PyTorch.
    """
    for h in handles:
        try:
            h.remove()
        except Exception as exc:  # noqa: BLE001 — best-effort removal
            LOG.warning("uninstall_hooks: handle.remove() failed: %s", exc)
    handles.clear()


__all__ = ["install_hooks", "uninstall_hooks"]
