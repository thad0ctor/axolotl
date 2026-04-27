"""Gradient-checkpointing wrapper for a single transformer block.

CKPT mode in the ProTrain three-way block strategy (§3.1.2). The wrapper
defers to ``torch.utils.checkpoint.checkpoint`` with ``use_reentrant=False``
so activations for the wrapped block are dropped after forward and
recomputed during backward.

Kwargs handling
---------------
HuggingFace transformer blocks take positional tensors plus keyword
arguments such as ``attention_mask``, ``position_ids``, ``past_key_value``,
``output_attentions``, ``use_cache``. The functional form of
``torch.utils.checkpoint.checkpoint`` only forwards positional arguments to
the wrapped function (kwargs are consumed by the checkpoint machinery
itself, not passed through). To route kwargs correctly we build a closure
that captures the kwargs dict and applies it internally, then pass only
positional tensors into ``checkpoint``. This preserves the block's native
call signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.utils.checkpoint as torch_checkpoint
from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class CheckpointedBlock(nn.Module):
    """Wrap an ``nn.Module`` so its forward activations are recomputed in backward.

    Marks the wrapper with ``_protrain_wrapped_mode = BlockMode.CKPT`` so the
    dispatcher can recognise and unwrap it idempotently.
    """

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block
        # Public marker consumed by dispatcher.unwrap_block and inspection code.
        self._protrain_wrapped_mode: BlockMode = BlockMode.CKPT
        # Optional callback installed by runtime.hooks. It re-gathers
        # this block's parameter chunks before checkpoint recompute,
        # because the recompute calls ``self.block`` directly and does
        # not pass through hooks attached to this wrapper module.
        self._protrain_recompute_pre_hook: Callable[[], None] | None = None

    def set_recompute_pre_hook(self, hook: Callable[[], None] | None) -> None:
        """Install a callback run before both original and recompute forwards."""
        self._protrain_recompute_pre_hook = hook

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # torch.utils.checkpoint.checkpoint only threads positional args into
        # the wrapped callable. Capture kwargs in a closure so HF blocks that
        # rely on e.g. attention_mask= still see them.
        block = self.block

        def _run(*inner_args: Any) -> Any:
            hook = self._protrain_recompute_pre_hook
            if hook is not None:
                hook()
            return block(*inner_args, **kwargs)

        return torch_checkpoint.checkpoint(
            _run,
            *args,
            use_reentrant=False,
        )

    def extra_repr(self) -> str:
        return f"mode={self._protrain_wrapped_mode.value}"


__all__ = ["CheckpointedBlock"]


# Silence unused import warnings when torch is present only for type hints.
_ = torch
