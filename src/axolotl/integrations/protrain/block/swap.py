"""Activation-swap wrapper — interface-only stub for M3.

SWAP mode in the ProTrain three-way block strategy (§3.1.2): forward
activations are offloaded to pinned CPU memory, then prefetched back
during backward. On RTX 3090 (communication-bound, no NVLink) the
searcher almost never selects ``n_swap > 0``, so M3 only provides the
wrapper surface; the full prefetch scheduler lands in M4.

Gating
------
Constructing ``SwappedBlock`` raises ``RuntimeError`` unless the process
has ``PROTRAIN_ENABLE_SWAP=1`` set. This is an intentional
feature-flag to prevent accidental use before M4's scheduler provides
end-to-end overlap.

When enabled, the forward pass runs the block normally and schedules an
async ``.to('cpu', non_blocking=True)`` copy on the output activation.
The backward path schedules an async ``.to('cuda', non_blocking=True)``
before the block's gradient computation. These are placeholders — **M4's
scheduler drives the actual overlap**. Without the scheduler the copies
still happen, but there is no pipelining, so peak memory is unaffected
and throughput degrades. Hence the feature flag.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


_ENV_FLAG = "PROTRAIN_ENABLE_SWAP"


def _swap_enabled() -> bool:
    """True iff the env flag is set to a truthy value (``"1"``)."""
    return os.environ.get(_ENV_FLAG, "0") == "1"


class _SwapOffloadFunction(torch.autograd.Function):
    """Autograd hook pair: offload in forward, prefetch in backward.

    This is a **stub**. M4's scheduler replaces the synchronous copy
    with a stream-scheduled, bandwidth-budgeted transfer.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Record device so backward knows where to prefetch to.
        ctx.src_device = tensor.device
        # Schedule async D2H. The returned tensor stays on GPU so the rest
        # of forward keeps working; the offloaded copy is saved for bwd.
        if tensor.is_cuda:
            cpu_copy = tensor.detach().to("cpu", non_blocking=True)
            ctx.save_for_backward(cpu_copy)
        else:
            ctx.save_for_backward(tensor.detach())
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        (saved,) = ctx.saved_tensors
        if saved.device != ctx.src_device:
            # Prefetch H2D before gradient computation continues upstream.
            saved = saved.to(ctx.src_device, non_blocking=True)
        # We only offloaded the activation for memory; grads flow through
        # unchanged. The reloaded tensor is dropped — scheduler (M4) will
        # replace this with an actual storage swap.
        del saved
        return grad_output


class SwappedBlock(nn.Module):
    """Wrap an ``nn.Module`` with the swap interface.

    M3 contract: construction gated by ``PROTRAIN_ENABLE_SWAP``; forward
    runs the block and registers offload/prefetch hooks on the output
    activation; backward is driven by autograd. Actual bandwidth-aware
    scheduling lands in M4.
    """

    def __init__(self, block: nn.Module) -> None:
        if not _swap_enabled():
            raise RuntimeError(
                "SWAP block mode is experimental; set PROTRAIN_ENABLE_SWAP=1 to enable."
            )
        super().__init__()
        self.block = block
        self._protrain_wrapped_mode: BlockMode = BlockMode.SWAP
        LOG.debug(
            "SwappedBlock constructed (stub mode; M4 scheduler drives actual overlap)"
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = self.block(*args, **kwargs)
        # Only the primary tensor output gets the swap hook. HF blocks
        # often return a tuple; wrap the first element and leave the rest
        # (masks, KV caches) untouched.
        if isinstance(out, torch.Tensor):
            return _SwapOffloadFunction.apply(out)
        if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            hooked = _SwapOffloadFunction.apply(out[0])
            return (hooked, *out[1:])
        return out

    def extra_repr(self) -> str:
        return f"mode={self._protrain_wrapped_mode.value}"


__all__ = ["SwappedBlock"]
