"""Allocate-before-use / free-after tensor context for profiling models > device memory.

M1 ships a PARTIAL implementation. The ``disabled`` fast path is a no-op context
manager used by the tiny-GPT2 test and the common 7B/13B case on a 3090 where
the forward pass fits normally. The ``enabled`` path is scaffolded with the
correct API shape but the replay logic raises ``NotImplementedError`` — full
replay-mode profiling is the M4 optimization called out in §3.2 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from axolotl.utils.logging import get_logger

from axolotl.integrations.protrain.types import OpRecord

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


@dataclass
class _LiveTensor:
    """Bookkeeping entry for a tensor currently materialized on GPU."""

    op_id: int
    tensor: Any  # torch.Tensor; Any here keeps import cost low


class OnDemandTensorMgr:
    """Context manager that materializes each op's inputs just-in-time.

    Disabled fast path
    ------------------
    When ``disabled=True`` (or the model fits on-device), the context manager
    is a no-op and the profiler runs a normal forward/backward pass. This is
    the M1 behavior for tiny-GPT2 and the default for any model that fits.

    Enabled replay-mode path (M4 follow-up)
    ---------------------------------------
    The caller first captures an op list (a "tape") with shape metadata, then
    re-enters this manager in replay mode. ``allocate_inputs`` materializes
    inputs for the next op; ``free_after`` releases them. Peak during profiling
    is then bounded by the largest single op rather than the full model
    footprint (§3.2). The replay driver itself is not wired up here — the
    method bodies raise ``NotImplementedError`` with a pointer to M4.

    The API shape is fixed so M4 can swap in the real implementation without
    touching the profiler driver.
    """

    def __init__(
        self,
        device: "torch.device | str | int | None" = None,
        *,
        disabled: bool = False,
    ) -> None:
        self.device = device
        self.disabled = disabled
        self._live: dict[int, _LiveTensor] = {}
        self._entered = False

    # ---- context-manager protocol --------------------------------------

    def __enter__(self) -> "OnDemandTensorMgr":
        self._entered = True
        if self.disabled:
            return self
        LOG.debug("OnDemandTensorMgr entered in replay mode (device=%s)", self.device)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._entered = False
        # Best-effort free of anything still live. Safe to call when disabled.
        self._live.clear()

    # ---- replay-mode API -----------------------------------------------

    def allocate_inputs(self, op: OpRecord) -> None:
        """Materialize the input tensors required by ``op`` on the GPU.

        Disabled fast path: no-op. Enabled path: not yet implemented — M4.
        """
        if self.disabled:
            return
        raise NotImplementedError(
            "on-demand replay TBD — M4 follow-up (profiler/on_demand.py). "
            "For M1 use disabled=True; the profiler runs a normal fwd+bwd."
        )

    def free_after(self, op: OpRecord) -> None:
        """Release any tensors allocated for ``op`` that no later op reads.

        Disabled fast path: no-op. Enabled path: not yet implemented — M4.
        """
        if self.disabled:
            return
        raise NotImplementedError(
            "on-demand replay TBD — M4 follow-up (profiler/on_demand.py)."
        )

    # ---- introspection --------------------------------------------------

    def live_tensor_ids(self) -> Iterable[int]:
        return tuple(self._live.keys())


__all__ = ["OnDemandTensorMgr"]
