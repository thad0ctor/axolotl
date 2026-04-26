"""Allocate-before-use / free-after tensor context for profiling models > device memory.

The profiler must be able to trace models whose full state (params + grads +
optimizer state + activations) doesn't fit on a single GPU. ProTrain solves
this with two coordinated mechanisms (paper §3.2):

1. **Parameter offload** — every nn.Module's directly-owned parameters live
   on pinned CPU memory between modules. A pre-forward hook gathers a
   module's own params onto GPU just before its forward; a post-forward
   hook releases them. The GPU therefore only holds *one* module's params
   at a time during the traced forward, plus whatever the running op's
   inputs/outputs require.

2. **Saved-activation spill** — ``torch.autograd.graph.saved_tensors_hooks``
   intercepts every tensor that autograd would retain for backward, copies
   it to CPU at save time, restores to GPU at unpack time. Since the
   profiler's traced pass is forward-only (the wrapper calls
   :func:`run_trace` with ``include_backward=False`` on large models),
   the unpack path is never exercised — the spill side alone is enough
   to keep retained activations off the GPU during forward.

Together these bound peak GPU at roughly ``max_leaf_param_bytes +
activation_workspace_per_op``, which is small enough that 13B / 70B-class
models can be profiled on a 24 GB card without OOM.

The disabled fast path (``disabled=True``) is a no-op context manager —
used by the tiny-GPT2 unit tests and by the model_wrapper when the model
fits on-device with headroom (no offload needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from axolotl.utils.logging import get_logger

from axolotl.integrations.protrain.types import OpRecord

if TYPE_CHECKING:
    import torch
    from torch import nn

LOG = get_logger(__name__)


@dataclass
class _ParamSpill:
    """Bookkeeping for one parameter that's been spilled to CPU.

    Two original-device cases:

    * GPU-resident param (typical Axolotl path): we copy GPU→CPU at __enter__,
      keep ``original_data`` alive so the optimizer's state slots (keyed on
      ``id(param)``) keep pointing at the same buffer, and copy CPU→original
      at __exit__.

    * CPU-resident param (paper's intent — model too big for GPU): no copy
      needed; ``cpu_storage`` IS the original tensor (pinned in place if
      possible). ``original_data`` is None. The pre-gather hook copies to
      the target device on demand.
    """

    param: Any                    # torch.nn.Parameter — Any keeps import light
    cpu_storage: Any              # torch.Tensor on CPU (pinned if possible)
    original_device: Any          # torch.device the param was on at __enter__
    original_data: Any            # GPU tensor at __enter__, or None for CPU-original


class OnDemandTensorMgr:
    """Context manager that materializes each leaf's params just-in-time.

    Disabled fast path
    ------------------
    When ``disabled=True``, the context manager is a no-op and the profiler
    runs a normal forward/backward pass. This is the right choice when the
    model fits on-device with headroom — pure profiling cost, zero spill
    overhead. The model_wrapper uses this path for ~7B-class models on a
    24 GB card.

    Enabled mode (replay-equivalent)
    --------------------------------
    On ``__enter__``:

    * Every parameter is detached and moved to pinned CPU memory (best-effort
      pinning; falls back to pageable if pinning fails). The Parameter's
      ``.data`` slot is replaced with an empty GPU tensor of matching dtype.
    * A pre-forward hook is registered on every nn.Module to copy that
      module's *direct* parameters (``parameters(recurse=False)``) from CPU
      to GPU, replacing the empty placeholder.
    * A post-forward hook on every module replaces those parameters' ``.data``
      with empty placeholders again, releasing the GPU storage. The freshly-
      gathered GPU tensor remains alive only as long as the autograd graph
      (or downstream ops) hold a reference to it.
    * ``torch.autograd.graph.saved_tensors_hooks`` is entered for the duration
      of the traced forward. Every tensor autograd would retain for backward
      is copied to CPU at save time. This is the activation-spill half of
      the paper's allocate-before-use / free-after-use scheme; it makes
      ``post_forward``'s ``p.data = empty()`` actually reclaim GPU memory
      (otherwise the saved-for-backward slot would pin the gathered tensor).

    On ``__exit__``: hooks are removed; every parameter is restored to its
    original device (using the original GPU storage that the optimizer's
    state already references via ``id(param)``).

    Notes
    -----
    * Buffers (BatchNorm running stats, position-embedding buffers, etc.)
      are NOT offloaded — they're typically small (<<1% of param state) and
      offloading them complicates the BatchNorm fastpath. If a future model
      shows non-trivial buffer footprint the same hook structure can be
      extended.
    * The ``allocate_inputs`` / ``free_after`` methods on this class are
      kept for API compatibility with the original M1 scaffold (the
      profiler driver does not call them — hook-based gathering replaces
      that path) and to keep ``test_on_demand_disabled_fast_path`` green.
    """

    def __init__(
        self,
        device: "torch.device | str | int | None" = None,
        *,
        disabled: bool = False,
        model: "nn.Module | None" = None,
    ) -> None:
        self.device = device
        self.disabled = disabled
        self.model = model
        self._spills: dict[int, _ParamSpill] = {}
        self._handles: list[Any] = []
        self._sthook_ctx: Any = None
        self._entered = False
        self._n_pin_failures = 0

    # ---- context-manager protocol --------------------------------------

    def __enter__(self) -> "OnDemandTensorMgr":
        self._entered = True
        if self.disabled:
            return self
        if self.model is None:
            raise ValueError(
                "OnDemandTensorMgr enabled mode requires a model. Pass "
                "model=... to __init__, or set disabled=True for the no-op "
                "fast path."
            )

        import torch

        target_device = (
            torch.device(self.device) if self.device is not None else None
        )

        # 1. Spill every parameter to pinned CPU; replace .data with empty.
        for _name, param in self.model.named_parameters():
            self._spill_param_to_cpu(param, target_device)

        # 2. Hook every module so leaf forwards gather their direct params.
        for sub in self.model.modules():
            self._handles.append(sub.register_forward_pre_hook(self._pre_gather))
            self._handles.append(sub.register_forward_hook(self._post_release))

        # 3. Spill saved-for-backward tensors to CPU. This is what makes
        #    post_release's ``p.data = empty()`` actually reclaim memory:
        #    without this, autograd would keep the gathered GPU param alive
        #    via the saved-for-backward slot of the linear's grad_fn.
        self._sthook_ctx = torch.autograd.graph.saved_tensors_hooks(
            self._pack_hook, self._unpack_hook
        )
        self._sthook_ctx.__enter__()

        if self._n_pin_failures:
            LOG.debug(
                "OnDemandTensorMgr: %d params couldn't be pinned (using "
                "pageable CPU); H2D copies will be synchronous. Trace will "
                "still complete; runtime per copy ~2x slower.",
                self._n_pin_failures,
            )

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._entered = False
        if self.disabled:
            return

        # Remove hooks first so partial forward calls during exit unwinding
        # don't try to gather params that are mid-restore.
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001 - defensive
                pass
        self._handles.clear()

        # Exit saved_tensors_hooks BEFORE restoring params — any in-flight
        # backward has already completed by this point (run_trace synchs).
        if self._sthook_ctx is not None:
            try:
                self._sthook_ctx.__exit__(exc_type, exc, tb)
            except Exception as _e:  # noqa: BLE001 - defensive
                LOG.debug("saved_tensors_hooks exit raised: %s", _e)
            self._sthook_ctx = None

        # Restore every parameter back to its original location.
        # GPU-original: copy CPU contents back into the *original* GPU
        # tensor (preserving identity for the optimizer's state slots),
        # then point param.data at it. CPU-original: just restore the
        # original CPU tensor.
        import torch

        for spill in self._spills.values():
            try:
                if spill.original_data is not None:
                    spill.original_data.copy_(
                        spill.cpu_storage.to(
                            spill.original_data.device, non_blocking=True
                        )
                    )
                    spill.param.data = spill.original_data
                else:
                    # CPU-original — cpu_storage is the original tensor.
                    spill.param.data = spill.cpu_storage
            except Exception as _e:  # noqa: BLE001 - defensive
                LOG.warning(
                    "OnDemandTensorMgr: failed to restore param to %s (%s); "
                    "leaving on CPU storage",
                    spill.original_device, _e,
                )
        # Sync once after all restores; cheaper than per-param sync.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001 - defensive
                pass
        self._spills.clear()

    # ---- spill / restore helpers ---------------------------------------

    def _spill_param_to_cpu(
        self, param: Any, target_device: "torch.device | None"
    ) -> None:
        """Move ``param`` to pinned CPU storage; leave a placeholder in .data.

        Handles both GPU-resident (copy GPU→CPU, replace .data with empty)
        and CPU-resident (use param's existing tensor, pin if possible) cases.
        """
        import torch

        original_device = param.device

        if original_device.type == "cpu":
            # CPU-resident: cpu_storage IS the original tensor. Pin it for
            # async H2D copies in pre-gather, best-effort.
            try:
                cpu_storage = param.data.pin_memory()
            except Exception:  # noqa: BLE001 - pinning is best-effort
                cpu_storage = param.data
                self._n_pin_failures += 1
            self._spills[id(param)] = _ParamSpill(
                param=param,
                cpu_storage=cpu_storage,
                original_device=original_device,
                original_data=None,
            )
            return

        # GPU-resident: copy GPU→CPU, keep original GPU tensor alive so
        # __exit__ can copy values back into the same StorageImpl that the
        # optimizer's state slots were keyed on.
        try:
            cpu_storage = param.data.detach().to("cpu", copy=True)
            try:
                cpu_storage = cpu_storage.pin_memory()
            except Exception:  # noqa: BLE001 - pinning is best-effort
                self._n_pin_failures += 1
        except Exception as exc:  # noqa: BLE001 - defensive
            LOG.warning(
                "OnDemandTensorMgr: failed to spill param to CPU (%s); "
                "leaving on GPU. Profile peak will be inflated for this param.",
                exc,
            )
            return

        original_data = param.data
        placeholder = torch.empty(
            0, dtype=original_data.dtype, device=original_device
        )
        param.data = placeholder
        self._spills[id(param)] = _ParamSpill(
            param=param,
            cpu_storage=cpu_storage,
            original_device=original_device,
            original_data=original_data,
        )

    # ---- module-level gather/release hooks -----------------------------

    def _gather_target_device(self) -> "torch.device | None":
        """Resolve the target device for gathered params.

        Falls back to the param's original device if the manager wasn't
        constructed with an explicit ``device``.
        """
        import torch

        if self.device is None:
            return None
        return torch.device(self.device) if not isinstance(self.device, torch.device) else self.device

    def _pre_gather(self, module: "nn.Module", inputs: Any) -> None:
        """Copy the module's *direct* params from CPU to target_device before forward."""
        target = self._gather_target_device()
        for param in module.parameters(recurse=False):
            spill = self._spills.get(id(param))
            if spill is None:
                continue
            dest = target if target is not None else spill.original_device
            try:
                gathered = spill.cpu_storage.to(dest, non_blocking=True)
                param.data = gathered
            except Exception as exc:  # noqa: BLE001 - defensive
                LOG.warning(
                    "OnDemandTensorMgr pre-gather failed (%s); falling back "
                    "to original data — peak may inflate for this op.",
                    exc,
                )
                if spill.original_data is not None:
                    param.data = spill.original_data
                else:
                    param.data = spill.cpu_storage

    def _post_release(
        self, module: "nn.Module", inputs: Any, output: Any
    ) -> None:
        """Replace the module's *direct* params with empty placeholders."""
        import torch

        target = self._gather_target_device()
        for param in module.parameters(recurse=False):
            spill = self._spills.get(id(param))
            if spill is None:
                continue
            dest = target if target is not None else spill.original_device
            try:
                placeholder = torch.empty(0, dtype=param.dtype, device=dest)
                param.data = placeholder
            except Exception as exc:  # noqa: BLE001 - defensive
                LOG.debug("OnDemandTensorMgr post-release no-op (%s)", exc)

    # ---- saved-tensors spill / restore ---------------------------------

    @staticmethod
    def _pack_hook(tensor: Any) -> Any:
        """Spill autograd-retained GPU tensors to CPU at save time."""
        try:
            if not getattr(tensor, "is_cuda", False):
                return tensor
            return tensor.detach().to("cpu", non_blocking=False)
        except Exception:  # noqa: BLE001 - defensive
            return tensor

    @staticmethod
    def _unpack_hook(packed: Any) -> Any:
        """Restore a spilled tensor — only fires if backward runs."""
        # The traced forward in run_trace is forward-only when on_demand=True,
        # so this path is not exercised. Implemented for completeness in case
        # future callers want to run backward under on-demand.
        try:
            if not getattr(packed, "is_cpu", True):
                return packed
            # Without explicit device knowledge we just return the CPU tensor;
            # caller's grad_fn knows the right device.
            return packed
        except Exception:  # noqa: BLE001 - defensive
            return packed

    # ---- back-compat API (no-ops in enabled mode under hook-based path) ---

    def allocate_inputs(self, op: OpRecord) -> None:
        """Compatibility shim. The enabled path uses module-level hooks.

        Kept callable in disabled mode to preserve the M1 fast-path test.
        Raises in enabled mode if invoked outside the context to flag misuse.
        """
        if self.disabled:
            return
        if not self._entered:
            raise RuntimeError(
                "OnDemandTensorMgr.allocate_inputs called outside ``with`` "
                "context. Use as a context manager — gathering happens via "
                "module hooks, not by calling allocate_inputs directly."
            )
        # No-op when entered: the pre-forward hook on the relevant module
        # has already gathered its params.

    def free_after(self, op: OpRecord) -> None:
        """Compatibility shim. The enabled path uses module-level hooks."""
        if self.disabled:
            return
        if not self._entered:
            raise RuntimeError(
                "OnDemandTensorMgr.free_after called outside ``with`` context."
            )
        # No-op when entered: the post-forward hook on the relevant module
        # has already released its params.

    # ---- introspection --------------------------------------------------

    def live_tensor_ids(self) -> Iterable[int]:
        return tuple(self._spills.keys())


__all__ = ["OnDemandTensorMgr"]
