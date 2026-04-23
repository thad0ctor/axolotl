"""Precise-size pinned host memory (Appendix B.2).

PyTorch's default ``CUDAHostAllocator`` rounds up pinned allocations to the
next power of two. For ``n_buffer * S_chunk`` that can waste hundreds of MB
on large chunks. We instead call ``cudaHostAlloc`` directly through
``ctypes`` for an exact byte count, and hand out zero-copy ``torch.Tensor``
views over the resulting buffer.

If the ``libcudart`` lookup fails (e.g. the system's CUDA runtime isn't
visible to ``ctypes.CDLL`` despite ``torch.cuda`` being available), we fall
back to ``torch.empty(size, pin_memory=True)`` and flag
``_is_precise_size = False`` so tests can detect and skip assertions that
depend on exact sizing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import TYPE_CHECKING

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)

# cudaHostAllocDefault from cuda_runtime_api.h: "Default page-locked allocation flag".
_CUDA_HOST_ALLOC_DEFAULT = 0
_CUDA_SUCCESS = 0


def _load_cudart() -> ctypes.CDLL | None:
    """Locate ``libcudart`` via several common names; return None if unavailable."""
    # ``torch.cuda.cudart()`` returns the loaded cudart handle on recent torch
    # versions; prefer that so we use exactly the same runtime torch linked
    # against. Fall back to ``ctypes.util.find_library`` / common SONAMEs.
    try:
        import torch

        handle = torch.cuda.cudart()
        if handle is not None:
            return handle  # type: ignore[return-value]
    except Exception as err:  # noqa: BLE001 — broad: torch may not even expose cudart
        LOG.debug("torch.cuda.cudart() unavailable: %s", err)

    for name in ("cudart", "libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            path = ctypes.util.find_library(name) or name
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


class PinnedHostMemory:
    """One large precise-size pinned host allocation split into ``n_buffer`` slots.

    Memory is allocated once in ``__init__`` and freed once in ``__del__``
    (or via :meth:`close`). Slots are contiguous and identically sized —
    ``buffer(i)`` hands out the ``i``-th slot as a pinned ``torch.Tensor``.
    """

    def __init__(self, n_buffer: int, S_chunk: int) -> None:
        if n_buffer <= 0:
            raise ValueError(f"n_buffer must be positive, got {n_buffer}")
        if S_chunk <= 0:
            raise ValueError(f"S_chunk must be positive, got {S_chunk}")

        self.n_buffer = int(n_buffer)
        self.S_chunk = int(S_chunk)
        self.total_bytes = self.n_buffer * self.S_chunk

        self._cudart: ctypes.CDLL | None = None
        self._ptr: int = 0  # device-facing pointer value (host-side VA)
        self._closed = False
        self._fallback_tensor: "torch.Tensor | None" = None
        self._torch_tensor: "torch.Tensor | None" = None
        self._is_precise_size: bool = False

        cudart = _load_cudart()
        if cudart is None:
            LOG.warning(
                "PinnedHostMemory: libcudart not found via ctypes; "
                "falling back to torch.empty(pin_memory=True). "
                "Pinned buffer may be rounded to a power of two."
            )
            self._init_fallback()
            return

        try:
            self._init_cudart(cudart)
        except Exception as err:  # noqa: BLE001
            LOG.warning(
                "PinnedHostMemory: ctypes cudaHostAlloc path failed (%s); "
                "falling back to torch.empty(pin_memory=True).",
                err,
            )
            self._init_fallback()

    # ---- initialization paths ------------------------------------------

    def _init_cudart(self, cudart: ctypes.CDLL) -> None:
        import torch

        # cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
        try:
            cudart.cudaHostAlloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            cudart.cudaHostAlloc.restype = ctypes.c_int
            cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
            cudart.cudaFreeHost.restype = ctypes.c_int
        except AttributeError as err:
            raise RuntimeError(f"cudart missing required symbol: {err}") from err

        ptr = ctypes.c_void_p(0)
        status = cudart.cudaHostAlloc(
            ctypes.byref(ptr),
            ctypes.c_size_t(self.total_bytes),
            ctypes.c_uint(_CUDA_HOST_ALLOC_DEFAULT),
        )
        if status != _CUDA_SUCCESS or not ptr.value:
            raise RuntimeError(
                f"cudaHostAlloc returned status={status} ptr={ptr.value} "
                f"for size={self.total_bytes}"
            )

        self._cudart = cudart
        self._ptr = int(ptr.value)
        self._is_precise_size = True

        # Build a single torch.Tensor viewing the whole region as uint8. We
        # use ``torch.frombuffer`` on a ``ctypes`` array cast so the tensor
        # shares storage with our cudaHostAlloc'd region with no copy.
        ArrayT = ctypes.c_uint8 * self.total_bytes
        # ``ArrayT.from_address(ptr)`` produces a ctypes array backed by the
        # pinned host region. ``torch.frombuffer`` takes any object that
        # supports the buffer protocol and exposes it as a zero-copy tensor.
        buf = ArrayT.from_address(self._ptr)
        self._torch_tensor = torch.frombuffer(buf, dtype=torch.uint8)
        # The buffer-protocol path doesn't carry the ``pin_memory`` flag
        # because PyTorch only sets that for allocations it made itself.
        # The underlying memory IS pinned (we called cudaHostAlloc), just
        # torch can't prove it. ``is_pinned()`` will therefore return False
        # on this path despite the memory being physically pinned. Callers
        # inspecting ``_is_precise_size`` know we're on the ctypes path.

    def _init_fallback(self) -> None:
        import torch

        self._fallback_tensor = torch.empty(
            self.total_bytes, dtype=torch.uint8, pin_memory=True
        )
        self._torch_tensor = self._fallback_tensor
        self._is_precise_size = False

    # ---- public API ----------------------------------------------------

    @property
    def is_precise_size(self) -> bool:
        """True iff the underlying bytes == exactly ``n_buffer * S_chunk``."""
        return self._is_precise_size

    def buffer(self, i: int) -> "torch.Tensor":
        """Return the ``i``-th slot as a 1D ``uint8`` tensor of length ``S_chunk``.

        The returned view shares storage with the pinned region; writes are
        immediately visible to CUDA transfers that use the same host pointer.
        """
        if self._closed:
            raise RuntimeError("PinnedHostMemory is closed")
        if not 0 <= i < self.n_buffer:
            raise IndexError(f"buffer index {i} out of range [0, {self.n_buffer})")
        assert self._torch_tensor is not None
        start = i * self.S_chunk
        return self._torch_tensor.narrow(0, start, self.S_chunk)

    def close(self) -> None:
        """Free the pinned allocation. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Drop torch views first so no tensor outlives the underlying memory.
        self._torch_tensor = None
        self._fallback_tensor = None
        if self._cudart is not None and self._ptr:
            status = self._cudart.cudaFreeHost(ctypes.c_void_p(self._ptr))
            if status != _CUDA_SUCCESS:
                LOG.warning("cudaFreeHost returned status=%d", status)
            self._ptr = 0
            self._cudart = None

    def __del__(self) -> None:  # noqa: D401
        try:
            self.close()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass


__all__ = ["PinnedHostMemory"]
