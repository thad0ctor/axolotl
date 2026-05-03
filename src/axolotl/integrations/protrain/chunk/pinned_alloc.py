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
    """Locate ``libcudart`` as a ``ctypes.CDLL`` handle; return None if unavailable.

    On recent PyTorch builds ``torch.cuda.cudart()`` returns a Python module
    (``torch._C._cudart``) rather than a ``ctypes.CDLL`` — the symbols are
    not the raw C functions we need to set ``argtypes``/``restype`` on, so
    we skip that path entirely and load the shared object directly via
    ``ctypes``. We try a handful of common SONAMEs (CUDA 11, 12, 13) and
    finally ``ctypes.util.find_library('cudart')`` which resolves to
    whichever ``libcudart.so.*`` ``ldconfig`` knows about.
    """
    # Explicit SONAMEs come first so we prefer a specific major version if
    # more than one is on the library search path. ``libcudart.so`` is the
    # unversioned symlink (only present with -dev packages); the versioned
    # names are what end-user CUDA toolkits install.
    candidates: list[str] = [
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    # Let ctypes locate whatever the current ld cache has, too.
    resolved = ctypes.util.find_library("cudart")
    if resolved:
        candidates.append(resolved)

    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class PinnedHostMemory:
    """One large precise-size pinned host allocation split into ``n_buffer`` slots.

    Memory is allocated once in ``__init__`` and freed once in ``__del__``
    (or via :meth:`close`). Slots are contiguous and identically sized —
    ``buffer(i)`` hands out the ``i``-th slot as a pinned ``torch.Tensor``.

    Lifetime hazard
    ---------------
    ``buffer(i)`` returns a ``narrow()`` view sharing storage with the
    underlying pinned region. If ``close()`` is called while a caller
    still holds such a view, the view becomes a dangling pointer —
    subsequent reads/writes (including async H2D copies) will touch
    freed memory. To guard against this, ``buffer(i)`` increments a
    borrow counter that the caller must decrement via
    :meth:`release_buffer` once the slot is no longer in use (the
    canonical pattern is acquire-via-``buffer`` then
    ``record_stream`` + ``release_buffer`` after enqueueing the H2D
    copy). :meth:`close` raises ``RuntimeError`` if any borrow is
    still outstanding so the lifetime violation is loud rather than
    silent.
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
        # Outstanding views handed out by ``buffer(i)`` that have not yet
        # been returned via ``release_buffer(i)``. Used by ``close()`` to
        # refuse free-while-borrowed (use-after-free guard).
        self._live_borrows: int = 0

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

        The slot is considered borrowed until the caller pairs this call
        with :meth:`release_buffer`. ``close()`` will refuse to free the
        underlying pinned region while any borrow is still outstanding
        (see the class docstring for the use-after-free hazard).
        """
        if self._closed:
            raise RuntimeError("PinnedHostMemory is closed")
        if not 0 <= i < self.n_buffer:
            raise IndexError(f"buffer index {i} out of range [0, {self.n_buffer})")
        assert self._torch_tensor is not None
        start = i * self.S_chunk
        view = self._torch_tensor.narrow(0, start, self.S_chunk)
        self._live_borrows += 1
        return view

    def release_buffer(self, i: int) -> None:
        """Decrement the borrow counter for slot ``i``.

        Pairs with :meth:`buffer`. The counter is the only ownership
        signal :meth:`close` consults; failing to release leaves
        ``close()`` raising. Index validation is best-effort so this
        is safe to call from cleanup paths even if the slot id was
        never borrowed in this allocator (logged but not fatal — we
        prefer not to derail destructor flows).
        """
        if not 0 <= i < self.n_buffer:
            LOG.warning(
                "PinnedHostMemory.release_buffer: index %d out of range "
                "[0, %d); ignored",
                i,
                self.n_buffer,
            )
            return
        if self._live_borrows <= 0:
            LOG.warning(
                "PinnedHostMemory.release_buffer(%d): no outstanding borrow; "
                "double-release?",
                i,
            )
            return
        self._live_borrows -= 1

    def close(self) -> None:
        """Free the pinned allocation. Idempotent.

        Raises ``RuntimeError`` if any slot view returned by
        :meth:`buffer` has not been returned via :meth:`release_buffer`
        — freeing the underlying pinned region while views are still
        live can create dangling pointers and silently corrupt any
        in-flight H2D copy or host write that targets the slot. The
        explicit ``close()`` path is the user-controlled deterministic
        teardown surface, so we want loud failure on lifetime
        violations. Destructor-driven cleanup falls through
        :meth:`__del__`, which logs and force-frees because destructors
        must not raise.
        """
        if self._closed:
            return
        if self._live_borrows > 0:
            raise RuntimeError(
                f"PinnedHostMemory.close(): {self._live_borrows} slot view(s) "
                "still borrowed; release them via release_buffer() before close() "
                "to avoid use-after-free on the pinned region."
            )
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
        # Destructors must not throw, so the borrow guard in ``close()``
        # is bypassed here: if the user dropped the allocator with views
        # outstanding it is too late to ask them to release. We log loudly
        # and force the free so we don't leak pinned memory at process
        # shutdown. The view-holders will fault if they touch the region
        # after this — that is the original hazard, surfaced rather than
        # hidden.
        try:
            if self._closed:
                return
            if self._live_borrows > 0:
                LOG.warning(
                    "PinnedHostMemory.__del__: %d slot view(s) still borrowed "
                    "at GC time; forcing free. Holders touching the region "
                    "after this point will hit freed memory.",
                    self._live_borrows,
                )
                self._live_borrows = 0
            self.close()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass


__all__ = ["PinnedHostMemory"]
