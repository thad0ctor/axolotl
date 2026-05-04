"""Single-stream memory allocation context (Appendix B.2).

PyTorch's caching allocator maintains a *per-stream* free list — a tensor
freed on stream A cannot be reused for an allocation on stream B without
``record_stream`` hand-holding. ProTrain sidesteps this entirely by
routing all chunk-manager allocations through a single managed stream
(the default stream by default). That way the allocator has a single
heap to amortize across prefetch, gather, offload, and optimizer
allocations, and we never need ``record_stream`` calls.

This module ships a minimal context-manager API. Full integration with
the chunk manager's gather/offload happens at call sites in M4
(runtime/scheduler.py is not part of M2).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


class SingleStreamAllocator:
    """Context manager forcing allocations onto one managed CUDA stream.

    Usage::

        alloc = SingleStreamAllocator()  # uses the default stream
        with alloc:
            buf = torch.empty(...)
        alloc.sync()

    The context is a thin wrapper over ``torch.cuda.stream(stream)``:
    inside the ``with`` block the current stream is set to ``self.stream``
    so any allocations made from Python-side code land on that stream.
    Exiting the context restores the previous current stream.

    Reentrancy: the wrapper is safe to nest with itself, but like all
    ``torch.cuda.stream`` usage it is not thread-safe.
    """

    def __init__(self, stream: "torch.cuda.Stream | None" = None) -> None:
        # Import lazily so the module remains importable without a CUDA
        # runtime (matters for docs builds and syntax-only CI lanes).
        import torch

        self._torch = torch
        if stream is None:
            if not torch.cuda.is_available():
                LOG.debug(
                    "SingleStreamAllocator constructed without CUDA available; "
                    "stream operations will be no-ops."
                )
                self.stream: "torch.cuda.Stream | None" = None
            else:
                self.stream = torch.cuda.default_stream()
        else:
            self.stream = stream

        self._ctx_stack: list[AbstractContextManager[object]] = []

    def __enter__(self) -> "SingleStreamAllocator":
        if self.stream is None:
            return self
        ctx = self._torch.cuda.stream(self.stream)
        ctx.__enter__()
        self._ctx_stack.append(ctx)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._ctx_stack:
            return
        ctx = self._ctx_stack.pop()
        ctx.__exit__(exc_type, exc, tb)

    def sync(self) -> None:
        """Synchronize the managed stream.

        Blocks until every operation previously enqueued on ``self.stream``
        has completed. No-op if CUDA isn't available or no stream is set.
        """
        if self.stream is None:
            return
        self.stream.synchronize()


__all__ = ["SingleStreamAllocator"]
