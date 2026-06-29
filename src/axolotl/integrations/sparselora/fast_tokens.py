"""Contiguous-block fast path for SparseLoRA token split / join.

SparseLoRA's per-step output-token mask (``api._compute_output_token_mask``) is
always a **single contiguous block** ``[left:right)`` shared by every row in the
batch: ``_compute_output_token_mask`` derives one ``left``/``right`` pair and does
``masks[..., left:right] = True``. The vendored ``SparseModule.token_splits`` /
``token_join`` express the split with boolean indexing (``x[~masks]`` /
``x[masks]``) and a ``scatter_add`` rejoin. For a contiguous block those are
mathematically identical to slicing + ``cat``, which avoids:

* materializing the boolean mask tensor every step,
* the boolean *gather* kernels in ``token_splits`` (two per split), and
* the ``scatter`` write in ``token_join``.

This module installs the fast path by rebinding the vendored entry points from
the plugin side (no ``_vendor`` edits). The numeric result is bit-identical to
the boolean path for any contiguous mask; ``tests/.../test_fast_tokens.py``
asserts this against the original implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["MaskContext", "install_fast_tokens"]


@dataclass(frozen=True)
class MaskContext:
    """Boundaries of the dense (output-token) block ``[left:right)``.

    All rows share the same boundaries, so the split is pure slicing. When
    ``right <= left`` there are no dense tokens (every token is sparse), which
    reproduces an all-``False`` boolean mask.
    """

    left: int
    right: int
    total: int

    @property
    def has_dense(self) -> bool:
        return self.right > self.left


def _compute_mask_context(labels: torch.Tensor, input_ids: torch.Tensor) -> MaskContext:
    """Drop-in replacement for ``api._compute_output_token_mask``.

    Returns a :class:`MaskContext` instead of a boolean tensor. Reproduces the
    exact ``left``/``right`` the vendored function used to fill the mask, so the
    split it implies is identical. ``input_ids`` is accepted for signature
    parity but unused (only the shape mattered, and ``labels`` carries it).
    """
    del input_ids
    is_ctx = labels == -100
    left = is_ctx.cumprod(dim=1).sum(dim=1).min().item()
    right_pad = is_ctx.flip(dims=[1]).cumprod(dim=1).sum(dim=1).min().item()
    total = labels.shape[-1]
    right = total - right_pad if right_pad > 0 else total
    return MaskContext(int(left), int(right), int(total))


def _fast_token_splits(self, x: torch.Tensor, masks: MaskContext):
    """Slice ``x`` into ``(sparse_tokens, dense_tokens)`` via a contiguous block.

    Equivalent to ``(x[~bool_mask], x[bool_mask])`` for the contiguous mask the
    context describes; the sparse half preserves leading-then-trailing order.
    """
    del self
    ctx = masks
    if not ctx.has_dense:
        return x, x[:, :0]
    dense = x[:, ctx.left : ctx.right]
    sparse = torch.cat([x[:, : ctx.left], x[:, ctx.right :]], dim=1)
    return sparse, dense


def _fast_token_join(
    sparse: torch.Tensor, dense: torch.Tensor, masks: MaskContext
) -> torch.Tensor:
    """Reassemble original token order with a single 3-way ``cat``."""
    ctx = masks
    if not ctx.has_dense:
        return sparse
    return torch.cat([sparse[:, : ctx.left], dense, sparse[:, ctx.left :]], dim=1)


_INSTALLED = False


def install_fast_tokens() -> None:
    """Rebind the vendored token split/join + mask builder to the fast path.

    Idempotent and behaviour-preserving (bit-identical). Call **before** the
    ``torch.compiler.disable`` boundaries are applied so the disable wraps the
    fast ``_compute_mask_context`` (it carries the per-step ``.item()`` syncs).
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from ._vendor.sparselora import api
    from ._vendor.sparselora.modules.base import SparseModule

    api._compute_output_token_mask = _compute_mask_context
    SparseModule.token_splits = _fast_token_splits  # type: ignore[method-assign]
    SparseModule.token_join = staticmethod(_fast_token_join)  # type: ignore[assignment]
    _INSTALLED = True
