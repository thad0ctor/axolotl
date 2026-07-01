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
from typing import Any

import torch

__all__ = ["MaskContext", "install_fast_tokens"]


def _compute_packed_output_token_mask(
    labels: torch.Tensor, input_ids: torch.Tensor | None = None
) -> torch.Tensor:
    """Packing-aware output-token mask: a boolean ``[batch, seq]`` tensor.

    Sample packing concatenates several sub-sequences into one row, so output
    tokens form *multiple* disjoint segments per row rather than one contiguous
    ``[left:right)`` block. The contiguous builder
    (:func:`_compute_mask_context`) would have to grow its single block to the
    whole envelope spanning all segments, sparsifying almost nothing. This keeps
    every output token (``labels != -100``) dense and leaves the inter-segment
    context tokens sparse.

    The boolean ``SparseModule.token_splits`` reshapes the gathered tokens to
    ``[batch, -1, hidden]``, which needs the same dense-token count in every
    row. Output-token counts differ per packed row, so each short row is topped
    up to the batch-wide maximum by marking its *leading* context tokens dense
    (always safe — extra dense tokens only cost a little speed, never
    correctness). The result is a uniform-count multi-segment mask the vendored
    boolean split/join handle exactly.
    """
    del input_ids
    is_out = labels != -100
    dense_count = is_out.sum(dim=1)
    max_dense = int(dense_count.max().item())
    is_ctx = ~is_out
    # cumsum over context positions is 1-indexed among them; keep the first
    # ``need`` per row so every row reaches ``max_dense`` dense tokens.
    need = (max_dense - dense_count).unsqueeze(1)
    pad = is_ctx & (is_ctx.cumsum(dim=1) <= need)
    return is_out | pad


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


def _compute_mask_context(
    labels: torch.Tensor, input_ids: torch.Tensor | None = None
) -> MaskContext:
    """Drop-in replacement for ``api._compute_output_token_mask``.

    Returns a :class:`MaskContext` instead of a boolean tensor. Reproduces the
    exact ``left``/``right`` the vendored function used to fill the mask, so the
    split it implies is identical. ``input_ids`` is accepted (optional) for
    backward signature parity but unused — the mask derives from ``labels``.
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


# The packing mode the hooks are currently installed for (``None`` = not yet
# installed). Keyed by ``packing`` so a mode change reinstalls: the two modes
# swap in mutually-incompatible token split/join semantics, so remembering only
# *that* something was installed would leave the wrong path active after a flip.
_INSTALLED_PACKING: bool | None = None
# Vendored split/join captured on first install so a flip back to packing can
# restore the boolean path the unpacked install replaces.
_ORIG_TOKEN_SPLITS: Any = None
_ORIG_TOKEN_JOIN: Any = None


def install_fast_tokens(packing: bool = False) -> None:
    """Rebind the vendored mask builder (and, unpacked, the token split/join).

    Idempotent **per mode**: a repeat call with the same ``packing`` is a no-op,
    but a call with a different ``packing`` reinstalls the correct hooks (the two
    modes are not interchangeable). Call **before** the ``torch.compiler.disable``
    boundaries are applied so the disable wraps the installed mask builder (it
    carries the per-step ``.item()`` syncs).

    Unpacked (``packing=False``): install the contiguous-block fast path
    (:func:`_compute_mask_context` + slice/cat split/join), which is
    bit-identical to the boolean path for the single ``[left:right)`` block the
    contiguous builder produces.

    Packed (``packing=True``): a packed row holds several output segments, so the
    single-block fast path cannot represent it (its slice/cat assumes one block
    shared by every row, and packed rows have different per-row boundaries).
    Install :func:`_compute_packed_output_token_mask` and keep the vendored
    boolean ``token_splits`` / ``token_join``, which gather/scatter an arbitrary
    multi-segment mask correctly.
    """
    global _INSTALLED_PACKING, _ORIG_TOKEN_SPLITS, _ORIG_TOKEN_JOIN
    if _INSTALLED_PACKING == packing:
        return

    from ._vendor.sparselora import api
    from ._vendor.sparselora.modules.base import SparseModule

    if _ORIG_TOKEN_SPLITS is None:
        _ORIG_TOKEN_SPLITS = SparseModule.token_splits
        _ORIG_TOKEN_JOIN = SparseModule.__dict__["token_join"]

    if packing:
        api._compute_output_token_mask = _compute_packed_output_token_mask
        # Restore the vendored boolean split/join in case an earlier unpacked
        # install replaced them with the contiguous fast path.
        SparseModule.token_splits = _ORIG_TOKEN_SPLITS  # type: ignore[method-assign]
        SparseModule.token_join = _ORIG_TOKEN_JOIN  # type: ignore[assignment]
    else:
        api._compute_output_token_mask = _compute_mask_context
        SparseModule.token_splits = _fast_token_splits  # type: ignore[method-assign]
        SparseModule.token_join = staticmethod(_fast_token_join)  # type: ignore[assignment]
    _INSTALLED_PACKING = packing
