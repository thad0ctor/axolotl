"""Placement rules for the interleaved block manager (§3.1.2).

Given ``n_swap``, ``n_checkpoint``, and ``N_block``, decide which block
index gets which ``BlockMode`` under ProTrain's three placement rules:

1. **Swap-early** — the first ``n_swap`` blocks get SWAP. Earlier blocks
   have more forward compute after them to hide the CPU->GPU prefetch.
2. **Interleave CKPT among the remaining blocks** — flattens peak memory
   by preventing activation accumulation in a contiguous run.
3. **Unopt-late** — blocks with NONE sit in the late tail so their
   activations are consumed first in backward, freeing PCIe bandwidth
   for the earlier swap-block prefetches.

Also ships ``discover_blocks`` — the heuristic that finds the
transformer-block ``nn.ModuleList`` inside a user model without needing
a central registry.
"""

from __future__ import annotations

from typing import Iterable

from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode, BlockStrategyMap
from axolotl.integrations.protrain.types import BlockId
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# assign_modes
# ---------------------------------------------------------------------------


def assign_modes(n_swap: int, n_checkpoint: int, N_block: int) -> BlockStrategyMap:
    """Return the per-block mode map under the three placement rules.

    Parameters
    ----------
    n_swap:
        Number of blocks that should use ``BlockMode.SWAP``. Must be
        non-negative and ``n_swap + n_checkpoint <= N_block``.
    n_checkpoint:
        Number of blocks that should use ``BlockMode.CKPT``.
    N_block:
        Total number of transformer blocks in the model.

    Returns
    -------
    BlockStrategyMap
        ``dict`` keyed ``0 .. N_block-1`` mapping to exactly
        ``n_swap`` SWAP entries, ``n_checkpoint`` CKPT entries, and
        ``N_block - n_swap - n_checkpoint`` NONE entries.

    Raises
    ------
    ValueError
        If any input is negative or ``n_swap + n_checkpoint > N_block``.
    """
    if N_block < 0:
        raise ValueError(f"N_block must be non-negative, got {N_block}")
    if n_swap < 0 or n_checkpoint < 0:
        raise ValueError(
            f"n_swap and n_checkpoint must be non-negative, got "
            f"n_swap={n_swap}, n_checkpoint={n_checkpoint}"
        )
    if n_swap + n_checkpoint > N_block:
        raise ValueError(
            f"n_swap + n_checkpoint ({n_swap} + {n_checkpoint} = "
            f"{n_swap + n_checkpoint}) exceeds N_block ({N_block})"
        )

    # Initialise everything to NONE (unopt-late default — positions that
    # do not receive SWAP/CKPT just stay NONE, and by construction those
    # positions land in the tail).
    modes: BlockStrategyMap = {BlockId(i): BlockMode.NONE for i in range(N_block)}

    # Rule 1: swap-early. First n_swap block ids are SWAP.
    for i in range(n_swap):
        modes[BlockId(i)] = BlockMode.SWAP

    # Rule 2: interleave CKPT evenly among the remaining (N_block - n_swap)
    # positions so checkpoint and non-checkpoint blocks alternate, flattening
    # peak memory. Strategy: pick n_checkpoint positions from [n_swap, N_block)
    # at an even stride.
    remaining = N_block - n_swap
    if n_checkpoint > 0 and remaining > 0:
        # Floor stride; n_checkpoint <= remaining guaranteed by validation.
        # Using stride = remaining // n_checkpoint puts a CKPT block at
        # position n_swap + k * stride for k in 0..n_checkpoint-1, which
        # distributes CKPT blocks evenly and leaves the last tail slots NONE
        # (satisfying rule 3: unopt-late).
        stride = remaining // n_checkpoint
        # Guard against stride==0 when remaining == n_checkpoint: every
        # remaining slot becomes CKPT, which is the correct behaviour.
        if stride == 0:
            stride = 1
        placed = 0
        k = 0
        while placed < n_checkpoint:
            idx = n_swap + k * stride
            if idx >= N_block:
                # Past the end — fill from the first available NONE slot
                # onward. This branch is only hit at the degenerate
                # boundary where stride * n_checkpoint overshoots.
                break
            if modes[BlockId(idx)] is BlockMode.NONE:
                modes[BlockId(idx)] = BlockMode.CKPT
                placed += 1
            k += 1
            # Safety: if k runs away, walk remaining NONE positions.
            if k > N_block:
                break
        # If we still haven't placed all CKPT blocks (only possible at the
        # ragged boundary), fill from the first available NONE position
        # after the swap band.
        if placed < n_checkpoint:
            for i in range(n_swap, N_block):
                if placed >= n_checkpoint:
                    break
                if modes[BlockId(i)] is BlockMode.NONE:
                    modes[BlockId(i)] = BlockMode.CKPT
                    placed += 1

    # Post-condition: counts match the request.
    _assert_counts(modes, n_swap=n_swap, n_checkpoint=n_checkpoint, N_block=N_block)
    return modes


def _assert_counts(
    modes: BlockStrategyMap, *, n_swap: int, n_checkpoint: int, N_block: int
) -> None:
    """Invariant check. Raises ``ValueError`` if counts diverge."""
    counts = {BlockMode.NONE: 0, BlockMode.CKPT: 0, BlockMode.SWAP: 0}
    for m in modes.values():
        counts[m] = counts[m] + 1
    expected_none = N_block - n_swap - n_checkpoint
    if (
        counts[BlockMode.SWAP] != n_swap
        or counts[BlockMode.CKPT] != n_checkpoint
        or counts[BlockMode.NONE] != expected_none
    ):
        raise ValueError(
            f"assign_modes invariant violation: got counts={counts}, "
            f"expected SWAP={n_swap}, CKPT={n_checkpoint}, NONE={expected_none}"
        )


# ---------------------------------------------------------------------------
# discover_blocks
# ---------------------------------------------------------------------------


# Dotted paths checked in order. Order rationale: GPT-2 style first (the
# project's canonical test target), then Llama/Mistral style (most common
# HF LLM layout), then less-common transformer variants, then the base_model
# layout used by PEFT-wrapped models.
_KNOWN_BLOCK_PATHS: tuple[str, ...] = (
    "transformer.h",                   # GPT-2, GPT-Neo, GPT-J (some), Falcon (some)
    "model.layers",                    # Llama, Mistral, Qwen, most modern HF LLMs
    "transformer.layers",              # MPT, some GPT-NeoX variants
    "base_model.layers",               # PEFT / LoRA-wrapped models (short form)
    "base_model.model.model.layers",   # PEFT + LlamaForCausalLM (LoraModel wraps CausalLM)
    "base_model.model.transformer.h",  # PEFT + GPT-2
)


def _resolve(root: nn.Module, dotted: str) -> nn.Module | None:
    obj: object = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    if isinstance(obj, nn.Module):
        return obj
    return None


def _looks_like_block(m: nn.Module) -> bool:
    """Heuristic: transformer blocks expose an ``attention`` or ``self_attn``
    attribute. Blocks wrapped by ProTrain's dispatcher expose
    ``_protrain_wrapped_mode``. Fall-back path when no known dotted path
    matches."""
    if hasattr(m, "attention") or hasattr(m, "self_attn"):
        return True
    if hasattr(m, "_protrain_wrapped_mode"):
        return True
    # CheckpointedBlock stores the original in ``.block``; check one level in.
    inner = getattr(m, "block", None)
    if inner is not None and (hasattr(inner, "attention") or hasattr(inner, "self_attn")):
        return True
    return False


def _iter_module_lists(root: nn.Module) -> Iterable[nn.ModuleList]:
    for m in root.modules():
        if isinstance(m, nn.ModuleList):
            yield m


def discover_blocks(model: nn.Module) -> list[nn.Module]:
    """Return the transformer-block ``ModuleList`` as a plain ``list``.

    Resolution order:

    1. Try each known dotted path (``transformer.h``, ``model.layers``,
       ``transformer.layers``, ``base_model.layers``). Return the first
       one that resolves to a ``nn.ModuleList``.
    2. Otherwise scan every ``nn.ModuleList`` under ``model`` and return
       the first whose children all look like transformer blocks
       (attribute ``attention`` or ``self_attn`` present). This catches
       custom models that do not match any known dotted path.

    Raises
    ------
    RuntimeError
        If no match is found. The error message names the paths tried.
    """
    for dotted in _KNOWN_BLOCK_PATHS:
        candidate = _resolve(model, dotted)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            LOG.debug("discover_blocks: matched %s (n=%d)", dotted, len(candidate))
            return list(candidate)

    # Fallback: scan for a ModuleList of block-shaped children.
    for mlist in _iter_module_lists(model):
        if len(mlist) == 0:
            continue
        if all(_looks_like_block(child) for child in mlist):
            LOG.debug(
                "discover_blocks: matched ModuleList via attention heuristic (n=%d)",
                len(mlist),
            )
            return list(mlist)

    raise RuntimeError(
        "discover_blocks: no transformer-block ModuleList found on model. "
        f"Tried dotted paths {_KNOWN_BLOCK_PATHS} and the "
        "attention/self_attn attribute heuristic."
    )


__all__ = ["assign_modes", "discover_blocks"]
