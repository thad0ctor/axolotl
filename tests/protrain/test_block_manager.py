"""Tests for the ProTrain block manager (M3).

Covers:

- ``assign_modes`` layout invariants (counts, swap-early placement,
  validation, monotonic CKPT count across a sweep).
- ``wrap_block`` dispatch semantics (NONE identity, CKPT forward/backward
  equivalence, SWAP env-gating).
- ``discover_blocks`` on a fresh-init GPT-2.
- A skeleton end-to-end memory sweep, skipped pending M5 integration.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402  (import after pytest.importorskip)

from axolotl.integrations.protrain.block import (  # noqa: E402
    BlockMode,
    assign_modes,
    discover_blocks,
    unwrap_block,
    wrap_block,
)
from axolotl.integrations.protrain.block.checkpoint import CheckpointedBlock  # noqa: E402
from axolotl.integrations.protrain.block.swap import SwappedBlock  # noqa: E402


# ---------------------------------------------------------------------------
# assign_modes
# ---------------------------------------------------------------------------


def test_assign_modes_basic() -> None:
    """N_block=12, n_swap=0, n_checkpoint=4 → 4 evenly-spaced CKPT.

    With stride = 12 // 4 = 3 and no swap band, CKPT should land at
    block indices 0, 3, 6, 9 and every other block be NONE.
    """
    N_block = 12
    modes = assign_modes(n_swap=0, n_checkpoint=4, N_block=N_block)

    expected_ckpt = {0, 3, 6, 9}
    actual_ckpt = {i for i, m in modes.items() if m is BlockMode.CKPT}
    actual_swap = {i for i, m in modes.items() if m is BlockMode.SWAP}
    actual_none = {i for i, m in modes.items() if m is BlockMode.NONE}

    assert actual_ckpt == expected_ckpt
    assert actual_swap == set()
    assert actual_none == set(range(N_block)) - expected_ckpt
    assert len(modes) == N_block


def test_assign_modes_swap_early() -> None:
    """N_block=10, n_swap=2, n_checkpoint=3 → blocks 0,1 are SWAP.

    SWAP positions must be exactly [0, 1] (swap-early rule). CKPT count
    must be exactly 3 and CKPT must not overlap SWAP. The three CKPT
    slots come from the [2, 10) tail with stride 8//3 = 2, so land at
    {2, 4, 6}.
    """
    N_block = 10
    modes = assign_modes(n_swap=2, n_checkpoint=3, N_block=N_block)

    swap_positions = sorted(i for i, m in modes.items() if m is BlockMode.SWAP)
    ckpt_positions = sorted(i for i, m in modes.items() if m is BlockMode.CKPT)

    assert swap_positions == [0, 1]
    assert len(ckpt_positions) == 3
    # No overlap with swap band.
    assert all(p >= 2 for p in ckpt_positions)
    # All ckpt positions within valid range.
    assert all(0 <= p < N_block for p in ckpt_positions)


def test_assign_modes_validation() -> None:
    """n_swap + n_checkpoint > N_block must raise ValueError."""
    with pytest.raises(ValueError):
        assign_modes(n_swap=5, n_checkpoint=6, N_block=10)
    with pytest.raises(ValueError):
        assign_modes(n_swap=-1, n_checkpoint=0, N_block=4)
    with pytest.raises(ValueError):
        assign_modes(n_swap=0, n_checkpoint=-1, N_block=4)


def test_assign_modes_monotonic_ckpt_count() -> None:
    """Sweep n_checkpoint; returned map has exactly n_checkpoint CKPT each time."""
    N_block = 12
    for n_ckpt in (0, 2, N_block):
        modes = assign_modes(n_swap=0, n_checkpoint=n_ckpt, N_block=N_block)
        count = sum(1 for m in modes.values() if m is BlockMode.CKPT)
        assert count == n_ckpt, f"n_ckpt={n_ckpt}: got {count}"
        assert len(modes) == N_block


# ---------------------------------------------------------------------------
# wrap_block dispatch
# ---------------------------------------------------------------------------


def test_wrap_block_none_is_identity() -> None:
    """NONE mode returns the exact same object (no wrapper)."""
    block = nn.Linear(8, 8)
    wrapped = wrap_block(block, BlockMode.NONE)
    assert wrapped is block


def test_wrap_block_ckpt_marks_wrapper() -> None:
    """CKPT mode produces a CheckpointedBlock with the correct marker."""
    block = nn.Linear(8, 8)
    wrapped = wrap_block(block, BlockMode.CKPT)
    assert isinstance(wrapped, CheckpointedBlock)
    assert wrapped._protrain_wrapped_mode is BlockMode.CKPT
    # Idempotent unwrap returns the original.
    assert unwrap_block(wrapped) is block


def test_wrap_block_idempotent_rewrap() -> None:
    """Re-wrapping an already-wrapped block unwraps then re-wraps."""
    block = nn.Linear(8, 8)
    once = wrap_block(block, BlockMode.CKPT)
    twice = wrap_block(once, BlockMode.NONE)
    # Second call with NONE unwraps and returns original.
    assert twice is block


@pytest.mark.gpu
def test_wrap_block_ckpt_roundtrip() -> None:
    """Forward+backward through a CKPT-wrapped Linear matches the unwrapped version."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(0)
    block = nn.Linear(8, 8).to(device)
    ref_block = nn.Linear(8, 8).to(device)
    ref_block.load_state_dict(block.state_dict())

    wrapped = wrap_block(block, BlockMode.CKPT)

    x_a = torch.randn(4, 8, device=device, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)

    out_wrapped = wrapped(x_a)
    out_ref = ref_block(x_b)

    assert torch.allclose(out_wrapped, out_ref, atol=1e-6)

    out_wrapped.sum().backward()
    out_ref.sum().backward()

    # Input grads match.
    assert torch.allclose(x_a.grad, x_b.grad, atol=1e-6)  # type: ignore[arg-type]
    # Parameter grads match — same underlying Linear weights.
    assert torch.allclose(
        unwrap_block(wrapped).weight.grad,  # type: ignore[union-attr]
        ref_block.weight.grad,  # type: ignore[arg-type]
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# SWAP env-gating
# ---------------------------------------------------------------------------


def test_swap_without_flag_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without PROTRAIN_ENABLE_SWAP, constructing SwappedBlock must raise."""
    monkeypatch.delenv("PROTRAIN_ENABLE_SWAP", raising=False)
    with pytest.raises(RuntimeError, match="PROTRAIN_ENABLE_SWAP"):
        SwappedBlock(nn.Linear(8, 8))


def test_swap_with_flag_constructs(monkeypatch: pytest.MonkeyPatch) -> None:
    """With PROTRAIN_ENABLE_SWAP=1, SwappedBlock must construct cleanly.

    We do NOT exercise forward here — that is integration work gated by
    M4's scheduler.
    """
    monkeypatch.setenv("PROTRAIN_ENABLE_SWAP", "1")
    wrapped = SwappedBlock(nn.Linear(8, 8))
    assert wrapped._protrain_wrapped_mode is BlockMode.SWAP


# ---------------------------------------------------------------------------
# discover_blocks
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_discover_blocks_gpt2() -> None:
    """Fresh-init GPT-2 with 3 layers; ``discover_blocks`` returns len==3."""
    transformers = pytest.importorskip("transformers")

    cfg = transformers.GPT2Config(n_layer=3)
    # Fresh init, no weight download — from_config, not from_pretrained.
    model = transformers.GPT2LMHeadModel(cfg)

    blocks = discover_blocks(model)
    assert len(blocks) == 3


# ---------------------------------------------------------------------------
# Full-sweep skeleton
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skip(
    reason=(
        "requires M2 chunk manager for end-to-end memory sweep; runs after M5 "
        "integration"
    )
)
def test_monotonic_memory_reduction_sweep() -> None:
    """Peak GPU memory should decrease monotonically as n_checkpoint grows.

    Intent: construct a small transformer, iterate n_checkpoint in
    [0, 1, ..., N_block], and measure peak CUDA memory after a single
    forward+backward. Higher n_checkpoint must never increase peak.
    This verifies that the block manager wiring actually recovers
    memory in backward.

    Blocked on M2's ChunkManager for realistic param-side memory
    accounting and M5 plugin wiring for the integration harness.
    """
    raise NotImplementedError
