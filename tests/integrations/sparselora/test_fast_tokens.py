"""Bit-identity of the contiguous-block fast token split/join vs the boolean path.

The fast path (``fast_tokens``) replaces SparseLoRA's boolean ``x[~mask]`` /
``x[mask]`` split and ``scatter`` rejoin with slicing + ``cat``. These tests pin
that the two are numerically identical for every contiguous mask the vendored
``_compute_output_token_mask`` can produce, including the degenerate all-context
and all-output cases.
"""

import pytest
import torch

from axolotl.integrations.sparselora import fast_tokens
from axolotl.integrations.sparselora._vendor.sparselora.modules.base import SparseModule


def _bool_mask(left, right, total):
    m = torch.zeros(1, total, dtype=torch.bool)
    if right > left:
        m[..., left:right] = True
    return m


def _orig_split(x, mask):
    return (
        x[~mask].view(x.shape[0], -1, x.shape[-1]).contiguous(),
        x[mask].view(x.shape[0], -1, x.shape[-1]).contiguous(),
    )


def _orig_join(sparse, dense, mask):
    out = torch.empty(
        sparse.shape[0],
        sparse.shape[1] + dense.shape[1],
        sparse.shape[-1],
        device=sparse.device,
        dtype=sparse.dtype,
    )
    out[mask] = dense.reshape(-1, dense.shape[-1])
    out[~mask] = sparse.reshape(-1, sparse.shape[-1])
    return out


_CASES = [
    (4, 12, 16),  # interior block
    (0, 12, 16),  # no leading context
    (4, 16, 16),  # no trailing context
    (0, 16, 16),  # all output (dense)
    (16, 0, 16),  # all context (degenerate -> all sparse)
    (8, 8, 16),  # empty dense block
    (1, 2, 16),  # single dense token
]


def test_compute_mask_context_matches_boundaries():
    labels = torch.full((1, 16), -100)
    labels[:, 5:11] = 1  # output tokens in [5, 11)
    ctx = fast_tokens._compute_mask_context(labels, labels)
    assert (ctx.left, ctx.right, ctx.total) == (5, 11, 16)
    assert ctx.has_dense


def test_split_join_bit_identical_to_boolean():
    torch.manual_seed(0)
    dummy = SparseModule.__new__(SparseModule)  # token_splits needs only self
    for left, right, total in _CASES:
        x = torch.randn(1, total, 8)
        mask = _bool_mask(left, right, total)
        ctx = fast_tokens.MaskContext(left, right, total)

        o_sparse, o_dense = _orig_split(x, mask)
        f_sparse, f_dense = fast_tokens._fast_token_splits(dummy, x, ctx)
        assert f_sparse.shape == o_sparse.shape
        assert f_dense.shape == o_dense.shape
        assert torch.equal(f_sparse, o_sparse)
        assert torch.equal(f_dense, o_dense)

        # Rejoin a *transformed* split (distinct values) to catch ordering bugs.
        sp, dn = o_sparse + 100.0, o_dense - 100.0
        o_join = _orig_join(sp, dn, mask)
        f_join = fast_tokens._fast_token_join(sp, dn, ctx)
        assert torch.equal(f_join, o_join)
        # Round-trip identity: split then join returns the original.
        assert torch.equal(fast_tokens._fast_token_join(f_sparse, f_dense, ctx), x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="predictors require CUDA")
def test_applied_model_logits_bit_identical(tiny_lora_model):
    """A real sparse-applied model gives identical logits under fast vs boolean."""
    import tempfile

    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        api,
        apply_sparselora,
    )
    from axolotl.integrations.sparselora.calibration import discover_target_modules
    from axolotl.integrations.sparselora.factors import (
        compute_factor_tensors,
        save_factors,
    )

    model = tiny_lora_model.to("cuda").to(torch.bfloat16).train()
    targets = discover_target_modules(model)
    factors = compute_factor_tensors(model, targets, rank=8)
    d = tempfile.mkdtemp()
    save_factors(factors, d)
    apply_sparselora(
        model,
        SparseLoRAConfig(
            layer_sparsity={t: 0.5 for t in targets}, predictor_rank=8, path=d
        ),
    )

    ids = torch.randint(0, 128, (2, 24), device="cuda")
    labels = ids.clone()
    labels[:, :10] = -100

    orig_mask = api._compute_output_token_mask
    orig_split = SparseModule.token_splits
    orig_join = SparseModule.token_join
    try:
        with torch.no_grad():
            ref = model(input_ids=ids, labels=labels).logits.clone()
        fast_tokens._INSTALLED = False
        fast_tokens.install_fast_tokens()
        with torch.no_grad():
            out = model(input_ids=ids, labels=labels).logits
        assert torch.equal(out, ref)
    finally:
        api._compute_output_token_mask = orig_mask
        SparseModule.token_splits = orig_split
        SparseModule.token_join = orig_join
        fast_tokens._INSTALLED = False


def test_install_is_idempotent_and_rebinds():
    orig_split = SparseModule.token_splits
    try:
        fast_tokens._INSTALLED = False
        fast_tokens.install_fast_tokens()
        fast_tokens.install_fast_tokens()  # idempotent
        assert SparseModule.token_splits is fast_tokens._fast_token_splits
    finally:
        SparseModule.token_splits = orig_split
        fast_tokens._INSTALLED = False
