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


@pytest.fixture
def restore_token_hooks():
    """Snapshot and restore every global ``install_fast_tokens`` patches.

    ``install_fast_tokens`` rewires three things — ``api._compute_output_token_mask``,
    ``SparseModule.token_splits`` and ``SparseModule.token_join`` — plus the
    module's installed-mode flag. Restoring only ``token_splits`` (as an earlier
    teardown did) leaks the mask builder and ``token_join`` into later tests.
    """
    from axolotl.integrations.sparselora._vendor.sparselora import api

    orig_mask = api._compute_output_token_mask
    orig_split = SparseModule.token_splits
    orig_join = SparseModule.__dict__["token_join"]  # keep the staticmethod descriptor
    orig_installed = fast_tokens._INSTALLED_PACKING
    fast_tokens._INSTALLED_PACKING = None
    try:
        yield
    finally:
        api._compute_output_token_mask = orig_mask
        SparseModule.token_splits = orig_split
        SparseModule.token_join = orig_join
        fast_tokens._INSTALLED_PACKING = orig_installed


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


def test_output_token_mask_needs_only_labels():
    """The mask builds from labels alone (input_ids may be positional/absent)."""
    from axolotl.integrations.sparselora._vendor.sparselora import api

    labels = torch.full((2, 16), -100)
    labels[:, 5:11] = 1
    mask = api._compute_output_token_mask(labels)  # no input_ids
    assert mask.dtype == torch.bool
    assert mask.shape == labels.shape
    expected = torch.zeros(2, 16, dtype=torch.bool)
    expected[:, 5:11] = True
    assert torch.equal(mask, expected)

    # The fast-path replacement accepts the same single-arg call.
    ctx = fast_tokens._compute_mask_context(labels)
    assert (ctx.left, ctx.right, ctx.total) == (5, 11, 16)


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
def test_applied_model_logits_bit_identical(tiny_lora_model, restore_token_hooks):
    """A real sparse-applied model gives identical logits under fast vs boolean."""
    import tempfile

    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
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

    with torch.no_grad():
        ref = model(input_ids=ids, labels=labels).logits.clone()
    fast_tokens.install_fast_tokens()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels).logits
    assert torch.equal(out, ref)


def test_packed_mask_keeps_outputs_dense_and_counts_uniform():
    """A packed row holds several output segments; the packing mask keeps every
    one dense and tops each row up to a uniform dense-token count."""
    labels = torch.full((2, 16), -100)
    # row 0: outputs at [3,5) and [9,12) -> 5 output tokens
    labels[0, 3:5] = 1
    labels[0, 9:12] = 1
    # row 1: outputs at [2,3) and [13,16) -> 4 output tokens
    labels[1, 2:3] = 1
    labels[1, 13:16] = 1

    mask = fast_tokens._compute_packed_output_token_mask(labels)
    assert mask.dtype == torch.bool
    assert mask.shape == labels.shape

    # Every output token is dense (never sparsified) -- the correctness invariant.
    assert torch.all(mask[labels != -100])
    # Multi-segment: row 0 must have >1 disjoint True run (not one contiguous block).
    row0 = mask[0].int()
    transitions = (row0[1:] != row0[:-1]).sum().item()
    assert transitions >= 3  # at least two separate True runs
    # Uniform dense-token count across rows (required by the boolean view reshape).
    counts = mask.sum(dim=1)
    assert counts[0] == counts[1] == 5  # max output count, both rows topped up


def test_packed_split_join_bit_identical_and_preserves_outputs():
    """The vendored boolean split/join used under packing round-trips the packed
    multi-segment mask exactly, and a sparse transform never touches outputs."""
    torch.manual_seed(0)
    dummy = SparseModule.__new__(SparseModule)

    labels = torch.full((3, 20), -100)
    labels[0, 2:4] = 1
    labels[0, 10:13] = 1
    labels[1, 5:6] = 1
    labels[1, 15:19] = 1
    labels[2, 1:2] = 1  # short row -> gets padded up with leading context
    mask = fast_tokens._compute_packed_output_token_mask(labels)

    x = torch.randn(3, 20, 8)
    # Vendored boolean path (what packing installs).
    sparse, dense = SparseModule.token_splits(dummy, x, mask)
    assert torch.equal(_orig_split(x, mask)[0], sparse)
    assert torch.equal(_orig_split(x, mask)[1], dense)

    # Round-trip identity.
    rejoined = SparseModule.token_join(sparse, dense, mask)
    assert torch.equal(rejoined, x)

    # Sparsifying context must leave every output token bit-identical.
    sparse_t = sparse + 123.0  # "approximate" the context tokens
    joined = SparseModule.token_join(sparse_t, dense, mask)
    assert torch.equal(joined[labels != -100], x[labels != -100])


def test_packing_install_keeps_boolean_split_join(restore_token_hooks):
    """With packing, install must set the multi-segment mask builder but leave the
    boolean (not contiguous-slice) token split/join in place."""
    from axolotl.integrations.sparselora._vendor.sparselora import api

    fast_tokens.install_fast_tokens(packing=True)
    assert api._compute_output_token_mask is (
        fast_tokens._compute_packed_output_token_mask
    )
    # Boolean split/join must NOT be replaced by the contiguous fast path.
    assert SparseModule.token_splits is not fast_tokens._fast_token_splits


def test_install_is_idempotent_and_rebinds(restore_token_hooks):
    fast_tokens.install_fast_tokens()
    fast_tokens.install_fast_tokens()  # same mode -> idempotent
    assert SparseModule.token_splits is fast_tokens._fast_token_splits


def test_install_reinstalls_on_packing_mode_flip(restore_token_hooks):
    """A packing-mode flip in the same interpreter must reinstall the correct
    hooks: the guard is keyed by ``packing``, not a bare "installed" flag, so a
    packed run never silently keeps the unpacked contiguous single-block path."""
    from axolotl.integrations.sparselora._vendor.sparselora import api

    # Unpacked first: contiguous fast split/join + contiguous mask builder.
    fast_tokens.install_fast_tokens(packing=False)
    assert SparseModule.token_splits is fast_tokens._fast_token_splits
    assert api._compute_output_token_mask is fast_tokens._compute_mask_context

    # Flip to packed: must swap back to the vendored boolean split/join and the
    # multi-segment mask builder, not reuse the unpacked path.
    fast_tokens.install_fast_tokens(packing=True)
    assert SparseModule.token_splits is not fast_tokens._fast_token_splits
    assert api._compute_output_token_mask is (
        fast_tokens._compute_packed_output_token_mask
    )

    # Flip back to unpacked: contiguous fast path restored.
    fast_tokens.install_fast_tokens(packing=False)
    assert SparseModule.token_splits is fast_tokens._fast_token_splits
    assert api._compute_output_token_mask is fast_tokens._compute_mask_context
