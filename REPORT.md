# SparseLoRA — Phi3 fused-projection support

Adds faithful SparseLoRA support for Phi3-style **fused projections**
(`qkv_proj`, `gate_up_proj`) on top of the existing model-agnostic
separate-projection wiring, without touching the `feat/sparselora-plugin`
branch. Branch: `feat/sparselora-arch-ext`.

## Verdict

**Phi3 works end-to-end with SparseLoRA, faithfully.** Dense-apply (sparsity 0)
is logit-exact vs the unmodified model for both GQA and MHA Phi3, sparse
forward+backward at 0.5 produces finite loss and non-zero LoRA gradients, and a
full `axolotl train` run (calibrate → apply → train) completes with a stable
finite loss. No approximation was shipped beyond the ones the existing
separate-projection path already makes (see *Caveats*).

## Design

Phi3 fuses q/k/v into one `qkv_proj` (out = `[q | k | v]`) and gate/up into one
`gate_up_proj` (out = `[gate | up]`). The predictors, the SVD factor machinery,
and the vendored `SparseLinear` all operate per logical projection. The key idea
is to **slice the fused weight into its logical sub-blocks** so everything
downstream is reused unchanged, and to **project the fused layer in one call with
a combined index** at run time.

### Phi3-specific sparse modules (`arch_wiring.py`)

- `SparseFusedQKVAttention(SparseAttention)` — overrides only the QKV-production
  step (new overridable `_qkv` hook on `SparseAttention`). The predictor selects
  `q_i / k_i / v_i`; the module builds the combined index
  `[q_i | q_size + k_i | q_size + kv_size + v_i]`, calls `qkv_proj(x, combined)`
  once, and splits the result back into Q/K/V at the (config-derived) sub-block
  sizes. RoPE (incl. partial-rotary), the attention interface (incl. GQA via
  `num_key_value_groups`, sliding-window, softcap), and `o_proj` are inherited
  from `SparseAttention` unchanged. `sliding_window` is read off `config` (Phi3
  carries it there, not on the module).
- `SparseFusedGateUpMLP(SparseSwiGLUMLP)` — overrides `_block`. For kept channel
  `i`, gate is row `i` and up is row `intermediate + i` of `gate_up_proj`, so the
  combined index is `[i | intermediate + i]`; the fused output is split into equal
  halves (width-derived, so it is correct whether the bare `gate_up_proj` returns
  the selected channels in mode `"out"` or a padded full width), gated with the
  model's own activation, then `down_proj` gathers column `i`.

Both reuse the existing predictors and SVD factors verbatim — `factors.py` slices
the fused weight into `gate_proj`/`up_proj` and `q_proj`/`k_proj`/`v_proj`
sub-blocks and writes the **same factor keys** the separate path uses, so
`_build_mlp_predictor` / `_build_attn_predictor` (extracted shared cores) load
them with no special-casing. GQA is handled by the predictor exactly as before
(k/v sub-blocks come out narrower than q, triggering the GQA predictor).

### Detection & registration

`is_fused_qkv_attention` / `is_fused_gate_up_mlp` (arch_wiring) and
`is_fused_attn_module` / `is_fused_mlp_module` (factors) recognize the Phi3
layout. `register_arch_wiring` maps `Phi3Attention → SparseFusedQKVAttention` and
`Phi3MLP → SparseFusedGateUpMLP`; the separate-projection path is untouched and
checked first.

### LoRA contract (`plugin._validate`)

- The attention LoRA target for Phi3 is `qkv_proj` (+ `o_proj`); `_validate`
  accepts `lora_target_modules: [qkv_proj, o_proj]`. `qkv_proj` lives inside the
  discovered `self_attn` target, so the `orphan_lora` check passes.
- `gate_up_proj` was added to the MLP-LoRA rejection set, so a LoRA adapter on
  the fused MLP (or `down_proj`) is still rejected — the recipe stays
  attention-only.
- The `predictor_rank` "projection too small" check now uses
  `output_sparse_weights`, which slices the fused weight, so the check sees the
  real per-sub-block dims.

### Calibration (`calibration.py`)

`discover_target_modules` and the per-module recon-error dispatch recognize fused
modules; the FFN/attention recon errors read weights through
`mlp_projection_weights` / `attn_projection_weights` (fused-aware), so the
sensitivity sweep works on Phi3.

### torch.compile boundaries (`plugin._apply_compile_boundaries`)

The two fused classes' forwards are added to the `torch.compiler.disable` set
(the fused MLP overrides `_block` under the already-disabled SwiGLU forward; the
fused attention's data-dependent combined-index slicing runs under the disabled
`SparseAttention` forward).

### One documented vendored edit

`get_sparsity_mode` (vendored `modules/registry.py`) keys the sparsity direction
off the projection's leaf name; `qkv_proj`/`gate_up_proj` were absent and would
default to `None` (dense), silently disabling sparsity on the fused base layer.
They were added to `_OUT_PROJS` (output-sparse, exactly like the separate
projections they concatenate). This is the only vendored change, documented in
`_vendor/PROVENANCE.md`; all the actual fused logic is un-vendored in
`arch_wiring.py`.

## Evidence

- **Dense-apply logit-exact** (GPU idx 5), Phi3 GQA *and* MHA: max logit diff
  `8.94e-08` vs the unmodified model — the fused qkv/gate_up slicing is bit-exact.
  Mirrors the Gemma2/StableLM logit-exact tests.
- **Sparse 0.5 fwd+bwd** (GPU idx 5), GQA & MHA: finite loss, non-zero LoRA grad
  norm, reconstruction within tolerance of the dense model.
- **End-to-end `axolotl train`** on a tiny real-arch Phi3 (random weights, Qwen3
  tokenizer) with `lora_target_modules: [qkv_proj, o_proj]`,
  `sample_packing: false`, `lora_qkv_kernel: false`:
  - `method: proxy` → calibrate → apply → 8 steps, finite stable loss, completes.
  - `method: none` forcing 0.5 on all 4 fused modules (both attn + both MLP in
    both layers): "applying sparsity to 4/4 modules", SparseLoRA enabled at step
    1, 8 steps of finite stable loss, completes — proves the fused **sparse**
    path runs in real training.
- **Tests**: full suite green — `88 passed` on GPU idx 5, `73 passed, 15 skipped`
  on CPU. New Phi3 tests: detection+registration, fused factor keys/shapes +
  validation (GQA & MHA), MLP-LoRA rejection, dense-apply logit-exact (GQA & MHA,
  GPU), sparse fwd+bwd (GQA & MHA, GPU).
- **Lint**: `pre-commit run` (pinned ruff/ruff-format/mypy/bandit) passes on all
  changed non-`_vendor` files; `_vendor/` stays lint-excluded.

## Caveats (pre-existing, not new)

- For GQA, `o_proj` is fed the value-channel indices `v_i` (which index the
  narrower kv space) against its q-space input — the exact behavior of the
  existing separate-projection GQA path, inherited unchanged. At sparsity 0 it is
  exact (full gather); at sparsity > 0 it is the same upstream contextual-sparsity
  approximation, not a new one introduced for Phi3.
- Fused attention assumes `qkv_proj` is the (LoRA-wrapped) attention adapter — the
  SparseLoRA recipe and `_validate` already require attention LoRA, and the
  vendored attention predictor already requires a `base_layer`. The fused MLP
  needs no LoRA (and rejects it), matching the recipe.

## Files changed

- `arch_wiring.py` — fused detection, `SparseFusedQKVAttention`,
  `SparseFusedGateUpMLP`, shared predictor cores, `_qkv` hook, `gate_activation`
  (Phi3 `activation_fn`), registration, `unsupported_reason`.
- `factors.py` — fused detection, `fused_qkv_sizes`/`fused_intermediate_size`,
  fused-aware weight extraction (`mlp_projection_weights`,
  `attn_projection_weights`, `output_sparse_weights`), sliced factor computation.
- `calibration.py` — fused detection in discovery + recon-error dispatch.
- `plugin.py` — accept `qkv_proj`/reject `gate_up_proj` LoRA, fused-aware rank
  check, fused compile boundaries.
- `_vendor/sparselora/modules/registry.py` (+ `PROVENANCE.md`) — `qkv_proj`,
  `gate_up_proj` added to `_OUT_PROJS`.
- tests `conftest.py` (tiny Phi3 builder + per-arch LoRA targets) and
  `test_sparselora_arch.py` (Phi3 test suite; the obsolete "not registered" test
  replaced with "detected and registered").
