# Vendored: z-lab/sparselora

This directory contains a verbatim copy of the [`sparselora`](https://github.com/z-lab/sparselora) Python package, vendored into Axolotl rather than installed via pip.

- **Upstream:** https://github.com/z-lab/sparselora
- **Commit:** `a2fd69de93b1168080346ec113c99501f0bb58b1`
- **License:** MIT (see `LICENSE`)
- **Paper:** SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity (ICML 2025), arXiv:2506.16500

## What was copied

The `sparselora/` package only (`api.py`, `callback.py`, `config.py`, `__init__.py`, and `modules/`). The upstream `experiments/` directory (training/eval harness and its extra dependencies) was **not** vendored.

## Local edits

Two changes were applied to the copied sources:

- Absolute intra-package imports (`from sparselora.… import …`) were rewritten as relative imports (`from .… import …`) so the package works under `axolotl.integrations.sparselora._vendor.sparselora`. Affected: `__init__.py`, `callback.py`, `modules/svd.py`. The example in `modules/registry.py`'s docstring was left as-is.
- `modules/svd.py`: a float-dtype guard (`_float_dtype`) so predictor factors load correctly when the base is a bitsandbytes 4-bit (QLoRA) weight, whose storage dtype is `uint8`. Upstream inferred the cast dtype directly from `base.weight.dtype`, which would cast the float SVD factors to `uint8` and corrupt them.

Each vendored file carries a one-line provenance header. **Do not hand-edit these files** — to update, re-copy from upstream at a new pinned commit and re-apply the import rewrite.

## Note on SVD predictor factors

Upstream `modules/svd.py` only *loads* pre-computed predictor factors from a `model.safetensors` (z-lab publishes these for Llama-2/3 only). The offline SVD computation that produces those factors is **not** in the upstream repo. Axolotl's plugin adds that step (`axolotl.integrations.sparselora.factors`) so predictors can be derived from any base model's weights, writing a `model.safetensors` in the exact key layout the vendored loader expects.
