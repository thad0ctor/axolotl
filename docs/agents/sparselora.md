# SparseLoRA — Agent Reference

Contextual-sparsity acceleration for LoRA fine-tuning. A training-free rank-8 SVD predictor skips redundant base-model neurons each step (LoRA stays dense). Vendored from [z-lab/sparselora](https://github.com/z-lab/sparselora) (ICML 2025) with **self-calibration** so it works on any (Llama) model + dataset, not just z-lab's published presets. Full usage: [src/axolotl/integrations/sparselora/README.md](../../src/axolotl/integrations/sparselora/README.md).

## Enable

```yaml
plugins:
  - axolotl.integrations.sparselora.SparseLoRAPlugin
adapter: lora
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
sample_packing: false
sparselora:
  target_sparsity: 0.9
  calibration: {method: faithful}
```

## Config keys (`sparselora:`)

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `true` | Master switch |
| `target_sparsity` | `0.9` | Overall base-path sparsity to aim for |
| `predictor_rank` | `8` | SVD predictor rank |
| `start_step` / `end_step` | `0.1` / `1.0` | Fraction (≤1) or absolute step; sparsity active in `[start, end)` |
| `layer_sparsity` | `null` | Explicit `{module_path: ratio}`; required for `method: none` |
| `calibration.method` | `faithful` | `faithful` \| `proxy` \| `none` |
| `calibration.num_samples` | `128` | Calibration examples from the same dataset |
| `calibration.warmup_steps` | `50` | `faithful` only; dense reference steps (rolled back) |
| `calibration.loss_budget` | `0.01` | Max per-layer reconstruction error when raising sparsity |
| `cache_dir` | `{output_dir}/sparselora_calibration` | Calibration artifacts location (logged at INFO) |
| `share_calibration` | `false` | Opt-in anonymous schedule sharing (schedule only, never data) |

## How it works

1. Hook: `post_trainer_create` (model + LoRA built, trainer created, before training / DDP wrap).
2. Cache lookup by hash of (model, adapter, dataset signature, target sparsity, rank, calibration params). Hit → reuse `schedule.json` + `model.safetensors`.
3. Miss → compute SVD predictor factors from base weights (`factors.py`), run the sensitivity sweep on a slice of the same dataset (`calibration.py`), allocate per-layer sparsity within `loss_budget`, cache it.
4. `apply_sparselora` swaps in sparse Llama modules; a `SparseLoRACallback` gates sparsity by `start_step`/`end_step`.

The SVD factors are training-free (`w1@w2 ≈ Wᵀ`, exact at full rank). z-lab ships factors only for Llama-2/3; computing them here is what makes the plugin model-agnostic.

## Constraints (v1)

- Attention-only LoRA (`q/k/v/o_proj`); MLP must not be LoRA-wrapped — **validated at startup**.
- Full-precision base only: `load_in_4bit`/`load_in_8bit` (QLoRA) rejected — the sparse linear needs a dense weight.
- `sample_packing: false`.
- Llama architecture only (extend via `register_sparse_module`).
- Single-GPU / DDP; no FSDP or DeepSpeed ZeRO-3.

## Source map

| File | Role |
|------|------|
| `plugin.py` | `SparseLoRAPlugin`, hook, validation, apply |
| `args.py` | `SparseLoRAArgs` Pydantic config |
| `factors.py` | SVD predictor-factor computation from base weights |
| `calibration.py` | Sensitivity sweep + schedule allocation |
| `cache.py` | Cache key/IO + opt-in telemetry stub |
| `_vendor/sparselora/` | Vendored z-lab code (see `_vendor/PROVENANCE.md`) |
