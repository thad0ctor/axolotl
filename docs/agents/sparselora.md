# SparseLoRA — Agent Reference

Contextual-sparsity acceleration for LoRA fine-tuning. A training-free rank-8 SVD predictor skips redundant base-model neurons each step (LoRA stays dense). Vendored from [z-lab/sparselora](https://github.com/z-lab/sparselora) (ICML 2025) with **self-calibration** (so it works on any model + dataset, not just z-lab's published presets) and **model-agnostic wiring** (Llama, Qwen2, Qwen3, Mistral, and any other SwiGLU-MLP / standard-attention architecture, auto-detected at apply time). Full usage: [src/axolotl/integrations/sparselora/README.md](../../src/axolotl/integrations/sparselora/README.md). Adding a new architecture's wiring: skill [`.agents/skills/sparselora-add-model`](../../.agents/skills/sparselora-add-model/SKILL.md).

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
4. `register_arch_wiring` introspects the loaded model and maps each MLP/attention class to the generic sparse wiring (Llama is pre-registered); `apply_sparselora` swaps in the sparse modules; a `SparseLoRACallback` gates sparsity by `start_step`/`end_step`.

The SVD factors are training-free (`w1@w2 ≈ Wᵀ`, exact at full rank). z-lab ships factors only for Llama-2/3; computing them here is what makes the plugin model-agnostic.

## Architecture support

MLP sparsity is near-universal: the gated SwiGLU forward (`down(act(gate(x)) * up(x))`) is the same across families up to the gate activation, which `SparseSwiGLUMLP` reads from the model (SiLU → fused liger kernel; `gelu_tanh` → Gemma path) for both the predictor's channel selection and the compute. Attention is handled by a generic `SparseAttention` (`arch_wiring.py`) that introspects per-arch differences:

| Arch | Handled by |
|------|-----------|
| Qwen2 | q/k/v projection **bias** via `SparseLinearBias`; `sliding_window` forwarded to the attention interface |
| Qwen3 | `q_norm`/`k_norm` (RMSNorm on the head dim) applied between view and RoPE transpose; `sliding_window`; **decoupled head_dim** (q_out ≠ hidden) via the fixed GQA predictor |
| Mistral | `sliding_window` from config |
| StableLM, Cohere | SiLU SwiGLU + standard attention — auto-registered, no special handling |
| Gemma2 | **supported** — `gelu_tanh` gated MLP (`SparseSwiGLUMLP` + `_GELUFFNPredictor`), attention-logit **softcapping** forwarded to the attention interface, `query_pre_attn_scalar` scaling and `sliding_window` inherited. Dense-apply is logit-exact; convergence tracks dense |
| Gemma3 (text) | **supported** — same `gelu_tanh` MLP path; `q_norm`/`k_norm`, `sliding_window`, no softcap by default |
| Phi3 | not auto-registered — fused `qkv_proj` / `gate_up_proj` are neither standard attention nor SwiGLU MLP |

Registration is automatic at apply time; no per-arch user config. The generic attention forwards Gemma2's `attn_logit_softcapping` and uses each architecture's own eager-attention fallback, so arch-specific attention terms are applied. Only **gated activations other than SiLU/`gelu_tanh`** and **fused-projection** layouts are unsupported. To extend further, register a custom sparse class via `register_sparse_module` — see the [`sparselora-add-model`](../../.agents/skills/sparselora-add-model/SKILL.md) skill.

## Constraints (v1)

- Attention-only LoRA (`q/k/v/o_proj`); MLP must not be LoRA-wrapped — **validated at startup**.
- Base: full-precision (`adapter: lora`) or 4-bit QLoRA (`adapter: qlora`, `load_in_4bit`; via `SparseLinear4bit`). 8-bit (`load_in_8bit`) not supported.
- `sample_packing: false`.
- SwiGLU-MLP (SiLU- or `gelu_tanh`-gated) + standard-attention architectures (Llama, Qwen2, Qwen3, Mistral, StableLM, Cohere, Gemma2, Gemma3-text, ...), auto-detected. Other gated activations and fused-projection models (Phi3) are rejected/skipped; extend custom wiring via `register_sparse_module`.
- Single-GPU / DDP; no FSDP or DeepSpeed ZeRO-3.
- `torch_compile: true` works — the dynamic sparse regions are `torch.compiler.disable` boundaries (graph-break + eager), the rest compiles.

## Source map

| File | Role |
|------|------|
| `plugin.py` | `SparseLoRAPlugin`, hook, validation, apply |
| `arch_wiring.py` | Model-agnostic sparse MLP/attention + auto-registration (`SparseSwiGLUMLP`, `SparseAttention`, `SparseLinearBias`, `register_arch_wiring`) |
| `args.py` | `SparseLoRAArgs` Pydantic config |
| `factors.py` | SVD predictor-factor computation from base weights (dequantizes 4-bit) |
| `sparse_linear_4bit.py` | `SparseLinear4bit` for QLoRA bases (dequantize-then-slice) |
| `calibration.py` | Sensitivity sweep + schedule allocation |
| `cache.py` | Cache key/IO + opt-in telemetry stub |
| `_vendor/sparselora/` | Vendored z-lab code (see `_vendor/PROVENANCE.md`) |
