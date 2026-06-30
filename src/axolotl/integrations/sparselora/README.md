# SparseLoRA

Faster LoRA fine-tuning via contextual sparsity: a training-free SVD predictor skips redundant frozen-base neurons each step, while the LoRA adapter stays dense. Wraps [z-lab/sparselora](https://github.com/z-lab/sparselora) (ICML 2025, MIT; vendored under `_vendor/`) with **self-calibration** so it works on *your* model and dataset (not just z-lab's Llama-2/3 presets), and auto-wiring for most modern architectures. The paper reports ~1.6× wall-clock / ~2.2× compute with no accuracy loss; we measure ~1.47× on Qwen3.5-9B (s=0.9).

## Quickstart

```yaml
plugins:
  - axolotl.integrations.sparselora.SparseLoRAPlugin

adapter: lora
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]   # attention-only (required)
sample_packing: false                                   # required

sparselora:
  target_sparsity: 0.9      # higher = faster, more loss
  start_step: 0.1           # first 10% of training stays dense
```

Calibration runs automatically before training and caches its result — nothing else to set up.

## Config

| Key | Default | What |
|---|---|---|
| `target_sparsity` | `0.9` | how aggressively to sparsify — lower it if loss matters more than speed |
| `predictor_rank` | `8` | SVD predictor rank |
| `start_step` | `0.1` | fraction of training kept dense up front (gradient stability) |
| `calibration.method` | `faithful` | `faithful` (accurate) · `proxy` (fast, forward-only) · `none` (supply your own `layer_sparsity`) |

`fused_attn_kernel: true` adds a small extra speedup on gated models (Qwen3.5/3.6).

## Supported models

Auto-detected at apply time: **Llama, Qwen2/3, Qwen3.5/3.6 (gated + MoE), Mistral, Cohere, StableLM, Gemma2/3/4, Phi3**, and most SwiGLU-MLP transformers. Only gated MLPs with an activation other than SiLU/`gelu_tanh` are unsupported (refused with a clear error). Full matrix and how to add a model: [`docs/agents/sparselora.md`](../../../../docs/agents/sparselora.md).

## Limitations (v1)

- Attention-only LoRA (`q/k/v/o_proj`, or `qkv_proj/o_proj` for Phi3) — don't LoRA the MLP.
- `sample_packing: false`.
- Single-GPU or DDP — no FSDP / DeepSpeed ZeRO-3.
- bf16 (`adapter: lora`) or 4-bit QLoRA (`adapter: qlora`); 8-bit not supported.
- `torch_compile: true` works (the sparse path runs eager, the rest compiles).

## Calibration & cache

Before training, a sensitivity sweep over a slice of *your* dataset sets per-layer sparsity to approach `target_sparsity` while keeping each layer within `loss_budget`; sensitive layers stay dense automatically. Results cache to `{output_dir}/sparselora_calibration/{key}/` (path logged at INFO) — delete that directory or change any input to recalibrate. `share_calibration` (off by default) would upload only the schedule + model id, never dataset content; no endpoint is wired, so nothing leaves your machine.

See `_vendor/PROVENANCE.md` for the vendored upstream commit and local edits.
