# SparseLoRA

Contextual-sparsity acceleration for LoRA fine-tuning, integrated as an Axolotl plugin. Wraps [z-lab/sparselora](https://github.com/z-lab/sparselora) (ICML 2025, MIT; vendored under `_vendor/`) and adds **self-calibration**: the per-layer sparsity schedule and the SVD predictor factors are derived from *your* base model and *your* dataset, rather than z-lab's Llama-2/3-only published presets.

## What it does

Each training step, a training-free rank-8 SVD predictor selects which base-model neurons matter for the current context and skips the rest in the frozen-weight forward/backward path (the LoRA adapter stays dense). The paper reports up to ~2.2× compute and ~1.6× wall-clock reduction without accuracy loss.

## Usage

```yaml
plugins:
  - axolotl.integrations.sparselora.SparseLoRAPlugin

base_model: meta-llama/Llama-3.1-8B-Instruct
adapter: lora
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]   # attention-only (required, see below)
sample_packing: false                                   # required (see below)

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

sparselora:
  target_sparsity: 0.9      # overall base-path sparsity to aim for
  predictor_rank: 8
  start_step: 0.1           # first 10% of training stays dense (gradient stability)
  calibration:
    method: faithful        # faithful (default) | proxy | none
    num_samples: 128
    warmup_steps: 50        # faithful only
    loss_budget: 0.01       # max per-layer reconstruction error when raising sparsity
```

## Calibration

Before training, the plugin runs a sensitivity sweep over a slice of the **same** tokenized dataset and config the run will train on, and allocates per-layer sparsity to approach `target_sparsity` while keeping each layer's reconstruction error within `loss_budget`. Sensitive layers (often the earliest) stay dense automatically.

- **`faithful`** (default) — a short dense LoRA warm-up first, so the sweep sees realistic activations (the paper's "start from a fine-tuned reference"). The warm-up's weight updates are rolled back so real training starts clean.
- **`proxy`** — forward-only sweep, no warm-up. Fast; works on any model.
- **`none`** — skip the sweep and use an explicit `sparselora.layer_sparsity` map (keyed by full module path, e.g. `model.layers.3.mlp`).

## Cache (where it lives)

Calibration artifacts — the schedule (`schedule.json`) and SVD factors (`model.safetensors`) — are written to:

```
{output_dir}/sparselora_calibration/{key}/
```

(or `sparselora.cache_dir/{key}/` if set). `key` is a hash of the base model, adapter, dataset signature, target sparsity, predictor rank, and calibration params. A matching key reuses the artifacts; **the resolved path is logged at INFO on every run**. To force recalibration, delete that directory or change any input.

### Optional upstream sharing (off by default)

`sparselora.share_calibration: true` assembles an anonymous payload — model id and sparsity params and the resulting schedule **only, never any dataset content** — intended for an authors' calibration database. No endpoint is wired yet, so v1 only assembles and logs the payload; nothing is transmitted. Leave it `false` (default) and nothing leaves your machine.

## Requirements & limitations (v1)

- **Attention-only LoRA.** `lora_target_modules` must be attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`). MLP projections must not be LoRA-wrapped — the MLP sparse path is incompatible with a LoRA-wrapped MLP linear. Validated at startup.
- **`sample_packing: false`.** The context/output token split assumes unpacked sequences.
- **Architecture: Llama.** Sparse module wiring is registered for Llama (`LlamaMLP`/`LlamaAttention`). Other architectures fail fast with a clear message; extend via `register_sparse_module`.
- **Single-GPU / DDP.** FSDP and DeepSpeed ZeRO-3 (parameter sharding) are not yet supported.
- **`torch_compile` compatible.** The plugin marks its data-dependent regions (top-k channel prediction, boolean-mask token splits, 4-bit dequant) as `torch.compiler.disable` boundaries, so `torch_compile: true` runs without error — Dynamo graph-breaks around the sparse path (which executes eagerly and is already fast) and compiles the rest of the model. Contextual sparsity is inherently dynamic, so the sparse matmuls themselves are not captured in the compiled graph.
- **Full-precision (`adapter: lora`) or 4-bit QLoRA (`adapter: qlora`, `load_in_4bit`) base.** For a 4-bit base, `SparseLinear4bit` dequantizes the frozen weight (which bitsandbytes does internally anyway at training batch sizes) and slices the matmul — a microbenchmark shows output-sparse projections (q/k/v/gate/up) land within ~5–15% of the full-precision SparseLoRA path (e.g. gate/up: 2.0×/3.2×/6.7× over dense QLoRA at sparsity 0.5/0.7/0.9). Input-sparse projections (o/down) only win at sparsity ≥0.7 — a pre-existing property of the gather-based path, not specific to 4-bit. 8-bit bases (`load_in_8bit`) are not supported.

See `_vendor/PROVENANCE.md` for the vendored upstream commit and the (import-only) local edits.
