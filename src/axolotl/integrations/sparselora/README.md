# SparseLoRA

Faster LoRA fine-tuning via contextual sparsity: a training-free SVD predictor skips redundant frozen-base neurons each step, while the LoRA adapter stays dense. Wraps [z-lab/sparselora](https://github.com/z-lab/sparselora) (ICML 2025, MIT; vendored under `_vendor/`) with **self-calibration** so it works on *your* model and dataset (not just z-lab's Llama-2/3 presets), and auto-wiring for most modern architectures. The paper reports ~1.6× wall-clock / ~2.2× compute with no accuracy loss; we measure ~1.47× on Qwen3.5-9B (s=0.9).

## Quickstart

```yaml
plugins:
  - axolotl.integrations.sparselora.SparseLoRAPlugin

adapter: lora
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]   # default safe target set

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
| `attn_sparsity` | `min(target, 0.75)` | attention band sparsity (attention tolerates far less than the MLP) |
| `calibration.method` | `structural` | `structural` (default, no forward pass) · `preset` (load a published z-lab schedule) · `faithful`/`proxy` (structural + a sensitivity sweep that demotes the most sensitive layers) · `none` (supply your own `layer_sparsity`) |

To reproduce z-lab's downstream-validated schedule on the Llama models they calibrated:

```yaml
sparselora:
  calibration:
    method: preset
  preset: z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA
  preset_mode: o2   # o1 = conservative, o2 = aggressive
```

`fused_attn_kernel: true` adds a small extra speedup on gated models (Qwen3.5/3.6).

## Supported models

Auto-detected at apply time: **Llama, Qwen2/3, Qwen3.5/3.6 (gated + MoE), Mistral, Cohere, StableLM, Gemma2/3/4, Phi3**, and most SwiGLU-MLP transformers. Vision attention is supported for Qwen3-VL, Gemma4, Gemma3/SigLIP, and InternVL-style towers; non-gated ViT MLPs with `fc1/fc2` or `linear_fc1/linear_fc2` are supported too. Only gated MLPs with an activation other than SiLU/`gelu_tanh` are unsupported (refused with a clear error). Full matrix and how to add a model: [`docs/agents/sparselora.md`](../../../../docs/agents/sparselora.md).

## Compatibility (v1)

| | Status | Notes |
|---|:---:|---|
| LoRA (bf16) | ✅ | `adapter: lora` |
| QLoRA (4-bit) | ✅ | `adapter: qlora` + `load_in_4bit` |
| 8-bit base | ❌ | |
| Attention LoRA | ✅ | `q/k/v/o_proj`, `qkv/proj`, `out_proj`, or `projection_layer` depending on architecture |
| MLP LoRA | ✅ | `gate/up/down`, fused `gate_up`, `fc1/fc2`, or Qwen3-VL `linear_fc1/linear_fc2` when the MLP layout is sparsifiable |
| Single GPU | ✅ | |
| DDP (multi-GPU) | ✅ | |
| FSDP | ✅ | FSDP1 + FSDP2; swap runs before wrapping, weight is all-gathered in forward |
| DeepSpeed ZeRO-1/2 | ✅ | |
| DeepSpeed ZeRO-3 | ❌ | parameter sharding |
| `sample_packing` | ✅ | multi-segment output mask; sparsity is diluted (shorter per-doc contexts) |
| `torch_compile` | ✅ | sparse path runs eager, the rest compiles |

## Calibration & cache

z-lab derive their published schedules from an offline *downstream-task* sensitivity analysis (progressively sparsifying each layer and measuring task accuracy) — their schedules run MLP layers at 0.97–0.99 sparsity, so an absolute per-block reconstruction-error budget cannot recover them (it returns near-zero sparsity). Calibration here therefore follows z-lab's *empirical structure* rather than reconstruction error: the default `structural` method applies their profile — dense shallow + final layers, aggressive deep MLP (`target_sparsity`), milder attention (`attn_sparsity`) over a still-deeper band — from the model's layer layout alone (no forward pass). On Llama-3-8B this lands at ~45/64 modules, mean sparsity ~0.59, matching z-lab's o2 (42/64, 0.597); `preset` reproduces o2 exactly. `faithful`/`proxy` additionally run a sweep over a slice of *your* dataset to demote the most sensitive band layers back to dense. Results cache to `{output_dir}/sparselora_calibration/{key}/` (path logged at INFO) — delete that directory or change any input to recalibrate. `share_calibration` (off by default) would upload only the schedule + model id, never dataset content; no endpoint is wired, so nothing leaves your machine.

See `_vendor/PROVENANCE.md` for the vendored upstream commit and local edits.
