# MegaTrain

MegaTrain enables full-parameter supervised fine-tuning when a model is too
large to remain in GPU memory. It keeps FP32 model parameters, gradients, and
Adam state in host RAM, then streams BF16 working copies one transformer layer
at a time to a CUDA device. Axolotl continues to own
configuration, dataset preparation, collation, scheduling, logging, and
checkpoint layout.

## Requirements

- At least one CUDA GPU with BF16 support. Additional GPUs are used by listing
  them in `megatrain_devices`.
- One Axolotl process. Launch through `axolotl train`, not `torchrun` or a
  distributed launcher.
- Enough host RAM for the model, optimizer state, working copies, dataset, and
  operating system.
- A built-in Transformers `LlamaForCausalLM` or `MistralForCausalLM` model with
  zero dropout.

## Usage

Add the plugin to a full-finetuning config:

```yaml
plugins:
  - axolotl.integrations.megatrain.MegaTrainPlugin

bf16: true
sample_packing: false
val_set_size: 0

megatrain_checkpoint_interval: 4
megatrain_num_grad_slabs: 12
```

Then run the normal Axolotl command:

```bash
axolotl train examples/smollm2/megatrain.yaml
```

The final output and intermediate model checkpoints use the standard Hugging
Face format and can be loaded without the MegaTrain integration.

## Configuration

| Option | Default | Description |
|---|---:|---|
| `megatrain_checkpoint_interval` | `4` | Number of transformer layers in a recomputation segment. Lower values retain more persistent checkpoints; higher values build a larger temporary recomputation cache. Benchmark both directions when tuning memory. |
| `megatrain_num_grad_slabs` | `12` | Number of reusable pinned host gradient buffers. More buffers can improve overlap but consume more RAM. Keep this at least twice the checkpoint interval. |
| `megatrain_devices` | current device | CUDA device indices to shard each microbatch across. Omit for single-GPU training. |
| `megatrain_fp32_head_grad` | `true` | Sum chunked cross-entropy gradients for the output head and final norm in FP32. Costs 4 bytes per `lm_head` element of GPU scratch on each device; set to `false` to trade some gradient precision for VRAM. |

## Multi-GPU

MegaTrain parallelizes over data, not parameters: it keeps one set of FP32 CPU
masters and spawns one worker process per GPU, each streaming the same layers
against its own shard of the microbatch. Gradients accumulate into shared-memory
buffers that the single CPU optimizer then steps.

```yaml
megatrain_devices: [0, 1, 2, 3]
micro_batch_size: 4
```

Axolotl still runs as a **single** process — launch with `axolotl train`, not
`torchrun` or an Accelerate multi-process runner. MegaTrain owns process
creation itself, so combining it with an external launcher is rejected.

`micro_batch_size` must be at least `len(megatrain_devices)`, because each
microbatch is sharded across the workers. Host RAM requirements are unchanged
(there is still exactly one CPU master), but each worker adds its own GPU
context, layer buffers, and pinned gradient slabs.

The integration derives the remaining MegaTrain settings from standard Axolotl
keys:

| Axolotl setting | MegaTrain use |
|---|---|
| `base_model` | Model identifier and CPU-loaded Hugging Face model |
| `bf16` / model dtype | Initial checkpoint loading and GPU compute dtype; CPU masters are promoted to FP32 |
| `attn_implementation` | `flash_attention_2`, `sdpa`, or `eager` attention |
| `micro_batch_size` | Per-step batch size |
| `gradient_accumulation_steps` | Number of streamed microsteps per optimizer step |
| `sequence_len` | Maximum sequence length |
| `learning_rate`, `weight_decay`, `adam_beta1`, `adam_beta2`, `adam_epsilon` | CPU optimizer settings |
| `megatrain_devices` | GPUs to shard each microbatch across |
| `max_grad_norm` | CPU-master gradient clipping threshold |

MegaTrain performs its own checkpointed layer streaming, so the example disables
Axolotl's `gradient_checkpointing` option.

An FP32 source checkpoint is loaded through Axolotl's requested BF16 dtype once
before the CPU masters are created. Subsequent optimizer updates, saved model
weights, and resumed training use FP32 masters without BF16 update rounding.

## Support matrix

| Feature | Status |
|---|---|
| Full-parameter SFT with FP32 CPU masters and BF16 streamed compute | Supported |
| Built-in Llama and Mistral causal LMs | Supported; other model types are rejected until loss and gradient parity is verified |
| Axolotl datasets and prompt strategies | Supported |
| Eager, SDPA, and Flash Attention 2 | Supported when available for the model |
| Uniform Mistral sliding-window attention | Supported |
| Standard Hugging Face model checkpoints | Supported |
| LoRA, QLoRA, PEFT adapters, and ReLoRA | Not supported |
| 4-bit, 8-bit, GPTQ, or other quantized model loading | Not supported |
| Sample packing | Not supported |
| Evaluation or validation datasets | Not supported; set `val_set_size: 0` |
| Resume from optimizer checkpoints | Supported through Axolotl's standard `resume_from_checkpoint` flow |
| DeepSpeed, FSDP, DDP, TP, context parallelism, Ray, or `torchrun` | Not supported |
| MegaTrain spawn-based multi-GPU training | Supported through `megatrain_devices` |
| DPO, KTO, ORPO, GRPO, reward modelling, or other RL/preference trainers | Not supported |
| Vision-language or multimodal training | Not currently supported |
| Nonzero dropout or mixed attention schedules | Not currently supported |
| Other Axolotl plugins or custom trainer/loss hooks | Not currently supported |
| Native MegaTrain CUDA extensions | Not included |

`torch_compile` is not recommended with CPU-master streaming. Unsupported
features are rejected early rather than falling back to Axolotl's stock trainer.

Two behaviours differ from a stock Axolotl run and are deliberate:

- Weight decay is not applied to embeddings or the output head, in addition to
  Axolotl's usual exclusion of norms and biases.
- Saved checkpoints hold FP32 weights, because the CPU masters are FP32.

## Memory sizing

Host memory must hold FP32 model parameters, accumulated FP32 gradients, and both
FP32 Adam moments. A rough steady-state tensor floor for transformer-layer
weights is approximately 18 GB per billion parameters: 4 GB for master weights,
4 GB for gradients, 8 GB for two Adam moments, and 2 GB for the pinned BF16
streaming copy.

Leave substantial additional capacity for model-loading and checkpoint-saving
transients, BF16 gradient slabs, pinned head and embedding buffers, Python and
framework allocations, prepared data, and the operating system. Measure a
representative small run before committing a host to a long training job.

GPU memory does not scale with the full parameter count in the usual way, but it
still includes the CUDA context, double-buffered copies of the largest layer,
embeddings and output head, and activations. Activation use grows with
micro-batch size, sequence length, hidden size, and checkpoint interval. If a run
is close to OOM, reduce `micro_batch_size` or `sequence_len`. Checkpoint-interval
memory is not monotonic, so test values above and below the current setting.

Gradient slabs are pinned host-memory buffers, not GPU buffers. Increasing
`megatrain_num_grad_slabs` raises pinned RAM use by roughly one largest-layer
gradient per slab. Keep the count at least twice the checkpoint interval.

Treat these estimates as planning guidance and leave headroom for architecture-
specific layers and allocator fragmentation.

## Provenance

The bundled runtime is a pruned, namespaced copy of MegaTrain. See the
[provenance record](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/integrations/megatrain/_vendor/PROVENANCE.md)
for its exact upstream revision,
license, excluded files, and local changes. MegaTrain is licensed under
Apache-2.0; the copied license is in this directory.
