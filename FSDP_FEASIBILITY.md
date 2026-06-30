# SparseLoRA under FSDP — Feasibility Report

## Verdict

**It already works — the guard was overly conservative.** SparseLoRA trains correctly under both **FSDP1 and FSDP2**, for both full-precision LoRA and 4-bit QLoRA, including the default `faithful` calibration path. The fix is to **drop the FSDP rejection** in the plugin's `_validate` (DeepSpeed ZeRO-3 stays rejected). No FSDP-aware sparse linear, no `ignore_modules`, no swap-reordering was needed — the existing ordering already makes the mechanism safe.

This was verified end-to-end on 2× RTX 3090 (GPU idx 1,2), tiny Llama (`axolotl-ai-co/tiny-llama-50m`, GQA), with sparsity active from step 0.

## Why the original guard was wrong

The suspected conflicts in the task brief all dissolve once you trace the *timing*. The single fact that makes FSDP safe:

> The SparseLoRA module swap runs **before** FSDP wraps the model, and FSDP **all-gathers the full base weight in the forward pre-hook**, so the contextual row/column slicing operates on the full materialized `self.weight` — exactly as it does single-GPU.

### 1. Timing/order (the load-bearing fact)

- The plugin's `post_trainer_create` hook is invoked at `src/axolotl/core/builders/base.py:457` (`get_post_trainer_create_callbacks` → `PluginManager.add_callbacks_post_trainer` → `SparseLoRAPlugin.post_trainer_create`), immediately after the `Trainer` object is constructed at `base.py:444`.
- FSDP wrapping does **not** happen at `Trainer.__init__`. It happens lazily inside `trainer.train()` (`accelerator.prepare`/`_wrap_model` in the inner training loop). Axolotl only sets the FSDP env flags up front (`src/axolotl/utils/trainer.py:600` `setup_fsdp_envs`); the actual `FullyShardedDataParallel`/`fully_shard` call is deferred to train time.
- Therefore `apply_sparselora` (`_vendor/sparselora/api.py:109`) — the in-place module swap (`api.py:129-147`), the `model.forward` patch (`api.py:59-70`), and the global `peft...lora.Linear.forward = lora_forward` patch (`api.py:124`) — **all run on the unwrapped model**. FSDP then flattens/shards the *already-sparse* modules. This is exactly the favorable "swap-before-wrap" path; the feared "re-class a module after FSDP wrapped it" never happens.

### 2. Weight access under FSDP (the core concern — resolved)

The sparse forward does `self.weight[sparse_indices]` (output-sparse) or `self.weight[:, sparse_indices]` (input-sparse) at `_vendor/sparselora/modules/linear.py:26-28`. Under FSDP the wrapped unit (the `LlamaDecoderLayer`) is unsharded in the forward pre-hook, and the original parameter view is restored, so `self.weight` is the **full `[out, in]` tensor during forward**. Proven directly with an isolated 2-GPU probe wrapping a `SparseLinear` in FSDP1:

```
[use_orig_params=True ] OK  weight_shape_in_fwd=(256, 256) numel=65536 (full=65536) sliced=(128, 256)
[use_orig_params=False] FSDP WRAP FAILED: ValueError: ... has both parameters with
                        requires_grad=True and False ... not support ... use_orig_params=False
```

- **FSDP1**: needs `use_orig_params=True` for the weight to be a real full-tensor view (and it is mandatory anyway for LoRA's mixed `requires_grad` flat param). In the real axolotl path this is satisfied automatically — even a config with `use_orig_params: false` trained fine, because PEFT/accelerate force the original-params path for adapter models (axolotl only *opts in* via env at `utils/trainer.py:614`, and accelerate's PEFT handling supplies the rest).
- **FSDP2** (`fully_shard`): params are DTensors that unshard to the full local tensor in the forward pre-hook; the fancy-index slicing executes without DTensor errors. Verified end-to-end.

The output-scatter path (`linear.py:31-42`) compares `x.shape[-1]` against `self.weight.shape[0]` (full out-dim) and scatters into a full-width buffer that feeds the residual stream — if `self.weight` were a half-size shard this would have thrown a shape error at the residual add. It didn't, on any run.

### 3. Predictor params/buffers

The SVD predictors register everything as **buffers**, not parameters (`_vendor/sparselora/modules/predictors.py:26-27,43-44,62-65`; `arch_wiring.py:427,439-440`). FSDP does not shard buffers — it replicates/broadcasts them — so each rank holds the full predictor factors. No flat-param accounting issue.

### 4. The mask `partial`-binding and the global `lora_forward` patch

Both survive. `_patch_forward` (`api.py:59-70`) patches the **top-level** `model.forward`; FSDP's root wrapper calls `self._fsdp_wrapped_module(...)` which dispatches to that patched bound method. Inside it, the per-step mask is bound onto each `SparseModule.forward` via `functools.partial` (`api.py:67`). Because the sparse submodules are *inside* an FSDP unit (not themselves separate FSDP wrappers under `TRANSFORMER_BASED_WRAP` at the decoder-layer granularity), `self.mlp`/`self.self_attn` are still the raw sparse modules and the instance-level `.forward` override is honored by `nn.Module.__call__`. The global `peft.tuners.lora.layer.Linear.forward = lora_forward` patch is a class-level patch applied pre-wrap and is unaffected by wrapping.

### 5. Calibration under multi-GPU (a real subtlety — but safe as-is)

The default `faithful`/`proxy` calibration runs inside `post_trainer_create`, i.e. on the **unwrapped, replicated** model on every rank, with **no distributed collectives** (no `all_reduce`/`barrier`/`DistributedSampler`). The calibration DataLoader uses `shuffle=False` over the first `num_samples` examples (`calibration.py:99-104`), factors come from the (identical) base weights, and the faithful warmup does identical LoRA steps on identical data per rank. So every rank deterministically computes the **same schedule** → the same module swap → matching FSDP flat-param structure across ranks. Verified: a non-empty calibrated schedule (16/16 modules, mean sparsity 0.50) applied consistently and trained to completion under FSDP2 with no cross-rank flat-param mismatch.

> Residual risk (not observed): bf16 reduction nondeterminism could in principle flip a single top-k/budget decision differently across ranks, which would desync the module swap. If that is ever seen in the wild, the robust hardening is *calibrate on rank 0 and broadcast the schedule*. It was not needed for any run here.

## What was tested (2× RTX 3090, idx 1,2)

| Run | FSDP | adapter | calibration | result |
|---|---|---|---|---|
| full-precision LoRA | FSDP1, `use_orig_params=True` | lora | none (explicit 0.5) | trained, loss 3.96→2.75, ckpt saved |
| full-precision LoRA | FSDP1, `use_orig_params=False` | lora | none | trained (PEFT forces orig-params) |
| full-precision LoRA | FSDP2 (`fully_shard`) | lora | none | trained, no DTensor errors |
| full-precision LoRA | FSDP2 | lora | **faithful** (empty sched) | trained (dense fallback, consistent) |
| full-precision LoRA | FSDP2 | lora | **faithful** (16/16 sparse) | trained, cross-rank consistent |
| QLoRA 4-bit | FSDP1 | qlora | none | trained, `SparseLinear4bit` dequant-slice OK |
| isolated probe | FSDP1 | — | — | confirmed full weight in forward |

## Implementation (branch `feat/sparselora-fsdp`)

Minimal, behavior-preserving:

1. `src/axolotl/integrations/sparselora/plugin.py` — `_validate`: removed the `cfg.fsdp` rejection; keep only the DeepSpeed ZeRO-3 reject (ZeRO-3 shards params without an in-forward all-gather of `self.weight`, so it stays unsupported, matching the existing ZeRO-1/2-allowed logic).
2. `tests/integrations/sparselora/test_sparselora_plugin.py` — `test_fsdp_rejected` → `test_fsdp_accepted` (FSDP must now pass validation).
3. `src/axolotl/integrations/sparselora/README.md` — compatibility matrix: FSDP ✅ (FSDP1 + FSDP2).
4. `docs/agents/sparselora.md` — constraints: "Single-GPU / DDP / FSDP (FSDP1 and FSDP2); no DeepSpeed ZeRO-3."
5. `examples/llama-3/lora-sparselora-fsdp.yml` — new example config.

All **90 sparselora tests pass** (17 skipped), pinned pre-commit (ruff/ruff-format/mypy/bandit) **clean**.

## Recommended next step

Land the guard drop. Optional follow-ups, in priority order: (a) a small e2e multi-GPU smoke test in CI mirroring the FSDP2 faithful run; (b) only if cross-rank schedule desync is ever observed, add rank-0-calibrate + `broadcast_object_list` of the schedule in `post_trainer_create`. DeepSpeed ZeRO-3 remains out of scope (would need an FSDP-style in-forward gather hook around the sparse slice).
