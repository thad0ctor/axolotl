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

### 5. Calibration under multi-GPU — base path is safe; broadcast hardening is PARKED

The default `faithful`/`proxy` calibration runs inside `post_trainer_create`, i.e. on the **unwrapped, replicated** model on every rank. The calibration DataLoader uses `shuffle=False` over the first `num_samples` examples (`calibration.py:99-104`), factors come from the (identical) base weights, and the faithful warm-up does identical local LoRA steps on identical data per rank. So every rank deterministically computes the **same schedule** → the same module swap → matching FSDP flat-param structure across ranks. This is the path that was **verified in round 1**: a non-empty calibrated schedule (16/16 modules, mean sparsity 0.50) applied consistently and trained to completion under FSDP2, with all ranks calibrating in lockstep. **The base FSDP support (committed) ships without any calibration hardening and is safe** on this determinism argument plus the round-1 evidence.

> **Optional hardening — implemented + unit-tested, but PARKED (e2e deadlocked twice).** The residual risk is bf16-reduction nondeterminism flipping a single top-k / loss-budget decision differently across ranks, desyncing the swap. A first attempt — *calibrate on rank 0 only, then broadcast* — is asymmetric and deadlocks (non-zero ranks issue collectives rank 0 never reaches). It was rewritten to a **symmetric** design: every rank runs the full calibration (collectives stay matched), then a single `broadcast_object_list` that **all ranks issue** overwrites every rank's `layer_sparsity` with rank 0's, followed by a `barrier`; only the disk write (`save_cached` / `maybe_share`) is rank-0-gated. Helpers: `plugin.py::_broadcast_rank0_schedule`, `_calibrate_distributed`, `_compute_calibration`; the rank-gating logic is unit-tested (`TestRank0Broadcast`, with a `dist` double).
>
> **Status: not e2e-verified.** Both 2-GPU verification runs hit the 600 s hard timeout (`ACCEL_EXIT=124`) and were aborted — no 3-hour hang, the timeout did its job. Critically, the second run's log froze at `"[RANK:1] Loading raw datasets"`, i.e. **during distributed dataset preparation, *before* `post_trainer_create` (and therefore before any of the broadcast code) runs**. That points at the probe harness / FSDP dataset-prep path, not necessarily the hardening logic. Because the committed base path does not need this change, the hardening is **left uncommitted on `feat/sparselora-fsdp` and parked pending a clean e2e repro** (a minimal 2-GPU run that reaches `post_trainer_create` without hanging in dataset prep, e.g. with a pre-tokenized dataset and no monkeypatch harness).

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
| full-precision LoRA | FSDP2 | lora | **faithful + broadcast hardening** | ⚠️ NOT verified — both 2-GPU runs hit the 600 s timeout; froze in distributed dataset prep, before `post_trainer_create`. Parked. |

## Implementation (branch `feat/sparselora-fsdp`)

### Committed (`836849a1e`) — base FSDP support, e2e-verified

Minimal, behavior-preserving:

1. `src/axolotl/integrations/sparselora/plugin.py` — `_validate`: removed the `cfg.fsdp` rejection; keep only the DeepSpeed ZeRO-3 reject (ZeRO-3 shards params without an in-forward all-gather of `self.weight`, so it stays unsupported, matching the existing ZeRO-1/2-allowed logic).
2. `tests/integrations/sparselora/test_sparselora_plugin.py` — `test_fsdp_rejected` → `test_fsdp_accepted` (FSDP must now pass validation).
3. `src/axolotl/integrations/sparselora/README.md` — compatibility matrix: FSDP ✅ (FSDP1 + FSDP2).
4. `docs/agents/sparselora.md` — constraints: "Single-GPU / DDP / FSDP (FSDP1 and FSDP2); no DeepSpeed ZeRO-3."
5. `examples/llama-3/lora-sparselora-fsdp.yml` — new example config.

This commit is **90 sparselora tests passing** (17 skipped), pinned pre-commit clean, and verified end-to-end on 2 GPUs (the matrix above, rows 1-7).

### Uncommitted / PARKED — multi-GPU calibration broadcast hardening (NOT e2e-verified)

Left in the working tree, **not committed**, pending a clean e2e repro (see §5):

6. `plugin.py` — split out the pure `_compute_calibration` (runs on all ranks, no I/O), added the `_calibrate_distributed` dispatch and the testable `_broadcast_rank0_schedule` helper (all ranks calibrate; all ranks broadcast; rank 0 writes; barrier). `test_sparselora_plugin.py::TestRank0Broadcast` covers the rank-gating with a `dist` double (no process group needed) — unit tests pass (92 total with these), pre-commit clean. **The two 2-GPU e2e verification runs both deadlocked and were timeout-aborted; the hang occurred in distributed dataset prep, before this code runs.**

## Recommended next step

Land the committed base FSDP support. For the parked hardening: build a minimal e2e repro that reaches `post_trainer_create` cleanly (pre-tokenized dataset, no monkeypatch harness, short NCCL timeout) to determine whether the dataset-prep hang is harness-specific; only then verify the broadcast and land it. Optional: a small e2e multi-GPU smoke test in CI mirroring the FSDP2 faithful run. DeepSpeed ZeRO-3 remains out of scope (would need an FSDP-style in-forward gather hook around the sparse slice).
