---
name: model-verification-harness
description: "Verifies that a model wires into axolotl correctly and trains sanely — a reproducible smoke gate for model-support PRs. Detects model_config_type + arch features (MoE/multimodal/SSM-hybrid), then runs a tiered gate ladder: config-resolution compat matrix (liger/CCE/kernel flags via the real load_cfg), an adaptive wired-everywhere integration checklist, preprocess + chat-template masking + sample-packing checks, and (on GPU) loss-sanity + cross-variant numerical consistency that catches silent kernel-shadowing. Emits a standardized report + manifest with a 0/1/2 exit code. Use when asked to verify model support, run the model verification harness, smoke-test a new-model PR, check whether a model wires into axolotl, or debug masking/packing/kernel-compat for a model type."
---

# Model verification harness

The bespoke part of adding a model (the attention/RoPE/mask patch) can't be a
skill — but *verifying* a model is a repeatable, scriptable gate, and that is what
this builds. It answers: does the model wire into every hook, do the kernel flags
resolve, is masking correct, does loss behave, and do kernels keep the math the
same?

Everything runs through `scripts/verify_model.py` (the reproducible feeder) plus
the `harness/` package. Run it **in the training environment** — kernels and
plugins are env-specific, so verify against the exact env you train with.

## The gate ladder (cheap → expensive, CPU → GPU)

| Gate | Cost | Checks |
|---|---|---|
| **G1** config | CPU ms | composite-first compat matrix through the real `load_cfg`; four-way verdict per cell (rejected / normalized / **warned-no-op** / supported). Bisects an unexpected reject to the minimal failing flag set. |
| **G2** integration | CPU ms | adaptive "wired-everywhere" checklist; flags **silent generic fallbacks** (the shadow failure mode), not just missing hooks. |
| **G3** preprocess | CPU/GPU s | tiny preprocess → prepared dataset has `input_ids`/`labels`/`attention_mask`. |
| **G4** masking | CPU s | decode trained vs masked spans; assert assistant-trained / prompt-masked; diff vs a saved snapshot. |
| **G5** packing | CPU/GPU s | multipack membership, packed len ≤ `sequence_len`, per-document `position_ids` reset. |
| **G6** loss | GPU min | short real train → loss finite (no NaN/inf) and non-diverging. |
| **G7** consistency | GPU min | **flagship**: step-0 loss with a kernel ON must ≈ baseline within `rtol`. Kernels change speed, not math — a delta is a silent numerical regression (the liger-0.8.0 shadow class) green CI misses. |
| **G8** coverage | CPU ms | which tests reference this type/patch-surface; advisory gap list; `--emit-test` scaffolds a regression test from the verified matrix. |

G1–G5 + G8 are CPU and CI-gateable. **G6/G7 need a GPU and are local/on-demand.**

## Stage 1: Detect

The harness reads the model's `config.json` and resolves `model_config_type` (the
routing key everything else keys off), plus whether it's MoE / multimodal /
SSM-hybrid / needs a patch / uses custom embedding names. Irrelevant gates and
checklist rows are pruned automatically from these features.

## Stage 2: Ask (the checklist prompt)

Present the user a short checklist (one multi-select for *what to test* + follow-up
prompts) and wait for answers before running — don't assume:

1. **Model & tiny strategy** — the `--base-model` (path or HF id) and
   `--tiny-strategy`: `path` (default — use the given model as-is),
   `checkpoint` (auto-match a tiny `axolotl-ai-co/tiny-*` by arch), or `shrink`
   (random-init a 2-layer model from the target's `config.json` — use this for a
   **brand-new arch with no published checkpoint**, so the run gates can still run).
2. **What to verify (multi-select):** the static gates (G1 config / G2 integration /
   G8 coverage — CPU, always cheap); the plumbing gates (G3 preprocess / G4 masking /
   G5 packing); the **model-loading + GPU gates** (G6 loss / G7 numerical
   consistency); and **opt-in extras** — quant axes (qlora/4bit/8bit) and RL config
   resolution (dpo/kto/orpo/grpo) via `--profile full`, packing loss-parity via
   `--features pack_parity`, multimodal image-token masking (auto when the model is
   multimodal). **Distributed correctness** (FSDP2 / expert-parallel / context-parallel)
   is *not* an automatic run — the harness verifies it wires + resolves
   (`--profile multigpu`), but actually executing a multi-GPU run is a separate,
   user-driven step; offer it as a checklist item and note it needs the real
   multi-GPU launch.
3. **Anything that loads a model or uses a GPU → pick GPUs.** If G6/G7/G5-parity or
   `--tiny-strategy shrink/path` will load weights, show the detected GPUs (the
   harness prints an inventory: index · name · memory) and ask **which devices** to
   use, as a checklist. Pass the choice via `--gpus 1,2`. If the chosen GPU is
   **incompatible** (too little memory for the model, or it's a card the user
   excludes), say so and re-prompt rather than failing mid-run. With no GPU, drop
   G6/G7 (they self-skip).
4. **Environment** — which `python` / venv / uv / conda (kernels are env-specific —
   run inside the training env).
5. **Report** — whether to attach the markdown report to a PR summary.

## Stage 3: Run

```bash
# CPU-only smoke (CI-gateable): config + integration + preprocess/masking/packing + coverage
python .agents/skills/model-verification-harness/scripts/verify_model.py \
  --base-model <path-or-hf-id> --gates G1,G2,G3,G4,G5,G8 \
  --report report.md --manifest manifest.json

# Full incl. GPU gates (loss + numerical consistency) on chosen GPUs, + quant/RL cells
python .agents/skills/model-verification-harness/scripts/verify_model.py \
  --base-model <path-or-hf-id> --gates all --profile full --gpus 1,2 --report report.md

# Brand-new arch with no checkpoint: shrink a 2-layer model from its config
python .agents/skills/model-verification-harness/scripts/verify_model.py \
  --base-model <hf-id-or-config-dir> --tiny-strategy shrink --gates all --gpus 1
```

Useful flags: `--tiny-strategy {path,checkpoint,shrink}`; `--gpus 1,2` pins which
CUDA devices the run gates use (the harness prints a GPU inventory when unset);
`--profile full` adds the quant + RL config-resolution cells (and G1 auto-bisect);
`--features sample_packing,pack_parity,…` marks G2 intents as expected / enables the
G5 packed-vs-unpacked loss-parity check; `--snapshot-dir DIR` enables the G4 masking
snapshot diff; `--emit-test` scaffolds the G8 regression test; `--trust-remote-code`
opts into remote model code (off by default); `--repo-root` points at the axolotl
checkout to introspect. See `--help` for all.

GPU gates retry a transient Triton/CUDA compile failure and classify a kernel that
can't compile in this env as **could-not-run** (not a false finding) — distinct from
a kernel that runs but changes the math (a real shadow finding).

Exit code: **`0`** all selected gates clean · **`1`** findings · **`2`** could not
run (model/env unloadable) — same contract as the liger audit, so CI can gate on
it.

> The harness must be invoked as the `verify_model.py` script (it has the
> `__main__` guard datasets' multiprocess save needs). Run it directly, not via
> `python -c`.

## Stage 4: Interpret

Summarize the report and call out the cells that matter:

| Result | Meaning | Action |
|---|---|---|
| G1 **warned-no-op** cell | a flag was accepted but warns / silently does nothing — the wrong-but-green class | confirm the flag is actually wanted for this arch; it is not doing what the name implies |
| G2 **generic-fallback** ⚠️ | a hook runs via a generic path that may be wrong for this arch (liger generic FLCE, CCE generic patch, default embed names) | review that the generic path is correct, or add a dedicated branch (use the model-add anatomy in the plan to find the file) |
| G2 **missing** ❌ | an expected hook for a detected feature is absent | wire it (multipack list, patch_manager branch, MoE arch block, …) |
| G4 finding | prompt tokens trained or assistant span masked — a silent chat-template/masking bug | fix the template / `train_on_*` setting |
| G7 **SHADOW SUSPECT** | a kernel's step-0 loss diverged from baseline beyond `rtol` | the kernel changed the math (wrong dispatch) — the highest-value finding; investigate the kernel routing for this type |

For a green smoke run, offer to scaffold the regression test (`--emit-test`) so the
model lands with a test that keeps it verified.

## Notes

- `manifest.json` records the seed, resolved config flags, package versions, git
  sha, and GPU, so a maintainer can re-run identically.
- The static gates introspect the **repo source** (AST/registry), the run gates
  import the **installed** axolotl — keep both the same tree (`--repo-root` +
  the env's axolotl) so verdicts agree.
- It reports; it does not edit `src/`. Fixes are a human decision.
