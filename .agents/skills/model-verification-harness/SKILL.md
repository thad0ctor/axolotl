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

Before running, confirm with the user:

1. **Model** — the `--base-model` (a path or HF id). A user-supplied base is
   **required** (there is no shrink-from-config default); for a brand-new arch
   with no checkpoint, point it at any small instance of that arch.
2. **Environment** — which `python` / venv / uv / conda to run in (the kernels are
   env-specific — run inside the training env).
3. **Profile** — `smoke` (≈5 maximal-compatible composites, every flag exercised
   once), `full` (smoke + auto-bisect any failure to the minimal flag set), or
   `multigpu` (adds the parallelism flags).
4. **Gates** — default all; drop G6/G7 if no GPU.
5. **Report** — whether to attach the markdown report to a PR summary.

## Stage 3: Run

```bash
# CPU-only smoke (CI-gateable): config + integration + preprocess/masking/packing + coverage
python .agents/skills/model-verification-harness/scripts/verify_model.py \
  --base-model <path-or-hf-id> --gates G1,G2,G3,G4,G5,G8 \
  --report report.md --manifest manifest.json

# Full incl. GPU gates (loss + numerical consistency), with bisection
python .agents/skills/model-verification-harness/scripts/verify_model.py \
  --base-model <path-or-hf-id> --gates all --profile full --report report.md
```

Useful flags: `--features sample_packing,fused_attn_kernel,…` marks those as
**expected** in G2 (else advisory); `--snapshot-dir DIR` enables the G4 masking
snapshot diff (first run captures, later runs compare); `--emit-test` scaffolds
the G8 regression test; `--repo-root` points at the axolotl checkout to
introspect (default: auto-locate). See `--help` for all.

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
