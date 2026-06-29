---
name: model-verification-harness
description: "Verifies that a model wires into axolotl correctly and trains sanely — a reproducible smoke gate for model-support PRs. Detects model_config_type + arch features (MoE/multimodal/SSM-hybrid), then runs a tiered gate ladder: config-resolution compat matrix (liger/CCE/kernel flags via the real load_cfg), an adaptive wired-everywhere integration checklist, preprocess + chat-template masking + sample-packing checks, and (on GPU) loss-sanity + cross-variant numerical consistency that catches silent kernel-shadowing. Emits a standardized report + manifest with a 0/1/2 exit code. Use when asked to verify model support, run the model verification harness, smoke-test a new-model PR, check whether a model wires into axolotl, or debug masking/packing/kernel-compat for a model type."
---

# Model verification harness

*Verifying* a model is a repeatable, scriptable gate (the bespoke attention/RoPE/mask patch isn't). Everything runs through `scripts/verify_model.py`; run it **in the training environment** (kernels/plugins are env-specific). `--help` lists every flag.

## Gates (cheap → expensive, CPU → GPU)

| Gate | Cost | Checks |
|---|---|---|
| **G1** config | CPU | compat matrix via the real `load_cfg`; 4-way verdict per cell (rejected / normalized / **warned-no-op** / supported); bisects a reject to the minimal flag set. |
| **G2** integration | CPU | adaptive "wired-everywhere" checklist; flags **silent generic fallbacks**, not just missing hooks. |
| **G3** preprocess | CPU/GPU | prepared dataset has `input_ids`/`labels`/`attention_mask`. |
| **G4** masking | CPU | native template (tokenizer_default) first; assistant-trained / prompt-masked spans; turn-terminator trained (#3754 silent stop-token); snapshot diff; image-token masking when multimodal. |
| **G5** packing | CPU/GPU | multipack membership, len ≤ `sequence_len`, per-doc `position_ids` reset (collate-time). |
| **G6** loss | GPU | short real train → loss finite + non-diverging. |
| **G7** consistency | GPU | **flagship**: a kernel's step-0 loss must ≈ baseline within `rtol` *and* actually engage — catches silent kernel-shadowing. |
| **G8** coverage | CPU | tests referencing this type; gap list; `--emit-test` scaffolds a regression test. |

G1–G5 + G8 are CPU and CI-gateable; **G6/G7 need a GPU (local/on-demand)**. `--profile full` adds quant + RL config-resolution cells.

## Flow

1. **Discover (PR mode).** If the user names a PR, run `verify_model.py --from-pr N` (or `--from-diff FILE`) — a fast, torch-free pass that prints the new `model_config_type`(s), their `base_model`, new-vs-extends, and warnings for un-greppable registration (CCE pin-bump, native-liger table, processor-class multimodal routing, transformers bump). Confirm the model, then continue with `--base-model` + `--repo-root <PR checkout>`.
2. **Detect.** The harness resolves `model_config_type` + arch features (MoE / multimodal / SSM-hybrid / needs-patch / custom-embeds) from `config.json` and prunes irrelevant gates.
3. **Ask (a checklist — wait for answers, don't assume):**
   - **Model & `--tiny-strategy`:** `path` (default, use as-is), `checkpoint` (auto-match a tiny `axolotl-ai-co/tiny-*` by arch), or `shrink` (random-init a 2-layer model from the target's config — for a new arch with no published checkpoint).
   - **What to test:** static (G1/G2/G8), plumbing (G3/G4/G5), GPU (G6/G7), extras (`--profile full` = quant+RL, `--features pack_parity`). Distributed (FSDP2 / expert-parallel / context-parallel) only *resolves* via `--profile multigpu`; an actual multi-GPU run is a separate user-driven step.
   - **GPUs:** anything that loads a model → show the printed GPU inventory (index · name · memory) and ask which devices (`--gpus 1,2`); re-prompt if incompatible. No GPU → G6/G7 self-skip.
   - **Env** (which venv/uv/conda) and whether to attach the report to the PR.
4. **Run.**
   ```bash
   verify_model.py --base-model <id> --gates G1,G2,G3,G4,G5,G8 --report report.md   # CPU smoke
   verify_model.py --base-model <id> --gates all --profile full --gpus 1,2          # + GPU gates
   ```
   Run bare it prints the report to stdout **and** returns an exit code (`--report`/`--manifest` also write files; `--quiet` = summary only). Invoke the script directly (it has the `__main__` guard datasets' multiprocess save needs), not via `python -c`.
   Exit **`0`** clean · **`1`** findings · **`2`** can't-run-here (no GPU / kernel won't compile / model unloadable). `--on-unavailable skip` makes can't-run non-blocking (offer as a checklist choice). GPU gates retry a transient Triton/CUDA compile error and mark a non-compiling kernel could-not-run, not a false finding.
5. **Interpret.** Call out: **G2 generic-fallback ⚠️** (runs a generic path that may be wrong — review or add a dedicated branch) and **missing ❌** (wire it: multipack / patch_manager / MoE block / …); **G1 warned-no-op** (flag accepted but inert); **G4 finding** (prompt trained, assistant masked, or turn terminator masked / #3754 — a chat-template/masking bug); **G7 SHADOW SUSPECT** (kernel changed the math — highest value). On a green run, offer `--emit-test`.
   **Frame your summary as a starting point, not a verdict.** Always tell the user explicitly that this is an automated heuristic smoke gate — it surfaces *leads*, not conclusions — and that every finding (and every PASS) should be verified independently against the model's own modeling code, the PR diff, and upstream docs before they act. Never say a model "is verified / is correct / passed"; say the gate "found no flagged issues in what it checks." Name the relevant blind spots (per-arch numerical/attention correctness, transformers-version-coupled breakage, VRAM regressions, subtle tokenization offsets, distributed-only failures). A green run ≠ a guarantee.

## Notes

- `manifest.json` captures seed, resolved flags, package versions, git sha, and GPU for reproducibility.
- Static gates read the **repo source** (`--repo-root`); run gates import the **installed** axolotl — keep them the same tree so verdicts agree. The harness reports; it never edits `src/`.
