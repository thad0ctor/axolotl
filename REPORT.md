# SparseLoRA model-agnostic + add-a-model skill — Report

Branch: `feat/sparselora-model-agnostic` (worktree `axolotl-sparselora-models`).
Two local commits, nothing pushed.

- `83f361d80` feat(sparselora): model-agnostic sparse wiring (Qwen2/Qwen3/Mistral)
- `b9ce5a46a` feat(skills): add sparselora-add-model agent skill

## TL;DR

SparseLoRA was Llama-only because the sparse *forward wiring* (MLP/attention) was hardcoded to Llama. It is now **model-agnostic**: any SwiGLU-MLP + standard-attention transformer auto-detects and registers the right sparse modules at apply time, with no per-arch user config. Verified end-to-end on **Qwen2, Qwen3, and Mistral** (real models, real sparse forward/backward, finite loss). A reproducible **`sparselora-add-model`** agent skill (SKILL.md + audit script + layout test) was added per the PR #3752 framework.

## What changed

### New file: `src/axolotl/integrations/sparselora/arch_wiring.py`

Plugin-side, model-agnostic wiring. **No vendored code was edited** — every new class subclasses a vendored one and registration goes through the public `register_sparse_module` API.

- `SparseSwiGLUMLP` — the SwiGLU MLP forward (`down(silu(gate(x)) * up(x))`) is identical across families, so this reuses the vendored `SparseLlamaMLP` forward verbatim; it exists to key the registry by each arch's own MLP class.
- `SparseAttention` — generic attention that introspects the three things that actually differ per arch: **projection bias** (Qwen2 q/k/v) handled by `SparseLinearBias`; **q_norm/k_norm** on the head dim (Qwen3) applied between `.view(hidden_shape)` and the RoPE `.transpose(1, 2)`, exactly as the real Qwen3 forward does; **sliding_window** (Qwen2/Qwen3/Mistral) forwarded to the attention interface (`None` for Llama → identical behavior).
- `SparseLinearBias(SparseLinear)` — the vendored `SparseLinear` **drops bias entirely** (Llama is bias-free, so it never mattered). This bias-aware drop-in applies the base layer's trained bias — sliced to selected channels in output-sparse modes, whole in input-sparse modes. With `bias is None` it is bit-identical to the vendored class, so the existing logit-exact Llama path is preserved.
- `register_arch_wiring(model)` — introspects every module, maps SwiGLU MLPs → `SparseSwiGLUMLP` and standard attention → `SparseAttention`, remaps `nn.Linear` → `SparseLinearBias` (safe superset), leaves pre-registered classes (Llama) untouched. Idempotent.
- `unsupported_reason(module)` — returns a clear message for semantics the generic path can't reproduce (currently Gemma2/Gemma3 attention-logit softcapping), so the plugin refuses instead of silently corrupting.

### `plugin.py`

- `_validate` now calls `register_arch_wiring(model)` first, then refuses any target whose `unsupported_reason` is non-None, then keeps the existing attention-only-LoRA / predictor-rank / orphan-LoRA / sample_packing / 8-bit / FSDP guards unchanged.
- `_apply_compile_boundaries` additionally marks `SparseAttention.forward` and `SparseLinearBias.forward` as `torch.compiler.disable` regions (the MLP/linear inherit the already-disabled vendored functions).

### Docs

`docs/agents/sparselora.md` and the plugin `README.md` updated: architecture support table, removed the "Llama only" limitation, added the source-map row for `arch_wiring.py`, and a pointer to the new skill.

## Architectures that now work (with evidence)

CPU unit tests (`tests/integrations/sparselora/test_sparselora_arch.py`, parametrized over qwen2/qwen3/mistral): target discovery (2 MLP + 2 attn), factor key/shape layout, auto-registration maps the arch's MLP/attn classes to the generic sparse classes, plugin validation accepts attention-only LoRA, and `SparseLinearBias` bias correctness (dense + output-sparse + None-bias parity).

GPU apply tests (idx 5, RTX 3090): for each of Qwen2/Qwen3/Mistral, apply SparseLoRA at 0.5 sparsity, assert `SparseAttention` + `SparseSwiGLUMLP` are installed, run a forward+backward, assert finite loss and non-zero LoRA grad norm. **All pass.**

End-to-end `axolotl train` (8 steps, `mhenrichsen/alpaca_2k_test`, tiny cached checkpoints, `--launcher python`, GPU 5):

| Arch (model) | Auto-registered | Calibrated sparsity | train_loss | Notes |
|---|---|---|---|---|
| Qwen2 (`tiny-qwen2-129m`) | `Qwen2Attention`, `Qwen2MLP` | 0/16 (calibration kept it dense at loss_budget 0.1) | 3.366 | validates calibrate→apply→train flow; q/k/v **bias** carried through |
| Qwen3 (`tiny-qwen3-129m`) | `Qwen3Attention`, `Qwen3MLP` | 12/16, mean 0.469 | 3.416 | exercises the **q_norm/k_norm** sparse path under real sparsity |
| Mistral (`tiny-mistral-25m`) | `MistralAttention`, `MistralMLP` | 14/16, mean 0.512 | 3.322 | exercises the **sliding_window** path under real sparsity |

(The Qwen2 e2e run left the schedule dense only because the tiny model + strict `loss_budget` produced no within-budget layer; the GPU unit test covers Qwen2's sparse forward/backward at 0.5 directly.)

Full suite: `58 passed` (sparselora unit/plugin/apply/arch/cache + validation + skill layout) on GPU 5. `44 passed, 7 skipped` on CPU. pre-commit (pinned ruff/ruff-format/mypy/bandit) clean on all changed non-`_vendor` files.

## What is still Llama-specific or unsupported, and why

- **Llama** keeps its pre-registered vendored `SparseLlamaMLP`/`SparseLlamaAttention` (not the generic classes). Deliberate: it preserves the existing logit-exact, already-validated Llama path byte-for-byte. The generic classes would also work for Llama, but there was no reason to disturb the tested path.
- **Gemma2 / Gemma3 — intentionally refused.** Their attention applies **attention-logit softcapping**, which the real forward passes as an explicit `softcap=` kwarg, not via `self`. The generic `SparseAttention` does not forward it, so sparsifying Gemma attention would silently drop the term. Rather than ship a wrong result, validation refuses with a clear message (`unsupported_reason`), with a unit test (`test_gemma2_softcapping_rejected`) and an audit-script demonstration (exit 1). Gemma's **MLP** is a normal SwiGLU and would sparsify fine; only its attention is blocked. Adding Gemma support is a clean follow-up (subclass `SparseAttention`, forward `softcap`), documented in the skill's Stage 3.
- **Fused-QKV / MoE / GLU-variant** architectures (single packed `qkv_proj`, MoE experts, non-SiLU gated activations) are **not** auto-supported: the SVD predictor and `SparseLinear` machinery assume separate `q_proj`/`k_proj`/`v_proj` Linears and a SiLU SwiGLU MLP. The audit script reports these as `UNMAPPED` / missing blocks (exit 1). The skill documents what to subclass. Out of the stated scope (Qwen2/Qwen3/Mistral) and an honestly non-trivial extension.
- **Distributed / packing / 8-bit** limitations are unchanged (FSDP, ZeRO-3, `sample_packing`, `load_in_8bit` still rejected). Orthogonal to architecture.

## The skill: `sparselora-add-model`

`.agents/skills/sparselora-add-model/` (canonical), exposed to Claude Code via the `.claude/skills` → `.agents/skills` symlink; read natively by Codex/Gemini.

- **`SKILL.md`** — portable agentskills.io frontmatter (name matches dir, description 767 chars). Step-by-step: audit → interpret → add custom wiring (what to subclass; how to handle bias / qk-norm / scaling / softcap / fused QKV; how to register; how to refuse cleanly) → validate. Includes trigger phrasings.
- **`scripts/audit_sparselora_model.py`** — given `--arch <name>` (synthetic tiny model, no download) or `--base-model <hf-id>` (built from config, no checkpoint, capped layers), checks gate/up/down + q/k/v/o presence, reports which sparse module each MLP/attn class maps to, refuses unsupported semantics, and runs a sparse apply+forward/backward smoke test on GPU (structural-only on CPU, since the vendored forwards need liger/Triton). Exit `0`/`1`/`2` like the liger audit. Verified: qwen2/qwen3/mistral/llama → SUPPORTED (exit 0), gemma2 → NOT SUPPORTED (exit 1, deduped refusal), CPU → structural SUPPORTED, real `--base-model SmolLM2-135M` → SUPPORTED.
- **Layout**: `.agents/skills/README.md`, an AGENTS.md skill table row, and `tests/test_agent_skill_layout.py` (adapted from PR #3752) — `4 passed`.

### Usage

```bash
# is my architecture supported?
python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py --arch qwen3
python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py --base-model Qwen/Qwen2.5-0.5B
```

In an assistant: "make SparseLoRA work on Mistral", "does SparseLoRA support Phi-3?", "SparseLoRA says wiring not available", or `/sparselora-add-model`.

## Honest assessment

- The core claim — **MLP sparsity is universal, attention differs in a small, introspectable set of ways** — held up. Qwen2/Qwen3/Mistral needed *zero* per-arch code beyond the one generic attention class + the bias-aware linear, and all three train with finite loss under real sparsity.
- The **bias bug** in the vendored `SparseLinear` (it silently drops bias) was a latent correctness issue that only surfaced once a bias-ful arch (Qwen2) was wired up; fixing it via a subclass keeps the Llama path bit-identical.
- I did **not** add Gemma/fused-QKV/MoE support — out of scope, and doing it half-way would be worse than refusing cleanly, which is what the plugin now does. The skill makes the follow-up reproducible.
- Sparse-vs-dense *quality* was not measured: the e2e runs are 8-step smoke runs on tiny checkpoints; loss is finite but this is not a convergence/accuracy claim. The guarantees here are structural + finite loss/grad, not accuracy parity with dense LoRA on a real benchmark (that needs a multi-hour run on a real model + dataset).
