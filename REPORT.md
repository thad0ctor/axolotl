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

---

# Pass 2 (overnight continuation)

Three more local commits on `feat/sparselora-model-agnostic` (still nothing pushed):

- `f4bd326e4` feat(sparselora): fix decoupled head_dim predictor; broaden arch coverage; Gemma verdict
- (+ this REPORT.md update)

## 1. Convergence parity (real Qwen3-0.6B, 80 steps, seed 42, same data)

Ran dense LoRA vs sparse-LoRA on the **real** Qwen3-0.6B (`Models-2/Qwen3/Qwen3-0.6B`), attention-only LoRA, GPU idx 5. This exercises the real q_norm/k_norm + decoupled-head_dim path.

**Run A — sparsity from step 0 (`start_step=0.0`, target 0.5, 42/56 modules sparse, mean 0.491):**

| phase | dense | sparse |
|---|---|---|
| steps 1–5 | 1.970 | 4.923 |
| steps 11–20 | 1.395 | 2.897 |
| steps 31–40 | 1.432 | 2.544 |
| steps 51–60 | 1.463 | 2.119 |
| steps 71–80 | 1.353 | 2.175 |
| min / NaN | 0.737 / no | 1.175 / **no** |

Sparse decreases monotonically (4.92 → 2.90 → 2.54 → 2.12), **no NaN, no divergence** — it tracks the dense downward trend at an offset. The offset is the cost of applying ~0.49 base-path sparsity from step 1 on an untrained predictor with no warm-up.

**Run B — recommended pattern (`start_step=0.1` warm-up, target 0.3):**

| phase | dense | sparse |
|---|---|---|
| steps 1–8 (dense warm-up) | **1.832** | **1.834** |
| steps 9–20 (sparsity on) | 1.386 | 3.045 |
| steps 31–40 | 1.432 | 2.448 |
| steps 71–80 | 1.353 | 1.985 |
| min / NaN | 0.737 / no | 1.043 / **no** |

The headline result: **during the dense warm-up the sparse run tracks the dense baseline to within 0.002 (1.834 vs 1.832)** — proving the plugin does not perturb training while sparsity is inactive (the wiring is transparent when off). When sparsity engages at step 9 the loss steps up (base path loses ~30% of channels) and then converges downward again, tracking dense without diverging. This matches the expected contextual-sparsity behavior; full recovery to the dense curve needs a full-length run (the paper's regime), not 80 steps.

**Verdict:** sparse converges and tracks dense (no divergence/NaN); the dense-phase identity is a strong correctness signal that the model-agnostic wiring is faithful.

## 2. Real bug found + fixed: decoupled `head_dim` predictor crash

The first sparse Qwen3-0.6B run **crashed** at predictor load: `size mismatch for q2: [2048,8] vs [1024,8]`. Root cause: the vendored `GQAAttentionPredictor` (and `AttentionPredictor`) size their `q2`/`w2` buffers at `hidden_size`, assuming `num_heads * head_dim == hidden_size`. Qwen3 **decouples** `head_dim` (128) from `hidden/num_heads`, so `q_proj` out (2048) ≠ hidden (1024). My tiny tests had used `head_dim*heads == hidden`, so they missed it.

Fix (plugin-side, no vendored edit): `arch_wiring._create_attn_predictor` + `_GQAAttentionPredictorHD` / `_AttentionPredictorHD` size the q/w buffers at the true projection dims (read from the factor tensors). `SparseAttention.__init__` uses this creator. For models where q_out == hidden it is identical to the vendored path. Regression-tested on CPU (`test_decoupled_head_dim_attn_predictor_loads`), the tiny Qwen3 fixture is now decoupled (`head_dim=32`, q_out=128≠hidden=64) so the GPU apply test exercises it, and the real Qwen3-0.6B e2e + audit-script smoke both pass.

This is the most important fix of the night — without it, **no real Qwen3 model** could use SparseLoRA.

## 3. Gemma verdict (precise)

Investigated whether the generic path can support Gemma. Two separate issues, and the binding one is **not** what we first thought:

- **Attention-logit softcapping (Gemma2 only):** *reproducible.* The attention interface's `eager_attention_forward` applies `softcap` when passed it. `SparseAttention` now forwards `attn_logit_softcapping` and resolves the eager fallback from the base module's **own** architecture module (Gemma2's softcap-aware eager), instead of Llama's. Proven by `test_sparseattention_applies_softcap`: attention-only sparsify of a tiny Gemma2 at sparsity 0 is **logit-exact** vs the unmodified reference (max diff 0.0). So softcap is **not** the blocker.
- **GELU-gated MLP (Gemma2 *and* Gemma3):** *not reproducible.* Gemma's SwiGLU uses `gelu_pytorch_tanh`, but the vendored `FFNPredictor` and `SparseLlamaMLP` hardcode liger `silu_mul` — for both channel *selection* and *compute*. Sparsifying (or even wrapping) a Gemma MLP would silently swap GELU→SiLU. Gemma3 **removed** softcapping (default `None`), confirming the MLP gate is the real, shared blocker.

**Decision:** keep the refusal, but make it precise and correct — `unsupported_reason` now rejects any **non-SiLU gated MLP** (detected by a numeric `is_silu_gated` probe against `F.silu`, since the activation class varies: `SiLUActivation` vs `GELUTanh`). This is the honest binding constraint and also protects any future non-SiLU SwiGLU family. Full Gemma support would need a GELU-aware sparse MLP + predictor (a scoped follow-up); shipping a silu-on-gelu approximation would be wrong, so we don't.

## 4. Broadened auto-detection coverage (registry test added)

| Family | MLP | Attention | Outcome |
|---|---|---|---|
| StableLM | SiLU SwiGLU | q/k/v/o | **supported** (auto-registered) |
| Cohere | SiLU SwiGLU | q/k/v/o | **supported** (auto-registered) |
| Gemma3-text | `gelu_pytorch_tanh` | q/k/v/o (no softcap) | MLP **refused**, attention registers |
| Phi3 | fused `gate_up_proj` | fused `qkv_proj` | **not registered** (neither standard); LoRA on `qkv_proj` is then refused by the existing orphan-LoRA guard |
| Qwen2-MoE | `Qwen2MoeSparseMoeBlock` (router + experts) | q/k/v/o | attention registers; the MoE block is not a SwiGLU MLP so it is not sparsified as one. Individual experts *are* SiLU SwiGLU and would be discovered, but MoE routing semantics for calibration are **unvalidated** — treat as out of scope. |

`test_registry_covers_silu_swiglu_families`, `test_phi3_fused_projections_not_registered`, and `test_gemma3_text_mlp_refused_attention_supported` lock these in.

## 5. Branch divergence from `feat/sparselora-plugin` (do NOT merge/rebase — note only)

`feat/sparselora-plugin` has advanced by one commit past my merge-base (`a32fb12`): **`e0eaa3e5d`** "fix(sparselora): address CodeRabbit review (PR #58)", touching `plugin.py`, `cache.py`, `calibration.py`, `pyproject.toml`, `README.md`, `test_sparselora_plugin.py`.

Expected conflicts at eventual merge — **low risk**:

- **`plugin.py`** — `e0eaa3e5d` adds a new `resolve_layer_sparsity()` function (above `_apply_compile_boundaries`) and rewrites the `method == "none"` branch in `post_trainer_create`. My changes are in a *different* region: `_apply_compile_boundaries` body (added `arch_wiring` import + two `disable(...)` lines) and the architecture-support block of `_validate`. Different hunks → should auto-merge; worst case a trivial adjacency resolve. **Semantically independent** (arch registration vs method=none key resolution compose cleanly).
- **`README.md`** — both edit it; `e0eaa3e5d` only re-tags a fenced code block as `text` (markdownlint), I edited the intro + the architecture limitation bullet. Different lines → likely clean, possibly a one-line manual resolve.
- **`cache.py`, `calibration.py`, `pyproject.toml`, `test_sparselora_plugin.py`** — touched only by `e0eaa3e5d`, untouched by me → apply cleanly. (Note: `e0eaa3e5d` declares the `safetensors` dependency my `arch_wiring` transitively relies on, so the merge is strictly beneficial.)

No reconciliation needed now; when the branches merge, resolve the two co-touched files (both straightforward) and the result composes.

## Test/quality status (end of pass 2)

- **62 passed** on GPU idx 5 (`tests/integrations/sparselora` + `tests/test_agent_skill_layout.py`); **55 passed, 7 skipped** on CPU.
- pre-commit (pinned ruff/ruff-format/mypy/bandit) clean on all changed non-`_vendor` files.
- e2e on real Qwen3-0.6B (dense + 2 sparse configs) and audit-script smoke on real Qwen3-0.6B all green; `_vendor/` still untouched.
