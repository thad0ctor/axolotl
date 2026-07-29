---
name: sparselora-add-model
description: "Adds or verifies a model architecture's SparseLoRA sparse wiring. Use when asked to launch the sparselora add-model skill, to make SparseLoRA work on a new architecture (Qwen, Mistral, Phi, Gemma, MoE, ...), when SparseLoRA errors with 'wiring not available' or 'cannot sparsify this model', when a model trains under SparseLoRA but loss looks wrong, or when reviewing changes to src/axolotl/integrations/sparselora/arch_wiring.py. Introspects a model's MLP/attention classes, checks the SwiGLU gate/up/down + standard q/k/v/o projections the generic wiring needs, flags semantics the generic path can't reproduce (e.g. a gated MLP activation other than SiLU/gelu_tanh, or fused qkv), and runs a smoke apply+forward. Covers what to subclass and how to register custom wiring for non-standard attention."
---

# SparseLoRA: add a model architecture

SparseLoRA's MLP/attention sparsity is **model-agnostic** for SwiGLU-MLP +
standard-attention transformers. `arch_wiring.register_arch_wiring(model)`
introspects the loaded model at apply time and maps each MLP class to
`SparseSwiGLUMLP` and each attention class to the generic `SparseAttention`;
Llama keeps its pre-registered vendored classes. Most new models therefore need
**zero code** — they just work and auto-register. This skill verifies that, and
guides the rare case (non-standard attention) that needs custom wiring.

Source: `src/axolotl/integrations/sparselora/arch_wiring.py`. Plugin validation
that consumes it: `plugin.py::_validate`.

## When to use

- "make SparseLoRA work on `<arch>`" / "does SparseLoRA support `<model>`?"
- SparseLoRA raises `SparseLoRA wiring not available...` or `cannot sparsify this model...`
- A SparseLoRA run trains but loss/quality looks wrong on a non-Llama model
- Reviewing a PR that touches `integrations/sparselora/arch_wiring.py`

## Stage 1: Audit the architecture

Run the bundled audit. It builds a small model of the architecture (real
hidden/head dims), attaches attention-only LoRA, reports which sparse module each
MLP/attention class maps to, refuses unsupported semantics, and smoke-tests
apply + a sparse forward/backward (on CUDA) or structural-only (on CPU):

```bash
# synthetic tiny model, no download (llama|qwen2|qwen3|mistral|gemma2)
PYTHONPATH=src python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py --arch qwen3

# any HuggingFace id — built from its config (random weights, no checkpoint),
# capped to --layers layers; run on a GPU for the sparse smoke test
PYTHONPATH=src python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py \
  --base-model Qwen/Qwen2.5-0.5B
```

Exit `0` supported (smoke passed, or structural-only on CPU), `1` not supported
(refused / missing projections / smoke failure), `2` could not audit. So it can
gate CI.

## Stage 2: Interpret

| Output | Meaning | Action |
|--------|---------|--------|
| `[1]`/`[2]` list classes → `SparseSwiGLUMLP` / `SparseAttention` (or `SparseLlama*` for Llama) and `[4]` smoke **PASSED** | The model is already supported by auto-registration. | Nothing to do — just enable the plugin. |
| `[3] Refused` with a reason | The generic wiring can't reproduce a semantic (e.g. a gated MLP activation other than SiLU/`gelu_tanh`). | Add custom wiring (Stage 3) or accept it's unsupported. The plugin will refuse with the same message. |
| `[1]`/`[2]` show `UNMAPPED`, or `[1]`/`[2]` missing a block | The block isn't a SwiGLU MLP / standard attention (e.g. fused QKV, GLU variant, MoE router). | Add custom wiring (Stage 3). |
| `[4]` smoke **FAILED** | apply or forward raised, or loss/grad was non-finite/zero. | Read the traceback note; usually a shape/attribute the generic forward doesn't handle → custom wiring (Stage 3). |

## Stage 3: Add custom wiring (only if Stage 2 says so)

The generic path already covers, automatically: **projection bias** (Qwen2 q/k/v)
via `SparseLinearBias`, **q_norm/k_norm** on the head dim (Qwen3/Gemma3),
**sliding_window** (Qwen2/Qwen3/Mistral/Gemma), **decoupled head_dim** (Qwen3
q_out ≠ hidden), **`query_pre_attn_scalar` scaling** and **attention-logit
softcapping** (Gemma2), and the **gated MLP activation** for SiLU and `gelu_tanh`
(Gemma). If the new architecture needs something else, subclass and register it.

1. **Find the real module.** Locate the architecture's MLP and attention classes
   in `transformers/models/<arch>/modeling_<arch>.py`. Read their `forward` and
   diff it against `SparseAttention.forward` / `SparseSwiGLUMLP` in
   `arch_wiring.py`. Note any term the generic forward drops: extra scaling,
   fused/packed QKV, a different RoPE, per-head bias, normalization placement, or
   a gate activation other than SiLU/`gelu_tanh`.

2. **MLP gate activation.** `SparseSwiGLUMLP` reads the base MLP's `act_fn` and
   uses the fused liger SiLU kernel for SiLU or `gelu_tanh` for Gemma — for both
   the predictor's channel *selection* (`_GELUFFNPredictor`) and the *compute*.
   A **new** gated activation needs (a) a new branch in `gate_kind` + a predictor
   variant scoring with that activation, and (b) the matching `_block` compute;
   the worked example is the `gelu_tanh` path added for Gemma. A structurally
   different MLP (fused gate+up, MoE) needs a fuller `SparseModule` subclass.

3. **Attention: subclass `SparseAttention`.** Reuse `_proj_qkv` / `_proj_o`
   (they apply the predictor token-mask machinery to q/k/v/o) and the
   sparse/dense split; override only `forward` to add the missing term. Preserve
   the arch's semantics:
   - **bias** → ensure `register_arch_wiring` maps `nn.Linear` to
     `SparseLinearBias` (it already does); nothing else needed.
   - **qk-norm** → apply between `.view(hidden_shape)` and `.transpose(1, 2)`,
     as `SparseAttention` does for `q_norm`/`k_norm`.
   - **scaling** → pass the right `scaling=` (it's read from `self.scaling`,
     copied off the base module, so a custom `query_pre_attn_scalar` already
     flows through).
   - **softcap / extra kwargs** → forward them to the attention interface call.
   - **fused QKV** → split the packed projection into q/k/v views first; the
     predictor + `SparseLinear` machinery assumes separate `q_proj`/`k_proj`/
     `v_proj` weights, so a fused layout needs its own predictor wiring.

4. **Register it.** Prefer extending `register_arch_wiring` so it stays
   automatic: detect the class and `register_sparse_module(TheAttention,
   SparseTheAttention)`. For a one-off, call `register_sparse_module` directly
   before `apply_sparselora`. If a semantic genuinely can't be reproduced, add a
   branch to `unsupported_reason(...)` so the plugin refuses with a clear message
   instead of silently dropping the term (this is how Gemma softcapping is
   handled).

5. **Guard the predictor-rank / projection-shape contract.** q/k/v must be
   separate Linears whose smaller dim ≥ `predictor_rank`; GQA (kv heads < q
   heads) is detected automatically in `svd.create_attn_predictor`.

## Stage 4: Validate

```bash
# structural + sparse smoke for the new arch (GPU for the real predictor path)
PYTHONPATH=src python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py --arch <arch>

# unit + apply tests (GPU exercises the liger/Triton sparse forward/backward)
PYTHONPATH=src pytest tests/integrations/sparselora -q
```

Add the new arch to the parametrized `ARCHES` list in
`tests/integrations/sparselora/test_sparselora_arch.py` (CPU discovery/factors/
registration/validation + GPU apply+forward) and, if you added a refusal, a
rejection test like `test_gemma2_softcapping_rejected`. Then run a few real steps
with `axolotl train` on a small checkpoint of the architecture and confirm
`registered generic sparse wiring for ...`, a non-empty calibrated schedule, and
finite loss.

## Notes

- The audit reflects the **installed** transformers; a future version that
  changes an attention forward can move an arch from supported to refused — run
  it whenever you bump transformers.
- It does not edit any source; adding wiring is a human decision. Keep the
  vendored `_vendor/` tree untouched — subclass its classes from
  `arch_wiring.py` and register through `register_sparse_module`.
