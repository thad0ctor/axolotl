---
name: sparselora-add-model
description: "Adds or verifies a model architecture's SparseLoRA sparse wiring, including text blocks and supported vision tower blocks. Use when asked to launch the sparselora add-model skill, to make SparseLoRA work on a new architecture (Qwen, Mistral, Phi, Gemma, MoE, VLM vision towers, ...), when SparseLoRA errors with 'wiring not available' or 'cannot sparsify this model', when a model trains under SparseLoRA but loss looks wrong, or when reviewing changes to src/axolotl/integrations/sparselora/arch_wiring.py. Introspects MLP/attention classes, checks supported SwiGLU, fused gate_up, non-gated ViT MLP, standard attention, fused qkv, and known vision attention layouts, flags semantics the generic path cannot reproduce, and runs a smoke apply+forward."
---

# SparseLoRA: add a model architecture

SparseLoRA's MLP/attention sparsity is **model-agnostic** for the supported
projection layouts in `arch_wiring.py`. `register_arch_wiring(model)` introspects
the loaded model at apply time and maps supported MLP classes to
`SparseSwiGLUMLP` or `SparseNonGatedMLP`, and supported attention classes to
`SparseAttention`, `SparseFusedQKVAttention`, `SparseQwen3VLVisionAttention`,
`SparseGemma4VisionAttention`, or `SparseNoRoPEAttention`; Llama keeps its
pre-registered vendored classes. Most new text models and many vision towers
therefore need **zero code** - they just work and auto-register. This skill
verifies that, and guides the rarer case where custom wiring is still needed.

Vision tower support is not "FFT-only". SparseLoRA can target LoRA/QLoRA
adapters on supported vision attention and MLP modules when the config exposes
those modules and the projection names are discoverable. Current generic vision
coverage includes Qwen3-VL fused `qkv`/`proj`, Gemma4 multidimensional RoPE
attention with wrapped linears, Gemma3/SigLIP and InternVL no-RoPE attention,
and non-gated ViT MLPs using `fc1`/`fc2` or Qwen3-VL
`linear_fc1`/`linear_fc2`.

Source: `src/axolotl/integrations/sparselora/arch_wiring.py`. Plugin validation
that consumes it: `plugin.py::_validate`.

## When to use

- "make SparseLoRA work on `<arch>`" / "does SparseLoRA support `<model>`?"
- SparseLoRA raises `SparseLoRA wiring not available...` or `cannot sparsify this model...`
- A SparseLoRA run trains but loss/quality looks wrong on a non-Llama model
- Reviewing a PR that touches `integrations/sparselora/arch_wiring.py`

## Stage 1: Audit the architecture

Run the bundled audit. It builds a small model of the architecture (real
hidden/head dims), attaches LoRA to every discovered sparsifiable projection,
reports which sparse module each MLP/attention class maps to, refuses unsupported
semantics, and smoke-tests sparse apply. Text model smokes run
forward/backward on CUDA and stay structural-only on CPU. Synthetic vision
tower smokes run an apply + forward on CPU so vision support can be checked
without downloading a checkpoint:

```bash
# synthetic tiny model, no download (llama|qwen2|qwen3|mistral|gemma2)
PYTHONPATH=src python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py --arch qwen3

# synthetic tiny vision tower, no download
# (qwen3vl|gemma4|gemma3|siglip|internvl)
PYTHONPATH=src python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py --vision-arch qwen3vl

# any HuggingFace id - built from its config (random weights, no checkpoint),
# capped to --layers layers; run on a GPU for the sparse smoke test
PYTHONPATH=src python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py \
  --base-model Qwen/Qwen2.5-0.5B
```

Exit `0` supported (smoke passed, or structural-only on CPU), `1` not supported
(refused / missing projections / smoke failure), `2` could not audit. So it can
gate CI.

For VLM repos, `--base-model` still uses `AutoModelForCausalLM` and is best for
text/decoder architectures. Use `--vision-arch` for the known synthetic tower
families, then add a real local config/module smoke when the model's vision
tower differs from those families.

## Stage 2: Interpret

| Output | Meaning | Action |
|--------|---------|--------|
| `[1]`/`[2]` list classes -> `SparseSwiGLUMLP`, `SparseNonGatedMLP`, `SparseAttention`, `SparseFusedQKVAttention`, a vision-specific sparse attention, or `SparseLlama*`, and `[4]` smoke **PASSED** | The model is already supported by auto-registration. | Nothing to do - just enable the plugin. |
| `[3] Refused` with a reason | The generic wiring can't reproduce a semantic (e.g. a gated MLP activation other than SiLU/`gelu_tanh`). | Add custom wiring (Stage 3) or accept it's unsupported. The plugin will refuse with the same message. |
| `[1]`/`[2]` show `UNMAPPED`, or `[1]`/`[2]` missing a block | The block is not one of the supported MLP/attention layouts, or its projection aliases were not discovered. | Add aliases or custom wiring (Stage 3). |
| `[4]` smoke **FAILED** | apply or forward raised, or loss/grad was non-finite/zero. | Read the traceback note; usually a shape, RoPE, normalization, or projection wrapper the sparse forward does not handle -> custom wiring (Stage 3). |

## Stage 3: Add custom wiring (only if Stage 2 says so)

The generic path already covers, automatically: **projection bias** (Qwen2 q/k/v)
via `SparseLinearBias`, **q_norm/k_norm** on the head dim (Qwen3/Gemma3/InternVL),
**sliding_window** (Qwen2/Qwen3/Mistral/Gemma), **decoupled head_dim** (Qwen3
q_out != hidden), **`query_pre_attn_scalar` scaling** and **attention-logit
softcapping** (Gemma2), **gated MLP activation** for SiLU and `gelu_tanh`
(Gemma), **fused `gate_up_proj` MLPs**, **non-gated ViT MLPs**, standard
`q_proj`/`k_proj`/`v_proj` attention, fused `qkv`/`qkv_proj` attention when the
split sizes are inferable, and output aliases including `o_proj`, `out_proj`,
`proj`, and `projection_layer`. If the new architecture needs something else,
subclass and register it.

1. **Find the real module.** Locate the architecture's MLP and attention classes
   in `transformers/models/<arch>/modeling_<arch>.py`. Read their `forward` and
   diff it against the closest sparse class in `arch_wiring.py`. Note any term
   the sparse forward would drop: extra scaling, unusual packed projection
   splits, a different RoPE, per-head bias, normalization placement, projector
   or merger modules, or a gate activation other than SiLU/`gelu_tanh`.

2. **MLP gate activation.** `SparseSwiGLUMLP` reads the base MLP's `act_fn` and
   uses the fused liger SiLU kernel for SiLU or `gelu_tanh` for Gemma - for both
   the predictor's channel *selection* (`_GELUFFNPredictor`) and the *compute*.
   A **new** gated activation needs (a) a new branch in `gate_kind` + a predictor
   variant scoring with that activation, and (b) the matching `_block` compute;
   the worked example is the `gelu_tanh` path added for Gemma. A structurally
   different MLP such as MoE needs a fuller `SparseModule` subclass.

3. **Non-gated vision MLP.** `SparseNonGatedMLP` handles
   `act(fc1_or_linear_fc1(x)) -> fc2_or_linear_fc2`. If the tower uses different
   names, add those aliases to `non_gated_mlp_projection_names(...)`,
   `discover_target_modules(...)`, and factor computation before adding a new
   sparse class. If the MLP has residuals, norms, gates, or experts inside the
   block, treat it as custom wiring instead.

4. **Attention: subclass `SparseAttention` when semantics differ.** Reuse `_proj_qkv` / `_proj_o`
   (they apply the predictor token-mask machinery to q/k/v/o) and the
   sparse/dense split; override only `forward` to add the missing term. Preserve
   the arch's semantics:
   - **bias** -> ensure `register_arch_wiring` maps `nn.Linear` to
     `SparseLinearBias` (it already does); nothing else needed.
   - **qk-norm** -> apply between `.view(hidden_shape)` and `.transpose(1, 2)`,
     as `SparseAttention` does for `q_norm`/`k_norm`.
   - **scaling** -> pass the right `scaling=` (it's read from `self.scaling`,
     copied off the base module, so a custom `query_pre_attn_scalar` already
     flows through).
   - **softcap / extra kwargs** -> forward them to the attention interface call.
   - **fused QKV** -> use `SparseFusedQKVAttention` when q/k/v split sizes can
     be inferred from attributes; otherwise add a custom sparse attention that
     splits the packed projection exactly like the base module.

5. **Vision tower attention.** Inspect vision attention and MLP separately; many
   VLMs use supported text-style MLPs but custom image attention. Preserve
   tensor rank, RoPE dimensionality, `position_embeddings`, `position_ids`,
   `cu_seqlens`, q/k/v norms, and output projection wrappers. Existing patterns
   cover Qwen3-VL `qkv`/`proj`, Gemma4 `q_proj`/`k_proj`/`v_proj`/`out_proj`
   with wrapped linears and multidimensional RoPE, and no-RoPE SigLIP/Gemma3 or
   InternVL attention. If the tower uses a patch merger/projector, keep that out
   of SparseLoRA unless it is explicitly one of the sparsifiable modules.

6. **Register it.** Prefer extending `register_arch_wiring` so it stays
   automatic: detect the class and `register_sparse_module(TheAttention,
   SparseTheAttention)`. For a one-off, call `register_sparse_module` directly
   before `apply_sparselora`. If a semantic genuinely can't be reproduced, add a
   branch to `unsupported_reason(...)` so the plugin refuses with a clear message
   instead of silently dropping the term (this is how Gemma softcapping is
   handled).

7. **Guard the predictor-rank / projection-shape contract.** Each selected
   projection's smaller dim must be >= `predictor_rank`; GQA (kv heads < q
   heads) and fused qkv split sizes are detected automatically when the module
   exposes enough shape metadata.

## Stage 4: Validate

```bash
# structural + sparse smoke for the new arch (GPU for the real predictor path)
PYTHONPATH=src python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py --arch <arch>

# synthetic vision tower apply + forward smoke
PYTHONPATH=src python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py --vision-arch <qwen3vl|gemma4|gemma3|siglip|internvl>

# unit + apply tests (GPU exercises the liger/Triton sparse forward/backward)
PYTHONPATH=src pytest tests/integrations/sparselora -q
```

Add the new arch to the parametrized `ARCHES` list in
`tests/integrations/sparselora/test_sparselora_arch.py` (CPU discovery/factors/
registration/validation + GPU apply+forward). For vision towers, add focused
tiny-module tests for the attention and MLP classes plus at least one real local
config/module smoke when possible. If you added a refusal, add a rejection test
like `test_gemma2_softcapping_rejected`. Then run a few real steps with
`axolotl train` on a small checkpoint of the architecture and confirm
`registered generic sparse wiring for ...`, a non-empty calibrated schedule, and
finite loss. For VLMs, also confirm the calibrated target list includes the
intended vision modules when the config is meant to train image layers.

## Notes

- The audit reflects the **installed** transformers; a future version that
  changes an attention forward can move an arch from supported to refused - run
  it whenever you bump transformers.
- Vision forward signatures vary by transformers release. If a synthetic audit
  starts failing after a dependency bump, inspect the upstream module signature
  before assuming SparseLoRA wiring regressed.
- It does not edit any source; adding wiring is a human decision. Keep the
  vendored `_vendor/` tree untouched - subclass its classes from
  `arch_wiring.py` and register through `register_sparse_module`.
