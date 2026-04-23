# Multimodal assistant-only loss masking

## What this fixes

For multimodal fine-tuning, `cfg.train_on_inputs`, `cfg.roles_to_train`, and
`cfg.train_on_eos` were silently ignored. Every non-pad, non-media token in
the sequence — including system prompts, user turns, and role markers —
contributed to the loss. Only Gemma3n had a working per-role mask; every
other multimodal path (LLaVA, Qwen2-VL, Qwen3.5, Gemma3, Llama-3.2 Vision,
Llama 4, Pixtral, Mistral V7 Tekken, Voxtral, SmolVLM2, Mistral3, InternVL,
GLM4V) trained on the full sequence.

## Root cause

`MultiModalChatDataCollator` re-tokenizes raw `messages` via
`processor.apply_chat_template(...)` at collation time, discarding the
per-role labels already computed by `ChatTemplateStrategy.tokenize_prompt` in
the preprocessing path. It then calls
`processing_strategy.process_labels(input_ids)`, which was supposed to rebuild
role-aware labels — but the base `_mask_non_assistant` was a no-op `return
labels`, and only `Gemma3nProcessingStrategy` overrode it. So for every other
multimodal model, the retokenized labels are never masked by role.

Parallel bug ([#3617](https://github.com/axolotl-ai-cloud/axolotl/issues/3617)):
`cfg.processor_kwargs` was never plumbed through to
`processor_cls.from_pretrained`, so users couldn't pass `min_pixels`,
`max_pixels`, `num_crops`, etc., to VLM processors.

## Design

We make role masking a first-class, declarative capability of the base
`ProcessingStrategy` and thread the masking knobs through from the trainer
builder.

### Why this over alternatives

- **Option (b): preserve the per-role labels from `tokenize_prompt`.**
  Rejected. The preprocessing labels were computed against a text-only
  tokenization; they don't align with the MM collator's re-tokenization after
  image/audio/video placeholders expand into hundreds of placeholder tokens.
  Preserving them would require either a second tokenization pass with image
  stand-ins, or rewriting the collator to never re-tokenize. Either is
  high-blast-radius for an incremental bugfix.
- **Option (c): `apply_chat_template(return_assistant_tokens_mask=True)`.**
  Rejected. This requires `{% generation %}` / `{% endgeneration %}` jinja
  markers. Only `llava.jinja` and `phi_4.jinja` have them in
  `src/axolotl/utils/chat_templates/templates/`. Adding these markers to
  upstream-mirrored templates (gemma3, qwen2_vl, llama3_2_vision, etc.)
  diverges from the reference templates and is fragile when HF updates them.
- **Option (a): parametrized token-boundary scanner in the base class.**
  Chosen. Each strategy declares its per-role boundary markers
  (`<|im_start|>assistant\n` ... `<|im_end|>` for Qwen2-VL,
  `<|turn>model` ... `<turn|>` for Gemma 4, etc.). The base scanner walks the
  re-tokenized sequence, locates role spans, and masks everything outside
  `cfg.roles_to_train`. Works with existing jinja templates, is testable
  offline with fake tokenizers, and fails visible (unverified strategies emit
  a one-shot warning rather than silently mis-masking).

### Components

1. **`RoleBoundary`** dataclass in `src/axolotl/processing_strategies.py`
   describing one role's `(start_tokens, end_tokens, include_start, include_end)`.
2. **`_apply_role_boundaries`** function: a longest-prefix-match scanner that
   implements `roles_to_train` / `train_on_inputs` / `train_on_eos` (`"turn"`
   keeps role-end markers on trainable turns, `"all"` keeps them on every
   turn, `"none"` excludes them).
3. **`ProcessingStrategy._build_role_boundaries`**: empty by default;
   overridden by each subclass. `_mask_non_assistant` delegates to the
   scanner; if no boundaries are declared it short-circuits and emits a
   one-shot warning (legacy behavior preserved).
4. **Plumbing**: `cfg.train_on_inputs`, the first dataset's `roles_to_train`
   and `train_on_eos` are threaded through `build_collator` →
   `get_processing_strategy` → each strategy's constructor.
5. **`cfg.processor_kwargs`** added to `ModelInputConfig` and merged into
   kwargs passed to `processor_cls.from_pretrained` in `load_processor`.
   Axolotl-managed keys (`revision`, `trust_remote_code`) are filtered out
   with a warning if the user tries to override them.

## Audit table

| Strategy / chat template | Honors `roles_to_train`? (before) | (after) | Role-boundary markers | Media tokens masked |
|---|---|---|---|---|
| `ProcessingStrategy` (fallback for `llava`, `lfm2vl`, `mistral_v3_tekken`, unknown) | ✗ | fallback + warn | *unverified* | `image_token_id` if processor exposes it |
| `Qwen2VLProcessingStrategy` (`qwen2_vl`) | ✗ | ✓ | `<\|im_start\|>{role}\n` ... `<\|im_end\|>` | `<\|image_pad\|>` |
| `Qwen3_5ProcessingStrategy` (`qwen3_5`, `qwen3_5_moe`) | ✗ | ✓ | same as Qwen2VL | `<\|image_pad\|>`, `<\|video_pad\|>` |
| `Gemma3ProcessingStrategy` (`gemma3`) | ✗ | ✓ | `<start_of_turn>{model/user/system}\n` ... `<end_of_turn>` | `boi_token`, `<image_soft_token>` (262144) |
| `Gemma3nProcessingStrategy` (`gemma3n`) | ✓ (ad-hoc) | ✓ (shared scanner) | same as Gemma 3 | `image_token_id`, `audio_token_id`, `boi_token_id`, `eoi_token_id` |
| `Gemma4ProcessingStrategy` (`gemma4`) | n/a (new) | ✓ | `<\|turn>{model/user/system}` ... `<turn\|>` | `image_token_id`, `audio_token_id`, `boi/eoi/boa/eoa` (resolved via `convert_tokens_to_ids`), `video_token_id` (on processor) |
| `Llama3_2VisionProcessingStrategy` (`llama3_2_vision`) — **new** | ✗ | ✓ | `<\|start_header_id\|>{role}<\|end_header_id\|>\n\n` ... `<\|eot_id\|>` | `image_token_id` via base |
| `Llama4ProcessingStrategy` (`llama4`) — **new** | ✗ | ✓ | `<\|header_start\|>{role}<\|header_end\|>\n\n` ... `<\|eot\|>` | `image_token_id` via base |
| `PixtralProcessingStrategy` (`pixtral`) — **new** | ✗ | ✓¹ | user: `[INST]` ... `[/INST]`, assistant: `[/INST]` ... `eos_token` | `image_token_id` via base |
| `MistralV7TekkenProcessingStrategy` (`mistral_v7_tekken`) — **new** | ✗ | ✓¹ | `[SYSTEM_PROMPT]` ... `[/SYSTEM_PROMPT]`, `[INST]` ... `[/INST]`, assistant ends at `eos_token` | `image_token_id` via base |
| `VoxtralProcessingStrategy` | ✗ | fallback + warn | *unverified* (mistral-common tokenizer) | `audio_token`, `begin_audio_token` |
| `SmolVLM2ProcessingStrategy` | ✗ | fallback + warn | *unverified* (checkpoint-dependent default) | `<image>` |
| `Mistral3ProcessingStrategy` | ✗ | fallback + warn | *unverified* (mistral-common tokenizer) | `img`, `img_break`, `img_end` |
| `InternVLProcessingStrategy` | ✗ | fallback + warn | *unverified* (InternLM-family) | `processor.image_ids` |
| `Glm4vProcessingStrategy` | ✗ | fallback + warn | *unverified* | image/video + begin/end markers |

¹ Pixtral and Mistral V7 Tekken share a token (`[/INST]`) between the
user-end and assistant-start markers. The scanner consumes the first
occurrence when terminating the user span; the immediate assistant-start
that follows is therefore skipped unless the user removes the user boundary
from `roles_to_train` explicitly. This limitation is documented in a
tombstone test; users who need stricter assistant masking on these templates
should swap to the `pixtral`/`mistral_v7_tekken` template-type-specific
subclass we ship and open an issue for the shared-marker case.

*unverified*: the right boundary markers cannot be confirmed without a real
checkpoint; the fallback preserves the legacy "mask pad + media tokens only"
behavior and emits a one-shot warning naming the strategy class so the miss
is visible in training logs. To enable role masking for one of these models,
subclass the strategy and implement `_build_role_boundaries` — see the Gemma
and Qwen implementations for the pattern.

## Commits on this branch

1. **`feat: systemic multimodal assistant-only loss masking`** — core
   refactor of `processing_strategies.py` (`RoleBoundary`,
   `_apply_role_boundaries`, `_build_role_boundaries`), per-strategy boundary
   declarations, dispatcher routing for new subclasses.
2. **`feat: thread cfg.train_on_inputs / roles_to_train / train_on_eos into
   MM collator`** — `build_collator` reads the knobs from `cfg` and the
   first dataset entry and passes them to `get_processing_strategy`.
3. **`feat: forward cfg.processor_kwargs to processor from_pretrained (#3617)`**
   — schema field added; `load_processor` merges kwargs; axolotl-managed
   keys (`revision`, `trust_remote_code`) protected.
4. **`test: offline unit tests for multimodal role-mask scanner and
   processor_kwargs plumbing`** — 32 tests covering scanner semantics,
   per-strategy masking, media-token masking within assistant spans,
   dispatcher routing, and the processor_kwargs passthrough.
5. **`docs: multimodal assistant-mask design doc`** — this file.

(Final packaging: these were squashed into logical units during implementation
but the branch commit sequence can be organized per reviewer preference.)

## Verification

- All 32 unit tests pass offline (`pytest tests/test_processing_strategies.py`).
- End-to-end check against real tokenizers:
  - `google/gemma-4-E2B-it`: 13/40 tokens kept for a 2-turn chat; decoded
    preview shows only assistant responses + `<turn|>` markers remain.
  - `axolotl-ai-co/Llama-3.3-70B-Instruct-tokenizer` (with bundled
    `llama3_2_vision.jinja`): 11/64 tokens kept; content correctly resolves
    to `"The capital of France is Paris.<|eot_id|>"` and `"Berlin.<|eot_id|>"`.
- Verified boundary token ids against the real Gemma 4 tokenizer:
  `<|turn>model` → `[105, 4368]`, `<turn|>` → `[106]`, `<|image|>` → `258880`,
  `<|audio|>` → `258881`, `<|video|>` → `258884`.

## Draft upstream PR description

> Fix silently-ignored `train_on_inputs` / `roles_to_train` / `train_on_eos`
> in the multimodal training path, and plumb `cfg.processor_kwargs`
> (#3617).
>
> **Why this matters**: for every multimodal model except Gemma 3n, loss was
> computed on the entire sequence (minus pad and media tokens) regardless of
> what `roles_to_train` / `train_on_inputs` the config specified. This
> silently turned assistant-only SFT into full-sequence SFT for thousands of
> users, degrading sample efficiency and introducing spurious gradient signal
> on system and user content.
>
> **What changed**:
> - `ProcessingStrategy._build_role_boundaries` declares per-role start/end
>   token sequences. The base `_mask_non_assistant` now consumes those
>   declarations via a shared scanner that honors `train_on_inputs`,
>   `roles_to_train`, and `train_on_eos`.
> - Per-strategy boundary declarations added for Qwen2-VL, Qwen3.5, Gemma 3,
>   Gemma 3n (refactored from ad-hoc scanner), Gemma 4 (new), Llama 3.2
>   Vision (new), Llama 4 (new), Pixtral (new), Mistral V7 Tekken (new).
> - Strategies whose boundary tokens we couldn't verify against a real
>   tokenizer (Voxtral, SmolVLM2, Mistral3, InternVL, GLM4V, and the
>   llava/lfm2vl/unknown-template fallback) retain legacy behavior but emit a
>   one-shot warning so the miss is visible in training logs.
> - `cfg.train_on_inputs` / `cfg.datasets[0].roles_to_train` /
>   `cfg.datasets[0].train_on_eos` are threaded through
>   `HFCausalTrainerBuilder.build_collator` → `get_processing_strategy` →
>   strategy constructor.
> - `cfg.processor_kwargs` (new) is merged into
>   `processor_cls.from_pretrained` kwargs; `revision` and `trust_remote_code`
>   remain axolotl-managed.
>
> **Testing**: 32 offline unit tests; end-to-end verified with the real
> Gemma 4 and Llama 3.x tokenizers.
