"""Multimodal CPT tokenization strategy (raw image+text, no chat template)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import BatchEncoding, PreTrainedTokenizerBase, ProcessorMixin

from axolotl.prompt_strategies.pretrain import (
    PretrainTokenizationStrategy,
    PretrainTokenizer,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _get_incompatible_processor_classes() -> tuple[type, ...]:
    """Real class refs for incompatible processors (subclass-safe via isinstance)."""
    classes: list[type] = []
    for mod_path, name in (
        ("transformers.models.mllama", "MllamaProcessor"),
        ("transformers.models.pixtral", "PixtralProcessor"),
        ("transformers.models.internvl", "InternVLProcessor"),
    ):
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            cls = getattr(mod, name, None)
            if cls is not None:
                classes.append(cls)
        except ImportError:
            continue
    return tuple(classes)


# Placeholder tokens axolotl knows about. Auto-detection probes these in
# order against `processor.tokenizer`; first hit wins. Only used as a
# fallback when `processor.image_token` is not exposed.
_KNOWN_IMAGE_TOKEN_CANDIDATES: tuple[str, ...] = (
    "<image>",
    "<|image|>",
    "<|image_pad|>",
    "<image_soft_token>",
    "<start_of_image>",
    "[IMG]",
    "<IMG_CONTEXT>",
)

# The full set of image-family tokens that should be masked out of labels
# (loss=-100). Includes wrappers like `<|vision_start|>` and `<end_of_image>`
# in addition to the visible placeholder. Empirically confirmed: without this
# masking, loss blows up ~10× on Qwen and SmolVLM families.
_IMAGE_FAMILY_TOKEN_CANDIDATES: tuple[str, ...] = (
    "<image>",
    "<|image|>",
    "<|image_pad|>",
    "<image_soft_token>",
    "<start_of_image>",
    "<end_of_image>",
    "<|vision_start|>",
    "<|vision_end|>",
    "[IMG]",
    "[IMG_END]",
    "<IMG_CONTEXT>",
)

# Processor classes we refuse for v1 multimodal CPT, with a user-facing reason.
# Keyed by class-name for the message, but the actual match uses `isinstance`
# against the real imports below — this catches user-defined subclasses too.
_INCOMPATIBLE_PROCESSOR_REASONS: dict[str, str] = {
    "MllamaProcessor": (
        "Llama-3.2-Vision (Mllama) uses cross-attention image injection, not "
        "in-stream placeholder tokens. Multimodal CPT is incompatible with "
        "this architecture; use chat-template SFT instead."
    ),
    "PixtralProcessor": (
        "Pixtral's tokenizer goes through mistral_common with a different "
        "API surface than AutoProcessor. Multimodal CPT not supported in v1; "
        "use chat-template SFT or Mistral-Small-3.1."
    ),
    "InternVLProcessor": (
        "InternVL ships a custom processing pipeline (AutoProcessor returns "
        "text-only); no pixel_values are produced. Multimodal CPT not "
        "supported in v1."
    ),
}
_INCOMPATIBLE_PROCESSOR_CLASSES = _get_incompatible_processor_classes()


@dataclass
class ImageTokenSpec:
    """Placeholder token + image-family id set for label masking."""

    image_token: str
    image_token_id: int
    image_family_token_ids: set[int]


def build_image_token_spec(
    processor: ProcessorMixin, override: str | None = None
) -> ImageTokenSpec:
    """Resolve placeholder token + family mask set. Raises if autodetect fails."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "Processor has no `tokenizer` attribute — multimodal CPT "
            "requires a processor with a text tokenizer (e.g. one produced "
            "by AutoProcessor.from_pretrained for a VLM)."
        )

    def resolve_id(tok: str) -> int | None:
        tid = tokenizer.convert_tokens_to_ids(tok)
        unk = getattr(tokenizer, "unk_token_id", None)
        if tid is None or tid == unk:
            return None
        return tid

    # Full set of tokens we consider "genuinely registered" for this
    # tokenizer. Used both to validate an override and to filter the
    # family-mask list below.
    known_special_tokens: set[str] = set()
    try:
        known_special_tokens |= set(tokenizer.get_added_vocab().keys())
    except Exception:
        pass
    known_special_tokens |= set(getattr(tokenizer, "all_special_tokens", None) or [])
    known_special_tokens |= set(
        getattr(tokenizer, "additional_special_tokens", None) or []
    )

    # Placeholder the user writes in the text column.
    image_token: str | None = None
    image_token_id: int | None = None
    if override is not None:
        # Require overrides to be actual registered special tokens — a plain
        # word like "image" BPE-tokenizes to a real id (not unk) but is not
        # a placeholder, and accepting it would silently break alignment.
        if override not in known_special_tokens:
            raise ValueError(
                f"image_token override {override!r} is not a registered "
                f"special token on this tokenizer. Pick one of the model's "
                f"actual image tokens (e.g. '<image>', '<|image_pad|>', "
                f"'<start_of_image>'), or leave unset to autodetect."
            )
        image_token_id = resolve_id(override)
        if image_token_id is None:
            raise ValueError(
                f"image_token override {override!r} did not resolve to a "
                f"token id (unk). Remove the override to autodetect."
            )
        image_token = override
    else:
        # Prefer the processor's own declaration when available.
        proc_token = getattr(processor, "image_token", None)
        if proc_token is not None:
            image_token_id = resolve_id(proc_token)
            if image_token_id is not None:
                image_token = proc_token
        if image_token is None:
            for cand in _KNOWN_IMAGE_TOKEN_CANDIDATES:
                tid = resolve_id(cand)
                if tid is not None:
                    image_token = cand
                    image_token_id = tid
                    break
        if image_token is None:
            raise ValueError(
                "Could not autodetect the image placeholder token for this "
                "processor. Set `image_token: <token>` in the dataset config "
                "(e.g. '<image>' for LLaVA, '<|image_pad|>' for Qwen-VL, "
                "'<start_of_image>' for Gemma-3)."
            )

    # Full family for label masking. Filter to genuine registered tokens so
    # we don't accidentally mask a legitimate text token whose string form
    # happens to resolve through BPE fallback.
    family: set[int] = {image_token_id}  # type: ignore[arg-type]
    for cand in _IMAGE_FAMILY_TOKEN_CANDIDATES:
        if cand != image_token and cand not in known_special_tokens:
            continue
        tid = resolve_id(cand)
        if tid is not None:
            family.add(tid)
    return ImageTokenSpec(
        image_token=image_token,
        image_token_id=image_token_id,  # type: ignore[arg-type]
        image_family_token_ids=family,
    )


def check_processor_compatibility(processor: ProcessorMixin) -> None:
    """Raise ValueError for v1-incompatible processors (Mllama/Pixtral/InternVL)."""
    if _INCOMPATIBLE_PROCESSOR_CLASSES and isinstance(
        processor, _INCOMPATIBLE_PROCESSOR_CLASSES
    ):
        for cls in _INCOMPATIBLE_PROCESSOR_CLASSES:
            if isinstance(processor, cls):
                raise ValueError(
                    f"Multimodal CPT is not supported for {cls.__name__}: "
                    f"{_INCOMPATIBLE_PROCESSOR_REASONS.get(cls.__name__, '')}"
                )
    # Fallback: walk the MRO class names (handles unit-test fakes and
    # cases where the concrete class couldn't be imported at module load).
    for base_cls in type(processor).__mro__:
        reason = _INCOMPATIBLE_PROCESSOR_REASONS.get(base_cls.__name__)
        if reason is not None:
            raise ValueError(
                f"Multimodal CPT is not supported for {base_cls.__name__}: " f"{reason}"
            )


class MultimodalPretrainTokenizationStrategy(PretrainTokenizationStrategy):
    """Pretrain tokenizer that preserves images + raw text columns for the collator."""

    def __init__(
        self,
        *args: Any,
        image_token: str,
        image_token_id: int,
        image_column: str = "images",
        image_base_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.image_column = image_column
        self.image_base_dir = image_base_dir

    def _tokenize(
        self,
        prompt: str,
        add_eos_token: bool = True,
        strip_bos_token: bool = False,
    ) -> BatchEncoding:
        # No overflow / stride — keep a 1:1 row-to-chunk mapping so images
        # don't need to be duplicated across chunks (ambiguous semantics).
        res = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length - 1,
            add_special_tokens=True,
        )
        # Restructure to the "list of one" format the base class expects.
        res["input_ids"] = [res["input_ids"] + [self.tokenizer.eos_token_id]]
        res["attention_mask"] = [res["attention_mask"] + [1]]
        return res

    def tokenize_prompt(self, prompt: dict[str, Any]) -> dict[str, list]:
        text = prompt[self.text_column]
        images = prompt.get(self.image_column) or []
        if not isinstance(images, (list, tuple)):
            raise ValueError(
                f"Row's `{self.image_column}` must be a list of image paths, "
                f"got {type(images).__name__}."
            )

        # Count placeholder occurrences by tokenizing once and counting token
        # ids — safer than `text.count(...)` which has prefix-match bugs
        # (e.g. "<image>" substring-matching inside "<image_soft_token>").
        probe_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        n_placeholders = sum(1 for t in probe_ids if t == self.image_token_id)
        if n_placeholders != len(images):
            raise ValueError(
                f"Multimodal CPT row has {n_placeholders} occurrence(s) of "
                f"{self.image_token!r} in text but {len(images)} image path(s) "
                f"in `{self.image_column}`. They must match — the text column "
                f"must contain exactly one placeholder per image. "
                f"(silent-failure guard: LLaVA/Qwen-VL would accept this "
                f"without error but drop the image at the model.)"
            )

        res = self._tokenize(text)
        n_chunks = len(res["input_ids"])
        # Parallel lists so `.map(batched=True)` keeps alignment.
        res["images"] = [list(images)] * n_chunks
        res["_mm_text"] = [text] * n_chunks
        return res


def load(
    tokenizer: PreTrainedTokenizerBase,
    cfg: Any,
    ds_cfg: dict | None = None,
    processor: ProcessorMixin | None = None,
) -> MultimodalPretrainTokenizationStrategy:
    """Factory for the non-streaming multimodal CPT path."""
    if processor is None:
        raise ValueError(
            "multimodal_pretrain requires a processor. Set `processor_type: "
            "AutoProcessor` (or the concrete processor class) in your config "
            "so axolotl loads it at startup."
        )
    check_processor_compatibility(processor)

    ds_cfg = dict(ds_cfg or {})
    # Accept config from either `pretraining_dataset[0]` or `datasets[i]`.
    text_column = ds_cfg.get("text_column") or ds_cfg.get("field") or "text"
    image_column = ds_cfg.get("image_column") or "images"
    image_base_dir = ds_cfg.get("image_base_dir")
    image_token_override = ds_cfg.get("image_token")

    spec = build_image_token_spec(processor, override=image_token_override)
    LOG.info(
        f"multimodal_pretrain: placeholder={spec.image_token!r} "
        f"(id={spec.image_token_id}), masking {len(spec.image_family_token_ids)} "
        f"image-family token ids in labels"
    )

    strat = MultimodalPretrainTokenizationStrategy(
        PretrainTokenizer(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        text_column=text_column,
        image_column=image_column,
        image_base_dir=image_base_dir,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        max_length=cfg.sequence_len,
    )
    # Stash spec on the strategy so downstream code (collator, validator)
    # can read it without re-probing the processor.
    strat.image_token_spec = spec  # type: ignore[attr-defined]
    return strat
