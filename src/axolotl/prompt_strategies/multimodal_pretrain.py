from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.image_utils import load_image

from axolotl.prompt_tokenizers import DatasetWrappingStrategy
from axolotl.utils.data.mm_image import resize_image_for_processor
from axolotl.utils.data.mm_image_transform import (
    MMImageTransform,
    resolve_mm_image_transform,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class MultiModalPretrainDatasetWrappingStrategy(DatasetWrappingStrategy):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin,
        sequence_len: int,
        text_column: str = "text",
        image_column: str = "images",
        image_token: str | None = None,
        image_base_dir: str | None = None,
        add_processor_lengths: bool = False,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Any | None = None,
        image_resize_buckets: list[tuple[int, int]] | None = None,
        image_resize_no_upscale: bool = False,
        image_resize_pad_color: Any | None = None,
        image_transform: MMImageTransform | None = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.sequence_len = sequence_len
        self.text_column = text_column
        self.image_column = image_column
        self.image_base_dir = image_base_dir
        self.add_processor_lengths = add_processor_lengths
        self.image_size = image_size
        self.image_resize_algorithm = image_resize_algorithm
        self.image_resize_buckets = image_resize_buckets
        self.image_resize_no_upscale = image_resize_no_upscale
        self.image_resize_pad_color = image_resize_pad_color
        self.image_transform = image_transform
        self.image_token_spec = build_image_token_spec(processor, override=image_token)

    def _encode_batch(self, examples: dict[str, list]) -> dict[str, list]:
        return encode_multimodal_pretrain(
            examples,
            tokenizer=self.tokenizer,
            max_tokens=self.sequence_len,
            image_token=self.image_token_spec.image_token,
            image_token_id=self.image_token_spec.image_token_id,
            text_column=self.text_column,
            image_column=self.image_column,
            processor=self.processor,
            image_base_dir=self.image_base_dir,
            add_processor_lengths=self.add_processor_lengths,
            image_size=self.image_size,
            image_resize_algorithm=self.image_resize_algorithm,
            image_resize_buckets=self.image_resize_buckets,
            image_resize_no_upscale=self.image_resize_no_upscale,
            image_resize_pad_color=self.image_resize_pad_color,
            image_transform=self.image_transform,
        )

    def wrap_dataset(
        self,
        dataset,
        process_count: int | None = None,
        keep_in_memory: bool | None = False,
        **kwargs,
    ) -> Dataset | IterableDataset:
        if isinstance(dataset, Dataset):
            remove_columns = list(dataset.column_names)
        elif getattr(dataset, "features", None):
            remove_columns = list(dataset.features.keys())
        else:
            remove_columns = None

        map_kwargs: dict[str, Any] = {
            "batched": True,
            "remove_columns": remove_columns,
            "desc": "Tokenizing multimodal CPT dataset",
        }
        if isinstance(dataset, Dataset):
            if process_count:
                map_kwargs["num_proc"] = process_count
            if keep_in_memory is not None:
                map_kwargs["keep_in_memory"] = keep_in_memory

        return dataset.map(self._encode_batch, **map_kwargs)


def load(
    tokenizer,
    cfg,
    ds_cfg: Optional[dict[str, Any]] = None,
    processor: ProcessorMixin | None = None,
):
    ds_cfg = ds_cfg or {}
    if processor is None:
        raise ValueError(
            "Multimodal CPT (type: multimodal_pretrain) requires a processor. "
            "Set `processor_type: AutoProcessor` (or the concrete processor "
            "class) in your config."
        )
    check_processor_compatibility(processor)
    processor_tokenizer = getattr(processor, "tokenizer", None)
    if processor_tokenizer is not None and processor_tokenizer is not tokenizer:
        raise ValueError(
            "Multimodal CPT requires `tokenizer` to be `processor.tokenizer` "
            "so image placeholder ids stay aligned during encoding."
        )

    text_column = ds_cfg.get("text_column") or "text"
    image_column = ds_cfg.get("image_column") or "images"
    LOG.info(
        "multimodal CPT dataset path: text_column=%r image_column=%r",
        text_column,
        image_column,
    )
    return MultiModalPretrainDatasetWrappingStrategy(
        tokenizer=tokenizer,
        processor=processor,
        sequence_len=cfg.sequence_len,
        text_column=text_column,
        image_column=image_column,
        image_token=ds_cfg.get("image_token"),
        image_base_dir=ds_cfg.get("image_base_dir"),
        add_processor_lengths=bool(cfg.sample_packing),
        image_size=cfg.image_size,
        image_resize_algorithm=cfg.image_resize_algorithm,
        image_resize_buckets=cfg.image_resize_buckets,
        image_resize_no_upscale=bool(cfg.image_resize_no_upscale),
        image_resize_pad_color=cfg.image_resize_pad_color,
        image_transform=resolve_mm_image_transform(cfg),
    )


def encode_multimodal_pretrain(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    image_token: str,
    image_token_id: int,
    text_column: str = "text",
    image_column: str = "images",
    enforce_max_length: bool = True,
    processor: ProcessorMixin | None = None,
    image_base_dir: str | None = None,
    add_processor_lengths: bool = False,
    add_eos_token: bool = True,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Any | None = None,
    image_resize_buckets: list[tuple[int, int]] | None = None,
    image_resize_no_upscale: bool = False,
    image_resize_pad_color: Any | None = None,
    image_transform: MMImageTransform | None = None,
) -> dict[str, list]:
    texts: list[str] = examples[text_column]
    imgs_list: list[list[str]] = examples[image_column]

    if len(texts) != len(imgs_list):
        raise ValueError(
            f"encode_multimodal_pretrain: text column has {len(texts)} rows "
            f"but image column has {len(imgs_list)}"
        )

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_mask: list[list[int]] = []
    keep_images: list[list[str]] = []
    keep_text: list[str] = []
    lengths: list[int] = []

    for text, imgs in zip(texts, imgs_list, strict=True):
        if not isinstance(text, str):
            raise TypeError(
                f"encode_multimodal_pretrain: `{text_column}` must be str, "
                f"got {type(text).__name__}."
            )
        if imgs is None:
            imgs = []
        if not isinstance(imgs, (list, tuple)):
            raise ValueError(
                f"encode_multimodal_pretrain: row's `{image_column}` must be "
                f"a list; got {type(imgs).__name__}"
            )
        for j, ip in enumerate(imgs):
            if not isinstance(ip, str) and not hasattr(ip, "resize"):
                raise TypeError(
                    f"encode_multimodal_pretrain: image {j} in row must be "
                    f"a path/URL string or PIL image, got {type(ip).__name__}."
                )
        # Avoid truncation before processor re-tokenization.
        enc = tokenizer(text, add_special_tokens=True)
        ids = list(enc["input_ids"])
        mask = list(enc["attention_mask"])
        if add_eos_token:
            ids.append(tokenizer.eos_token_id)
            mask.append(1)
        # Count by id; text.count can match <image> inside <image_soft_token>.
        n_placeholders = sum(1 for t in ids if t == image_token_id)
        if n_placeholders != len(imgs):
            raise ValueError(
                f"Multimodal CPT row has {n_placeholders} occurrence(s) of "
                f"{image_token!r} in text but {len(imgs)} image path(s). "
                f"Text and image count must match (one placeholder per image)."
            )
        if enforce_max_length and len(ids) > max_tokens:
            raise ValueError(
                f"Multimodal CPT row tokenizes to {len(ids)} tokens which "
                f"exceeds sequence_len={max_tokens}. Pre-chunk your text or "
                f"raise sequence_len (image patch expansion at the processor "
                f"may push the final length even higher)."
            )
        # Labels = ids; collator masks image-family ids after re-tokenization.
        input_ids.append(ids)
        labels.append(list(ids))
        attention_mask.append(mask)
        keep_images.append(list(imgs))
        keep_text.append(text)

    if add_processor_lengths:
        if processor is None:
            raise ValueError(
                "encode_multimodal_pretrain: processor is required when "
                "add_processor_lengths=True."
            )
        lengths = compute_multimodal_processor_lengths(
            keep_text,
            keep_images,
            tokenizer=tokenizer,
            processor=processor,
            image_base_dir=image_base_dir,
            add_eos_token=add_eos_token,
            image_size=image_size,
            image_resize_algorithm=image_resize_algorithm,
            image_resize_buckets=image_resize_buckets,
            image_resize_no_upscale=image_resize_no_upscale,
            image_resize_pad_color=image_resize_pad_color,
            image_transform=image_transform,
            image_token=image_token,
        )
        if enforce_max_length:
            oversized = [
                (row_idx, length)
                for row_idx, length in enumerate(lengths)
                if length > max_tokens
            ]
            if oversized:
                # The text-only pre-check above can pass while image patch / tiling
                # expansion pushes the real processor length over the budget. Surface
                # it here (at encode time) rather than letting it fail later inside
                # packing, far from the cause.
                LOG.warning(
                    "%d multimodal CPT row(s) expand beyond sequence_len=%d after "
                    "image patch / tiling expansion (e.g. row %d -> %d tokens); "
                    "they cannot be packed. Pre-chunk the text, lower the tile grid, "
                    "or raise sequence_len.",
                    len(oversized),
                    max_tokens,
                    oversized[0][0],
                    oversized[0][1],
                )

    out: dict[str, list[Any]] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "images": keep_images,
        "_mm_text": keep_text,
    }
    if add_processor_lengths:
        out["length"] = lengths
    return out


def append_eos_for_processor(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    add_eos_token: bool = True,
) -> str:
    eos = getattr(tokenizer, "eos_token", None)
    if add_eos_token and eos and not text.endswith(eos):
        return text + eos
    return text


def row_starts_with_bos(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None or "input_ids" not in row:
        return False
    input_ids = row["input_ids"]
    try:
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        return bool(input_ids) and int(input_ids[0]) == int(bos_id)
    except (TypeError, ValueError):
        return False


def prepare_text_for_packed_boundary(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    is_first: bool,
    add_eos_token: bool = True,
) -> str:
    text = row["_mm_text"]
    if not isinstance(text, str):
        raise TypeError(
            f"`_mm_text` must be str, got {type(text).__name__}. "
            "Check dataset encoding."
        )
    text = append_eos_for_processor(text, tokenizer, add_eos_token=add_eos_token)
    bos = getattr(tokenizer, "bos_token", None)
    if not is_first and bos and row_starts_with_bos(row, tokenizer):
        return bos + text
    return text


def _resolve_image_source(image_base_dir: str | None, src: Any) -> Any:
    if (
        image_base_dir
        and isinstance(src, str)
        and not os.path.isabs(src)
        and "://" not in src
    ):
        return os.path.join(image_base_dir, src)
    return src


def _load_images_for_lengths(
    imgs_list: list[list[Any]],
    image_base_dir: str | None,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Any | None = None,
    image_resize_buckets: list[tuple[int, int]] | None = None,
    image_resize_no_upscale: bool = False,
    image_resize_pad_color: Any | None = None,
    image_transform: MMImageTransform | None = None,
) -> list[list[Any]]:
    loaded: list[list[Any]] = []
    tile_cache = image_transform.new_cache() if image_transform else None
    for row_idx, sources in enumerate(imgs_list):
        if image_transform is not None:
            row = []
            try:
                for raw in sources or []:
                    tiles, _labels = image_transform.per_source(
                        raw,
                        image_base_dir=image_base_dir,
                        resize_algorithm=image_resize_algorithm,
                        cache=tile_cache,
                    )
                    row.extend(tiles)
            except Exception as exc:
                raise RuntimeError(
                    f"Row {row_idx}: failed to tile image(s) while estimating "
                    "multimodal processor length "
                    f"({type(exc).__name__})."
                ) from exc
            loaded.append(row)
            continue

        row = []
        for raw in sources:
            try:
                # Route every input (incl. already-PIL) through load_image to match
                # the runtime collator, which converts L/RGBA->RGB; skipping it for
                # PIL inputs would make the estimated length diverge from the real one.
                image = load_image(_resolve_image_source(image_base_dir, raw))
                row.append(
                    resize_image_for_processor(
                        image,
                        image_size,
                        image_resize_algorithm,
                        image_resize_buckets,
                        image_resize_no_upscale,
                        image_resize_pad_color,
                    )
                )
            except Exception as exc:
                label = (
                    os.path.basename(raw)
                    if isinstance(raw, str)
                    else type(raw).__name__
                )
                raise RuntimeError(
                    f"Row {row_idx}: failed to load image {label!r} while "
                    "estimating multimodal processor length "
                    f"({type(exc).__name__})."
                ) from exc
        loaded.append(row)
    return loaded


def _batch_lengths_from_processor_output(batch: Any) -> list[int]:
    if "attention_mask" in batch:
        attn = batch["attention_mask"]
        lengths = (
            attn.sum(dim=-1) if hasattr(attn, "sum") else [sum(row) for row in attn]
        )
        if hasattr(lengths, "tolist"):
            return [int(x) for x in lengths.tolist()]
        return [int(x) for x in lengths]
    input_ids = batch["input_ids"]
    if hasattr(input_ids, "shape"):
        return [int(input_ids.shape[-1])] * int(input_ids.shape[0])
    return [len(row) for row in input_ids]


def compute_multimodal_processor_lengths(
    texts: list[str],
    imgs_list: list[list[Any]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    processor: ProcessorMixin,
    image_base_dir: str | None = None,
    add_eos_token: bool = True,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Any | None = None,
    image_resize_buckets: list[tuple[int, int]] | None = None,
    image_resize_no_upscale: bool = False,
    image_resize_pad_color: Any | None = None,
    image_transform: MMImageTransform | None = None,
    image_token: str | None = None,
) -> list[int]:
    if len(texts) != len(imgs_list):
        raise ValueError(
            "compute_multimodal_processor_lengths: text/image row count mismatch "
            f"({len(texts)} text row(s), {len(imgs_list)} image row(s))."
        )

    tile_cache = image_transform.new_cache() if image_transform else None
    if image_transform is not None:
        tiling_image_token = (
            image_token or build_image_token_spec(processor).image_token
        )
        prepared = [
            image_transform.prepare(
                text,
                list(images or []),
                image_token=tiling_image_token,
                image_base_dir=image_base_dir,
                resize_algorithm=image_resize_algorithm,
                cache=tile_cache,
            )
            for text, images in zip(texts, imgs_list, strict=True)
        ]
        texts = [text for text, _images in prepared]
        imgs_list = [images for _text, images in prepared]

    processor_texts = [
        append_eos_for_processor(text, tokenizer, add_eos_token=add_eos_token)
        for text in texts
    ]
    if all(len(imgs) == 0 for imgs in imgs_list):
        return _batch_lengths_from_processor_output(
            tokenizer(text=processor_texts, return_tensors="pt", padding=True)
        )

    loaded_images = _load_images_for_lengths(
        imgs_list,
        image_base_dir,
        image_size=None if image_transform is not None else image_size,
        image_resize_algorithm=image_resize_algorithm,
        image_resize_buckets=(
            None if image_transform is not None else image_resize_buckets
        ),
        image_resize_no_upscale=(
            False if image_transform is not None else image_resize_no_upscale
        ),
        image_resize_pad_color=(
            None if image_transform is not None else image_resize_pad_color
        ),
        image_transform=None,
    )
    try:
        batch = processor(
            text=processor_texts,
            images=loaded_images,
            return_tensors="pt",
            padding=True,
        )
        return _batch_lengths_from_processor_output(batch)
    except Exception:
        lengths: list[int] = []
        for text, row_images in zip(processor_texts, loaded_images, strict=True):
            if row_images:
                batch = processor(
                    text=[text],
                    images=[row_images],
                    return_tensors="pt",
                    padding=True,
                )
            else:
                batch = tokenizer(text=[text], return_tensors="pt", padding=True)
            lengths.extend(_batch_lengths_from_processor_output(batch))
        return lengths


def _get_incompatible_processor_classes() -> tuple[type, ...]:
    import importlib

    classes: list[type] = []
    for mod_path, name in (
        ("transformers.models.mllama", "MllamaProcessor"),
        ("transformers.models.pixtral", "PixtralProcessor"),
        ("transformers.models.internvl", "InternVLProcessor"),
    ):
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, name, None)
            if cls is not None:
                classes.append(cls)
        except ImportError:
            continue
    return tuple(classes)


_KNOWN_IMAGE_TOKEN_CANDIDATES: tuple[str, ...] = (
    "<image>",
    "<|image|>",
    "<|image_pad|>",
    "<image_soft_token>",
    "<start_of_image>",
    "[IMG]",
    "<IMG_CONTEXT>",
)

# Without masking these in labels, loss blows up ~10x on Qwen/SmolVLM.
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
    image_token: str
    image_token_id: int
    image_family_token_ids: set[int]


def build_image_token_spec(
    processor: ProcessorMixin, override: str | None = None
) -> ImageTokenSpec:
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

    known_special_tokens: set[str] = set()
    try:
        known_special_tokens |= set(tokenizer.get_added_vocab().keys())
    except Exception as exc:  # noqa: BLE001
        LOG.debug(
            "tokenizer.get_added_vocab() failed on %s: %s",
            type(tokenizer).__name__,
            exc,
        )
    known_special_tokens |= set(getattr(tokenizer, "all_special_tokens", None) or [])
    known_special_tokens |= set(
        getattr(tokenizer, "additional_special_tokens", None) or []
    )

    image_token: str | None = None
    image_token_id: int | None = None
    if override is not None:
        # Reject plain words that BPE-tokenize cleanly but aren't placeholders.
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
        proc_token = getattr(processor, "image_token", None)
        # Gemma-3-style only: `image_token` is the post-expansion soft token
        # (its name literally contains "soft_token"); the user-facing
        # placeholder is `boi_token`. Gemma-4 reverses this — `image_token`
        # IS the user-facing placeholder (`<|image|>`) and `boi_token`
        # (`<|image>`) is just a bracket marker, so don't blindly swap.
        boi_token = getattr(processor, "boi_token", None)
        if (
            boi_token
            and proc_token
            and boi_token != proc_token
            and boi_token in known_special_tokens
            and "soft_token" in proc_token
        ):
            proc_token = boi_token
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

    # Filter to registered tokens so BPE-fallback ids don't get masked.
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
    if _INCOMPATIBLE_PROCESSOR_CLASSES and isinstance(
        processor, _INCOMPATIBLE_PROCESSOR_CLASSES
    ):
        for cls in _INCOMPATIBLE_PROCESSOR_CLASSES:
            if isinstance(processor, cls):
                raise ValueError(
                    f"Multimodal CPT is not supported for {cls.__name__}: "
                    f"{_INCOMPATIBLE_PROCESSOR_REASONS.get(cls.__name__, '')}"
                )
    # MRO-name fallback for test fakes and unimportable concrete classes.
    for base_cls in type(processor).__mro__:
        reason = _INCOMPATIBLE_PROCESSOR_REASONS.get(base_cls.__name__)
        if reason is not None:
            raise ValueError(
                f"Multimodal CPT is not supported for {base_cls.__name__}: {reason}"
            )
