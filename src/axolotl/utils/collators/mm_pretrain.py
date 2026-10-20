"""Collator for multimodal CPT."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import torch
from PIL import Image
from torch import Tensor
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.image_utils import load_image
from transformers.utils import PaddingStrategy

from axolotl.prompt_strategies.multimodal_pretrain import (
    ImageTokenSpec,
    check_processor_compatibility,
    compute_multimodal_processor_lengths,
    prepare_text_for_packed_boundary,
)
from axolotl.utils.data.mm_image import resize_image_for_processor
from axolotl.utils.data.mm_image_transform import MMImageTransform
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass
class MultiModalPretrainDataCollator(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    processor: ProcessorMixin
    image_token_spec: ImageTokenSpec
    image_base_dir: Optional[str] = None
    return_tensors: Literal["pt"] = "pt"
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    max_length: Optional[int] = None
    skip_bad_images: bool = False
    add_eos_token: bool = True
    sample_packing: bool = False
    image_size: int | tuple[int, int] | None = None
    image_resize_algorithm: Image.Resampling | None = None
    image_resize_buckets: list[tuple[int, int]] | None = None
    image_resize_no_upscale: bool = False
    image_resize_pad_color: Any | None = None
    image_transform: MMImageTransform | None = None

    _image_family_token_ids: set[int] = field(init=False, default_factory=set)
    _image_tile_cache: Any | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.return_tensors != "pt":
            raise ValueError(
                "MultiModalPretrainDataCollator only supports "
                "return_tensors='pt' (in-place torch ops are used downstream)."
            )
        check_processor_compatibility(self.processor)
        # All-text batches use self.tokenizer; image batches use self.processor.
        # If they don't share the same tokenizer instance, the two paths can
        # tokenize the same text differently.
        proc_tokenizer = getattr(self.processor, "tokenizer", None)
        if proc_tokenizer is not None and proc_tokenizer is not self.tokenizer:
            LOG.warning(
                "MultiModalPretrainDataCollator.tokenizer is not "
                "processor.tokenizer; all-text and image batches may "
                "tokenize inconsistently."
            )
        self._image_family_token_ids = set(self.image_token_spec.image_family_token_ids)
        if self.image_transform is not None:
            self._image_tile_cache = self.image_transform.new_cache()

    def _resolve_image_source(self, src: Any) -> Any:
        # Only join base_dir for relative string paths; pass everything else
        # (PIL images, URLs, base64, absolute paths) through to load_image.
        if (
            self.image_base_dir
            and isinstance(src, str)
            and not os.path.isabs(src)
            and "://" not in src
        ):
            return os.path.join(self.image_base_dir, src)
        return src

    def _raw_image_sources(self, raw: Any, row_index: int) -> list:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            return list(raw)
        raise TypeError(
            f"Row {row_index}: `images` must be a list (or None), got "
            f"{type(raw).__name__}."
        )

    def _load_images_for_row(self, sources: list, row_index: int) -> list[Image.Image]:
        out: list[Image.Image] = []
        for raw in sources:
            try:
                img = load_image(self._resolve_image_source(raw))
            except Exception as exc:
                label = (
                    os.path.basename(raw)
                    if isinstance(raw, str)
                    else type(raw).__name__
                )
                msg = (
                    f"Row {row_index}: failed to load image {label!r} "
                    f"({type(exc).__name__})"
                )
                LOG.debug("failed image full source: %r; error: %s", raw, exc)
                if self.skip_bad_images:
                    LOG.warning("%s — skipping", msg)
                    continue
                raise RuntimeError(msg) from exc
            out.append(
                resize_image_for_processor(
                    img,
                    self.image_size,
                    self.image_resize_algorithm,
                    self.image_resize_buckets,
                    self.image_resize_no_upscale,
                    self.image_resize_pad_color,
                )
            )
        return out

    def _prepare_row_text_and_images(
        self,
        text: str,
        sources: list,
        row_index: int,
    ) -> tuple[str, list[Image.Image]] | None:
        if self.image_transform is None:
            loaded = self._load_images_for_row(sources, row_index=row_index)
            if self.skip_bad_images and len(loaded) != len(sources):
                return None
            return text, loaded

        out: list[Image.Image] = []
        counts: list[int] = []
        labels: list[list[str] | None] = []
        for raw in sources:
            try:
                tiles, tile_labels = self.image_transform.per_source(
                    raw,
                    image_base_dir=self.image_base_dir,
                    resize_algorithm=self.image_resize_algorithm,
                    cache=self._image_tile_cache,
                )
            except Exception as exc:
                label = (
                    os.path.basename(raw)
                    if isinstance(raw, str)
                    else type(raw).__name__
                )
                msg = (
                    f"Row {row_index}: failed to tile image {label!r} "
                    f"({type(exc).__name__})"
                )
                LOG.debug("failed tile full source: %r; error: %s", raw, exc)
                if self.skip_bad_images:
                    LOG.warning("%s — skipping row", msg)
                    return None
                raise RuntimeError(msg) from exc
            counts.append(len(tiles))
            labels.append(tile_labels)
            out.extend(tiles)
        return (
            self.image_transform.expand_placeholders(
                text,
                image_token=self.image_token_spec.image_token,
                counts=counts,
                labels=labels,
            ),
            out,
        )

    @staticmethod
    def _coerce_length(value: Any) -> int | None:
        if value is None:
            return None
        if hasattr(value, "numel") and value.numel() == 1:
            return int(value.item())
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list):
            if len(value) != 1:
                return None
            value = value[0]
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_sample_lengths(value: Any) -> list[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise TypeError("`_mm_sample_lengths` must be a list of integers.")
        return [int(v.item() if hasattr(v, "item") else v) for v in value]

    def _fallback_row_lengths(self, rows: list[dict]) -> list[int]:
        texts = [row["_mm_text"] for row in rows]
        image_sources = [
            self._raw_image_sources(row.get("images"), row_index=i)
            for i, row in enumerate(rows)
        ]
        return compute_multimodal_processor_lengths(
            texts,
            image_sources,
            tokenizer=self.tokenizer,
            processor=self.processor,
            image_base_dir=self.image_base_dir,
            add_eos_token=self.add_eos_token,
            image_size=self.image_size,
            image_resize_algorithm=self.image_resize_algorithm,
            image_resize_buckets=self.image_resize_buckets,
            image_resize_no_upscale=self.image_resize_no_upscale,
            image_resize_pad_color=self.image_resize_pad_color,
            image_transform=self.image_transform,
            image_token=self.image_token_spec.image_token,
        )

    def _pack_rows(self, rows: list[dict], pack_index: int) -> dict[str, Any] | None:
        missing_lengths: list[int] = []
        lengths_from_processor: list[int] | None = None

        packed_text_parts: list[str] = []
        packed_images: list[Image.Image] = []
        sample_lengths: list[int] = []
        kept_rows: list[dict] = []

        for row_index, row in enumerate(rows):
            if "_mm_text" not in row or "images" not in row:
                raise KeyError(
                    f"MultiModalPretrainDataCollator: packed row {row_index} "
                    "is missing '_mm_text' or 'images'."
                )
            raw_sources = self._raw_image_sources(row.get("images"), row_index)
            prepared = self._prepare_row_text_and_images(
                prepare_text_for_packed_boundary(
                    row,
                    self.tokenizer,
                    is_first=len(packed_text_parts) == 0,
                    add_eos_token=self.add_eos_token,
                ),
                raw_sources,
                row_index=row_index,
            )
            if prepared is None:
                LOG.warning(
                    "Packed row %d.%d: image preparation failed; dropping row.",
                    pack_index,
                    row_index,
                )
                continue
            prepared_text, loaded = prepared

            length = self._coerce_length(row.get("length"))
            if length is None:
                missing_lengths.append(len(kept_rows))

            kept_rows.append(row)
            sample_lengths.append(length or 0)
            packed_images.extend(loaded)
            packed_text_parts.append(prepared_text)

        if not kept_rows:
            return None

        if missing_lengths:
            lengths_from_processor = self._fallback_row_lengths(kept_rows)
            for idx in missing_lengths:
                sample_lengths[idx] = lengths_from_processor[idx]

        return {
            "text": "".join(packed_text_parts),
            "images": packed_images,
            "sample_lengths": sample_lengths,
        }

    def _prepacked_row(self, row: dict, pack_index: int) -> dict[str, Any] | None:
        if "_mm_text" not in row or "images" not in row:
            raise KeyError(
                f"MultiModalPretrainDataCollator: packed row {pack_index} "
                "is missing '_mm_text' or 'images'."
            )
        sample_lengths = self._coerce_sample_lengths(row.get("_mm_sample_lengths"))
        raw_sources = self._raw_image_sources(row.get("images"), row_index=pack_index)
        prepared = self._prepare_row_text_and_images(
            row["_mm_text"],
            raw_sources,
            row_index=pack_index,
        )
        if prepared is None:
            LOG.warning(
                "Packed row %d: image preparation failed; dropping packed row.",
                pack_index,
            )
            return None
        prepared_text, loaded = prepared
        return {
            "text": prepared_text,
            "images": loaded,
            "sample_lengths": sample_lengths,
        }

    def _packed_examples(self, examples: list[Any]) -> list[dict[str, Any]]:
        packed: list[dict[str, Any]] = []
        if isinstance(examples[0], list):
            for pack_index, rows in enumerate(examples):
                pack = self._pack_rows(rows, pack_index)
                if pack is not None:
                    packed.append(pack)
        elif "_mm_sample_lengths" in examples[0]:
            for pack_index, row in enumerate(examples):
                pack = self._prepacked_row(row, pack_index)
                if pack is not None:
                    packed.append(pack)
        else:
            pack = self._pack_rows(examples, pack_index=0)
            if pack is not None:
                packed.append(pack)
        return packed

    def _add_packing_masks(
        self,
        batch: dict[str, Any],
        sample_lengths: list[list[int]],
    ) -> None:
        input_ids: Tensor = batch["input_ids"]
        max_len = input_ids.shape[-1]
        if "attention_mask" in batch:
            actual_lengths = [
                int(v)
                for v in batch["attention_mask"].sum(dim=-1).detach().cpu().tolist()
            ]
        else:
            actual_lengths = [max_len for _ in sample_lengths]

        position_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        for row_index, (lengths, actual_len) in enumerate(
            zip(sample_lengths, actual_lengths, strict=True)
        ):
            expected_len = sum(lengths)
            if expected_len != actual_len:
                raise RuntimeError(
                    "Packed multimodal length mismatch on row "
                    f"{row_index}: sample lengths sum to {expected_len}, "
                    f"processor produced {actual_len}. Rebuild the dataset "
                    "cache with this processor/tokenizer pair."
                )
            pos_row: list[int] = []
            mask_row: list[int] = []
            for segment_id, length in enumerate(lengths, start=1):
                pos_row.extend(range(length))
                mask_row.extend([segment_id] * length)
            pad_len = max_len - actual_len
            pad_positions = [0] * pad_len
            pad_mask = [0] * pad_len
            if padding_side == "left":
                pos_row = pad_positions + pos_row
                mask_row = pad_mask + mask_row
            else:
                pos_row.extend(pad_positions)
                mask_row.extend(pad_mask)
            position_ids.append(pos_row)
            attention_mask.append(mask_row)

        batch["position_ids"] = torch.tensor(
            position_ids, dtype=torch.long, device=input_ids.device
        )
        batch["attention_mask"] = torch.tensor(
            attention_mask, dtype=torch.long, device=input_ids.device
        )

    def _torch_call_packed(self, examples: list[dict]) -> dict[str, Any]:
        packed = self._packed_examples(examples)
        if not packed:
            raise RuntimeError(
                "All rows in the packed batch were dropped due to image load "
                "failures. Check dataset integrity."
            )

        texts = [item["text"] for item in packed]
        images = [item["images"] for item in packed]
        sample_lengths = [item["sample_lengths"] for item in packed]

        if all(len(im) == 0 for im in images):
            tok_kwargs: dict[str, Any] = {
                "text": texts,
                "return_tensors": self.return_tensors,
                "padding": self.padding,
            }
            if self.pad_to_multiple_of is not None:
                tok_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
            batch = self.tokenizer(**tok_kwargs)
        else:
            proc_kwargs: dict[str, Any] = {
                "text": texts,
                "images": images,
                "return_tensors": self.return_tensors,
                "padding": self.padding,
            }
            if self.pad_to_multiple_of is not None:
                proc_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
            batch = self.processor(**proc_kwargs)

        input_ids_len = batch["input_ids"].shape[-1]
        if self.max_length is not None and input_ids_len > self.max_length:
            LOG.warning(
                "Packed batch input_ids length %d exceeds configured capacity %d.",
                input_ids_len,
                self.max_length,
            )

        input_ids: Tensor = batch["input_ids"]
        labels = input_ids.clone()
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for tid in self._image_family_token_ids:
            labels[labels == tid] = -100

        batch["labels"] = labels
        self._add_packing_masks(batch, sample_lengths)
        return batch

    def torch_call(self, examples: list[dict]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Empty batch passed to MultiModalPretrainDataCollator.")
        if self.sample_packing:
            return self._torch_call_packed(examples)

        texts: list[str] = []
        images: list[list[Image.Image]] = []
        for i, ex in enumerate(examples):
            if "_mm_text" not in ex or "images" not in ex:
                raise KeyError(
                    f"MultiModalPretrainDataCollator: row {i} is missing "
                    f"'_mm_text' or 'images'. Did you wire the multimodal CPT "
                    f"encoder (encode_streaming_multimodal)?"
                )
            mm_text = ex["_mm_text"]
            if not isinstance(mm_text, str):
                raise TypeError(
                    f"Row {i}: `_mm_text` must be str, got "
                    f"{type(mm_text).__name__}. Check dataset encoding "
                    f"(Parquet BINARY columns may surface as bytes)."
                )
            raw_sources = self._raw_image_sources(ex["images"], row_index=i)
            # Processor re-tokenizes below, discarding the encoder's EOS — re-append.
            if self.add_eos_token and self.tokenizer.eos_token:
                if not mm_text.endswith(self.tokenizer.eos_token):
                    mm_text = mm_text + self.tokenizer.eos_token
            prepared = self._prepare_row_text_and_images(
                mm_text,
                raw_sources,
                row_index=i,
            )
            if prepared is None:
                LOG.warning(
                    "Row %d: image preparation failed; dropping row.",
                    i,
                )
                continue
            mm_text, loaded = prepared
            texts.append(mm_text)
            images.append(loaded)

        if not texts:
            raise RuntimeError(
                "All rows in the batch were dropped due to image load "
                "failures. Check dataset integrity."
            )

        # All-text batch: bypass the processor and tokenize directly.
        if all(len(im) == 0 for im in images):
            LOG.debug(
                "MultiModalPretrainDataCollator: all-text batch (%d rows); "
                "using tokenizer-only fallback (no pixel_values).",
                len(texts),
            )
            tok_kwargs: dict[str, Any] = {
                "text": texts,
                "return_tensors": self.return_tensors,
                "padding": self.padding,
            }
            if self.pad_to_multiple_of is not None:
                tok_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
            batch = self.tokenizer(**tok_kwargs)
            tok_input_ids: Tensor = batch["input_ids"]
            tok_labels = tok_input_ids.clone()
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                tok_labels[tok_labels == pad_id] = -100
            for tid in self._image_family_token_ids:
                tok_labels[tok_labels == tid] = -100
            batch["labels"] = tok_labels
            return batch

        # No truncation: it chops input_ids mid-placeholder while pixel_values
        # keep every image — silent text/pixel mismatch. We warn post-hoc instead.
        proc_kwargs: dict[str, Any] = {
            "text": texts,
            "images": images,
            "return_tensors": self.return_tensors,
            "padding": self.padding,
        }
        if self.pad_to_multiple_of is not None:
            proc_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
        try:
            batch = self.processor(**proc_kwargs)
        except Exception as exc:
            # Pinpoint the bad row; bail to "inconclusive" if retry raises a different class.
            LOG.warning(
                "MultiModalPretrainDataCollator: processor failed on a batch "
                "of %d rows (%s); retrying each row individually to locate "
                "the offender. This adds up to %d extra processor calls.",
                len(texts),
                type(exc).__name__,
                len(texts),
            )
            offender_idx: Optional[int] = None
            retry_ok = True
            retry_kwargs: dict[str, Any] = {
                "return_tensors": self.return_tensors,
                "padding": self.padding,
            }
            if self.pad_to_multiple_of is not None:
                retry_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
            for i, (t, imgs) in enumerate(zip(texts, images, strict=True)):
                try:
                    if len(imgs) == 0:
                        # Some processors reject `images=[[]]` — would mislabel
                        # text-only rows as the offender.
                        self.tokenizer(text=[t], **retry_kwargs)
                    else:
                        self.processor(text=[t], images=[imgs], **retry_kwargs)
                except Exception as retry_exc:
                    if isinstance(retry_exc, type(exc)) or isinstance(
                        exc, type(retry_exc)
                    ):
                        offender_idx = i
                    else:
                        retry_ok = False
                    break
            if offender_idx is not None:
                location = f"row {offender_idx}"
            elif retry_ok:
                location = (
                    f"batch of {len(texts)} rows "
                    f"(individual rows all succeed; see __cause__ for details)"
                )
            else:
                location = f"batch of {len(texts)} rows (retry inconclusive)"
            raise RuntimeError(
                f"MultiModalPretrainDataCollator: processor call failed on "
                f"{location} ({type(exc).__name__}: {exc}). Common causes: "
                f"placeholder token absent from the row's text, image count "
                f"mismatch, or an unsupported processor class."
            ) from exc

        input_ids_len = batch["input_ids"].shape[-1]
        if self.max_length is not None and input_ids_len > self.max_length:
            LOG.warning(
                "Batch input_ids length %d exceeds configured sequence_len %d "
                "(image placeholder expansion). Raise sequence_len if this "
                "fires repeatedly.",
                input_ids_len,
                self.max_length,
            )

        input_ids: Tensor = batch["input_ids"]
        labels = input_ids.clone()

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Without this, image-family ids dominate loss and blow it up ~10x.
        for tid in self._image_family_token_ids:
            labels[labels == tid] = -100

        batch["labels"] = labels
        return batch
