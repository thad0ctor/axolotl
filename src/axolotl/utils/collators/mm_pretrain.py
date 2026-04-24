"""Collator for multimodal CPT — re-runs processor on the batch, masks image tokens."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from PIL import Image
from torch import Tensor
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy

from axolotl.prompt_strategies.multimodal_pretrain import (
    ImageTokenSpec,
    check_processor_compatibility,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Raised by PIL (elevated to ValueError below) when a decoded image exceeds
# this pixel count. 50M is ~7070×7070 — generous for document crops, but
# blocks gigapixel decompression-bomb inputs well before they blow up RAM.
_DEFAULT_MAX_IMAGE_PIXELS = 50_000_000

# Default cap on images per row — defense in depth against malicious datasets
# containing thousands of placeholders in a single row. Override via config.
_DEFAULT_MAX_IMAGES_PER_ROW = 32


@dataclass
class MultiModalPretrainDataCollator(DataCollatorMixin):
    """Collator for raw image+text CPT (no chat template)."""

    tokenizer: PreTrainedTokenizerBase
    processor: ProcessorMixin
    image_token_spec: ImageTokenSpec
    image_base_dir: Optional[str] = None
    return_tensors: str = "pt"
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    # Cap the token length the processor produces — without this a few images
    # can silently produce 10k+ tokens of placeholders and OOM the model.
    max_length: Optional[int] = None
    # Allow bad-image rows to be skipped instead of crashing the run. Off by
    # default — fail loud unless the user explicitly opts in.
    skip_bad_images: bool = False
    # Decompression-bomb guard. PIL raises DecompressionBombWarning above
    # this; we elevate it to a hard error.
    max_image_pixels: int = _DEFAULT_MAX_IMAGE_PIXELS
    max_images_per_row: int = _DEFAULT_MAX_IMAGES_PER_ROW

    # Populated in __post_init__. Kept on the instance so workers can mask
    # without re-probing the tokenizer.
    _image_family_token_ids: set[int] = field(init=False, default_factory=set)
    _base_dir_real: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        check_processor_compatibility(self.processor)
        self._image_family_token_ids = set(
            self.image_token_spec.image_family_token_ids
        )
        if self.image_base_dir is not None:
            self._base_dir_real = os.path.realpath(self.image_base_dir)

    # --- helpers ---------------------------------------------------------

    def _resolve_image_path(self, p: str) -> str:
        """Canonicalize path and enforce `image_base_dir` containment if set."""
        if not isinstance(p, str):
            raise ValueError(
                f"Image path must be str, got {type(p).__name__}."
            )
        # Embedded NUL bytes are a classic filesystem-trick vector; most
        # syscalls stop at the NUL but some libc/tools don't.
        if "\x00" in p:
            raise ValueError("Image path contains embedded NUL byte.")
        # Reject non-local schemes explicitly (v1 = local files only).
        # Scheme-check is case-insensitive (HTTP:// and ftp:// both fail).
        # UNC paths on Windows (`\\host\share\...`) are also non-local.
        p_lower = p.lower()
        if p_lower.startswith(
            ("http://", "https://", "ftp://", "ftps://", "file://", "data:")
        ) or p.startswith(("\\\\", "//")):
            raise ValueError(
                f"Non-local image path scheme is not supported in v1 "
                f"multimodal CPT (got {p!r})."
            )
        if self._base_dir_real is not None:
            if os.path.isabs(p):
                raise ValueError(
                    f"Absolute image path {p!r} is rejected when "
                    f"`image_base_dir` is configured. All image paths must be "
                    f"relative to the configured base directory."
                )
            resolved = os.path.realpath(os.path.join(self._base_dir_real, p))
            # Containment check (post-symlink). Trailing os.sep prevents
            # "/base/../basemalicious" escape.
            if not (
                resolved == self._base_dir_real
                or resolved.startswith(self._base_dir_real + os.sep)
            ):
                raise ValueError(
                    f"Image path {p!r} resolves outside `image_base_dir` "
                    f"after symlink resolution. Refusing to load."
                )
            return resolved
        # No base dir → trust absolute paths as-is but still canonicalize.
        return os.path.realpath(p) if os.path.isabs(p) else p

    def _open_image_hardened(self, resolved: str) -> Image.Image:
        """Open, check pixel+frame caps, load, return RGB. fd-safe via `with`."""
        # O_NOFOLLOW refuses a terminal symlink at the final path component.
        # `realpath` has already resolved any symlinks on the path, so this
        # only catches the narrow TOCTOU window where a symlink appears AT
        # the resolved location between `realpath` and `os.open`. It does
        # NOT protect against ancestor-directory symlink swaps — for those,
        # `image_base_dir` itself is assumed to be under admin control.
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(resolved, os.O_RDONLY | nofollow)
        except OSError as exc:
            raise ValueError(
                f"Cannot open image (os.open failed: {type(exc).__name__})."
            ) from exc
        # Wrap fd in a file object so PIL's `Image.open` gets the read/seek
        # interface it expects. `os.fdopen` transfers ownership — closing
        # the file object closes the fd.
        file_obj = os.fdopen(fd, "rb")
        try:
            with Image.open(file_obj) as src:
                w, h = src.size
                if w * h > self.max_image_pixels:
                    raise ValueError(
                        f"Image pixels ({w}×{h}) exceed "
                        f"max_image_pixels ({self.max_image_pixels})."
                    )
                # GIF/TIFF/WebP multi-frame bomb guard: decoding frame 0
                # is cheap, but an attacker can stuff 10k frames. We only
                # need frame 0 for static VLM input.
                n_frames = getattr(src, "n_frames", 1)
                if n_frames > 1:
                    raise ValueError(
                        f"Multi-frame images are not supported "
                        f"(got {n_frames} frames)."
                    )
                img = src.convert("RGB")
                img.load()
                return img
        finally:
            # Image.open's context manager closes `src`, which also closes
            # `file_obj` in recent Pillow — but we defensively close here
            # to cover the error-before-with-entry case.
            if not file_obj.closed:
                file_obj.close()

    def _load_images_for_row(
        self, paths: list[str], row_index: int
    ) -> list[Image.Image]:
        if len(paths) > self.max_images_per_row:
            raise ValueError(
                f"Row {row_index}: {len(paths)} images exceeds "
                f"`max_images_per_row={self.max_images_per_row}`. Split the "
                f"row or raise the cap if this is expected."
            )
        out: list[Image.Image] = []
        for raw in paths:
            try:
                resolved = self._resolve_image_path(raw)
                img = self._open_image_hardened(resolved)
            except Exception as exc:
                # Only leak the basename to the top-level log — full resolved
                # paths can contain cluster layout / user dirs that end up in
                # third-party log aggregators. Full path stays on the DEBUG
                # stream and in the chained exception.
                basename = os.path.basename(str(raw))
                msg = (
                    f"Row {row_index}: failed to load image {basename!r} "
                    f"({type(exc).__name__})"
                )
                LOG.debug("failed image full path: %r; error: %s", raw, exc)
                if self.skip_bad_images:
                    LOG.warning("%s — skipping", msg)
                    continue
                raise RuntimeError(msg) from exc
            out.append(img)
        return out

    # --- DataCollatorMixin -----------------------------------------------

    def torch_call(self, examples: list[dict]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Empty batch passed to MultiModalPretrainDataCollator.")

        texts: list[str] = []
        images: list[list[Image.Image]] = []
        for i, ex in enumerate(examples):
            if "_mm_text" not in ex or "images" not in ex:
                raise KeyError(
                    f"MultiModalPretrainDataCollator: row {i} is missing "
                    f"'_mm_text' or 'images'. Did you wire the multimodal CPT "
                    f"encoder (encode_streaming_multimodal or "
                    f"MultimodalPretrainTokenizationStrategy)?"
                )
            mm_text = ex["_mm_text"]
            if not isinstance(mm_text, str):
                raise TypeError(
                    f"Row {i}: `_mm_text` must be str, got "
                    f"{type(mm_text).__name__}. Check dataset encoding "
                    f"(Parquet BINARY columns may surface as bytes)."
                )
            raw = ex["images"]
            if raw is None:
                raw_paths: list[str] = []
            elif isinstance(raw, (list, tuple)):
                raw_paths = list(raw)
            else:
                raise TypeError(
                    f"Row {i}: `images` must be a list (or None), got "
                    f"{type(raw).__name__}."
                )
            # Enforce str type at the boundary — the dataset can hold dicts
            # or None; we want a clear error, not a confusing PIL failure.
            for j, rp in enumerate(raw_paths):
                if not isinstance(rp, str):
                    raise TypeError(
                        f"Row {i}, image {j}: path must be str, got "
                        f"{type(rp).__name__}."
                    )
            texts.append(mm_text)
            loaded = self._load_images_for_row(raw_paths, row_index=i)
            if self.skip_bad_images and len(loaded) != len(raw_paths):
                # Drop the row entirely rather than leave a placeholder/image
                # count mismatch for the processor (which would silently
                # corrupt alignment on LLaVA/Qwen families).
                LOG.warning(
                    "Row %d: %d/%d images failed to load; dropping row.",
                    i, len(raw_paths) - len(loaded), len(raw_paths),
                )
                texts.pop()
                continue
            images.append(loaded)

        if not texts:
            raise RuntimeError(
                "All rows in the batch were dropped due to image load "
                "failures. Check dataset integrity."
            )

        # Re-tokenize + encode pixels on the whole batch. Each processor
        # knows its own layout (flat [sum_patches, D] for Qwen,
        # [B, tiles, C, H, W] for SmolVLM, [B, C, H, W] for LLaVA/Gemma-3).
        #
        # NOTE: we do NOT pass `truncation=True` here. Truncation would chop
        # `input_ids` mid-placeholder-expansion while `pixel_values` retains
        # every image — producing a silent text/pixel alignment mismatch
        # (round-3 finding). A too-small `sequence_len` instead produces a
        # visible failure at forward time (position-embedding overflow or OOM),
        # which is the safer failure mode. If `max_length` is set, we warn
        # post-hoc when the produced input_ids exceed it.
        proc_kwargs: dict[str, Any] = {
            "text": texts,
            "images": images,
            "return_tensors": self.return_tensors,
            "padding": self.padding,
        }
        try:
            batch = self.processor(**proc_kwargs)
        except Exception as exc:
            # Narrow the error — pinpoint the problematic row by retrying
            # one-by-one. Use `isinstance` instead of exact-type match so a
            # subclass raise in a row still counts as the same failure. If
            # a retry raises a *different* exception class (e.g. OOM that
            # wasn't in the original), we mark the retry inconclusive
            # rather than false-blame a row.
            offender_idx: Optional[int] = None
            retry_ok = True
            for i, (t, imgs) in enumerate(zip(texts, images)):
                try:
                    self.processor(
                        text=[t], images=[imgs],
                        return_tensors=self.return_tensors, padding=self.padding,
                    )
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

        # Post-hoc length warning — informational, not a corruption guard
        # (since we removed truncation there's no silent-corruption path).
        input_ids_len = batch["input_ids"].shape[-1]
        if self.max_length is not None and input_ids_len > self.max_length:
            LOG.warning(
                "Batch input_ids length %d exceeds configured sequence_len %d "
                "(image placeholder expansion). Reduce max_images_per_row or "
                "raise sequence_len if this fires repeatedly.",
                input_ids_len, self.max_length,
            )

        # Build labels from the processor's (re-)tokenized input_ids.
        # CPT trains on all text tokens → start from input_ids.clone().
        input_ids: Tensor = batch["input_ids"]
        labels = input_ids.clone()

        # Mask padding.
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Mask image-family tokens — essential: these ids never correspond to
        # a predicted text token, so including them in the loss dominates
        # gradient signal and blows up training loss ~10× in practice.
        for tid in self._image_family_token_ids:
            labels[labels == tid] = -100

        batch["labels"] = labels
        return batch
