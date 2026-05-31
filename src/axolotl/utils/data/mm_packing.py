"""Multimodal sample-packing metadata and packing helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from hashlib import md5
from pathlib import Path
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.image_utils import load_image

from axolotl.prompt_strategies.multimodal_pretrain import append_eos_for_processor
from axolotl.utils.data.mm_image import resize_image_for_processor
from axolotl.utils.data.mm_tiling import (
    ImageTileCache,
    ImageTilingConfig,
    _tiling_policy_payload,
    prepare_tiled_text_and_images,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)
DEFAULT_METADATA_PROCESSOR_BATCH_SIZE = 32


@dataclass(frozen=True)
class MultimodalPackingMetadata:
    length: int
    visual_tokens: int
    image_count: int
    visual_signature: str


class MultimodalPackingMetadataCache:
    """Small LRU RAM cache with optional persistent JSON records."""

    def __init__(self, cache_path: str | None = None, ram_budget_mb: int | None = None):
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self.ram_budget_bytes = max(0, int(ram_budget_mb or 0) * 1024 * 1024)
        self._ram: OrderedDict[str, tuple[MultimodalPackingMetadata, int]] = (
            OrderedDict()
        )
        self._ram_bytes = 0
        if self.cache_path:
            self.cache_path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> MultimodalPackingMetadata | None:
        if key in self._ram:
            value, size = self._ram.pop(key)
            self._ram[key] = (value, size)
            return value
        if not self.cache_path:
            return None
        path = self._path_for_key(key)
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = MultimodalPackingMetadata(**json.load(handle))
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001
            LOG.debug("failed reading multimodal packing metadata cache: %s", exc)
            return None
        self.set(key, value, persist=False)
        return value

    def set(
        self,
        key: str,
        value: MultimodalPackingMetadata,
        *,
        persist: bool = True,
    ) -> None:
        payload = json.dumps(asdict(value), sort_keys=True, separators=(",", ":"))
        size = len(payload)
        if self.ram_budget_bytes:
            if key in self._ram:
                _, old_size = self._ram.pop(key)
                self._ram_bytes -= old_size
            self._ram[key] = (value, size)
            self._ram_bytes += size
            while self._ram_bytes > self.ram_budget_bytes and self._ram:
                _, (_, evicted_size) = self._ram.popitem(last=False)
                self._ram_bytes -= evicted_size

        if not persist or not self.cache_path:
            return
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=str(path.parent),
                text=True,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, path)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("failed writing multimodal packing metadata cache: %s", exc)
            try:
                os.unlink(tmp_name)  # type: ignore[possibly-undefined]
            except Exception:  # noqa: BLE001
                pass

    def _path_for_key(self, key: str) -> Path:
        if self.cache_path is None:
            raise RuntimeError("cache_path is not set")
        return self.cache_path / key[:2] / key[2:4] / f"{key}.json"


def multimodal_metadata_cache_key(
    *,
    text: str,
    images: list[Any],
    tokenizer: PreTrainedTokenizerBase,
    processor: ProcessorMixin,
    image_base_dir: str | None,
    image_token_id: int,
    image_token: str | None = None,
    add_eos_token: bool = True,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Any | None = None,
    image_resize_buckets: list[tuple[int, int]] | None = None,
    image_resize_no_upscale: bool = False,
    image_resize_pad_color: Any | None = None,
    image_tiling_config: ImageTilingConfig | None = None,
) -> str:
    payload = {
        "text": text,
        "images": [
            _image_fingerprint(src, image_base_dir=image_base_dir) for src in images
        ],
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "processor": getattr(processor, "name_or_path", None),
        "processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
        "image_processor_class": _image_processor_class(processor),
        "image_processor_size": _jsonable(
            getattr(getattr(processor, "image_processor", None), "size", None)
        ),
        "image_token_id": image_token_id,
        "image_token": image_token,
        "eos_token": getattr(tokenizer, "eos_token", None),
        "add_eos_token": add_eos_token,
        "image_size": _jsonable(image_size),
        "image_resize_algorithm": _jsonable(image_resize_algorithm),
        "image_resize_buckets": _jsonable(image_resize_buckets),
        "image_resize_no_upscale": bool(image_resize_no_upscale),
        "image_resize_pad_color": _jsonable(image_resize_pad_color),
        "image_tiling_config": _jsonable(
            _tiling_policy_payload(image_tiling_config)
            if image_tiling_config is not None
            else None
        ),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()


def compute_multimodal_packing_metadata(
    texts: list[str],
    imgs_list: list[list[Any]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    processor: ProcessorMixin,
    image_token_id: int,
    image_token: str | None = None,
    image_base_dir: str | None = None,
    add_eos_token: bool = True,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Any | None = None,
    image_resize_buckets: list[tuple[int, int]] | None = None,
    image_resize_no_upscale: bool = False,
    image_resize_pad_color: Any | None = None,
    image_tiling_config: ImageTilingConfig | None = None,
    cache: MultimodalPackingMetadataCache | None = None,
) -> list[MultimodalPackingMetadata]:
    if len(texts) != len(imgs_list):
        raise ValueError(
            "compute_multimodal_packing_metadata: text/image row count mismatch "
            f"({len(texts)} text row(s), {len(imgs_list)} image row(s))."
        )

    keys = [
        multimodal_metadata_cache_key(
            text=text,
            images=list(images or []),
            tokenizer=tokenizer,
            processor=processor,
            image_base_dir=image_base_dir,
            image_token_id=image_token_id,
            image_token=image_token,
            add_eos_token=add_eos_token,
            image_size=image_size,
            image_resize_algorithm=image_resize_algorithm,
            image_resize_buckets=image_resize_buckets,
            image_resize_no_upscale=image_resize_no_upscale,
            image_resize_pad_color=image_resize_pad_color,
            image_tiling_config=image_tiling_config,
        )
        for text, images in zip(texts, imgs_list, strict=True)
    ]
    output: list[MultimodalPackingMetadata | None] = [None] * len(texts)
    misses: list[int] = []
    if cache is not None:
        for idx, key in enumerate(keys):
            cached = cache.get(key)
            if cached is None:
                misses.append(idx)
            else:
                output[idx] = cached
    else:
        misses = list(range(len(texts)))

    if misses:
        batch_size = _metadata_processor_batch_size()
        for start in range(0, len(misses), batch_size):
            chunk = misses[start : start + batch_size]
            computed = _compute_uncached_metadata(
                [texts[idx] for idx in chunk],
                [imgs_list[idx] for idx in chunk],
                tokenizer=tokenizer,
                processor=processor,
                image_token_id=image_token_id,
                image_token=image_token,
                image_base_dir=image_base_dir,
                add_eos_token=add_eos_token,
                image_size=image_size,
                image_resize_algorithm=image_resize_algorithm,
                image_resize_buckets=image_resize_buckets,
                image_resize_no_upscale=image_resize_no_upscale,
                image_resize_pad_color=image_resize_pad_color,
                image_tiling_config=image_tiling_config,
            )
            for idx, value in zip(chunk, computed, strict=True):
                output[idx] = value
                if cache is not None:
                    cache.set(keys[idx], value)

    if any(value is None for value in output):
        raise RuntimeError("Failed to compute multimodal packing metadata.")
    return [value for value in output if value is not None]


def _metadata_processor_batch_size() -> int:
    raw_value = os.environ.get("AXOLOTL_MM_PACKING_METADATA_BATCH_SIZE")
    if raw_value is None:
        return DEFAULT_METADATA_PROCESSOR_BATCH_SIZE
    try:
        return max(1, int(raw_value))
    except ValueError:
        LOG.warning(
            "Ignoring invalid AXOLOTL_MM_PACKING_METADATA_BATCH_SIZE=%r",
            raw_value,
        )
        return DEFAULT_METADATA_PROCESSOR_BATCH_SIZE


def pack_2d_first_fit_decreasing(
    token_lengths: list[int],
    visual_lengths: list[int],
    *,
    token_capacity: int,
    visual_capacity: int,
) -> list[list[int]]:
    if len(token_lengths) != len(visual_lengths):
        raise ValueError("token_lengths and visual_lengths must have the same length")
    if token_capacity <= 0 or visual_capacity <= 0:
        raise ValueError(
            "pack_2d_first_fit_decreasing requires positive capacities; got "
            f"token_capacity={token_capacity}, visual_capacity={visual_capacity}."
        )
    order = sorted(
        range(len(token_lengths)),
        key=lambda idx: max(
            token_lengths[idx] / token_capacity,
            visual_lengths[idx] / visual_capacity,
        ),
        reverse=True,
    )
    bins: list[list[int]] = []
    used: list[list[int]] = []
    for idx in order:
        token_len = int(token_lengths[idx])
        visual_len = int(visual_lengths[idx])
        if token_len > token_capacity:
            raise ValueError(
                "Multimodal packed streaming row exceeds token packing capacity "
                f"{token_capacity}: row length={token_len}."
            )
        if visual_len > visual_capacity:
            raise ValueError(
                "Multimodal packed streaming row exceeds visual packing capacity "
                f"{visual_capacity}: row visual tokens={visual_len}."
            )
        for bin_idx, bin_used in enumerate(used):
            if (
                bin_used[0] + token_len <= token_capacity
                and bin_used[1] + visual_len <= visual_capacity
            ):
                bins[bin_idx].append(idx)
                bin_used[0] += token_len
                bin_used[1] += visual_len
                break
        else:
            bins.append([idx])
            used.append([token_len, visual_len])
    return bins


def _compute_uncached_metadata(
    texts: list[str],
    imgs_list: list[list[Any]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    processor: ProcessorMixin,
    image_token_id: int,
    image_token: str | None,
    image_base_dir: str | None,
    add_eos_token: bool,
    image_size: int | tuple[int, int] | None,
    image_resize_algorithm: Any | None,
    image_resize_buckets: list[tuple[int, int]] | None,
    image_resize_no_upscale: bool,
    image_resize_pad_color: Any | None,
    image_tiling_config: ImageTilingConfig | None,
) -> list[MultimodalPackingMetadata]:
    if image_tiling_config is not None:
        tile_cache = ImageTileCache(image_tiling_config.cache_path)
        prepared = [
            prepare_tiled_text_and_images(
                text,
                list(images or []),
                image_token=image_token or _processor_image_token(processor),
                tiling_config=image_tiling_config,
                image_base_dir=image_base_dir,
                resize_algorithm=image_resize_algorithm,
                cache=tile_cache,
            )
            for text, images in zip(texts, imgs_list, strict=True)
        ]
        texts = [text for text, _images in prepared]
        imgs_list = [images for _text, images in prepared]
        image_size = None
        image_resize_buckets = None
        image_resize_no_upscale = False
        image_resize_pad_color = None

    processor_texts = [
        append_eos_for_processor(text, tokenizer, add_eos_token=add_eos_token)
        for text in texts
    ]
    if all(len(images or []) == 0 for images in imgs_list):
        encoded = tokenizer(text=processor_texts, return_tensors=None, padding=False)
        return _metadata_from_processor_output(
            encoded,
            [0] * len(texts),
            image_token_id=image_token_id,
        )

    loaded_images = _load_images(
        imgs_list,
        image_base_dir,
        image_size=image_size,
        image_resize_algorithm=image_resize_algorithm,
        image_resize_buckets=image_resize_buckets,
        image_resize_no_upscale=image_resize_no_upscale,
        image_resize_pad_color=image_resize_pad_color,
    )
    image_counts = [len(row) for row in loaded_images]
    try:
        encoded = processor(
            text=processor_texts,
            images=loaded_images,
            return_tensors=None,
            padding=False,
        )
        return _metadata_from_processor_output(
            encoded,
            image_counts,
            image_token_id=image_token_id,
        )
    except Exception:
        metadata: list[MultimodalPackingMetadata] = []
        for text, images in zip(processor_texts, loaded_images, strict=True):
            if images:
                encoded = processor(
                    text=[text],
                    images=[images],
                    return_tensors=None,
                    padding=False,
                )
                image_count = len(images)
            else:
                encoded = tokenizer(text=[text], return_tensors=None, padding=False)
                image_count = 0
            metadata.extend(
                _metadata_from_processor_output(
                    encoded,
                    [image_count],
                    image_token_id=image_token_id,
                )
            )
        return metadata


def _metadata_from_processor_output(
    encoded: Any,
    image_counts: list[int],
    *,
    image_token_id: int,
) -> list[MultimodalPackingMetadata]:
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    grids = _grid_rows(encoded.get("image_grid_thw"))
    grid_idx = 0
    metadata: list[MultimodalPackingMetadata] = []
    for ids, image_count in zip(input_ids, image_counts, strict=True):
        row_grids = grids[grid_idx : grid_idx + image_count]
        grid_idx += image_count
        placeholder_tokens = sum(
            1 for token_id in ids if int(token_id) == image_token_id
        )
        grid_tokens = sum(t * h * w for t, h, w in row_grids)
        visual_tokens = placeholder_tokens or grid_tokens or image_count
        signature = _visual_signature(row_grids, image_count, visual_tokens)
        metadata.append(
            MultimodalPackingMetadata(
                length=len(ids),
                visual_tokens=visual_tokens,
                image_count=image_count,
                visual_signature=signature,
            )
        )
    return metadata


def _load_images(
    imgs_list: list[list[Any]],
    image_base_dir: str | None,
    *,
    image_size: int | tuple[int, int] | None,
    image_resize_algorithm: Any | None,
    image_resize_buckets: list[tuple[int, int]] | None,
    image_resize_no_upscale: bool,
    image_resize_pad_color: Any | None,
) -> list[list]:
    loaded: list[list] = []
    for row in imgs_list:
        loaded_row = []
        for image in row or []:
            # Always go through load_image (even for PIL) to match the runtime
            # collator's L/RGBA->RGB conversion; otherwise estimated length can
            # diverge from the real tokenized length for non-RGB inputs.
            raw_image = load_image(_resolve_image_source(image, image_base_dir))
            loaded_row.append(
                resize_image_for_processor(
                    raw_image,
                    image_size,
                    image_resize_algorithm,
                    image_resize_buckets,
                    image_resize_no_upscale,
                    image_resize_pad_color,
                )
            )
        loaded.append(loaded_row)
    return loaded


def _processor_image_token(processor: ProcessorMixin) -> str:
    token = getattr(processor, "image_token", None)
    if token:
        return token
    tokenizer = getattr(processor, "tokenizer", None)
    for candidate in ("<|image_pad|>", "<image>", "<|image|>", "<image_soft_token>"):
        if tokenizer is None:
            continue
        try:
            token_id = tokenizer.convert_tokens_to_ids(candidate)
        except (KeyError, TypeError, ValueError) as exc:
            LOG.debug("convert_tokens_to_ids failed for %r: %s", candidate, exc)
            continue
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is not None and token_id != unk_id:
            return candidate
    return "<image>"


def _resolve_image_source(src: Any, image_base_dir: str | None) -> Any:
    if (
        image_base_dir
        and isinstance(src, str)
        and not os.path.isabs(src)
        and "://" not in src
    ):
        return os.path.join(image_base_dir, src)
    return src


def _image_fingerprint(src: Any, *, image_base_dir: str | None) -> dict[str, Any]:
    resolved = _resolve_image_source(src, image_base_dir)
    fp: dict[str, Any] = {
        "type": type(src).__name__,
        "source": str(resolved),
    }
    if isinstance(resolved, str) and "://" not in resolved:
        try:
            stat = os.stat(resolved)
            fp["size"] = stat.st_size
            fp["mtime_ns"] = stat.st_mtime_ns
        except OSError:
            pass
    return fp


def _grid_rows(value: Any) -> list[tuple[int, int, int]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [tuple(int(item) for item in row) for row in value]


def _visual_signature(
    grids: list[tuple[int, int, int]],
    image_count: int,
    visual_tokens: int,
) -> str:
    if grids:
        return "+".join("x".join(str(item) for item in grid) for grid in grids)
    return f"images={image_count}:tokens={visual_tokens}"


def _image_processor_class(processor: ProcessorMixin) -> str | None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return None
    return f"{type(image_processor).__module__}.{type(image_processor).__name__}"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
