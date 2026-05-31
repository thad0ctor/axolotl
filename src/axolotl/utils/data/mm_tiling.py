"""Multimodal image tiling helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import PIL
from PIL import Image
from transformers.image_utils import load_image

from axolotl.utils.data.mm_image import resize_image_for_processor
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Bump when tiling/cropping/encoding math changes so old cached tiles are ignored.
CACHE_FORMAT_VERSION = 1

# Textual position markers injected before each tile (DocOwl/InternVL style): the
# LLM reconstructs spatial layout from simple text tokens better than from order
# alone. Columns are numbered left->right regardless of reading-order traversal.
GLOBAL_IMAGE_LABEL = "<global_img>"


@dataclass(frozen=True)
class ImageTilingConfig:
    tile_size: int | tuple[int, int]
    grid: tuple[int, int]
    overlap: float = 0.1
    min_area: int = 0
    overview_size: int | tuple[int, int] | None = None
    overview_buckets: list[tuple[int, int]] | None = None
    no_upscale: bool = True
    pad_color: Any | None = None
    reading_order: str = "rtl"
    cache_path: str | None = None
    cache_hash_images: bool = False
    shape_buckets: tuple["ImageTilingBucketConfig", ...] | None = None
    tile_labels: bool = True


@dataclass(frozen=True)
class ImageTilingBucketConfig:
    name: str | None
    grid: tuple[int, int]
    min_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None
    tile_size: int | tuple[int, int] | None = None
    overlap: float | None = None
    min_area: int | None = None
    overview_size: int | tuple[int, int] | None = None
    overview_buckets: tuple[tuple[int, int], ...] | None = None
    no_upscale: bool | None = None
    pad_color: Any | None = None
    reading_order: str | None = None


DEFAULT_OCR_IMAGE_TILING_BUCKETS: tuple[ImageTilingBucketConfig, ...] = (
    ImageTilingBucketConfig(
        name="landscape",
        grid=(3, 2),
        max_aspect_ratio=0.9,
    ),
    ImageTilingBucketConfig(
        name="portrait",
        grid=(2, 3),
        min_aspect_ratio=0.9,
        max_aspect_ratio=1.7,
    ),
    ImageTilingBucketConfig(
        name="tall",
        grid=(2, 4),
        min_aspect_ratio=1.7,
    ),
)


class ImageTileCache:
    """Persistent PNG cache for processor-ready image tiles."""

    def __init__(self, cache_path: str | None):
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        if self.cache_path:
            self.cache_path.mkdir(parents=True, exist_ok=True)

    def get_or_create(
        self,
        source: Any,
        config: ImageTilingConfig,
        *,
        image_base_dir: str | None = None,
        resize_algorithm: Image.Resampling | None = None,
    ) -> tuple[list[Image.Image], list[str] | None]:
        resolved = _resolve_image_source(source, image_base_dir)

        # In-memory PIL or remote sources: tile directly, no persistence.
        if (
            isinstance(resolved, Image.Image)
            or self.cache_path is None
            or not _is_local_path(resolved)
        ):
            image = resolved if isinstance(resolved, Image.Image) else load_image(resolved)
            selected = select_tiling_config_for_image(image.size, config)
            tiles = tile_image_for_processor(
                image, selected, resize_algorithm=resize_algorithm
            )
            return tiles, tile_position_labels(selected, len(tiles))

        path = Path(resolved)
        try:
            with Image.open(path) as probe:
                # load_image (used to cut tiles) applies exif_transpose, so the
                # bucket/grid/labels must be selected from the ORIENTED size.
                image_size = _exif_oriented_size(probe)
        except Exception:
            image = load_image(str(path))
            image_size = image.size
        else:
            image = None

        selected = select_tiling_config_for_image(image_size, config)
        key = image_tile_cache_key(
            str(path),
            selected,
            resize_algorithm=resize_algorithm,
            hash_image=config.cache_hash_images,
        )
        cache_dir = self.cache_path / key[:2] / key[2:4] / key
        manifest_path = cache_dir / "manifest.json"
        cached_paths = _read_cached_manifest(manifest_path, expected_key=key)
        if cached_paths:
            try:
                tiles = [
                    load_image(str(cache_dir / rel_path)) for rel_path in cached_paths
                ]
                return tiles, tile_position_labels(selected, len(tiles))
            except Exception as exc:  # noqa: BLE001
                LOG.debug("failed loading image tile cache entry %s: %s", key, exc)

        if image is None:
            image = load_image(str(path))
        tiles = tile_image_for_processor(
            image,
            selected,
            resize_algorithm=resize_algorithm,
        )
        _write_cached_tiles(cache_dir, manifest_path, tiles, key)
        return tiles, tile_position_labels(selected, len(tiles))


def build_image_tiling_config(
    *,
    enabled: bool | None = False,
    tile_size: int | tuple[int, int] | None = None,
    grid: tuple[int, int] | None = None,
    overlap: float | None = None,
    min_area: int | None = None,
    overview_size: int | tuple[int, int] | None = None,
    overview_buckets: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    no_upscale: bool | None = None,
    pad_color: Any | None = None,
    reading_order: str | None = None,
    cache_path: str | None = None,
    cache_hash_images: bool | None = None,
    shape_buckets: Any | None = None,
    tile_labels: bool | None = None,
) -> ImageTilingConfig | None:
    if not enabled:
        return None
    normalized_shape_buckets = _normalize_shape_buckets(shape_buckets)
    return ImageTilingConfig(
        tile_size=_normalize_size_pair(tile_size or 1024),
        grid=_normalize_grid(grid or (2, 3)),
        overlap=0.1 if overlap is None else float(overlap),
        min_area=0 if min_area is None else int(min_area),
        overview_size=(
            None if overview_size is None else _normalize_size_pair(overview_size)
        ),
        overview_buckets=_normalize_bucket_list(overview_buckets),
        no_upscale=True if no_upscale is None else bool(no_upscale),
        pad_color=pad_color,
        reading_order=reading_order or "rtl",
        cache_path=cache_path,
        cache_hash_images=bool(cache_hash_images),
        shape_buckets=normalized_shape_buckets,
        tile_labels=True if tile_labels is None else bool(tile_labels),
    )


def image_tiling_config_from_cfg(cfg: Any) -> ImageTilingConfig | None:
    def get(key: str, default=None):
        if hasattr(cfg, "get"):
            try:
                return cfg.get(key, default)
            except (AttributeError, KeyError, TypeError):
                pass
        return getattr(cfg, key, default)

    return build_image_tiling_config(
        enabled=bool(get("image_tiling", False)),
        tile_size=get("image_tiling_tile_size"),
        grid=get("image_tiling_grid"),
        overlap=get("image_tiling_overlap"),
        min_area=get("image_tiling_min_area"),
        overview_size=get("image_tiling_overview_size"),
        overview_buckets=get("image_tiling_overview_buckets"),
        no_upscale=get("image_tiling_no_upscale"),
        pad_color=get("image_tiling_pad_color"),
        reading_order=get("image_tiling_reading_order"),
        cache_path=get("image_tiling_cache_path"),
        cache_hash_images=get("image_tiling_cache_hash_images"),
        shape_buckets=get("image_tiling_shape_buckets"),
        tile_labels=get("image_tiling_tile_labels"),
    )


def select_tiling_config_for_image(
    image_size: tuple[int, int],
    config: ImageTilingConfig,
) -> ImageTilingConfig:
    if not config.shape_buckets:
        return config
    width, height = image_size
    aspect_ratio = height / max(1, width)
    for bucket in config.shape_buckets:
        if (
            bucket.min_aspect_ratio is not None
            and aspect_ratio < bucket.min_aspect_ratio
        ):
            continue
        if (
            bucket.max_aspect_ratio is not None
            and aspect_ratio >= bucket.max_aspect_ratio
        ):
            continue
        return _merge_bucket_config(config, bucket)
    return config


def tile_image_for_processor(
    image: Image.Image,
    config: ImageTilingConfig,
    *,
    resize_algorithm: Image.Resampling | None = None,
) -> list[Image.Image]:
    if image.width * image.height < config.min_area:
        # Still canvas to the fixed tile size so small images don't reintroduce
        # variable visual shapes (which would defeat compile-stability).
        return [
            resize_image_for_processor(
                image,
                config.tile_size,
                resize_algorithm,
                None,
                config.no_upscale,
                config.pad_color,
            )
        ]

    output: list[Image.Image] = []
    if config.overview_size is not None or config.overview_buckets:
        output.append(
            resize_image_for_processor(
                image,
                config.overview_size,
                resize_algorithm,
                config.overview_buckets,
                config.no_upscale,
                config.pad_color,
            )
        )

    for box in _tile_boxes(
        image.size, config.grid, config.overlap, config.reading_order
    ):
        crop = image.crop(box)
        output.append(
            resize_image_for_processor(
                crop,
                config.tile_size,
                resize_algorithm,
                None,
                config.no_upscale,
                config.pad_color,
            )
        )
    return output


def tile_position_labels(
    config: ImageTilingConfig, n_tiles: int
) -> list[str] | None:
    """Per-tile textual position markers aligned to tile_image_for_processor order.

    None when labels are disabled, the source wasn't tiled (n_tiles<=1, e.g. the
    min_area short-circuit), or the count doesn't match the grid (don't mislabel).
    """
    if not getattr(config, "tile_labels", True) or n_tiles <= 1:
        return None
    has_overview = config.overview_size is not None or bool(config.overview_buckets)
    cols, rows = config.grid
    if n_tiles != (1 if has_overview else 0) + cols * rows:
        return None
    labels: list[str] = []
    if has_overview:
        labels.append(GLOBAL_IMAGE_LABEL)
    col_order = (
        range(cols - 1, -1, -1) if config.reading_order == "rtl" else range(cols)
    )
    for row in range(rows):
        for col in col_order:
            labels.append(f"<row{row}_col{col}>")
    return labels


def tile_image_source_with_labels(
    source: Any,
    config: ImageTilingConfig,
    *,
    image_base_dir: str | None = None,
    resize_algorithm: Image.Resampling | None = None,
    cache: ImageTileCache | None = None,
) -> tuple[list[Image.Image], list[str] | None]:
    if cache is None:
        cache = ImageTileCache(config.cache_path)
    return cache.get_or_create(
        source,
        config,
        image_base_dir=image_base_dir,
        resize_algorithm=resize_algorithm,
    )


def tile_image_source_for_processor(
    source: Any,
    config: ImageTilingConfig,
    *,
    image_base_dir: str | None = None,
    resize_algorithm: Image.Resampling | None = None,
    cache: ImageTileCache | None = None,
) -> list[Image.Image]:
    tiles, _labels = tile_image_source_with_labels(
        source,
        config,
        image_base_dir=image_base_dir,
        resize_algorithm=resize_algorithm,
        cache=cache,
    )
    return tiles


def prepare_tiled_text_and_images(
    text: str,
    images: list[Any],
    *,
    image_token: str,
    tiling_config: ImageTilingConfig | None,
    image_base_dir: str | None = None,
    resize_algorithm: Image.Resampling | None = None,
    cache: ImageTileCache | None = None,
) -> tuple[str, list[Image.Image] | list[Any]]:
    if tiling_config is None:
        return text, images

    tiled_images: list[Image.Image] = []
    counts: list[int] = []
    labels: list[list[str] | None] = []
    if cache is None:
        cache = ImageTileCache(tiling_config.cache_path)
    for image in images:
        tiles, tile_labels = tile_image_source_with_labels(
            image,
            tiling_config,
            image_base_dir=image_base_dir,
            resize_algorithm=resize_algorithm,
            cache=cache,
        )
        counts.append(len(tiles))
        labels.append(tile_labels)
        tiled_images.extend(tiles)
    return (
        expand_image_placeholders_for_tiling(
            text, image_token=image_token, counts=counts, labels=labels
        ),
        tiled_images,
    )


def expand_image_placeholders_for_tiling(
    text: str,
    *,
    image_token: str,
    counts: list[int] | tuple[int, ...],
    labels: list[list[str] | None] | None = None,
) -> str:
    if not counts or all(count == 1 for count in counts):
        return text

    token = image_token if image_token in text else "<image>"
    cursor = 0
    parts: list[str] = []
    for idx, count in enumerate(counts):
        if count < 1:
            raise ValueError("image tiling placeholder counts must be positive")
        start, end = _find_next_image_placeholder_unit(text, token, cursor)
        unit = text[start:end]
        parts.append(text[cursor:start])
        src_labels = labels[idx] if labels and idx < len(labels) else None
        if src_labels and len(src_labels) == count:
            tile_units = [f"{label}{unit}" for label in src_labels]
        else:
            tile_units = [unit] * count
        parts.append("\n".join(tile_units))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def replace_first_image_placeholder(
    text: str,
    *,
    image_token: str,
    count: int,
) -> str:
    return expand_image_placeholders_for_tiling(
        text,
        image_token=image_token,
        counts=[count],
    )


def image_tile_cache_key(
    source_path: str,
    config: ImageTilingConfig,
    *,
    resize_algorithm: Image.Resampling | None = None,
    hash_image: bool = False,
) -> str:
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "pil_version": PIL.__version__,
        "source": _source_fingerprint(source_path, hash_image=hash_image),
        "policy": _pixel_policy_payload(config),
        "resize_algorithm": str(resize_algorithm),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pixel_policy_payload(config: ImageTilingConfig) -> dict[str, Any]:
    # Tile labels are pure text (injected later), so they don't affect cached PNG
    # pixels — exclude them so toggling labels doesn't invalidate the tile cache.
    payload = _tiling_policy_payload(config)
    payload.pop("tile_labels", None)
    return payload


def _tile_boxes(
    image_size: tuple[int, int],
    grid: tuple[int, int],
    overlap: float,
    reading_order: str,
) -> list[tuple[int, int, int, int]]:
    width, height = image_size
    cols, rows = grid
    if cols <= 0 or rows <= 0:
        raise ValueError("tile grid dimensions must be positive")
    overlap = max(0.0, min(float(overlap), 0.49))
    cell_w = width / cols
    cell_h = height / rows
    pad_w = cell_w * overlap
    pad_h = cell_h * overlap

    boxes: list[tuple[int, int, int, int]] = []
    col_order = (
        list(range(cols - 1, -1, -1)) if reading_order == "rtl" else list(range(cols))
    )
    for row in range(rows):
        for col in col_order:
            left = max(0, int(round(col * cell_w - pad_w)))
            top = max(0, int(round(row * cell_h - pad_h)))
            right = min(width, int(round((col + 1) * cell_w + pad_w)))
            bottom = min(height, int(round((row + 1) * cell_h + pad_h)))
            boxes.append((left, top, right, bottom))
    return boxes


def _find_next_image_placeholder_unit(
    text: str,
    token: str,
    cursor: int,
) -> tuple[int, int]:
    token_pos = text.find(token, cursor)
    if token_pos < 0:
        raise ValueError("Could not find an image placeholder to replace for tiling.")

    start = token_pos
    end = token_pos + len(token)
    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"
    wrapped_start = text.rfind(vision_start, cursor, token_pos + len(token))
    wrapped_end = text.find(vision_end, token_pos)
    if wrapped_start >= cursor and wrapped_end >= 0:
        start = wrapped_start
        end = wrapped_end + len(vision_end)
    return start, end


def _merge_bucket_config(
    config: ImageTilingConfig,
    bucket: ImageTilingBucketConfig,
) -> ImageTilingConfig:
    return replace(
        config,
        tile_size=(
            _normalize_size_pair(bucket.tile_size)
            if bucket.tile_size is not None
            else config.tile_size
        ),
        grid=bucket.grid,
        overlap=config.overlap if bucket.overlap is None else float(bucket.overlap),
        min_area=config.min_area if bucket.min_area is None else int(bucket.min_area),
        overview_size=(
            config.overview_size
            if bucket.overview_size is None
            else _normalize_size_pair(bucket.overview_size)
        ),
        overview_buckets=(
            config.overview_buckets
            if bucket.overview_buckets is None
            else list(bucket.overview_buckets)
        ),
        no_upscale=(
            config.no_upscale if bucket.no_upscale is None else bool(bucket.no_upscale)
        ),
        pad_color=config.pad_color if bucket.pad_color is None else bucket.pad_color,
        reading_order=bucket.reading_order or config.reading_order,
        shape_buckets=None,
    )


def _normalize_shape_buckets(
    value: Any | None,
) -> tuple[ImageTilingBucketConfig, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in {"auto", "ocr_pages"}:
            raise ValueError(
                "image_tiling_shape_buckets must be 'auto', 'ocr_pages', or a list"
            )
        return DEFAULT_OCR_IMAGE_TILING_BUCKETS
    buckets: list[ImageTilingBucketConfig] = []
    for raw_bucket in value:
        if isinstance(raw_bucket, ImageTilingBucketConfig):
            buckets.append(raw_bucket)
            continue
        # Check the class, not the instance: DictDefault (post-validation config
        # buckets) returns None for any missing attr, so instance hasattr(...,
        # "model_dump") is always True and would call None(). Pydantic models
        # define model_dump on the class; DictDefault/dict do not.
        if hasattr(type(raw_bucket), "model_dump"):
            raw_bucket = raw_bucket.model_dump()
        bucket = dict(raw_bucket)
        bucket.setdefault("name", None)  # optional; bare dicts may omit it
        bucket["grid"] = _normalize_grid(bucket["grid"])
        if bucket.get("tile_size") is not None:
            bucket["tile_size"] = _normalize_size_pair(bucket["tile_size"])
        if bucket.get("overview_size") is not None:
            bucket["overview_size"] = _normalize_size_pair(bucket["overview_size"])
        if bucket.get("overview_buckets") is not None:
            bucket["overview_buckets"] = tuple(
                _normalize_bucket_list(bucket["overview_buckets"]) or ()
            )
        buckets.append(ImageTilingBucketConfig(**bucket))
    return tuple(buckets)


def _normalize_size_pair(
    value: int | tuple[int, int] | list[int],
) -> int | tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("image tiling sizes must be positive")
        return value
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "image tiling size entries must be integers or width/height pairs"
        )
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError("image tiling sizes must be positive")
    return (width, height)


def _normalize_grid(value: tuple[int, int] | list[int]) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("image tiling grid must be a two-item columns/rows pair")
    cols, rows = int(value[0]), int(value[1])
    if cols <= 0 or rows <= 0:
        raise ValueError("image tiling grid dimensions must be positive")
    return (cols, rows)


def _normalize_bucket_list(
    value: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None,
) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    converted = []
    for bucket in value:
        if not isinstance(bucket, (list, tuple)) or len(bucket) != 2:
            raise ValueError("image tiling overview buckets must be width/height pairs")
        width, height = int(bucket[0]), int(bucket[1])
        if width <= 0 or height <= 0:
            raise ValueError("image tiling overview buckets must be positive")
        converted.append((width, height))
    return converted


def _resolve_image_source(image: Any, image_base_dir: str | None) -> Any:
    if (
        image_base_dir
        and isinstance(image, str)
        and not os.path.isabs(image)
        and "://" not in image
    ):
        return os.path.join(image_base_dir, image)
    return image


def _is_local_path(source: Any) -> bool:
    return isinstance(source, (str, os.PathLike)) and "://" not in str(source)


def _exif_oriented_size(image: Image.Image) -> tuple[int, int]:
    """Size after exif_transpose (W/H swapped for rotated orientations), computed
    from EXIF without decoding pixels so the cache-hit probe stays cheap."""
    width, height = image.size
    try:
        orientation = image.getexif().get(0x0112)
    except Exception:  # noqa: BLE001
        orientation = None
    if orientation in (5, 6, 7, 8):
        return height, width
    return width, height


def _source_fingerprint(path: str, *, hash_image: bool) -> dict[str, Any]:
    resolved = str(Path(path).expanduser().resolve())
    payload: dict[str, Any] = {"path": resolved}
    try:
        stat = os.stat(resolved)
        payload["size"] = stat.st_size
        payload["mtime_ns"] = stat.st_mtime_ns
    except OSError:
        return payload
    if hash_image:
        payload["sha256"] = _file_sha256(resolved)
    return payload


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tiling_policy_payload(config: ImageTilingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("cache_path", None)
    payload.pop("cache_hash_images", None)
    return payload


def _read_cached_manifest(
    manifest_path: Path, *, expected_key: str | None = None
) -> list[str] | None:
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        LOG.debug("failed reading image tile cache manifest %s: %s", manifest_path, exc)
        return None
    # Reject a manifest written for a different policy/source (defense in depth
    # against a partially-written or hash-colliding entry).
    if expected_key is not None and manifest.get("key") != expected_key:
        return None
    tile_paths = manifest.get("tiles")
    if not isinstance(tile_paths, list):
        return None
    if all((manifest_path.parent / rel_path).exists() for rel_path in tile_paths):
        return [str(rel_path) for rel_path in tile_paths]
    return None


def _write_cached_tiles(
    cache_dir: Path,
    manifest_path: Path,
    tiles: list[Image.Image],
    key: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tile_paths: list[str] = []
    try:
        for idx, tile in enumerate(tiles):
            rel_path = f"{idx:04d}.png"
            out_path = cache_dir / rel_path
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{rel_path}.", suffix=".tmp", dir=str(cache_dir)
            )
            os.close(fd)
            tile.save(tmp_name, format="PNG")
            # fsync the data before the rename so a crash can't leave the manifest
            # pointing at a tile whose bytes never reached disk (truncated PNG).
            _fsync_path(tmp_name)
            os.replace(tmp_name, out_path)
            tile_paths.append(rel_path)
        payload = json.dumps(
            {"key": key, "tiles": tile_paths},
            sort_keys=True,
            separators=(",", ":"),
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=str(cache_dir),
            text=True,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Manifest is the commit point and must land only after all tiles are durable.
        os.replace(tmp_name, manifest_path)
    except OSError as exc:
        # ENOSPC/IO errors here are silent perf cliffs (re-tiling every epoch);
        # surface at WARNING, not DEBUG, so a full SSD is noticed.
        LOG.warning("image tile cache write failed for %s: %s", key, exc)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("failed writing image tile cache entry %s: %s", key, exc)


def _fsync_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
