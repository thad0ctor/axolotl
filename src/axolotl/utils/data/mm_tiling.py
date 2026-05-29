"""Multimodal image tiling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from axolotl.utils.data.mm_image import resize_image_for_processor


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


def tile_image_for_processor(
    image: Image.Image,
    config: ImageTilingConfig,
    *,
    resize_algorithm: Image.Resampling | None = None,
) -> list[Image.Image]:
    if image.width * image.height < config.min_area:
        return [image]

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

    for box in _tile_boxes(image.size, config.grid, config.overlap, config.reading_order):
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


def replace_first_image_placeholder(
    text: str,
    *,
    image_token: str,
    count: int,
) -> str:
    if count <= 1:
        return text

    token = image_token if image_token in text else "<image>"
    token_pos = text.find(token)
    if token_pos < 0:
        raise ValueError("Could not find an image placeholder to replace for tiling.")

    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"
    start = text.rfind(vision_start, 0, token_pos + len(token))
    end = text.find(vision_end, token_pos)
    if start >= 0 and end >= 0:
        end += len(vision_end)
        unit = text[start:end]
        return text[:start] + "\n".join([unit] * count) + text[end:]

    return text[:token_pos] + "\n".join([token] * count) + text[token_pos + len(token) :]


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
    col_order = list(range(cols - 1, -1, -1)) if reading_order == "rtl" else list(range(cols))
    for row in range(rows):
        for col in col_order:
            left = max(0, int(round(col * cell_w - pad_w)))
            top = max(0, int(round(row * cell_h - pad_h)))
            right = min(width, int(round((col + 1) * cell_w + pad_w)))
            bottom = min(height, int(round((row + 1) * cell_h + pad_h)))
            boxes.append((left, top, right, bottom))
    return boxes
