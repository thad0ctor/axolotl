"""Shared multimodal image preprocessing helpers."""

from __future__ import annotations

from typing import Any

from PIL import Image


def select_image_resize_bucket(
    image_size: tuple[int, int],
    image_resize_buckets: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    no_upscale: bool = False,
) -> tuple[int, int]:
    """Select the most detail-preserving static canvas for an image."""
    width, height = int(image_size[0]), int(image_size[1])
    buckets = sorted(
        {
            (int(bucket[0]), int(bucket[1]))
            for bucket in image_resize_buckets
            if int(bucket[0]) > 0 and int(bucket[1]) > 0
        },
        key=lambda bucket: (bucket[0] * bucket[1], bucket[0], bucket[1]),
    )
    if not buckets:
        raise ValueError("image_resize_buckets must contain at least one valid bucket.")

    containing = [
        bucket for bucket in buckets if width <= bucket[0] and height <= bucket[1]
    ]
    if containing:
        return min(
            containing,
            key=lambda bucket: (
                bucket[0] * bucket[1],
                abs((bucket[0] / bucket[1]) - (width / height)),
            ),
        )

    if no_upscale:
        return max(
            buckets,
            key=lambda bucket: (
                min(bucket[0] / width, bucket[1] / height),
                -(bucket[0] * bucket[1]),
            ),
        )

    return max(
        buckets,
        key=lambda bucket: (
            min(bucket[0] / width, bucket[1] / height),
            -(bucket[0] * bucket[1]),
        ),
    )


def resize_image_for_processor(
    image: Any,
    image_size: int | tuple[int, int] | list[int] | None,
    image_resize_algorithm: Image.Resampling | None = None,
    image_resize_buckets: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    image_resize_no_upscale: bool = False,
    image_resize_pad_color: Any | None = None,
) -> Any:
    if image_size is None and not image_resize_buckets:
        return image
    if not hasattr(image, "resize"):
        raise TypeError("Image does not have a resize method.")

    resize_algorithm = image_resize_algorithm or Image.Resampling.BILINEAR
    pad_color = _normalize_pad_color(image_resize_pad_color)
    if image_resize_buckets:
        target = select_image_resize_bucket(
            image.size,
            image_resize_buckets,
            no_upscale=image_resize_no_upscale,
        )
        return _pad_to_canvas(
            image,
            target,
            resize_algorithm,
            no_upscale=image_resize_no_upscale,
            color=pad_color,
        )

    if isinstance(image_size, list):
        image_size = tuple(image_size)
    if isinstance(image_size, tuple):
        return image.resize(
            (int(image_size[0]), int(image_size[1])),
            resize_algorithm,
        )
    size = int(image_size)
    return _pad_to_canvas(
        image,
        (size, size),
        resize_algorithm,
        no_upscale=image_resize_no_upscale,
        color=pad_color,
    )


def _pad_to_canvas(
    image: Image.Image,
    target: tuple[int, int],
    method: Image.Resampling,
    *,
    no_upscale: bool,
    color: Any,
) -> Image.Image:
    width, height = image.size
    target_w, target_h = int(target[0]), int(target[1])
    scale = min(target_w / width, target_h / height)
    if no_upscale:
        scale = min(scale, 1.0)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    fitted = image if new_size == image.size else image.resize(new_size, method)
    canvas = Image.new(fitted.mode, (target_w, target_h), color)
    offset = ((target_w - fitted.width) // 2, (target_h - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _normalize_pad_color(value: Any | None) -> Any:
    if value is None:
        return (0, 0, 0)
    if isinstance(value, list):
        return tuple(value)
    return value
