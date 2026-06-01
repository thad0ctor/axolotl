"""Pydantic models for multimodal-related configuration"""

from typing import Literal

from PIL.Image import Resampling
from pydantic import BaseModel, Field, field_validator


class RoleBoundarySpec(BaseModel):
    """One ``cfg.role_boundaries`` row; see docs/multimodal_assistant_mask.md."""

    role: str = Field(
        json_schema_extra={
            "description": (
                "Role name as it appears in cfg.roles_to_train (e.g. "
                "'assistant', 'user', 'system', 'tool', 'ipython')."
            )
        },
    )
    start: str = Field(
        json_schema_extra={
            "description": (
                "Literal string that marks the start of this role's span in "
                "the rendered chat template. Tokenized via "
                "``tokenizer.encode(..., add_special_tokens=False)`` at "
                "strategy init."
            )
        },
    )
    end: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Literal string that marks the end of this role's span. "
                "Set to ``eos_token`` to terminate at the tokenizer's EOS. "
                "Leave unset / null to terminate at end-of-sequence."
            )
        },
    )
    include_start: bool = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Whether the start marker tokens contribute to loss on "
                "trainable turns. Default False."
            )
        },
    )
    include_end: bool = Field(
        default=True,
        json_schema_extra={
            "description": (
                "Whether the end marker tokens contribute to loss on "
                "trainable turns (honoring cfg.train_on_eos). Default True."
            )
        },
    )


class MultiModalConfig(BaseModel):
    """Multi-modal configuration subset"""

    image_size: int | tuple[int, int] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "The size of the image to resize to. It can be an integer (resized into padded-square image) or a tuple (width, height)."
                "If not provided, we will attempt to load from preprocessor.size, otherwise, images won't be resized."
            )
        },
    )
    image_resize_algorithm: (
        Literal["bilinear", "bicubic", "lanczos"] | Resampling | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": "The resampling algorithm to use for image resizing. Default is bilinear. Please refer to PIL.Image.Resampling for more details."
        },
    )
    image_resize_buckets: list[tuple[int, int]] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Optional fixed image canvas buckets as (width, height). Images preserve aspect ratio and pad to the chosen bucket."
        },
    )
    image_resize_no_upscale: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "When using padded square or bucketed resizing, do not enlarge images that are already smaller than the target canvas."
        },
    )
    image_resize_pad_color: int | tuple[int, int, int] | str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Padding color for padded square or bucketed resizing. Defaults to black; OCR/document datasets often prefer 'white'."
        },
    )
    role_boundaries: list[RoleBoundarySpec] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Opt-in override for the MM mask scanner's per-role boundary "
                "markers. Non-empty list replaces built-ins wholesale; unset "
                "or empty falls back to built-ins. See "
                "docs/multimodal_assistant_mask.md."
            )
        },
    )

    @field_validator("image_resize_algorithm", mode="before")
    @classmethod
    def convert_image_resize_algorithm(cls, image_resize_algorithm):
        """
        Convert the image resize algorithm to a PIL.Image.Resampling enum.
        """
        if isinstance(image_resize_algorithm, str):
            image_resize_algorithm = image_resize_algorithm.lower()
            if image_resize_algorithm == "bilinear":
                image_resize_algorithm = Resampling.BILINEAR
            elif image_resize_algorithm == "bicubic":
                image_resize_algorithm = Resampling.BICUBIC
            elif image_resize_algorithm == "lanczos":
                image_resize_algorithm = Resampling.LANCZOS
            else:
                raise ValueError(
                    f"Invalid image resize algorithm: {image_resize_algorithm}"
                )
        return image_resize_algorithm

    @field_validator("image_resize_buckets", mode="before")
    @classmethod
    def convert_image_resize_buckets(cls, image_resize_buckets):
        """
        Convert image resize buckets to width/height tuples.
        """
        if image_resize_buckets is None:
            return None
        converted = []
        for bucket in image_resize_buckets:
            if not isinstance(bucket, (list, tuple)) or len(bucket) != 2:
                raise ValueError(
                    "image_resize_buckets entries must be two-item width/height pairs"
                )
            width, height = int(bucket[0]), int(bucket[1])
            if width <= 0 or height <= 0:
                raise ValueError("image_resize_buckets entries must be positive")
            converted.append((width, height))
        return converted

    @field_validator("image_resize_pad_color", mode="before")
    @classmethod
    def convert_image_resize_pad_color(cls, image_resize_pad_color):
        """
        Convert YAML list colors to tuples for PIL.
        """
        if isinstance(image_resize_pad_color, list):
            return tuple(int(part) for part in image_resize_pad_color)
        return image_resize_pad_color
