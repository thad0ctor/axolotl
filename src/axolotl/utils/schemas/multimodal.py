"""Pydantic models for multimodal-related configuration"""

from typing import Literal

from PIL.Image import Resampling
from pydantic import BaseModel, Field, field_validator, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ImageTilingBucketSpec(BaseModel):
    """Shape bucket for selecting a tiling policy by height/width ratio."""

    name: str | None = None
    grid: tuple[int, int] = Field(
        json_schema_extra={"description": "Tile grid as (columns, rows), e.g. [2, 3]."},
    )
    min_aspect_ratio: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Minimum image height/width ratio for this policy."
        },
    )
    max_aspect_ratio: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Exclusive maximum image height/width ratio for this policy."
        },
    )
    tile_size: int | tuple[int, int] | None = None
    overlap: float | None = None
    min_area: int | None = None
    overview_size: int | tuple[int, int] | None = None
    overview_buckets: list[tuple[int, int]] | None = None
    no_upscale: bool | None = None
    pad_color: int | tuple[int, int, int] | str | None = None
    reading_order: Literal["rtl", "ltr"] | None = None

    @field_validator("grid", mode="before")
    @classmethod
    def convert_grid(cls, grid):
        return _convert_pair(grid, "image_tiling_shape_buckets.grid")

    @field_validator("tile_size", "overview_size", mode="before")
    @classmethod
    def convert_optional_size(cls, size):
        if size is None:
            return None
        if isinstance(size, int):
            if size <= 0:
                raise ValueError("image tiling sizes must be positive")
            return size
        return _convert_pair(size, "image tiling size")

    @field_validator("overview_buckets", mode="before")
    @classmethod
    def convert_overview_buckets(cls, buckets):
        return _convert_bucket_list(
            buckets, "image_tiling_shape_buckets.overview_buckets"
        )

    @field_validator("pad_color", mode="before")
    @classmethod
    def convert_pad_color(cls, color):
        if isinstance(color, list):
            return tuple(int(part) for part in color)
        return color

    @field_validator("overlap", mode="before")
    @classmethod
    def validate_bucket_overlap(cls, overlap):
        return _validate_overlap(overlap, "image_tiling_shape_buckets.overlap")


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
    image_tiling: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Enable processor-side image tiling for multimodal CPT/SFT."
        },
    )
    image_tiling_tile_size: int | tuple[int, int] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Processor canvas for each tile. Defaults to 1024 when image_tiling is enabled."
        },
    )
    image_tiling_grid: tuple[int, int] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Default tile grid as (columns, rows). Defaults to [2, 3]."
        },
    )
    image_tiling_overlap: float | None = Field(
        default=0.1,
        json_schema_extra={
            "description": "Tile overlap fraction per cell edge, clamped below 0.5."
        },
    )
    image_tiling_min_area: int | None = Field(
        default=0,
        json_schema_extra={
            "description": "Only tile images with width*height at least this value."
        },
    )
    image_tiling_overview_size: int | tuple[int, int] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Optional overview image canvas prepended before tiles."
        },
    )
    image_tiling_overview_buckets: list[tuple[int, int]] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Optional overview image canvas buckets as (width, height)."
        },
    )
    image_tiling_no_upscale: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Do not upscale page overviews or tiles when padding to fixed canvases."
        },
    )
    image_tiling_pad_color: int | tuple[int, int, int] | str | None = Field(
        default=None,
        json_schema_extra={"description": "Padding color for tiled image canvases."},
    )
    image_tiling_reading_order: Literal["rtl", "ltr"] | None = Field(
        default="rtl",
        json_schema_extra={
            "description": "Column traversal order within each tile row."
        },
    )
    image_tiling_cache_path: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Optional SSD path for cached processor-ready tile PNGs."
        },
    )
    image_tiling_cache_hash_images: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Include a full source-image sha256 in tile cache keys. Path/stat keys are used when false."
        },
    )
    image_tiling_shape_buckets: (
        list[ImageTilingBucketSpec] | Literal["auto", "ocr_pages"] | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": "Optional aspect-ratio policies. 'ocr_pages' maps landscape to 3x2, portrait to 2x3, tall to 2x4."
        },
    )
    image_tiling_tile_labels: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Prefix each tile with a textual position token (<rowX_colY>, plus <global_img> for the overview) so the model knows tile layout. Recommended for OCR."
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

    @field_validator(
        "image_tiling_tile_size", "image_tiling_overview_size", mode="before"
    )
    @classmethod
    def convert_image_tiling_size(cls, size):
        if size is None:
            return None
        if isinstance(size, int):
            if size <= 0:
                raise ValueError("image tiling sizes must be positive")
            return size
        return _convert_pair(size, "image tiling size")

    @field_validator("image_tiling_grid", mode="before")
    @classmethod
    def convert_image_tiling_grid(cls, grid):
        if grid is None:
            return None
        return _convert_pair(grid, "image_tiling_grid")

    @field_validator("image_tiling_overview_buckets", mode="before")
    @classmethod
    def convert_image_tiling_overview_buckets(cls, buckets):
        return _convert_bucket_list(buckets, "image_tiling_overview_buckets")

    @field_validator("image_tiling_pad_color", mode="before")
    @classmethod
    def convert_image_tiling_pad_color(cls, color):
        if isinstance(color, list):
            return tuple(int(part) for part in color)
        return color

    @field_validator("image_tiling_overlap", mode="before")
    @classmethod
    def validate_image_tiling_overlap(cls, overlap):
        return _validate_overlap(overlap, "image_tiling_overlap")

    @field_validator("image_tiling_shape_buckets", mode="before")
    @classmethod
    def convert_image_tiling_shape_buckets(cls, buckets):
        if buckets is None or isinstance(buckets, str):
            if buckets not in (None, "auto", "ocr_pages"):
                raise ValueError(
                    "image_tiling_shape_buckets must be 'auto', 'ocr_pages', or a "
                    "list of bucket specs"
                )
            return buckets
        return list(buckets)

    @model_validator(mode="after")
    def warn_unused_image_tiling_fields(self):
        if not self.image_tiling:
            set_fields = [
                name
                for name in (
                    "image_tiling_tile_size",
                    "image_tiling_grid",
                    "image_tiling_overview_size",
                    "image_tiling_overview_buckets",
                    "image_tiling_pad_color",
                    "image_tiling_cache_path",
                    "image_tiling_shape_buckets",
                )
                if getattr(self, name) is not None
            ]
            if set_fields:
                LOG.warning(
                    "image_tiling is disabled but these tiling fields are set and "
                    "will be ignored: %s. Set image_tiling: true to enable tiling.",
                    ", ".join(set_fields),
                )
        return self


def _validate_overlap(overlap, label: str):
    if overlap is None:
        return None
    overlap = float(overlap)
    if not 0.0 <= overlap < 0.5:
        raise ValueError(f"{label} must be in [0.0, 0.5)")
    return overlap


def _convert_pair(value, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-item pair")
    first, second = int(value[0]), int(value[1])
    if first <= 0 or second <= 0:
        raise ValueError(f"{label} entries must be positive")
    return (first, second)


def _convert_bucket_list(buckets, label: str):
    if buckets is None:
        return None
    converted = []
    for bucket in buckets:
        converted.append(_convert_pair(bucket, label))
    return converted
