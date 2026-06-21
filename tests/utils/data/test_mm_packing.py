"""Unit tests for multimodal packing metadata helpers."""

from __future__ import annotations

import json

import pytest

from axolotl.integrations.mm_tiling.tiling import (
    DEFAULT_OCR_IMAGE_TILING_BUCKETS,
    ImageTileCache,
    ImageTilingConfig,
    build_image_tiling_config,
    expand_image_placeholders_for_tiling,
    image_tile_cache_key,
    replace_first_image_placeholder,
    select_tiling_config_for_image,
    tile_image_for_processor,
    tile_image_source_for_processor,
)
from axolotl.utils.data import mm_packing
from axolotl.utils.data.mm_image import (
    resize_image_for_processor,
    select_image_resize_bucket,
)
from axolotl.utils.data.mm_packing import (
    MultimodalPackingMetadata,
    MultimodalPackingMetadataCache,
    _metadata_from_processor_output,
    compute_multimodal_packing_metadata,
    multimodal_metadata_cache_key,
    pack_2d_first_fit_decreasing,
)


def test_select_image_resize_bucket_prefers_nearest_static_canvas():
    buckets = [(1024, 1024), (1024, 1536), (1536, 1024), (1536, 1536)]

    assert select_image_resize_bucket((900, 1300), buckets) == (1024, 1536)
    assert select_image_resize_bucket((2200, 900), buckets) == (1536, 1024)


def test_resize_image_for_processor_bucket_no_upscale_and_pad_color():
    from PIL import Image

    image = Image.new("RGB", (64, 32), "black")

    resized = resize_image_for_processor(
        image,
        image_size=None,
        image_resize_buckets=[(128, 128)],
        image_resize_no_upscale=True,
        image_resize_pad_color="white",
    )

    assert resized.size == (128, 128)
    assert resized.getpixel((0, 0)) == (255, 255, 255)
    assert resized.getpixel((64, 64)) == (0, 0, 0)


def test_tile_image_for_processor_adds_overview_and_ordered_tiles():
    from PIL import Image

    image = Image.new("RGB", (2000, 3000), "black")
    tiled = tile_image_for_processor(
        image,
        ImageTilingConfig(
            tile_size=512,
            overview_buckets=[(512, 768)],
            grid=(2, 3),
            overlap=0.1,
            min_area=1,
            pad_color="white",
        ),
    )

    assert len(tiled) == 7
    assert tiled[0].size == (512, 768)
    assert all(tile.size == (512, 512) for tile in tiled[1:])


def test_replace_first_image_placeholder_repeats_qwen_wrapped_placeholder():
    text = "<|vision_start|><|image_pad|><|vision_end|>\nOCR text"

    tiled = replace_first_image_placeholder(
        text,
        image_token="<|image_pad|>",
        count=3,
    )

    assert tiled.count("<|image_pad|>") == 3
    assert tiled.endswith("\nOCR text")


def test_expand_image_placeholders_for_tiling_handles_multiple_placeholders():
    text = "a <image> b <image> c <image> d"

    tiled = expand_image_placeholders_for_tiling(
        text,
        image_token="<image>",
        counts=[2, 1, 3],
    )

    assert tiled == "a <image>\n<image> b <image> c <image>\n<image>\n<image> d"


def test_select_tiling_config_for_shape_buckets():
    config = ImageTilingConfig(
        tile_size=512,
        grid=(2, 3),
        shape_buckets=DEFAULT_OCR_IMAGE_TILING_BUCKETS,
    )

    assert select_tiling_config_for_image((2000, 1000), config).grid == (3, 2)
    assert select_tiling_config_for_image((1000, 1500), config).grid == (2, 3)
    assert select_tiling_config_for_image((1000, 2200), config).grid == (2, 4)


def test_build_image_tiling_config_accepts_ocr_pages_preset():
    config = build_image_tiling_config(
        enabled=True,
        tile_size=[256, 512],
        shape_buckets="ocr_pages",
    )

    assert config is not None
    assert config.tile_size == (256, 512)
    assert config.shape_buckets == DEFAULT_OCR_IMAGE_TILING_BUCKETS


def test_tile_cache_round_trips_from_ssd(tmp_path, monkeypatch):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (120, 180), "white").save(image_path)
    config = ImageTilingConfig(
        tile_size=32,
        grid=(2, 3),
        cache_path=str(tmp_path / "tiles"),
    )

    first = tile_image_source_for_processor(
        str(image_path),
        config,
        cache=ImageTileCache(config.cache_path),
    )
    assert len(first) == 6
    assert list((tmp_path / "tiles").rglob("manifest.json"))

    def _fail(*_args, **_kwargs):
        raise AssertionError("tile cache hit should not recreate tiles")

    monkeypatch.setattr(
        "axolotl.integrations.mm_tiling.tiling.tile_image_for_processor", _fail
    )
    second = tile_image_source_for_processor(
        str(image_path),
        config,
        cache=ImageTileCache(config.cache_path),
    )
    assert len(second) == 6


def test_tile_cache_key_changes_with_source_stat_and_policy(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (120, 180), "white").save(image_path)
    base_config = ImageTilingConfig(tile_size=32, grid=(2, 3))

    key = image_tile_cache_key(str(image_path), base_config)
    policy_key = image_tile_cache_key(
        str(image_path),
        ImageTilingConfig(tile_size=32, grid=(2, 4)),
    )
    Image.new("RGB", (121, 180), "white").save(image_path)
    stat_key = image_tile_cache_key(str(image_path), base_config)

    assert key != policy_key
    assert key != stat_key


def test_tile_cache_key_can_include_image_hash(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    config = ImageTilingConfig(tile_size=16, grid=(1, 1))

    key = image_tile_cache_key(str(image_path), config, hash_image=True)
    Image.new("RGB", (16, 16), "black").save(image_path)
    hashed_key = image_tile_cache_key(str(image_path), config, hash_image=True)

    assert key != hashed_key


class _Tokenizer:
    name_or_path = "tok"
    eos_token = "</s>"


class _ImageProcessor:
    size = {"height": 224, "width": 224}


class _Processor:
    name_or_path = "proc"
    image_processor = _ImageProcessor()


def test_pack_2d_first_fit_respects_token_and_visual_constraints():
    bins = pack_2d_first_fit_decreasing(
        [6, 6, 4],
        [6, 1, 4],
        token_capacity=10,
        visual_capacity=7,
    )

    assert len(bins) == 2
    for bin_indices in bins:
        assert sum([6, 6, 4][idx] for idx in bin_indices) <= 10
        assert sum([6, 1, 4][idx] for idx in bin_indices) <= 7


def test_pack_2d_first_fit_rejects_bad_inputs():
    with pytest.raises(ValueError, match="same length"):
        pack_2d_first_fit_decreasing([1], [], token_capacity=2, visual_capacity=2)
    with pytest.raises(ValueError, match="token packing capacity"):
        pack_2d_first_fit_decreasing([3], [1], token_capacity=2, visual_capacity=2)
    with pytest.raises(ValueError, match="visual packing capacity"):
        pack_2d_first_fit_decreasing([1], [3], token_capacity=2, visual_capacity=2)


def test_metadata_cache_round_trips_from_ssd_without_recomputing(tmp_path, monkeypatch):
    cache = MultimodalPackingMetadataCache(str(tmp_path), ram_budget_mb=0)
    key = multimodal_metadata_cache_key(
        text="hello",
        images=[],
        tokenizer=_Tokenizer(),
        processor=_Processor(),
        image_base_dir=None,
        image_token_id=7,
        add_eos_token=True,
    )
    expected = MultimodalPackingMetadata(
        length=12,
        visual_tokens=0,
        image_count=0,
        visual_signature="images=0:tokens=0",
    )
    cache.set(key, expected)

    def _fail(*_args, **_kwargs):
        raise AssertionError("cache miss should not recompute metadata")

    monkeypatch.setattr(mm_packing, "_compute_uncached_metadata", _fail)

    actual = compute_multimodal_packing_metadata(
        ["hello"],
        [[]],
        tokenizer=_Tokenizer(),
        processor=_Processor(),
        image_token_id=7,
        cache=MultimodalPackingMetadataCache(str(tmp_path), ram_budget_mb=0),
    )

    assert actual == [expected]


def test_metadata_cache_key_changes_with_resize_policy():
    base_kwargs = {
        "text": "hello",
        "images": ["img.png"],
        "tokenizer": _Tokenizer(),
        "processor": _Processor(),
        "image_base_dir": None,
        "image_token_id": 7,
        "add_eos_token": True,
    }

    default_key = multimodal_metadata_cache_key(**base_kwargs)
    sized_key = multimodal_metadata_cache_key(**base_kwargs, image_size=512)
    algorithm_key = multimodal_metadata_cache_key(
        **base_kwargs,
        image_size=512,
        image_resize_algorithm="bicubic",
    )
    bucket_key = multimodal_metadata_cache_key(
        **base_kwargs,
        image_resize_buckets=[(1024, 1536), (1536, 1536)],
        image_resize_no_upscale=True,
        image_resize_pad_color="white",
    )

    assert default_key != sized_key
    assert sized_key != algorithm_key
    assert default_key != bucket_key
    assert algorithm_key != bucket_key


def test_metadata_cache_ignores_corrupt_ssd_entry(tmp_path):
    cache = MultimodalPackingMetadataCache(str(tmp_path), ram_budget_mb=0)
    path = cache._path_for_key("a" * 32)  # pylint: disable=protected-access
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"length": "bad"}), encoding="utf-8")

    assert cache.get("a" * 32) is None


def test_metadata_uses_grid_fallback_when_processor_does_not_expand_placeholder():
    metadata = _metadata_from_processor_output(
        {
            "input_ids": [[1, 2, 3]],
            "image_grid_thw": [[1, 4, 5]],
        },
        [1],
        image_token_id=99,
    )

    assert metadata[0].length == 3
    assert metadata[0].visual_tokens == 20
    assert metadata[0].visual_signature == "1x4x5"


def test_metadata_computation_chunks_processor_batches(monkeypatch):
    calls = []

    def _fake_uncached(texts, imgs_list, **_kwargs):
        calls.append(len(texts))
        return [
            MultimodalPackingMetadata(
                length=len(text),
                visual_tokens=len(images),
                image_count=len(images),
                visual_signature=f"images={len(images)}",
            )
            for text, images in zip(texts, imgs_list, strict=True)
        ]

    monkeypatch.setenv("AXOLOTL_MM_PACKING_METADATA_BATCH_SIZE", "2")
    monkeypatch.setattr(mm_packing, "_compute_uncached_metadata", _fake_uncached)

    metadata = compute_multimodal_packing_metadata(
        ["a", "bb", "ccc", "dddd", "eeeee"],
        [[], [], [], [], []],
        tokenizer=_Tokenizer(),
        processor=_Processor(),
        image_token_id=7,
    )

    assert calls == [2, 2, 1]
    assert [item.length for item in metadata] == [1, 2, 3, 4, 5]
