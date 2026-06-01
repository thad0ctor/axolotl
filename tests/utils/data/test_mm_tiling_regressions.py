"""Regression tests for multimodal tiling commit-readiness fixes."""

from __future__ import annotations

from axolotl.integrations.mm_tiling.tiling import image_tiling_config_from_cfg
from axolotl.utils.data.shared import _mm_processing_fingerprint
from axolotl.utils.dict import DictDefault


def test_explicit_shape_bucket_list_via_dictdefault_does_not_crash():
    """B1: post-validation buckets are DictDefault; hasattr(model_dump) used to crash."""
    cfg = DictDefault(
        {
            "image_tiling": True,
            "image_tiling_shape_buckets": [
                {"name": "portrait", "grid": [2, 3], "min_aspect_ratio": 0.9},
                {"name": "tall", "grid": [2, 4], "min_aspect_ratio": 1.7},
            ],
        }
    )
    tiling = image_tiling_config_from_cfg(cfg)
    assert [b.grid for b in tiling.shape_buckets] == [(2, 3), (2, 4)]


def test_string_preset_shape_buckets_still_work():
    cfg = DictDefault({"image_tiling": True, "image_tiling_shape_buckets": "ocr_pages"})
    tiling = image_tiling_config_from_cfg(cfg)
    assert tiling.shape_buckets and len(tiling.shape_buckets) == 3


def test_dataset_hash_fingerprint_tracks_tile_policy(mm_tiling_plugin):
    """B2: changing the tile policy must change the dataset-cache fingerprint."""
    grid_2x3 = DictDefault({"image_tiling": True, "image_tiling_grid": [2, 3]})
    grid_2x4 = DictDefault({"image_tiling": True, "image_tiling_grid": [2, 4]})
    buckets = DictDefault(
        {"image_tiling": True, "image_tiling_shape_buckets": "ocr_pages"}
    )
    assert _mm_processing_fingerprint(grid_2x3) != _mm_processing_fingerprint(grid_2x4)
    assert _mm_processing_fingerprint(grid_2x3) != _mm_processing_fingerprint(buckets)


def test_dataset_hash_fingerprint_empty_without_image_config():
    """No churn for text-only / non-tiling users."""
    assert _mm_processing_fingerprint(DictDefault({"image_tiling": False})) == ""
    assert _mm_processing_fingerprint(DictDefault({})) == ""


def test_dataset_hash_fingerprint_tracks_resize_knobs():
    plain = DictDefault({})
    resized = DictDefault({"image_size": 512, "image_resize_no_upscale": True})
    assert _mm_processing_fingerprint(plain) != _mm_processing_fingerprint(resized)


class TestTilePositionLabels:
    """DocOwl/InternVL-style positional tile labels (<rowX_colY> + <global_img>)."""

    def test_rtl_overview_labels(self):
        from axolotl.integrations.mm_tiling.tiling import (
            ImageTilingConfig,
            tile_position_labels,
        )

        cfg = ImageTilingConfig(
            tile_size=64, grid=(2, 3), overview_size=64, reading_order="rtl"
        )
        assert tile_position_labels(cfg, 7) == [
            "<global_img>",
            "<row0_col1>",
            "<row0_col0>",
            "<row1_col1>",
            "<row1_col0>",
            "<row2_col1>",
            "<row2_col0>",
        ]

    def test_ltr_no_overview_labels(self):
        from axolotl.integrations.mm_tiling.tiling import (
            ImageTilingConfig,
            tile_position_labels,
        )

        cfg = ImageTilingConfig(tile_size=64, grid=(3, 2), reading_order="ltr")
        assert tile_position_labels(cfg, 6) == [
            "<row0_col0>",
            "<row0_col1>",
            "<row0_col2>",
            "<row1_col0>",
            "<row1_col1>",
            "<row1_col2>",
        ]

    def test_labels_disabled(self):
        from axolotl.integrations.mm_tiling.tiling import (
            ImageTilingConfig,
            tile_position_labels,
        )

        cfg = ImageTilingConfig(tile_size=64, grid=(2, 3), tile_labels=False)
        assert tile_position_labels(cfg, 6) is None

    def test_single_tile_unlabeled(self):
        from axolotl.integrations.mm_tiling.tiling import (
            ImageTilingConfig,
            tile_position_labels,
        )

        cfg = ImageTilingConfig(tile_size=64, grid=(2, 3))
        assert tile_position_labels(cfg, 1) is None

    def test_count_mismatch_returns_none(self):
        from axolotl.integrations.mm_tiling.tiling import (
            ImageTilingConfig,
            tile_position_labels,
        )

        cfg = ImageTilingConfig(tile_size=64, grid=(2, 3))  # expects 6
        assert tile_position_labels(cfg, 5) is None

    def test_expand_interleaves_labels(self):
        from axolotl.integrations.mm_tiling.tiling import (
            expand_image_placeholders_for_tiling,
        )

        out = expand_image_placeholders_for_tiling(
            "<img>\nx",
            image_token="<img>",
            counts=[3],
            labels=[["<global_img>", "<row0_col1>", "<row0_col0>"]],
        )
        assert out == "<global_img><img>\n<row0_col1><img>\n<row0_col0><img>\nx"

    def test_default_tile_labels_on(self):
        from axolotl.integrations.mm_tiling.tiling import image_tiling_config_from_cfg
        from axolotl.utils.dict import DictDefault

        cfg = image_tiling_config_from_cfg(DictDefault({"image_tiling": True}))
        assert cfg.tile_labels is True


class TestTilingReviewFixes:
    """Fixes from the multi-agent review: EXIF orientation, label cache key, hashes."""

    def test_exif_oriented_image_buckets_consistently(self, tmp_path):
        """Cached and in-memory paths must pick the same grid/labels for EXIF-rotated
        images (load_image applies exif_transpose; the bucket must use oriented size)."""
        from PIL import Image

        from axolotl.integrations.mm_tiling.tiling import (
            DEFAULT_OCR_IMAGE_TILING_BUCKETS,
            ImageTileCache,
            ImageTilingConfig,
        )

        p = tmp_path / "rot.jpg"
        img = Image.new("RGB", (120, 360), "white")  # on-disk portrait...
        exif = img.getexif()
        exif[0x0112] = 6  # ...displays as 360x120 landscape after transpose
        img.save(p, exif=exif)

        cfg = ImageTilingConfig(
            tile_size=64,
            grid=(2, 3),
            shape_buckets=DEFAULT_OCR_IMAGE_TILING_BUCKETS,
            reading_order="rtl",
        )
        tiles_cached, labels_cached = ImageTileCache(str(tmp_path / "c")).get_or_create(
            str(p), cfg
        )
        tiles_mem, labels_mem = ImageTileCache(None).get_or_create(str(p), cfg)
        assert len(tiles_cached) == len(tiles_mem)
        assert labels_cached == labels_mem
        # landscape -> 3x2 grid (6 tiles), not portrait 2x3
        assert labels_cached[0] == "<row0_col2>"

    def test_tile_labels_excluded_from_pixel_cache_key(self):
        from dataclasses import replace

        from axolotl.integrations.mm_tiling.tiling import (
            ImageTilingConfig,
            image_tile_cache_key,
        )

        on = ImageTilingConfig(tile_size=64, grid=(2, 3), tile_labels=True)
        off = replace(on, tile_labels=False)
        assert image_tile_cache_key("/x.png", on) == image_tile_cache_key("/x.png", off)

    def test_tile_labels_still_in_metadata_policy(self):
        """Labels DO change token length, so they must stay in the metadata payload."""
        from dataclasses import replace

        from axolotl.integrations.mm_tiling.tiling import (
            ImageTilingConfig,
            _tiling_policy_payload,
        )

        on = ImageTilingConfig(tile_size=64, grid=(2, 3), tile_labels=True)
        off = replace(on, tile_labels=False)
        assert _tiling_policy_payload(on) != _tiling_policy_payload(off)

    def test_metadata_cache_key_tracks_tiling_policy(self):
        from axolotl.integrations.mm_tiling.tiling import (
            ImageTilingConfig,
            TilingImageTransform,
        )
        from axolotl.utils.data.mm_packing import multimodal_metadata_cache_key

        class _Tok:
            name_or_path = "tok"
            eos_token = "</s>"

        class _Proc:
            name_or_path = "proc"

        common = dict(
            text="<image>",
            images=["/a.png"],
            tokenizer=_Tok(),
            processor=_Proc(),
            image_base_dir=None,
            image_token_id=1,
            image_token="<image>",
        )
        grid_a = TilingImageTransform(ImageTilingConfig(tile_size=64, grid=(2, 3)))
        grid_b = TilingImageTransform(ImageTilingConfig(tile_size=64, grid=(2, 4)))
        ka = multimodal_metadata_cache_key(**common, image_transform=grid_a)
        kb = multimodal_metadata_cache_key(**common, image_transform=grid_b)
        none = multimodal_metadata_cache_key(**common, image_transform=None)
        assert ka != kb and ka != none

    def test_manifest_rejects_key_mismatch(self, tmp_path):
        import json

        from axolotl.integrations.mm_tiling.tiling import _read_cached_manifest

        manifest = tmp_path / "manifest.json"
        (tmp_path / "0000.png").write_bytes(b"x")
        manifest.write_text(json.dumps({"key": "OTHER", "tiles": ["0000.png"]}))
        assert _read_cached_manifest(manifest, expected_key="EXPECTED") is None
        assert _read_cached_manifest(manifest, expected_key="OTHER") == ["0000.png"]


def test_bare_dict_bucket_without_name_does_not_crash():
    """Hardening: a bucket dict lacking 'name' should default it, not TypeError."""
    from axolotl.integrations.mm_tiling.tiling import _normalize_shape_buckets

    buckets = _normalize_shape_buckets([{"grid": [2, 3], "min_aspect_ratio": 0.9}])
    assert buckets[0].name is None
    assert buckets[0].grid == (2, 3)


def test_native_resolution_keeps_crop_size():
    """native_resolution: tiles are the raw crops (no resize), so the reconstructed
    page ~ the original; fixed tile_size instead pads/resizes to a constant canvas."""
    from PIL import Image

    from axolotl.integrations.mm_tiling.tiling import (
        ImageTileCache,
        build_image_tiling_config,
    )

    img = Image.new("RGB", (600, 900), "white")  # portrait -> ocr_pages 2x3
    native = build_image_tiling_config(
        enabled=True, shape_buckets="ocr_pages", overlap=0.0, native_resolution=True
    )
    fixed = build_image_tiling_config(
        enabled=True, shape_buckets="ocr_pages", overlap=0.0, tile_size=512
    )
    n_tiles, _ = ImageTileCache(None).get_or_create(img, native)
    f_tiles, _ = ImageTileCache(None).get_or_create(img, fixed)
    assert len(n_tiles) == 6 and len(f_tiles) == 6
    # native tiles ~ region size (600/2 x 900/3 = 300x300); fixed are 512x512
    assert all(t.size == (300, 300) for t in n_tiles)
    assert all(t.size == (512, 512) for t in f_tiles)
    # reconstructed native area == original (overlap 0)
    assert sum(w * h for w, h in (t.size for t in n_tiles)) == 600 * 900
