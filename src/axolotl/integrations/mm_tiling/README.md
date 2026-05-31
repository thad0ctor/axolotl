# Image Tiling (multimodal OCR)

Splits high-resolution document images into shape-bucketed tiles for OCR
continual-pretraining and SFT, with textual position labels, an optional
whole-page overview, and a persistent SSD tile cache.

The plugin provides axolotl's multimodal image transform via the
`get_mm_image_transform` hook; the length-estimation paths and the runtime
collator apply the same transform, so packed-sample lengths stay aligned.

## Usage

```yaml
plugins:
  - axolotl.integrations.mm_tiling.MMTilingPlugin

image_tiling: true
image_tiling_shape_buckets: ocr_pages   # landscape->3x2, portrait->2x3, tall->2x4
image_tiling_tile_size: 1024
image_tiling_tile_labels: true          # <global_img> + <rowX_colY> per tile
image_tiling_native_resolution: false   # true => tiles at original page resolution
image_tiling_cache_path: /path/to/ssd/tile-cache
image_tiling_reading_order: rtl
```

See `docs/multimodal.qmd` for the full option reference.
