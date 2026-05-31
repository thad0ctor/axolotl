"""Image-tiling plugin: provides the multimodal image transform for OCR.

Splits a high-res page into shape-bucketed, position-labeled tiles (with an SSD
cache and an optional whole-page overview) and exposes it through the core
``get_mm_image_transform`` hook, so the rest of axolotl stays tiling-agnostic.
"""

from __future__ import annotations

from axolotl.integrations.base import BasePlugin


class MMTilingPlugin(BasePlugin):
    """Registers the tiling image transform when ``image_tiling`` is set."""

    def get_mm_image_transform(self, cfg):
        from axolotl.utils.data.mm_tiling import (
            TilingImageTransform,
            image_tiling_config_from_cfg,
        )

        config = image_tiling_config_from_cfg(cfg)
        if config is None:
            return None
        return TilingImageTransform(config)
