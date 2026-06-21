"""MMTilingPlugin provides the image transform via the core hook."""

from __future__ import annotations

import pytest

from axolotl.utils.dict import DictDefault


@pytest.fixture
def _registered_plugin():
    from axolotl.integrations.base import PluginManager
    from axolotl.integrations.mm_tiling import MMTilingPlugin

    pm = PluginManager.get_instance()
    saved = dict(pm.plugins)
    pm.plugins.clear()
    pm.plugins["mm_tiling"] = MMTilingPlugin()
    try:
        yield
    finally:
        pm.plugins.clear()
        pm.plugins.update(saved)


def test_plugin_provides_transform(_registered_plugin):
    from axolotl.utils.data.mm_image_transform import (
        MMImageTransform,
        resolve_mm_image_transform,
    )

    t = resolve_mm_image_transform(
        DictDefault(
            {
                "image_tiling": True,
                "image_tiling_shape_buckets": "ocr_pages",
                "image_tiling_tile_size": 1024,
            }
        )
    )
    assert isinstance(t, MMImageTransform)
    assert t.policy_payload()["tile_size"] == 1024


def test_plugin_declines_without_tiling(_registered_plugin):
    from axolotl.integrations.mm_tiling import MMTilingPlugin

    assert (
        MMTilingPlugin().get_mm_image_transform(DictDefault({"image_tiling": False}))
        is None
    )
