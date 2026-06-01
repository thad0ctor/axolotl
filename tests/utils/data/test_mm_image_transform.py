"""Core MMImageTransform extension-point seam."""

from __future__ import annotations

from axolotl.utils.data.mm_image_transform import (
    MMImageTransform,
    resolve_mm_image_transform,
)
from axolotl.utils.dict import DictDefault


def test_resolves_tiling_transform_when_enabled(mm_tiling_plugin):
    t = resolve_mm_image_transform(
        DictDefault({"image_tiling": True, "image_tiling_shape_buckets": "ocr_pages"})
    )
    assert isinstance(t, MMImageTransform)
    assert "grid" in (t.policy_payload() or {})


def test_resolves_none_without_plugin():
    # No mm_tiling plugin registered -> no transform even if cfg requests tiling.
    assert resolve_mm_image_transform(DictDefault({"image_tiling": True})) is None


def test_resolves_none_without_tiling(mm_tiling_plugin):
    assert resolve_mm_image_transform(DictDefault({"image_tiling": False})) is None
    assert resolve_mm_image_transform(DictDefault({})) is None


def test_plugin_hook_takes_precedence(monkeypatch):
    """A plugin-provided transform overrides the built-in fallback."""
    import axolotl.integrations.base as base

    class _Fake:
        def per_source(self, *a, **k):
            return [], None

        def expand_placeholders(self, text, **k):
            return text

        def prepare(self, text, images, **k):
            return text, images

        def policy_payload(self):
            return {"fake": 1}

        def new_cache(self):
            return None

    class _Mgr:
        def get_mm_image_transform(self, cfg):
            return _Fake()

    monkeypatch.setattr(
        base.PluginManager, "get_instance", staticmethod(lambda: _Mgr())
    )
    t = resolve_mm_image_transform(DictDefault({"image_tiling": True}))
    assert isinstance(t, _Fake)
