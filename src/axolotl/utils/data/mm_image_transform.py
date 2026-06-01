"""Generic multimodal image-transform extension point.

A transform rewrites a row's ``(text, image_sources)`` before the processor sees
it (e.g. tiling a page into labeled tiles). The SAME transform must run in the
length-estimation paths and the runtime collator so packed lengths stay aligned,
so core resolves ONE transform per cfg and threads it through every site.

Tiling itself lives in the ``mm_tiling`` plugin; core only knows this protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MMImageTransform(Protocol):
    """Row-level image transform applied identically in length + collator paths."""

    def per_source(
        self,
        source: Any,
        *,
        image_base_dir: str | None = None,
        resize_algorithm: Any | None = None,
        cache: Any | None = None,
    ) -> tuple[list[Any], list[str] | None]:
        """Expand one image source into >=1 processor-ready images + optional
        per-image text labels (aligned to the returned image order)."""
        ...

    def expand_placeholders(
        self,
        text: str,
        *,
        image_token: str,
        counts: list[int],
        labels: list[list[str] | None] | None = None,
    ) -> str:
        """Expand each image placeholder in ``text`` into ``counts[i]`` units,
        optionally prefixing each with its label."""
        ...

    def prepare(
        self,
        text: str,
        images: list[Any],
        *,
        image_token: str,
        image_base_dir: str | None = None,
        resize_algorithm: Any | None = None,
        cache: Any | None = None,
    ) -> tuple[str, list[Any]]:
        """Convenience: per_source over all images + one expand. Used by the
        length-estimation paths; the collator calls per_source/expand directly
        so it can drop bad rows."""
        ...

    def policy_payload(self) -> dict | None:
        """Length-affecting policy, folded into dataset + metadata cache hashes
        so a policy change invalidates stale prepared datasets."""
        ...

    def new_cache(self) -> Any:
        """A fresh transform-owned cache handle (or None)."""
        ...


def resolve_mm_image_transform(cfg: Any) -> MMImageTransform | None:
    """Resolve the image transform for this cfg, or None if none applies.

    The transform is provided by a plugin via ``BasePlugin.get_mm_image_transform``
    (e.g. ``axolotl.integrations.mm_tiling``). Returns None when no plugin claims
    the cfg, so non-tiling runs are untouched.
    """
    try:
        from axolotl.integrations.base import PluginManager

        return PluginManager.get_instance().get_mm_image_transform(cfg)
    except Exception:  # noqa: BLE001 - manager may be uninitialized in tooling
        return None
