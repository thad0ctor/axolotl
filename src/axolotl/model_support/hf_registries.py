"""Apply model-support declarations to transformers' process-wide registries.

Two registries decide how an architecture is built and how its weights move in
and out of a checkpoint: the patch mapping (class replacements applied while a
model is constructed) and the checkpoint conversion mapping (applied on load,
and *reversed* on save). Profiles declare both; these helpers apply them
idempotently at the model-support lifecycle boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from axolotl.utils.logging import get_logger

from .profile import CheckpointConversionContext, resolve_model_support

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from axolotl.utils.dict import DictDefault

    from .base import ModelSupport

LOG = get_logger(__name__)

# Patch-mapping keys this process registered, so they can be released again.
_REGISTERED_PATCH_KEYS: set[str] = set()


def apply_patch_mappings(
    support: ModelSupport | None,
    cfg: DictDefault | None = None,  # pylint: disable=unused-argument
) -> tuple[str, ...]:
    """Register the profile's class replacements with transformers.

    Returns the registered keys. Safe to call repeatedly: re-registering the
    same class is a no-op, and a key already claimed by a different class is
    reported before it is replaced.
    """
    if support is None:
        return ()
    provider = resolve_model_support(support).strategies.patch_mappings
    if provider is None:
        return ()
    mapping = provider() or {}
    if not mapping:
        return ()

    from transformers.monkey_patching import get_patch_mapping, register_patch_mapping

    registered = get_patch_mapping()
    for key, replacement in mapping.items():
        existing = registered.get(key)
        if existing is not None and existing is not replacement:
            LOG.warning(
                "Replacing an existing patch mapping for %s (%s -> %s)",
                key,
                existing.__name__,
                replacement.__name__,
            )
    register_patch_mapping(dict(mapping), overwrite=True)
    _REGISTERED_PATCH_KEYS.update(mapping)
    LOG.debug("Registered patch mappings: %s", ", ".join(sorted(mapping)))
    return tuple(mapping)


def reset_patch_mappings() -> None:
    """Release every patch mapping registered through a model-support profile."""
    if not _REGISTERED_PATCH_KEYS:
        return

    from transformers.monkey_patching import (
        get_patch_mapping,
        unregister_patch_mapping,
    )

    live = set(get_patch_mapping())
    stale = sorted(_REGISTERED_PATCH_KEYS & live)
    if stale:
        unregister_patch_mapping(stale)
    _REGISTERED_PATCH_KEYS.clear()


def _ops_without_reverse(transforms: Sequence[Any]) -> list[str]:
    """Names of conversion ops that cannot be reversed when saving."""
    missing: list[str] = []
    for transform in transforms:
        for op in getattr(transform, "operations", ()):
            try:
                _ = op.reverse_op
            except NotImplementedError:
                missing.append(type(op).__name__)
    return missing


def _warn_if_shadowed(key: str, model: PreTrainedModel | None) -> None:
    """Warn when a class-name mapping outranks the ``model_type`` key we edited.

    transformers resolves conversions by class name first, so an edit keyed on
    ``model_type`` silently does nothing when the loaded class has its own entry.
    """
    if model is None:
        return

    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    class_name = type(model).__name__
    if class_name == key:
        return
    if get_checkpoint_conversion_mapping(class_name) is not None:
        LOG.warning(
            "Checkpoint conversions for model_type=%s are shadowed by the "
            "class-specific mapping registered for %s",
            key,
            class_name,
        )


def apply_checkpoint_conversions(
    support: ModelSupport | None,
    cfg: DictDefault | None = None,
    *,
    model: PreTrainedModel | None = None,
) -> tuple[str, ...]:
    """Apply the profile's checkpoint-conversion edits, once per ``model_type``.

    Returns the keys that were re-registered. Called at each boundary where the
    mapping matters — before the build, after it (model ``__init__`` may
    re-register its own mapping), and before saving — so adapters must be
    idempotent.
    """
    if support is None:
        return ()
    resolved = resolve_model_support(support)
    provider = resolved.strategies.checkpoint_conversions
    if provider is None:
        return ()

    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )

    updated_keys: list[str] = []
    for key in resolved.model_types:
        current = get_checkpoint_conversion_mapping(key) or []
        updated = provider(
            CheckpointConversionContext(
                key=key,
                conversions=tuple(current),
                cfg=cfg,
                model=model,
            )
        )
        if updated is None:
            continue

        updated = list(updated)
        missing = _ops_without_reverse(updated)
        if missing:
            LOG.warning(
                "Checkpoint conversions for %s use operations without a reverse "
                "(%s); saving with save_original_format=True will fail",
                key,
                ", ".join(sorted(set(missing))),
            )
        register_checkpoint_conversion_mapping(key, updated, overwrite=True)
        _warn_if_shadowed(key, model)
        updated_keys.append(key)

    if updated_keys:
        LOG.debug(
            "Applied checkpoint conversions for %s", ", ".join(sorted(updated_keys))
        )
    return tuple(updated_keys)
