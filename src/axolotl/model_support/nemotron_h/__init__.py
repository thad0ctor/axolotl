"""Nemotron-H model support (hybrid Mamba/attention decoder)."""

from typing import Any, Sequence

from axolotl.model_support.base import ModelSupport
from axolotl.model_support.profile import (
    CheckpointConversionContext,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM


def _drop_legacy_embedding_rename(
    context: CheckpointConversionContext,
) -> Sequence[Any] | None:
    """Remove the spurious ``embedding`` → ``embeddings`` renaming.

    The nvidia Hub model registers that renaming to read a legacy checkpoint
    variant. Its reverse, applied on save, renames
    ``backbone.embeddings.weight`` → ``backbone.embedding.weight``, corrupting
    merged checkpoints. ``NemotronHForCausalLM.__init__`` re-registers the
    mapping with ``overwrite=True``, so this edit is re-applied after the model
    is built and again before saving.
    """
    from transformers.conversion_mapping import WeightRenaming

    kept = [
        transform
        for transform in context.conversions
        if not (
            isinstance(transform, WeightRenaming)
            and transform.source_patterns == ["embedding.weight"]
            and transform.target_patterns == ["embeddings.weight"]
        )
    ]
    return kept if len(kept) != len(context.conversions) else None


@register_model_support
class NemotronHSupport(ModelSupport):
    """Descriptor for Nemotron-H."""

    model_types = ("nemotron_h",)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        strategies=ModelStrategyOverrides(
            checkpoint_conversions=_drop_legacy_embedding_rename,
        ),
    )
