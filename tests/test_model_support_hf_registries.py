"""Offline contracts for the transformers registries a model profile can claim."""

import pytest
from transformers import conversion_mapping as hf_conversion_mapping
from transformers.conversion_mapping import (
    WeightRenaming,
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.core_model_loading import ConversionOps, WeightConverter
from transformers.monkey_patching import get_patch_mapping

from axolotl.model_support import (
    VANILLA_CAUSAL_LM,
    CheckpointConversionContext,
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
    ModelStrategyOverrides,
    ModelSupport,
    apply_checkpoint_conversions,
    apply_patch_mappings,
    hf_registries,
    prepare_model_for_save,
    registry as support_registry,
    reset_patch_mappings,
    resolve_model_support,
    run_model_support_hooks,
)
from axolotl.utils.dict import DictDefault


@pytest.fixture
def isolated_registry(monkeypatch):
    support_registry._ensure_builtins()
    monkeypatch.setattr(
        support_registry, "_REGISTRY", support_registry._REGISTRY.copy()
    )
    return support_registry


@pytest.fixture
def conversion_keys():
    """Drop any conversion mapping a test registered, restoring prior entries."""
    claimed: list[str] = []

    def claim(key):
        claimed.append(key)
        return key

    yield claim

    cache = hf_conversion_mapping._checkpoint_conversion_mapping_cache or {}
    user_registered = getattr(hf_conversion_mapping, "USER_REGISTERED_MAPPINGS", set())
    for key in claimed:
        cache.pop(key, None)
        user_registered.discard(key)


@pytest.fixture
def patch_mapping_cleanup(monkeypatch):
    monkeypatch.setattr(hf_registries, "_REGISTERED_PATCH_KEYS", set())
    yield
    reset_patch_mappings()


class _IrreversibleOp(ConversionOps):
    """Conversion op that cannot be reversed, i.e. cannot be saved back out."""

    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        return input_dict


def _legacy_embedding_rename():
    return WeightRenaming("embedding.weight", "embeddings.weight")


def test_profile_declared_patch_mappings_reach_transformers(patch_mapping_cleanup):
    import torch.nn as nn

    class Replacement(nn.Module):
        pass

    class PatchedSupport(ModelSupport):
        model_types = ("patch_mapping_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            strategies=ModelStrategyOverrides(
                patch_mappings=lambda: {"PatchMappingTestAttention": Replacement},
            ),
        )

    keys = apply_patch_mappings(PatchedSupport(), DictDefault({}))

    assert keys == ("PatchMappingTestAttention",)
    assert get_patch_mapping()["PatchMappingTestAttention"] is Replacement

    # Re-applying is a no-op rather than a duplicate-registration error.
    apply_patch_mappings(PatchedSupport(), DictDefault({}))
    assert get_patch_mapping()["PatchMappingTestAttention"] is Replacement

    reset_patch_mappings()
    assert "PatchMappingTestAttention" not in get_patch_mapping()


def test_patch_mappings_replace_the_class_transformers_builds_with(
    patch_mapping_cleanup,
):
    from transformers import AutoModelForCausalLM, LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP

    class MarkedMLP(LlamaMLP):
        pass

    class PatchedSupport(ModelSupport):
        model_types = ("llama",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            strategies=ModelStrategyOverrides(
                patch_mappings=lambda: {"LlamaMLP": MarkedMLP},
            ),
        )

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )

    def built_mlp_cls():
        return type(AutoModelForCausalLM.from_config(config).model.layers[0].mlp)

    assert built_mlp_cls() is LlamaMLP

    apply_patch_mappings(PatchedSupport(), DictDefault({}))
    assert built_mlp_cls() is MarkedMLP

    reset_patch_mappings()
    assert built_mlp_cls() is LlamaMLP


def test_models_without_patch_mappings_touch_nothing(patch_mapping_cleanup):
    class PlainSupport(ModelSupport):
        model_types = ("plain_patch_test",)
        profile = ModelProfile(family=VANILLA_CAUSAL_LM)

    before = get_patch_mapping()

    assert apply_patch_mappings(PlainSupport(), DictDefault({})) == ()
    assert apply_patch_mappings(None, DictDefault({})) == ()
    assert get_patch_mapping() == before


def test_checkpoint_conversions_can_filter_an_existing_mapping(conversion_keys):
    key = conversion_keys("conversion_filter_test")
    register_checkpoint_conversion_mapping(
        key, [_legacy_embedding_rename()], overwrite=True
    )

    def drop_renamings(context):
        kept = [
            transform
            for transform in context.conversions
            if not isinstance(transform, WeightRenaming)
        ]
        return kept if len(kept) != len(context.conversions) else None

    class FilteringSupport(ModelSupport):
        model_types = (key,)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            strategies=ModelStrategyOverrides(checkpoint_conversions=drop_renamings),
        )

    assert apply_checkpoint_conversions(FilteringSupport(), DictDefault({})) == (key,)
    assert get_checkpoint_conversion_mapping(key) == []

    # Idempotent: a second pass has nothing left to filter and reports no change.
    assert apply_checkpoint_conversions(FilteringSupport(), DictDefault({})) == ()
    assert get_checkpoint_conversion_mapping(key) == []


def test_checkpoint_conversions_can_add_a_mapping_for_an_unmapped_model(
    conversion_keys,
):
    key = conversion_keys("conversion_add_test")
    added = _legacy_embedding_rename()

    class AddingSupport(ModelSupport):
        model_types = (key,)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            strategies=ModelStrategyOverrides(
                checkpoint_conversions=lambda context: [added]
            ),
        )

    assert get_checkpoint_conversion_mapping(key) is None
    assert apply_checkpoint_conversions(AddingSupport(), DictDefault({})) == (key,)

    registered = get_checkpoint_conversion_mapping(key)
    assert [type(transform).__name__ for transform in registered] == ["WeightRenaming"]


def test_checkpoint_conversion_context_reports_the_registered_state(conversion_keys):
    key = conversion_keys("conversion_context_test")
    register_checkpoint_conversion_mapping(
        key, [_legacy_embedding_rename()], overwrite=True
    )
    seen: list[CheckpointConversionContext] = []
    cfg = DictDefault({"model_config_type": key})

    class InspectingSupport(ModelSupport):
        model_types = (key,)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            strategies=ModelStrategyOverrides(
                checkpoint_conversions=lambda context: seen.append(context) or None
            ),
        )

    apply_checkpoint_conversions(InspectingSupport(), cfg, model=None)

    assert len(seen) == 1
    assert seen[0].key == key
    assert seen[0].cfg is cfg
    assert [type(transform).__name__ for transform in seen[0].conversions] == [
        "WeightRenaming"
    ]


def test_irreversible_conversions_warn_because_saving_reverses_them(
    conversion_keys, caplog
):
    key = conversion_keys("conversion_reverse_test")

    class IrreversibleSupport(ModelSupport):
        model_types = (key,)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            strategies=ModelStrategyOverrides(
                checkpoint_conversions=lambda context: [
                    WeightConverter("a.weight", "b.weight", [_IrreversibleOp()])
                ]
            ),
        )

    with caplog.at_level("WARNING"):
        apply_checkpoint_conversions(IrreversibleSupport(), DictDefault({}))

    assert "_IrreversibleOp" in caplog.text
    assert "save_original_format" in caplog.text


def test_save_boundary_runs_hooks_and_reapplies_conversions(
    isolated_registry, conversion_keys
):
    key = conversion_keys("save_boundary_test")
    register_checkpoint_conversion_mapping(
        key, [_legacy_embedding_rename()], overwrite=True
    )
    saved: list[str] = []

    class SavingSupport(ModelSupport):
        model_types = (key,)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            strategies=ModelStrategyOverrides(
                checkpoint_conversions=lambda context: [],
            ),
            hooks=ModelHooks(
                {ModelHookPhase.BEFORE_SAVE: (lambda context: saved.append("hook"),)}
            ),
        )

    isolated_registry.register_model_support(SavingSupport)
    cfg = DictDefault({"model_config_type": key})

    prepare_model_for_save(cfg, model=None)
    prepare_model_for_save(cfg, model=None)

    assert saved == ["hook", "hook"]
    assert get_checkpoint_conversion_mapping(key) == []


def test_save_boundary_is_a_noop_without_a_descriptor(isolated_registry):
    prepare_model_for_save(DictDefault({"model_config_type": "unregistered_arch"}))
    prepare_model_for_save(None)


def test_legacy_methods_still_override_profile_registrations():
    class LegacySupport(ModelSupport):
        model_types = ("legacy_registry_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            strategies=ModelStrategyOverrides(
                patch_mappings=lambda: {"FromProfile": object},
                checkpoint_conversions=lambda context: ["from-profile"],
            ),
        )

        def get_patch_mappings(self):
            return {"FromLegacyMethod": object}

        def get_checkpoint_conversions(self, context):
            return ["from-legacy-method"]

    resolved = resolve_model_support(LegacySupport())

    assert resolved.strategies.patch_mappings() == {"FromLegacyMethod": object}
    assert resolved.strategies.checkpoint_conversions(
        CheckpointConversionContext(key="legacy_registry_test")
    ) == ["from-legacy-method"]


def test_legacy_pre_save_method_is_dispatched_as_a_before_save_hook():
    calls: list[str] = []
    model = object()

    class LegacySaveSupport(ModelSupport):
        model_types = ("legacy_pre_save_test",)

        def pre_save(self, cfg, model):  # pylint: disable=redefined-outer-name
            calls.append(cfg.model_config_type)

    support = LegacySaveSupport()
    cfg = DictDefault({"model_config_type": "legacy_pre_save_test"})

    run_model_support_hooks(
        support,
        ModelHookPhase.BEFORE_SAVE,
        ModelHookContext(cfg=cfg, model=model),
    )

    assert calls == ["legacy_pre_save_test"]

    # The legacy method itself dispatches the phase; the guard stops it re-entering.
    support.pre_save(cfg, model)
    assert calls == ["legacy_pre_save_test", "legacy_pre_save_test"]


def test_nemotron_h_profile_drops_the_save_corrupting_rename(conversion_keys):
    key = conversion_keys("nemotron_h")
    support = support_registry.get_model_support("nemotron_h")
    register_checkpoint_conversion_mapping(
        key,
        [_legacy_embedding_rename(), WeightRenaming("other.weight", "kept.weight")],
        overwrite=True,
    )

    assert apply_checkpoint_conversions(support, DictDefault({})) == (key,)

    remaining = get_checkpoint_conversion_mapping(key)
    assert [transform.target_patterns for transform in remaining] == [["kept.weight"]]
