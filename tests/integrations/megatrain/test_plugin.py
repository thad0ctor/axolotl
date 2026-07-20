"""CPU-only tests for MegaTrain plugin registration and config composition."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
)

from axolotl.cli.config import load_cfg
from axolotl.integrations.base import BasePlugin, PluginManager, load_plugin
from axolotl.integrations.config import merge_input_args
from axolotl.utils.dict import DictDefault

PLUGIN = "axolotl.integrations.megatrain.MegaTrainPlugin"


def _minimal_config(**overrides):
    config = {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "datasets": [{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "plugins": [PLUGIN],
    }
    config.update(overrides)
    return config


def _runtime_config(**overrides):
    config = DictDefault(
        {
            "plugins": [PLUGIN],
            "world_size": 1,
            "torch_dtype": torch.bfloat16,
            "is_multimodal": False,
            "attn_implementation": "eager",
            "optimizer": "adamw_torch",
            "val_set_size": 0,
        }
    )
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "plugin_name",
    [PLUGIN, "megatrain.MegaTrainPlugin"],
)
def test_load_plugin(plugin_name):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    plugin = load_plugin(plugin_name)

    assert isinstance(plugin, MegaTrainPlugin)
    assert (
        plugin.get_input_args() == "axolotl.integrations.megatrain.args.MegaTrainArgs"
    )


def test_config_schema_merge_accepts_megatrain_keys(monkeypatch):
    manager = PluginManager.get_instance()
    monkeypatch.setattr(
        manager,
        "get_input_args",
        lambda: ["axolotl.integrations.megatrain.args.MegaTrainArgs"],
    )

    _, merged_input_config = merge_input_args()
    config = merged_input_config.model_validate(
        _minimal_config(
            megatrain_checkpoint_interval=3,
            megatrain_num_grad_slabs=8,
        )
    )

    assert "megatrain_checkpoint_interval" in merged_input_config.model_fields
    assert "megatrain_num_grad_slabs" in merged_input_config.model_fields
    assert config.megatrain_checkpoint_interval == 3
    assert config.megatrain_num_grad_slabs == 8


@pytest.mark.parametrize("optimizer", [None, "adamw_torch"])
def test_register_configures_cpu_loading_and_supported_optimizer(optimizer):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    config = {"plugins": [PLUGIN], "optimizer": optimizer}

    MegaTrainPlugin().register(config)

    assert config["device_map"] == "cpu"
    assert config["experimental_skip_move_to_device"] is True
    assert config["optimizer"] == "adamw_torch"


@pytest.mark.parametrize(
    ("unsupported", "message"),
    [
        ({"device_map": "auto"}, "device_map"),
        ({"device_map": {"model.layers.0": 0}}, "device_map"),
        ({"max_memory": {0: "10GiB"}}, "max_memory"),
        ({"gpu_memory_limit": "10GiB"}, "gpu_memory_limit"),
    ],
)
def test_register_rejects_non_cpu_loading(unsupported, message):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    with pytest.raises(ValueError, match=message):
        MegaTrainPlugin().register({"plugins": [PLUGIN], **unsupported})


def test_schema_merge_deduplicates_canonical_and_shorthand_instances():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    manager = PluginManager.get_instance()
    manager.plugins[PLUGIN] = MegaTrainPlugin()
    manager.plugins["megatrain.MegaTrainPlugin"] = MegaTrainPlugin()

    _, merged_input_config = merge_input_args()
    config = merged_input_config.model_validate(
        _minimal_config(plugins=["megatrain.MegaTrainPlugin"])
    )

    assert manager.get_input_args() == [
        "axolotl.integrations.megatrain.args.MegaTrainArgs"
    ]
    assert config.megatrain_checkpoint_interval is None


def test_pre_model_load_requires_cuda(monkeypatch):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA GPU"):
        MegaTrainPlugin().pre_model_load(SimpleNamespace(plugins=[PLUGIN]))


def test_pre_model_load_rejects_multi_process_runtime(monkeypatch):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    config = SimpleNamespace(
        plugins=[PLUGIN],
        world_size=1,
        torch_dtype=torch.bfloat16,
        is_multimodal=False,
        attn_implementation="sdpa",
    )

    with pytest.raises(RuntimeError, match="one Axolotl process"):
        MegaTrainPlugin().pre_model_load(config)


def test_pre_model_load_rejects_initialized_distributed_runtime(monkeypatch):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(RuntimeError, match="one Axolotl process"):
        MegaTrainPlugin().pre_model_load(_runtime_config())


def test_pre_model_load_requires_bf16_capable_gpu(monkeypatch):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(RuntimeError, match="BF16 support"):
        MegaTrainPlugin().pre_model_load(_runtime_config())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"torch_dtype": torch.float16}, "BF16 streamed compute"),
        ({"is_multimodal": True}, "multimodal"),
        ({"attn_implementation": "flex_attention"}, "supports only"),
        ({"test_datasets": [{"path": "unused"}]}, "evaluation"),
        ({"batch_flattening": True}, "batch_flattening"),
        ({"fsdp_config": {}}, "FSDP or DeepSpeed"),
        ({"model_quantization_config": {}}, "unquantized BF16"),
        ({"optimizer": "adamw_torch_fused"}, "adamw_torch"),
        ({"use_pose": True}, "use_pose"),
    ],
)
def test_pre_model_load_rejects_normalized_incompatibilities(
    monkeypatch, overrides, message
):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(RuntimeError, match=message):
        MegaTrainPlugin().pre_model_load(_runtime_config(**overrides))


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        (SimpleNamespace(model_type="gpt2"), "model types"),
        (
            SimpleNamespace(model_type="llama", quantization_config={}),
            "quantization metadata",
        ),
        (
            LlamaConfig(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                attention_dropout=0.1,
            ),
            "attention_dropout",
        ),
        (
            SimpleNamespace(
                model_type="mistral",
                layer_types=["full_attention", "sliding_attention"],
            ),
            "mixed attention",
        ),
        (
            SimpleNamespace(
                model_type="llama",
                layer_types=["linear_attention"],
            ),
            "linear_attention",
        ),
        (
            SimpleNamespace(
                model_type="llama",
                layer_types=["full_attention"],
                attention_chunk_size=64,
            ),
            "chunked attention",
        ),
    ],
)
def test_pre_model_load_rejects_unsupported_model_configs(
    monkeypatch, model_config, message
):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        "axolotl.loaders.utils.load_model_config", lambda _cfg: model_config
    )
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(RuntimeError, match=message):
        MegaTrainPlugin().pre_model_load(_runtime_config())


def test_pre_model_load_validates_stock_path_then_forces_cpu_placement(monkeypatch):
    import axolotl.integrations.megatrain as megatrain

    plugin = megatrain.MegaTrainPlugin()
    PluginManager.get_instance().plugins[PLUGIN] = plugin
    model_config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        "axolotl.loaders.utils.load_model_config", lambda _cfg: model_config
    )
    monkeypatch.setattr(
        "axolotl.monkeypatch.attention.sdpa_varlen.unpatch_sdpa_varlen",
        lambda: events.append("unpatch_sdpa_varlen"),
    )
    monkeypatch.setattr(
        megatrain,
        "_assert_unpatched_model_classes",
        lambda model_type: events.append(("model_classes", model_type)),
    )
    monkeypatch.setattr(
        megatrain,
        "_assert_flash_attention_2_is_unpatched",
        lambda implementation: events.append(("flash_attention", implementation)),
    )
    monkeypatch.setattr(
        megatrain,
        "_assert_cross_entropy_is_unpatched",
        lambda: events.append("cross_entropy"),
    )
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    config = _runtime_config(device_map=None, experimental_skip_move_to_device=False)

    plugin.pre_model_load(config)

    assert events == [
        "unpatch_sdpa_varlen",
        ("model_classes", "llama"),
        ("flash_attention", "eager"),
        "cross_entropy",
    ]
    assert config.device_map == "cpu"
    assert config.experimental_skip_move_to_device is True


@pytest.mark.parametrize(
    ("unsupported", "message"),
    [
        ({"fsdp_config": {}}, "fsdp_config"),
        ({"do_bench_eval": True}, "do_bench_eval"),
    ],
)
def test_full_load_cfg_rejects_normalization_bypasses(
    monkeypatch, unsupported, message
):
    monkeypatch.setattr(
        "axolotl.cli.config.gpu_capabilities",
        lambda: (
            {
                "bf16": True,
                "compute_capability": "sm_89",
                "fp8": False,
                "n_gpu": 1,
                "n_node": 1,
                "tf32": True,
            },
            {"torch_version": torch.__version__.split("+")[0]},
        ),
    )
    config = DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "plugins": [PLUGIN],
            "datasets": [{"path": "unused", "type": "alpaca"}],
            "sequence_len": 32,
            "max_steps": 1,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "bf16": True,
            "sample_packing": False,
            "val_set_size": 0,
            **unsupported,
        }
    )

    with pytest.raises(ValueError, match=message):
        load_cfg(config)


def test_stale_plugin_is_inert_for_stock_config():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    manager = PluginManager.get_instance()
    plugin = MegaTrainPlugin()
    manager.plugins[PLUGIN] = plugin
    stock_config = DictDefault({"plugins": []})
    raw_stock_config = {"plugins": [], "device_map": "auto"}
    model = Mock()

    manager.cfg = DictDefault({"plugins": [PLUGIN]})
    plugin.register(raw_stock_config)
    plugin.post_model_load(stock_config, model)

    assert raw_stock_config["device_map"] == "auto"
    assert manager.get_trainer_cls(stock_config) is None
    assert manager.get_training_args(stock_config) == {}
    assert manager.create_optimizer(Mock()) is None
    model.to.assert_not_called()


def test_conflicting_stale_trainer_plugin_is_rejected(monkeypatch):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    class ConflictingPlugin(BasePlugin):
        def get_trainer_cls(self, cfg):
            return object

    manager = PluginManager.get_instance()
    plugin = MegaTrainPlugin()
    manager.plugins["conflict.Plugin"] = ConflictingPlugin()
    manager.plugins[PLUGIN] = plugin
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    config = DictDefault(
        {
            "plugins": [PLUGIN],
            "world_size": 1,
            "torch_dtype": torch.bfloat16,
            "is_multimodal": False,
            "attn_implementation": "sdpa",
            "optimizer": "adamw_torch",
        }
    )

    with pytest.raises(RuntimeError, match="stale registered plugin"):
        plugin.pre_model_load(config)


def test_configured_trainer_plugin_cannot_take_precedence(monkeypatch):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    class ConflictingPlugin(BasePlugin):
        def get_trainer_cls(self, cfg):
            return object

    manager = PluginManager.get_instance()
    plugin = MegaTrainPlugin()
    manager.plugins["conflict.Plugin"] = ConflictingPlugin()
    manager.plugins[PLUGIN] = plugin
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(RuntimeError, match="selected as the trainer"):
        plugin.pre_model_load(_runtime_config(plugins=["conflict.Plugin", PLUGIN]))


def test_quantized_model_is_rejected_before_dtype_move():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    model.is_quantized = True
    model.to = Mock()

    with pytest.raises(RuntimeError, match="quantized"):
        MegaTrainPlugin().post_model_load(SimpleNamespace(plugins=[PLUGIN]), model)

    model.to.assert_not_called()


def test_empty_checkpoint_quantization_metadata_is_rejected():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    model.config.quantization_config = {}

    with pytest.raises(RuntimeError, match="quantized"):
        MegaTrainPlugin().post_model_load(SimpleNamespace(plugins=[PLUGIN]), model)


def test_functional_dropout_in_loaded_model_config_is_rejected():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0.1,
        )
    )

    with pytest.raises(RuntimeError, match="attention_dropout"):
        MegaTrainPlugin().post_model_load(SimpleNamespace(plugins=[PLUGIN]), model)


def test_dropout_module_is_rejected_even_when_model_config_is_zero():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    model.extra_dropout = torch.nn.Dropout(p=0.1)

    with pytest.raises(RuntimeError, match="extra_dropout"):
        MegaTrainPlugin().post_model_load(SimpleNamespace(plugins=[PLUGIN]), model)


def test_encoder_decoder_flag_is_rejected_on_supported_model_class():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    model.config.is_encoder_decoder = True

    with pytest.raises(RuntimeError, match="decoder-only"):
        MegaTrainPlugin().post_model_load(SimpleNamespace(plugins=[PLUGIN]), model)


def test_post_model_load_verifies_fp32_conversion_succeeded():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).to(dtype=torch.bfloat16)
    model.to = Mock(return_value=model)

    with pytest.raises(RuntimeError, match="FP32"):
        MegaTrainPlugin().post_model_load(SimpleNamespace(plugins=[PLUGIN]), model)


def test_post_model_load_verifies_cpu_placement_succeeded():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    with torch.device("meta"):
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        )
    model.to = Mock(return_value=model)

    with pytest.raises(RuntimeError, match="loaded on CPU"):
        MegaTrainPlugin().post_model_load(SimpleNamespace(plugins=[PLUGIN]), model)


def test_custom_subclass_of_supported_model_is_rejected():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    class CustomLlamaForCausalLM(LlamaForCausalLM):
        pass

    model = CustomLlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )

    with pytest.raises(RuntimeError, match="CustomLlamaForCausalLM"):
        MegaTrainPlugin().post_model_load(SimpleNamespace(plugins=[PLUGIN]), model)


@pytest.mark.parametrize(
    ("config", "model_cls"),
    [
        (
            LlamaConfig(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                tie_word_embeddings=True,
            ),
            LlamaForCausalLM,
        ),
        (
            MistralConfig(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                tie_word_embeddings=True,
            ),
            MistralForCausalLM,
        ),
    ],
    ids=["llama", "mistral"],
)
def test_post_model_load_builds_fp32_cpu_master(config, model_cls):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    model = model_cls(config).to(dtype=torch.bfloat16)

    MegaTrainPlugin().post_model_load(SimpleNamespace(plugins=[PLUGIN]), model)

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(
        parameter.dtype == torch.float32
        for parameter in model.parameters()
        if parameter.is_floating_point()
    )
    floating_buffers = {
        name: buffer
        for name, buffer in model.named_buffers()
        if buffer.is_floating_point()
    }
    assert floating_buffers
    assert all(buffer.dtype == torch.float32 for buffer in floating_buffers.values())
    assert any(name.endswith("inv_freq") for name in floating_buffers)
    assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("use_pose", True),
        ("tiled_mlp", True),
        ("flash_attn_fuse_mlp", True),
        ("flash_optimum", True),
        ("lora_qkv_kernel", True),
        ("sdpa_varlen", True),
        ("large_head_attention", "auto"),
        ("flash_attn_d512", True),
        ("fused_attn_kernel", True),
    ],
)
def test_normalized_runtime_rejects_model_patch_settings(setting, value):
    from axolotl.integrations.megatrain import _unsupported_model_patch

    assert _unsupported_model_patch(DictDefault({setting: value})) == setting


def test_persistent_model_forward_patch_is_rejected(monkeypatch):
    from transformers.models.llama.modeling_llama import LlamaMLP

    from axolotl.integrations.megatrain import _assert_unpatched_model_classes

    def patched_forward(self, hidden_states):
        return hidden_states

    patched_forward.__module__ = "axolotl.monkeypatch.test"
    monkeypatch.setattr(LlamaMLP, "forward", patched_forward)

    with pytest.raises(RuntimeError, match="persistent model-forward"):
        _assert_unpatched_model_classes("llama")


def test_replaced_model_component_class_is_rejected(monkeypatch):
    from transformers.models.llama import modeling_llama

    from axolotl.integrations.megatrain import _assert_unpatched_model_classes

    class ForeignLlamaMLP(modeling_llama.LlamaMLP):
        pass

    monkeypatch.setattr(modeling_llama, "LlamaMLP", ForeignLlamaMLP)

    with pytest.raises(RuntimeError, match="persistent model-forward"):
        _assert_unpatched_model_classes("llama")


def test_persistent_rotary_embedding_patch_is_rejected(monkeypatch):
    from transformers.models.llama import modeling_llama

    from axolotl.integrations.megatrain import _assert_unpatched_model_classes

    def patched_rotary(*args, **kwargs):
        del args, kwargs

    patched_rotary.__module__ = "axolotl.monkeypatch.test"
    monkeypatch.setattr(modeling_llama, "apply_rotary_pos_emb", patched_rotary)

    with pytest.raises(RuntimeError, match="persistent rotary-embedding"):
        _assert_unpatched_model_classes("llama")


def test_stock_hub_rotary_kernel_is_accepted(monkeypatch):
    from transformers.models.llama import modeling_llama

    from axolotl.integrations.megatrain import _assert_unpatched_model_classes

    def hub_rotary(*args, **kwargs):
        del args, kwargs

    hub_rotary.__module__ = "kernels.layer.func"
    hub_rotary.kernel_layer_name = "rotary_pos_emb"
    monkeypatch.setattr(modeling_llama, "apply_rotary_pos_emb", hub_rotary)

    _assert_unpatched_model_classes("llama")


def test_flash_attention_4_patch_is_rejected(monkeypatch):
    import transformers.modeling_flash_attention_utils as flash_attention_utils

    from axolotl.integrations.megatrain import (
        _assert_flash_attention_2_is_unpatched,
    )

    monkeypatch.setattr(
        flash_attention_utils._lazy_imports,
        "_axolotl_patched",
        True,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="Flash Attention 4"):
        _assert_flash_attention_2_is_unpatched("flash_attention_2")


def test_flash_attention_4_marker_is_ignored_for_other_backends(monkeypatch):
    import transformers.modeling_flash_attention_utils as flash_attention_utils

    from axolotl.integrations.megatrain import (
        _assert_flash_attention_2_is_unpatched,
    )

    monkeypatch.setattr(
        flash_attention_utils._lazy_imports,
        "_axolotl_patched",
        True,
        raising=False,
    )

    _assert_flash_attention_2_is_unpatched("eager")


def test_persistent_cross_entropy_patch_is_rejected(monkeypatch):
    from axolotl.integrations.megatrain import _assert_cross_entropy_is_unpatched

    def patched_cross_entropy(*args, **kwargs):
        del args, kwargs

    patched_cross_entropy.__module__ = "liger_kernel.transformers.functional"
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", patched_cross_entropy)

    with pytest.raises(RuntimeError, match="persistent cross-entropy"):
        _assert_cross_entropy_is_unpatched()


@pytest.mark.parametrize("plugin_name", [PLUGIN, "megatrain.MegaTrainPlugin"])
def test_patch_manager_does_not_auto_upgrade_megatrain_to_fa4(monkeypatch, plugin_name):
    from axolotl.loaders.patch_manager import PatchManager
    from axolotl.monkeypatch.attention import flash_attn_4

    patch = Mock()
    monkeypatch.setattr(flash_attn_4, "patch_flash_attn_4", patch)
    # Core consults registered plugins, so mirror what `prepare_plugins` does.
    PluginManager.get_instance().register(plugin_name)
    patch_manager = object.__new__(PatchManager)
    patch_manager.cfg = SimpleNamespace(
        attn_uses_flash_lib=True,
        plugins=[plugin_name],
    )
    patch_manager.model_config = LlamaConfig()

    patch_manager._apply_flash_attn_4_patches()

    patch.assert_not_called()


def test_patch_manager_preserves_stock_fa4_upgrade(monkeypatch):
    from axolotl.loaders.patch_manager import PatchManager
    from axolotl.monkeypatch.attention import flash_attn_4

    patch = Mock()
    model_config = LlamaConfig()
    monkeypatch.setattr(flash_attn_4, "patch_flash_attn_4", patch)
    patch_manager = object.__new__(PatchManager)
    patch_manager.cfg = SimpleNamespace(
        attn_uses_flash_lib=True,
        plugins=[],
    )
    patch_manager.model_config = model_config

    patch_manager._apply_flash_attn_4_patches()

    patch.assert_called_once_with(model_config)


def test_active_plugin_hook_contracts():
    from axolotl.integrations.megatrain import MegaTrainPlugin
    from axolotl.integrations.megatrain.trainer import MegaTrainTrainer

    plugin = MegaTrainPlugin()
    config = _runtime_config()

    assert plugin.get_trainer_cls(config) is MegaTrainTrainer
    assert plugin.get_training_args(config) == {
        "use_cpu": True,
        "bf16": False,
        "fp16": False,
        "tf32": False,
        "gradient_checkpointing": False,
        "torch_compile": False,
        "use_liger_kernel": False,
    }
    assert plugin.create_optimizer(config, Mock()) is None

    with pytest.raises(RuntimeError, match="different trainer"):
        plugin.post_trainer_create(config, Mock())


def test_stale_hook_only_plugin_is_rejected_before_model_load(monkeypatch):
    from axolotl.integrations.megatrain import MegaTrainPlugin

    class HookOnlyPlugin(BasePlugin):
        pass

    manager = PluginManager.get_instance()
    plugin = MegaTrainPlugin()
    manager.plugins[PLUGIN] = plugin
    manager.plugins["stale.HookOnlyPlugin"] = HookOnlyPlugin()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    config = DictDefault(
        {
            "plugins": [PLUGIN],
            "world_size": 1,
            "torch_dtype": torch.bfloat16,
            "is_multimodal": False,
            "attn_implementation": "eager",
            "optimizer": "adamw_torch",
        }
    )

    with pytest.raises(RuntimeError, match="stale registered plugin"):
        plugin.pre_model_load(config)


def test_optimizer_groups_match_axolotl_decay_policy():
    from axolotl.integrations.megatrain import MegaTrainPlugin

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
    )
    parameters = list(model.parameters())
    cpu_master = SimpleNamespace(
        get_parameters=lambda: parameters,
        _sync_params_to_gpu=Mock(),
    )
    from axolotl.integrations.megatrain.trainer import MegaTrainTrainer

    trainer = object.__new__(MegaTrainTrainer)
    trainer.model = model
    trainer.cpu_master = cpu_master
    trainer.get_decay_parameter_names = lambda _: {
        name for name, parameter in model.named_parameters() if parameter.ndim > 1
    }
    config = DictDefault(
        {
            "plugins": [PLUGIN],
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
        }
    )

    optimizer = MegaTrainPlugin().create_optimizer(config, trainer)
    decay_by_id = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert decay_by_id[id(model.model.layers[0].self_attn.q_proj.weight)] == 0.1
    assert decay_by_id[id(model.model.layers[0].input_layernorm.weight)] == 0.0
    assert decay_by_id[id(model.model.embed_tokens.weight)] == 0.0
    assert decay_by_id[id(model.lm_head.weight)] == 0.0
