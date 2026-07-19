"""CPU-only validation tests for the MegaTrain config surface."""

import pytest
from pydantic import ValidationError

from axolotl.integrations.megatrain.args import MegaTrainArgs

PLUGIN = "axolotl.integrations.megatrain.MegaTrainPlugin"


@pytest.mark.parametrize(
    ("unsupported", "message"),
    [
        ({"adapter": "lora"}, "adapter"),
        ({"peft": {"loftq_config": {}}}, "peft"),
        ({"load_in_8bit": True}, "load_in_8bit"),
        ({"load_in_4bit": True}, "load_in_4bit"),
        ({"gptq": True}, "gptq"),
        ({"model_quantization_config_kwargs": {}}, "model_quantization_config_kwargs"),
        ({"quantization": {}}, "quantization"),
        ({"fsdp": ["full_shard"]}, "fsdp"),
        ({"fsdp_config": {}}, "fsdp_config"),
        ({"fsdp_config": {"offload_params": True}}, "fsdp_config"),
        ({"deepspeed": "deepspeed_configs/zero2.json"}, "deepspeed"),
        ({"ddp": True}, "ddp"),
        ({"tensor_parallel_size": 2}, "tensor_parallel_size"),
        ({"context_parallel_size": 2}, "context_parallel_size"),
        ({"model_quantization_config": "Mxfp4Config"}, "model_quantization_config"),
        ({"qat": {"weight_dtype": "int8"}}, "qat"),
        ({"fp8": True}, "fp8"),
        ({"use_onebitllms": True}, "use_onebitllms"),
        ({"sample_packing": True}, "sample_packing"),
        ({"batch_flattening": True}, "batch_flattening"),
        ({"batch_flattening": "auto"}, "batch_flattening"),
        ({"relora": True}, "relora"),
        ({"relora_prune_ratio": 0.5}, "relora_prune_ratio"),
        ({"relora_prune_method": "random"}, "relora_prune_method"),
        ({"relora_cpu_offload": True}, "relora_cpu_offload"),
        ({"jagged_restart_steps": 10}, "jagged_restart_steps"),
        ({"val_set_size": 0.1}, "val_set_size"),
        ({"test_datasets": [{"path": "unused", "type": "alpaca"}]}, "test_datasets"),
        ({"do_bench_eval": True}, "do_bench_eval"),
        ({"do_causal_lm_eval": True}, "do_causal_lm_eval"),
        ({"generate_samples": True}, "generate_samples"),
        ({"load_best_model_at_end": True}, "load_best_model_at_end"),
        ({"eval_steps": 5}, "eval_steps"),
        ({"eval_strategy": "steps"}, "eval_strategy"),
        ({"rl": "dpo"}, "rl"),
        ({"reward_model": True}, "reward_model"),
        ({"pretraining_dataset": [{"path": "unused"}]}, "pretraining_dataset"),
        ({"use_ray": True}, "use_ray"),
        ({"use_eaft": True}, "use_eaft"),
        ({"flash_attn_cross_entropy": True}, "flash_attn_cross_entropy"),
        ({"chunked_cross_entropy": True}, "chunked_cross_entropy"),
        ({"cut_cross_entropy": True}, "cut_cross_entropy"),
        ({"use_liger_kernel": True}, "use_liger_kernel"),
        ({"liger_rope": True}, "liger"),
        ({"use_pose": True}, "use_pose"),
        ({"pose_num_chunks": 4}, "pose_num_chunks"),
        ({"tiled_mlp": True}, "tiled_mlp"),
        ({"tiled_mlp_num_shards": 4}, "tiled_mlp_num_shards"),
        ({"flash_attn_fuse_mlp": True}, "flash_attn_fuse_mlp"),
        ({"flash_optimum": True}, "flash_optimum"),
        ({"lora_mlp_kernel": True}, "lora_mlp_kernel"),
        ({"lora_qkv_kernel": True}, "lora_qkv_kernel"),
        ({"lora_o_kernel": True}, "lora_o_kernel"),
        ({"lora_embedding_kernel": True}, "lora_embedding_kernel"),
        ({"sdpa_varlen": True}, "sdpa_varlen"),
        ({"sdpa_varlen": False}, "sdpa_varlen"),
        ({"large_head_attention": "auto"}, "large_head_attention"),
        ({"flash_attn_d512": True}, "flash_attn_d512"),
        ({"fused_attn_kernel": True}, "fused_attn_kernel"),
        ({"gradient_checkpointing": True}, "gradient_checkpointing"),
        ({"selective_checkpointing": True}, "selective_checkpointing"),
        ({"activation_offloading": True}, "activation_offloading"),
        ({"layer_offloading": True}, "layer_offloading"),
        ({"unfrozen_parameters": ["lm_head.weight"]}, "unfrozen_parameters"),
        ({"freeze_mm_modules": True}, "freeze_mm_modules"),
        ({"fp32_norms": True}, "fp32_norms"),
        ({"lisa_n_layers": 2}, "lisa_n_layers"),
        ({"neftune_noise_alpha": 0.1}, "neftune_noise_alpha"),
        ({"float16": True}, "float16"),
        ({"device_map": "auto"}, "device_map"),
        ({"max_memory": {0: "10GiB"}}, "max_memory"),
        ({"optimizer": "adamw_torch_fused"}, "optimizer"),
        ({"optim_args": "momentum=0.9"}, "optim_args"),
        ({"embedding_lr": 1e-5}, "embedding_lr"),
        ({"trainer_cls": "custom.Trainer"}, "trainer_cls"),
        ({"trust_remote_code": True}, "trust_remote_code"),
        (
            {"plugins": [PLUGIN, "axolotl.integrations.liger.LigerPlugin"]},
            "plugins",
        ),
    ],
)
def test_rejects_unsupported_training_modes(unsupported, message):
    with pytest.raises(ValidationError, match=message):
        MegaTrainArgs.model_validate({"plugins": [PLUGIN], **unsupported})


def test_guardrails_are_inactive_without_megatrain_plugin():
    config = MegaTrainArgs.model_validate(
        {
            "plugins": ["axolotl.integrations.kd.KDPlugin"],
            "adapter": "lora",
            "sample_packing": True,
            "val_set_size": 0.1,
        }
    )

    assert config.megatrain_checkpoint_interval is None


def test_torch_compile_warns():
    with pytest.warns(UserWarning, match="torch_compile"):
        MegaTrainArgs.model_validate(
            {
                "plugins": [PLUGIN],
                "torch_compile": True,
            }
        )


def test_similar_plugin_name_does_not_enable_guardrails():
    config = MegaTrainArgs.model_validate(
        {
            "plugins": ["third_party.NotReallyMegaTrainPlugin"],
            "adapter": "lora",
        }
    )

    assert config.megatrain_checkpoint_interval is None


def test_rejects_duplicate_megatrain_aliases():
    with pytest.raises(ValidationError, match="exactly once"):
        MegaTrainArgs.model_validate(
            {
                "plugins": [PLUGIN, "megatrain.MegaTrainPlugin"],
            }
        )


def test_shorthand_alias_enables_guardrails():
    with pytest.raises(ValidationError, match="sample_packing"):
        MegaTrainArgs.model_validate(
            {
                "plugins": ["megatrain.MegaTrainPlugin"],
                "sample_packing": True,
            }
        )


@pytest.mark.parametrize(
    "values",
    [
        {"megatrain_checkpoint_interval": 0},
        {"megatrain_num_grad_slabs": 0},
        {
            "megatrain_checkpoint_interval": 7,
            "megatrain_num_grad_slabs": 13,
        },
    ],
)
def test_rejects_invalid_buffer_configuration(values):
    with pytest.raises(ValidationError):
        MegaTrainArgs.model_validate({"plugins": [PLUGIN], **values})


def test_accepts_exact_gradient_slab_boundary():
    config = MegaTrainArgs.model_validate(
        {
            "plugins": [PLUGIN],
            "megatrain_checkpoint_interval": 7,
            "megatrain_num_grad_slabs": 14,
        }
    )

    assert config.megatrain_checkpoint_interval == 7
    assert config.megatrain_num_grad_slabs == 14
