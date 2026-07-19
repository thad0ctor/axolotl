# SPDX-License-Identifier: Apache-2.0

"""Configuration schema for the MegaTrain integration."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, model_validator

MEGATRAIN_PLUGIN_NAMES = {
    "axolotl.integrations.megatrain.MegaTrainPlugin",
    "megatrain.MegaTrainPlugin",
}


def _megatrain_enabled(data: Mapping[str, Any] | Any) -> bool:
    plugins = (
        data.get("plugins") if hasattr(data, "get") else getattr(data, "plugins", None)
    ) or []
    return any(str(plugin) in MEGATRAIN_PLUGIN_NAMES for plugin in plugins)


class MegaTrainArgs(BaseModel):
    """Top-level MegaTrain configuration fields and compatibility checks."""

    megatrain_checkpoint_interval: int | None = Field(default=None, ge=1)
    megatrain_num_grad_slabs: int | None = Field(default=None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def validate_compatibility(cls, data):
        if not isinstance(data, Mapping) or not _megatrain_enabled(data):
            return data

        def reject(key: str, reason: str, replacement: str) -> None:
            raise ValueError(
                f"MegaTrain does not support `{key}` because {reason}. {replacement}"
            )

        configured_plugins = [str(plugin) for plugin in data.get("plugins") or []]
        megatrain_plugins = [
            plugin
            for plugin in configured_plugins
            if plugin in MEGATRAIN_PLUGIN_NAMES
        ]
        if len(megatrain_plugins) > 1:
            reject(
                "plugins",
                "MegaTrain must be registered exactly once",
                "Keep one canonical or shorthand MegaTrain plugin entry.",
            )

        other_plugins = [
            str(plugin)
            for plugin in configured_plugins
            if str(plugin) not in MEGATRAIN_PLUGIN_NAMES
        ]
        if other_plugins:
            reject(
                "plugins",
                "model, trainer, optimizer, and loss hooks from other plugins have not "
                "been validated with CPU-master streaming",
                f"Remove the other plugin(s): {', '.join(other_plugins)}.",
            )

        if data.get("adapter") is not None or data.get("peft") is not None:
            reject(
                "adapter/peft",
                "adapters are incompatible with full-parameter CPU-master streaming",
                "Remove the adapter or PEFT configuration.",
            )

        for key in (
            "load_in_4bit",
            "load_in_8bit",
            "gptq",
            "quantize_moe_experts",
            "fp8",
            "use_onebitllms",
        ):
            if data.get(key):
                reject(
                    key,
                    "quantized parameters cannot initialize FP32 CPU masters and BF16 streamed copies",
                    f"Set `{key}: false`.",
                )

        for key in (
            "model_quantization_config",
            "model_quantization_config_kwargs",
            "qat",
            "quantization",
        ):
            if key in data and data.get(key) is not None:
                reject(
                    key,
                    "training and loading quantization are outside the FP32-master full-finetune path",
                    f"Remove `{key}`.",
                )

        for key in ("fsdp", "fsdp_config", "deepspeed"):
            if key in data and data.get(key) is not None:
                reject(
                    key,
                    "distributed parameter sharding conflicts with MegaTrain's CPU master",
                    f"Remove `{key}` and launch one Axolotl process.",
                )

        if data.get("ddp"):
            reject(
                "ddp",
                "MegaTrain currently supports one Axolotl process",
                "Set `ddp: false` and do not use torchrun.",
            )

        for key in ("tensor_parallel_size", "context_parallel_size"):
            if (data.get(key) or 1) > 1:
                reject(
                    key,
                    "Axolotl parallelism conflicts with layer streaming",
                    f"Set `{key}: 1`.",
                )

        if data.get("sample_packing"):
            reject(
                "sample_packing",
                "the streamed attention path expects an ordinary padded causal mask",
                "Set `sample_packing: false`.",
            )

        if data.get("batch_flattening"):
            reject(
                "batch_flattening",
                "the streamed path does not consume flattened sequence boundaries or their position IDs",
                "Set `batch_flattening: false`.",
            )

        relora_keys = (
            "relora",
            "relora_prune_ratio",
            "relora_prune_method",
            "relora_cpu_offload",
            "jagged_restart_steps",
            "jagged_restart_warmup_steps",
            "jagged_restart_anneal_steps",
        )
        for key in relora_keys:
            if key in data and data.get(key) is not None:
                reject(
                    key,
                    "ReLoRA changes adapter and optimizer state outside the CPU-master loop",
                    "Remove all ReLoRA and jagged-restart settings.",
                )

        if float(data.get("val_set_size") or 0) > 0 or data.get("test_datasets"):
            reject(
                "val_set_size/test_datasets",
                "the MegaTrain trainer does not implement evaluation",
                "Set `val_set_size: 0` and remove `test_datasets`.",
            )

        for key in (
            "do_bench_eval",
            "do_causal_lm_eval",
            "generate_samples",
            "load_best_model_at_end",
        ):
            if data.get(key):
                reject(
                    key,
                    "the MegaTrain trainer does not implement evaluation or generation callbacks",
                    f"Set `{key}: false`.",
                )

        for key in ("eval_steps", "evals_per_epoch"):
            if key in data and data.get(key) is not None:
                reject(
                    key,
                    "the MegaTrain trainer does not implement evaluation",
                    f"Remove `{key}`.",
                )

        for key in ("eval_strategy", "evaluation_strategy"):
            if data.get(key) not in (None, "no"):
                reject(
                    key,
                    "the MegaTrain trainer does not implement evaluation",
                    f"Set `{key}: no` or remove it.",
                )

        for key in (
            "rl",
            "reward_model",
            "process_reward_model",
            "pretraining_dataset",
        ):
            if data.get(key):
                reject(
                    key,
                    "this integration currently implements supervised fine-tuning only",
                    f"Remove `{key}` and use an SFT dataset.",
                )

        if data.get("use_ray"):
            reject(
                "use_ray",
                "Ray workers conflict with the single-process CUDA runtime",
                "Set `use_ray: false`.",
            )

        if data.get("use_eaft"):
            reject(
                "use_eaft",
                "MegaTrain computes token cross entropy inside its streamed backward pass",
                "Set `use_eaft: false`.",
            )

        for key in (
            "flash_attn_cross_entropy",
            "chunked_cross_entropy",
            "cut_cross_entropy",
            "use_liger_kernel",
            "use_kernels",
            "use_pose",
            "tiled_mlp",
            "flash_attn_fuse_mlp",
            "flash_optimum",
            "lora_mlp_kernel",
            "lora_qkv_kernel",
            "lora_o_kernel",
            "lora_embedding_kernel",
            "flash_attn_d512",
            "fused_attn_kernel",
        ):
            if data.get(key):
                reject(
                    key,
                    "external loss or model-kernel patches bypass the validated streamed path",
                    f"Set `{key}: false` and remove its plugin if configured.",
                )

        for key in (
            "pose_split_on_token_ids",
            "pose_max_context_len",
            "pose_num_chunks",
            "tiled_mlp_num_shards",
            "tiled_mlp_use_original_mlp",
        ):
            if key in data and data.get(key) is not None:
                reject(
                    key,
                    "the option configures a model-forward transformation outside the validated streamed path",
                    f"Remove `{key}`.",
                )

        if "sdpa_varlen" in data and data.get("sdpa_varlen") is not None:
            reject(
                "sdpa_varlen",
                "the variable-length SDPA patch is outside the validated streamed attention path",
                "Remove `sdpa_varlen`.",
            )

        if data.get("large_head_attention") not in (None, "sdpa"):
            reject(
                "large_head_attention",
                "the Triton large-head router is outside the validated streamed attention path",
                "Remove it or set `large_head_attention: sdpa`.",
            )

        if any(key.startswith("liger_") and data.get(key) for key in data):
            reject(
                "liger_*",
                "Liger model and loss patches have not been validated with streamed layer templates",
                "Remove all Liger settings and its plugin.",
            )

        for key in (
            "gradient_checkpointing",
            "selective_checkpointing",
            "activation_offloading",
            "layer_offloading",
        ):
            if data.get(key):
                reject(
                    key,
                    "MegaTrain performs its own layer checkpointing and activation streaming",
                    f"Set `{key}: false`.",
                )

        for key in ("unfrozen_parameters", "freeze_mm_modules", "fp32_norms"):
            if data.get(key):
                reject(
                    key,
                    "partial freezing or mixed parameter dtypes are outside FP32-master fine-tuning",
                    f"Remove `{key}`.",
                )

        for key in ("lisa_n_layers", "lisa_step_interval"):
            if key in data and data.get(key) is not None:
                reject(
                    key,
                    "LISA changes trainable layers after the CPU master has mapped parameters",
                    "Remove all LISA settings.",
                )

        for key in ("neftune_noise_alpha", "noisy_embedding_alpha"):
            if key in data and data.get(key) is not None:
                reject(
                    key,
                    "the embedding noise hook is not installed on MegaTrain's streamed GPU copy",
                    f"Remove `{key}`.",
                )

        if data.get("fp16") or data.get("float16") or data.get("float32"):
            reject(
                "fp16/float16/float32",
                "the streamed runtime uses BF16 compute with FP32 CPU master parameters",
                "Set `bf16: true` and remove the other dtype flags.",
            )

        device_map = data.get("device_map")
        if device_map not in (None, "cpu", {"": "cpu"}):
            reject(
                "device_map",
                "the Hugging Face model must be loaded wholly on CPU",
                "Remove `device_map` or set it to `cpu`.",
            )
        for key in ("max_memory", "gpu_memory_limit"):
            if data.get(key) is not None:
                reject(
                    key,
                    "automatic model dispatch can place master parameters on CUDA",
                    f"Remove `{key}`.",
                )

        optimizer = data.get("optimizer")
        if optimizer is not None and str(optimizer) != "adamw_torch":
            reject(
                "optimizer",
                "the integration currently provides only unfused CPU AdamW",
                "Set `optimizer: adamw_torch`.",
            )

        for key in (
            "optim_args",
            "optim_target_modules",
            "embedding_lr",
            "embedding_lr_scale",
            "lr_groups",
        ):
            if key in data and data.get(key) is not None:
                reject(
                    key,
                    "the setting changes optimizer groups outside the supported CPU AdamW contract",
                    f"Remove `{key}`.",
                )

        if data.get("trainer_cls"):
            reject(
                "trainer_cls",
                "MegaTrain must own forward, backward, and optimizer synchronization",
                "Remove `trainer_cls`.",
            )

        if data.get("trust_remote_code"):
            reject(
                "trust_remote_code",
                "custom model forwards may apply transformations that the streamed path cannot discover",
                "Set `trust_remote_code: false` and use a supported Transformers model class.",
            )

        if data.get("torch_compile"):
            warnings.warn(
                "MegaTrain disables `torch_compile`; streamed layer templates are not "
                "compatible with a compiled whole-model graph.",
                UserWarning,
                stacklevel=2,
            )

        return data

    @model_validator(mode="after")
    def validate_buffer_counts(self):
        checkpoint_interval = self.megatrain_checkpoint_interval or 4
        num_grad_slabs = self.megatrain_num_grad_slabs or 12
        if num_grad_slabs < 2 * checkpoint_interval:
            raise ValueError(
                "`megatrain_num_grad_slabs` must be at least twice "
                "`megatrain_checkpoint_interval` to prevent gradient-buffer starvation."
            )
        return self
