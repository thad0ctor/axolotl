# SPDX-License-Identifier: Apache-2.0

"""MegaTrain integration for Axolotl."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping

import torch
from torch import nn

from axolotl.integrations.base import BasePlugin, PluginManager

from .args import (
    DEFAULT_ATTN_IMPLEMENTATION,
    MEGATRAIN_PLUGIN_NAMES,
    SUPPORTED_ATTN_IMPLEMENTATIONS,
    MegaTrainArgs as MegaTrainArgs,
    _megatrain_enabled,
)

_SUPPORTED_MODEL_TYPES = {"llama", "mistral"}
_MODEL_PATCH_FLAGS = (
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
    "flash_attn_cross_entropy",
    "flash_attn_cross_entropy_loss",
    "chunked_cross_entropy",
    "cut_cross_entropy",
    "use_liger_kernel",
    "use_kernels",
)


def _cfg_value(cfg, key: str, default=None):
    value = cfg.get(key) if hasattr(cfg, "get") else getattr(cfg, key, None)
    return default if value is None else value


def _nonzero_dropout_fields(config) -> list[str]:
    values = config.to_dict() if hasattr(config, "to_dict") else config
    found = []

    def visit(value, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if (
                    "dropout" in str(key).lower()
                    and isinstance(child, (int, float))
                    and float(child) > 0
                ):
                    found.append(child_path)
                visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(values, "")
    return found


def _unsupported_model_patch(cfg) -> str | None:
    for key in _MODEL_PATCH_FLAGS:
        if _cfg_value(cfg, key, False):
            return key
    if _cfg_value(cfg, "sdpa_varlen") is True:
        return "sdpa_varlen"
    if _cfg_value(cfg, "large_head_attention") not in (None, "sdpa"):
        return "large_head_attention"
    return None


def _assert_unpatched_model_classes(model_type: str) -> None:
    module = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )
    expected_module = module.__name__
    prefix = "Llama" if model_type == "llama" else "Mistral"
    for class_name in (
        f"{prefix}Attention",
        f"{prefix}MLP",
        f"{prefix}RMSNorm",
        f"{prefix}DecoderLayer",
    ):
        component_class = getattr(module, class_name)
        if (
            component_class.__module__ != expected_module
            or component_class.forward.__module__ != expected_module
        ):
            raise RuntimeError(
                "MegaTrain detected a persistent model-forward patch from a previous "
                "run. Start this training run in a fresh Python process."
            )

    rotary = module.apply_rotary_pos_emb
    rotary_module = getattr(rotary, "__module__", type(rotary).__module__)
    stock_hub_kernel = (
        rotary_module == "kernels.layer.func"
        and getattr(rotary, "kernel_layer_name", None) == "rotary_pos_emb"
    )
    if rotary_module != expected_module and not stock_hub_kernel:
        raise RuntimeError(
            "MegaTrain detected a persistent rotary-embedding patch from a previous "
            "run. Start this training run in a fresh Python process."
        )


def _assert_flash_attention_2_is_unpatched(attention_implementation: str) -> None:
    if attention_implementation != "flash_attention_2":
        return
    import transformers.modeling_flash_attention_utils as flash_attention_utils

    if getattr(flash_attention_utils._lazy_imports, "_axolotl_patched", False):
        raise RuntimeError(
            "MegaTrain supports Flash Attention 2, but this Python process was already "
            "patched to use Flash Attention 4. Start the run in a fresh process."
        )


def _assert_cross_entropy_is_unpatched() -> None:
    if nn.functional.cross_entropy.__module__ != "torch.nn.functional":
        raise RuntimeError(
            "MegaTrain detected a persistent cross-entropy patch from a previous "
            "run. Start this training run in a fresh Python process."
        )


class MegaTrainPlugin(BasePlugin):
    """Configure Axolotl to train through MegaTrain's CPU master."""

    def __init__(self):
        self._trainer = None

    def register(self, cfg: dict):
        if not _megatrain_enabled(cfg):
            return
        device_map = cfg.get("device_map")
        if device_map not in (None, "cpu", {"": "cpu"}):
            raise ValueError(
                "MegaTrain requires `device_map: cpu`; remove custom or automatic "
                "model device placement."
            )
        if cfg.get("max_memory") is not None or cfg.get("gpu_memory_limit") is not None:
            raise ValueError(
                "MegaTrain cannot use `max_memory` or `gpu_memory_limit` because the "
                "Hugging Face model must be loaded wholly on CPU."
            )
        cfg["device_map"] = "cpu"
        cfg["experimental_skip_move_to_device"] = True
        if cfg.get("optimizer") is None:
            cfg["optimizer"] = "adamw_torch"

    def get_input_args(self):
        manager = PluginManager.get_instance()
        first_instance = next(
            (
                plugin
                for plugin in manager.plugins.values()
                if isinstance(plugin, MegaTrainPlugin)
            ),
            self,
        )
        if first_instance is not self:
            return None
        return "axolotl.integrations.megatrain.args.MegaTrainArgs"

    def pre_model_load(self, cfg):
        if not _megatrain_enabled(cfg):
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MegaTrain requires one CUDA GPU, but CUDA is not available."
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("MegaTrain requires a CUDA GPU with BF16 support.")

        world_size = max(int(os.environ.get("WORLD_SIZE", "1")), int(cfg.world_size))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = max(world_size, torch.distributed.get_world_size())
        if world_size != 1:
            raise RuntimeError(
                "MegaTrain supports one Axolotl process. Launch with `axolotl train` "
                "instead of torchrun, Accelerate multi-process, or a distributed runner."
            )
        devices = list(_cfg_value(cfg, "megatrain_devices", []) or [])
        if devices:
            device_count = torch.cuda.device_count()
            out_of_range = [index for index in devices if index >= device_count]
            if out_of_range:
                raise RuntimeError(
                    f"`megatrain_devices` names CUDA device {out_of_range[0]}, but "
                    f"only {device_count} device(s) are visible."
                )
            micro_batch_size = int(_cfg_value(cfg, "micro_batch_size", 1))
            if micro_batch_size < len(devices):
                raise RuntimeError(
                    "MegaTrain shards each microbatch across its GPUs, so "
                    f"`micro_batch_size` ({micro_batch_size}) must be at least the "
                    f"number of `megatrain_devices` ({len(devices)})."
                )
        if cfg.torch_dtype != torch.bfloat16:
            raise RuntimeError(
                "MegaTrain requires BF16 streamed compute. Set `bf16: true`."
            )
        if cfg.is_multimodal:
            raise RuntimeError(
                "MegaTrain's Axolotl integration currently supports decoder-only text "
                "models, not multimodal models."
            )
        # Resolve before validating: an unset value would otherwise be rejected as
        # `None`, and the streamed runtime needs an explicit backend to mirror.
        attn_implementation = cfg.attn_implementation or DEFAULT_ATTN_IMPLEMENTATION
        if attn_implementation not in SUPPORTED_ATTN_IMPLEMENTATIONS:
            supported = ", ".join(sorted(SUPPORTED_ATTN_IMPLEMENTATIONS))
            raise RuntimeError(
                f"MegaTrain supports only {supported} attention, not "
                f"{cfg.attn_implementation!r}."
            )
        cfg.attn_implementation = attn_implementation

        if (
            float(_cfg_value(cfg, "val_set_size", 0)) > 0
            or _cfg_value(cfg, "test_datasets")
            or _cfg_value(cfg, "do_bench_eval", False)
            or _cfg_value(cfg, "do_causal_lm_eval", False)
            or _cfg_value(cfg, "generate_samples", False)
            or _cfg_value(cfg, "load_best_model_at_end", False)
        ):
            raise RuntimeError(
                "MegaTrain does not implement evaluation or generation callbacks. "
                "Set `val_set_size: 0` and disable all evaluation settings."
            )
        if _cfg_value(cfg, "batch_flattening", False):
            raise RuntimeError(
                "MegaTrain does not support `batch_flattening`; set it to false."
            )
        if any(
            _cfg_value(cfg, key) is not None
            for key in ("fsdp", "fsdp_config", "deepspeed")
        ):
            raise RuntimeError(
                "MegaTrain cannot run with FSDP or DeepSpeed. Remove their settings."
            )
        if any(
            _cfg_value(cfg, key, False)
            for key in (
                "load_in_4bit",
                "load_in_8bit",
                "gptq",
                "quantize_moe_experts",
                "fp8",
                "use_onebitllms",
            )
        ) or any(
            _cfg_value(cfg, key) is not None
            for key in ("model_quantization_config", "qat", "quantization")
        ):
            raise RuntimeError(
                "MegaTrain requires an unquantized BF16 model. Remove all loading, "
                "training, and expert quantization settings."
            )
        if _cfg_value(cfg, "optimizer") != "adamw_torch":
            raise RuntimeError("MegaTrain supports `optimizer: adamw_torch` only.")
        unsupported_patch = _unsupported_model_patch(cfg)
        if unsupported_patch is not None:
            raise RuntimeError(
                f"MegaTrain does not support `{unsupported_patch}` because it changes "
                "the validated model-forward or loss path. Disable or remove it."
            )

        manager = PluginManager.get_instance()
        configured_plugins = {
            str(plugin) for plugin in (_cfg_value(cfg, "plugins", []) or [])
        }
        stale_plugins = [
            name
            for name, plugin in manager.plugins.items()
            if name not in configured_plugins
            and name not in MEGATRAIN_PLUGIN_NAMES
            and not isinstance(plugin, MegaTrainPlugin)
        ]
        if stale_plugins:
            raise RuntimeError(
                "MegaTrain detected a stale registered plugin whose hooks are not "
                f"validated with CPU-master streaming: {stale_plugins[0]!r}. Start "
                "the run in a fresh Python process."
            )
        if self in manager.plugins.values():
            from .trainer import MegaTrainTrainer

            if manager.get_trainer_cls(cfg) is not MegaTrainTrainer:
                raise RuntimeError(
                    "Another plugin was selected as the trainer before MegaTrain. "
                    "Use MegaTrain as the only configured plugin."
                )

        from axolotl.loaders.utils import load_model_config

        model_config = load_model_config(cfg)
        text_config = (
            model_config.get_text_config()
            if hasattr(model_config, "get_text_config")
            else getattr(model_config, "text_config", model_config)
        )
        model_type = getattr(text_config, "model_type", "")
        if model_type not in _SUPPORTED_MODEL_TYPES:
            supported = ", ".join(sorted(_SUPPORTED_MODEL_TYPES))
            raise RuntimeError(
                f"MegaTrain has verified loss and gradient parity only for model types "
                f"{supported}; got {model_type!r}."
            )
        if getattr(model_config, "quantization_config", None) is not None:
            raise RuntimeError(
                "MegaTrain cannot load a checkpoint whose model config contains "
                "quantization metadata. Use an unquantized BF16 checkpoint."
            )
        dropout_fields = _nonzero_dropout_fields(model_config)
        if dropout_fields:
            raise RuntimeError(
                "MegaTrain currently requires all model dropout probabilities to be "
                f"zero; found {dropout_fields[0]!r}."
            )
        layer_types = set(getattr(text_config, "layer_types", []) or [])
        unsupported_layer_types = layer_types - {
            "full_attention",
            "sliding_attention",
        }
        if len(layer_types) > 1:
            raise RuntimeError(
                "MegaTrain does not support mixed attention layer schedules."
            )
        if (
            unsupported_layer_types
            or getattr(text_config, "attention_chunk_size", None) is not None
        ):
            unsupported = ", ".join(sorted(unsupported_layer_types)) or "chunked"
            raise RuntimeError(
                "MegaTrain supports full and sliding-window decoder attention, "
                f"not {unsupported} attention."
            )

        from axolotl.monkeypatch.attention.sdpa_varlen import unpatch_sdpa_varlen

        unpatch_sdpa_varlen()
        _assert_unpatched_model_classes(model_type)
        _assert_flash_attention_2_is_unpatched(cfg.attn_implementation)
        _assert_cross_entropy_is_unpatched()

        cfg.device_map = "cpu"
        cfg.experimental_skip_move_to_device = True

    def post_model_load(self, cfg, model):
        if not _megatrain_enabled(cfg):
            return
        model_quantization_config = getattr(model.config, "quantization_config", None)
        if (
            getattr(model, "is_quantized", False)
            or getattr(model, "hf_quantizer", None)
            or model_quantization_config is not None
        ):
            raise RuntimeError(
                "MegaTrain cannot train a quantized base model; use an unquantized BF16 "
                "checkpoint."
            )
        if getattr(model.config, "is_encoder_decoder", False):
            raise RuntimeError(
                "MegaTrain currently supports decoder-only causal language models."
            )

        from transformers import LlamaForCausalLM, MistralForCausalLM

        supported_classes = (LlamaForCausalLM, MistralForCausalLM)
        if type(model) not in supported_classes:
            raise RuntimeError(
                "MegaTrain currently supports the Transformers LlamaForCausalLM and "
                f"MistralForCausalLM classes, not {type(model).__name__}."
            )
        dropout_fields = _nonzero_dropout_fields(model.config)
        if dropout_fields:
            raise RuntimeError(
                "MegaTrain currently requires all model dropout probabilities to be "
                f"zero; found {dropout_fields[0]!r}."
            )
        dropout_modules = [
            name
            for name, module in model.named_modules()
            if isinstance(
                module,
                (
                    nn.Dropout,
                    nn.Dropout1d,
                    nn.Dropout2d,
                    nn.Dropout3d,
                    nn.AlphaDropout,
                    nn.FeatureAlphaDropout,
                ),
            )
            and module.p > 0
        ]
        if dropout_modules:
            raise RuntimeError(
                "MegaTrain currently requires model dropout modules to have `p=0`; "
                f"found {dropout_modules[0]!r}."
            )

        model.to(device="cpu", dtype=torch.float32)
        parameters = list(model.parameters())
        if any(parameter.device.type != "cpu" for parameter in parameters):
            raise RuntimeError(
                "MegaTrain requires every model parameter to be loaded on CPU."
            )
        if any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in parameters
        ):
            raise RuntimeError(
                "MegaTrain requires all floating-point CPU master parameters to use FP32."
            )

    def get_trainer_cls(self, cfg):
        if not _megatrain_enabled(cfg):
            return None
        from .trainer import MegaTrainTrainer

        return MegaTrainTrainer

    def get_training_args(self, cfg):
        if not _megatrain_enabled(cfg):
            return {}
        return {
            "use_cpu": True,
            "bf16": False,
            "fp16": False,
            "tf32": False,
            "gradient_checkpointing": False,
            "torch_compile": False,
            "use_liger_kernel": False,
        }

    def create_optimizer(self, cfg, trainer):
        if not _megatrain_enabled(cfg):
            return None
        from .trainer import MegaTrainAdamW, MegaTrainTrainer

        if not isinstance(trainer, MegaTrainTrainer):
            return None

        decay_parameter_names = trainer.get_decay_parameter_names(trainer.model)
        # Embeddings and the output head are additionally excluded from weight
        # decay; this deviates from Axolotl's stock optimizer grouping.
        decay_parameter_ids = {
            id(parameter)
            for name, parameter in trainer.model.named_parameters()
            if name in decay_parameter_names
            and not any(part in name for part in ("embed_tokens", "lm_head"))
        }
        parameters = trainer.cpu_master.get_parameters()
        optimizer_groups = [
            {
                "params": [
                    parameter
                    for parameter in parameters
                    if id(parameter) in decay_parameter_ids
                ],
                "weight_decay": float(cfg.weight_decay or 0.0),
            },
            {
                "params": [
                    parameter
                    for parameter in parameters
                    if id(parameter) not in decay_parameter_ids
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_groups = [group for group in optimizer_groups if group["params"]]
        return MegaTrainAdamW(
            optimizer_groups,
            cpu_master=trainer.cpu_master,
            lr=float(cfg.learning_rate),
            betas=(
                float(cfg.adam_beta1 if cfg.adam_beta1 is not None else 0.9),
                float(cfg.adam_beta2 if cfg.adam_beta2 is not None else 0.999),
            ),
            eps=float(cfg.adam_epsilon if cfg.adam_epsilon is not None else 1e-8),
            weight_decay=0.0,
            fused=False,
            foreach=False,
        )

    def post_trainer_create(self, cfg, trainer):
        if not _megatrain_enabled(cfg):
            return
        from .trainer import MegaTrainTrainer

        if not isinstance(trainer, MegaTrainTrainer):
            raise RuntimeError(
                "MegaTrain was active but Axolotl constructed a different trainer. "
                "Use MegaTrain as the only configured plugin."
            )
        self._trainer = trainer

    def post_train(self, cfg, model):
        if not _megatrain_enabled(cfg):
            return
        trainer, self._trainer = self._trainer, None
        if trainer is not None:
            trainer.close()

    def post_train_unload(self, cfg):
        if not _megatrain_enabled(cfg):
            return
        trainer, self._trainer = self._trainer, None
        if trainer is not None:
            trainer.close()


__all__ = ["MegaTrainArgs", "MegaTrainPlugin"]
