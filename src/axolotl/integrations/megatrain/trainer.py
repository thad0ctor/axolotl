# SPDX-License-Identifier: Apache-2.0

"""Trainer bridge for MegaTrain's CPU-master runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.integrations.base import PluginManager
from axolotl.integrations.megatrain._vendor.infinity.config.training import (
    CPUMasterConfig,
)
from axolotl.integrations.megatrain._vendor.infinity.model.cpu_master import (
    CPUMasterModel,
)
from axolotl.integrations.megatrain.args import DEFAULT_ATTN_IMPLEMENTATION
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _cfg_value(cfg, key: str, default=None):
    value = cfg.get(key) if hasattr(cfg, "get") else getattr(cfg, key, None)
    return default if value is None else value


def _attention_implementation(cfg) -> str:
    implementation = _cfg_value(cfg, "attn_implementation")
    if implementation:
        return str(implementation)
    if _cfg_value(cfg, "flash_attention", False):
        return "flash_attention_2"
    if _cfg_value(cfg, "sdp_attention", False):
        return "sdpa"
    return DEFAULT_ATTN_IMPLEMENTATION


def build_megatrain_config(cfg) -> CPUMasterConfig:
    """Map normalized Axolotl settings to MegaTrain's runtime config."""

    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else int(_cfg_value(cfg, "local_rank", 0))
    )
    devices = [int(index) for index in _cfg_value(cfg, "megatrain_devices", []) or []]
    if not devices:
        devices = [device]
    max_steps = int(_cfg_value(cfg, "max_steps", -1))
    return CPUMasterConfig(
        model_name=str(_cfg_value(cfg, "base_model", "unknown")),
        device=devices[0],
        devices=devices,
        dtype=torch.bfloat16,
        attn_implementation=_attention_implementation(cfg),
        trust_remote_code=bool(_cfg_value(cfg, "trust_remote_code", False)),
        dataset_path="__axolotl__",
        max_seq_len=int(_cfg_value(cfg, "sequence_len", 512)),
        batch_size=int(_cfg_value(cfg, "micro_batch_size", 1)),
        gradient_accumulation_steps=int(
            _cfg_value(cfg, "gradient_accumulation_steps", 1)
        ),
        num_steps=max_steps if max_steps > 0 else 100,
        learning_rate=float(_cfg_value(cfg, "learning_rate", 5e-5)),
        weight_decay=float(_cfg_value(cfg, "weight_decay", 0.0)),
        max_grad_norm=float(_cfg_value(cfg, "max_grad_norm", 1.0)),
        beta1=float(_cfg_value(cfg, "adam_beta1", 0.9)),
        beta2=float(_cfg_value(cfg, "adam_beta2", 0.999)),
        eps=float(_cfg_value(cfg, "adam_epsilon", 1e-8)),
        seed=int(_cfg_value(cfg, "seed", 42)),
        log_interval=int(_cfg_value(cfg, "logging_steps", 1)),
        checkpoint_interval=int(_cfg_value(cfg, "megatrain_checkpoint_interval", 4)),
        num_grad_slabs=int(_cfg_value(cfg, "megatrain_num_grad_slabs", 12)),
        fp32_head_grad=bool(_cfg_value(cfg, "megatrain_fp32_head_grad", True)),
        enable_timing=False,
    )


class MegaTrainAdamW(torch.optim.AdamW):
    """CPU AdamW that refreshes MegaTrain's streamed parameter copies."""

    def __init__(self, params, *, cpu_master: CPUMasterModel, **kwargs):
        self.cpu_master = cpu_master
        super().__init__(params, **kwargs)

    @override
    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure=closure)
        self.cpu_master._sync_params_to_gpu()
        return loss


class MegaTrainTrainer(AxolotlTrainer):
    """Axolotl Trainer that delegates forward and backward to MegaTrain."""

    def __init__(self, *args, **kwargs):
        self._megatrain_closed = False
        super().__init__(*args, **kwargs)

        # Transformers scales the dataloader batch by `n_gpu` for DataParallel, but
        # MegaTrain streams one unwrapped model and shards batches itself, so on a
        # multi-GPU host that would silently multiply `micro_batch_size`.
        if getattr(self, "args", None) is not None:
            self.args._n_gpu = 1
            self._train_batch_size = self.args.per_device_train_batch_size

        cfg = PluginManager.get_instance().cfg
        if cfg is None:
            raise RuntimeError(
                "MegaTrainTrainer requires a loaded Axolotl plugin configuration."
            )

        parameters = list(self.model.parameters())
        if any(parameter.device.type != "cpu" for parameter in parameters):
            raise RuntimeError(
                "MegaTrain requires every model parameter to remain on CPU. "
                "Remove custom device placement and use the plugin's CPU device map."
            )
        if any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in parameters
        ):
            raise RuntimeError(
                "MegaTrain requires FP32 CPU master parameters and BF16 streamed "
                "working parameters."
            )
        if any(not parameter.requires_grad for parameter in parameters):
            raise RuntimeError(
                "MegaTrain requires a full fine-tune, but the loaded model contains "
                "frozen parameters. Remove selective freezing settings."
            )

        self.megatrain_config = build_megatrain_config(cfg)
        self.model_accepts_loss_kwargs = True

    def _ensure_cpu_master(self) -> CPUMasterModel:
        cpu_master = getattr(self, "cpu_master", None)
        if cpu_master is not None:
            return cpu_master
        if self._megatrain_closed:
            raise RuntimeError("MegaTrain resources have already been released.")

        self.model.train()
        cpu_master = CPUMasterModel(self.model, self.megatrain_config)
        self.cpu_master = cpu_master

        model_parameter_ids = {id(parameter) for parameter in self.model.parameters()}
        master_parameter_ids = {
            id(parameter) for parameter in cpu_master.get_parameters()
        }
        if model_parameter_ids != master_parameter_ids:
            self.close()
            raise RuntimeError(
                "MegaTrain could not map every model parameter to its streamed "
                "components. This model architecture is not supported."
            )

        LOG.info(
            "Initialized MegaTrain CPU master with %s streamed layers on cuda:%s",
            len(cpu_master.cpu_layers),
            self.megatrain_config.device,
        )
        return cpu_master

    @override
    def _build_accelerator_args(self, **kwargs) -> dict[str, Any]:
        accelerator_args = super()._build_accelerator_args(**kwargs)
        accelerator_args["device_placement"] = False
        return accelerator_args

    @override
    def _move_model_to_device(self, model: nn.Module, device: torch.device) -> None:
        return None

    @override
    def _wrap_model(self, model: nn.Module, training=True, dataloader=None):
        return model

    @override
    def create_optimizer(self, model=None):
        if self.optimizer is None:
            cpu_master = self._ensure_cpu_master()
            from . import MegaTrainPlugin

            manager = PluginManager.get_instance()
            cfg = manager.cfg
            if cfg is None:
                raise RuntimeError(
                    "MegaTrainTrainer requires a loaded Axolotl plugin configuration."
                )
            configured_plugins = {
                str(plugin) for plugin in (_cfg_value(cfg, "plugins", []) or [])
            }
            plugins = [
                plugin
                for name, plugin in manager.plugins.items()
                if name in configured_plugins and isinstance(plugin, MegaTrainPlugin)
            ]
            if len(plugins) != 1:
                raise RuntimeError(
                    "MegaTrainTrainer requires exactly one active MegaTrainPlugin."
                )
            optimizer = plugins[0].create_optimizer(cfg, self)
            if not isinstance(optimizer, MegaTrainAdamW):
                raise RuntimeError(
                    "MegaTrainPlugin did not provide its CPU-master AdamW optimizer."
                )
            if optimizer.cpu_master is not cpu_master:
                raise RuntimeError(
                    "MegaTrain optimizer is bound to a different CPU-master runtime."
                )
            self.optimizer = optimizer
        return self.optimizer

    @override
    def _get_num_items_in_batch(
        self, batch_samples: list, device: torch.device
    ) -> torch.Tensor | int | None:
        labels = [batch.get("labels") for batch in batch_samples]
        if not labels or any(label is None for label in labels):
            return None
        return sum(label[..., 1:].ne(-100).sum() for label in labels).to(device)

    @override
    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        if self._megatrain_closed:
            raise RuntimeError("MegaTrain resources have already been released.")

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        try:
            input_ids = inputs["input_ids"].detach().cpu()
            labels = inputs["labels"].detach().cpu()
        except KeyError as exc:
            raise KeyError(
                f"MegaTrain batch is missing required field {exc.args[0]!r}."
            ) from exc
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.detach().cpu()
        position_ids = inputs.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.detach().cpu()
            expected_position_ids = torch.arange(input_ids.shape[-1]).expand_as(
                input_ids
            )
            if not torch.equal(position_ids, expected_position_ids):
                raise RuntimeError(
                    "MegaTrain does not support custom `position_ids`; each row must "
                    "use the canonical 0..sequence_length-1 positions."
                )

        local_valid_tokens = int(labels[..., 1:].ne(-100).sum().item())
        if local_valid_tokens == 0:
            return torch.zeros((), device=self.args.device)

        if num_items_in_batch is not None:
            global_valid_tokens = int(torch.as_tensor(num_items_in_batch).item())
        else:
            global_valid_tokens = 0

        if global_valid_tokens > 0:
            gradient_denominator = global_valid_tokens
            loss_weight = local_valid_tokens / global_valid_tokens
        else:
            accumulation_steps = max(1, self.current_gradient_accumulation_steps)
            gradient_denominator = local_valid_tokens * accumulation_steps
            loss_weight = 1.0 / accumulation_steps

        cpu_master = getattr(self, "cpu_master", None)
        if cpu_master is None:
            cpu_master = self._ensure_cpu_master()
        loss_value, _, timing = cpu_master.forward_and_backward(
            input_ids,
            attention_mask,
            labels,
            global_valid_tokens=gradient_denominator,
        )
        self._megatrain_last_timing = timing
        return torch.tensor(
            float(loss_value) * loss_weight,
            dtype=torch.float32,
            device=self.args.device,
        )

    @override
    def _clip_grad_norm(self, model):
        return torch.nn.utils.clip_grad_norm_(
            self._ensure_cpu_master().get_parameters(), self.args.max_grad_norm
        )

    @override
    def _get_grad_norm(self, model, grad_norm=None):
        if grad_norm is not None:
            return grad_norm
        return torch.nn.utils.clip_grad_norm_(
            self._ensure_cpu_master().get_parameters(), float("inf")
        )

    @override
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None) -> None:
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        self._ensure_cpu_master()._sync_params_to_gpu()

    @override
    def _load_best_model(self) -> None:
        super()._load_best_model()
        self._ensure_cpu_master()._sync_params_to_gpu()

    @override
    def train(self, *args, **kwargs):
        try:
            self._ensure_cpu_master()
            return super().train(*args, **kwargs)
        finally:
            self.close()

    def close(self) -> None:
        if self._megatrain_closed:
            return
        self._megatrain_closed = True
        cpu_master = getattr(self, "cpu_master", None)
        if cpu_master is None:
            return
        try:
            cpu_master.release_gpu_buffers()
        finally:
            cpu_master.cleanup()
