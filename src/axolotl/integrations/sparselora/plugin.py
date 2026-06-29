"""SparseLoRA contextual-sparsity plugin for Axolotl.

Composes on top of a standard full-precision LoRA adapter: after the trainer is
built (model + adapter ready, before the training loop and any DDP wrapping), it
calibrates a per-layer sparsity schedule on a slice of the *same* dataset,
computes the SVD predictor factors from the base weights, and applies the
vendored SparseLoRA machinery.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .cache import compute_cache_key, entry_dir, load_cached, maybe_share, save_cached
from .calibration import calibrate, discover_target_modules, method_name
from .factors import compute_factor_tensors

LOG = get_logger(__name__)

_SUPPORTED_ADAPTERS = {"lora"}


class SparseLoRAPlugin(BasePlugin):
    """Enable SparseLoRA contextual sparsity for full-precision LoRA fine-tuning."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.sparselora.SparseLoRAArgs"

    def _settings(self, cfg: DictDefault):
        settings = getattr(cfg, "sparselora", None)
        if not settings or not settings.enabled:
            return None
        return settings

    def _validate(
        self, cfg: DictDefault, model: nn.Module, target_names: list[str]
    ) -> None:
        if cfg.adapter not in _SUPPORTED_ADAPTERS:
            raise ValueError(
                f"SparseLoRA composes on a full-precision LoRA adapter; got "
                f"adapter={cfg.adapter!r}. Set `adapter: lora`."
            )
        if cfg.load_in_4bit or cfg.load_in_8bit:
            raise ValueError(
                "SparseLoRA v1 requires a full-precision base model. The sparse "
                "linear channel-slices a dense weight, which is incompatible with "
                "a quantized (4-bit/8-bit) base. Disable load_in_4bit/load_in_8bit "
                "and use adapter: lora."
            )
        if cfg.sample_packing:
            raise ValueError(
                "SparseLoRA's context/output token split is incompatible with "
                "sample_packing. Set `sample_packing: false`."
            )
        if cfg.fsdp or (cfg.deepspeed and "zero3" in str(cfg.deepspeed)):
            raise ValueError(
                "SparseLoRA v1 supports single-GPU / DDP only; FSDP and DeepSpeed "
                "ZeRO-3 shard parameters and are not yet supported."
            )

        # Architecture support: the parent MLP/attention module type must have
        # registered sparse wiring (Llama in v1).
        from ._vendor.sparselora.modules import get_module_mapping

        module_map = get_module_mapping()
        modules = dict(model.named_modules())
        unsupported = {
            type(modules[n]).__name__
            for n in target_names
            if type(modules[n]) not in module_map
        }
        if unsupported:
            raise ValueError(
                f"SparseLoRA wiring not available for module type(s) {sorted(unsupported)}. "
                "Supported architectures: Llama. Register sparse modules via "
                "`register_sparse_module` to extend support."
            )

        # The vendored MLP sparse path passes indices as a kwarg that only a
        # plain SparseLinear accepts; a LoRA-wrapped MLP projection (gate/up/down)
        # collides with the patched lora_forward. z-lab's recipe is attention-only.
        mlp_proj_names = {"gate_proj", "up_proj", "down_proj"}
        mlp_lora = [
            name
            for name, module in model.named_modules()
            if name.rsplit(".", 1)[-1] in mlp_proj_names
            and hasattr(module, "base_layer")
        ]
        if mlp_lora:
            raise ValueError(
                "SparseLoRA v1 requires attention-only LoRA; found LoRA adapters on "
                f"MLP projections: {mlp_lora}. Remove gate_proj/up_proj/down_proj from "
                "`lora_target_modules` (e.g. use q_proj,k_proj,v_proj,o_proj)."
            )

        # The vendored predictor buffers are sized with the configured rank, so a
        # projection smaller than predictor_rank fails at load_state_dict after
        # calibration has already run. Fail early with the offending projection.
        from .factors import linear_weight, projections_for

        rank = cfg.sparselora.predictor_rank
        too_small = [
            f"{name}.{proj} (min dim {min(linear_weight(getattr(modules[name], proj)).shape)})"
            for name in target_names
            for proj in projections_for(modules[name])
            if min(linear_weight(getattr(modules[name], proj)).shape) < rank
        ]
        if too_small:
            raise ValueError(
                f"sparselora.predictor_rank={rank} exceeds the smaller dimension of "
                f"these projections: {too_small}. Lower predictor_rank."
            )

        # Every remaining LoRA layer must live inside a covered target module, else
        # the global lora.Linear patch would call a plain nn.Linear with indices.
        covered = tuple(n + "." for n in target_names)
        orphan_lora = [
            name
            for name, module in model.named_modules()
            if "lora_" not in name
            and hasattr(module, "base_layer")
            and not name.startswith(covered)
        ]
        if orphan_lora:
            raise ValueError(
                "SparseLoRA requires every LoRA-adapted module to sit inside a "
                f"sparsifiable attention block; found adapters outside: {orphan_lora}. "
                "Restrict `lora_target_modules` to attention projections."
            )

    def post_trainer_create(self, cfg: DictDefault, trainer: Any) -> None:
        settings = self._settings(cfg)
        if settings is None:
            return

        model = trainer.model
        target_names = discover_target_modules(model)
        self._validate(cfg, model, target_names)

        rank = settings.predictor_rank
        key = compute_cache_key(cfg)
        cached = load_cached(cfg, key)

        if cached is not None:
            layer_sparsity = {k: float(v) for k, v in cached["layer_sparsity"].items()}
            cache_dir = entry_dir(cfg, key)
        else:
            method = method_name(settings.calibration.method)
            all_factors: dict = {}
            if method == "none":
                schedule = {
                    k: float(v) for k, v in dict(settings.layer_sparsity).items()
                }
            else:
                # Factors for all targets feed the sensitivity sweep.
                all_factors = compute_factor_tensors(model, target_names, rank)
                schedule = calibrate(cfg, model, all_factors, trainer)

            # Cover every target module so the global LoRA patch stays safe;
            # calibration-dense modules are explicit 0.0.
            layer_sparsity = {n: float(schedule.get(n, 0.0)) for n in target_names}
            positive = [n for n, s in layer_sparsity.items() if s > 0.0]
            # Reuse sweep factors where available; only compute the rest.
            missing = [
                n
                for n in positive
                if f"{n}.gate_proj.w1" not in all_factors
                and f"{n}.q_proj.w1" not in all_factors
            ]
            factor_tensors = {
                k: v
                for k, v in all_factors.items()
                if k.rsplit(".", 2)[0] in set(positive)
            }
            factor_tensors.update(compute_factor_tensors(model, missing, rank))

            meta = {
                "base_model": cfg.base_model,
                "model_type": type(model).__name__,
                "predictor_rank": rank,
                "target_sparsity": settings.target_sparsity,
                "calibration_method": method_name(settings.calibration.method),
                "num_sparsified": len(positive),
            }
            save_cached(cfg, key, layer_sparsity, meta, factor_tensors)
            maybe_share(cfg, layer_sparsity, meta)
            cache_dir = entry_dir(cfg, key)

        self._apply(cfg, model, trainer, layer_sparsity, cache_dir, rank)

    def _apply(
        self,
        cfg: DictDefault,
        model: nn.Module,
        trainer: Any,
        layer_sparsity: dict[str, float],
        cache_dir: str,
        rank: int,
    ) -> None:
        from ._vendor.sparselora import SparseLoRAConfig, apply_sparselora
        from ._vendor.sparselora.callback import SparseLoRACallback

        settings = cfg.sparselora
        sl_config = SparseLoRAConfig(
            layer_sparsity=layer_sparsity,
            predictor_rank=rank,
            path=cache_dir,
            start_step=settings.start_step,
            end_step=settings.end_step,
        )
        n_active = sum(1 for v in layer_sparsity.values() if v > 0)
        LOG.info(
            "SparseLoRA: applying sparsity to %d/%d modules (start_step=%s, end_step=%s).",
            n_active,
            len(layer_sparsity),
            settings.start_step,
            settings.end_step,
        )
        apply_sparselora(model, sl_config)
        trainer.add_callback(SparseLoRACallback(settings.start_step, settings.end_step))
