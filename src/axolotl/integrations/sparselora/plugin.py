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

_SUPPORTED_ADAPTERS = {"lora", "qlora"}
_COMPILE_BOUNDARIES_SET = False


def _is_deepspeed_zero3(deepspeed: Any) -> bool:
    """True if the DeepSpeed config selects ZeRO stage 3.

    ``cfg.deepspeed`` may be a path string to a JSON config, an already-parsed
    dict, or a JSON string. Inspect the parsed ``zero_optimization.stage`` when
    possible and fall back to a substring match on the path/string form.
    """
    if not deepspeed:
        return False

    config = deepspeed
    if isinstance(deepspeed, str):
        import json
        import os

        if os.path.isfile(deepspeed):
            try:
                with open(deepspeed, encoding="utf-8") as f:
                    config = json.load(f)
            except (OSError, ValueError):
                return "zero3" in deepspeed
        else:
            try:
                config = json.loads(deepspeed)
            except ValueError:
                return "zero3" in deepspeed

    if isinstance(config, dict):
        zero = config.get("zero_optimization", {})
        if not isinstance(zero, dict):
            # Legacy boolean form maps to stage 1 (or 0), never 3.
            return False
        return int(zero.get("stage", 0)) == 3

    return "zero3" in str(deepspeed)


def resolve_layer_sparsity(
    layer_sparsity: dict[str, float], target_names: list[str]
) -> dict[str, float]:
    """Map explicit ``layer_sparsity`` keys onto discovered module names.

    Keys may be a full module path or a suffix of one (e.g. ``model.layers.3.mlp``).
    A key matching no module is an error rather than a silent drop to dense.
    """
    schedule: dict[str, float] = {}
    for raw_key, val in dict(layer_sparsity).items():
        matches = [n for n in target_names if n == raw_key or n.endswith("." + raw_key)]
        if not matches:
            raise ValueError(
                f"sparselora.layer_sparsity key {raw_key!r} matches no sparsifiable "
                "module. Use a full module path or a suffix of one "
                "(e.g. 'model.layers.3.mlp')."
            )
        for n in matches:
            schedule[n] = float(val)
    return schedule


def _apply_compile_boundaries() -> None:
    """Mark SparseLoRA's data-dependent code as ``torch.compiler.disable`` regions.

    Contextual sparsity (top-k channel selection, boolean-mask token splits,
    quantized dequant) cannot be captured in a static graph — under ``torch
    .compile`` it raises a backend failure. Disabling Dynamo on these entry
    points makes the compiler graph-break around them (running them eagerly,
    which is already fast) and compile the rest of the model. No-op at eager
    runtime. Applied once at the class level.
    """
    global _COMPILE_BOUNDARIES_SET
    if _COMPILE_BOUNDARIES_SET:
        return

    import torch

    from . import arch_wiring
    from ._vendor.sparselora import api
    from ._vendor.sparselora.modules import llama, predictors
    from .fast_tokens import install_fast_tokens
    from .sparse_linear_4bit import SparseLinear4bit

    # Install the contiguous-block token split/join before wrapping the mask
    # builder, so the torch.compiler.disable below wraps the fast version.
    install_fast_tokens()

    disable = torch.compiler.disable
    llama.SparseLlamaMLP.forward = disable(llama.SparseLlamaMLP.forward)  # type: ignore[method-assign]
    llama.SparseLlamaAttention.forward = disable(llama.SparseLlamaAttention.forward)  # type: ignore[method-assign]
    # Generic attention has its own forward; the MLP/linear paths inherit the
    # (already-disabled) vendored functions, so only the override needs marking.
    arch_wiring.SparseAttention.forward = disable(arch_wiring.SparseAttention.forward)  # type: ignore[method-assign]
    # Phi3 fused attention overrides forward indirectly via _qkv (data-dependent
    # combined-index slicing); the fused MLP overrides the (already-disabled)
    # SwiGLU _block. Disable their forwards explicitly so Dynamo graph-breaks
    # around the contextual-sparsity selection in both.
    arch_wiring.SparseFusedQKVAttention.forward = disable(  # type: ignore[method-assign]
        arch_wiring.SparseFusedQKVAttention.forward
    )
    arch_wiring.SparseFusedGateUpMLP.forward = disable(  # type: ignore[method-assign]
        arch_wiring.SparseFusedGateUpMLP.forward
    )
    arch_wiring.SparseLinearBias.forward = disable(arch_wiring.SparseLinearBias.forward)  # type: ignore[method-assign]
    api._compute_output_token_mask = disable(api._compute_output_token_mask)
    for cls in (
        predictors.FFNPredictor,
        predictors.AttentionPredictor,
        predictors.GQAAttentionPredictor,
        arch_wiring._GELUFFNPredictor,
    ):
        cls.predict = disable(cls.predict)  # type: ignore[method-assign]
    SparseLinear4bit.forward = disable(SparseLinear4bit.forward)  # type: ignore[method-assign]

    _COMPILE_BOUNDARIES_SET = True


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
                f"SparseLoRA composes on a LoRA/QLoRA adapter; got "
                f"adapter={cfg.adapter!r}. Set `adapter: lora` or `qlora`."
            )
        if cfg.load_in_8bit:
            raise ValueError(
                "SparseLoRA does not support 8-bit bases. Use a full-precision "
                "base (adapter: lora) or a 4-bit base (adapter: qlora, "
                "load_in_4bit: true)."
            )
        if cfg.sample_packing:
            raise ValueError(
                "SparseLoRA's context/output token split is incompatible with "
                "sample_packing. Set `sample_packing: false`."
            )
        if cfg.fsdp or _is_deepspeed_zero3(cfg.deepspeed):
            raise ValueError(
                "SparseLoRA v1 supports single-GPU / DDP only; FSDP and DeepSpeed "
                "ZeRO-3 shard parameters and are not yet supported."
            )

        # Architecture support: auto-register sparse wiring for the loaded
        # model's MLP/attention classes (Llama is pre-registered; Qwen2/Qwen3/
        # Mistral/... are introspected and mapped to the generic wiring), then
        # confirm every target is covered and has supported semantics.
        from ._vendor.sparselora.modules import get_module_mapping
        from .arch_wiring import register_arch_wiring, unsupported_reason

        registered = register_arch_wiring(model)
        if registered:
            LOG.info(
                "SparseLoRA: registered generic sparse wiring for %s.",
                ", ".join(
                    f"{cls} ({role})" for cls, role in sorted(registered.items())
                ),
            )

        module_map = get_module_mapping()
        modules = dict(model.named_modules())

        reasons = sorted(
            {r for n in target_names if (r := unsupported_reason(modules[n]))}
        )
        if reasons:
            raise ValueError(
                "SparseLoRA cannot sparsify this model: " + "; ".join(reasons)
            )

        unsupported = {
            type(modules[n]).__name__
            for n in target_names
            if type(modules[n]) not in module_map
        }
        if unsupported:
            raise ValueError(
                f"SparseLoRA wiring not available for module type(s) {sorted(unsupported)}. "
                "Supported: Llama + any SwiGLU-MLP / standard-attention architecture "
                "(Qwen2, Qwen3, Mistral, ...). Register custom wiring via "
                "`register_sparse_module` to extend support."
            )

        # The vendored MLP sparse path passes indices as a kwarg that only a
        # plain SparseLinear accepts; a LoRA-wrapped MLP projection (gate/up/down,
        # or the fused gate_up_proj) collides with the patched lora_forward.
        # z-lab's recipe is attention-only.
        mlp_proj_names = {"gate_proj", "up_proj", "down_proj", "gate_up_proj"}
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
        # output_sparse_weights slices fused qkv_proj/gate_up_proj into their
        # logical sub-blocks, so the check sees the real per-sub-block dims.
        from .factors import output_sparse_weights

        rank = cfg.sparselora.predictor_rank
        too_small = [
            f"{name}.{proj} (min dim {min(weight.shape)})"
            for name in target_names
            for proj, weight in output_sparse_weights(modules[name]).items()
            if min(weight.shape) < rank
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
                schedule = resolve_layer_sparsity(settings.layer_sparsity, target_names)
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
        _apply_compile_boundaries()
        if cfg.load_in_4bit:
            from .sparse_linear_4bit import register_4bit_support

            if not register_4bit_support():
                raise ImportError(
                    "load_in_4bit is set but bitsandbytes is unavailable for "
                    "SparseLoRA's 4-bit sparse linear."
                )

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
        if cfg.load_in_4bit:
            self._patch_4bit_lora_forward(model)
        trainer.add_callback(SparseLoRACallback(settings.start_step, settings.end_step))

    def _patch_4bit_lora_forward(self, model: nn.Module) -> None:
        """Route sparse indices through PEFT 4-bit LoRA wrappers.

        ``apply_sparselora`` patches ``peft...lora.layer.Linear.forward``, but a
        4-bit base is wrapped by a different PEFT class (``lora...Linear4bit``)
        that the global patch misses. Bind the vendored ``lora_forward`` to each
        such wrapper so the predicted indices reach the sparse base layer.
        """
        import types

        from ._vendor.sparselora.modules.linear import lora_forward
        from .sparse_linear_4bit import SparseLinear4bit

        patched = 0
        for module in model.modules():
            base = getattr(module, "base_layer", None)
            if isinstance(base, SparseLinear4bit) and hasattr(module, "lora_A"):
                module.forward = types.MethodType(lora_forward, module)
                patched += 1
        LOG.debug("SparseLoRA: routed indices through %d 4-bit LoRA wrappers.", patched)
