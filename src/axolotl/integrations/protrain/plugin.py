# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BasePlugin subclass for ProTrain (M5, DESIGN.md §Plugin Integration).

Thin shim over the M1-M4 runtime primitives: wires Axolotl's plugin hook
points (``post_model_load`` / ``create_optimizer`` / ``post_trainer_create``)
to ``protrain_model_wrapper`` / ``protrain_optimizer_wrapper``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from transformers import Trainer

    from axolotl.utils.dict import DictDefault

LOG = get_logger(__name__)


# Default PCIe H2D bandwidth assumed for HardwareProfile construction when
# no measured value is available. 13 GB/s matches a typical PCIe Gen4 x16
# 3090 rig; the profiler's microbench will overwrite this once the cache
# key misses and a full profile runs — this constant only seeds the
# constructor for the cost model's effective-bandwidth prior.
_DEFAULT_PCIE_BPS = 13e9


def _is_plugin_active(cfg) -> bool:
    """Return True iff both the plugin is registered and auto_memory is on.

    Matches the enable-gate documented on ``ProTrainArgs.protrain_auto_memory``
    and mirrors the ``LigerPlugin`` pattern of reading ``cfg.*`` attributes
    without touching Axolotl-internal state.
    """
    if not getattr(cfg, "protrain_auto_memory", False):
        return False
    plugins = getattr(cfg, "plugins", None) or []
    return any(isinstance(p, str) and "protrain" in p.lower() for p in plugins)


def _build_hardware_profile(cfg):
    """Construct a ``HardwareProfile`` from the first visible CUDA device."""
    import torch

    from axolotl.integrations.protrain.types import HardwareProfile

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ProTrain plugin requires a CUDA device; torch.cuda.is_available() is False."
        )

    # Honour CUDA_VISIBLE_DEVICES — the ordinal here is logical (0), which
    # resolves to whatever the user masked in via the env var. The
    # searcher consumes total GPU memory; the M5 plan scopes ProTrain to
    # single-3090 runs so we read device 0 without enumerating the rest.
    device = 0
    props = torch.cuda.get_device_properties(device)
    gpu_memory_bytes = int(props.total_memory)
    gpu_sku = torch.cuda.get_device_name(device)

    # Measured PCIe bandwidth lives in the profiler trace; at plugin load
    # time we seed a reasonable prior. The cost model uses hardware_profile
    # for effective-bandwidth derating (cost/bandwidth.py) where the
    # absolute value matters less than the ratio against n_swap traffic.
    pcie_h2d_bps = _DEFAULT_PCIE_BPS
    pcie_d2h_bps = _DEFAULT_PCIE_BPS

    world_size = max(1, int(torch.cuda.device_count()))

    return HardwareProfile(
        gpu_sku=gpu_sku,
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_count=world_size,
        pcie_h2d_bps=pcie_h2d_bps,
        pcie_d2h_bps=pcie_d2h_bps,
        has_nvlink=False,
    )


class ProTrainPlugin(BasePlugin):
    """Plugin for ProTrain integration with Axolotl.

    Paper: MLSys 2026, arXiv 2406.08334. Exposes:

    * ``get_input_args`` — dotted path to ``ProTrainArgs``.
    * ``post_model_load`` — builds ``HardwareProfile``, calls
      ``protrain_model_wrapper``, stashes the returned ``WrappedModel``
      on ``cfg._protrain_wrapped`` for ``post_trainer_create`` to pick up.
    * ``create_optimizer`` — returns the ``_ProTrainOptimizer`` facade
      constructed from the stashed ``WrappedModel``. Per BasePlugin
      contract, but NOT the wiring path — Axolotl's ``OptimizerMixin``
      does not currently dispatch to ``PluginManager.create_optimizer``,
      so actual optimizer install happens in ``post_trainer_create``.
    * ``post_trainer_create`` — installs ``_ProTrainOptimizer`` on
      ``trainer.optimizer`` directly (this is the real wiring). Also
      auto-detects DDP composition and flips
      ``skip_internal_grad_reduce``.
    """

    def get_input_args(self) -> str:
        return "axolotl.integrations.protrain.args.ProTrainArgs"

    def get_training_args(self, cfg):
        """Force ``save_only_model=True`` so HF Trainer skips optim state save.

        ``_ProTrainOptimizer.state_dict`` / ``load_state_dict`` raise
        ``NotImplementedError`` — optimizer-state checkpointing lives
        in the M6 scope. Without this, ``save_steps`` would trigger a
        ``NotImplementedError`` at the first checkpoint. Setting
        ``save_only_model`` skips the ``_save_optimizer_and_scheduler``
        call entirely; the adapter / model weights still round-trip.
        """
        if not _is_plugin_active(cfg):
            return None
        return {"save_only_model": True}

    def post_model_load(self, cfg, model: "nn.Module") -> None:
        """Wrap the post-adapter model with the ProTrain runtime.

        Silently no-ops when the plugin is inactive (see
        ``_is_plugin_active``). Called after LoRA adapters are attached
        so persistent-chunk sizing reflects the trainable surface.
        """
        if not _is_plugin_active(cfg):
            return

        from axolotl.integrations.protrain.api import protrain_model_wrapper

        hw = _build_hardware_profile(cfg)

        # Pull knobs / overrides off the merged cfg. Pydantic already
        # validated the mutex with deepspeed/fsdp; here we just read.
        micro_batch_size = int(getattr(cfg, "micro_batch_size", 1) or 1)
        seq_len = int(getattr(cfg, "sequence_len", 1024) or 1024)
        capacity_bytes = getattr(cfg, "protrain_capacity_bytes", None)
        cache_dir = getattr(cfg, "protrain_cache_dir", None)
        force_all_persistent = bool(
            getattr(cfg, "protrain_force_all_persistent", False)
        )

        n_persist_override = getattr(cfg, "protrain_n_persist_override", None)
        n_buffer_override = getattr(cfg, "protrain_n_buffer_override", None)
        n_swap_override = getattr(cfg, "protrain_n_swap_override", None)
        n_checkpoint_override = getattr(
            cfg, "protrain_n_checkpoint_override", None
        )
        zero3_shard = getattr(cfg, "protrain_zero3_shard", None)

        wrapped = protrain_model_wrapper(
            model,
            model_config=getattr(model, "config", None),
            hardware_profile=hw,
            batch_size=micro_batch_size,
            seq_len=seq_len,
            capacity_bytes=capacity_bytes,
            cache_dir=cache_dir,
            force_all_persistent=force_all_persistent,
            n_persist_override=n_persist_override,
            n_buffer_override=n_buffer_override,
            n_swap_override=n_swap_override,
            n_checkpoint_override=n_checkpoint_override,
            zero3_shard=zero3_shard,
        )

        # Stash on cfg so post_trainer_create (which only receives cfg +
        # trainer) can recover the WrappedModel. Using a leading
        # underscore to signal "runtime state, not YAML-serialisable".
        cfg._protrain_wrapped = wrapped  # type: ignore[attr-defined]

        picked = wrapped.search_result.cfg
        LOG.info(
            "ProTrain: %s config picked (n_persist=%d, n_buffer=%d, "
            "n_checkpoint=%d, force_all_persistent=%s)",
            type(getattr(model, "base_model", model)).__name__,
            getattr(picked, "n_persist", -1),
            getattr(picked, "n_buffer", -1),
            getattr(picked, "n_checkpoint", -1),
            force_all_persistent,
        )

    def create_optimizer(
        self, cfg, trainer: "Trainer"
    ) -> "Optimizer | None":
        """Return the ProTrain optimizer facade, or ``None`` when inactive."""
        if not _is_plugin_active(cfg):
            return None

        wrapped = getattr(cfg, "_protrain_wrapped", None)
        if wrapped is None:
            # post_model_load wasn't called (or the model was None) —
            # fall through to Axolotl's default optimizer path rather
            # than raise, since that matches every other plugin's
            # "inactive -> return None" contract.
            LOG.warning(
                "ProTrain.create_optimizer: no _protrain_wrapped on cfg; "
                "post_model_load must have been skipped. Falling through to "
                "the default optimizer."
            )
            return None

        from axolotl.integrations.protrain.api import protrain_optimizer_wrapper

        args = trainer.args
        lr = float(args.learning_rate)
        betas = (float(args.adam_beta1), float(args.adam_beta2))
        eps = float(args.adam_epsilon)
        weight_decay = float(args.weight_decay)

        LOG.info(
            "ProTrain.create_optimizer: lr=%.3e betas=%s eps=%.1e wd=%.3e",
            lr,
            betas,
            eps,
            weight_decay,
        )

        return protrain_optimizer_wrapper(
            wrapped,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def post_trainer_create(self, cfg, trainer: "Trainer") -> None:
        """Install the ProTrain optimizer on the trainer.

        Axolotl's ``OptimizerMixin.create_optimizer`` does not dispatch
        to ``PluginManager.create_optimizer`` (unlike
        ``SchedulerMixin.create_scheduler``), so relying on
        :meth:`create_optimizer` alone leaves the plugin inert and the
        trainer falls back to vanilla AdamW. HuggingFace ``Trainer``
        checks ``self.optimizer`` before rebuilding one — setting
        ``trainer.optimizer`` here intercepts that path.

        Also auto-detects DDP composition and flips
        ``chunk_manager.skip_internal_grad_reduce`` so the outer DDP
        wrapper owns the cross-rank grad all-reduce rather than fighting
        with ProTrain's per-chunk reduce.
        """
        if not _is_plugin_active(cfg):
            return

        wrapped = getattr(cfg, "_protrain_wrapped", None)
        if wrapped is None:
            LOG.warning(
                "ProTrain: post_trainer_create fired without wrapped model; "
                "skipping optimizer install. post_model_load must have been "
                "skipped (non-CUDA run?) — falling back to the default "
                "optimizer."
            )
            return

        from axolotl.integrations.protrain.api import protrain_optimizer_wrapper

        args = trainer.args
        optim = protrain_optimizer_wrapper(
            wrapped,
            lr=float(args.learning_rate),
            betas=(float(args.adam_beta1), float(args.adam_beta2)),
            eps=float(args.adam_epsilon),
            weight_decay=float(args.weight_decay),
        )

        # ``_ProTrainOptimizer.state_dict`` raises NotImplementedError
        # (optim-state checkpointing is M6 scope). HF Trainer and
        # Accelerate both call ``state_dict`` unconditionally — HF at
        # checkpoint save (silenced via ``save_only_model=True`` in
        # ``get_training_args``) and Accelerate at ``prepare`` time for
        # device-placement (NOT silenced). Override the two methods on
        # this instance with safe no-ops so the bring-up path survives
        # without having to edit the api/ module (out-of-scope per the
        # fix plan). The safe no-op returns an empty param-state dict
        # preserving HF's ``{"param_groups": ...}`` shape so
        # Accelerate's ``move_to_device(state_dict, ...)`` +
        # ``load_state_dict(state_dict)`` round-trip does not crash.
        def _empty_state_dict(_self=optim):  # type: ignore[misc]
            return {
                "state": {},
                "param_groups": [
                    {k: v for k, v in g.items() if k != "params"}
                    | {"params": [i for i, _ in enumerate(g["params"])]}
                    for g in _self.param_groups
                ],
            }

        def _noop_load_state_dict(_state_dict, _self=optim):  # type: ignore[misc]
            # Accelerate re-loads the same (device-moved) state we just
            # returned — since neither adapter owns persistent state on
            # the torch side, discarding it is safe for the M5 scope.
            return None

        optim.state_dict = _empty_state_dict  # type: ignore[method-assign]
        optim.load_state_dict = _noop_load_state_dict  # type: ignore[method-assign]

        trainer.optimizer = optim
        LOG.info(
            "ProTrain: installed protrain_optimizer_wrapper on trainer.optimizer "
            "(lr=%.3e betas=%s eps=%.1e wd=%.3e)",
            float(args.learning_rate),
            (float(args.adam_beta1), float(args.adam_beta2)),
            float(args.adam_epsilon),
            float(args.weight_decay),
        )

        # ---- DDP composition detection ----------------------------------
        # If the trainer's model is wrapped in DistributedDataParallel,
        # defer cross-rank grad all-reduce to DDP and silence ProTrain's
        # internal reduce. Conversely, surface the case of multi-rank
        # init without DDP so the operator knows ProTrain's own reduce
        # path is still active (which is correct — just unusual).
        try:
            import torch
            from torch.nn.parallel import DistributedDataParallel
        except ImportError:
            return

        is_ddp = isinstance(trainer.model, DistributedDataParallel) or (
            hasattr(trainer, "model_wrapped")
            and isinstance(
                getattr(trainer, "model_wrapped", None), DistributedDataParallel
            )
        )
        if is_ddp:
            wrapped.chunk_manager.skip_internal_grad_reduce = True
            # DDP composition is incompatible with ZeRO-3 sharding —
            # the sharded path's reduce_scatter would overlap with
            # DDP's bucketed all_reduce. If sharding was auto-enabled
            # in post_model_load (before the DDP wrap), warn loudly:
            # at this point materialize_offload has already created
            # per-rank shards, so we can't cleanly revert. The
            # operator should have set ``protrain_zero3_shard: false``
            # in the YAML when composing with DDP.
            if getattr(wrapped.chunk_manager, "zero3_shard", False):
                LOG.warning(
                    "ProTrain: DDP composition detected but ZeRO-3 sharding "
                    "is active on the chunk manager. The two paths are not "
                    "composable (DDP + reduce_scatter would double-reduce). "
                    "Set ``protrain_zero3_shard: false`` in YAML to silence."
                )
            LOG.info(
                "ProTrain: detected DDP composition; set "
                "skip_internal_grad_reduce=True (DDP owns the cross-rank grad "
                "all-reduce)"
            )
        elif (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            LOG.warning(
                "ProTrain: multi-rank init (world_size=%d) detected but "
                "trainer.model is not wrapped in DistributedDataParallel; "
                "ProTrain's internal per-chunk grad all-reduce path remains "
                "active. This is the correct path for non-DDP multi-rank "
                "runs, but surface it here because it is unusual.",
                torch.distributed.get_world_size(),
            )


__all__ = ["ProTrainPlugin"]
