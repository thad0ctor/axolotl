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

"""Pydantic argument model for the ProTrain plugin (M5, DESIGN.md §Plugin Integration).

Merged into the top-level Axolotl config schema at validation time via the
``plugins:`` entry in the user YAML. Mirrors the shape of
``axolotl.integrations.liger.LigerArgs`` / ``axolotl.integrations.spectrum.SpectrumArgs``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ProTrainArgs(BaseModel):
    """Input args for the ProTrain plugin.

    The plugin is opt-in at two levels: (1) the YAML must list
    ``axolotl.integrations.protrain`` in ``plugins:``, and (2)
    ``protrain_auto_memory`` must be True. The second gate lets users add
    the plugin import for args-schema registration without actually
    rewiring the training path (useful for validation / documentation).
    """

    protrain_auto_memory: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Master enable flag for ProTrain automatic memory management. "
                "When True, the plugin's post_model_load hook wraps the model "
                "with the hierarchical chunk manager + interleaved block manager, "
                "and create_optimizer returns the ProTrain optimizer. "
                "Requires ``plugins: [axolotl.integrations.protrain]``. "
                "Mutually exclusive with ``deepspeed:`` and ``fsdp:`` / ``fsdp_config:``."
            )
        },
    )

    protrain_force_all_persistent: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": (
                "Override the searcher and force every chunk to stay GPU-resident "
                "(n_persist = N_chunk, n_swap = 0, n_checkpoint = N_block). "
                "Recommended on 24 GB cards with LoRA until the M4.5 runtime "
                "primitives (init-time chunk offload, per-param grad offload) land. "
                "With those gaps in place, search-picked configs that rely on CPU-"
                "hosted non-persistent chunks OOM on 7B-class models; "
                "force_all_persistent keeps model state GPU-resident and relies on "
                "activation checkpointing to trim peak memory — a valid and useful "
                "ProTrain configuration for LoRA on single 3090s."
            )
        },
    )

    protrain_capacity_bytes: int | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Override the GPU memory budget (bytes) the searcher respects. "
                "When None, defaults to ``gpu_memory_bytes - 2 GiB`` headroom "
                "for the CUDA context + allocator reserve."
            )
        },
    )

    protrain_cache_dir: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Override the profiler-cache directory. When None, the cache "
                "lives under the standard XDG cache root."
            )
        },
    )

    # Debugging escape hatches — bypass the searcher. Intended for
    # reproducibility experiments and bug-hunting; production runs should
    # leave these None and let the cost model pick.
    protrain_n_persist_override: int | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Debug override: force the number of persistent chunks. "
                "Bypasses the exhaustive searcher when set alongside the other "
                "three overrides."
            )
        },
    )
    protrain_n_buffer_override: int | None = Field(
        default=None,
        json_schema_extra={"description": "Debug override for n_buffer."},
    )
    protrain_n_swap_override: int | None = Field(
        default=None,
        json_schema_extra={"description": "Debug override for n_swap."},
    )
    protrain_n_checkpoint_override: int | None = Field(
        default=None,
        json_schema_extra={"description": "Debug override for n_checkpoint."},
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _require_plugin_registration(cls, data):
        """``protrain_auto_memory=True`` requires the plugin in ``plugins:``.

        Clone of the enable-guard pattern used by Liger / Spectrum: the
        plugin being present in ``plugins:`` is what causes its args
        model to be merged in, but a user could set the YAML flag without
        the plugin import — this validator surfaces that misconfiguration
        as a clear ValueError instead of a silently-ignored flag.
        """
        if not isinstance(data, dict):
            return data
        if not data.get("protrain_auto_memory"):
            return data
        plugins = data.get("plugins") or []
        has_protrain = any(
            isinstance(p, str) and "protrain" in p.lower() for p in plugins
        )
        if not has_protrain:
            raise ValueError(
                "`protrain_auto_memory: true` requires the ProTrain plugin to be "
                "listed in `plugins:`. Add "
                "`- axolotl.integrations.protrain` to the `plugins` list."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_deepspeed_fsdp_coexistence(cls, data):
        """Mutex with DeepSpeed / FSDP — mirror ``spectrum/args.py:32-47``.

        ProTrain owns per-rank memory policy; running it inside a
        DeepSpeed / FSDP model factory would double-manage model state,
        grads, and optim state. Refuse the combination at load-time.
        """
        if not isinstance(data, dict):
            return data
        if not data.get("protrain_auto_memory"):
            return data
        plugins = data.get("plugins") or []
        if not any(
            isinstance(p, str) and "protrain" in p.lower() for p in plugins
        ):
            return data
        if data.get("deepspeed"):
            raise ValueError(
                "ProTrain + DeepSpeed cannot be used together: both manage "
                "per-rank model-state placement. Remove `deepspeed:` or disable "
                "`protrain_auto_memory`."
            )
        if data.get("fsdp") or data.get("fsdp_config"):
            raise ValueError(
                "ProTrain + FSDP cannot be used together: both manage "
                "per-rank model-state placement. Remove `fsdp:` / `fsdp_config:` "
                "or disable `protrain_auto_memory`."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _require_model_or_adapter(cls, data):
        """Basic sanity: a training run needs a base model (adapter is optional)."""
        if not isinstance(data, dict):
            return data
        if not data.get("protrain_auto_memory"):
            return data
        plugins = data.get("plugins") or []
        if not any(
            isinstance(p, str) and "protrain" in p.lower() for p in plugins
        ):
            return data
        if not (data.get("base_model") or data.get("model_name_or_path")):
            raise ValueError(
                "`protrain_auto_memory: true` requires a `base_model` (or "
                "`model_name_or_path`) to be configured."
            )
        return data
