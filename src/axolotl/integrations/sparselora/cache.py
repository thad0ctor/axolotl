"""Transparent on-disk cache for SparseLoRA calibration artifacts.

A cache entry is a directory ``{cache_root}/{key}/`` holding:

- ``schedule.json`` -- the calibrated ``layer_sparsity`` map plus metadata.
- ``model.safetensors`` -- the SVD predictor factors (see ``factors.py``).

The cache key is a stable hash of everything the artifacts depend on (base
model, adapter, dataset signature, target sparsity, predictor rank, calibration
method and params). A matching key reuses the artifacts; change any input or
delete the directory to force recalibration. The resolved path is logged at
INFO on every run so it is never hidden from the user.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

SCHEDULE_FILE = "schedule.json"
FACTORS_FILE = "model.safetensors"


def _dataset_signature(cfg: Any) -> list:
    """Stable, content-light signature of the configured datasets."""
    sig = []
    for ds in cfg.datasets or []:
        ds = dict(ds) if not isinstance(ds, dict) else ds
        sig.append(
            {
                "path": ds.get("path"),
                "type": ds.get("type"),
                "name": ds.get("name"),
                "split": ds.get("split"),
                "data_files": ds.get("data_files"),
                "revision": ds.get("revision"),
            }
        )
    return sig


def compute_cache_key(cfg: Any) -> str:
    from .calibration import method_name

    settings = cfg.sparselora
    cal = settings.calibration
    payload = {
        "base_model": cfg.base_model,
        "adapter": cfg.adapter,
        "sequence_len": cfg.sequence_len,
        "train_on_inputs": cfg.train_on_inputs,
        "datasets": _dataset_signature(cfg),
        # The faithful schedule depends on the warm-up, which depends on LoRA.
        "lora": {
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "target_modules": cfg.lora_target_modules,
            "target_linear": cfg.lora_target_linear,
            "learning_rate": cfg.learning_rate,
        },
        "target_sparsity": settings.target_sparsity,
        "attn_sparsity": settings.attn_sparsity,
        "predictor_rank": settings.predictor_rank,
        "layer_sparsity": settings.layer_sparsity,
        "preset": settings.preset,
        "preset_mode": settings.preset_mode,
        "calibration": {
            "method": method_name(cal.method),
            "num_samples": cal.num_samples,
            "batch_size": cal.batch_size,
            "warmup_steps": cal.warmup_steps,
            "dense_prefix": cal.dense_prefix,
            "attn_dense_prefix": cal.attn_dense_prefix,
            "sensitivity_demote": cal.sensitivity_demote,
            # `loss_budget` is deprecated/ignored but kept in the key for
            # backward compatibility with entries written before it was retired.
            "loss_budget": cal.loss_budget,
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def resolve_cache_root(cfg: Any) -> str:
    if cfg.sparselora.cache_dir:
        return cfg.sparselora.cache_dir
    return os.path.join(cfg.output_dir or ".", "sparselora_calibration")


def entry_dir(cfg: Any, key: str) -> str:
    return os.path.join(resolve_cache_root(cfg), key)


def load_cached(cfg: Any, key: str) -> dict | None:
    """Return ``{"layer_sparsity", "meta", "factors_path"}`` if a valid entry exists."""
    path = entry_dir(cfg, key)
    schedule_path = os.path.join(path, SCHEDULE_FILE)
    factors_path = os.path.join(path, FACTORS_FILE)
    if not (os.path.isfile(schedule_path) and os.path.isfile(factors_path)):
        return None
    with open(schedule_path) as f:
        data = json.load(f)
    LOG.info("SparseLoRA: reusing cached calibration at %s", path)
    return {
        "layer_sparsity": data["layer_sparsity"],
        "meta": data.get("meta", {}),
        "factors_path": factors_path,
    }


def save_cached(
    cfg: Any,
    key: str,
    layer_sparsity: dict[str, float],
    meta: dict,
    factor_tensors: dict[str, torch.Tensor],
) -> str:
    """Persist schedule + factors and return the entry directory."""
    from safetensors.torch import save_file

    path = entry_dir(cfg, key)
    os.makedirs(path, exist_ok=True)
    # Atomic writes (temp + os.replace) so concurrent DDP ranks writing the same
    # deterministic entry can't leave a half-written file for a reader.
    factors_path = os.path.join(path, FACTORS_FILE)
    tmp_factors = f"{factors_path}.tmp.{os.getpid()}"
    save_file({k: v.cpu().contiguous() for k, v in factor_tensors.items()}, tmp_factors)
    os.replace(tmp_factors, factors_path)

    schedule_path = os.path.join(path, SCHEDULE_FILE)
    tmp_schedule = f"{schedule_path}.tmp.{os.getpid()}"
    with open(tmp_schedule, "w") as f:
        json.dump({"layer_sparsity": layer_sparsity, "meta": meta}, f, indent=2)
    os.replace(tmp_schedule, schedule_path)
    LOG.info("SparseLoRA: wrote calibration to %s", path)
    return path


def maybe_share(cfg: Any, layer_sparsity: dict[str, float], meta: dict) -> None:
    """Opt-in upstream sharing of the *schedule only* (never dataset content).

    Disabled unless ``sparselora.share_calibration`` is true. v1 ships the
    privacy-safe payload assembly only; no endpoint is wired until z-lab
    provides one, so this logs the payload at DEBUG rather than transmitting it.
    """
    if not cfg.sparselora.share_calibration:
        return
    from .calibration import method_name

    payload = {
        "base_model": cfg.base_model,
        "model_type": meta.get("model_type"),
        "target_sparsity": cfg.sparselora.target_sparsity,
        "predictor_rank": cfg.sparselora.predictor_rank,
        "calibration_method": method_name(cfg.sparselora.calibration.method),
        "layer_sparsity": layer_sparsity,
    }
    LOG.info(
        "SparseLoRA: share_calibration is enabled; assembled an anonymous schedule "
        "payload (no dataset content). No upstream endpoint is configured yet, so "
        "nothing was transmitted."
    )
    LOG.debug("SparseLoRA share payload: %s", json.dumps(payload, sort_keys=True))
