"""Derive ModelFeatures from a model's HF config + axolotl registries; an unavailable registry degrades one field to its default rather than failing (only an unresolvable config raises)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from . import ModelFeatures

DEFAULT_EMBED_LAYERS = ["embed_tokens", "lm_head"]
PATCH_MANAGER_REL = Path("src/axolotl/loaders/patch_manager.py")
VALIDATION_REL = Path("src/axolotl/utils/schemas/validation.py")
MODEL_LOADER_REL = Path("src/axolotl/loaders/model.py")

_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "num_experts_per_tok")
# fallback if validation.py can't be read (see _ssm_hybrid_types)
_SSM_HYBRID_FALLBACK = {"nemotron_h", "falcon_h1", "granitemoehybrid"}


# --------------------------------------------------------------------------- #
# config.json resolution
# --------------------------------------------------------------------------- #
def _load_config_dict(
    base_model: str, trust_remote_code: bool = False
) -> dict[str, Any]:
    local = Path(base_model).expanduser()
    local_cfg = local / "config.json"
    if local.is_dir() and local_cfg.is_file():
        return json.loads(local_cfg.read_text(encoding="utf-8"))
    if local.is_file() and local.name == "config.json":
        return json.loads(local.read_text(encoding="utf-8"))

    from transformers import AutoConfig

    # only execute remote model code behind an explicit opt-in
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    return config.to_dict()


# --------------------------------------------------------------------------- #
# static AST scans (registry-shape resilient, liger-audit style)
# --------------------------------------------------------------------------- #
def _str_constants(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [
            elt.value
            for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
    return []


def _type_dispatch_constants(tree: ast.AST, attrs: set[str]) -> set[str]:
    # Eq/In only (not NotIn): a `not in (...)` guard must not invert into every other type
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Attribute) and node.left.attr in attrs):
            continue
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            if isinstance(op, (ast.Eq, ast.In)):
                out.update(_str_constants(comparator))
    return out


def _patch_dispatch_types(repo_root: Path, warnings: list[str]) -> set[str] | None:
    path = repo_root / PATCH_MANAGER_REL
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as exc:
        warnings.append(f"patch_manager.py unreadable ({exc.__class__.__name__})")
        return None
    # both attrs hold the same un-normalized HF type
    return _type_dispatch_constants(tree, {"model_config_type", "model_type"})


def _ssm_hybrid_types(repo_root: Path, warnings: list[str]) -> set[str]:
    # the real set is a function-local in validation.py, so read it statically
    path = repo_root / VALIDATION_REL
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as exc:
        warnings.append(
            f"_SSM_HYBRID_MODEL_TYPES unreadable ({exc.__class__.__name__}); "
            "using pinned fallback"
        )
        return set(_SSM_HYBRID_FALLBACK)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SSM_HYBRID_MODEL_TYPES"
            for t in node.targets
        ):
            literals = set(_str_constants(node.value))
            if literals:
                return literals
    warnings.append(
        "_SSM_HYBRID_MODEL_TYPES not found in validation.py; using pinned fallback"
    )
    return set(_SSM_HYBRID_FALLBACK)


def _check_model_requirement_types(repo_root: Path) -> set[str]:
    # types whose _check_model_requirements branch gates on a dep (lfm2: causal-conv1d must be absent)
    path = repo_root / MODEL_LOADER_REL
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_check_model_requirements"
        ):
            return _type_dispatch_constants(node, {"model_config_type", "model_type"})
    return set()


# --------------------------------------------------------------------------- #
# registry-backed flags (degrade per-field on import failure)
# --------------------------------------------------------------------------- #
def _moe_arch_block(warnings: list[str]) -> set[str]:
    try:
        from axolotl.common.architectures import MOE_ARCH_BLOCK

        return set(MOE_ARCH_BLOCK)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"MOE_ARCH_BLOCK unavailable ({exc.__class__.__name__})")
        return set()


def _multimodal_types(warnings: list[str]) -> set[str]:
    try:
        from axolotl.loaders.constants import MULTIMODAL_AUTO_MODEL_MAPPING

        return set(MULTIMODAL_AUTO_MODEL_MAPPING)
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            f"MULTIMODAL_AUTO_MODEL_MAPPING unavailable ({exc.__class__.__name__})"
        )
        return set()


def _embed_layers(model_config_type: str, warnings: list[str]) -> list[str]:
    try:
        from axolotl.loaders.utils import get_linear_embedding_layers

        return get_linear_embedding_layers(model_config_type)
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            f"get_linear_embedding_layers unavailable ({exc.__class__.__name__})"
        )
        return list(DEFAULT_EMBED_LAYERS)


def _expert_count(config: dict[str, Any]) -> int | None:
    # VLM wrappers nest the backbone under text_config; check both levels
    for scope in (config, config.get("text_config") or {}):
        if not isinstance(scope, dict):
            continue
        for key in _EXPERT_COUNT_KEYS:
            val = scope.get(key)
            if isinstance(val, int) and val > 0:
                return val
    return None


def detect_model(
    base_model: str, repo_root: Path, trust_remote_code: bool = False
) -> ModelFeatures:
    warnings: list[str] = []
    config = _load_config_dict(base_model, trust_remote_code=trust_remote_code)

    model_config_type = config.get("model_type")
    if not isinstance(model_config_type, str) or not model_config_type:
        raise ValueError("config.json must contain a non-empty string model_type")
    architectures = list(config.get("architectures") or [])
    extra: dict[str, Any] = {}

    # MoE: registry membership OR a positive expert-count signal
    moe_block = _moe_arch_block(warnings)
    expert_count = _expert_count(config)
    is_moe = model_config_type in moe_block or expert_count is not None
    if expert_count is not None:
        extra["num_experts"] = expert_count

    # multimodal: image-text-to-text routing, pixtral name heuristic, or nested vision config
    mm_types = _multimodal_types(warnings)
    is_multimodal = (
        model_config_type in mm_types
        or "pixtral" in base_model.lower()
        or isinstance(config.get("vision_config"), dict)
    )

    is_ssm_hybrid = model_config_type in _ssm_hybrid_types(repo_root, warnings)

    patch_types = _patch_dispatch_types(repo_root, warnings)
    needs_patch = bool(patch_types and model_config_type in patch_types)

    custom_embed_names = (
        _embed_layers(model_config_type, warnings) != DEFAULT_EMBED_LAYERS
    )

    if model_config_type in _check_model_requirement_types(repo_root):
        # lfm2: _check_model_requirements raises if causal-conv1d IS present
        extra["causal_conv1d_conflict"] = True

    if warnings:
        extra["detect_warnings"] = warnings

    return ModelFeatures(
        model_config_type=model_config_type,
        base_model=base_model,
        is_moe=is_moe,
        is_multimodal=is_multimodal,
        is_ssm_hybrid=is_ssm_hybrid,
        needs_patch=needs_patch,
        custom_embed_names=custom_embed_names,
        architectures=architectures,
        config_json=config,
        extra=extra,
    )
