"""G2 — integration gate: derive the *expected* hook set from ``ctx.features`` and classify each four-way (present_explicit / generic_fallback / missing / not_expected). generic_fallback is the shadow failure mode — surfaced, never green; an unreadable dispatch shape leans toward COULD_NOT_RUN."""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import GateContext, GateResult, GateStatus

GATE_ID = "G2"
GATE_NAME = "integration"

PRESENT = "present_explicit"
GENERIC = "generic_fallback"
MISSING = "missing"
NOT_EXPECTED = "not_expected"

_ICON = {PRESENT: "✅", GENERIC: "⚠️", MISSING: "❌", NOT_EXPECTED: "·"}

# default module names get_linear_embedding_layers falls back to
_DEFAULT_EMBED = ["embed_tokens", "lm_head"]


def applies(ctx: GateContext) -> bool:  # noqa: ARG001 - always applies
    return True


@dataclass
class HookRow:
    hook: str
    file: str
    gated_on: str
    status: str
    note: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "hook": self.hook,
            "file": self.file,
            "gated_on": self.gated_on,
            "status": self.status,
            "note": self.note,
        }


@dataclass
class Probe:
    """Shared state for the checks: paths, feature facts, reliability warnings."""

    ctx: GateContext
    warnings: list[str] = field(default_factory=list)
    rows: list[HookRow] = field(default_factory=list)
    # set when a load-bearing read failed, so the aggregate can't be trusted -> COULD_NOT_RUN
    critical_unreadable: bool = False

    @property
    def repo(self) -> Path:
        return self.ctx.repo_root

    @property
    def mct(self) -> str:
        return self.ctx.features.model_config_type

    @property
    def opt(self) -> dict[str, Any]:
        return self.ctx.options

    def add(self, row: HookRow) -> None:
        self.rows.append(row)

    def warn(self, msg: str, critical: bool = False) -> None:
        self.warnings.append(msg)
        if critical:
            self.critical_unreadable = True

    def src(self, rel: str) -> Path:
        return self.repo / "src" / "axolotl" / rel


# ---------------------------------------------------------------------------
# AST helpers (mirrors audit_liger_sync.py: read dispatch ladders, not bodies)
# ---------------------------------------------------------------------------


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


_TYPE_ATTRS = {"model_config_type", "model_type", "model_config_type_text"}


def _compared_model_types(test: ast.expr) -> set[str]:
    # Any `<...>.model_(config_)type == "x"` / `in ("x", ...)` comparison.
    out: set[str] = set()
    for node in ast.walk(test):
        if isinstance(node, ast.Compare):
            attr = node.left
            if isinstance(attr, ast.Attribute) and attr.attr in _TYPE_ATTRS:
                for op, comp in zip(node.ops, node.comparators, strict=False):
                    if isinstance(op, (ast.Eq, ast.In)):
                        out.update(_str_constants(comp))
    return out


def _file_model_types(path: Path) -> tuple[set[str], bool]:
    """Every string a file dispatches on via a model-type comparison (whole-file). readable=False on a missing/syntax-error file so the caller can degrade."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return set(), False
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            out |= _compared_model_types(node)
    return out, True


def _named_set(path: Path, name: str) -> tuple[set[str], bool]:
    """String members of a literal set/tuple/list assigned to ``name``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):  # noqa: BLE001
        return set(), False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            if isinstance(node.value, (ast.Set, ast.Tuple, ast.List)):
                vals = _str_constants(node.value)
                if vals:
                    return set(vals), True
    return set(), False


def _func_string_keys(path: Path, func_name: str) -> tuple[set[str], bool]:
    """All string constants a function dispatches on via ``==``/``in`` (get_processing_strategy keys on chat_template_type, not model_type)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):  # noqa: BLE001
        return set(), False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            out: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Compare):
                    for op, comp in zip(sub.ops, sub.comparators, strict=False):
                        if isinstance(op, (ast.Eq, ast.In)):
                            out.update(_str_constants(comp))
            return out, True
    return set(), False


def _method_model_types(path: Path) -> tuple[dict[str, set[str]], bool]:
    """Per-method (``FunctionDef.name`` -> types it dispatches on) so the patch_manager lifecycle can be reported per-stage."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return {}, False
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            types: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Compare):
                    types |= _compared_model_types(sub)
            if types:
                out[node.name] = types
    return out, True


def _preconfig_intercept_strings(path: Path) -> tuple[set[str], bool]:
    """Model tokens the pre-config/pre-tokenizer remote-code intercepts key on (substring of cfg.base_model_config, not model_config_type)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return set(), False
    targets = {"base_model_config", "tokenizer_config"}
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant):
            for op, comp in zip(node.ops, node.comparators, strict=False):
                if isinstance(op, ast.In) and isinstance(node.left.value, str):
                    refs = {
                        n.attr for n in ast.walk(comp) if isinstance(n, ast.Attribute)
                    }
                    if refs & targets:
                        out.add(node.left.value)
    return out, True


def _registered_adapter_modules(init_path: Path) -> tuple[set[str], bool]:
    """Module stems imported by ``_all_adapters()`` — the actual registry, so a stray unwired file isn't counted as a dedicated adapter."""
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return set(), False
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_all_adapters":
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom) and sub.module:
                    out.add(sub.module.rsplit(".", 1)[-1])
    return out, True


def _adapter_info(path: Path) -> tuple[set[str], str | None, bool]:
    """(model-type tokens, adapter class name, readable) for one adapter file; tokens = stem + ``name=`` attr + type-compare constants."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return set(), None, False
    tokens = {path.stem}
    cls_name: str | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name.endswith("Adapter")
            and node.name != "ModelAdapter"
        ):
            cls_name = node.name
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "name" for t in node.targets
        ):
            tokens.update(_str_constants(node.value))
        if isinstance(node, ast.Compare):
            tokens |= _compared_model_types(node)
        # str(getattr(cfg, "model_type", "")).startswith("gemma4") — gated on a type ref
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("startswith", "endswith")
            and any(
                isinstance(n, ast.Constant) and n.value in _TYPE_ATTRS
                for n in ast.walk(node)
            )
        ):
            for arg in node.args:
                tokens.update(_str_constants(arg))
    return tokens, cls_name, True


def _family_match(mct: str, token: str) -> bool:
    """mct matches token on equality or a segment-boundary family prefix (``gemma4_text`` matches ``gemma4`` but ``gemma4x`` does not)."""
    if not mct or not token:
        return False
    if mct == token:
        return True
    if mct.startswith(token) and mct[len(token) : len(token) + 1] in ("", "_"):
        return True
    if token.startswith(mct) and token[len(mct) : len(mct) + 1] in ("", "_"):
        return True
    return False


def _safe_import(modpath: str, attr: str) -> tuple[Any, str | None]:
    try:
        mod = importlib.import_module(modpath)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the gate
        return None, f"{exc.__class__.__name__}: {exc}"
    if not hasattr(mod, attr):
        return None, f"{modpath} has no attribute {attr}"
    return getattr(mod, attr), None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_model_loads(p: Probe) -> None:
    feats = p.ctx.features
    mm_map, mm_err = _safe_import(
        "axolotl.loaders.constants", "MULTIMODAL_AUTO_MODEL_MAPPING"
    )
    cfg_names, cfg_err = _safe_import(
        "transformers.models.auto.configuration_auto", "CONFIG_MAPPING_NAMES"
    )
    if cfg_err is not None and (mm_err is not None or not feats.is_multimodal):
        p.warn(f"cannot resolve AutoModel mapping ({cfg_err})", critical=True)
        p.add(
            HookRow(
                "model loads (AutoModel resolves)",
                "transformers auto-mapping",
                "always",
                MISSING,
                f"unreadable: {cfg_err}",
            )
        )
        return
    in_cfg = bool(cfg_names) and p.mct in cfg_names
    in_mm = bool(mm_map) and p.mct in mm_map
    has_arch = bool(feats.architectures) or bool(feats.config_json.get("architectures"))
    if feats.is_multimodal:
        if in_mm:
            p.add(
                HookRow(
                    "model loads (multimodal AutoModel)",
                    "loaders/constants.py",
                    "is_multimodal",
                    PRESENT,
                    "in MULTIMODAL_AUTO_MODEL_MAPPING",
                )
            )
        elif in_cfg or has_arch:
            p.add(
                HookRow(
                    "model loads (multimodal AutoModel)",
                    "loaders/constants.py",
                    "is_multimodal",
                    GENERIC,
                    "resolves as text/remote but absent from MULTIMODAL_AUTO_MODEL_MAPPING",
                )
            )
        else:
            p.add(
                HookRow(
                    "model loads (multimodal AutoModel)",
                    "loaders/constants.py",
                    "is_multimodal",
                    MISSING,
                    "no AutoModel mapping for this type",
                )
            )
        return
    if in_cfg:
        p.add(
            HookRow(
                "model loads (AutoModel resolves)",
                "transformers CONFIG_MAPPING",
                "always",
                PRESENT,
                "registered transformers config type",
            )
        )
    elif has_arch:
        p.add(
            HookRow(
                "model loads (AutoModel resolves)",
                "transformers CONFIG_MAPPING",
                "always",
                GENERIC,
                "not a builtin transformers type; relies on custom/remote architectures",
            )
        )
    else:
        p.add(
            HookRow(
                "model loads (AutoModel resolves)",
                "transformers CONFIG_MAPPING",
                "always",
                MISSING,
                "type not registered and no architectures in config",
            )
        )


def _check_chat_template(p: Probe) -> None:
    # advisory: inline jinja and tokenizer_default are equally valid, so never a gap
    enum_vals: set[str] = set()
    ev, _ = _safe_import("axolotl.utils.schemas.enums", "ChatTemplate")
    if ev is not None:
        enum_vals = {e.value for e in ev}
    jinja_dir = p.src("utils/chat_templates/templates")
    stems = {f.stem for f in jinja_dir.glob("*.jinja")} if jinja_dir.is_dir() else set()
    if p.mct in enum_vals or p.mct in stems:
        where = "enum" if p.mct in enum_vals else "bundled .jinja"
        p.add(
            HookRow(
                "chat template resolves",
                "utils/chat_templates/",
                "always (advisory)",
                PRESENT,
                f"named template available ({where})",
            )
        )
    else:
        p.add(
            HookRow(
                "chat template resolves",
                "utils/chat_templates/",
                "always (advisory)",
                NOT_EXPECTED,
                "no named/bundled template; inline jinja or tokenizer_default still valid",
            )
        )


def _check_multipack(p: Probe) -> None:
    types, err = _safe_import(
        "axolotl.monkeypatch.multipack", "SUPPORTED_MULTIPACK_MODEL_TYPES"
    )
    wants_packing = bool(p.opt.get("sample_packing"))
    gated = "sample_packing" if wants_packing else "sample_packing (advisory)"
    if err is not None:
        p.warn(f"SUPPORTED_MULTIPACK_MODEL_TYPES unreadable ({err})")
        p.add(
            HookRow(
                "multipack support",
                "monkeypatch/multipack.py",
                gated,
                MISSING if wants_packing else NOT_EXPECTED,
                f"unreadable: {err}",
            )
        )
        return
    if p.mct in set(types):
        p.add(
            HookRow(
                "multipack support",
                "monkeypatch/multipack.py",
                gated,
                PRESENT,
                "in SUPPORTED_MULTIPACK_MODEL_TYPES",
            )
        )
    elif wants_packing:
        p.add(
            HookRow(
                "multipack support",
                "monkeypatch/multipack.py",
                gated,
                MISSING,
                "sample_packing requested but type not in SUPPORTED_MULTIPACK_MODEL_TYPES",
            )
        )
    else:
        p.add(
            HookRow(
                "multipack support",
                "monkeypatch/multipack.py",
                gated,
                NOT_EXPECTED,
                "not packing-enabled (sample_packing not requested)",
            )
        )


# patch_manager lifecycle stages (hook label, feeding methods, present-note); split so a reviewer sees which stage a type hooks
_PM_STAGES: list[tuple[str, set[str], str]] = [
    (
        "PatchManager: pre-model dispatch",
        {"_apply_model_specific_patches", "_apply_mistral_cross_entropy_patch"},
        "model-specific branch in the pre-model-load patch sequence",
    ),
    (
        "PatchManager: Voxtral/Apertus pre-model",
        {"_apply_voxtral_patches", "_apply_apertus_patches"},
        "Voxtral/Apertus forward/activation patch",
    ),
    (
        "PatchManager: post-build loss/conversion",
        {"apply_post_model_build_patches", "_apply_gemma4_loss_kwargs"},
        "post-build Gemma loss-kwargs / nemotron_h conversion fix",
    ),
    (
        "PatchManager: post-load model patch",
        {"_apply_llama_flash_attn_patches"},
        "post-load flash-attn / swiglu patch",
    ),
]
# known hijack types so a missing branch reads as drift, not "not a hijack type"
_HIJACK_TYPES = {"btlm", "stablelm_epoch", "mistral3", "llava"}


def _check_patch_manager(p: Probe) -> tuple[set[str], bool]:
    path = p.src("loaders/patch_manager.py")
    method_types, ok = _method_model_types(path)
    needs = p.ctx.features.needs_patch
    if not ok:
        p.warn("patch_manager.py dispatch unreadable", critical=needs)
        p.add(
            HookRow(
                "PatchManager dispatch",
                "loaders/patch_manager.py",
                "needs_patch",
                MISSING if needs else NOT_EXPECTED,
                "dispatch ladder unreadable",
            )
        )
        return set(), False

    all_types: set[str] = set().union(*method_types.values()) if method_types else set()

    # pre-config / pre-tokenizer remote-code intercept (substring keyed, not model_type)
    pre_strs, _ = _preconfig_intercept_strings(path)
    norm = lambda s: s.lower().replace("-", "").replace("_", "")  # noqa: E731
    mct_n = norm(p.mct)
    pre_hit = bool(mct_n) and any(
        norm(s) and (mct_n == norm(s) or mct_n.startswith(norm(s))) for s in pre_strs
    )
    p.add(
        HookRow(
            "PatchManager: pre-config/tokenizer intercept",
            "loaders/patch_manager.py",
            "needs_patch",
            PRESENT if pre_hit else NOT_EXPECTED,
            "remote-code config/tokenizer intercept for this type"
            if pre_hit
            else "no pre-config/tokenizer remote-code intercept",
        )
    )

    for hook, methods, present_note in _PM_STAGES:
        stage_types: set[str] = set().union(
            *(method_types.get(m, set()) for m in methods)
        )
        hit = p.mct in stage_types
        p.add(
            HookRow(
                hook,
                "loaders/patch_manager.py",
                "needs_patch",
                PRESENT if hit else NOT_EXPECTED,
                present_note if hit else "no hook for this type at this stage",
            )
        )

    # needs_patch guard: a patch was expected but the type hooks no lifecycle stage
    if needs and not (pre_hit or p.mct in all_types):
        p.add(
            HookRow(
                "PatchManager dispatch",
                "loaders/patch_manager.py",
                "needs_patch",
                MISSING,
                "needs_patch=True but no branch keys on this type at any lifecycle stage",
            )
        )

    # attention hijack (btlm/stablelm/mistral3/llava) lives in _patch_attention
    if p.mct in _HIJACK_TYPES:
        present = p.mct in method_types.get("_patch_attention", set())
        p.add(
            HookRow(
                "attention hijack",
                "loaders/patch_manager.py",
                "hijack types",
                PRESENT if present else MISSING,
                "flash-attn / sdpa hijack branch present"
                if present
                else "hijack type but no dispatch branch found",
            )
        )
    return all_types, ok


def _check_fused_attn(p: Probe) -> None:
    path = p.src("loaders/patch_manager.py")
    types, ok = _named_set(path, "_FUSED_ATTN_KERNEL_SUPPORTED")
    wants = bool(p.opt.get("fused_attn_kernel")) or p.mct in types
    gated = (
        "fused_attn_kernel"
        if p.opt.get("fused_attn_kernel")
        else "fused_attn_kernel (advisory)"
    )
    if not ok:
        p.warn("_FUSED_ATTN_KERNEL_SUPPORTED unreadable")
    if not wants:
        p.add(
            HookRow(
                "fused attn kernel support",
                "loaders/patch_manager.py",
                gated,
                NOT_EXPECTED,
                "fused_attn_kernel not requested for this type",
            )
        )
    elif not ok:
        p.add(
            HookRow(
                "fused attn kernel support",
                "loaders/patch_manager.py",
                gated,
                MISSING,
                "fused_attn_kernel requested but support set unreadable",
            )
        )
    elif p.mct in types:
        p.add(
            HookRow(
                "fused attn kernel support",
                "loaders/patch_manager.py",
                gated,
                PRESENT,
                "in _FUSED_ATTN_KERNEL_SUPPORTED",
            )
        )
    else:
        p.add(
            HookRow(
                "fused attn kernel support",
                "loaders/patch_manager.py",
                gated,
                MISSING,
                "fused_attn_kernel requested but type not supported",
            )
        )


def _check_liger(p: Probe) -> None:
    # explicit = dedicated elif branch OR native MODEL_TYPE_TO_APPLY_LIGER_FN; generic_fallback = neither (only experimental patch_lce_forward FLCE)
    native, nerr = _safe_import(
        "liger_kernel.transformers.monkey_patch", "MODEL_TYPE_TO_APPLY_LIGER_FN"
    )
    elif_types, ok = _file_model_types(p.src("integrations/liger/plugin.py"))
    if not ok:
        p.warn("liger plugin.py dispatch ladder unreadable")
    native_keys = set(native) if native is not None else set()
    if nerr is not None:
        p.warn(
            f"liger-kernel not importable ({nerr}); native-path classification degraded"
        )
    if p.mct in elif_types:
        p.add(
            HookRow(
                "liger routing",
                "integrations/liger/plugin.py",
                "liger (opt-in)",
                PRESENT,
                "dedicated elif branch in axolotl liger plugin",
            )
        )
    elif p.mct in native_keys:
        p.add(
            HookRow(
                "liger routing",
                "integrations/liger/plugin.py",
                "liger (opt-in)",
                PRESENT,
                "native upstream support (MODEL_TYPE_TO_APPLY_LIGER_FN)",
            )
        )
    elif native is None and not ok:
        p.add(
            HookRow(
                "liger routing",
                "integrations/liger/plugin.py",
                "liger (opt-in)",
                NOT_EXPECTED,
                "could not read liger plugin or upstream table",
            )
        )
    else:
        p.add(
            HookRow(
                "liger routing",
                "integrations/liger/plugin.py",
                "liger (opt-in)",
                GENERIC,
                "no dedicated branch / native entry; falls to experimental generic "
                "FLCE (patch_lce_forward) — review before relying on liger",
            )
        )


def _check_cce(p: Probe) -> None:
    # explicit = an upstream PATCH_FNS entry; else patch_llama_like installs a generic llama-shaped patch = generic_fallback
    patch_fns, err = _safe_import("cut_cross_entropy.transformers.patch", "PATCH_FNS")
    if err is not None:
        p.warn(f"CCE PATCH_FNS not importable ({err}); routing classification degraded")
        p.add(
            HookRow(
                "CCE routing",
                "integrations/cut_cross_entropy/__init__.py",
                "cut_cross_entropy (opt-in)",
                NOT_EXPECTED,
                f"unreadable: {err}",
            )
        )
        return
    if p.mct in set(patch_fns):
        p.add(
            HookRow(
                "CCE routing",
                "integrations/cut_cross_entropy/__init__.py",
                "cut_cross_entropy (opt-in)",
                PRESENT,
                "dedicated upstream PATCH_FNS entry",
            )
        )
    else:
        p.add(
            HookRow(
                "CCE routing",
                "integrations/cut_cross_entropy/__init__.py",
                "cut_cross_entropy (opt-in)",
                GENERIC,
                "not in PATCH_FNS; patch_llama_like installs an experimental generic "
                "llama-shaped patch — review masking/loss for this arch",
            )
        )


def _check_moe(p: Probe) -> None:
    is_moe = p.ctx.features.is_moe
    block, err = _safe_import("axolotl.common.architectures", "MOE_ARCH_BLOCK")
    if err is not None:
        p.warn(f"MOE_ARCH_BLOCK unreadable ({err})")
    if not is_moe:
        present = block is not None and p.mct in block
        p.add(
            HookRow(
                "MoE arch block",
                "common/architectures.py",
                "is_moe",
                PRESENT if present else NOT_EXPECTED,
                "in MOE_ARCH_BLOCK" if present else "not a MoE architecture",
            )
        )
        return
    if err is not None:
        p.add(
            HookRow(
                "MoE arch block",
                "common/architectures.py",
                "is_moe",
                MISSING,
                f"is_moe but MOE_ARCH_BLOCK unreadable: {err}",
            )
        )
    elif p.mct in block:
        p.add(
            HookRow(
                "MoE arch block",
                "common/architectures.py",
                "is_moe",
                PRESENT,
                f"MOE_ARCH_BLOCK[{p.mct}] = {block[p.mct]!r}",
            )
        )
    else:
        p.add(
            HookRow(
                "MoE arch block",
                "common/architectures.py",
                "is_moe",
                MISSING,
                "is_moe=True but no MOE_ARCH_BLOCK entry (MoE FSDP wrap will be wrong)",
            )
        )
    # dedicated dsv4/gemma4/glm adapter vs the generic scattermoe/sonicmoe backend (the designed default for any MoE, not a bug)
    adapters = p.src("integrations/kernels/adapters")
    registered, reg_ok = _registered_adapter_modules(adapters / "__init__.py")
    if not reg_ok:
        p.warn("kernels adapter registry (_all_adapters) unreadable")
        registered = (
            {f.stem for f in adapters.glob("*.py")} - {"__init__"}
            if adapters.is_dir()
            else set()
        )
    dedicated_cls: str | None = None
    for stem in sorted(registered):
        toks, cls_name, tok_ok = _adapter_info(adapters / f"{stem}.py")
        if not tok_ok:
            p.warn(f"kernels adapter {stem}.py unreadable")
            continue
        if any(_family_match(p.mct, t) for t in toks):
            dedicated_cls = cls_name or stem
            break
    p.add(
        HookRow(
            "MoE kernel adapter",
            "integrations/kernels/adapters/",
            "is_moe",
            PRESENT,
            f"dedicated {dedicated_cls} matches this type"
            if dedicated_cls
            else f"generic scattermoe/sonicmoe backend (dedicated adapters: {sorted(registered)})",
        )
    )


def _check_multimodal(p: Probe) -> None:
    is_mm = p.ctx.features.is_multimodal
    # processing strategy (masking) — keyed on chat_template_type
    keys, ok = _func_string_keys(
        p.src("processing_strategies.py"), "get_processing_strategy"
    )
    if not ok:
        p.warn("get_processing_strategy keys unreadable")
    if not is_mm:
        p.add(
            HookRow(
                "MM processing strategy",
                "processing_strategies.py",
                "is_multimodal",
                NOT_EXPECTED,
                "not a multimodal model",
            )
        )
        return
    if not ok:
        p.add(
            HookRow(
                "MM processing strategy",
                "processing_strategies.py",
                "is_multimodal",
                MISSING,
                "is_multimodal but get_processing_strategy unreadable",
            )
        )
    elif p.mct in keys:
        p.add(
            HookRow(
                "MM processing strategy",
                "processing_strategies.py",
                "is_multimodal (via chat_template_type)",
                PRESENT,
                f"model_config_type {p.mct!r} matches a get_processing_strategy key; "
                "note: selection is keyed on chat_template_type, not model_type",
            )
        )
    else:
        p.add(
            HookRow(
                "MM processing strategy",
                "processing_strategies.py",
                "is_multimodal (via chat_template_type)",
                GENERIC,
                f"model_config_type {p.mct!r} not among get_processing_strategy keys "
                "(best-effort: strategy is keyed on chat_template_type, not model_type); "
                "generic image-token masking may mislabel for this arch",
            )
        )


def _check_ssm_hybrid(p: Probe, pm_types: set[str], pm_ok: bool) -> None:
    is_hybrid = p.ctx.features.is_ssm_hybrid
    types, ok = _named_set(
        p.src("utils/schemas/validation.py"), "_SSM_HYBRID_MODEL_TYPES"
    )
    if not ok:
        p.warn("_SSM_HYBRID_MODEL_TYPES unreadable")
    if not is_hybrid:
        present = ok and p.mct in types
        p.add(
            HookRow(
                "SSM-hybrid registration",
                "utils/schemas/validation.py",
                "is_ssm_hybrid",
                PRESENT if present else NOT_EXPECTED,
                "in _SSM_HYBRID_MODEL_TYPES"
                if present
                else "not an SSM/attention hybrid",
            )
        )
        return
    if not ok:
        p.add(
            HookRow(
                "SSM-hybrid registration",
                "utils/schemas/validation.py",
                "is_ssm_hybrid",
                MISSING,
                "is_ssm_hybrid but set unreadable",
            )
        )
    elif p.mct in types:
        p.add(
            HookRow(
                "SSM-hybrid registration",
                "utils/schemas/validation.py",
                "is_ssm_hybrid",
                PRESENT,
                "in _SSM_HYBRID_MODEL_TYPES",
            )
        )
    else:
        p.add(
            HookRow(
                "SSM-hybrid registration",
                "utils/schemas/validation.py",
                "is_ssm_hybrid",
                MISSING,
                "is_ssm_hybrid=True but not in _SSM_HYBRID_MODEL_TYPES",
            )
        )
    # hybrid packing patch (per-type branch in patch_manager)
    wants_packing = bool(p.opt.get("sample_packing"))
    gated = "hybrid + packing" if wants_packing else "hybrid + packing (advisory)"
    if not wants_packing:
        p.add(
            HookRow(
                "hybrid packing patch",
                "loaders/patch_manager.py",
                gated,
                NOT_EXPECTED,
                "sample_packing not requested",
            )
        )
    elif not pm_ok:
        p.add(
            HookRow(
                "hybrid packing patch",
                "loaders/patch_manager.py",
                gated,
                MISSING,
                "patch_manager unreadable",
            )
        )
    elif p.mct in pm_types:
        p.add(
            HookRow(
                "hybrid packing patch",
                "loaders/patch_manager.py",
                gated,
                PRESENT,
                "model-specific GC-compatible packing patch present",
            )
        )
    else:
        p.add(
            HookRow(
                "hybrid packing patch",
                "loaders/patch_manager.py",
                gated,
                MISSING,
                "hybrid + packing but no modeling-packing patch for this type",
            )
        )


def _check_embed_names(p: Probe) -> None:
    custom = p.ctx.features.custom_embed_names
    fn, err = _safe_import("axolotl.loaders.utils", "get_linear_embedding_layers")
    if err is not None:
        p.warn(f"get_linear_embedding_layers unreadable ({err})")
        p.add(
            HookRow(
                "embedding layer names",
                "loaders/utils.py",
                "custom_embed_names",
                MISSING if custom else NOT_EXPECTED,
                f"unreadable: {err}",
            )
        )
        return
    names = fn(p.mct)
    is_default = names == _DEFAULT_EMBED
    if not custom:
        p.add(
            HookRow(
                "embedding layer names",
                "loaders/utils.py",
                "custom_embed_names",
                PRESENT if not is_default else NOT_EXPECTED,
                f"-> {names}"
                if not is_default
                else "standard embed_tokens/lm_head names",
            )
        )
    elif not is_default:
        p.add(
            HookRow(
                "embedding layer names",
                "loaders/utils.py",
                "custom_embed_names",
                PRESENT,
                f"custom names mapped: {names}",
            )
        )
    else:
        p.add(
            HookRow(
                "embedding layer names",
                "loaders/utils.py",
                "custom_embed_names",
                GENERIC,
                "non-standard embed/lm_head expected but falls back to default "
                f"{_DEFAULT_EMBED} — embeddings_to_save / lora targets may be wrong",
            )
        )


def _check_model_py(p: Probe) -> None:
    types, ok = _file_model_types(p.src("loaders/model.py"))
    if not ok:
        p.warn("loaders/model.py model-type branches unreadable")
        return
    # quantization skip-modules / quant_storage (jamba/falcon_h1 special blocks)
    wants_quant = bool(p.opt.get("quantization"))
    special_quant = {"jamba", "falcon_h1"}
    gated = "quantization + special block"
    if wants_quant and p.mct in special_quant:
        p.add(
            HookRow(
                "quant skip-modules / quant_storage",
                "loaders/model.py",
                gated,
                PRESENT if p.mct in types else MISSING,
                "quant_storage / skip-block handling present"
                if p.mct in types
                else "special-quant type but no branch in model.py",
            )
        )
    else:
        p.add(
            HookRow(
                "quant skip-modules / quant_storage",
                "loaders/model.py",
                gated,
                NOT_EXPECTED,
                "no special quant handling required",
            )
        )
    # _check_model_requirements extra deps (e.g. causal_conv1d for lfm2/lfm2-vl)
    needs_dep = bool(p.ctx.features.extra.get("causal_conv1d_conflict")) or p.mct in {
        "lfm2",
        "lfm2-vl",
    }
    g2 = "extra runtime dep"
    if needs_dep:
        p.add(
            HookRow(
                "_check_model_requirements deps",
                "loaders/model.py",
                g2,
                PRESENT if p.mct in types else MISSING,
                "dep check present"
                if p.mct in types
                else "needs extra dep but no _check_model_requirements branch",
            )
        )
    else:
        p.add(
            HookRow(
                "_check_model_requirements deps",
                "loaders/model.py",
                g2,
                NOT_EXPECTED,
                "no extra runtime dependency",
            )
        )


def _check_trainer(p: Probe) -> None:
    base_types, ok_b = _file_model_types(p.src("core/trainers/base.py"))
    tr_types, ok_t = _file_model_types(p.src("utils/trainer.py"))
    if not ok_b:
        p.warn("core/trainers/base.py unreadable")
    if not ok_t:
        p.warn("utils/trainer.py unreadable")
    # both are informational: a type that needs no special trainer handling is fine.
    p.add(
        HookRow(
            "trainer special handling",
            "core/trainers/base.py",
            "trainer branch (advisory)",
            PRESENT if (ok_b and p.mct in base_types) else NOT_EXPECTED,
            "has a model-type branch in base trainer"
            if (ok_b and p.mct in base_types)
            else "no special base-trainer handling",
        )
    )
    p.add(
        HookRow(
            "trainer flags (drop_attn_mask/skip_estimates)",
            "utils/trainer.py",
            "trainer flag (advisory)",
            PRESENT if (ok_t and p.mct in tr_types) else NOT_EXPECTED,
            "type referenced by a trainer flag"
            if (ok_t and p.mct in tr_types)
            else "no special trainer flags",
        )
    )


def _check_densemixer(p: Probe) -> None:
    types, ok = _file_model_types(p.src("integrations/densemixer/plugin.py"))
    supported = ok and p.mct in types
    wants = bool(p.opt.get("densemixer"))
    gated = "densemixer + moe" if wants else "densemixer (advisory)"
    if p.mct in {"olmoe", "qwen2_moe", "qwen3_moe"}:
        p.add(
            HookRow(
                "densemixer handling",
                "integrations/densemixer/plugin.py",
                gated,
                PRESENT if supported else MISSING,
                "dedicated densemixer patch"
                if supported
                else "densemixer-eligible type but no patch branch",
            )
        )
    else:
        p.add(
            HookRow(
                "densemixer handling",
                "integrations/densemixer/plugin.py",
                gated,
                NOT_EXPECTED,
                "type not densemixer-eligible",
            )
        )


def _check_custom_kernel_module(p: Probe) -> None:
    mod_dir = p.src("monkeypatch/models") / p.mct
    if not mod_dir.is_dir():
        p.add(
            HookRow(
                "custom kernel module",
                "monkeypatch/models/",
                "has custom module dir",
                NOT_EXPECTED,
                "no monkeypatch/models/<type>/ module",
            )
        )
        return
    modpath = f"axolotl.monkeypatch.models.{p.mct}"
    try:
        importlib.import_module(modpath)
        p.add(
            HookRow(
                "custom kernel module",
                f"monkeypatch/models/{p.mct}/",
                "has custom module dir",
                PRESENT,
                "module imports and compiles",
            )
        )
    except Exception as exc:  # noqa: BLE001
        p.warn(
            f"monkeypatch/models/{p.mct} import failed ({exc.__class__.__name__})",
            critical=True,
        )
        p.add(
            HookRow(
                "custom kernel module",
                f"monkeypatch/models/{p.mct}/",
                "has custom module dir",
                MISSING,
                f"module present but import failed: {exc.__class__.__name__}: {exc}",
            )
        )


def _check_tiled_mlp(p: Probe) -> None:
    tdir = p.src("monkeypatch/tiled_mlp")
    # advisory unless tiled_mlp is requested: an absent generic package isn't a gap
    wants = bool(p.opt.get("tiled_mlp"))
    if tdir.is_dir():
        status, note = PRESENT, "generic tiled-MLP patch available (type-agnostic)"
    elif wants:
        status, note = MISSING, "tiled_mlp requested but package missing"
    else:
        status, note = NOT_EXPECTED, "tiled_mlp package missing (not requested)"
    p.add(
        HookRow(
            "tiled-MLP",
            "monkeypatch/tiled_mlp/",
            "tiled_mlp" if wants else "tiled_mlp (advisory)",
            status,
            note,
        )
    )


def _check_example(p: Probe) -> None:
    ex = p.repo / "examples"
    norm = lambda s: s.lower().replace("-", "").replace("_", "")  # noqa: E731
    mct_n = norm(p.mct)
    matched: str | None = None
    if ex.is_dir() and mct_n:
        for d in ex.iterdir():
            if not d.is_dir():
                continue
            dn = norm(d.name)
            if not dn:
                continue
            # require a real segment-prefix overlap (>=3 chars), not a loose substring, so the family must actually match the dir name
            short, long = (dn, mct_n) if len(dn) <= len(mct_n) else (mct_n, dn)
            if len(short) >= 3 and long.startswith(short):
                matched = d.name
                break
    p.add(
        HookRow(
            "example config",
            "examples/",
            "recommended (advisory)",
            PRESENT if matched else NOT_EXPECTED,
            f"examples/{matched}/ matches this type/family"
            if matched
            else "no examples/<model>/ for this type/family (advisory; see G8 for "
            "test coverage)",
        )
    )


def run(ctx: GateContext) -> GateResult:
    p = Probe(ctx)

    _check_model_loads(p)
    _check_chat_template(p)
    _check_multipack(p)
    pm_types, pm_ok = _check_patch_manager(p)
    _check_fused_attn(p)
    _check_liger(p)
    _check_cce(p)
    _check_moe(p)
    _check_multimodal(p)
    _check_ssm_hybrid(p, pm_types, pm_ok)
    _check_embed_names(p)
    _check_model_py(p)
    _check_trainer(p)
    _check_densemixer(p)
    _check_custom_kernel_module(p)
    _check_tiled_mlp(p)
    _check_example(p)

    rows = p.rows
    missing = [r for r in rows if r.status == MISSING]
    generic = [r for r in rows if r.status == GENERIC]
    explicit = [r for r in rows if r.status == PRESENT]
    not_exp = [r for r in rows if r.status == NOT_EXPECTED]
    expected = len(rows) - len(not_exp)

    data: dict[str, Any] = {
        "model_config_type": p.mct,
        "checklist": [r.as_dict() for r in rows],
        "counts": {
            "expected": expected,
            "present_explicit": len(explicit),
            "generic_fallback": len(generic),
            "missing": len(missing),
            "not_expected": len(not_exp),
        },
        "reliability_warnings": list(p.warnings),
    }

    details: list[str] = [
        f"{_ICON[r.status]} {r.hook} [{r.gated_on}] — {r.status}"
        + (f": {r.note}" if r.note else "")
        for r in rows
    ]
    if p.warnings:
        details.append("")
        details.append("reliability warnings (dispatch shape drifted / unreadable):")
        details.extend(f"  ! {w}" for w in p.warnings)

    gen_note = ""
    if generic:
        gen_note = " (" + ", ".join(sorted(r.hook for r in generic)) + ")"
    miss_note = ""
    if missing:
        miss_note = " (" + ", ".join(sorted(r.hook for r in missing)) + ")"
    summary = (
        f"{expected} expected hooks: {len(explicit)} explicit, "
        f"{len(generic)} generic-fallback{gen_note}, {len(missing)} missing{miss_note}"
    )

    if p.critical_unreadable:
        status = GateStatus.COULD_NOT_RUN
        summary = "load-bearing dispatch could not be read; " + summary
    elif missing or generic:
        status = GateStatus.FINDINGS
    else:
        status = GateStatus.PASS

    return GateResult(
        GATE_ID, GATE_NAME, status, summary=summary, details=details, data=data
    )
