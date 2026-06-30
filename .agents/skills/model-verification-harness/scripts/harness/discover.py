"""Static PR model auto-discovery: parse a unified git diff and surface candidate new ``model_config_type``(s), their canonical ``base_model``, which registration hooks the PR wired, and reliability warnings for dynamic/external registration the diff can't statically resolve."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except Exception:  # noqa: BLE001 - yaml is a dep, but degrade to regex if absent
    yaml = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# diff model
# --------------------------------------------------------------------------- #
@dataclass
class FileDiff:
    path: str = ""  # post-image (b-side) path
    is_new: bool = False
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)

    @property
    def added_text(self) -> str:
        return "\n".join(self.added)

    @property
    def pre_image(self) -> list[str]:
        # context + removed == the lines that existed before this PR
        return self.context + self.removed


def _strip_ab(path: str) -> str:
    path = path.strip()
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def parse_diff(diff_text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            cur = FileDiff()
            files.append(cur)
            # b-side path as a fallback; +++ refines it below
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                cur.path = parts[1].strip()
            continue
        if cur is None:
            continue
        if line.startswith("new file mode"):
            cur.is_new = True
        elif line.startswith("--- "):
            if line[4:].strip() == "/dev/null":
                cur.is_new = True
        elif line.startswith("+++ "):
            tgt = line[4:].strip()
            if tgt != "/dev/null":
                cur.path = _strip_ab(tgt)
        elif line.startswith("@@") or line.startswith("\\"):
            continue
        elif line.startswith("+"):
            cur.added.append(line[1:])
        elif line.startswith("-"):
            cur.removed.append(line[1:])
        elif line.startswith(" "):
            cur.context.append(line[1:])
    return files


# --------------------------------------------------------------------------- #
# signal taxonomy
# --------------------------------------------------------------------------- #
# point-class -> signal-ids, for the "≥N independent classes" new-type heuristic (§4/§6)
_CLASSES = {
    "model_dir": {"model_dir"},
    "example": {"example"},
    "container": {
        "multipack",
        "moe_arch_block",
        "experts_only_block",
        "fused_attn_set",
        "ssm_hybrid_set",
        "mm_automodel",
    },
    "template": {"chat_template_enum", "jinja_template"},
    "dispatch": {
        "pm_dispatch",
        "liger",
        "cce_routing",
        "densemixer",
        "embed_names",
        "model_py",
        "lora_kernel_attn",
        "config_postprocess",
        "builder",
        "trainer",
        "tokenizer",
        "chat_strategy",
        "trl_schema",
        "processing_strategy",
    },
    "kernels": {"kernels_adapter", "expert_kernel_libs", "liger_model_file"},
}
_SIGNAL_CLASS = {sig: cls for cls, sigs in _CLASSES.items() for sig in sigs}

_WEIGHTS = {
    "model_dir": 5,
    "kernels_adapter": 3,
    "example": 3,
    "multipack": 2,
    "moe_arch_block": 2,
    "experts_only_block": 2,
    "fused_attn_set": 2,
    "ssm_hybrid_set": 2,
    "mm_automodel": 2,
    "chat_template_enum": 1,
    "jinja_template": 1,
    "pm_dispatch": 1,
    "liger": 1,
    "cce_routing": 1,
    "densemixer": 1,
    "embed_names": 1,
    "model_py": 1,
    "lora_kernel_attn": 1,
    "config_postprocess": 1,
    "builder": 1,
    "trainer": 1,
    "tokenizer": 1,
    "chat_strategy": 1,
    "trl_schema": 1,
    "processing_strategy": 1,
    "expert_kernel_libs": 1,
    "liger_model_file": 1,
}

# named-container files -> (signal-id, member shape)
_CONTAINERS = {
    "src/axolotl/monkeypatch/multipack.py": ("multipack", "member"),
    "src/axolotl/common/architectures.py": ("moe_arch_block", "dictkey"),
    "src/axolotl/integrations/kernels/constants.py": ("experts_only_block", "dictkey"),
    "src/axolotl/loaders/patch_manager.py": ("fused_attn_set", "member"),
    "src/axolotl/utils/schemas/validation.py": ("ssm_hybrid_set", "member"),
    "src/axolotl/loaders/constants.py": ("mm_automodel", "subscript"),
    "src/axolotl/utils/schemas/enums.py": ("chat_template_enum", "enum"),
}

# files whose type-compare branches (==/in/startswith on model_(config_)type) name a type
_DISPATCH_FILES = {
    "src/axolotl/loaders/patch_manager.py": "pm_dispatch",
    "src/axolotl/integrations/liger/plugin.py": "liger",
    "src/axolotl/integrations/liger/__init__.py": "liger",
    "src/axolotl/integrations/cut_cross_entropy/__init__.py": "cce_routing",
    "src/axolotl/integrations/densemixer/plugin.py": "densemixer",
    "src/axolotl/loaders/utils.py": "embed_names",
    "src/axolotl/loaders/model.py": "model_py",
    "src/axolotl/monkeypatch/lora_kernels.py": "lora_kernel_attn",
    "src/axolotl/utils/config/__init__.py": "config_postprocess",
    "src/axolotl/core/builders/base.py": "builder",
    "src/axolotl/core/builders/causal.py": "builder",
    "src/axolotl/core/trainers/base.py": "trainer",
    "src/axolotl/utils/trainer.py": "trainer",
    "src/axolotl/loaders/tokenizer.py": "tokenizer",
    "src/axolotl/prompt_strategies/chat_template.py": "chat_strategy",
    "src/axolotl/utils/schemas/trl.py": "trl_schema",
}

_FAMILY_SUFFIXES = ("_moe_text", "_unified_text", "_text", "_moe", "_unified", "_vl")

# tokens that are never a model_config_type but match the literal shapes
_STOPWORDS = {"true", "false", "none", "auto", "default", "model_type", "self", "cfg"}

_TOKEN = r"[A-Za-z][\w.\-]*"  # nosec B105  # noqa: S105 - regex fragment
_RE_MEMBER = re.compile(rf"^['\"]({_TOKEN})['\"]\s*,?\s*$")
_RE_DICTKEY = re.compile(rf"^['\"]({_TOKEN})['\"]\s*:")
_RE_ENUM = re.compile(rf"^[A-Za-z_]\w*\s*=\s*['\"]({_TOKEN})['\"]")
_RE_SUBSCRIPT = re.compile(
    rf"MULTIMODAL_AUTO_MODEL_MAPPING\[\s*['\"]({_TOKEN})['\"]\s*\]"
)
_RE_QUOTED = re.compile(rf"['\"]({_TOKEN})['\"]")
_RE_YAML_BASE = re.compile(r"^\s*base_model:\s*(.+?)\s*(?:#.*)?$")
_RE_YAML_CHAT = re.compile(r"^\s*chat_template:\s*(.+?)\s*(?:#.*)?$")


# --------------------------------------------------------------------------- #
# candidate accumulation
# --------------------------------------------------------------------------- #
def _normalize(name: str) -> str:
    # cluster key: lowercase, separators -> '_' (qwen3.5 -> qwen3_5, falcon-h1 -> falcon_h1)
    return re.sub(r"[.\-/ ]+", "_", name.strip().lower()).strip("_")


def _family_root(key: str) -> str:
    for suf in _FAMILY_SUFFIXES:
        if key.endswith(suf) and len(key) > len(suf):
            return key[: -len(suf)]
    return key


@dataclass
class _Cand:
    key: str
    display: str = ""
    base_model: str | None = None
    signals: set[str] = field(default_factory=set)
    preimage: bool = False
    new_file: bool = False  # backed by a new model/example dir or adapter file


class _Bag:
    def __init__(self) -> None:
        self.cands: dict[str, _Cand] = {}

    def add(
        self,
        literal: str,
        signal: str,
        *,
        authoritative: bool,
        base_model: str | None = None,
        new_file: bool = False,
    ) -> _Cand | None:
        key = _normalize(literal)
        if not key or key in _STOPWORDS or len(key) < 2:
            return None
        c = self.cands.setdefault(key, _Cand(key=key))
        c.signals.add(signal)
        if authoritative and (not c.display):
            c.display = literal
        if base_model and not c.base_model:
            c.base_model = base_model
        if new_file:
            c.new_file = True
        return c

    def mark_preimage(self, literal: str) -> None:
        key = _normalize(literal)
        c = self.cands.get(key)
        if c is not None:
            c.preimage = True


# --------------------------------------------------------------------------- #
# extractors
# --------------------------------------------------------------------------- #
def _container_literals(lines: list[str], shape: str) -> list[str]:
    out: list[str] = []
    for raw in lines:
        s = raw.strip()
        if shape == "member":
            m = _RE_MEMBER.match(s)
        elif shape == "dictkey":
            m = _RE_DICTKEY.match(s)
        elif shape == "enum":
            m = _RE_ENUM.match(s)
        elif shape == "subscript":
            m = _RE_SUBSCRIPT.search(s)
        else:
            m = None
        if m:
            out.append(m.group(1))
    return out


def _compare_literals(lines: list[str]) -> list[str]:
    # literals on a line that compares a model-type ref (==/!=/in/startswith)
    out: list[str] = []
    for raw in lines:
        if "model_type" not in raw and "model_config_type" not in raw:
            continue
        if not re.search(r"==|!=|\bin\b|startswith", raw):
            continue
        for tok in _RE_QUOTED.findall(raw):
            if tok.lower() not in _STOPWORDS:
                out.append(tok)
    return out


def _adapter_literals(fd: FileDiff) -> list[str]:
    # AST the new adapter file: `name = "x"` on a *Adapter class, plus type compares
    out: list[str] = []
    try:
        tree = ast.parse(fd.added_text)
    except SyntaxError:
        return _compare_literals(fd.added)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.endswith("Adapter"):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if (
                            isinstance(t, ast.Name)
                            and t.id == "name"
                            and isinstance(stmt.value, ast.Constant)
                            and isinstance(stmt.value.value, str)
                        ):
                            out.append(stmt.value.value)
    out.extend(_compare_literals(fd.added))
    return out


def _yaml_base(fd: FileDiff) -> tuple[str | None, str | None]:
    base = chat = None
    if yaml is not None:
        try:
            data = yaml.safe_load(fd.added_text)
            if isinstance(data, dict):
                base = data.get("base_model")
                chat = data.get("chat_template")
        except Exception:  # noqa: BLE001 - partial/invalid YAML -> regex fallback
            data = None
    for raw in fd.added:
        if base is None:
            m = _RE_YAML_BASE.match(raw)
            if m:
                base = m.group(1).strip().strip("'\"")
        if chat is None:
            m = _RE_YAML_CHAT.match(raw)
            if m:
                chat = m.group(1).strip().strip("'\"")
    base = str(base).strip().strip("'\"") if base else None
    chat = str(chat).strip().strip("'\"") if chat else None
    return base or None, chat or None


def _path_segment(path: str, prefix: str) -> str | None:
    # first segment after `prefix` (which ends in '/')
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix) :]
    return rest.split("/", 1)[0] if rest else None


# --------------------------------------------------------------------------- #
# reliability warnings (dynamic / external registration — §5/§7)
# --------------------------------------------------------------------------- #
def _spec(pkg: str, lines: list[str]) -> str:
    for raw in lines:
        m = re.search(rf'["\']({re.escape(pkg)}[^"\']*)["\']', raw)
        if m:
            return m.group(1)
    return ""


def _reliability_warnings(files: list[FileDiff]) -> list[str]:
    w: list[str] = []
    by_path = {f.path: f for f in files}

    if "scripts/cutcrossentropy_install.py" in by_path:
        w.append(
            "reliability: scripts/cutcrossentropy_install.py changed — CCE model "
            "support lands as an external commit-pin bump; the supported type is in "
            "the cut_cross_entropy package, not a statically extractable literal."
        )
    for req in ("pyproject.toml", "requirements.txt", "setup.py"):
        fd = by_path.get(req)
        if fd and any("transformers" in a for a in fd.added):
            old = _spec("transformers", fd.removed)
            new = _spec("transformers", fd.added)
            w.append(
                f"reliability: transformers pin bumped in {req} ({old or '?'} -> "
                f"{new or '?'}); a brand-new arch may only resolve via AutoConfig "
                "under the newer transformers — verify model_type after install."
            )
            break
    proc = by_path.get("src/axolotl/processing_strategies.py")
    if proc and any("isinstance(" in a for a in proc.added):
        w.append(
            "reliability: processing_strategies.py routes some multimodal models by "
            "isinstance(processor, ...) with no type literal — a processor-class "
            "route can't be matched by string; corroborate via example/base_model."
        )
    const = by_path.get("src/axolotl/loaders/constants.py")
    if const and any("dict(" in a and "MAPPING" in a for a in const.added):
        w.append(
            "reliability: MULTIMODAL_AUTO_MODEL_MAPPING is seeded from a transformers "
            "dict(...) — multimodal types may register without a subscript literal."
        )
    mp = by_path.get("src/axolotl/monkeypatch/multipack.py")
    if mp and any("remote_code" in (a + " ") for a in mp.added + mp.context):
        w.append(
            "reliability: multipack has a remote-code branch — remote-code models "
            "multipack without being listed in SUPPORTED_MULTIPACK_MODEL_TYPES."
        )
    for lf in (
        "src/axolotl/integrations/liger/plugin.py",
        "src/axolotl/integrations/liger/__init__.py",
    ):
        fd = by_path.get(lf)
        if fd and not _compare_literals(fd.added):
            w.append(
                f"reliability: {lf} changed but no model-type literal extracted — the "
                "native liger MODEL_TYPE_TO_APPLY_LIGER_FN table may cover the type."
            )
            break
    if any(f.path.startswith("src/axolotl/utils/mistral/") for f in files):
        w.append(
            "reliability: utils/mistral/* changed — mistral-common tokenizer routing "
            "is class-based (no type literal); confirm tokenizer support separately."
        )
    return w


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def discover_from_diff(diff_text: str, repo_root: Path | None = None) -> dict:
    files = parse_diff(diff_text)
    bag = _Bag()
    # corroborating-only stems (prefix-attached to primary candidates, never created)
    corroborators: list[tuple[str, str]] = []  # (signal-id, stem)
    preimage: list[str] = []

    for fd in files:
        path = fd.path

        # --- new dirs / files ------------------------------------------------
        if fd.is_new:
            seg = _path_segment(path, "src/axolotl/monkeypatch/models/")
            if seg:
                bag.add(seg, "model_dir", authoritative=False, new_file=True)
            seg = _path_segment(path, "examples/")
            if seg and seg not in {"", "README.md"}:
                base = None
                if path.endswith((".yml", ".yaml")):
                    base, _chat = _yaml_base(fd)
                bag.add(
                    seg, "example", authoritative=False, base_model=base, new_file=True
                )
            if path.startswith(
                "src/axolotl/utils/chat_templates/templates/"
            ) and path.endswith(".jinja"):
                bag.add(Path(path).stem, "jinja_template", authoritative=False)
            seg = _path_segment(path, "src/axolotl/integrations/kernels/libs/")
            if seg:
                corroborators.append(("expert_kernel_libs", seg))
            seg = _path_segment(path, "src/axolotl/integrations/liger/models/")
            if seg and path.endswith(".py") and not path.endswith("__init__.py"):
                corroborators.append(("liger_model_file", Path(path).stem))

        # adapter new file: AST out `name=` + type compares (stem != type, §7)
        if (
            fd.is_new
            and path.startswith("src/axolotl/integrations/kernels/adapters/")
            and path.endswith(".py")
            and not path.endswith("__init__.py")
        ):
            for lit in _adapter_literals(fd):
                bag.add(lit, "kernels_adapter", authoritative=True, new_file=True)

        # --- named containers (added members + pre-image members) ------------
        if path in _CONTAINERS:
            signal, shape = _CONTAINERS[path]
            for lit in _container_literals(fd.added, shape):
                auth = signal != "chat_template_enum"
                bag.add(lit, signal, authoritative=auth)
            preimage.extend(_container_literals(fd.pre_image, shape))

        # --- type-compare dispatch files ------------------------------------
        if path in _DISPATCH_FILES:
            signal = _DISPATCH_FILES[path]
            for lit in _compare_literals(fd.added):
                bag.add(lit, signal, authoritative=True)
            preimage.extend(_compare_literals(fd.pre_image))

    # mark extensions: any candidate that pre-existed in a container/dispatch
    for lit in preimage:
        bag.mark_preimage(lit)

    # attach corroborating signals to existing same-family candidates
    unmatched: list[tuple[str, str]] = []
    for signal, stem in corroborators:
        key = _normalize(stem)
        seg = key.split("_", 1)[0]
        hit = False
        for c in bag.cands.values():
            if c.key.split("_", 1)[0] == seg:
                c.signals.add(signal)
                hit = True
        if not hit:
            unmatched.append((signal, stem))

    warnings = _reliability_warnings(files)
    for signal, stem in unmatched:
        warnings.append(
            f"reliability: new {signal} module '{stem}' matched no candidate family "
            "— a model may be registered only via this perf module."
        )

    candidates = _rank(bag)
    if not candidates and not warnings:
        warnings.append("no model registration signals found in this diff.")
    return {"candidates": candidates, "warnings": warnings}


def _rank(bag: _Bag) -> list[dict]:
    out: list[dict] = []
    for c in bag.cands.values():
        classes = {_SIGNAL_CLASS[s] for s in c.signals if s in _SIGNAL_CLASS}
        score = sum(_WEIGHTS.get(s, 0) for s in c.signals)
        if c.base_model:
            score += 1
        if len(classes) >= 3:
            score += 2

        strong_new = c.new_file
        if c.preimage and not strong_new:
            is_new = False
        elif strong_new or len(classes) >= 2:
            is_new = True
        else:
            is_new = False

        out.append(
            {
                "model_config_type": c.display or c.key,
                "base_model": c.base_model,
                "signals": sorted(c.signals),
                "score": score,
                "is_new": is_new,
                "family_root": _family_root(c.key),
            }
        )
    out.sort(key=lambda d: (-d["score"], d["model_config_type"]))
    return out
