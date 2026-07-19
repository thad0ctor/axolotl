"""Verify that the plugins declared in a config are already installed.

A ``plugins:`` entry may be a dotted class path or a mapping pointing at an
externally installed plugin::

    plugins:
      - axolotl.integrations.liger.LigerPlugin          # dotted path, unchanged
      - cls: my_plugin.MyPlugin                          # external plugin
        source: https://github.com/org/repo.git          # provenance, not an install
        ref: 4f2a9c1
        subdir: src

Verification is deliberately inert: it reads the plugin manifest, extends
``sys.path`` with the entries recorded there, and checks that each class is
importable. Nothing is fetched, built, or written -- installing is an explicit
``axolotl plugins install`` step. ``verify_plugins`` rewrites ``cfg["plugins"]``
in place to a flat list of dotted class paths, so every downstream consumer keeps
seeing ``list[str]``.
"""

from __future__ import annotations

import importlib
import importlib.util

from axolotl.integrations.plugin_manifest import apply_syspath_entries, find_entries
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class PluginNotInstalledError(RuntimeError):
    """A plugin declared in the config is not importable in this environment."""


def _install_command(spec) -> str:
    parts = ["axolotl", "plugins", "install", spec.source]
    if spec.ref:
        parts += ["--ref", spec.ref]
    if spec.subdir:
        parts += ["--subdir", spec.subdir]
    if isinstance(spec.cls, str):
        parts += ["--cls", spec.cls]
    return " ".join(parts)


def _not_importable(cls_path: str, spec) -> PluginNotInstalledError:
    if not spec.source:
        return PluginNotInstalledError(
            f"Plugin {cls_path!r} is not importable. Install the package that "
            "provides it, or add a `source:` to the plugin entry and install it "
            "with `axolotl plugins install <source>`."
        )
    return PluginNotInstalledError(
        f"Plugin {cls_path!r} is not installed.\n"
        f"To install it, run:\n  {_install_command(spec)}"
    )


def _unresolved(spec) -> PluginNotInstalledError:
    return PluginNotInstalledError(
        f"No installed plugin is recorded for source {spec.source!r}.\n"
        f"To install it, run:\n  {_install_command(spec)}"
    )


def _check_importable(cls_path: str, spec) -> None:
    """Confirm the class really loads, so a bad entry fails here and not mid-training.

    ``PluginManager.register`` swallows ``ImportError`` and carries on without the
    plugin, and surfaces a typo'd class name as a bare ``AttributeError``.
    """
    module_name, _, class_name = cls_path.rpartition(".")
    if not module_name:
        raise _not_importable(cls_path, spec)

    try:
        if importlib.util.find_spec(module_name) is None:
            raise _not_importable(cls_path, spec)
    except (ImportError, ValueError):
        raise _not_importable(cls_path, spec) from None

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        message = (
            f"Plugin {cls_path!r} is present but cannot be imported: {exc}. Its "
            "dependencies are not importable in this environment."
        )
        if spec.source:
            message += f"\nTo reinstall it, run:\n  {_install_command(spec)}"
        raise PluginNotInstalledError(message) from exc

    if not hasattr(module, class_name):
        raise PluginNotInstalledError(
            f"Plugin class {cls_path!r} was not found: module {module_name!r} has no "
            f"attribute {class_name!r}. Check the `cls` in your config."
        )


def _spec_cls_paths(spec) -> list[str]:
    if spec.cls:
        return [spec.cls] if isinstance(spec.cls, str) else list(spec.cls)
    # Only the manifest can name the class; discovery needs the source tree.
    for entry in find_entries(spec):
        if entry.get("cls"):
            return list(entry["cls"])
    raise _unresolved(spec)


def verify_plugins(cfg) -> None:
    """Normalize ``cfg['plugins']`` to ``list[str]``, checking external plugins exist.

    Idempotent: a no-op when every entry is already a dotted path string.
    """
    plugins = cfg.get("plugins") if cfg else None
    if not plugins or all(isinstance(p, str) for p in plugins):
        return

    from axolotl.utils.schemas.config import PluginSpec

    parsed = [
        entry if isinstance(entry, (str, PluginSpec)) else PluginSpec(**dict(entry))
        for entry in plugins
    ]
    apply_syspath_entries([entry for entry in parsed if not isinstance(entry, str)])

    normalized: list[str] = []
    for entry in parsed:
        if isinstance(entry, str):
            normalized.append(entry)
            continue
        for cls_path in _spec_cls_paths(entry):
            _check_importable(cls_path, entry)
            normalized.append(cls_path)
    cfg["plugins"] = normalized
