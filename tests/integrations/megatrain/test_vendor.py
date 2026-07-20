"""Import and namespace integrity tests for the vendored MegaTrain package."""

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

VENDOR_PACKAGE = "axolotl.integrations.megatrain._vendor.infinity"


def _top_level_infinity_modules():
    return {
        name
        for name in sys.modules
        if name == "infinity" or name.startswith("infinity.")
    }


def test_all_vendored_modules_import_without_top_level_infinity():
    before = _top_level_infinity_modules()
    package = importlib.import_module(VENDOR_PACKAGE)
    modules = [
        module.name
        for module in pkgutil.walk_packages(
            package.__path__, prefix=f"{VENDOR_PACKAGE}."
        )
    ]

    assert modules
    for module in modules:
        importlib.import_module(module)

    assert _top_level_infinity_modules() == before


def test_vendored_sources_have_no_absolute_infinity_imports():
    package = importlib.import_module(VENDOR_PACKAGE)
    package_dir = Path(package.__file__).parent
    absolute_imports = []

    for source in package_dir.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "infinity" or name.startswith("infinity."):
                    absolute_imports.append(
                        f"{source.relative_to(package_dir)}: {name}"
                    )

    assert absolute_imports == []
