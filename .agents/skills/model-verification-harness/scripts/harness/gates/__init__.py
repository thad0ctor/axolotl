"""Gate registry.

Each gate lives in its own module (``gN_<name>.py``) and exposes the
:class:`~harness.Gate` surface. ``load_gates`` imports the ones that exist, in
ladder order, so a partially-built harness still runs the gates it has and
reports the rest as unavailable rather than crashing.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from .. import GATE_ORDER

# gate id -> module name under this package
_GATE_MODULES: dict[str, str] = {
    "G1": "g1_config",
    "G2": "g2_integration",
    "G3": "g3_preprocess",
    "G4": "g4_masking",
    "G5": "g5_packing",
    "G6": "g6_loss",
    "G7": "g7_consistency",
    "G8": "g8_coverage",
}


def load_gates() -> tuple[list[ModuleType], list[str]]:
    """Return ``(modules, errors)`` — gate modules in ladder order plus a list of
    human-readable import failures for gates that are missing or broken."""
    modules: list[ModuleType] = []
    errors: list[str] = []
    for gate_id in GATE_ORDER:
        mod_name = _GATE_MODULES.get(gate_id)
        if not mod_name:
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{mod_name}")
        except Exception as exc:  # noqa: BLE001 - a broken gate must not kill the run
            errors.append(f"{gate_id} ({mod_name}): {exc.__class__.__name__}: {exc}")
            continue
        if not (hasattr(mod, "GATE_ID") and hasattr(mod, "run")):
            errors.append(f"{gate_id} ({mod_name}): does not satisfy the Gate contract")
            continue
        modules.append(mod)
    return modules, errors
