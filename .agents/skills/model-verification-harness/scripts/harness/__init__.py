"""Shared contracts (GateContext / GateResult / GateStatus) coupling the gates to the orchestrator."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


class GateStatus(enum.Enum):
    """Per-gate outcome; aggregates into the exit code."""

    PASS = "pass"  # nosec B105
    FINDINGS = "findings"
    COULD_NOT_RUN = "could_not_run"
    SKIPPED = "skipped"

    @property
    def icon(self) -> str:
        return {
            GateStatus.PASS: "✅",
            GateStatus.FINDINGS: "❌",
            GateStatus.COULD_NOT_RUN: "⚠️",
            GateStatus.SKIPPED: "·",
        }[self]


class Verdict(enum.Enum):
    """Four-way verdict for a compat-matrix cell. WARNED_NO_OP is the silent-shadowing class and must never render as a green check."""

    REJECTED = "rejected"
    NORMALIZED = "normalized"
    WARNED_NO_OP = "warned_no_op"
    SUPPORTED = "supported"

    @property
    def icon(self) -> str:
        return {
            Verdict.REJECTED: "⛔",
            Verdict.NORMALIZED: "✏️",
            Verdict.WARNED_NO_OP: "⚠️",
            Verdict.SUPPORTED: "✅",
        }[self]


@dataclass
class ModelFeatures:
    """Architecture facts driving gate applicability; produced by harness.detect, read-only."""

    model_config_type: str
    base_model: str  # path or HF id
    is_moe: bool = False
    is_multimodal: bool = False
    is_ssm_hybrid: bool = False
    needs_patch: bool = False  # has a patch_manager dispatch branch
    custom_embed_names: bool = False  # non-standard embed/lm_head module names
    architectures: list[str] = field(default_factory=list)
    config_json: dict[str, Any] = field(default_factory=dict)
    # free-form extras for gates without widening the contract
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateContext:
    """Built once and passed read-only. Gates that drive preprocess/train (G3..G7) must write only under ``output_dir``."""

    features: ModelFeatures
    repo_root: Path  # checkout the static gates introspect
    output_dir: Path  # isolated scratch
    profile: str = "smoke"  # smoke | full | multigpu
    seed: int = 42
    gpu_available: bool = False
    selected_gates: set[str] = field(default_factory=set)
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def model_config_type(self) -> str:
        return self.features.model_config_type


@dataclass
class GateResult:
    """Uniform gate output; ``data`` carries structured rows, ``details`` human lines."""

    gate_id: str
    name: str
    status: GateStatus
    summary: str = ""
    details: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def skipped(cls, gate_id: str, name: str, why: str) -> "GateResult":
        return cls(gate_id, name, GateStatus.SKIPPED, summary=why)

    @classmethod
    def could_not_run(cls, gate_id: str, name: str, why: str) -> "GateResult":
        return cls(gate_id, name, GateStatus.COULD_NOT_RUN, summary=why)


@runtime_checkable
class Gate(Protocol):
    """Structural contract each ``harness.gates.gN_*`` module satisfies."""

    GATE_ID: str
    GATE_NAME: str

    def applies(self, ctx: GateContext) -> bool: ...
    def run(self, ctx: GateContext) -> GateResult: ...


def exit_code(results: list[GateResult]) -> int:
    """Aggregate the exit contract: COULD_NOT_RUN (2) dominates FINDINGS (1); SKIPPED never counts."""
    statuses = {r.status for r in results}
    if GateStatus.COULD_NOT_RUN in statuses:
        return 2
    if GateStatus.FINDINGS in statuses:
        return 1
    return 0


GateModule = Any
GATE_ORDER: list[str] = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]

__all__ = [
    "GateStatus",
    "Verdict",
    "ModelFeatures",
    "GateContext",
    "GateResult",
    "Gate",
    "exit_code",
    "GATE_ORDER",
    "GateModule",
    "Callable",
]
