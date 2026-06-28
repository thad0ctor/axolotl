"""Shared contracts for the model-verification harness.

Every gate (G1..G8) is a small module under ``harness.gates`` that exposes a
uniform surface (``GATE_ID``, ``GATE_NAME``, ``applies(ctx)``, ``run(ctx)``) and
returns a :class:`GateResult`. The orchestrator (`verify_model.py`) builds one
:class:`GateContext`, runs the selected+applicable gates in ladder order, renders
a report, and maps the aggregate to the liger-style exit contract:

    0 = all selected gates clean
    1 = findings need review
    2 = could not run reliably (model/env unloadable, pipeline import error)

These types are the *only* coupling between gates, so a gate can be authored and
tested in isolation against this contract.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


class GateStatus(enum.Enum):
    """Per-gate outcome. Maps to the aggregate exit code (see :func:`exit_code`)."""

    PASS = "pass"  # nosec B105 - enum value, not a secret  # gate ran, nothing to flag
    FINDINGS = "findings"  # gate ran, surfaced issues a human must review
    COULD_NOT_RUN = "could_not_run"  # genuine inability (model won't load, etc.)
    SKIPPED = "skipped"  # not applicable to this model, or not selected

    @property
    def icon(self) -> str:
        return {
            GateStatus.PASS: "✅",
            GateStatus.FINDINGS: "❌",
            GateStatus.COULD_NOT_RUN: "⚠️",
            GateStatus.SKIPPED: "·",
        }[self]


class Verdict(enum.Enum):
    """Four-way verdict for a single compat-matrix cell (G1 §1.1).

    The validator oracle is *partial*: many bad combos warn or silently no-op
    rather than raise, so a cell is never just pass/fail. ``WARNED_NO_OP`` is the
    silent-shadowing class (same failure mode as the liger-0.8.0 bug) and must be
    surfaced, never rendered as a green check.
    """

    REJECTED = "rejected"  # a validator raised — combo is genuinely invalid
    NORMALIZED = "normalized"  # accepted but rewritten — note the rewrite
    WARNED_NO_OP = "warned_no_op"  # accepted, but warns / silently does nothing
    SUPPORTED = "supported"  # explicitly handled for this model type

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
    """Architecture facts that drive gate applicability and the G2 expected-hook
    set. Produced by ``harness.detect``; consumed read-only by every gate."""

    model_config_type: str
    base_model: str  # user-supplied path or HF id (resolution is required input)
    is_moe: bool = False
    is_multimodal: bool = False
    is_ssm_hybrid: bool = False
    needs_patch: bool = False  # has a patch_manager dispatch branch
    custom_embed_names: bool = False  # non-standard embed/lm_head module names
    architectures: list[str] = field(default_factory=list)  # from config.json
    config_json: dict[str, Any] = field(default_factory=dict)
    # free-form extras a detector wants to pass to gates without widening the
    # contract (e.g. {"causal_conv1d_required": True})
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateContext:
    """Everything a gate needs. Built once by the orchestrator, passed read-only.

    ``output_dir`` is an isolated, harness-owned scratch dir — gates that drive
    preprocess/train (G3..G7) MUST write only under it and must not touch the
    user's tree.
    """

    features: ModelFeatures
    repo_root: Path  # axolotl checkout the static gates introspect
    output_dir: Path  # isolated scratch (configs, prepared data, model artifacts)
    profile: str = "smoke"  # smoke | full | multigpu
    seed: int = 42
    gpu_available: bool = False
    selected_gates: set[str] = field(default_factory=set)
    # per-gate knobs (auto_bisect, rtol, emit_test, snapshot_dir, ...) so the
    # context stays stable as gates grow options
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def model_config_type(self) -> str:
        return self.features.model_config_type


@dataclass
class GateResult:
    """Uniform gate output. ``data`` carries structured rows for the report and
    ``manifest.json`` (e.g. G1 matrix cells); ``details`` is human-readable lines.
    """

    gate_id: str  # "G1"
    name: str  # "config"
    status: GateStatus
    summary: str = ""  # one-line for the report summary block
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
    """Aggregate the liger-style exit contract over all gate results.

    ``COULD_NOT_RUN`` dominates (we could not vouch for the verdict), then
    ``FINDINGS``; ``SKIPPED`` never affects the code.
    """
    statuses = {r.status for r in results}
    if GateStatus.COULD_NOT_RUN in statuses:
        return 2
    if GateStatus.FINDINGS in statuses:
        return 1
    return 0


# Populated by harness.gates at import time: ordered gate-id -> module.
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
