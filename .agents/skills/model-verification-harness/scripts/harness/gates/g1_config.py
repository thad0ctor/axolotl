"""G1 — config-resolution compat matrix.

G1 drives the REAL ``axolotl.cli.config.load_cfg`` pipeline (plugins ->
gpu_capabilities -> validate_config -> normalize_config) over a small set of
*maximal-compatible* composites and a couple of oracle probes, then assigns each
matrix cell a four-way :class:`Verdict`. It is resolution-only: no GPU, no model
load beyond ``config.json``, no training — so the full set is cheap.

The validator oracle is partial: many bad combos warn or silently no-op instead
of raising. WARNED_NO_OP is the silent-shadowing class (same failure mode as the
liger-0.8.0 bug) and is the high-value signal, so warnings are captured on BOTH
channels — Python ``warnings`` and the ``axolotl`` logging tree — and surfaced as
FINDINGS, never a green check.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import yaml

from .. import GateContext, GateResult, GateStatus, Verdict

GATE_ID = "G1"
GATE_NAME = "config"

# Cross-entropy group: exactly one may be enabled (validation.py
# check_cross_entropy_conflicts). Drives one composite per option.
_CE_OPTIONS = (
    "none",
    "liger_cross_entropy",
    "liger_fused_linear_cross_entropy",
    "cut_cross_entropy",
    "chunked_cross_entropy",
)

_LIGER_PLUGIN = "axolotl.integrations.liger.LigerPlugin"
_CCE_PLUGIN = "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"
_KERNELS_PLUGIN = "axolotl.integrations.kernels.KernelsPlugin"

# Optional flag key -> substrings that mark a captured warning as "about" it. Used
# to attribute a warning to a flag the cell set (so incidental warnings — FSDP1
# deprecation, bf16 auto — don't falsely paint a cell WARNED_NO_OP).
_FLAG_WARN_TOKENS = {
    "sample_packing": ("sample_packing",),
    "batch_flattening": ("batch_flattening",),
    "scaling_softmax": ("scaling_softmax",),
    "liger_use_token_scaling": ("token_scaling",),
    "liger_glu_activation": ("glu", "swiglu"),
    "liger_rms_norm": ("rms_norm",),
    "activation_offloading": ("activation_offloading", "offload"),
    # NB: no bare "kernel" — it appears in unrelated triton/liger advisories and
    # would wrongly paint a healthy MoE cell WARNED_NO_OP.
    "expert_backend": ("expert", "scattermoe", "sonicmoe"),
    "use_scattermoe": ("scattermoe",),
    "moe_grouped_backend": ("moe_grouped", "grouped"),
}

# Flag names too generic to use as warning-attribution tokens (they appear in
# incidental advisories — bf16 auto-cast, fp16 notices). A flag here only matters
# via an explicit _FLAG_WARN_TOKENS entry, never as a bare-name fallback.
_GENERIC_FLAG_DENY = {"bf16", "fp16", "tf32", "fp8", "use_kernels", "kernel"}

# Keys whose presence in the resolved cfg (but not the input) signals a known
# canonicalization rewrite (NORMALIZED) even though the literal input key is intact.
_CANONICALIZATION_WATCH = (
    "use_scattermoe",
    "use_sonicmoe",
    "experts_implementation",
)


def applies(ctx: GateContext) -> bool:
    # Config resolution is meaningful for every model; G1 always runs.
    return True


@dataclass
class _Cell:
    composite_id: str
    flags: dict[str, Any]
    expect: str  # "resolve" | "reject"
    purpose: str


@dataclass
class _Outcome:
    cell: _Cell
    verdict: Verdict
    note: str
    normalized_changes: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    bisect: list[str] = field(default_factory=list)


@contextmanager
def _capture():
    """Capture warnings on both channels for one ``load_cfg`` call.

    axolotl emits most advisories through ``logging`` (get_logger(__name__), all
    children of the ``axolotl`` logger), not ``warnings.warn`` — so a logging
    handler on the ``axolotl`` root is required in addition to catch_warnings.
    ``warning_once`` is lru_cached process-wide; clearing it lets the same advisory
    re-fire for each composite that triggers it.
    """
    records: list[str] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                records.append(record.getMessage())
            except Exception:  # noqa: BLE001 - a bad format arg must not abort capture
                records.append(str(record.msg))

    sink = _Sink(level=logging.WARNING)
    axolotl_log = logging.getLogger("axolotl")
    axolotl_log.addHandler(sink)

    try:
        from axolotl.utils.logging import MultiProcessAdapter

        MultiProcessAdapter.warning_once.cache_clear()
    except Exception:  # noqa: BLE001
        pass

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            yield records
        finally:
            for w in wlist:
                records.append(str(w.message))
            axolotl_log.removeHandler(sink)


def _reset_plugins() -> None:
    # The PluginManager is a process singleton that accumulates registrations; reset
    # it so each composite's load_cfg sees only the plugins its own YAML declares.
    try:
        from axolotl.integrations.base import PluginManager

        PluginManager._instance = None  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass


def _base_cfg(ctx: GateContext, composite_id: str) -> dict[str, Any]:
    return {
        "base_model": ctx.features.base_model,
        "output_dir": str(ctx.output_dir / composite_id),
        "datasets": [{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "sequence_len": 512,
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "val_set_size": 0,
    }


def _run_cfg(
    ctx: GateContext, flags: dict[str, Any], composite_id: str
) -> tuple[dict | None, BaseException | None, list[str]]:
    """Write a minimal YAML and run it through the real load_cfg pipeline."""
    from axolotl.cli.config import load_cfg

    cfg_dict = _base_cfg(ctx, composite_id)
    cfg_dict.update(flags)
    cfg_path = ctx.output_dir / f"{composite_id}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    _reset_plugins()
    resolved: dict | None = None
    exc: BaseException | None = None
    with _capture() as records:
        try:
            resolved = dict(load_cfg(str(cfg_path)))
        except BaseException as err:  # noqa: BLE001 - any raise is a REJECTED verdict
            exc = err
    return resolved, exc, list(records)


def _relevant_warnings(flags: dict[str, Any], records: list[str]) -> list[str]:
    tokens: list[str] = []
    for key in flags:
        if key in _FLAG_WARN_TOKENS:
            tokens.extend(_FLAG_WARN_TOKENS[key])
        elif key not in _GENERIC_FLAG_DENY and key != "plugins":
            tokens.append(key)  # specific flag name as its own token
    out = []
    for msg in records:
        low = msg.lower()
        if any(tok in low for tok in tokens):
            out.append(msg)
    return out


def _normalized_changes(flags: dict[str, Any], resolved: dict) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for key, val in flags.items():
        if key in ("plugins",):
            continue
        if key in resolved and resolved[key] != val:
            changes[key] = {"in": val, "out": resolved[key]}
    # canonicalizations that add a derived key rather than rewriting the input one
    for key in _CANONICALIZATION_WATCH:
        if key not in flags and key in resolved:
            changes[key] = {"in": None, "out": resolved[key]}
    return changes


def _support_note(ctx: GateContext, flags: dict[str, Any]) -> str:
    """Registry cross-check that distinguishes a real SUPPORTED branch from an
    accepted-but-inert flag."""
    notes = []
    if flags.get("sample_packing"):
        from axolotl.monkeypatch.multipack import SUPPORTED_MULTIPACK_MODEL_TYPES

        mt = ctx.features.model_config_type
        attn = flags.get("attn_implementation")
        from axolotl.utils.schemas.enums import ATTN_IMPLS_SUPPORTING_PACKING

        if (
            mt in SUPPORTED_MULTIPACK_MODEL_TYPES
            and attn in ATTN_IMPLS_SUPPORTING_PACKING
        ):
            notes.append(f"sample_packing: {mt} in multipack registry + varlen attn")
        elif attn not in ATTN_IMPLS_SUPPORTING_PACKING:
            notes.append("sample_packing: attn not varlen-capable (no-op risk)")
        else:
            notes.append(f"sample_packing: {mt} NOT in multipack registry")
    return "; ".join(notes)


def _classify(
    ctx: GateContext, cell: _Cell, resolved: dict | None, exc, records: list[str]
) -> _Outcome:
    rel = _relevant_warnings(cell.flags, records)

    if exc is not None:
        note = f"{type(exc).__name__}: {exc}".split("\n")[0]
        if cell.expect == "reject":
            note = f"expected reject — {note}"
        else:
            note = f"UNEXPECTED reject — {note}"
        return _Outcome(cell, Verdict.REJECTED, note, warnings=rel)

    changes = _normalized_changes(cell.flags, resolved or {})
    support = _support_note(ctx, cell.flags)

    if rel:
        return _Outcome(
            cell,
            Verdict.WARNED_NO_OP,
            support or "warned / possible no-op",
            normalized_changes=changes,
            warnings=rel,
        )
    if changes:
        return _Outcome(cell, Verdict.NORMALIZED, support or "value rewritten", changes)
    return _Outcome(cell, Verdict.SUPPORTED, support or "resolved clean")


# --- composite construction ----------------------------------------------------


def _composites(ctx: GateContext) -> list[_Cell]:
    """One maximal-compatible composite per cross-entropy option, spreading the
    optional flags so each is exercised >=1 across the set, then oracle probes."""
    is_moe = ctx.features.is_moe
    cells: list[_Cell] = []

    # C0 / none: sample_packing (SUPPORTED on a multipack model) + gc + offloading.
    cells.append(
        _Cell(
            "c0_none_packing",
            {
                "attn_implementation": "flash_attention_2",
                "sample_packing": True,
                "gradient_checkpointing": True,
                "activation_offloading": True,
            },
            "resolve",
            "CE=none; exercises sample_packing, gradient_checkpointing, activation_offloading",
        )
    )

    # C1 / liger_cross_entropy: batch_flattening (varlen + not sample_packing) + glu/rms.
    cells.append(
        _Cell(
            "c1_liger_ce_batchflatten",
            {
                "plugins": [_LIGER_PLUGIN],
                "liger_cross_entropy": True,
                "liger_glu_activation": True,
                "liger_rms_norm": True,
                "attn_implementation": "flash_attention_2",
                "batch_flattening": True,
            },
            "resolve",
            "CE=liger_cross_entropy; exercises batch_flattening, liger_glu_activation, liger_rms_norm",
        )
    )

    # C2 / FLCE: token_scaling (requires FLCE) + scaling_softmax (requires flex).
    cells.append(
        _Cell(
            "c2_flce_flex_softmax",
            {
                "plugins": [_LIGER_PLUGIN],
                "liger_fused_linear_cross_entropy": True,
                "liger_use_token_scaling": True,  # nosec B105 - axolotl flag name, not a secret
                "attn_implementation": "flex_attention",
                "scaling_softmax": True,
            },
            "resolve",
            "CE=FLCE; exercises liger_use_token_scaling, scaling_softmax, flex_attention",
        )
    )

    # C3 / cut_cross_entropy: requires bf16/fp16 + the CCE plugin.
    cells.append(
        _Cell(
            "c3_cce_bf16",
            {
                "plugins": [_CCE_PLUGIN],
                "cut_cross_entropy": True,
                "bf16": True,
                "attn_implementation": "flash_attention_2",
                "gradient_checkpointing": True,
            },
            "resolve",
            "CE=cut_cross_entropy; exercises cut_cross_entropy + bf16 co-requisite",
        )
    )

    # C4 / chunked: sample_packing on a non-varlen backend. This is a deliberate
    # oracle probe for the WARNED_NO_OP channel (expect="warn"), NOT a model
    # property — its firing is EXPECTED and must not count as a model finding
    # (mirrors p_dual_ce for the REJECTED channel). chunked_cross_entropy keeps
    # this cell in its own CE slot so every CE option is still exercised once.
    cells.append(
        _Cell(
            "c4_chunked_packing_sdpa",
            {
                "chunked_cross_entropy": True,
                "attn_implementation": "sdpa",
                "sample_packing": True,
            },
            "warn",
            "CE=chunked; sample_packing without varlen attn -> oracle probe for WARNED_NO_OP",
        )
    )

    if is_moe:
        # expert_backend canonicalizes onto use_scattermoe -> NORMALIZED probe.
        cells.append(
            _Cell(
                "c5_moe_expert_backend",
                {
                    "plugins": [_KERNELS_PLUGIN],
                    "use_kernels": True,
                    "expert_backend": "scattermoe",
                    "attn_implementation": "flash_attention_2",
                },
                "resolve",
                "MoE; exercises expert_backend canonicalization (-> use_scattermoe)",
            )
        )

    if ctx.profile == "multigpu":
        cells.extend(_multigpu_composites(ctx))

    # Oracle probes (expected rejects) — confirm the partial oracle still fires.
    cells.append(
        _Cell(
            "p_dual_ce",
            {
                "plugins": [_LIGER_PLUGIN],
                "liger_cross_entropy": True,
                "cut_cross_entropy": True,
                "bf16": True,
            },
            "reject",
            "two cross-entropy flags -> expected REJECTED (mutual exclusivity)",
        )
    )
    return cells


def _multigpu_composites(ctx: GateContext) -> list[_Cell]:
    cells = [
        _Cell(
            "mg_fsdp2",
            {
                "fsdp_version": 2,
                "fsdp_config": {"offload_params": False},
                "attn_implementation": "flash_attention_2",
            },
            "resolve",
            "multigpu: FSDP2 resolves",
        ),
        # liger_rms_norm + tensor parallelism is rejected by the liger args.
        _Cell(
            "mg_liger_rmsnorm_tp",
            {
                "plugins": [_LIGER_PLUGIN],
                "liger_rms_norm": True,
                "tensor_parallel_size": 2,
                "attn_implementation": "flash_attention_2",
            },
            "reject",
            "multigpu: liger_rms_norm + tensor_parallel -> expected REJECTED",
        ),
    ]
    return cells


# --- delta-debug ---------------------------------------------------------------


def _bisect(ctx: GateContext, cell: _Cell) -> list[str]:
    """ddmin over the cell's optional flags: smallest subset whose load_cfg still
    rejects. Used on an UNEXPECTED reject to name the culprit flag set."""
    items = [(k, v) for k, v in cell.flags.items() if k != "plugins"]
    plugins = cell.flags.get("plugins")

    def _fails(subset: list[tuple[str, Any]]) -> bool:
        flags = dict(subset)
        if plugins:
            flags["plugins"] = plugins
        _, exc, _ = _run_cfg(ctx, flags, f"{cell.composite_id}__bisect")
        return exc is not None

    if not _fails(items):
        return []  # not reproducible in isolation
    n = 2
    while len(items) >= 2:
        chunk = max(1, len(items) // n)
        subsets = [items[i : i + chunk] for i in range(0, len(items), chunk)]
        reduced = False
        for s in subsets:
            complement = [x for x in items if x not in s]
            if complement and _fails(complement):
                items = complement
                n = max(n - 1, 2)
                reduced = True
                break
        if not reduced:
            if n >= len(items):
                break
            n = min(len(items), n * 2)
    return [k for k, _ in items]


def run(ctx: GateContext) -> GateResult:
    try:
        import axolotl.cli.config  # noqa: F401
    except BaseException as err:  # noqa: BLE001
        return GateResult.could_not_run(
            GATE_ID, GATE_NAME, f"axolotl import failed: {type(err).__name__}: {err}"
        )

    cells = _composites(ctx)
    do_bisect = ctx.profile == "full" or bool(ctx.options.get("auto_bisect"))

    outcomes: list[_Outcome] = []
    could_not_run = 0
    for cell in cells:
        resolved, exc, records = _run_cfg(ctx, cell.flags, cell.composite_id)
        # A pipeline that cannot even import the model config (offline / bad id) is
        # an environment problem, not a config verdict.
        if exc is not None and _is_environment_error(exc):
            could_not_run += 1
            outcomes.append(
                _Outcome(
                    cell,
                    Verdict.REJECTED,
                    f"COULD-NOT-RUN ({type(exc).__name__}: {str(exc).splitlines()[0]})",
                    warnings=[],
                )
            )
            continue
        outcome = _classify(ctx, cell, resolved, exc, records)
        if (
            do_bisect
            and outcome.verdict == Verdict.REJECTED
            and cell.expect == "resolve"
        ):
            outcome.bisect = _bisect(ctx, cell)
        outcomes.append(outcome)

    return _assemble(ctx, outcomes, could_not_run)


_ENV_ERROR_MARKERS = (
    "couldn't connect",
    "could not connect",
    "offline",
    "is not a local folder",
    "is not the path to a directory",
    "does not appear to have a file named config.json",
    "connectionerror",
    "max retries",
    "failed to resolve",
)


def _is_environment_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("OSError", "ConnectionError", "HTTPError", "GatedRepoError"):
        return True
    low = f"{name}: {exc}".lower()
    return any(m in low for m in _ENV_ERROR_MARKERS)


def _assemble(
    ctx: GateContext, outcomes: list[_Outcome], could_not_run: int
) -> GateResult:
    matrix = []
    details = []
    findings = 0
    unexpected_reject = 0
    warned = 0
    resolved_ok = 0
    total_resolvable = 0

    for o in outcomes:
        cell = o.cell
        matrix.append(
            {
                "composite_id": cell.composite_id,
                "flags": cell.flags,
                "expect": cell.expect,
                "verdict": o.verdict.value,
                "note": o.note,
                "normalized_changes": o.normalized_changes,
                "warnings": o.warnings,
                "bisect": o.bisect,
            }
        )
        line = f"{o.verdict.icon} {cell.composite_id}: {o.verdict.value} — {o.note}"
        if o.bisect:
            line += f" [minimal failing flags: {o.bisect}]"
        details.append(line)

        if cell.expect == "resolve":
            total_resolvable += 1
            if o.verdict == Verdict.REJECTED and "COULD-NOT-RUN" not in o.note:
                unexpected_reject += 1
                findings += 1
            elif o.verdict == Verdict.WARNED_NO_OP:
                # a cell built to be valid that warns/no-ops is the real signal
                warned += 1
                findings += 1
                resolved_ok += 1
            elif o.verdict != Verdict.REJECTED:
                resolved_ok += 1
        elif cell.expect == "warn":
            # oracle probe for the WARNED_NO_OP channel: firing is expected (not a
            # finding); resolving clean means the warn path silently stopped working.
            total_resolvable += 1
            if o.verdict == Verdict.WARNED_NO_OP:
                resolved_ok += 1
            elif o.verdict == Verdict.REJECTED and "COULD-NOT-RUN" not in o.note:
                unexpected_reject += 1
                findings += 1
            else:
                findings += 1
                details.append(
                    f"  !! {cell.composite_id} was built to warn/no-op but resolved "
                    "clean (oracle gap — the warn path may have regressed)"
                )
        else:  # expect reject
            if o.verdict != Verdict.REJECTED:
                # the oracle did NOT fire on a combo we built to be invalid
                findings += 1
                details.append(
                    f"  !! {cell.composite_id} was built invalid but RESOLVED "
                    "(oracle gap)"
                )

    if could_not_run and resolved_ok == 0:
        return GateResult(
            GATE_ID,
            GATE_NAME,
            GateStatus.COULD_NOT_RUN,
            summary=f"{could_not_run} composite(s) could not run (model/env unloadable)",
            details=details,
            data={"matrix": matrix},
        )

    status = GateStatus.FINDINGS if findings else GateStatus.PASS
    summary = (
        f"{resolved_ok}/{total_resolvable} composites resolve; "
        f"{warned} warned-no-op, {unexpected_reject} unexpected-reject"
    )
    if could_not_run:
        summary += f"; {could_not_run} could-not-run"
    return GateResult(
        GATE_ID,
        GATE_NAME,
        status,
        summary=summary,
        details=details,
        data={"matrix": matrix, "profile": ctx.profile},
    )
