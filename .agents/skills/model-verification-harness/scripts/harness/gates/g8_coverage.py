"""G8 — test-coverage gaps + a regression-test scaffolder. Axolotl has no test_<model>.py convention, so G8 content-greps the whole test tree for this type; a gap is ADVISORY (FINDINGS only when ZERO references exist). --emit-test scaffolds a parametrized e2e from the verified G1 matrix."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .. import GateContext, GateResult, GateStatus

GATE_ID = "G8"
GATE_NAME = "coverage"

# capability label -> content tokens (case-insensitive substrings) marking a test
# that references this type as also exercising that capability
_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "lora": ("lora",),
    "qlora / 4bit": ("qlora", "load_in_4bit"),
    "8bit quant": ("load_in_8bit",),
    "full fine-tune": ("fft", "full_finetune", "test_ft", "fullfinetune"),
    "sample_packing": ("sample_packing",),
    "cut_cross_entropy": ("cut_cross_entropy", "cutcrossentropy"),
    "liger": ("liger",),
    "flash_attention": ("flash_attention", "flash_attn"),
    "gradient_checkpointing": ("gradient_checkpointing",),
    "rl / dpo / grpo": ("rl:", '"rl"', "'rl'", "dpo", "grpo", "kto", "orpo"),
    "multimodal": ("pixel_values", "image_token", "processing_strategy"),
}


def applies(ctx: GateContext) -> bool:  # noqa: ARG001 - coverage check is universal
    return True


# --------------------------------------------------------------------------- #
# token construction
# --------------------------------------------------------------------------- #
def _word_re(token: str) -> re.Pattern[str]:
    # avoid `phi` matching `graphics`: require non-alnum/underscore boundaries
    return re.compile(rf"(?<![a-z0-9_]){re.escape(token.lower())}(?![a-z0-9_])")


def _model_tokens(ctx: GateContext) -> list[str]:
    feats = ctx.features
    toks: set[str] = set()
    mct = (feats.model_config_type or "").lower()
    if mct:
        toks.add(mct)
    base = (feats.base_model or "").rstrip("/").split("/")[-1].lower()
    if base:
        toks.add(base)
    # drop 1-2 char tokens: too promiscuous for a content search (fake coverage)
    return sorted(t for t in toks if len(t) >= 3)


def _patch_surfaces(ctx: GateContext) -> list[str]:
    mct = ctx.features.model_config_type or ""
    if not mct:
        return []
    return [
        f"monkeypatch.models.{mct}",
        f"monkeypatch/models/{mct}",
        f"integrations/liger/{mct}",
    ]


def _test_root(ctx: GateContext) -> Path:
    return ctx.repo_root / "tests"


# --------------------------------------------------------------------------- #
# static coverage
# --------------------------------------------------------------------------- #
def _scan(ctx: GateContext) -> dict[str, Any]:
    root = _test_root(ctx)
    model_toks = _model_tokens(ctx)
    surfaces = _patch_surfaces(ctx)
    surface_low = [s.lower() for s in surfaces]
    tok_res = [_word_re(t) for t in model_toks]

    searched: list[str] = []
    referencing: list[str] = []
    surface_hits: dict[str, bool] = {s: False for s in surfaces}
    cap_files: dict[str, list[str]] = {cap: [] for cap in _CAPABILITIES}

    if not root.is_dir():
        return {
            "searched_paths": [],
            "model_tokens": model_toks,
            "patch_surfaces": surfaces,
            "referencing_files": [],
            "covered": [],
            "uncovered": list(_CAPABILITIES),
            "surface_covered": surface_hits,
            "error": f"no tests/ tree at {root}",
        }

    searched = sorted(
        str(p.relative_to(ctx.repo_root))
        for p in root.iterdir()
        if p.is_dir() or p.suffix == ".py"
    )

    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        rel = str(path.relative_to(ctx.repo_root))
        file_surface_hit = False
        for idx, s in enumerate(surface_low):
            if s in text:
                surface_hits[surfaces[idx]] = True
                file_surface_hit = True
        # a patch-surface mention counts as a reference too, else surface-only coverage
        # still reports the model "undefended"
        if not (any(rx.search(text) for rx in tok_res) or file_surface_hit):
            continue
        referencing.append(rel)
        for cap, tokens in _CAPABILITIES.items():
            if any(tok in text for tok in tokens):
                cap_files[cap].append(rel)

    covered = sorted(cap for cap, files in cap_files.items() if files)
    uncovered = sorted(cap for cap, files in cap_files.items() if not files)
    return {
        "searched_paths": searched,
        "model_tokens": model_toks,
        "patch_surfaces": surfaces,
        "referencing_files": referencing,
        "covered": covered,
        "uncovered": uncovered,
        "cap_files": {k: v for k, v in cap_files.items() if v},
        "surface_covered": surface_hits,
    }


# --------------------------------------------------------------------------- #
# emit-test scaffolder
# --------------------------------------------------------------------------- #
_PASSING_VERDICTS = {"supported", "normalized"}


def _load_matrix(ctx: GateContext) -> list[dict[str, Any]] | None:
    """Read a G1 matrix the orchestrator may have dropped, since gates can't see each other's GateResults."""
    candidates = []
    opt_path = ctx.options.get("g1_matrix_file")
    if opt_path:
        candidates.append(Path(opt_path))
    candidates.append(ctx.output_dir / "g1_matrix.json")
    for path in candidates:
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data = data.get("matrix", [])
                if isinstance(data, list):
                    return data
        except (OSError, ValueError):
            continue
    return None


def _passing_composites(matrix: list[dict[str, Any]]) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for cell in matrix:
        if cell.get("expect") != "resolve":
            continue
        if cell.get("verdict") not in _PASSING_VERDICTS:
            continue
        flags = cell.get("flags") or {}
        if isinstance(flags, dict):
            out.append((str(cell.get("composite_id", f"case{len(out)}")), flags))
    return out


def _ident(model_type: str) -> str:
    ident = re.sub(r"[^0-9a-zA-Z]+", "_", model_type or "model").strip("_")
    return ident or "model"


def _scaffold_source(
    model_type: str, base_model: str, composites: list[tuple[str, dict]]
) -> str:
    cases = "\n".join(
        f"    pytest.param({flags!r}, id={cid!r})," for cid, flags in composites
    )
    ident = _ident(model_type)
    return f'''"""Auto-generated regression test for {model_type}.

Scaffolded by the model-verification harness (--emit-test) from the verified G1
compat matrix. Review thresholds and move into tests/e2e/ before committing.
"""

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_tensorboard_loss_decreased

_BASE_MODEL = {base_model!r}

_VERIFIED_COMPOSITES = [
{cases}
]


@pytest.mark.parametrize("flags", _VERIFIED_COMPOSITES)
def test_{ident}_verified_composite(flags, temp_dir):
    cfg = DictDefault(
        {{
            "base_model": _BASE_MODEL,
            "sequence_len": 512,
            "val_set_size": 0.0,
            "datasets": [{{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}}],
            "num_epochs": 1,
            "max_steps": 10,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "optimizer": "adamw_torch",
            "lr_scheduler": "cosine",
            "logging_steps": 1,
            "output_dir": temp_dir,
            "use_tensorboard": True,
            "save_first_step": False,
        }}
    )
    cfg.update(flags)

    cfg = validate_config(cfg)
    normalize_config(cfg)
    dataset_meta = load_datasets(cfg=cfg)
    train(cfg=cfg, dataset_meta=dataset_meta)
    check_tensorboard_loss_decreased(
        temp_dir + "/runs", initial_window=5, final_window=5
    )
'''


def _emit_scaffold(ctx: GateContext) -> dict[str, Any]:
    mct = ctx.features.model_config_type or "model"
    base_model = ctx.features.base_model
    matrix = _load_matrix(ctx)
    limitation = ""
    if matrix:
        composites = _passing_composites(matrix)
    else:
        composites = []
    if not composites:
        # no matrix/passing cells: emit a default placeholder (no model- or
        # hardware-specific flags) so the scaffold is runnable for any target model
        composites = [("representative", {})]
        limitation = (
            "no G1 matrix file available to G8 (gates can't read each other's "
            "results); scaffolded a single representative case — re-run with a "
            "matrix dropped at output_dir/g1_matrix.json for full coverage"
        )
    path = ctx.output_dir / f"scaffold_test_{_ident(mct)}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_scaffold_source(mct, base_model, composites), encoding="utf-8")
    return {
        "scaffold_path": str(path),
        "scaffold_cases": [cid for cid, _ in composites],
        "scaffold_limitation": limitation,
    }


# --------------------------------------------------------------------------- #
# measured coverage (best-effort)
# --------------------------------------------------------------------------- #
def _measured(ctx: GateContext) -> str:
    try:
        import coverage  # noqa: F401
    except Exception:  # noqa: BLE001
        return "not run (coverage not importable)"
    data_file = ctx.output_dir / ".coverage"
    if not data_file.is_file():
        return "not run (no smoke-run coverage data under output_dir)"
    return "coverage data present (patch-surface line coverage not computed here)"


# --------------------------------------------------------------------------- #
def run(ctx: GateContext) -> GateResult:
    scan = _scan(ctx)
    details: list[str] = []

    details.append(
        "searched: tests/ tree ("
        + ", ".join(scan["searched_paths"][:12])
        + (" ..." if len(scan["searched_paths"]) > 12 else "")
        + ")"
    )
    details.append(f"model tokens: {scan['model_tokens']}")
    details.append(f"{len(scan['referencing_files'])} test file(s) reference this type")

    surface_cov = scan.get("surface_covered", {})
    for surface, hit in surface_cov.items():
        details.append(
            f"  {'✅' if hit else '·'} patch surface {surface}: "
            + ("referenced by a test" if hit else "no test references it")
        )

    if scan["covered"]:
        details.append("covered capabilities: " + ", ".join(scan["covered"]))
    if scan["uncovered"]:
        details.append("uncovered (advisory gaps): " + ", ".join(scan["uncovered"]))

    measured_note = _measured(ctx)
    details.append(f"measured coverage: {measured_note}")

    data: dict[str, Any] = dict(scan)
    data["measured"] = measured_note
    reliability: list[str] = []

    if ctx.options.get("emit_test"):
        emitted = _emit_scaffold(ctx)
        data.update(emitted)
        details.append(f"emitted scaffold: {emitted['scaffold_path']}")
        details.append(f"  cases: {emitted['scaffold_cases']}")
        if emitted["scaffold_limitation"]:
            details.append(f"  ! {emitted['scaffold_limitation']}")
            reliability.append(emitted["scaffold_limitation"])

    if reliability:
        data["reliability_warnings"] = reliability

    undefended = not scan["referencing_files"]
    if undefended:
        return GateResult(
            GATE_ID,
            GATE_NAME,
            GateStatus.FINDINGS,
            summary=(
                f"no test references '{ctx.model_config_type}' anywhere "
                "(undefended — add at least one e2e/integration test)"
            ),
            details=details,
            data=data,
        )

    n_cov = len(scan["covered"])
    n_total = len(_CAPABILITIES)
    summary = (
        f"{len(scan['referencing_files'])} test(s) reference this type; "
        f"{n_cov}/{n_total} capabilities covered"
    )
    if scan["uncovered"]:
        summary += f"; advisory gaps: {', '.join(scan['uncovered'])}"
    return GateResult(
        GATE_ID, GATE_NAME, GateStatus.PASS, summary=summary, details=details, data=data
    )
