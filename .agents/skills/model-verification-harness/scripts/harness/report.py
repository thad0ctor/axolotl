"""Standardized markdown report + reproducibility manifest. Every section renders what is present and never raises, so a partially-built ladder still produces output."""

from __future__ import annotations

import dataclasses
import platform
import subprocess  # nosec B404 - only `git rev-parse` on the repo path
from pathlib import Path
from typing import Any

from . import GateResult, Verdict, exit_code

_VERDICT_ICON = {v.value: v.icon for v in Verdict}

# every report leads with this — the harness is a heuristic smoke gate, not an oracle
_DISCLAIMER = [
    "> ⚠️ **Starting point, not an authority.** This is an automated smoke gate built on "
    "static registry checks + tiny-model probes. It flags *likely* gaps and exercises "
    "*some* paths — it does not prove correctness, and a clean run is not a guarantee. "
    "Treat every finding (and every PASS) as a lead to verify independently against the "
    "model's own modeling code, the PR diff, and upstream docs before acting on it. Known "
    "blind spots: per-arch numerical/attention correctness, transformers-version-coupled "
    "breakage, VRAM/memory regressions, subtle tokenization offsets, and distributed-only "
    "failures (see the gate caveats below).",
]

# G2 hook status -> checklist marker
_HOOK_MARKER = {
    "present_explicit": "[x]",
    "generic_fallback": "[⚠️]",
    "missing": "[ ]",
    "not_expected": "[·]",
}


# --------------------------------------------------------------------------- #
# env facts (all guarded — a missing import degrades to "unknown")
# --------------------------------------------------------------------------- #
def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:  # noqa: BLE001
        return "n/a"


def _transformers_version() -> str:
    try:
        import transformers

        return str(transformers.__version__)
    except Exception:  # noqa: BLE001
        return "n/a"


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.run(  # nosec B603 B607 - fixed `git` argv, no shell, no user input
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        pass
    return "none"


def _versions(ctx) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": _torch_version(),
        "transformers": _transformers_version(),
        "axolotl": _git_sha(ctx.repo_root),
    }


def _by_id(results: list[GateResult]) -> dict[str, GateResult]:
    return {r.gate_id: r for r in results}


# --------------------------------------------------------------------------- #
# render_report
# --------------------------------------------------------------------------- #
def render_report(ctx, features, results: list[GateResult]) -> str:
    vers = _versions(ctx)
    lines: list[str] = []
    lines.append(
        f"# Model Verification Report — {features.model_config_type} "
        f"({features.base_model})"
    )
    lines.append(
        f"Env: python {vers['python']} · axolotl {vers['axolotl']} · "
        f"torch {vers['torch']} · transformers {vers['transformers']} · "
        f"GPU {_gpu_name()}"
    )
    lines.append(f"Profile: {ctx.profile} · seed {ctx.seed}")
    lines.append("")
    lines.extend(_DISCLAIMER)
    lines.append("")

    lines.extend(_summary_block(results))
    lines.extend(_compat_matrix(results))
    lines.extend(_integration_checklist(results))
    lines.extend(_masking_sample(results))
    lines.extend(_consistency_table(results))
    lines.extend(_gate_details(results))
    lines.extend(_reliability(results))

    # honor the run's --on-unavailable so this line matches the process exit code
    on_unavailable = (
        ctx.options.get("on_unavailable", "fail") if ctx.options else "fail"
    )
    code = exit_code(results, on_unavailable=on_unavailable)
    meaning = {
        0: "all selected gates clean",
        1: "findings need review",
        2: "could not run reliably",
    }
    lines.append("")
    lines.append(f"Exit code: {code} — {meaning.get(code, 'unknown')}")
    lines.append("")
    lines.append(
        "_Automated starting point — verify findings against the model's own code/docs "
        "before acting; a green run is not a guarantee of correctness (see caveats above)._"
    )
    return "\n".join(lines) + "\n"


def _summary_block(results: list[GateResult]) -> list[str]:
    out = ["## Summary"]
    for r in results:
        label = f"{r.gate_id} {r.name}".ljust(17)
        out.append(f"{label}{r.status.icon} {r.summary}")
    out.append("")
    return out


def _compat_matrix(results: list[GateResult]) -> list[str]:
    g1 = _by_id(results).get("G1")
    out = [
        "## Compat matrix  (verdict ∈ rejected | normalized | warned/no-op | supported)"
    ]
    matrix = (g1.data.get("matrix") if g1 and isinstance(g1.data, dict) else None) or []
    if not matrix:
        out.append("_(no G1 matrix data)_")
        out.append("")
        return out
    out.append("| composite | verdict | note | warnings / bisect |")
    out.append("|---|---|---|---|")
    for cell in matrix:
        if not isinstance(cell, dict):
            continue
        cid = cell.get("composite_id", "?")
        verdict = cell.get("verdict", "?")
        icon = _VERDICT_ICON.get(verdict, "")
        note = _cell_text(cell.get("note", ""))
        extras: list[str] = []
        warns = cell.get("warnings") or []
        if warns:
            extras.append("warns: " + _cell_text("; ".join(str(w) for w in warns), 80))
        bisect = cell.get("bisect") or []
        if bisect:
            extras.append(f"minimal-fail: {bisect}")
        out.append(
            f"| {cid} | {icon} {verdict} | {note} | {_cell_text('; '.join(extras))} |"
        )
    out.append("")
    return out


def _cell_text(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value).split())
    text = text.replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _integration_checklist(results: list[GateResult]) -> list[str]:
    g2 = _by_id(results).get("G2")
    out = ["## Integration checklist"]
    checklist = (
        g2.data.get("checklist") if g2 and isinstance(g2.data, dict) else None
    ) or []
    if not checklist:
        out.append("_(no G2 checklist data)_")
        out.append("")
        return out

    gaps = [
        r
        for r in checklist
        if isinstance(r, dict) and r.get("status") in ("missing", "generic_fallback")
    ]
    if gaps:
        out.append("Gaps (review first):")
        for r in gaps:
            out.append(_hook_line(r))
        out.append("")
    out.append("All expected hooks:")
    for r in checklist:
        if isinstance(r, dict):
            out.append(_hook_line(r))
    out.append("")
    return out


def _hook_line(row: dict[str, Any]) -> str:
    marker = _HOOK_MARKER.get(row.get("status", ""), "[?]")
    hook = row.get("hook", "?")
    gated = row.get("gated_on", "")
    note = row.get("note", "")
    suffix = f" — {note}" if note else ""
    gated_s = f" ({gated})" if gated else ""
    return f"- {marker} {hook}{gated_s}{suffix}"


def _masking_sample(results: list[GateResult]) -> list[str]:
    g4 = _by_id(results).get("G4")
    if not (g4 and isinstance(g4.data, dict)):
        return []
    # prefer the gate's pre-rendered string; else reconstruct from structured spans
    render = g4.data.get("render")
    out = ["## Masking sample", "(trained spans in **bold**)"]
    if isinstance(render, str) and render:
        out.append(render)
        out.append("")
        return out
    spans = None
    for key in ("spans", "sample_spans", "masking_sample", "decoded_sample"):
        if g4.data.get(key):
            spans = g4.data[key]
            break
    if spans is None:
        return []
    out.append(_render_spans(spans) if isinstance(spans, list) else str(spans))
    out.append("")
    return out


def _render_spans(spans: list[Any]) -> str:
    parts: list[str] = []
    for span in spans:
        if isinstance(span, dict):
            text = str(span.get("text", ""))
            trained = bool(span.get("trained", not span.get("masked", True)))
        elif isinstance(span, (list, tuple)) and len(span) >= 2:
            # G4 stores [trained_bool, text]; tolerate the reverse order too.
            if isinstance(span[0], bool):
                trained, text = bool(span[0]), str(span[1])
            else:
                text, trained = str(span[0]), bool(span[1])
        else:
            text, trained = str(span), False
        parts.append(f"**{text}**" if trained else text)
    return "".join(parts)


def _consistency_table(results: list[GateResult]) -> list[str]:
    g7 = _by_id(results).get("G7")
    if not (g7 and isinstance(g7.data, dict)):
        return []
    rows = g7.data.get("variants") or []
    if not rows:
        return []
    base = g7.data.get("baseline_step0")
    out = ["## Numerical consistency (G7)"]
    if base is not None:
        out.append(f"baseline step-0 loss: {base}")
    out.append("| variant | step0 | Δ vs base | rel | tol | engaged | within tol |")
    out.append("|---|---|---|---|---|---|---|")
    eng_label = {True: "yes", False: "**NO**", None: "unknown"}
    for r in rows:
        if not isinstance(r, dict):
            continue
        out.append(
            "| {v} | {s} | {d} | {rel} | {t} | {eng} | {w} |".format(
                v=_cell_text(r.get("variant", "?"), 40),
                s=_num(r.get("step0")),
                d=_num(r.get("delta")),
                rel=_num(r.get("rel_delta")),
                t=_num(r.get("tol")),
                eng=eng_label.get(r.get("engaged"), "—") if "engaged" in r else "—",
                w=""
                if r.get("within_tol") is None
                else ("yes" if r.get("within_tol") else "**NO**"),
            )
        )
    out.append("")
    return out


def _num(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.4g}"
    return "—" if v is None else str(v)


# Gates whose verdict lives in their `details` lines (no dedicated section above).
_DETAIL_GATES = ("G3", "G5", "G6", "G7", "G8")


def _gate_details(results: list[GateResult]) -> list[str]:
    out: list[str] = []
    by_id = _by_id(results)
    for gid in _DETAIL_GATES:
        r = by_id.get(gid)
        if not r or not r.details:
            continue
        out.append(f"## {r.gate_id} {r.name} — detail")
        out.extend(f"- {line}" for line in r.details)
        out.append("")
    return out


def _reliability(results: list[GateResult]) -> list[str]:
    collected: list[str] = []
    for r in results:
        if isinstance(r.data, dict):
            for w in r.data.get("reliability_warnings", []) or []:
                collected.append(f"[{r.gate_id}] {w}")
    if not collected:
        return []
    out = ["## Notes / reliability warnings"]
    out.extend(f"- {w}" for w in collected)
    out.append("")
    return out


# --------------------------------------------------------------------------- #
# build_manifest
# --------------------------------------------------------------------------- #
def _import_under_repo_src(repo_root: Path, import_path: str) -> bool:
    # structural containment, not substring: `/work/src-old` must not match `/work/src`
    if not import_path:
        return False
    try:
        src_dir = (repo_root / "src").resolve()
        import_file = Path(import_path).resolve()
    except OSError:
        return False
    return src_dir == import_file or src_dir in import_file.parents


def build_manifest(args, ctx, features, results: list[GateResult]) -> dict[str, Any]:
    import_path = ctx.options.get("axolotl_import_path", "") if ctx.options else ""
    manifest = {
        "tool": "model-verification-harness",
        "seed": ctx.seed,
        "profile": ctx.profile,
        "gpu": _gpu_name(),
        "gpu_available": bool(ctx.gpu_available),
        "repo_root": str(ctx.repo_root),
        "output_dir": str(ctx.output_dir),
        "versions": _versions(ctx),
        "axolotl_git_sha": _git_sha(ctx.repo_root),
        # which axolotl the run gates imported, and whether it matches repo_root
        "axolotl_import_path": import_path,
        "repo_import_match": _import_under_repo_src(ctx.repo_root, import_path),
        "emitted_test": _emitted_test(results),
        "gates": {
            "selected": sorted(ctx.selected_gates),
            "run": [r.gate_id for r in results],
        },
        "model": _model_block(features),
        "results": [
            {"gate_id": r.gate_id, "status": r.status.value, "summary": r.summary}
            for r in results
        ],
        "invocation": _invocation(args),
    }
    flags = _config_flags(results)
    if flags is not None:
        manifest["config_flags"] = flags
    return _jsonable(manifest)


def _emitted_test(results: list[GateResult]) -> str | None:
    for r in results:
        if isinstance(r.data, dict) and r.data.get("scaffold_path"):
            return str(r.data["scaffold_path"])
    return None


def _model_block(features) -> dict[str, Any]:
    try:
        feat = dataclasses.asdict(features)
    except Exception:  # noqa: BLE001
        feat = {"model_config_type": getattr(features, "model_config_type", "")}
    feat.pop("config_json", None)  # large + redundant with base_model
    return feat


def _invocation(args) -> dict[str, Any]:
    keys = (
        "base_model",
        "gates",
        "profile",
        "seed",
        "auto_bisect",
        "on_unavailable",
        "emit_test",
        "snapshot_dir",
        "features",
        "tiny_strategy",
        "gpus",
        "trust_remote_code",
    )
    return {k: getattr(args, k, None) for k in keys}


def _config_flags(results: list[GateResult]) -> list[dict[str, Any]] | None:
    g1 = _by_id(results).get("G1")
    matrix = (g1.data.get("matrix") if g1 and isinstance(g1.data, dict) else None) or []
    if not matrix:
        return None
    out = []
    for cell in matrix:
        if isinstance(cell, dict):
            out.append(
                {
                    "composite_id": cell.get("composite_id"),
                    "verdict": cell.get("verdict"),
                    "flags": cell.get("flags", {}),
                }
            )
    return out or None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
