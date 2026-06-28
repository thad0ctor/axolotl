#!/usr/bin/env python3
"""Resolve a model, run the selected gates, and emit a report + manifest with a 0/1/2 exit code (clean / findings / could-not-run). Run inside the training env."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# importable when run as a bare script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    GATE_ORDER,
    GateContext,
    GateResult,
    GateStatus,
    exit_code,
)
from harness.gates import load_gates  # noqa: E402


def _find_repo_root(start: Path) -> Path | None:
    for base in (start, *start.parents):
        if (base / "src" / "axolotl").is_dir():
            return base
    return None


def _parse_gates(spec: str) -> set[str]:
    if spec.strip().lower() == "all":
        return set(GATE_ORDER)
    out: set[str] = set()
    for tok in spec.replace(",", " ").split():
        tok = tok.strip().upper()
        if tok in GATE_ORDER:
            out.add(tok)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--base-model",
        required=True,
        help="model to verify: a local path or HF id (user-supplied is required)",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="axolotl checkout to introspect (default: auto-locate from cwd)",
    )
    p.add_argument(
        "--gates",
        default="all",
        help="comma-separated gate ids to run (e.g. G1,G2) or 'all' (default)",
    )
    p.add_argument(
        "--profile",
        choices=["smoke", "full", "multigpu"],
        default="smoke",
        help="matrix breadth (G1): smoke≈5 composites, full=+auto-bisect, "
        "multigpu=+parallelism flags",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="scratch dir for configs/prepared-data/artifacts (default: a tempdir)",
    )
    p.add_argument(
        "--report", type=Path, default=None, help="write markdown report here"
    )
    p.add_argument(
        "--manifest", type=Path, default=None, help="write manifest.json here"
    )
    p.add_argument(
        "--auto-bisect",
        action="store_true",
        help="G1: decompose a failing composite to the minimal failing flag set",
    )
    p.add_argument(
        "--emit-test",
        action="store_true",
        help="G8: scaffold a regression test from the verified matrix",
    )
    p.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="G4: directory of masking snapshots to diff against",
    )
    p.add_argument(
        "--features",
        default="",
        help="comma-separated feature intents to treat as EXPECTED in G2 "
        "(e.g. sample_packing,fused_attn_kernel,quantization,densemixer); "
        "absent ones stay advisory",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow AutoConfig to execute remote model code for --base-model "
        "(off by default; only enable for fully trusted sources)",
    )
    return p


def _parse_features(spec: str) -> dict[str, bool]:
    return {tok.strip(): True for tok in spec.replace(",", " ").split() if tok.strip()}


def _detect(base_model: str, repo_root: Path, trust_remote_code: bool = False):
    # lazy so a broken detect module surfaces as could-not-run, not a startup crash

    from harness.detect import detect_model

    return detect_model(
        base_model=base_model,
        repo_root=repo_root,
        trust_remote_code=trust_remote_code,
    )


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _is_under_repo_src(repo_root: Path, import_path: str) -> bool:
    # structural containment, not substring: `/work/src-old` must not match `/work/src`
    if not import_path:
        return False
    try:
        src_dir = (repo_root / "src").resolve()
        import_file = Path(import_path).resolve()
    except OSError:
        return False
    return src_dir == import_file or src_dir in import_file.parents


def _axolotl_import_path() -> str:
    try:
        import importlib

        mod = importlib.import_module("axolotl")
        return getattr(mod, "__file__", "") or ""
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # AXOLOTL_LOG_LEVEL survives configure_logging() where a plain setLevel would not
    import os

    os.environ.setdefault("AXOLOTL_LOG_LEVEL", "WARNING")

    repo_root = (
        args.repo_root
        or _find_repo_root(Path.cwd())
        or _find_repo_root(Path(__file__).resolve())
    )
    if repo_root is None:
        print("could not locate an axolotl checkout (src/axolotl). Use --repo-root.")
        return 2

    # make `import axolotl` resolve to repo_root in-process AND on PYTHONPATH, so the
    # G6/G7 subprocess workers introspect the same tree as the static gates
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))
        prev = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(src_dir)] + ([prev] if prev else [])
        )
    axolotl_import_path = _axolotl_import_path()
    if axolotl_import_path and not _is_under_repo_src(repo_root, axolotl_import_path):
        print(
            f"  ! WARNING: `import axolotl` resolves to {axolotl_import_path}, NOT "
            f"under --repo-root {repo_root}. Static gates read repo_root while run "
            "gates import the resolved axolotl — verdicts could mix trees. Run inside "
            "the env whose axolotl matches --repo-root (or set --repo-root to it)."
        )

    selected = _parse_gates(args.gates)
    if not selected:
        print(f"no valid gates in --gates {args.gates!r}; choose from {GATE_ORDER}")
        return 2

    try:
        features = _detect(args.base_model, repo_root, args.trust_remote_code)
    except Exception as exc:  # noqa: BLE001
        print(
            f"could not detect model '{args.base_model}': {exc.__class__.__name__}: {exc}"
        )
        return 2

    tmp_ctx = None
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = args.output_dir
    elif args.emit_test:
        # persist to cwd: a tempdir would delete the emitted scaffold on exit
        output_dir = Path.cwd() / "verify-model-out"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  (--emit-test) artifacts kept under {output_dir}")
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="verify-model-")
        output_dir = Path(tmp_ctx.name)

    ctx = GateContext(
        features=features,
        repo_root=repo_root,
        output_dir=output_dir,
        profile=args.profile,
        seed=args.seed,
        gpu_available=_gpu_available(),
        selected_gates=selected,
        options={
            "auto_bisect": args.auto_bisect or args.profile == "full",
            "emit_test": args.emit_test,
            "snapshot_dir": args.snapshot_dir,
            "axolotl_import_path": axolotl_import_path,
            **_parse_features(args.features),
        },
    )

    print(
        f"Model verification harness — {features.model_config_type} ({features.base_model})"
    )
    print(f"repo: {repo_root}  ·  profile: {args.profile}  ·  gpu: {ctx.gpu_available}")
    print(f"gates: {sorted(selected)}\n")

    modules, import_errors = load_gates()
    for err in import_errors:
        print(f"  ! gate unavailable: {err}")

    results: list[GateResult] = []
    available_ids = set()
    for mod in modules:
        gate_id = mod.GATE_ID
        available_ids.add(gate_id)
        if gate_id not in selected:
            continue
        name = getattr(mod, "GATE_NAME", gate_id)
        try:
            if hasattr(mod, "applies") and not mod.applies(ctx):
                results.append(GateResult.skipped(gate_id, name, "not applicable"))
                print(f"{gate_id} {name}: {GateStatus.SKIPPED.icon} skipped (n/a)")
                continue
            result = mod.run(ctx)
        except Exception as exc:  # noqa: BLE001 - a gate crash is a could-not-run, not fatal
            result = GateResult.could_not_run(
                gate_id, name, f"{exc.__class__.__name__}: {exc}"
            )
        results.append(result)
        print(f"{gate_id} {result.name}: {result.status.icon} {result.summary}")
        # gates can't see each other's results; drop G1's matrix for G8 --emit-test
        if gate_id == "G1" and result.data.get("matrix"):
            (output_dir / "g1_matrix.json").write_text(
                json.dumps({"matrix": result.data["matrix"]}, default=str),
                encoding="utf-8",
            )

    for gate_id in sorted(selected - available_ids):
        results.append(
            GateResult.could_not_run(gate_id, gate_id, "gate module not available")
        )

    for r in results:
        scaffold = r.data.get("scaffold_path") if isinstance(r.data, dict) else None
        if scaffold:
            print(f"  emitted regression test → {scaffold}")

    code = exit_code(results)

    if args.report is not None or args.manifest is not None:
        # a requested-but-unwritable report/manifest forces exit 2 (could-not-run)
        if not _emit_outputs(args, ctx, features, results):
            code = 2

    print(
        f"\nSUMMARY: exit {code}  "
        f"({sum(r.status is GateStatus.FINDINGS for r in results)} findings, "
        f"{sum(r.status is GateStatus.COULD_NOT_RUN for r in results)} could-not-run)"
    )

    if tmp_ctx is not None:
        tmp_ctx.cleanup()
    return code


def _emit_outputs(args, ctx, features, results) -> bool:
    """Returns False on any import/write failure so the caller can map it to exit 2."""
    try:
        from harness.report import build_manifest, render_report
    except Exception as exc:  # noqa: BLE001
        print(f"  ! report module unavailable: {exc}")
        return False
    try:
        if args.report is not None:
            args.report.write_text(
                render_report(ctx, features, results), encoding="utf-8"
            )
            print(f"  report → {args.report}")
        if args.manifest is not None:
            args.manifest.write_text(
                json.dumps(build_manifest(args, ctx, features, results), indent=2),
                encoding="utf-8",
            )
            print(f"  manifest → {args.manifest}")
    except OSError as exc:
        print(f"  ! failed to write outputs: {exc}")
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
