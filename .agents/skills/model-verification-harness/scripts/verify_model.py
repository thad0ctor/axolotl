#!/usr/bin/env python3
"""Resolve a model, run the selected gates, and emit a report + manifest with a 0/1/2 exit code (clean / findings / could-not-run). Run inside the training env."""

from __future__ import annotations

import argparse
import json
import shlex
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
    tiny,  # noqa: E402
)
from harness.gates import load_gates  # noqa: E402


def _find_repo_root(start: Path) -> Path | None:
    for base in (start, *start.parents):
        if (base / "src" / "axolotl").is_dir():
            return base
    return None


def _parse_gates(spec: str) -> tuple[set[str], list[str]]:
    if spec.strip().lower() == "all":
        return set(GATE_ORDER), []
    out: set[str] = set()
    invalid: list[str] = []
    for tok in spec.replace(",", " ").split():
        tok = tok.strip().upper()
        if tok in GATE_ORDER:
            out.add(tok)
        elif tok:
            invalid.append(tok)
    return out, invalid


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--base-model",
        default=None,
        help="model to verify: a local path or HF id (required unless --from-pr/"
        "--from-diff selects discovery report mode)",
    )
    p.add_argument(
        "--from-pr",
        type=int,
        default=None,
        help="report mode: discover candidate new model_config_type(s) from a GitHub "
        "PR's diff (fetched via `gh pr diff`); does not run the gates",
    )
    p.add_argument(
        "--pr-repo",
        default="axolotl-ai-cloud/axolotl",
        help="OWNER/REPO for --from-pr (default: axolotl-ai-cloud/axolotl)",
    )
    p.add_argument(
        "--from-diff",
        type=Path,
        default=None,
        help="report mode: discover candidates from a unified diff file (offline)",
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
    p.add_argument(
        "--tiny-strategy",
        choices=["path", "checkpoint", "shrink"],
        default="path",
        help="how to resolve the model the gates exercise: 'path' (use --base-model "
        "as-is), 'checkpoint' (auto-match a tiny-* checkpoint by arch), 'shrink' "
        "(build a 2-layer model from --base-model's own config)",
    )
    p.add_argument(
        "--model-config-type",
        default="",
        help="model_config_type for --tiny-strategy checkpoint when --base-model is omitted",
    )
    p.add_argument(
        "--gpus",
        default=None,
        help="comma-separated CUDA device indices for the train gates (e.g. 1,2); "
        "default leaves CUDA visibility as-is",
    )
    p.add_argument(
        "--on-unavailable",
        choices=["fail", "skip"],
        default="fail",
        help="how a gate that cannot run here (no GPU, kernel won't compile, model "
        "unloadable) affects the exit code: fail=exit 2 (default, strict gate), "
        "skip=non-blocking (exit reflects only real findings)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the full report sections on stdout (keep only per-gate "
        "summary lines + exit code)",
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


def _gpu_inventory() -> list[dict]:
    """Per visible CUDA device: index, name, total_mem_gb — feeds the agent's GPU picker."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        out: list[dict] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            out.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_mem_gb": round(props.total_memory / (1024**3), 1),
                }
            )
        return out
    except Exception:  # noqa: BLE001
        return []


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


def _fetch_pr_diff(pr: int, repo: str) -> tuple[str | None, str]:
    """Return (diff_text, error). Uses `gh pr diff` with a fixed argv (no shell)."""
    import subprocess  # nosec B404 - fixed argv below, no shell

    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell, no user interpolation into a command string
            ["gh", "pr", "diff", str(pr), "--repo", repo],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except FileNotFoundError:
        return (
            None,
            "the `gh` CLI is not installed; install GitHub CLI or use --from-diff",
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"`gh pr diff` failed: {exc.__class__.__name__}: {exc}"
    if proc.returncode != 0:
        return None, f"`gh pr diff` exit {proc.returncode}: {proc.stderr.strip()}"
    return proc.stdout, ""


def _run_discovery(args, repo_root: Path | None) -> int:
    # report mode stays torch-free: only the static diff parser is imported here
    from harness.discover import discover_from_diff

    if args.from_diff is not None:
        try:
            diff_text = args.from_diff.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"could not read --from-diff {args.from_diff}: {exc}")
            return 2
        source = str(args.from_diff)
    else:
        diff_text, err = _fetch_pr_diff(args.from_pr, args.pr_repo)
        if diff_text is None:
            print(f"could not fetch PR #{args.from_pr} from {args.pr_repo}: {err}")
            return 2
        source = f"{args.pr_repo}#{args.from_pr}"

    result = discover_from_diff(diff_text, repo_root=repo_root)
    candidates = result["candidates"]
    warnings = result["warnings"]

    print(f"PR model auto-discovery — {source}")
    print("=" * 60)
    if not candidates:
        print("\nNo candidate model_config_type found in the diff.")
        for w in warnings:
            print(f"  ! {w}")
        return 2

    print(f"\n{len(candidates)} candidate(s), ranked by evidence:\n")
    for i, c in enumerate(candidates, 1):
        kind = "NEW" if c["is_new"] else "extends"
        print(f"  [{i}] {c['model_config_type']}  (score {c['score']}, {kind})")
        print(
            f"        base_model : {c['base_model'] or '(none — check example/README)'}"
        )
        print(f"        family     : {c['family_root']}")
        print(f"        signals    : {', '.join(c['signals'])}")

    if warnings:
        print("\nReliability warnings (dynamic/external registration):")
        for w in warnings:
            print(f"  ! {w}")

    top = candidates[0]
    target = shlex.quote(str(top["base_model"] or top["model_config_type"]))
    print("\nSuggested follow-up (check out the PR, pick GPUs, then run the gates):")
    print(
        f"  verify_model.py --base-model {target} "
        f"--repo-root <PR checkout> --gates G1,G2 [--trust-remote-code] [--gpus ...]"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # AXOLOTL_LOG_LEVEL survives configure_logging() where a plain setLevel would not
    import os

    os.environ.setdefault("AXOLOTL_LOG_LEVEL", "WARNING")

    if args.from_pr is not None or args.from_diff is not None:
        # report mode: discover candidates, never auto-run the gates (needs a checkout + GPU)
        repo_root = args.repo_root or _find_repo_root(Path.cwd())
        return _run_discovery(args, repo_root)

    # checkpoint strategy can resolve a tiny model by arch alone; other strategies need a base_model
    if not args.base_model and not (
        args.tiny_strategy == "checkpoint" and args.model_config_type
    ):
        print("--base-model is required (or use --from-pr/--from-diff for discovery).")
        return 2

    repo_root = (
        args.repo_root
        or _find_repo_root(Path.cwd())
        or _find_repo_root(Path(__file__).resolve())
    )
    if repo_root is None:
        print("could not locate an axolotl checkout (src/axolotl). Use --repo-root.")
        return 2

    # make `import axolotl` resolve to repo_root in-process AND on PYTHONPATH, so G6/G7 subprocess workers see the same tree as the static gates
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
        # warn-only: a normal pip-installed axolotl resolves outside the checkout (same code,
        # different location), so hard-failing here would false-positive in most real envs + CI

    selected, invalid = _parse_gates(args.gates)
    if invalid:
        print(
            f"invalid gate(s) in --gates {args.gates!r}: {invalid}; choose from {GATE_ORDER}"
        )
        return 2
    if not selected:
        print(f"no valid gates in --gates {args.gates!r}; choose from {GATE_ORDER}")
        return 2

    # output_dir is resolved before detect: 'shrink' writes the tiny model under it
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

    # model_config_type is unknown pre-detect; checkpoint/shrink rely on base_model here
    try:
        effective_base = tiny.resolve_base_model(
            args.base_model,
            args.model_config_type,
            strategy=args.tiny_strategy,
            output_dir=output_dir,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"could not resolve base model (strategy={args.tiny_strategy}): "
            f"{exc.__class__.__name__}: {exc}"
        )
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return 2
    if effective_base != args.base_model:
        print(
            f"  tiny-strategy={args.tiny_strategy}: resolved model → {effective_base}"
        )

    try:
        features = _detect(effective_base, repo_root, args.trust_remote_code)
    except Exception as exc:  # noqa: BLE001
        print(
            f"could not detect model '{effective_base}': {exc.__class__.__name__}: {exc}"
        )
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return 2

    # spread features FIRST so a feature flag can never overwrite explicit control keys below
    feature_options = _parse_features(args.features)
    ctx = GateContext(
        features=features,
        repo_root=repo_root,
        output_dir=output_dir,
        profile=args.profile,
        seed=args.seed,
        gpu_available=_gpu_available(),
        selected_gates=selected,
        options={
            **feature_options,
            "auto_bisect": args.auto_bisect or args.profile == "full",
            "emit_test": args.emit_test,
            "snapshot_dir": args.snapshot_dir,
            "axolotl_import_path": axolotl_import_path,
            # train gates (G6/G7) read this and set CUDA_VISIBLE_DEVICES for their subprocess
            "gpus": args.gpus,
            # so the report's exit-code line matches the process's actual exit code
            "on_unavailable": args.on_unavailable,
            # MM gate forwards this to processor/tokenizer load for remote-code VLMs
            "trust_remote_code": args.trust_remote_code,
        },
    )

    print(
        f"Model verification harness — {features.model_config_type} ({features.base_model})"
    )
    print(f"repo: {repo_root}  ·  profile: {args.profile}  ·  gpu: {ctx.gpu_available}")
    if args.gpus is None:
        for dev in _gpu_inventory():
            print(
                f"  gpu[{dev['index']}] {dev['name']} ({dev['total_mem_gb']} GiB)  "
                "— pass --gpus to pin train gates"
            )
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

    code = exit_code(results, on_unavailable=args.on_unavailable)

    # dual output: print the full report to stdout AND write the markdown/manifest when requested; a requested-but-unwritable file forces exit 2
    if not _emit_outputs(args, ctx, features, results):
        if args.report is not None or args.manifest is not None:
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
    """Render once: print to stdout (unless --quiet) + write --report/--manifest when requested. False on import/write failure so a missing requested output maps to exit 2."""
    try:
        from harness.report import build_manifest, render_report
    except Exception as exc:  # noqa: BLE001
        print(f"  ! report module unavailable: {exc}")
        return False
    report_md = render_report(ctx, features, results)
    if not args.quiet:
        print("\n" + report_md)
    try:
        if args.report is not None:
            args.report.write_text(report_md, encoding="utf-8")
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
