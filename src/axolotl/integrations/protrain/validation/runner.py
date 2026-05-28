from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve()
_PROTRAIN_DIR = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[5]
_RECIPE_DIR = _HERE.parent / "recipes"
_DEFAULT_RECIPE = _RECIPE_DIR / "3090-8b-qlora-acceptance.yml"

_CPU_CORE_TESTS = (
    "tests/protrain/test_plugin_args_validators.py",
    "tests/protrain/test_cost_search.py",
    "tests/protrain/test_optimizer_checkpoint.py::test_layout_signature_stable_across_calls",
    "tests/protrain/test_optimizer_checkpoint.py::test_layout_signature_changes_with_persistent_ids",
    "tests/protrain/test_optimizer_checkpoint.py::test_layout_signature_changes_with_world_size_or_zero3",
    "tests/protrain/test_optimizer_checkpoint.py::test_load_rejects_v2_metadata_missing_save_mode",
    "tests/protrain/test_world_size_reshard.py::test_reshard_repartitions_persistent_gpu_rank_files",
    "tests/protrain/test_world_size_reshard.py::test_reshard_persistent_gpu_uses_saved_param_group_order",
    "tests/protrain/test_model_family_ownership.py",
    "tests/protrain/test_modec_optimizer_boundary.py::test_optimizer_rejects_nonfinite_hidden_cpu_grad_before_step",
)

_TWO_GPU_TESTS = (
    "tests/protrain/test_multi_gpu_7b.py::test_protrain_2gpu_mistral_modec_smoke",
    "tests/protrain/test_resume_robustness.py::test_resume_hook_inprocess_cycle_continues_training",
    "tests/protrain/test_modec_optimizer_boundary.py::test_optimizer_rejects_nonfinite_hidden_cpu_grad_before_step",
)

_LOSS_RE = re.compile(
    r"['\"]loss['\"]\s*:\s*['\"]?([+-]?(?:nan|inf|[0-9.eE+-]+))", re.I
)
_GRAD_RE = re.compile(
    r"['\"]grad_norm['\"]\s*:\s*['\"]?([+-]?(?:nan|inf|[0-9.eE+-]+))", re.I
)
_NONFINITE_RE = re.compile(r"(?<![A-Za-z])(?:nan|[+-]?inf)(?![A-Za-z])", re.I)


@dataclass
class LaneResult:
    lane: str
    status: str
    seconds: float
    summary: str
    commands: list[list[str]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "status": self.status,
            "seconds": round(self.seconds, 3),
            "summary": self.summary,
            "commands": self.commands,
            "logs": self.logs,
            "gaps": self.gaps,
            "metrics": self.metrics,
        }


def _repo_root() -> Path:
    return _REPO_ROOT


def _base_env(repo_root: Path, gpu_devices: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    src = str(repo_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("DS_SKIP_CUDA_CHECK", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices
    return env


def _gpu_memories_mib() -> list[int]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []
    try:
        out = subprocess.check_output(  # nosec B603
            [
                nvidia_smi,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8", errors="replace")
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return []

    values: list[int] = []
    for raw in out.splitlines():
        try:
            values.append(int(raw.strip()))
        except ValueError:
            continue
    return values


def _visible_gpu_count(gpu_devices: str | None) -> int:
    if gpu_devices:
        return len([part for part in gpu_devices.split(",") if part.strip()])
    return len(_gpu_memories_mib())


def _has_24g_gpu(gpu_devices: str | None) -> bool:
    memories = _gpu_memories_mib()
    if not memories:
        return False
    if gpu_devices:
        selected: list[int] = []
        for part in gpu_devices.split(","):
            try:
                selected.append(int(part.strip()))
            except ValueError:
                return False
        memories = [memories[i] for i in selected if 0 <= i < len(memories)]
    return any(mib >= 23 * 1024 for mib in memories)


def _run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_s: int,
    dry_run: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(argv)}\n")
        log.write(f"cwd={cwd}\n\n")
        if dry_run:
            log.write("DRY RUN: command not executed.\n")
            return 0
        log.flush()
        proc = subprocess.run(  # nosec B603
            argv,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return proc.returncode


def parse_train_metrics(text: str) -> dict[str, Any]:
    losses = [_to_float(match.group(1)) for match in _LOSS_RE.finditer(text)]
    grad_norms = [_to_float(match.group(1)) for match in _GRAD_RE.finditer(text)]
    losses = [value for value in losses if value is not None]
    grad_norms = [value for value in grad_norms if value is not None]
    markers = sorted(
        set(match.group(0).lower() for match in _NONFINITE_RE.finditer(text))
    )
    return {"losses": losses, "grad_norms": grad_norms, "nonfinite_markers": markers}


def _to_float(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def numerical_stability_issues(
    text: str, *, require_loss: bool = False
) -> tuple[list[str], dict[str, Any]]:
    metrics = parse_train_metrics(text)
    issues: list[str] = []
    losses = metrics["losses"]
    grad_norms = metrics["grad_norms"]

    if require_loss and not losses:
        issues.append("no per-step loss values were parsed from the log")
    for idx, loss in enumerate(losses):
        if not math.isfinite(loss):
            issues.append(f"loss[{idx}] is non-finite: {loss}")
        elif loss < 0.0 or loss > 100.0:
            issues.append(f"loss[{idx}] is outside the stability band: {loss}")
    for idx, grad_norm in enumerate(grad_norms):
        if not math.isfinite(grad_norm):
            issues.append(f"grad_norm[{idx}] is non-finite: {grad_norm}")
    if metrics["nonfinite_markers"]:
        issues.append(
            "non-finite marker(s) found in log: "
            + ", ".join(metrics["nonfinite_markers"])
        )
    return issues, metrics


def write_single_gpu_yaml(
    *,
    recipe_path: Path,
    dest_path: Path,
    output_dir: Path,
    max_steps: int,
    save_steps: int,
    resume_from_checkpoint: Path | None = None,
) -> None:
    with recipe_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["output_dir"] = str(output_dir)
    cfg["max_steps"] = int(max_steps)
    cfg["save_steps"] = int(save_steps)
    if resume_from_checkpoint is None:
        cfg.pop("resume_from_checkpoint", None)
    else:
        cfg["resume_from_checkpoint"] = str(resume_from_checkpoint)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _pytest_cmd(
    paths: tuple[str, ...], *, quiet: bool = True, clear_addopts: bool = False
) -> list[str]:
    cmd = [sys.executable, "-m", "pytest"]
    if quiet:
        cmd.append("-q")
    cmd.extend(paths)
    cmd.extend(["-x", "-s"])
    if clear_addopts:
        cmd.extend(["-o", "addopts="])
    return cmd


def run_cpu_core(
    args: argparse.Namespace, work_dir: Path, repo_root: Path
) -> LaneResult:
    start = time.monotonic()
    log_path = work_dir / "cpu-core.log"
    cmd = _pytest_cmd(_CPU_CORE_TESTS)
    rc = _run_subprocess(
        cmd,
        cwd=repo_root,
        env=_base_env(repo_root),
        log_path=log_path,
        timeout_s=args.timeout_s,
        dry_run=args.dry_run,
    )
    status = "PASS" if rc == 0 else "FAIL"
    summary = "core CPU tests passed" if rc == 0 else f"core CPU tests failed rc={rc}"
    if args.dry_run:
        status = "DRY-RUN"
        summary = "core CPU command rendered"
    return LaneResult(
        lane="cpu-core",
        status=status,
        seconds=time.monotonic() - start,
        summary=summary,
        commands=[cmd],
        logs=[str(log_path)],
    )


def run_cpu_full(
    args: argparse.Namespace, work_dir: Path, repo_root: Path
) -> LaneResult:
    start = time.monotonic()
    log_path = work_dir / "cpu-full.log"
    cmd = _pytest_cmd(("tests/protrain",))
    rc = _run_subprocess(
        cmd,
        cwd=repo_root,
        env=_base_env(repo_root),
        log_path=log_path,
        timeout_s=args.timeout_s,
        dry_run=args.dry_run,
    )
    status = "PASS" if rc == 0 else "FAIL"
    summary = (
        "full default ProTrain pytest suite passed"
        if rc == 0
        else f"pytest failed rc={rc}"
    )
    if args.dry_run:
        status = "DRY-RUN"
        summary = "full CPU command rendered"
    return LaneResult(
        lane="cpu-full",
        status=status,
        seconds=time.monotonic() - start,
        summary=summary,
        commands=[cmd],
        logs=[str(log_path)],
    )


def run_single_gpu(
    args: argparse.Namespace, work_dir: Path, repo_root: Path
) -> LaneResult:
    start = time.monotonic()
    lane_dir = work_dir / "single-gpu"
    output_dir = lane_dir / "out"
    train_yaml = lane_dir / "train.yml"
    resume_yaml = lane_dir / "resume.yml"
    train_log = lane_dir / "train.log"
    resume_log = lane_dir / "resume.log"
    train_steps = int(args.single_steps)
    resume_steps = int(args.single_resume_steps)
    checkpoint_dir = output_dir / f"checkpoint-{train_steps}"

    gaps: list[str] = []
    if not args.dry_run and not _has_24g_gpu(args.gpu_devices):
        return LaneResult(
            lane="single-gpu",
            status="SKIP",
            seconds=time.monotonic() - start,
            summary="no visible 24 GiB GPU for 8B QLoRA acceptance",
            gaps=["1x 24 GiB hardware lane not executed"],
        )

    write_single_gpu_yaml(
        recipe_path=Path(args.single_recipe),
        dest_path=train_yaml,
        output_dir=output_dir,
        max_steps=train_steps,
        save_steps=train_steps,
    )
    write_single_gpu_yaml(
        recipe_path=Path(args.single_recipe),
        dest_path=resume_yaml,
        output_dir=output_dir,
        max_steps=resume_steps,
        save_steps=resume_steps,
        resume_from_checkpoint=checkpoint_dir,
    )

    env = _base_env(repo_root, args.gpu_devices)
    cmd_train = [sys.executable, "-m", "axolotl.cli.train", str(train_yaml)]
    rc_train = _run_subprocess(
        cmd_train,
        cwd=repo_root,
        env=env,
        log_path=train_log,
        timeout_s=args.train_timeout_s,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return LaneResult(
            lane="single-gpu",
            status="DRY-RUN",
            seconds=time.monotonic() - start,
            summary="single-GPU train/resume commands rendered",
            commands=[
                cmd_train,
                [sys.executable, "-m", "axolotl.cli.train", str(resume_yaml)],
            ],
            logs=[str(train_log), str(resume_log)],
        )
    if rc_train != 0:
        return LaneResult(
            lane="single-gpu",
            status="FAIL",
            seconds=time.monotonic() - start,
            summary=f"initial 8B QLoRA train failed rc={rc_train}",
            commands=[cmd_train],
            logs=[str(train_log)],
            gaps=gaps,
        )
    if not checkpoint_dir.is_dir():
        return LaneResult(
            lane="single-gpu",
            status="FAIL",
            seconds=time.monotonic() - start,
            summary=f"expected checkpoint missing: {checkpoint_dir}",
            commands=[cmd_train],
            logs=[str(train_log)],
        )

    cmd_resume = [sys.executable, "-m", "axolotl.cli.train", str(resume_yaml)]
    rc_resume = _run_subprocess(
        cmd_resume,
        cwd=repo_root,
        env=env,
        log_path=resume_log,
        timeout_s=args.train_timeout_s,
        dry_run=False,
    )

    train_text = train_log.read_text(encoding="utf-8", errors="replace")
    resume_text = resume_log.read_text(encoding="utf-8", errors="replace")
    train_issues, train_metrics = numerical_stability_issues(
        train_text, require_loss=True
    )
    resume_issues, resume_metrics = numerical_stability_issues(
        resume_text, require_loss=True
    )
    issues = train_issues + resume_issues
    if rc_resume != 0:
        issues.append(f"resume train failed rc={rc_resume}")

    train_losses = train_metrics["losses"]
    resume_losses = resume_metrics["losses"]
    if train_losses and resume_losses:
        limit = 5.0 * train_losses[-1] + 1.0
        if resume_losses[0] >= limit:
            issues.append(
                "resume loss continuity failed: "
                f"pre_end={train_losses[-1]:.4f}, "
                f"resume_start={resume_losses[0]:.4f}, limit={limit:.4f}"
            )

    status = "PASS" if not issues else "FAIL"
    summary = (
        "8B QLoRA save/resume was finite and loss-continuous"
        if not issues
        else "; ".join(issues[:3])
    )
    return LaneResult(
        lane="single-gpu",
        status=status,
        seconds=time.monotonic() - start,
        summary=summary,
        commands=[cmd_train, cmd_resume],
        logs=[str(train_log), str(resume_log)],
        gaps=gaps,
        metrics={
            "train_loss_count": len(train_losses),
            "resume_loss_count": len(resume_losses),
            "last_train_loss": train_losses[-1] if train_losses else None,
            "first_resume_loss": resume_losses[0] if resume_losses else None,
        },
    )


def run_two_gpu(
    args: argparse.Namespace, work_dir: Path, repo_root: Path
) -> LaneResult:
    start = time.monotonic()
    log_path = work_dir / "two-gpu.log"
    if not args.dry_run and _visible_gpu_count(args.gpu_devices) < 2:
        return LaneResult(
            lane="two-gpu",
            status="SKIP",
            seconds=time.monotonic() - start,
            summary="fewer than two visible GPUs",
            gaps=["2-GPU forced Mode C lane not executed"],
        )

    cmd = _pytest_cmd(_TWO_GPU_TESTS, clear_addopts=True)
    rc = _run_subprocess(
        cmd,
        cwd=repo_root,
        env=_base_env(repo_root, args.gpu_devices),
        log_path=log_path,
        timeout_s=args.timeout_s,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return LaneResult(
            lane="two-gpu",
            status="DRY-RUN",
            seconds=time.monotonic() - start,
            summary="two-GPU command rendered",
            commands=[cmd],
            logs=[str(log_path)],
        )

    text = log_path.read_text(encoding="utf-8", errors="replace")
    issues, metrics = numerical_stability_issues(text, require_loss=False)
    if rc != 0:
        issues.append(f"pytest failed rc={rc}")
    status = "PASS" if not issues else "FAIL"
    summary = (
        "forced Mode C, resume robustness, and non-finite boundary checks passed"
        if not issues
        else "; ".join(issues[:3])
    )
    return LaneResult(
        lane="two-gpu",
        status=status,
        seconds=time.monotonic() - start,
        summary=summary,
        commands=[cmd],
        logs=[str(log_path)],
        metrics=metrics,
    )


def _suite_lanes(suite: str) -> tuple[str, ...]:
    if suite == "cpu-core":
        return ("cpu-core",)
    if suite == "cpu-full":
        return ("cpu-full",)
    if suite == "single-gpu":
        return ("single-gpu",)
    if suite == "two-gpu":
        return ("two-gpu",)
    if suite == "full":
        return ("cpu-core", "cpu-full", "single-gpu", "two-gpu")
    raise ValueError(f"unknown suite: {suite}")


def run_validation(args: argparse.Namespace) -> list[LaneResult]:
    repo_root = Path(args.repo_root).resolve()
    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="protrain-validation-")).resolve()

    runners = {
        "cpu-core": run_cpu_core,
        "cpu-full": run_cpu_full,
        "single-gpu": run_single_gpu,
        "two-gpu": run_two_gpu,
    }
    results: list[LaneResult] = []
    for lane in _suite_lanes(args.suite):
        results.append(runners[lane](args, work_dir, repo_root))
    return results


def _print_text(results: list[LaneResult]) -> None:
    print("ProTrain validation results")
    for result in results:
        print(f"[{result.status}] {result.lane}: {result.summary}")
        for log in result.logs:
            print(f"  log: {log}")
        for gap in result.gaps:
            print(f"  gap: {gap}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run maintainer-scale ProTrain validation lanes."
    )
    parser.add_argument(
        "--suite",
        choices=("cpu-core", "cpu-full", "single-gpu", "two-gpu", "full"),
        default="cpu-core",
    )
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--gpu-devices", default=None)
    parser.add_argument("--single-recipe", default=str(_DEFAULT_RECIPE))
    parser.add_argument("--single-steps", type=int, default=50)
    parser.add_argument("--single-resume-steps", type=int, default=60)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--train-timeout-s", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = run_validation(args)
    if args.as_json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        _print_text(results)
    return 1 if any(result.status == "FAIL" for result in results) else 0
