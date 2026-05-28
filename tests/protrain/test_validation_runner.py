from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

from axolotl.integrations.protrain.validation import runner


def test_parse_train_metrics_flags_nonfinite_values() -> None:
    text = "{'loss': '1.25', 'grad_norm': 'nan'}\n{'loss': 'inf'}"

    issues, metrics = runner.numerical_stability_issues(text, require_loss=True)

    assert metrics["losses"] == [1.25, math.inf]
    assert any("loss[1] is non-finite" in issue for issue in issues)
    assert any("grad_norm[0] is non-finite" in issue for issue in issues)


def test_single_gpu_recipe_writer_sets_resume_and_qlora(tmp_path: Path) -> None:
    dest = tmp_path / "resume.yml"
    output_dir = tmp_path / "out"
    checkpoint = output_dir / "checkpoint-50"

    runner.write_single_gpu_yaml(
        recipe_path=runner._DEFAULT_RECIPE,  # noqa: SLF001
        dest_path=dest,
        output_dir=output_dir,
        max_steps=60,
        save_steps=60,
        resume_from_checkpoint=checkpoint,
    )

    cfg = yaml.safe_load(dest.read_text())
    assert cfg["adapter"] == "qlora"
    assert cfg["load_in_4bit"] is True
    assert cfg["max_steps"] == 60
    assert cfg["save_steps"] == 60
    assert cfg["resume_from_checkpoint"] == str(checkpoint)
    assert cfg["plugins"] == ["axolotl.integrations.protrain.ProTrainPlugin"]
    assert cfg["protrain_auto_memory"] is True


def test_validation_module_dry_run_json(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "axolotl.integrations.protrain.validation",
        "--suite",
        "single-gpu",
        "--dry-run",
        "--json",
        "--work-dir",
        str(tmp_path),
    ]

    proc = subprocess.run(  # nosec B603
        cmd,
        cwd=runner._repo_root(),  # noqa: SLF001
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload[0]["lane"] == "single-gpu"
    assert payload[0]["status"] == "DRY-RUN"
    rendered = " ".join(" ".join(cmd) for cmd in payload[0]["commands"])
    assert "axolotl.cli.train" in rendered
