"""Shallow wrapper test around ``scripts/benchmark_multi_gpu.py``.

Runs the benchmark as a subprocess and sanity-checks mode-engagement
(not throughput targets — numbers are hardware-dependent). The default
test cadence skips this because the full benchmark takes ~2.5 minutes
wall-clock; users opt in by dropping the ``skip`` marker or running the
script directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _nvidia_smi_gpu_count() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8", errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 0
    return sum(1 for line in out.splitlines() if line.strip())


# Skipped by default — full benchmark takes ~2.5 min end-to-end, and
# the assertions validate mode-engagement not hardware-specific throughput
# targets (those live in the README / DESIGN.md for reference).
# Users opt in with:
#   CUDA_VISIBLE_DEVICES=1,4,5,7 CUDA_DEVICE_ORDER=PCI_BUS_ID \
#       python scripts/benchmark_multi_gpu.py
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skip(
    reason="full benchmark, run manually via scripts/benchmark_multi_gpu.py; "
    "assertions validate mode-engagement, not throughput targets"
)
def test_benchmark_multi_gpu_runs(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    gpu_count = _nvidia_smi_gpu_count()
    if gpu_count < 4:
        pytest.skip(f"requires >= 4 GPUs; nvidia-smi reports {gpu_count}")

    # Repo root — the script lives at scripts/benchmark_multi_gpu.py
    # and writes its results file to the same directory. To avoid
    # mutating the checked-in results file we run from a tmp_path
    # copy of the script; the JSON output file will land next to the
    # script (i.e. inside tmp_path).
    repo_root = Path(__file__).resolve().parents[2]
    src_script = repo_root / "scripts" / "benchmark_multi_gpu.py"
    assert src_script.exists(), f"missing benchmark script at {src_script}"

    script_copy = tmp_path / "benchmark_multi_gpu.py"
    script_copy.write_text(src_script.read_text())

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1,4,5,7"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    proc = subprocess.run(
        [sys.executable, str(script_copy)],
        env=env,
        cwd=str(tmp_path),
        check=False,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert proc.returncode == 0, (
        f"benchmark exited {proc.returncode}\n"
        f"stdout tail:\n{proc.stdout[-3000:]}\n"
        f"stderr tail:\n{proc.stderr[-3000:]}"
    )

    json_path = tmp_path / "multi_gpu_benchmark_results.json"
    assert json_path.exists(), "benchmark did not write the JSON results file"
    payload = json.loads(json_path.read_text())
    summaries = {s["mode"]: s for s in payload["summaries"]}

    # (1) Every mode completed.
    for mode in ("single", "ddp", "replicated", "zero3"):
        assert mode in summaries, f"mode {mode!r} missing from benchmark output"
        assert summaries[mode]["throughput_samples_per_s"] > 0, (
            f"mode {mode!r} produced zero throughput"
        )

    # (2) Sharding actually saves CPU: ZeRO-3 per-rank CPU bytes
    # should be well below the replicated-mode footprint. Threshold
    # 0.4 gives headroom over the ideal 1/world_size = 0.25 to absorb
    # allocator / alignment overhead.
    z3_cpu = summaries["zero3"]["cpu_pinned_bytes_max"]
    rep_cpu = summaries["replicated"]["cpu_pinned_bytes_max"]
    assert rep_cpu > 0, "replicated mode reported zero CPU bytes — mode did not engage"
    assert z3_cpu <= 0.4 * rep_cpu, (
        f"ZeRO-3 CPU footprint {z3_cpu/1e9:.3f} GB not <= 0.4 x replicated "
        f"{rep_cpu/1e9:.3f} GB (sharding may not have engaged)"
    )

    # (3) DDP scaling invariant (M6 threshold): DDP throughput > 2.5x
    # single-rank. Same bar the test_protrain_4gpu_throughput_scaling
    # test asserts.
    single_tp = summaries["single"]["throughput_samples_per_s"]
    ddp_tp = summaries["ddp"]["throughput_samples_per_s"]
    assert ddp_tp > 2.5 * single_tp, (
        f"DDP throughput {ddp_tp:.2f} not > 2.5 x single-rank {single_tp:.2f}"
    )
