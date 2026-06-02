"""Default-off JIT loader for an experimental native NVFP4 dK/dV backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


ENV_FLAG = "AXOLOTL_NVFP4_DKDV_CUDA"
ENV_REFERENCE_FLAG = "AXOLOTL_NVFP4_DKDV_CUDA_REFERENCE"
ENV_MMA_FLAG = "AXOLOTL_NVFP4_DKDV_CUDA_MMA"
ENV_SRC = "AXOLOTL_NVFP4_DKDV_CUDA_SRC"
ENV_ARCH_LIST = "TORCH_CUDA_ARCH_LIST"
DEFAULT_ARCH_LIST = "12.0a"
EXTENSION_NAME = "axolotl_nvfp4_dkdv"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").lower() in {"1", "true", "yes", "on"}


def reference_enabled() -> bool:
    return os.environ.get(ENV_REFERENCE_FLAG, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def mma_enabled() -> bool:
    return os.environ.get(ENV_MMA_FLAG, "").lower() in {"1", "true", "yes", "on"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _source_path() -> Path:
    override = os.environ.get(ENV_SRC)
    if override:
        return Path(override).expanduser().resolve()
    return _repo_root() / "csrc" / "nvfp4_attention" / "dkdv_smoke.cu"


def _candidate_cutlass_roots() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("CUTLASS_PATH", "CUTE_PATH"):
        value = os.environ.get(env_name)
        if value:
            root = Path(value).expanduser().resolve()
            candidates.extend([root, root / "include"])
    for parent in _repo_root().parents:
        candidates.extend([parent / "cutlass", parent / "cutlass" / "include"])
    if CUDA_HOME:
        candidates.append(Path(CUDA_HOME) / "include")
    return candidates


def cutlass_header_paths() -> list[Path]:
    paths: list[Path] = []
    for root in _candidate_cutlass_roots():
        if (
            root not in paths
            and (root / "cutlass" / "cutlass.h").exists()
            and (root / "cute" / "config.hpp").exists()
        ):
            paths.append(root)
    return paths


def cutlass_available() -> bool:
    return bool(cutlass_header_paths())


def cutlass_missing_message() -> str:
    checked = "\n".join(f"  - {path}" for path in _candidate_cutlass_roots())
    return (
        "CUTLASS/CuTe headers were not found. Set CUTLASS_PATH to a checkout "
        "root or include directory containing cutlass/cutlass.h and cute/config.hpp. "
        f"Checked:\n{checked}"
    )


def load_extension(*, verbose: bool = False, force: bool = False) -> Any | None:
    if not (force or enabled()):
        return None
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set; cannot build NVFP4 dK/dV CUDA extension")
    source = _source_path()
    if not source.exists():
        raise RuntimeError(f"NVFP4 dK/dV CUDA source is missing: {source}")

    old_arch_list = os.environ.get(ENV_ARCH_LIST)
    if old_arch_list is None:
        os.environ[ENV_ARCH_LIST] = DEFAULT_ARCH_LIST
    try:
        return load(
            name=EXTENSION_NAME,
            sources=[str(source)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_include_paths=[str(path) for path in cutlass_header_paths()],
            verbose=verbose,
            with_cuda=True,
        )
    finally:
        if old_arch_list is None:
            os.environ.pop(ENV_ARCH_LIST, None)


def extension_has_cutlass_headers() -> bool:
    extension = load_extension()
    if extension is None:
        raise RuntimeError(f"set {ENV_FLAG}=1 to build the experimental NVFP4 dK/dV CUDA extension")
    return bool(extension.has_cutlass_headers())


def sm120_mxf4nvf4_ue4m3_available() -> bool:
    extension = load_extension()
    if extension is None:
        raise RuntimeError(
            f"set {ENV_FLAG}=1 to build the experimental NVFP4 dK/dV CUDA extension"
        )
    return bool(extension.sm120_mxf4nvf4_ue4m3_available())


def fp4_mma_microbench(
    *,
    tiles: int = 4096,
    iterations: int = 256,
    repeat: int = 20,
    warmup: int = 3,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run an opt-in SM120 packed FP4 MMA microbench for the dK/dV tile shape.

    The native kernel issues ``m16n8k64`` mxf4nvf4/ue4m3 MMA instructions, matching
    the large-K dV/dK substep shape: ``[N=16, M=64] @ [D=8, M=64].T``.
    """
    extension = load_extension()
    if extension is None:
        raise RuntimeError(
            f"set {ENV_FLAG}=1 to build the experimental NVFP4 dK/dV CUDA extension"
        )
    if device is None:
        device = torch.device("cuda")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("fp4_mma_microbench requires a CUDA device")
    if tiles <= 0 or iterations <= 0 or repeat <= 0 or warmup < 0:
        raise ValueError(
            "tiles, iterations, and repeat must be positive; warmup must be "
            "non-negative"
        )

    with torch.cuda.device(device):
        if not bool(extension.sm120_mxf4nvf4_ue4m3_available()):
            raise RuntimeError(
                "SM120 mxf4nvf4 ue4m3 MMA is not available on the current "
                "CUDA device"
            )
        output = None
        for _ in range(warmup):
            output = extension.fp4_mma_microbench(int(tiles), int(iterations))
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            output = extension.fp4_mma_microbench(int(tiles), int(iterations))
        end.record()
        torch.cuda.synchronize(device)

    assert output is not None
    elapsed_ms = float(start.elapsed_time(end))
    per_launch_ms = elapsed_ms / repeat
    mma_ops = tiles * iterations * repeat
    flops = mma_ops * (16 * 8 * 64 * 2)
    tflops = flops / (elapsed_ms / 1000.0) / 1.0e12
    finite = bool(torch.isfinite(output).all().item())
    checksum = float(output.float().sum().item())
    return {
        "output": output,
        "elapsed_ms": elapsed_ms,
        "per_launch_ms": per_launch_ms,
        "tflops": tflops,
        "mma_ops": mma_ops,
        "checksum": checksum,
        "finite": finite,
        "tiles": tiles,
        "iterations": iterations,
        "repeat": repeat,
    }


def fp4_mma_smoke_test(
    *,
    tiles: int = 8,
    iterations: int = 1,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    result = fp4_mma_microbench(
        tiles=tiles,
        iterations=iterations,
        repeat=1,
        warmup=0,
        device=device,
    )
    if not result["finite"]:
        raise AssertionError("SM120 FP4 MMA microbench produced non-finite output")
    return result


def dkdv_backward_reference(
    qnv: torch.Tensor,
    qsc: torch.Tensor,
    donv: torch.Tensor,
    dosc: torch.Tensor,
    knv: torch.Tensor,
    ksc: torch.Tensor,
    vnv: torch.Tensor,
    vsc: torch.Tensor,
    bias: torch.Tensor | None,
    lse: torch.Tensor,
    delta: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    scaling: float,
    sq: int,
    skv: int,
    *,
    D: int,
    H: int,
    HK: int,
    sb_z: int,
    sdk_n: int,
    sdv_n: int,
    has_bias: bool,
    causal: bool,
) -> None:
    """Correctness-first CUDA dK/dV implementation; scalar, not yet tensor-core."""
    extension = load_extension()
    if extension is None:
        raise RuntimeError(
            f"set {ENV_FLAG}=1 before calling the experimental NVFP4 dK/dV backend"
        )
    bias_arg = (
        bias
        if bias is not None
        else torch.empty(0, device=qnv.device, dtype=torch.float32)
    )
    extension.dkdv_reference(
        qnv,
        qsc.view(torch.uint8),
        donv,
        dosc.view(torch.uint8),
        knv,
        ksc.view(torch.uint8),
        vnv,
        vsc.view(torch.uint8),
        bias_arg,
        lse,
        delta,
        dk,
        dv,
        float(scaling),
        int(sq),
        int(skv),
        int(D),
        int(H),
        int(HK),
        int(sb_z),
        int(sdk_n),
        int(sdv_n),
        bool(has_bias),
        bool(causal),
    )


def dkdv_backward_mma_rtn(
    qnv: torch.Tensor,
    qsc: torch.Tensor,
    qtnv: torch.Tensor,
    qtsc: torch.Tensor,
    donv: torch.Tensor,
    dosc: torch.Tensor,
    dotnv: torch.Tensor,
    dotsc: torch.Tensor,
    knv: torch.Tensor,
    ksc: torch.Tensor,
    vnv: torch.Tensor,
    vsc: torch.Tensor,
    bias: torch.Tensor | None,
    lse: torch.Tensor,
    delta: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    scaling: float,
    sq: int,
    sq_pad: int,
    skv: int,
    *,
    D: int,
    H: int,
    HK: int,
    sb_z: int,
    sdk_n: int,
    sdv_n: int,
    has_bias: bool,
    causal: bool,
) -> None:
    """Experimental SM120 native-FP4 MMA dK/dV backend; RTN-only, D=256."""
    extension = load_extension()
    if extension is None:
        raise RuntimeError(
            f"set {ENV_FLAG}=1 before calling the experimental NVFP4 dK/dV backend"
        )
    bias_arg = (
        bias
        if bias is not None
        else torch.empty(0, device=qnv.device, dtype=torch.float32)
    )
    extension.dkdv_mma_rtn(
        qnv,
        qsc.view(torch.uint8),
        qtnv,
        qtsc.view(torch.uint8),
        donv,
        dosc.view(torch.uint8),
        dotnv,
        dotsc.view(torch.uint8),
        knv,
        ksc.view(torch.uint8),
        vnv,
        vsc.view(torch.uint8),
        bias_arg,
        lse,
        delta,
        dk,
        dv,
        float(scaling),
        int(sq),
        int(sq_pad),
        int(skv),
        int(D),
        int(H),
        int(HK),
        int(sb_z),
        int(sdk_n),
        int(sdv_n),
        bool(has_bias),
        bool(causal),
    )
