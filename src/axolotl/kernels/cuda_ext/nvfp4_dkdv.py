"""Default-off JIT loader for an experimental native NVFP4 dK/dV backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


ENV_FLAG = "AXOLOTL_NVFP4_DKDV_CUDA"
ENV_SRC = "AXOLOTL_NVFP4_DKDV_CUDA_SRC"
ENV_ARCH_LIST = "TORCH_CUDA_ARCH_LIST"
DEFAULT_ARCH_LIST = "12.0"
EXTENSION_NAME = "axolotl_nvfp4_dkdv_smoke"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").lower() in {"1", "true", "yes", "on"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _source_path() -> Path:
    override = os.environ.get(ENV_SRC)
    if override:
        return Path(override).expanduser().resolve()
    return _repo_root() / "csrc" / "nvfp4_attention" / "dkdv_smoke.cu"


def cutlass_header_paths() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("CUTLASS_PATH", "CUTE_PATH"):
        value = os.environ.get(env_name)
        if value:
            root = Path(value).expanduser().resolve()
            roots.extend([root, root / "include"])
    if CUDA_HOME:
        roots.append(Path(CUDA_HOME) / "include")
    return [
        root
        for root in roots
        if (root / "cutlass" / "cutlass.h").exists() or (root / "cute" / "config.hpp").exists()
    ]


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


def smoke_test(n: int = 1024, *, device: torch.device | str | None = None) -> torch.Tensor:
    extension = load_extension()
    if extension is None:
        raise RuntimeError(f"set {ENV_FLAG}=1 to build the experimental NVFP4 dK/dV CUDA extension")
    if device is None:
        device = torch.device("cuda")
    x = torch.arange(n, device=device, dtype=torch.float32)
    y = extension.smoke_add_one(x.contiguous())
    torch.cuda.synchronize(device)
    torch.testing.assert_close(y, x + 1.0)
    return y


def dkdv_backward(
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
    seed: int,
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
    sr: bool,
    sr_p_dv: bool,
    block_m: int,
    block_n: int,
) -> None:
    if load_extension() is None:
        raise RuntimeError(f"set {ENV_FLAG}=1 before calling the experimental NVFP4 dK/dV backend")
    raise NotImplementedError("CUDA dK/dV kernel is not implemented; smoke JIT is the current feasibility boundary")
