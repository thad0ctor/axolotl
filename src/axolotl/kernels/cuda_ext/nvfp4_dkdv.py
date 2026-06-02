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
EXTENSION_NAME = "axolotl_nvfp4_dkdv"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").lower() in {"1", "true", "yes", "on"}


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


def extension_has_cutlass_headers() -> bool:
    extension = load_extension()
    if extension is None:
        raise RuntimeError(f"set {ENV_FLAG}=1 to build the experimental NVFP4 dK/dV CUDA extension")
    return bool(extension.has_cutlass_headers())


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
    """Real-signature launch probe; writes sentinels, not gradient math."""
    extension = load_extension()
    if extension is None:
        raise RuntimeError(f"set {ENV_FLAG}=1 before calling the experimental NVFP4 dK/dV backend")
    bias_arg = (
        bias
        if bias is not None
        else torch.empty(0, device=qnv.device, dtype=torch.float32)
    )
    extension.dkdv_signature_probe(
        qnv,
        qsc,
        qtnv,
        qtsc,
        donv,
        dosc,
        dotnv,
        dotsc,
        knv,
        ksc,
        vnv,
        vsc,
        bias_arg,
        lse,
        delta,
        dk,
        dv,
        float(scaling),
        int(seed),
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
        bool(sr),
        bool(sr_p_dv),
        int(block_m),
        int(block_n),
    )


def signature_probe_test(
    *,
    z: int = 1,
    h: int = 16,
    hk: int = 4,
    sq: int = 64,
    skv: int = 64,
    d: int = 256,
    block_m: int = 64,
    block_n: int = 32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = torch.device("cuda")
    sq_pad = ((sq + block_m - 1) // block_m) * block_m
    zh = z * h
    zhk = z * hk
    tensors = {
        "qnv": torch.full((zh, sq, d // 2), 1, device=device, dtype=torch.uint8),
        "qsc": torch.full((zh, sq, d // 16), 2, device=device, dtype=torch.uint8),
        "qtnv": torch.full((zh, d, sq_pad // 2), 4, device=device, dtype=torch.uint8),
        "qtsc": torch.full((zh, d, sq_pad // 16), 8, device=device, dtype=torch.uint8),
        "donv": torch.full((zh, sq, d // 2), 16, device=device, dtype=torch.uint8),
        "dosc": torch.full((zh, sq, d // 16), 32, device=device, dtype=torch.uint8),
        "dotnv": torch.full((zh, d, sq_pad // 2), 64, device=device, dtype=torch.uint8),
        "dotsc": torch.full((zh, d, sq_pad // 16), 128, device=device, dtype=torch.uint8),
        "knv": torch.full((zhk, skv, d // 2), 3, device=device, dtype=torch.uint8),
        "ksc": torch.full((zhk, skv, d // 16), 5, device=device, dtype=torch.uint8),
        "vnv": torch.full((zhk, skv, d // 2), 7, device=device, dtype=torch.uint8),
        "vsc": torch.full((zhk, skv, d // 16), 11, device=device, dtype=torch.uint8),
    }
    lse = torch.full((zh, sq), 0.125, device=device, dtype=torch.float32)
    delta = torch.full((zh, sq), 0.25, device=device, dtype=torch.float32)
    dk = torch.empty((zh, skv, d), device=device, dtype=torch.float32)
    dv = torch.empty((zh, skv, d), device=device, dtype=torch.float32)
    dkdv_backward(
        tensors["qnv"],
        tensors["qsc"],
        tensors["qtnv"],
        tensors["qtsc"],
        tensors["donv"],
        tensors["dosc"],
        tensors["dotnv"],
        tensors["dotsc"],
        tensors["knv"],
        tensors["ksc"],
        tensors["vnv"],
        tensors["vsc"],
        None,
        lse,
        delta,
        dk,
        dv,
        1.0,
        1234,
        sq,
        sq_pad,
        skv,
        D=d,
        H=h,
        HK=hk,
        sb_z=0,
        sdk_n=dk.stride(1),
        sdv_n=dv.stride(1),
        has_bias=False,
        causal=True,
        sr=True,
        sr_p_dv=False,
        block_m=block_m,
        block_n=block_n,
    )
    torch.cuda.synchronize(device)
    expected_dk = torch.tensor(1009.375, device=device)
    expected_dv = torch.tensor(2252.375, device=device)
    torch.testing.assert_close(dk[0, 0, 0], expected_dk)
    torch.testing.assert_close(dv[0, 0, 0], expected_dv)
    return dk, dv
