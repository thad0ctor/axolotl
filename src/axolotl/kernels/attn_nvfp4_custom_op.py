"""torch.compile custom-op boundary for native-NVFP4 attention."""

from __future__ import annotations

import torch

from axolotl.kernels.attn_nvfp4_flash import nvfp4_flash_attention_packed

_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def _dtype_to_code(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_TO_CODE[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported NVFP4 attention output dtype: {dtype}") from exc


def _code_to_dtype(code: int) -> torch.dtype:
    try:
        return _CODE_TO_DTYPE[int(code)]
    except KeyError as exc:
        raise TypeError(f"unsupported NVFP4 attention output dtype code: {code}") from exc


@torch.library.custom_op("axolotl_nvfp4::flash_attention_packed", mutates_args=())
def _flash_attention_packed_op(
    qnv: torch.Tensor,
    qsc: torch.Tensor,
    knv: torch.Tensor,
    ksc: torch.Tensor,
    vnv: torch.Tensor,
    vsc: torch.Tensor,
    key_pad_bias: torch.Tensor,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    scaling: float,
    out_dtype_code: int,
    causal: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    bias = None if key_pad_bias.numel() == 0 else key_pad_bias
    return nvfp4_flash_attention_packed(
        qnv,
        qsc,
        knv,
        ksc,
        vnv,
        vsc,
        z=z,
        h=h,
        hk=hk,
        s_q=s_q,
        s_kv=s_kv,
        d=d,
        scaling=scaling,
        out_dtype=_code_to_dtype(out_dtype_code),
        causal=causal,
        key_pad_bias=bias,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@_flash_attention_packed_op.register_fake
def _(
    qnv,
    qsc,
    knv,
    ksc,
    vnv,
    vsc,
    key_pad_bias,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    scaling: float,
    out_dtype_code: int,
    causal: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
):
    del qsc, knv, ksc, vnv, vsc, key_pad_bias
    del hk, s_kv, scaling, causal, block_m, block_n, num_warps, num_stages
    return qnv.new_empty((z, h, s_q, d), dtype=_code_to_dtype(out_dtype_code))


def nvfp4_flash_attention_packed_custom_op(
    qnv: torch.Tensor,
    qsc: torch.Tensor,
    knv: torch.Tensor,
    ksc: torch.Tensor,
    vnv: torch.Tensor,
    vsc: torch.Tensor,
    *,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    scaling: float,
    out_dtype: torch.dtype,
    causal: bool = False,
    key_pad_bias: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    bias = (
        qnv.new_empty((0,), dtype=torch.float32)
        if key_pad_bias is None
        else key_pad_bias
    )
    return _flash_attention_packed_op(
        qnv,
        qsc,
        knv,
        ksc,
        vnv,
        vsc,
        bias,
        z,
        h,
        hk,
        s_q,
        s_kv,
        d,
        scaling,
        _dtype_to_code(out_dtype),
        causal,
        block_m,
        block_n,
        num_warps,
        num_stages,
    )
