"""torch.compile custom-op boundary for native-NVFP4 attention."""

from __future__ import annotations

import torch

from axolotl.kernels.attn_nvfp4_flash import (
    _gqa_reduce_cast_dkdv,
    _run_bwd,
    nvfp4_flash_attention,
    nvfp4_flash_attention_packed,
)

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
        raise TypeError(
            f"unsupported NVFP4 attention output dtype code: {code}"
        ) from exc


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


# ---------------------------------------------------------------------------
# Differentiable training custom op.
#
# The forward custom op above is inference-only: it operates on already-packed
# NVFP4 operands and has no registered backward, so it can't carry gradients
# across the torch.compile boundary. This op wraps the FULL high-precision
# attention (internal pre-quant + native-NVFP4 flash forward) AND registers a
# native-NVFP4 backward via torch.library, so Inductor treats the whole attention
# as one opaque-but-DIFFERENTIABLE op: it compiles AROUND it (no attempt to trace
# tl.dot_scaled, so no InductorError + no silent eager fallback of the backward
# subgraph), while the registered backward computes the SAME native-NVFP4 grads as
# nvfp4_flash_attn_func.
#
# The forward returns ``out`` plus the per-row LSE as an auxiliary output. The LSE
# is marked non-differentiable in setup_context, so the public wrapper still
# exposes a single differentiable tensor while backward can reuse the forward
# softmax stats and skip the extra FA2-prep QK^T pass.
# ---------------------------------------------------------------------------
@torch.library.custom_op("axolotl_nvfp4::flash_attention_train", mutates_args=())
def _flash_attention_train_op(
    query: torch.Tensor,  # [Z, H, Sq, D] high precision
    key: torch.Tensor,  # [Z, Hk, Skv, D]
    value: torch.Tensor,  # [Z, Hk, Skv, D]
    key_pad_bias: torch.Tensor,  # [Z, Skv] fp32 or empty
    scaling: float,
    causal: bool,
    num_key_value_groups: int,
    sr: bool,
    backward_p_dv_sr: bool,
    backward_dot_dv_sr: bool,
    backward_ds_dq_sr: bool,
    dkdv_scratch_bf16: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del sr, backward_p_dv_sr, backward_dot_dv_sr, backward_ds_dq_sr
    del dkdv_scratch_bf16
    bias = None if key_pad_bias.numel() == 0 else key_pad_bias
    out, lse = nvfp4_flash_attention(
        query,
        key,
        value,
        scaling,
        causal=causal,
        num_key_value_groups=num_key_value_groups,
        key_pad_bias=bias,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
        return_lse=True,
    )
    return out, lse


@_flash_attention_train_op.register_fake
def _(
    query,
    key,
    value,
    key_pad_bias,
    scaling: float,
    causal: bool,
    num_key_value_groups: int,
    sr: bool,
    backward_p_dv_sr: bool,
    backward_dot_dv_sr: bool,
    backward_ds_dq_sr: bool,
    dkdv_scratch_bf16: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
):
    del key, value, key_pad_bias, scaling, causal, num_key_value_groups
    del sr, backward_p_dv_sr, backward_dot_dv_sr, backward_ds_dq_sr
    del dkdv_scratch_bf16, block_m, block_n, num_warps, num_stages
    z, h, s_q, d = query.shape
    return query.new_empty((z, h, s_q, d)), query.new_empty(
        (z * h, s_q), dtype=torch.float32
    )


def _flash_attention_train_setup_context(ctx, inputs, output):
    (
        query,
        key,
        value,
        key_pad_bias,
        scaling,
        causal,
        num_key_value_groups,
        sr,
        backward_p_dv_sr,
        backward_dot_dv_sr,
        backward_ds_dq_sr,
        dkdv_scratch_bf16,
        block_m,
        block_n,
        num_warps,
        num_stages,
    ) = inputs
    del num_key_value_groups
    out, lse = output
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(lse)
    ctx.save_for_backward(query, key, value, out, key_pad_bias, lse)
    ctx.scaling = scaling
    ctx.causal = causal
    ctx.sr = sr
    ctx.backward_p_dv_sr = backward_p_dv_sr
    ctx.backward_dot_dv_sr = backward_dot_dv_sr
    ctx.backward_ds_dq_sr = backward_ds_dq_sr
    ctx.dkdv_scratch_bf16 = dkdv_scratch_bf16
    ctx.tiles = (block_m, block_n, num_warps, num_stages)


# The backward must ALSO be an opaque custom op. register_autograd makes the bwd
# part of the autograd graph, but under compiled-autograd torch.compile traces the
# bwd too — and _run_bwd's raw Triton kernels hit FakeTensor .data_ptr()/.stride()
# accesses (the "Cannot access data pointer of FakeTensor" InductorError). Wrapping
# _run_bwd in its own custom op + fake makes the whole backward opaque to Inductor.
@torch.library.custom_op("axolotl_nvfp4::flash_attention_train_bwd", mutates_args=())
def _flash_attention_train_bwd_op(
    grad_out: torch.Tensor,  # [Z, H, Sq, D]
    query: torch.Tensor,  # [Z, H, Sq, D]
    key: torch.Tensor,  # [Z, Hk, Skv, D]
    value: torch.Tensor,  # [Z, Hk, Skv, D]
    out: torch.Tensor,  # [Z, H, Sq, D]
    key_pad_bias: torch.Tensor,  # [Z, Skv] fp32 or empty
    lse: torch.Tensor,  # [Z*H, Sq] fp32, forward softmax stats
    scaling: float,
    causal: bool,
    sr: bool,
    backward_p_dv_sr: bool,
    backward_dot_dv_sr: bool,
    backward_ds_dq_sr: bool,
    dkdv_scratch_bf16: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    bias = None if key_pad_bias.numel() == 0 else key_pad_bias.to(torch.float32)
    do = grad_out.reshape(z * h, s_q, d).contiguous()
    dq, dk, dv = _run_bwd(
        query.reshape(z * h, s_q, d).contiguous(),
        key.reshape(z * hk, s_kv, d).contiguous(),
        value.reshape(z * hk, s_kv, d).contiguous(),
        do,
        out.reshape(z * h, s_q, d).contiguous(),
        bias,
        z,
        h,
        hk,
        s_q,
        s_kv,
        d,
        scaling,
        causal,
        sr,
        block_m,
        block_n,
        num_warps,
        num_stages,
        lse=lse.reshape(z * h, s_q).contiguous(),
        sr_p_dv=backward_p_dv_sr,
        sr_dot_dv=backward_dot_dv_sr,
        sr_ds_dq=backward_ds_dq_sr,
        dkdv_scratch_bf16=dkdv_scratch_bf16,
    )
    dq = dq.reshape(z, h, s_q, d).to(grad_out.dtype).contiguous()
    ng = h // hk
    if ng > 1:
        dk, dv = _gqa_reduce_cast_dkdv(dk, dv, z, h, hk, s_kv, d, grad_out.dtype)
    else:
        dk = dk.reshape(z, hk, s_kv, d).to(grad_out.dtype)
        dv = dv.reshape(z, hk, s_kv, d).to(grad_out.dtype)
    return dq.contiguous(), dk.contiguous(), dv.contiguous()


@_flash_attention_train_bwd_op.register_fake
def _(
    grad_out,
    query,
    key,
    value,
    out,
    key_pad_bias,
    lse,
    scaling: float,
    causal: bool,
    sr: bool,
    backward_p_dv_sr: bool,
    backward_dot_dv_sr: bool,
    backward_ds_dq_sr: bool,
    dkdv_scratch_bf16: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
):
    del grad_out, out, key_pad_bias, lse, scaling, causal, sr
    del backward_p_dv_sr, backward_dot_dv_sr, backward_ds_dq_sr, dkdv_scratch_bf16
    del block_m, block_n, num_warps, num_stages
    # The real op returns CONTIGUOUS dq/dk/dv (reshape / GQA-reduce empty), even when
    # the q/k/v inputs are non-contiguous roped views — the fake must match strides.
    dq = query.new_empty(query.shape)
    dk = key.new_empty(key.shape)
    dv = value.new_empty(value.shape)
    return dq, dk, dv


def _flash_attention_train_backward(ctx, grad_out, grad_lse):
    del grad_lse
    query, key, value, out, key_pad_bias, lse = ctx.saved_tensors
    block_m, block_n, num_warps, num_stages = ctx.tiles
    dq, dk, dv = torch.ops.axolotl_nvfp4.flash_attention_train_bwd(
        grad_out.contiguous(),
        query,
        key,
        value,
        out,
        key_pad_bias,
        lse,
        ctx.scaling,
        ctx.causal,
        ctx.sr,
        ctx.backward_p_dv_sr,
        ctx.backward_dot_dv_sr,
        ctx.backward_ds_dq_sr,
        ctx.dkdv_scratch_bf16,
        block_m,
        block_n,
        num_warps,
        num_stages,
    )
    # one grad slot per forward input (16 inputs)
    return (
        dq,
        dk,
        dv,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "axolotl_nvfp4::flash_attention_train",
    _flash_attention_train_backward,
    setup_context=_flash_attention_train_setup_context,
)


def nvfp4_flash_attn_train_custom_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    *,
    causal: bool = False,
    num_key_value_groups: int = 1,
    key_pad_bias: torch.Tensor | None = None,
    stochastic_rounding: bool = True,
    backward_p_dv_stochastic_rounding: bool | None = None,
    backward_dot_dv_stochastic_rounding: bool | None = None,
    backward_ds_dq_stochastic_rounding: bool | None = None,
    dkdv_scratch_bf16: bool = False,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """Differentiable native-NVFP4 attention as an opaque torch.compile custom op.

    Same forward/backward math as ``nvfp4_flash_attn_func`` but wrapped so Inductor
    compiles around it (no ``tl.dot_scaled`` trace, no eager fallback of the bwd
    subgraph). All SR knobs are op arguments (so they survive compile/serialization);
    the bwd reuses the forward LSE but recomputes the FP4 packs (no saved-pack reuse
    across the opaque boundary).
    """
    bias = (
        query.new_empty((0,), dtype=torch.float32)
        if key_pad_bias is None
        else key_pad_bias
    )
    sr = stochastic_rounding
    p_dv = (
        sr
        if backward_p_dv_stochastic_rounding is None
        else backward_p_dv_stochastic_rounding
    )
    dot_dv = (
        sr
        if backward_dot_dv_stochastic_rounding is None
        else backward_dot_dv_stochastic_rounding
    )
    ds_dq = (
        sr
        if backward_ds_dq_stochastic_rounding is None
        else backward_ds_dq_stochastic_rounding
    )
    out, _lse = torch.ops.axolotl_nvfp4.flash_attention_train(
        query,
        key,
        value,
        bias,
        scaling,
        causal,
        num_key_value_groups,
        sr,
        p_dv,
        dot_dv,
        ds_dq,
        dkdv_scratch_bf16,
        block_m,
        block_n,
        num_warps,
        num_stages,
    )
    return out
