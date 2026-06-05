"""torch.compile custom-op boundary for native-NVFP4 attention."""

from __future__ import annotations

import torch

from axolotl.kernels.attn_nvfp4_flash import (
    _gqa_reduce_cast_dkdv,
    _next_mult,
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


def _empty_pack_outputs(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.new_empty((0,), dtype=torch.uint8) for _ in range(10))


def _static_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _use_saved_train_packs(requested: bool, s_q, s_kv) -> bool:
    if not requested:
        return False
    s_q_i = _static_int(s_q)
    s_kv_i = _static_int(s_kv)
    if s_q_i is None or s_kv_i is None:
        return False
    return max(s_q_i, s_kv_i) <= 4096


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
# The forward returns ``out`` plus non-differentiable auxiliary outputs: per-row
# LSE, and optionally the forward Q/K/V FP4 packs needed by backward. The public
# wrapper still exposes a single differentiable tensor while backward can reuse
# those tensors across the opaque boundary.
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
    save_backward_packs: bool,
    out_zshd: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del sr, backward_p_dv_sr, backward_dot_dv_sr, backward_ds_dq_sr
    del dkdv_scratch_bf16
    bias = None if key_pad_bias.numel() == 0 else key_pad_bias
    use_saved_packs = _use_saved_train_packs(
        save_backward_packs, query.shape[2], key.shape[2]
    )
    if use_saved_packs:
        out, lse, packs = nvfp4_flash_attention(
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
            return_packs=True,
            out_layout="zshd" if out_zshd else "zhsd",
        )
    else:
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
            out_layout="zshd" if out_zshd else "zhsd",
        )
        packs = _empty_pack_outputs(query)
    return (out, lse, *packs)


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
    save_backward_packs: bool,
    out_zshd: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
):
    del key_pad_bias, scaling, causal, num_key_value_groups
    del sr, backward_p_dv_sr, backward_dot_dv_sr, backward_ds_dq_sr
    del dkdv_scratch_bf16, num_warps, num_stages
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    out = query.new_empty((z, s_q, h, d) if out_zshd else (z, h, s_q, d))
    lse = query.new_empty((z * h, s_q), dtype=torch.float32)
    if not _use_saved_train_packs(save_backward_packs, s_q, s_kv):
        return (out, lse, *_empty_pack_outputs(query))

    bwd_block_m = min(block_m, 64)
    s_q_bwd_pad = _next_mult(s_q, max(bwd_block_m, 32))
    s_kv_bwd_pad = _next_mult(s_kv, 64)
    return (
        out,
        lse,
        query.new_empty((z * h, s_q, d // 2), dtype=torch.uint8),
        query.new_empty((z * h, s_q, d // 16), dtype=torch.uint8),
        query.new_empty((z * h, d, s_q_bwd_pad // 2), dtype=torch.uint8),
        query.new_empty((z * h, d, s_q_bwd_pad // 16), dtype=torch.uint8),
        key.new_empty((z * hk, s_kv, d // 2), dtype=torch.uint8),
        key.new_empty((z * hk, s_kv, d // 16), dtype=torch.uint8),
        value.new_empty((z * hk, s_kv, d // 2), dtype=torch.uint8),
        value.new_empty((z * hk, s_kv, d // 16), dtype=torch.uint8),
        key.new_empty((z * hk, d, s_kv_bwd_pad // 2), dtype=torch.uint8),
        key.new_empty((z * hk, d, s_kv_bwd_pad // 16), dtype=torch.uint8),
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
        save_backward_packs,
        out_zshd,
        block_m,
        block_n,
        num_warps,
        num_stages,
    ) = inputs
    del num_key_value_groups
    out, lse, qnv, qsc, qtnv, qtsc, knv, ksc, vdnv, vdsc, ktnv, ktsc = output
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(
        lse, qnv, qsc, qtnv, qtsc, knv, ksc, vdnv, vdsc, ktnv, ktsc
    )
    use_saved_packs = save_backward_packs and qnv.numel() > 0
    if use_saved_packs:
        query_save = key_save = value_save = out
    else:
        query_save, key_save, value_save = query, key, value
    ctx.save_for_backward(
        query_save,
        key_save,
        value_save,
        out,
        key_pad_bias,
        lse,
        qnv,
        qsc,
        qtnv,
        qtsc,
        knv,
        ksc,
        vdnv,
        vdsc,
        ktnv,
        ktsc,
    )
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    ctx.dims = (z, h, hk, s_q, s_kv, d)
    ctx.scaling = scaling
    ctx.causal = causal
    ctx.sr = sr
    ctx.backward_p_dv_sr = backward_p_dv_sr
    ctx.backward_dot_dv_sr = backward_dot_dv_sr
    ctx.backward_ds_dq_sr = backward_ds_dq_sr
    ctx.dkdv_scratch_bf16 = dkdv_scratch_bf16
    ctx.save_backward_packs = use_saved_packs
    ctx.out_zshd = bool(out_zshd)
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
    qnv: torch.Tensor,
    qsc: torch.Tensor,
    qtnv: torch.Tensor,
    qtsc: torch.Tensor,
    knv: torch.Tensor,
    ksc: torch.Tensor,
    vdnv: torch.Tensor,
    vdsc: torch.Tensor,
    ktnv: torch.Tensor,
    ktsc: torch.Tensor,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    save_backward_packs: bool,
    out_zshd: bool,
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
    bias = None if key_pad_bias.numel() == 0 else key_pad_bias.to(torch.float32)
    if out_zshd:
        do = grad_out
        o = out
    else:
        do = grad_out.reshape(z * h, s_q, d).contiguous()
        o = out.reshape(z * h, s_q, d).contiguous()
    if save_backward_packs:
        # HP q/k/v were intentionally not saved. _run_bwd does not dereference them
        # when LSE plus all deterministic forward packs are supplied; it only needs
        # a same-device tensor for scratch allocations and dummy bias pointers.
        query_bwd = key_bwd = value_bwd = o
    else:
        query_bwd = query.reshape(z * h, s_q, d).contiguous()
        key_bwd = key.reshape(z * hk, s_kv, d).contiguous()
        value_bwd = value.reshape(z * hk, s_kv, d).contiguous()
    dq, dk, dv = _run_bwd(
        query_bwd,
        key_bwd,
        value_bwd,
        do,
        o,
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
        qnv_saved=qnv if qnv.numel() else None,
        qsc_saved=qsc if qsc.numel() else None,
        qtnv_saved=qtnv if qtnv.numel() else None,
        qtsc_saved=qtsc if qtsc.numel() else None,
        knv_saved=knv if knv.numel() else None,
        ksc_saved=ksc if ksc.numel() else None,
        vnv_saved=vdnv if vdnv.numel() else None,
        vsc_saved=vdsc if vdsc.numel() else None,
        ktnv_saved=ktnv if ktnv.numel() else None,
        ktsc_saved=ktsc if ktsc.numel() else None,
        do_zshd=out_zshd,
        o_zshd=out_zshd,
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
    qnv,
    qsc,
    qtnv,
    qtsc,
    knv,
    ksc,
    vdnv,
    vdsc,
    ktnv,
    ktsc,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    save_backward_packs: bool,
    out_zshd: bool,
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
    del grad_out, out, key_pad_bias, lse
    del qnv, qsc, qtnv, qtsc, knv, ksc, vdnv, vdsc, ktnv, ktsc
    del scaling, causal, sr, save_backward_packs, out_zshd
    del backward_p_dv_sr, backward_dot_dv_sr, backward_ds_dq_sr, dkdv_scratch_bf16
    del block_m, block_n, num_warps, num_stages
    # The real op returns CONTIGUOUS dq/dk/dv (reshape / GQA-reduce empty), even when
    # the q/k/v inputs are non-contiguous roped views — the fake must match strides.
    dq = query.new_empty((z, h, s_q, d))
    dk = key.new_empty((z, hk, s_kv, d))
    dv = value.new_empty((z, hk, s_kv, d))
    return dq, dk, dv


def _flash_attention_train_backward(
    ctx,
    grad_out,
    grad_lse,
    grad_qnv,
    grad_qsc,
    grad_qtnv,
    grad_qtsc,
    grad_knv,
    grad_ksc,
    grad_vdnv,
    grad_vdsc,
    grad_ktnv,
    grad_ktsc,
):
    del grad_lse, grad_qnv, grad_qsc, grad_qtnv, grad_qtsc
    del grad_knv, grad_ksc, grad_vdnv, grad_vdsc, grad_ktnv, grad_ktsc
    (
        query,
        key,
        value,
        out,
        key_pad_bias,
        lse,
        qnv,
        qsc,
        qtnv,
        qtsc,
        knv,
        ksc,
        vdnv,
        vdsc,
        ktnv,
        ktsc,
    ) = ctx.saved_tensors
    z, h, hk, s_q, s_kv, d = ctx.dims
    block_m, block_n, num_warps, num_stages = ctx.tiles
    dq, dk, dv = torch.ops.axolotl_nvfp4.flash_attention_train_bwd(
        grad_out.contiguous(),
        query,
        key,
        value,
        out,
        key_pad_bias,
        lse,
        qnv,
        qsc,
        qtnv,
        qtsc,
        knv,
        ksc,
        vdnv,
        vdsc,
        ktnv,
        ktsc,
        z,
        h,
        hk,
        s_q,
        s_kv,
        d,
        ctx.save_backward_packs,
        ctx.out_zshd,
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
    # one grad slot per forward input (18 inputs)
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
    save_backward_packs: bool = False,
    out_layout: str = "zhsd",
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """Differentiable native-NVFP4 attention as an opaque torch.compile custom op.

    Same forward/backward math as ``nvfp4_flash_attn_func`` but wrapped so Inductor
    compiles around it (no ``tl.dot_scaled`` trace, no eager fallback of the bwd
    subgraph). All SR knobs are op arguments (so they survive compile/serialization);
    the bwd reuses the forward LSE and, when requested, the deterministic forward
    FP4 packs across the opaque boundary for short/mid static sequence lengths.
    ``out_layout="zshd"`` writes the HF attention layout directly.
    """
    if out_layout not in ("zhsd", "zshd"):
        raise ValueError("out_layout must be 'zhsd' or 'zshd'")
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
    out, *_aux = torch.ops.axolotl_nvfp4.flash_attention_train(
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
        save_backward_packs,
        out_layout == "zshd",
        block_m,
        block_n,
        num_warps,
        num_stages,
    )
    return out
