"""Experimental Qwen3.5 layer-boundary autograd for native NVFP4 attention."""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from axolotl.kernels.attn_nvfp4_flash import (
    _gqa_reduce_cast_dkdv,
    _run_bwd,
    nvfp4_flash_attention,
    nvfp4_flash_attn_func,
)

_MANUAL_BWD_ENV = "AXOLOTL_NVFP4_QWEN35_LAYER_AUTOGRAD_MANUAL_BWD"


def _empty_bias_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(0)


def _bias_or_none(bias: torch.Tensor) -> torch.Tensor | None:
    return None if bias.numel() == 0 else bias


def _manual_backward_enabled() -> bool:
    return os.environ.get(_MANUAL_BWD_ENV, "").lower() in {"1", "true", "yes"}


def _qwen3_5_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def _rotary_backward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    grad_query_roped: torch.Tensor,
    grad_key_roped: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    need_cos: bool,
    need_sin: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    rotary_dim = cos_u.shape[-1]

    query_rot = query_states[..., :rotary_dim]
    key_rot = key_states[..., :rotary_dim]
    grad_query_rot, grad_query_pass = (
        grad_query_roped[..., :rotary_dim],
        grad_query_roped[..., rotary_dim:],
    )
    grad_key_rot, grad_key_pass = (
        grad_key_roped[..., :rotary_dim],
        grad_key_roped[..., rotary_dim:],
    )

    grad_query_states = grad_query_rot * cos_u - _rotate_half(grad_query_rot * sin_u)
    grad_key_states = grad_key_rot * cos_u - _rotate_half(grad_key_rot * sin_u)
    grad_query_states = torch.cat([grad_query_states, grad_query_pass], dim=-1)
    grad_key_states = torch.cat([grad_key_states, grad_key_pass], dim=-1)

    grad_cos = None
    if need_cos:
        grad_cos = (grad_query_rot * query_rot).sum(dim=1)
        grad_cos = grad_cos + (grad_key_rot * key_rot).sum(dim=1)
        grad_cos = grad_cos.to(cos.dtype)

    grad_sin = None
    if need_sin:
        grad_sin = (grad_query_rot * _rotate_half(query_rot)).sum(dim=1)
        grad_sin = grad_sin + (grad_key_rot * _rotate_half(key_rot)).sum(dim=1)
        grad_sin = grad_sin.to(sin.dtype)

    return grad_query_states, grad_key_states, grad_cos, grad_sin


def _qwen3_5_rms_norm_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    grad_output: torch.Tensor,
    need_x: bool,
    need_weight: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    x_float = x.float()
    grad_output = grad_output.float()
    inv_rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    normed = x_float * inv_rms

    grad_weight = None
    if need_weight:
        reduce_dims = tuple(range(grad_output.ndim - 1))
        grad_weight = (grad_output * normed).sum(dim=reduce_dims).to(weight.dtype)

    grad_x = None
    if need_x:
        grad_normed = grad_output * (1.0 + weight.float())
        inner = (grad_normed * x_float).mean(dim=-1, keepdim=True)
        grad_x = inv_rms * (grad_normed - x_float * inner * inv_rms.pow(2))
        grad_x = grad_x.to(x.dtype)

    return grad_x, grad_weight


def _linear_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
    need_x: bool,
    need_weight: bool,
    need_bias: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    x_2d = x.reshape(-1, x.shape[-1])
    grad_2d = grad_output.reshape(-1, grad_output.shape[-1])

    grad_x = None
    if need_x:
        grad_x = grad_2d.matmul(weight).reshape_as(x).to(x.dtype)

    grad_weight = None
    if need_weight:
        grad_weight = grad_2d.t().matmul(x_2d).to(weight.dtype)

    grad_bias = None
    if need_bias and bias.numel() != 0:
        grad_bias = grad_2d.sum(dim=0).to(bias.dtype)

    return grad_x, grad_weight, grad_bias


def _forward_impl(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    o_weight: torch.Tensor,
    o_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    head_dim: int,
    q_norm_eps: float,
    k_norm_eps: float,
    scaling: float,
    causal: bool,
    stochastic_rounding: bool,
    save_backward_packs: bool,
    backward_p_dv_stochastic_rounding: bool,
    backward_dot_dv_stochastic_rounding: bool,
    backward_ds_dq_stochastic_rounding: bool,
    dkdv_scratch_bf16: bool,
) -> torch.Tensor:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states, gate = torch.chunk(
        F.linear(hidden_states, q_weight, _bias_or_none(q_bias)).view(
            *input_shape, -1, head_dim * 2
        ),
        2,
        dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = _qwen3_5_rms_norm(
        query_states.view(hidden_shape), q_norm_weight, q_norm_eps
    ).transpose(1, 2)
    key_states = _qwen3_5_rms_norm(
        F.linear(hidden_states, k_weight, _bias_or_none(k_bias)).view(hidden_shape),
        k_norm_weight,
        k_norm_eps,
    ).transpose(1, 2)
    value_states = (
        F.linear(hidden_states, v_weight, _bias_or_none(v_bias))
        .view(hidden_shape)
        .transpose(1, 2)
    )

    query_roped, key_roped = _apply_rotary_pos_emb(query_states, key_states, cos, sin)
    attn_output = nvfp4_flash_attn_func(
        query_roped,
        key_roped,
        value_states,
        scaling,
        causal=causal,
        num_key_value_groups=query_states.shape[1] // key_states.shape[1],
        stochastic_rounding=stochastic_rounding,
        save_backward_packs=save_backward_packs,
        backward_p_dv_stochastic_rounding=backward_p_dv_stochastic_rounding,
        backward_dot_dv_stochastic_rounding=backward_dot_dv_stochastic_rounding,
        backward_ds_dq_stochastic_rounding=backward_ds_dq_stochastic_rounding,
        dkdv_scratch_bf16=dkdv_scratch_bf16,
    ).transpose(1, 2)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return F.linear(attn_output, o_weight, _bias_or_none(o_bias))


def _backward_recompute_autograd(ctx, grad_output: torch.Tensor):
    saved = ctx.saved_tensors
    (
        head_dim,
        q_norm_eps,
        k_norm_eps,
        scaling,
        causal,
        stochastic_rounding,
        save_backward_packs,
        backward_p_dv_stochastic_rounding,
        backward_dot_dv_stochastic_rounding,
        backward_ds_dq_stochastic_rounding,
        dkdv_scratch_bf16,
    ) = ctx.meta

    detached = [
        tensor.detach().requires_grad_(need_grad)
        for tensor, need_grad in zip(saved, ctx.needs_input_grad[:13], strict=False)
    ]

    grad_targets: list[torch.Tensor] = []
    grad_target_indices: list[int] = []
    for idx, (tensor, need_grad) in enumerate(
        zip(detached, ctx.needs_input_grad[:13], strict=False)
    ):
        if need_grad and tensor.numel() != 0:
            grad_targets.append(tensor)
            grad_target_indices.append(idx)

    devices: Iterable[torch.device] = ()
    if ctx.cuda_device is not None:
        devices = (ctx.cuda_device,)

    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.set_rng_state(ctx.cpu_rng_state)
        if ctx.cuda_rng_state is not None:
            torch.cuda.set_rng_state(ctx.cuda_rng_state, ctx.cuda_device)
        with torch.enable_grad():
            output = _forward_impl(
                *detached,
                head_dim,
                q_norm_eps,
                k_norm_eps,
                scaling,
                causal,
                stochastic_rounding,
                save_backward_packs,
                backward_p_dv_stochastic_rounding,
                backward_dot_dv_stochastic_rounding,
                backward_ds_dq_stochastic_rounding,
                dkdv_scratch_bf16,
            )
            grads = torch.autograd.grad(
                output,
                grad_targets,
                grad_output,
                allow_unused=True,
            )

    result: list[torch.Tensor | None] = [None] * 24
    for idx, grad in zip(grad_target_indices, grads, strict=False):
        result[idx] = grad
    return tuple(result)


def _backward_native_attention(ctx, grad_output: torch.Tensor):
    (
        hidden_states,
        cos,
        sin,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        o_bias,
        q_norm_weight,
        k_norm_weight,
    ) = ctx.saved_tensors
    (
        head_dim,
        q_norm_eps,
        k_norm_eps,
        scaling,
        causal,
        stochastic_rounding,
        save_backward_packs,
        backward_p_dv_stochastic_rounding,
        backward_dot_dv_stochastic_rounding,
        backward_ds_dq_stochastic_rounding,
        dkdv_scratch_bf16,
    ) = ctx.meta
    (
        need_hidden,
        need_cos,
        need_sin,
        need_q_weight,
        need_q_bias,
        need_k_weight,
        need_k_bias,
        need_v_weight,
        need_v_bias,
        need_o_weight,
        need_o_bias,
        need_q_norm_weight,
        need_k_norm_weight,
    ) = ctx.needs_input_grad[:13]

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_linear = F.linear(hidden_states, q_weight, _bias_or_none(q_bias))
    query_states, gate = torch.chunk(
        query_linear.view(*input_shape, -1, head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)

    query_norm_input = query_states.view(hidden_shape)
    key_linear = F.linear(hidden_states, k_weight, _bias_or_none(k_bias))
    key_norm_input = key_linear.view(hidden_shape)
    value_linear = F.linear(hidden_states, v_weight, _bias_or_none(v_bias))

    query_states = _qwen3_5_rms_norm(
        query_norm_input, q_norm_weight, q_norm_eps
    ).transpose(1, 2)
    key_states = _qwen3_5_rms_norm(key_norm_input, k_norm_weight, k_norm_eps).transpose(
        1, 2
    )
    value_states = value_linear.view(hidden_shape).transpose(1, 2)
    query_roped, key_roped = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    z, h, s_q, d = query_roped.shape
    hk = key_roped.shape[1]
    s_kv = key_roped.shape[2]
    num_key_value_groups = h // hk

    if save_backward_packs:
        attn_heads, lse, packs = nvfp4_flash_attention(
            query_roped,
            key_roped,
            value_states,
            scaling,
            causal=causal,
            num_key_value_groups=num_key_value_groups,
            return_lse=True,
            return_packs=True,
        )
        qnv, qsc, qtnv, qtsc, knv, ksc, vdnv, vdsc, ktnv, ktsc = packs
    else:
        attn_heads, lse = nvfp4_flash_attention(
            query_roped,
            key_roped,
            value_states,
            scaling,
            causal=causal,
            num_key_value_groups=num_key_value_groups,
            return_lse=True,
        )
        qnv = qsc = qtnv = qtsc = knv = ksc = vdnv = vdsc = ktnv = ktsc = None

    attn_flat = attn_heads.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    sigmoid_gate = torch.sigmoid(gate)
    gated_attn = attn_flat * sigmoid_gate

    need_pre_o = any(
        (
            need_hidden,
            need_cos,
            need_sin,
            need_q_weight,
            need_q_bias,
            need_k_weight,
            need_k_bias,
            need_v_weight,
            need_v_bias,
            need_q_norm_weight,
            need_k_norm_weight,
        )
    )
    grad_gated_attn, grad_o_weight, grad_o_bias = _linear_backward(
        gated_attn,
        o_weight,
        o_bias,
        grad_output,
        need_pre_o,
        need_o_weight,
        need_o_bias,
    )

    result: list[torch.Tensor | None] = [None] * 24
    result[9] = grad_o_weight
    result[10] = grad_o_bias
    if not need_pre_o:
        return tuple(result)

    grad_attn_flat = grad_gated_attn * sigmoid_gate
    grad_gate = grad_gated_attn * attn_flat * sigmoid_gate * (1.0 - sigmoid_gate)
    grad_attn = (
        grad_attn_flat.reshape(*input_shape, -1, head_dim).transpose(1, 2).contiguous()
    )

    query_2d = query_roped.reshape(z * h, s_q, d).contiguous()
    key_2d = key_roped.reshape(z * hk, s_kv, d).contiguous()
    value_2d = value_states.reshape(z * hk, s_kv, d).contiguous()
    grad_attn_2d = grad_attn.reshape(z * h, s_q, d).contiguous()
    attn_2d = attn_heads.reshape(z * h, s_q, d)
    grad_query, grad_key, grad_value = _run_bwd(
        query_2d,
        key_2d,
        value_2d,
        grad_attn_2d,
        attn_2d,
        None,
        z,
        h,
        hk,
        s_q,
        s_kv,
        d,
        scaling,
        causal,
        stochastic_rounding,
        64,
        128,
        4,
        1,
        lse=lse,
        sr_p_dv=backward_p_dv_stochastic_rounding,
        sr_dot_dv=backward_dot_dv_stochastic_rounding,
        sr_ds_dq=backward_ds_dq_stochastic_rounding,
        dkdv_scratch_bf16=dkdv_scratch_bf16,
        qnv_saved=qnv,
        qsc_saved=qsc,
        qtnv_saved=qtnv,
        qtsc_saved=qtsc,
        knv_saved=knv,
        ksc_saved=ksc,
        vnv_saved=vdnv,
        vsc_saved=vdsc,
        ktnv_saved=ktnv,
        ktsc_saved=ktsc,
    )
    grad_query = grad_query.reshape(z, h, s_q, d).to(query_roped.dtype)
    if num_key_value_groups > 1:
        grad_key, grad_value = _gqa_reduce_cast_dkdv(
            grad_key, grad_value, z, h, hk, s_kv, d, query_roped.dtype
        )
    else:
        grad_key = grad_key.reshape(z, hk, s_kv, d).to(key_roped.dtype)
        grad_value = grad_value.reshape(z, hk, s_kv, d).to(value_states.dtype)

    grad_query_states, grad_key_states, grad_cos, grad_sin = _rotary_backward(
        query_states,
        key_states,
        grad_query,
        grad_key,
        cos,
        sin,
        need_cos,
        need_sin,
    )
    result[1] = grad_cos
    result[2] = grad_sin

    need_q_linear = need_hidden or need_q_weight or need_q_bias
    need_k_linear = need_hidden or need_k_weight or need_k_bias
    grad_query_norm_input, grad_q_norm_weight = _qwen3_5_rms_norm_backward(
        query_norm_input,
        q_norm_weight,
        q_norm_eps,
        grad_query_states.transpose(1, 2),
        need_q_linear,
        need_q_norm_weight,
    )
    grad_key_norm_input, grad_k_norm_weight = _qwen3_5_rms_norm_backward(
        key_norm_input,
        k_norm_weight,
        k_norm_eps,
        grad_key_states.transpose(1, 2),
        need_k_linear,
        need_k_norm_weight,
    )
    result[11] = grad_q_norm_weight
    result[12] = grad_k_norm_weight

    grad_hidden = None
    if need_q_linear:
        grad_query_linear = torch.cat(
            [
                grad_query_norm_input,
                grad_gate.reshape(*input_shape, -1, head_dim),
            ],
            dim=-1,
        ).reshape(*input_shape, -1)
        grad_q_hidden, grad_q_weight, grad_q_bias = _linear_backward(
            hidden_states,
            q_weight,
            q_bias,
            grad_query_linear,
            need_hidden,
            need_q_weight,
            need_q_bias,
        )
        result[3] = grad_q_weight
        result[4] = grad_q_bias
        grad_hidden = grad_q_hidden

    if need_k_linear:
        grad_key_linear = grad_key_norm_input.reshape(*input_shape, -1)
        grad_k_hidden, grad_k_weight, grad_k_bias = _linear_backward(
            hidden_states,
            k_weight,
            k_bias,
            grad_key_linear,
            need_hidden,
            need_k_weight,
            need_k_bias,
        )
        result[5] = grad_k_weight
        result[6] = grad_k_bias
        if grad_hidden is None:
            grad_hidden = grad_k_hidden
        elif grad_k_hidden is not None:
            grad_hidden = grad_hidden + grad_k_hidden

    need_v_linear = need_hidden or need_v_weight or need_v_bias
    if need_v_linear:
        grad_value_linear = grad_value.transpose(1, 2).reshape(*input_shape, -1)
        grad_v_hidden, grad_v_weight, grad_v_bias = _linear_backward(
            hidden_states,
            v_weight,
            v_bias,
            grad_value_linear,
            need_hidden,
            need_v_weight,
            need_v_bias,
        )
        result[7] = grad_v_weight
        result[8] = grad_v_bias
        if grad_hidden is None:
            grad_hidden = grad_v_hidden
        elif grad_v_hidden is not None:
            grad_hidden = grad_hidden + grad_v_hidden

    result[0] = grad_hidden
    return tuple(result)


class _Qwen35NVFP4LayerAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_weight: torch.Tensor,
        q_bias: torch.Tensor,
        k_weight: torch.Tensor,
        k_bias: torch.Tensor,
        v_weight: torch.Tensor,
        v_bias: torch.Tensor,
        o_weight: torch.Tensor,
        o_bias: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        head_dim: int,
        q_norm_eps: float,
        k_norm_eps: float,
        scaling: float,
        causal: bool,
        stochastic_rounding: bool,
        save_backward_packs: bool,
        backward_p_dv_stochastic_rounding: bool,
        backward_dot_dv_stochastic_rounding: bool,
        backward_ds_dq_stochastic_rounding: bool,
        dkdv_scratch_bf16: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            hidden_states,
            cos,
            sin,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            o_weight,
            o_bias,
            q_norm_weight,
            k_norm_weight,
        )
        ctx.meta = (
            head_dim,
            q_norm_eps,
            k_norm_eps,
            scaling,
            causal,
            stochastic_rounding,
            save_backward_packs,
            backward_p_dv_stochastic_rounding,
            backward_dot_dv_stochastic_rounding,
            backward_ds_dq_stochastic_rounding,
            dkdv_scratch_bf16,
        )
        ctx.cpu_rng_state = torch.get_rng_state()
        if hidden_states.is_cuda:
            ctx.cuda_device = hidden_states.device
            ctx.cuda_rng_state = torch.cuda.get_rng_state(hidden_states.device)
        else:
            ctx.cuda_device = None
            ctx.cuda_rng_state = None
        return _forward_impl(
            hidden_states,
            cos,
            sin,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            o_weight,
            o_bias,
            q_norm_weight,
            k_norm_weight,
            head_dim,
            q_norm_eps,
            k_norm_eps,
            scaling,
            causal,
            stochastic_rounding,
            save_backward_packs,
            backward_p_dv_stochastic_rounding,
            backward_dot_dv_stochastic_rounding,
            backward_ds_dq_stochastic_rounding,
            dkdv_scratch_bf16,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        devices: Iterable[torch.device] = ()
        if ctx.cuda_device is not None:
            devices = (ctx.cuda_device,)

        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.set_rng_state(ctx.cpu_rng_state)
            if ctx.cuda_rng_state is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng_state, ctx.cuda_device)
            if _manual_backward_enabled():
                return _backward_native_attention(ctx, grad_output)
            return _backward_recompute_autograd(ctx, grad_output)


def supports_qwen35_layer_autograd(module: nn.Module) -> bool:
    linears = (module.q_proj, module.k_proj, module.v_proj, module.o_proj)
    if any(type(linear) is not nn.Linear for linear in linears):
        return False
    if module.training and float(getattr(module, "attention_dropout", 0.0)) != 0.0:
        return False
    return True


def qwen35_nvfp4_layer_attention(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    causal: bool,
) -> torch.Tensor:
    cos, sin = position_embeddings
    return _Qwen35NVFP4LayerAttention.apply(
        hidden_states,
        cos,
        sin,
        module.q_proj.weight,
        (
            module.q_proj.bias
            if module.q_proj.bias is not None
            else _empty_bias_like(hidden_states)
        ),
        module.k_proj.weight,
        (
            module.k_proj.bias
            if module.k_proj.bias is not None
            else _empty_bias_like(hidden_states)
        ),
        module.v_proj.weight,
        (
            module.v_proj.bias
            if module.v_proj.bias is not None
            else _empty_bias_like(hidden_states)
        ),
        module.o_proj.weight,
        (
            module.o_proj.bias
            if module.o_proj.bias is not None
            else _empty_bias_like(hidden_states)
        ),
        module.q_norm.weight,
        module.k_norm.weight,
        module.head_dim,
        float(module.q_norm.eps),
        float(module.k_norm.eps),
        float(module.scaling),
        causal,
        bool(getattr(module, "_nvfp4_stochastic_rounding", True)),
        bool(getattr(module, "_nvfp4_save_backward_packs", False)),
        (
            bool(getattr(module, "_nvfp4_stochastic_rounding", True))
            and not bool(getattr(module, "_nvfp4_backward_dv_p_rtn", False))
        ),
        (
            bool(getattr(module, "_nvfp4_stochastic_rounding", True))
            and not bool(getattr(module, "_nvfp4_backward_dv_dot_rtn", False))
        ),
        (
            bool(getattr(module, "_nvfp4_stochastic_rounding", True))
            and not bool(getattr(module, "_nvfp4_backward_dq_ds_rtn", False))
        ),
        bool(getattr(module, "_nvfp4_backward_dkdv_scratch_bf16", False)),
    )
