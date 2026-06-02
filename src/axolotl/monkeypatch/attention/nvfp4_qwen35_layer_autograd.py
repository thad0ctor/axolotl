"""Experimental Qwen3.5 layer-boundary autograd for native NVFP4 attention."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from axolotl.kernels.attn_nvfp4_flash import nvfp4_flash_attn_func


def _empty_bias_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(0)


def _bias_or_none(bias: torch.Tensor) -> torch.Tensor | None:
    return None if bias.numel() == 0 else bias


def _qwen3_5_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.type_as(x)


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
) -> torch.Tensor:
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

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

    query_roped, key_roped = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
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
    ).transpose(1, 2)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return F.linear(attn_output, o_weight, _bias_or_none(o_bias))


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
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
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
        ) = ctx.meta

        detached = [
            tensor.detach().requires_grad_(need_grad)
            for tensor, need_grad in zip(saved, ctx.needs_input_grad[:13])
        ]

        grad_targets: list[torch.Tensor] = []
        grad_target_indices: list[int] = []
        for idx, (tensor, need_grad) in enumerate(
            zip(detached, ctx.needs_input_grad[:13])
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
                )
                grads = torch.autograd.grad(
                    output,
                    grad_targets,
                    grad_output,
                    allow_unused=True,
                )

        result: list[torch.Tensor | None] = [None] * 22
        for idx, grad in zip(grad_target_indices, grads):
            result[idx] = grad
        return tuple(result)


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
    )
