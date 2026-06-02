"""Native-NVFP4 acceleration for the Qwen3.5 dense SwiGLU MLP.

Qwen3.5 (2B/9B) uses a DENSE SwiGLU MLP (``Qwen3_5MLP``: gate_proj, up_proj,
down_proj; no MoE/router) on EVERY decoder layer — both full-attention and
linear-attention layers share the same MLP block. At the 9B dims the MLP is the
single largest whole-model FLOP sink: ``gate_proj`` and ``up_proj`` are each
4096->12288 and ``down_proj`` is 12288->4096, so the three GEMMs together are
~2.25x the linear-attn projection FLOP and hit all 32 layers. All three contract
over a large K (4096 / 12288) — the regime where native NVFP4 ``tl.dot_scaled``
beats bf16.

    down_proj( act_fn(gate_proj(x)) * up_proj(x) )

Novel shared-activation fusion: ``gate_proj`` and ``up_proj`` consume the SAME
``hidden_states`` and contract over the SAME K=hidden. The activation is
NVFP4-packed ONCE and reused for both GEMMs (one quant pass feeds two FP4
tensor-core ops), removing a redundant activation round-trip. ``down_proj`` takes
the SwiGLU(gate)*up product (a different activation, K=intermediate) and gets its
own NVFP4 path.

Forward/inference only (no autograd). Opt-in: call
``patch_qwen3_5_nvfp4_mlp(model)``; default OFF. Falls back to the stock forward
for grad/training and any K not divisible by 16.
"""

from __future__ import annotations

import torch
from torch import nn

from axolotl.kernels.attn_nvfp4_flash import _quant_nvfp4
from axolotl.kernels.nvfp4_linear import (
    nvfp4_linear,
    prepack_weight_nvfp4,
)
from axolotl.monkeypatch.attention.nvfp4_linear_attn import _gemm_from_packed_act
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def make_nvfp4_mlp_forward(orig_forward):
    """Patched ``Qwen3_5MLP.forward`` routing all three SwiGLU GEMMs through
    native NVFP4, with the gate/up input activation shared (quantized once).

    Falls back to ``orig_forward`` when grad is enabled (training) — the fast
    path is forward/inference only.
    """

    def forward(self, x):
        if torch.is_grad_enabled():
            return orig_forward(self, x)

        out_dtype = x.dtype
        k = x.shape[-1]
        lead = x.shape[:-1]
        x2d = x.reshape(-1, k)
        m = x2d.shape[0]

        # SHARED-ACTIVATION FUSION: pack hidden_states to NVFP4 ONCE, feed both
        # gate_proj and up_proj (same activation, same K=hidden contraction).
        anv, asc = _quant_nvfp4(x2d.unsqueeze(0))
        anv = anv[0]
        asc = asc[0]

        gate = _gemm_from_packed_act(
            anv, asc, self._gate_wnv, self._gate_wsc, m, self._inter, k, out_dtype
        )
        up = _gemm_from_packed_act(
            anv, asc, self._up_wnv, self._up_wsc, m, self._inter, k, out_dtype
        )

        inter = self.act_fn(gate) * up

        # down_proj: separate activation (post-SwiGLU), K=intermediate, own path.
        out = nvfp4_linear(inter, self._down_wnv, self._down_wsc, self._hidden)
        return out.reshape(*lead, self._hidden)

    return forward


def patch_qwen3_5_nvfp4_mlp(model: nn.Module) -> int:
    """Patch every Qwen3.5 dense SwiGLU MLP forward to route its three GEMMs
    through native NVFP4. Prepacks gate/up/down weights once per layer. Returns
    the count of patched MLP modules. Idempotent.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    patched = 0
    seen_forward = None
    for module in model.modules():
        if isinstance(module, Qwen3_5MLP):
            if getattr(module, "_nvfp4_patched", False):
                continue
            hidden = module.gate_proj.in_features
            inter = module.gate_proj.out_features
            if hidden % 16 != 0 or inter % 16 != 0:
                continue
            module._gate_wnv, module._gate_wsc = prepack_weight_nvfp4(
                module.gate_proj.weight
            )
            module._up_wnv, module._up_wsc = prepack_weight_nvfp4(
                module.up_proj.weight
            )
            module._down_wnv, module._down_wsc = prepack_weight_nvfp4(
                module.down_proj.weight
            )
            module._inter = inter
            module._hidden = hidden
            orig = type(module).forward
            if seen_forward is None:
                seen_forward = make_nvfp4_mlp_forward(orig)
            module.forward = seen_forward.__get__(module, type(module))
            module._nvfp4_patched = True
            patched += 1
    LOG.info("nvfp4 mlp: patched %d Qwen3.5 dense SwiGLU MLP modules", patched)
    return patched
