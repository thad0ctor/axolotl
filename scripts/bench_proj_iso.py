"""Isolated: does FP4 q/k/o_proj (shared-pack) beat bf16 at the real Qwen3.5-4B
dims? This is the deciding factor for the dense-model rationale (per-GEMM, where a
dense model would hit it every layer)."""
import torch
from transformers import AutoModelForCausalLM
from axolotl.kernels.attn_nvfp4_flash import _quant_nvfp4
from axolotl.kernels.nvfp4_linear import nvfp4_linear
from axolotl.monkeypatch.attention.nvfp4_linear_attn import (
    _gemm_from_packed_act, _get_packed_weight,
)

dev = "cuda"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16).to(dev).eval()
attn = None
for mod in m.modules():
    if all(hasattr(mod, p) for p in ("q_proj", "k_proj", "o_proj")) and type(mod.q_proj) is torch.nn.Linear:
        attn = mod; break
H = attn.q_proj.in_features
qO, kO, oI, oO = attn.q_proj.out_features, attn.k_proj.out_features, attn.o_proj.in_features, attn.o_proj.out_features
print(f"dims: hidden={H}  q_out={qO}  k_out={kO}  o_in={oI}  o_out={oO}")

M = 1024
x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
xo = torch.randn(M, oI, device=dev, dtype=torch.bfloat16)
qw = _get_packed_weight(attn, "_q", attn.q_proj)
kw = _get_packed_weight(attn, "_k", attn.k_proj)
ow = _get_packed_weight(attn, "_o", attn.o_proj)


@torch.no_grad()
def bf16():
    return attn.q_proj(x), attn.k_proj(x), attn.o_proj(xo)


@torch.no_grad()
def fp4():
    anv, asc = _quant_nvfp4(x.unsqueeze(0)); anv, asc = anv[0], asc[0]
    q = _gemm_from_packed_act(anv, asc, qw[0], qw[1], M, qO, H, torch.bfloat16)
    k = _gemm_from_packed_act(anv, asc, kw[0], kw[1], M, kO, H, torch.bfloat16)
    o = nvfp4_linear(xo, ow[0], ow[1], oO)
    return q, k, o


def t(fn, it=100):
    torch.cuda.synchronize()
    for _ in range(10): fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it


bf, fp = t(bf16), t(fp4)
print(f"q+k+o proj (M={M}): bf16 {bf*1000:6.1f}us   FP4-shared {fp*1000:6.1f}us   speedup {bf/fp:.2f}x")
