"""Model-level parity + prefill latency for FP4 q/k/o_proj (shared-pack).
Loads Qwen3.5-4B, patches full-attn layers, compares prefill logits + latency
with the proj-FP4 flag ON vs OFF (FP4-attention-only) vs unpatched bf16."""
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from axolotl.monkeypatch.attention.nvfp4_flash_attn import patch_qwen3_5_nvfp4_attention

dev = "cuda"
m = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
).to(dev).eval()
torch.manual_seed(0)
ids = torch.randint(0, 10000, (1, 1024), device=dev)


def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


with torch.no_grad():
    base = m(ids).logits.float()

n = patch_qwen3_5_nvfp4_attention(m, fuse_attn_proj=True)
ok = sum(int(getattr(mod, "_nvfp4_attn_proj_ok", False)) for mod in m.modules())
print(f"patched full-attn layers: {n}; proj-FP4-eligible: {ok}")


def set_flag(v):
    for mod in m.modules():
        if hasattr(mod, "_nvfp4_fuse_attn_proj"):
            mod._nvfp4_fuse_attn_proj = v


with torch.no_grad():
    set_flag(True);  on = m(ids).logits.float()
    set_flag(False); off = m(ids).logits.float()

print(f"logit cos  proj-FP4 ON  vs bf16-unpatched : {cos(on, base):.5f}")
print(f"logit cos  FP4-attn OFF vs bf16-unpatched : {cos(off, base):.5f}")
print(f"logit cos  ON vs OFF (marginal proj-FP4)  : {cos(on, off):.5f}")
print(f"argmax agree ON vs bf16: {(on.argmax(-1)==base.argmax(-1)).float().mean().item():.4f}")


def t(v, it=10):
    set_flag(v)
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(3): m(ids)
        torch.cuda.synchronize()
        a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
        for _ in range(it): m(ids)
        b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it


on_ms, off_ms = t(True), t(False)
print(f"prefill 1x1024: proj-FP4 ON {on_ms:.2f}ms  OFF {off_ms:.2f}ms  speedup {off_ms/on_ms:.3f}x")
