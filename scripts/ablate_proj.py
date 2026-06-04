import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from axolotl.monkeypatch.attention.nvfp4_flash_attn import patch_qwen3_5_nvfp4_attention
dev="cuda"
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(dev).eval()
torch.manual_seed(0); ids=torch.randint(0,10000,(1,1024),device=dev)
def cos(a,b): return F.cosine_similarity(a.flatten(),b.flatten(),dim=0).item()
with torch.no_grad(): base=m(ids).logits.float()
patch_qwen3_5_nvfp4_attention(m, fuse_attn_proj=True)
def setg(qk,o):
    for mod in m.modules():
        if hasattr(mod,"_nvfp4_fuse_attn_proj"): mod._nvfp4_fp4_qk=qk; mod._nvfp4_fp4_o=o
for name,qk,o in [("o-only",False,True),("qk-only",True,False),("all",True,True)]:
    setg(qk,o)
    with torch.no_grad(): lg=m(ids).logits.float()
    print(f"{name:8s}: logit cos {cos(lg,base):.5f}  argmax-agree {(lg.argmax(-1)==base.argmax(-1)).float().mean().item():.3f}")
