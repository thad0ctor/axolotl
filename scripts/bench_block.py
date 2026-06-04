"""End-to-end prefill attention-block compute (producers Q/K + V + flash kernel),
timed. Q/K passed as transposed (non-contiguous) views — the production layout —
so #2's strided path is exercised. Dumps output for cross-worktree bit-compare."""
import sys, math, torch
from axolotl.kernels.nvfp4_fused_producers import fused_rope_quant_qk, quant_v_keyaxis
from axolotl.kernels.attn_nvfp4_flash import nvfp4_flash_attention_packed

dev = "cuda"; dt = torch.bfloat16
tag = sys.argv[1]
BLOCK_N = 128
outs = {}
for Z, H, Hk, S, D in [(1, 16, 4, 2048, 256), (1, 16, 4, 4096, 256)]:
    torch.manual_seed(0)
    rot = D; sc = 1.0 / math.sqrt(D)
    qb = torch.randn(Z, S, H, D, device=dev, dtype=dt)
    kb = torch.randn(Z, S, Hk, D, device=dev, dtype=dt)
    q_t = qb.transpose(1, 2)   # [Z,H,S,D] non-contig (prod layout)
    k_t = kb.transpose(1, 2)   # [Z,Hk,S,D]
    v = torch.randn(Z, Hk, S, D, device=dev, dtype=dt)
    cos = torch.randn(Z, S, rot, device=dev, dtype=dt)
    sin = torch.randn(Z, S, rot, device=dev, dtype=dt)

    def fn():
        qnv, qsc = fused_rope_quant_qk(q_t, cos, sin)
        knv, ksc = fused_rope_quant_qk(k_t, cos, sin)
        vnv, vsc, _ = quant_v_keyaxis(v, block_n=BLOCK_N)
        return nvfp4_flash_attention_packed(
            qnv, qsc, knv, ksc, vnv, vsc, z=Z, h=H, hk=Hk, s_q=S, s_kv=S, d=D,
            scaling=sc, out_dtype=dt, causal=True, block_n=BLOCK_N, out_layout="zshd")

    outs[S] = fn().cpu()
    torch.cuda.synchronize()
    for _ in range(5): fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(50): fn()
    b.record(); torch.cuda.synchronize()
    print(f"S={S}: block(producers+flash) {a.elapsed_time(b)/50*1000:7.1f} us")
torch.save(outs, f"/tmp/block_{tag}.pt")
print(f"saved /tmp/block_{tag}.pt")
