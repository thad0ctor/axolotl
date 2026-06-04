"""#3 forward V-load hoist: dump forward output (for cross-worktree bit-compare)
+ forward prefill latency. Run under each worktree's PYTHONPATH."""
import sys, math, torch
from axolotl.kernels.attn_nvfp4_flash import nvfp4_flash_attention

dev = "cuda"; dt = torch.bfloat16
tag = sys.argv[1]
outs = {}
for Z, H, Hk, S, D in [(1, 16, 4, 2048, 256), (1, 16, 4, 4096, 256)]:
    torch.manual_seed(0)
    q = torch.randn(Z, H, S, D, device=dev, dtype=dt)
    k = torch.randn(Z, Hk, S, D, device=dev, dtype=dt)
    v = torch.randn(Z, Hk, S, D, device=dev, dtype=dt)
    sc = 1.0 / math.sqrt(D)
    fn = lambda: nvfp4_flash_attention(q, k, v, sc, causal=True, num_key_value_groups=H // Hk)
    out = fn()
    outs[S] = out.cpu()

    torch.cuda.synchronize()
    for _ in range(5): fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(50): fn()
    b.record(); torch.cuda.synchronize()
    print(f"S={S}: forward {a.elapsed_time(b)/50*1000:7.1f} us")
torch.save(outs, f"/tmp/p3_{tag}.pt")
print(f"saved /tmp/p3_{tag}.pt")
