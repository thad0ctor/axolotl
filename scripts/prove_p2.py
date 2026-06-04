"""#2 fused_rope_quant_qk: parity (strided==contiguous, bit-identical) + the
saved-copy latency on a realistic transposed (non-contiguous) Q input."""
import torch
from axolotl.kernels.nvfp4_fused_producers import fused_rope_quant_qk

dev = "cuda"; dt = torch.bfloat16
torch.manual_seed(0)
for Z, H, S, D in [(1, 16, 2048, 256), (1, 16, 4096, 256)]:
    rot = D
    base = torch.randn(Z, S, H, D, device=dev, dtype=dt)  # [Z,S,H,D] contiguous
    x_t = base.transpose(1, 2)                              # [Z,H,S,D] non-contig (D contig)
    cos = torch.randn(Z, S, rot, device=dev, dtype=dt)
    sin = torch.randn(Z, S, rot, device=dev, dtype=dt)

    q_s, sc_s = fused_rope_quant_qk(x_t, cos, sin)              # strided (new, no copy)
    q_c, sc_c = fused_rope_quant_qk(x_t.contiguous(), cos, sin)  # contiguous reference
    ok = torch.equal(q_s, q_c) and torch.equal(sc_s.view(torch.uint8), sc_c.view(torch.uint8))

    def t(fn, it=50):
        torch.cuda.synchronize()
        for _ in range(5): fn()
        torch.cuda.synchronize()
        a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
        for _ in range(it): fn()
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / it
    new = t(lambda: fused_rope_quant_qk(x_t, cos, sin))
    old = t(lambda: fused_rope_quant_qk(x_t.contiguous(), cos, sin))
    print(f"S={S}: parity_bit_identical={ok}  new(strided) {new*1000:6.1f}us  "
          f"old(contig+kernel) {old*1000:6.1f}us  speedup {old/new:.2f}x")
