"""B1 parity probe: dump dq/dk/dv + peak mem for the saved-packs FP4 backward.

Run on BOTH the pre-B1 commit and the B1 commit (same args, SR OFF so the FP4
backward is deterministic), then diff the two dumps — B1 only removes dead HP q/k/v
saves, so grads must be BIT-IDENTICAL and peak memory should drop.

    PYTHONPATH=<worktree>/src python scripts/check_b1_parity.py /tmp/b1_<tag>.pt
"""
import sys
import torch

from axolotl.kernels.attn_nvfp4_flash import nvfp4_flash_attn_func

torch.manual_seed(0)
dev = "cuda"
Z, H, Hk, Sq, Skv, D = 1, 8, 2, 1024, 1024, 256  # GQA, D=256 (Qwen3.5 full-attn)
q = torch.randn(Z, H, Sq, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(Z, Hk, Skv, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(Z, Hk, Skv, D, device=dev, dtype=torch.bfloat16, requires_grad=True)

torch.cuda.reset_peak_memory_stats()
out = nvfp4_flash_attn_func(
    q, k, v, 1.0 / (D**0.5),
    causal=True, num_key_value_groups=H // Hk,
    stochastic_rounding=False,        # deterministic for cross-commit bit-compare
    save_backward_packs=True,
    dkdv_scratch_bf16=True,
)
g = torch.randn_like(out)
out.backward(g)
peak = torch.cuda.max_memory_allocated() / 2**30

path = sys.argv[1]
torch.save(
    {"dq": q.grad.cpu(), "dk": k.grad.cpu(), "dv": v.grad.cpu(), "peak_GiB": peak},
    path,
)
print(f"saved {path}  peak_GiB={peak:.4f}")
