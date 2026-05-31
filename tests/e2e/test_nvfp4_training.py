"""Tests for NVFP4-GEMM training (real FP4 forward + backward).

Gated on Blackwell (sm >= 10.0) since the FP4 tensor-core GEMM has no CPU/
non-Blackwell path.
"""

import pytest
import torch
from torch import nn

from tests.e2e.utils import require_torch_2_8_0, requires_sm_ge_100


@require_torch_2_8_0
@requires_sm_ge_100
class TestNVFP4Training:
    def test_supported(self):
        from axolotl.utils.nvfp4_training import nvfp4_supported

        ok, reason = nvfp4_supported()
        assert ok, reason

    def test_linear_forward_backward_finite(self):
        """Module runs with a non-32-aligned token count (padding path)."""
        from axolotl.utils.nvfp4_training import NVFP4Linear, NVFP4Recipe

        torch.manual_seed(0)
        linear = nn.Linear(512, 256, bias=True).cuda().bfloat16()
        mod = NVFP4Linear.from_linear(linear, NVFP4Recipe())
        x = torch.randn(
            37, 7, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )  # 259 tokens, not a multiple of 32
        out = mod(x)
        assert out.shape == (37, 7, 256)
        out.sum().backward()
        for t in (out, x.grad, mod.weight.grad, mod.bias.grad):
            assert torch.isfinite(t).all().item()
        assert x.grad.shape == x.shape

    def test_learns(self):
        """Loss decreases when fitting a teacher weight through FP4 GEMMs."""
        from axolotl.utils.nvfp4_training import NVFP4Linear, NVFP4Recipe

        torch.manual_seed(1)
        w_star = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 0.05
        linear = nn.Linear(512, 256, bias=False).cuda().bfloat16()
        with torch.no_grad():
            linear.weight.mul_(0.02)
        mod = NVFP4Linear.from_linear(linear, NVFP4Recipe())
        opt = torch.optim.SGD(mod.parameters(), lr=0.5)
        x = torch.randn(259, 512, device="cuda", dtype=torch.bfloat16)
        tgt = (x @ w_star.t()).detach()
        losses = []
        for _ in range(80):
            opt.zero_grad()
            loss = (mod(x) - tgt).float().pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        assert losses[-1] < 0.5 * losses[0]
        assert all(torch.isfinite(torch.tensor(losses)))

    def test_gemm_wiring_matches_dequant_matmul(self):
        """The FP4 GEMM must equal dequant(a) @ dequant(b)."""
        from axolotl.utils.nvfp4_training import _fp4_mm, QuantPolicy, _quantize

        torch.manual_seed(0)
        a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
        got = _fp4_mm(a, b, QuantPolicy(), QuantPolicy())
        a_q = _quantize(a, QuantPolicy())
        b_q = _quantize(b.t().contiguous(), QuantPolicy()).t()
        ref = a_q.dequantize(torch.bfloat16) @ b_q.dequantize(torch.bfloat16)
        rel = ((got - ref).norm() / ref.norm()).item()
        assert rel < 2e-2, f"GEMM wiring rel-err {rel}"

    def test_convert_excludes_sensitive_and_odd_dims(self):
        """Swap eligible linears; keep lm_head/embeddings high-precision."""
        from transformers import AutoModelForCausalLM

        from axolotl.utils.nvfp4_training import NVFP4Linear, convert_to_nvfp4_training

        model = AutoModelForCausalLM.from_pretrained(
            "axolotl-ai-co/tiny-qwen2-129m", device_map="auto", dtype=torch.bfloat16
        )
        n = convert_to_nvfp4_training(model)
        assert n > 0
        assert not isinstance(model.lm_head, NVFP4Linear)
        ids = torch.randint(0, 1000, (2, 48), device=model.device)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        assert torch.isfinite(out.loss).item()

    def test_compile_no_graph_breaks(self):
        from axolotl.utils.nvfp4_training import NVFP4Linear, NVFP4Recipe

        import torch._dynamo as dyn

        torch.manual_seed(0)
        linear = nn.Linear(512, 256).cuda().bfloat16()
        # recipe on: SR + RHT live at the quant boundary, must not break the graph
        mod = NVFP4Linear.from_linear(
            linear, NVFP4Recipe(stochastic_rounding=True, hadamard=True)
        )
        x = torch.randn(
            128, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        dyn.reset()
        explanation = dyn.explain(lambda z: mod(z).sum())(x)
        assert explanation.graph_break_count == 0

    def test_sr_unbiased(self):
        """Averaging stochastic-rounded quant converges to the true value; RTN
        keeps a systematic bias."""
        from axolotl.utils.nvfp4_training import QuantPolicy, _quantize

        torch.manual_seed(0)
        t = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16) * 0.01
        true = t.float()
        rtn = _quantize(t, QuantPolicy()).dequantize(torch.float32)
        acc = torch.zeros_like(true)
        n = 3000
        for _ in range(n):
            acc += _quantize(t, QuantPolicy(stochastic=True)).dequantize(torch.float32)
        sr_mean = acc / n
        rtn_bias = (rtn - true).abs().mean().item()
        sr_bias = (sr_mean - true).abs().mean().item()
        # SR is unbiased: its averaged error collapses far below RTN's fixed bias
        assert sr_bias < 0.25 * rtn_bias, (rtn_bias, sr_bias)

    def test_rht_cancels(self):
        """With FP4 quant bypassed, the Hadamard rotation applied to both wgrad
        operands cancels in the product (orthonormal H^T H = I)."""
        from axolotl.utils.nvfp4_training import _apply_rht

        torch.manual_seed(1)
        g = torch.randn(32, 64, device="cuda", dtype=torch.float32)  # [N,M]
        x = torch.randn(64, 48, device="cuda", dtype=torch.float32)  # [M,K]
        ref = g @ x
        g_r = _apply_rht(g)  # rotate along contraction M (last dim)
        x_r = _apply_rht(x.t().contiguous()).t()  # rotate along M
        rel = ((g_r @ x_r) - ref).norm() / ref.norm()
        assert rel < 1e-4, rel

    def test_recipe_reduces_wgrad_noise(self):
        """RHT rotates contraction-dim outliers toward Gaussian, cutting the FP4
        wgrad quantization noise vs no-RHT."""
        from axolotl.utils.nvfp4_training import QuantPolicy, _fp4_mm, _pad_to_block

        torch.manual_seed(7)
        n, m, k = 128, 512, 256
        g = torch.randn(n, m, device="cuda", dtype=torch.bfloat16) * 0.01
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.01
        out_tok = torch.randperm(m)[:8]  # outliers concentrated on a few tokens (dim M)
        g[:, out_tok] *= 30.0
        x[out_tok, :] *= 30.0
        ref = g.float() @ x.float()
        gt, _ = _pad_to_block(g.contiguous(), 1)
        xp, _ = _pad_to_block(x.contiguous(), 0)

        def noise(hadamard):
            p = QuantPolicy(stochastic=True, hadamard=hadamard)
            vals = [
                ((_fp4_mm(gt, xp, p, p).float() - ref).norm() / ref.norm()).item()
                for _ in range(10)
            ]
            return sum(vals) / len(vals)

        assert noise(True) < noise(False)
