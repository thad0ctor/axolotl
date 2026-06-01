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

    def test_swap_frozen_lm_head_compute_and_storage(self):
        """A bare frozen lm_head swaps to the matching NVFP4 base and stays finite."""
        from axolotl.utils.nvfp4_training import (
            NVFP4ComputeBaseLinear,
            NVFP4FrozenBaseLinear,
            NVFP4Recipe,
            swap_frozen_linear_to_nvfp4,
        )

        torch.manual_seed(0)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(2048, 4096, bias=False)

        for mode, cls in (
            ("compute", NVFP4ComputeBaseLinear),
            ("storage", NVFP4FrozenBaseLinear),
        ):
            m = M().cuda().bfloat16()
            ok = swap_frozen_linear_to_nvfp4(
                m, "lm_head", NVFP4Recipe(), base_mode=mode
            )
            assert ok
            # mslk-fast variants are subclasses; both modes must not stay nn.Linear
            assert not isinstance(m.lm_head, nn.Linear)
            x = torch.randn(2, 16, 2048, device="cuda", dtype=torch.bfloat16)
            out = m.lm_head(x)
            assert out.shape == (2, 16, 4096)
            assert torch.isfinite(out).all().item()

    def test_swap_frozen_lm_head_skips_odd_dims(self):
        """A non-%32 output dim is left in high precision (no crash)."""
        from axolotl.utils.nvfp4_training import (
            NVFP4Recipe,
            swap_frozen_linear_to_nvfp4,
        )

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(2048, 4095, bias=False)  # 4095 % 32 != 0

        m = M().cuda().bfloat16()
        assert not swap_frozen_linear_to_nvfp4(m, "lm_head", NVFP4Recipe())
        assert isinstance(m.lm_head, nn.Linear)

    def test_quantize_lm_head_guard_tied_embeddings(self):
        """Tied + FROZEN now succeeds (quantize-once); tied + TRAINABLE still
        raises; untied removes lm_head from the exclusion as before."""
        from axolotl.loaders.patch_manager import PatchManager
        from axolotl.utils.dict import DictDefault

        class Cfg:
            tie_word_embeddings = False

        class TinyModel(nn.Module):
            def __init__(self, tie, frozen=True):
                super().__init__()
                self.config = Cfg()
                self.config.tie_word_embeddings = tie
                self.embed = nn.Embedding(64, 32)
                self.lm_head = nn.Linear(32, 64, bias=False)
                if tie:
                    self.lm_head.weight = self.embed.weight
                if frozen:
                    for p in self.parameters():
                        p.requires_grad_(False)

            def get_output_embeddings(self):
                return self.lm_head

            def get_input_embeddings(self):
                return self.embed

        pm = PatchManager.__new__(PatchManager)
        pm.cfg = DictDefault({"cut_cross_entropy": False})

        assert PatchManager._model_ties_embeddings(TinyModel(tie=True))
        assert not PatchManager._model_ties_embeddings(TinyModel(tie=False))

        # tied + frozen: no longer raises — the shared weight is quantized once
        # (the name-fragment exclusion is returned unchanged for the tied path).
        out = pm._nvfp4_unexclude_lm_head(
            TinyModel(tie=True, frozen=True), ["lm_head", "embed_tokens"]
        )
        assert out == ["lm_head", "embed_tokens"]

        # tied + trainable shared weight: still raises (FP4-storing it corrupts
        # training).
        with pytest.raises(RuntimeError, match="FROZEN"):
            pm._nvfp4_unexclude_lm_head(
                TinyModel(tie=True, frozen=False), ["lm_head", "embed_tokens"]
            )

        # untied: lm_head removed from exclusion, embed_tokens retained
        out = pm._nvfp4_unexclude_lm_head(
            TinyModel(tie=False), ["lm_head", "embed_tokens"]
        )
        assert out == ["embed_tokens"]

        # untied + cut_cross_entropy: raises
        pm_cce = PatchManager.__new__(PatchManager)
        pm_cce.cfg = DictDefault({"cut_cross_entropy": True})
        with pytest.raises(RuntimeError, match="cut_cross_entropy"):
            pm_cce._nvfp4_unexclude_lm_head(
                TinyModel(tie=False), ["lm_head", "embed_tokens"]
            )

    def test_embedding_quant_forward_and_memory(self):
        """NVFP4Embedding lookup matches bf16 F.embedding within FP4 tolerance and
        the weight memory drops ~3.5x."""
        import torch.nn.functional as F

        from axolotl.utils.nvfp4_training import NVFP4Embedding

        torch.manual_seed(0)
        emb = nn.Embedding(2048, 512).cuda().bfloat16()
        emb.weight.requires_grad_(False)
        ne = NVFP4Embedding.from_embedding(emb)
        idx = torch.randint(0, 2048, (4, 32), device="cuda")
        ref = F.embedding(idx, emb.weight)
        got = ne(idx)
        cos = F.cosine_similarity(
            ref.flatten().float(), got.flatten().float(), dim=0
        ).item()
        assert cos > 0.99, cos

        bf16_bytes = emb.weight.numel() * 2
        fp4_bytes = (
            ne.w_q.qdata.numel()
            + ne.w_q.scale.numel()
            + ne.w_q.per_tensor_scale.numel() * 4
        )
        assert bf16_bytes / fp4_bytes > 3.0, bf16_bytes / fp4_bytes

    def test_embedding_quant_skips_trainable(self):
        """A trainable embedding is left in high precision (no FP4 master)."""
        from axolotl.utils.nvfp4_training import (
            NVFP4Embedding,
            swap_frozen_embedding_to_nvfp4,
        )

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(256, 64)

        m = M().cuda().bfloat16()  # weight trainable by default
        assert swap_frozen_embedding_to_nvfp4(m, "embed") is None
        assert isinstance(m.embed, nn.Embedding)
        assert not isinstance(m.embed, NVFP4Embedding)

    def test_tied_quantize_once_shared_store(self):
        """Tied frozen weight: the embedding lookup and the lm_head GEMM read the
        SAME dequantized FP4 store (quantize-once)."""
        from axolotl.utils.nvfp4_training import (
            NVFP4Embedding,
            NVFP4Recipe,
            NVFP4TiedLMHead,
            swap_tied_embedding_and_lm_head_to_nvfp4,
        )

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(2048, 512)
                self.lm_head = nn.Linear(512, 2048, bias=False)
                self.lm_head.weight = self.embed.weight
                for p in self.parameters():
                    p.requires_grad_(False)

            def get_input_embeddings(self):
                return self.embed

            def get_output_embeddings(self):
                return self.lm_head

        m = M().cuda().bfloat16()
        ok = swap_tied_embedding_and_lm_head_to_nvfp4(
            m, "embed", "lm_head", NVFP4Recipe()
        )
        assert ok
        assert isinstance(m.embed, NVFP4Embedding)
        assert isinstance(m.lm_head, NVFP4TiedLMHead)
        # one store, two roles
        assert m.lm_head.w_q is m.embed.w_q
        assert torch.equal(m.embed.weight, m.lm_head.weight)

        x = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16)
        logits = m.lm_head(x)
        ref = x @ m.embed.weight.t()  # same dequantized weight as the lookup
        rel = ((logits - ref).norm() / ref.norm()).item()
        assert rel < 0.2 and torch.isfinite(logits).all().item(), rel

    def test_vision_tower_targets_only_vision_linears(self):
        """The vision-tower swap touches ONLY eligible frozen linears under the
        vision module — merger/patch-embed/non-%32 dims and the language head are
        left untouched."""
        from axolotl.utils.nvfp4_training import (
            NVFP4Recipe,
            _find_vision_tower,
            convert_vision_tower_to_nvfp4,
            is_nvfp4_base,
        )

        class VisionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Module()
                self.attn.qkv = nn.Linear(1152, 3456, bias=True)
                self.attn.proj = nn.Linear(1152, 1152)
                self.mlp = nn.Module()
                self.mlp.fc1 = nn.Linear(1152, 4304)  # 4304 not %32 -> skip
                self.mlp.fc2 = nn.Linear(4304, 1152)

        class VisionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Module()
                self.patch_embed.proj = nn.Conv2d(3, 1152, 16, 16)
                self.blocks = nn.ModuleList([VisionBlock() for _ in range(2)])
                self.merger = nn.Module()
                self.merger.linear_fc1 = nn.Linear(1152, 2048)

        class FakeVL(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.visual = VisionModel()
                self.lm_head = nn.Linear(2048, 1024)

        m = FakeVL().cuda().bfloat16()
        for p in m.parameters():
            p.requires_grad_(False)

        name, vt = _find_vision_tower(m)
        assert name == "model.visual"
        n = convert_vision_tower_to_nvfp4(m, NVFP4Recipe(), base_mode="storage")
        # qkv + proj per block (fc1/fc2 skipped for %32), 2 blocks => 4
        assert n == 4, n
        assert is_nvfp4_base(m.model.visual.blocks[0].attn.qkv)
        assert is_nvfp4_base(m.model.visual.blocks[0].attn.proj)
        assert isinstance(m.model.visual.blocks[0].mlp.fc1, nn.Linear)
        assert isinstance(m.model.visual.merger.linear_fc1, nn.Linear)
        assert isinstance(m.lm_head, nn.Linear)

    def test_vision_tower_warns_on_text_model(self):
        """No vision tower => no swap, count 0 (text-only model)."""
        from axolotl.utils.nvfp4_training import (
            NVFP4Recipe,
            convert_vision_tower_to_nvfp4,
        )

        class TextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Linear(64, 64)])
                self.lm_head = nn.Linear(64, 64)

        m = TextModel().cuda().bfloat16()
        assert convert_vision_tower_to_nvfp4(m, NVFP4Recipe()) == 0
        assert isinstance(m.lm_head, nn.Linear)

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
        # fullgraph fwd+bwd must trace cleanly (RHT's Hadamard build must not
        # reach a torch.Generator inside the backward graph)
        compiled = torch.compile(lambda z: mod(z).sum(), fullgraph=True)
        compiled(x).backward()
        assert torch.isfinite(mod.weight.grad).all().item()

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


@require_torch_2_8_0
@requires_sm_ge_100
class TestNVFP4Adapters:
    """NVFP4 base under LoRA adapters: FP4 compute, and FP4 storage (QLoRA)."""

    def _lora_model(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "axolotl-ai-co/tiny-qwen2-129m", device_map="auto", dtype=torch.bfloat16
        )
        lc = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lc)

    def _train(self, model, steps=10):
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        ids = torch.randint(0, 1000, (2, 64), device=model.device)
        first = None
        for _ in range(steps):
            opt.zero_grad()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            opt.step()
            first = first if first is not None else float(out.loss)
        return first, float(out.loss)

    def test_lora_fp4_compute(self):
        """base_layer -> NVFP4Linear (frozen HP weight), adapters train."""
        from axolotl.utils.nvfp4_training import (
            NVFP4Linear,
            NVFP4Recipe,
            convert_lora_base_to_nvfp4,
        )

        model = self._lora_model()
        n = convert_lora_base_to_nvfp4(model, NVFP4Recipe(), quantized_storage=False)
        assert n > 0
        assert sum(1 for m in model.modules() if isinstance(m, NVFP4Linear)) == n
        l0, l1 = self._train(model)
        assert l1 < l0 and torch.isfinite(torch.tensor(l1))

    def test_lora_fp4_compute_base_prequant(self):
        """compute_base: frozen base pre-quantized into 2 FP4 layouts; fprop
        matches per-step requant bit-exactly, base quant prologue paid once."""
        from torch import nn

        from axolotl.utils.nvfp4_training import (
            NVFP4ComputeBaseLinear,
            NVFP4Linear,
            NVFP4Recipe,
            is_nvfp4_base,
            convert_lora_base_to_nvfp4,
        )

        # bit-identical to the per-step path (same quantization, just cached)
        lin = nn.Linear(512, 768, bias=False).cuda().bfloat16()
        lin.weight.requires_grad_(False)
        x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
        oc = NVFP4ComputeBaseLinear.from_linear(lin, NVFP4Recipe())(x)
        op = NVFP4Linear.from_linear(lin, NVFP4Recipe())(x)
        assert torch.equal(oc, op)

        # two FP4 layouts are smaller than the bf16 weight
        cb = NVFP4ComputeBaseLinear.from_linear(lin, NVFP4Recipe())
        fp4 = (
            cb.w_fprop.qdata.numel()
            + cb.w_fprop.scale.numel()
            + cb.w_dgrad.qdata.numel()
            + cb.w_dgrad.scale.numel()
        )
        assert (512 * 768 * 2) / fp4 > 1.5

        model = self._lora_model()
        n = convert_lora_base_to_nvfp4(model, NVFP4Recipe(), compute_base=True)
        assert n > 0
        assert sum(1 for m in model.modules() if is_nvfp4_base(m)) == n
        l0, l1 = self._train(model)
        assert l1 < l0 and torch.isfinite(torch.tensor(l1))

    def test_qlora_fp4_storage(self):
        """base_layer -> NVFP4FrozenBaseLinear: weight stored in FP4 (memory win)."""
        from axolotl.utils.nvfp4_training import (
            NVFP4Recipe,
            convert_lora_base_to_nvfp4,
            is_nvfp4_base,
        )

        model = self._lora_model()
        n = convert_lora_base_to_nvfp4(model, NVFP4Recipe(), quantized_storage=True)
        assert n > 0
        bases = [m for m in model.modules() if is_nvfp4_base(m)]
        assert len(bases) == n
        # FP4 storage is materially smaller than the bf16 weight it replaced
        b = bases[0]
        if hasattr(b, "w_q"):  # torchao base
            fp4_bytes = b.w_q.qdata.numel() + b.w_q.scale.numel()
        else:  # mslk-fast base: packed qdata + e4m3 block scales
            fp4_bytes = b.wq.numel() + b.wsc.numel()
        bf16_bytes = b.in_features * b.out_features * 2
        assert bf16_bytes / fp4_bytes > 3.0
        l0, l1 = self._train(model)
        assert l1 < l0 and torch.isfinite(torch.tensor(l1))

    def test_fsdp_hooks_present_and_reconstruct(self):
        """The FSDP-wrapped FP4 base carries all-gather hooks and reconstructs
        bit-exactly from row-sharded qdata/scale (the shard the hooks gather)."""
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            NVFP4Tensor,
            per_tensor_amax_to_scale,
        )

        from axolotl.utils.nvfp4_training import _to_fsdp_nvfp4

        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 0.05
        wq = NVFP4Tensor.to_nvfp4(
            w.contiguous(),
            block_size=16,
            per_tensor_scale=per_tensor_amax_to_scale(torch.max(torch.abs(w))),
        )
        fq = _to_fsdp_nvfp4(wq)
        assert hasattr(fq, "fsdp_pre_all_gather")
        assert hasattr(fq, "fsdp_post_all_gather")
        assert torch.equal(
            fq.dequantize(torch.bfloat16), wq.dequantize(torch.bfloat16)
        )

        # simulate the gather: split qdata/scale by row, concat, reconstruct
        (qd, sc), (ctx, pts) = fq.fsdp_pre_all_gather(mesh=None)
        half = qd.shape[0] // 2
        gathered = (
            torch.cat([qd[:half], qd[half:]], dim=0),
            torch.cat([sc[:half], sc[half:]], dim=0),
        )
        rebuilt, _ = fq.fsdp_post_all_gather(gathered, (ctx, pts), torch.bfloat16)
        rel = (
            rebuilt.dequantize(torch.bfloat16) - wq.dequantize(torch.bfloat16)
        ).norm() / wq.dequantize(torch.bfloat16).norm()
        assert rel.item() < 1e-6

    def test_qlora_checkpoint_roundtrip(self):
        """The FP4-packed base must survive save/load (buffer in state_dict),
        else QLoRA resume silently reinitializes the base."""
        import io

        from axolotl.utils.nvfp4_training import (
            NVFP4Recipe,
            convert_lora_base_to_nvfp4,
        )

        model = self._lora_model()
        convert_lora_base_to_nvfp4(model, NVFP4Recipe(), quantized_storage=True)
        sd = model.state_dict()
        # torchao base stores "w_q"; the mslk-fast base stores "wq"/"wsc"/"w_inv".
        assert any(k.endswith(("w_q", "wq")) for k in sd)

        ids = torch.randint(0, 1000, (2, 32), device=model.device)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        opt.zero_grad()
        model(input_ids=ids, labels=ids).loss.backward()
        opt.step()

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        reloaded = self._lora_model()
        convert_lora_base_to_nvfp4(reloaded, NVFP4Recipe(), quantized_storage=True)
        missing, _ = reloaded.load_state_dict(
            torch.load(buf, weights_only=False), strict=False
        )
        assert not [
            k for k in missing if k.rsplit(".", 1)[-1] in ("w_q", "wq", "wsc", "w_inv")
        ]

        model.eval()
        reloaded.eval()
        with torch.no_grad():
            o1 = model(input_ids=ids).logits
            o2 = reloaded(input_ids=ids).logits
        assert torch.equal(o1, o2)

    def test_compute_base_checkpoint_roundtrip(self):
        """NVFP4ComputeBaseLinear: the two FP4 layouts (float4_e2m1fn_x2 qdata +
        float8_e4m3fn scales) must survive state_dict save/load bit-exactly,
        else compute-base resume reinitializes the base."""
        import io

        from torch import nn

        from axolotl.utils.nvfp4_training import (
            NVFP4ComputeBaseLinear,
            NVFP4Recipe,
        )

        torch.manual_seed(0)
        lin = nn.Linear(512, 256, bias=True).cuda().bfloat16()
        mod = NVFP4ComputeBaseLinear.from_linear(lin, NVFP4Recipe())
        x = torch.randn(48, 512, device="cuda", dtype=torch.bfloat16)
        ref = mod(x)

        sd = mod.state_dict()
        assert any(k.endswith("w_fprop") for k in sd)
        assert any(k.endswith("w_dgrad") for k in sd)

        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        fresh = NVFP4ComputeBaseLinear.from_linear(
            nn.Linear(512, 256, bias=True).cuda().bfloat16(), NVFP4Recipe()
        )
        fresh.load_state_dict(torch.load(buf, weights_only=False))
        assert torch.equal(fresh.w_fprop.qdata, mod.w_fprop.qdata)
        assert torch.equal(fresh.w_fprop.scale, mod.w_fprop.scale)
        assert torch.equal(fresh(x), ref)

    def test_fast_compute_base_checkpoint_roundtrip(self):
        """NVFP4FastComputeBaseLinear (MSLK): the packed FP4 qdata + swizzled e4m3
        scale + inv-global buffers must survive state_dict save/load bit-exactly."""
        import io

        from torch import nn

        from axolotl.utils.nvfp4_training import (
            NVFP4FastComputeBaseLinear,
            NVFP4Recipe,
            _mslk_available,
        )

        if not _mslk_available():
            pytest.skip("MSLK fused FP4 quant kernel not available")

        torch.manual_seed(0)
        lin = nn.Linear(512, 256, bias=True).cuda().bfloat16()
        mod = NVFP4FastComputeBaseLinear.from_linear(lin, NVFP4Recipe())
        x = torch.randn(48, 512, device="cuda", dtype=torch.bfloat16)
        ref = mod(x)

        sd = mod.state_dict()
        assert any(k.endswith("wq_f") for k in sd)
        assert any(k.endswith("wq_d") for k in sd)

        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        fresh = NVFP4FastComputeBaseLinear.from_linear(
            nn.Linear(512, 256, bias=True).cuda().bfloat16(), NVFP4Recipe()
        )
        fresh.load_state_dict(torch.load(buf, weights_only=False))
        assert fresh.wq_f.dtype == torch.float4_e2m1fn_x2
        assert fresh.wsc_f.dtype == torch.float8_e4m3fn
        assert torch.equal(fresh.wq_f, mod.wq_f)
        assert torch.equal(fresh.wsc_f, mod.wsc_f)
        assert torch.equal(fresh(x), ref)

    def test_fast_storage_base_checkpoint_roundtrip(self):
        """NVFP4FastFrozenBaseLinear (MSLK storage): single FP4 weight layout
        survives save/load and reproduces forward bit-exactly."""
        import io

        from torch import nn

        from axolotl.utils.nvfp4_training import (
            NVFP4FastFrozenBaseLinear,
            NVFP4Recipe,
            _mslk_available,
        )

        if not _mslk_available():
            pytest.skip("MSLK fused FP4 quant kernel not available")

        torch.manual_seed(0)
        lin = nn.Linear(512, 256, bias=True).cuda().bfloat16()
        mod = NVFP4FastFrozenBaseLinear.from_linear(lin, NVFP4Recipe())
        x = torch.randn(48, 512, device="cuda", dtype=torch.bfloat16)
        ref = mod(x)

        sd = mod.state_dict()
        assert any(k.endswith("wq") for k in sd)

        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        fresh = NVFP4FastFrozenBaseLinear.from_linear(
            nn.Linear(512, 256, bias=True).cuda().bfloat16(), NVFP4Recipe()
        )
        fresh.load_state_dict(torch.load(buf, weights_only=False))
        assert fresh.wq.dtype == torch.float4_e2m1fn_x2
        assert torch.equal(fresh.wq, mod.wq)
        assert torch.equal(fresh(x), ref)
