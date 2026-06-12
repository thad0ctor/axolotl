"""Regression tests: the Qwen3.5 decoder LOOP must compile under torch.compile.

The packed-sequence cu_seqlens helper used aten.nonzero inside the decoder
loop; dynamo cannot resume from a graph break inside a for loop, so a single
break made it skip the whole ``Qwen3_5TextModel.forward`` frame ("graph break
in loop", frame reported at modeling_qwen3_5.py:1235) and EVERY decoder layer
ran eagerly. The FLA training kernels now run behind opaque custom ops
(``axolotl_qwen3_5::gdn_conv`` / ``gdn_chunk``, see fla_ops.py) that derive
cu_seqlens eagerly inside the op, and FusedRMSNormGated routes through its
custom-op wrapper whenever torch_compile is on — so the loop body traces with
zero breaks (gradient checkpointing path; without checkpointing the
intentional eager self-attention boundary remains, which still skips the loop
frame but must never reintroduce an aten.nonzero break).
"""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("fla")


@pytest.fixture
def packing_patched():
    """Apply the packing patch (torch_compile on) and restore globals after."""
    import fla.modules.fused_norm_gate  # noqa: F401  (ensure importable)
    from fla.modules import FusedRMSNormGated
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    from axolotl.monkeypatch.models.qwen3_5 import modeling as qm

    saved = {
        "decoder_forward": hf.Qwen3_5DecoderLayer.forward,
        "gdn_forward": hf.Qwen3_5GatedDeltaNet.forward,
        "norm_forward": FusedRMSNormGated.forward,
        "norm_flag": getattr(FusedRMSNormGated, "_axolotl_compile_boundary", False),
        "chunk": getattr(hf, "chunk_gated_delta_rule", None),
        "recurrent": getattr(hf, "fused_recurrent_gated_delta_rule", None),
        "norm_cls": getattr(hf, "FusedRMSNormGated", None),
        "fast_path": getattr(hf, "is_fast_path_available", None),
        "fla_ops_flag": qm._FLA_COMPILED_OPS,
    }
    qm.patch_qwen3_5_modeling_packing(torch_compile=True)
    yield qm
    hf.Qwen3_5DecoderLayer.forward = saved["decoder_forward"]
    hf.Qwen3_5GatedDeltaNet.forward = saved["gdn_forward"]
    FusedRMSNormGated.forward = saved["norm_forward"]
    if saved["norm_flag"]:
        FusedRMSNormGated._axolotl_compile_boundary = saved["norm_flag"]
    else:
        try:
            delattr(FusedRMSNormGated, "_axolotl_compile_boundary")
        except AttributeError:
            pass
    hf.chunk_gated_delta_rule = saved["chunk"]
    hf.fused_recurrent_gated_delta_rule = saved["recurrent"]
    hf.FusedRMSNormGated = saved["norm_cls"]
    hf.is_fast_path_available = saved["fast_path"]
    qm._FLA_COMPILED_OPS = saved["fla_ops_flag"]


def _build_model(seed: int = 0):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3_5TextModel(cfg).cuda().to(torch.bfloat16)


def _packed_inputs(T: int = 64):
    torch.manual_seed(123)
    input_ids = torch.randint(0, 128, (1, T), device="cuda")
    pos = torch.cat(
        [torch.arange(40, device="cuda"), torch.arange(T - 40, device="cuda")]
    ).view(1, T)
    position_ids = pos[None, ...].expand(4, 1, T).contiguous()
    return input_ids, position_ids


def _fwd_bwd(model, fn=None):
    input_ids, position_ids = _packed_inputs()
    if fn is None:
        fn = model
    out = fn(
        input_ids=input_ids, position_ids=position_ids, use_cache=False
    ).last_hidden_state
    loss = out.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.detach(), grads


class TestDecoderLoopCompiles:
    def test_no_graph_breaks_with_grad_checkpointing(self, packing_patched):
        """Production shape (GC on): the loop must trace with ZERO breaks."""
        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        model = _build_model()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        _fwd_bwd(model, torch.compile(model))

        breaks = dict(counters["graph_break"])
        assert not breaks, f"decoder loop graph-broke: {list(breaks)}"
        assert counters["stats"]["unique_graphs"] >= 1

    def test_no_nonzero_break_without_grad_checkpointing(self, packing_patched):
        """GC off: the intentional eager self-attn boundary remains, but the
        aten.nonzero break at the patched site must never come back."""
        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        model = _build_model()
        model.train()
        _fwd_bwd(model, torch.compile(model))

        breaks = dict(counters["graph_break"])
        offenders = [k for k in breaks if "nonzero" in k.lower()]
        assert not offenders, f"aten.nonzero graph break is back: {offenders}"
        # only the intentional dynamo.disable boundary around self-attention
        for k in breaks:
            assert "disable" in k.lower(), f"unexpected graph break: {k}"

    def test_eager_parity_ops_vs_legacy_bitwise(self, packing_patched):
        """The opaque-op path must be numerically IDENTICAL to the legacy
        (pre-fix) eager path: same FLA kernels, same saved tensors."""
        qm = packing_patched
        assert qm._FLA_COMPILED_OPS
        model = _build_model()
        model.train()
        state = {k: v.clone() for k, v in model.state_dict().items()}

        qm._FLA_COMPILED_OPS = False
        loss_a, grads_a = _fwd_bwd(model)
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        qm._FLA_COMPILED_OPS = True
        loss_b, grads_b = _fwd_bwd(model)

        assert torch.equal(loss_a, loss_b)
        assert set(grads_a) == set(grads_b)
        for n in grads_a:
            assert torch.equal(grads_a[n], grads_b[n]), f"grad {n} not bitwise"

    def test_compiled_matches_eager(self, packing_patched):
        """Compiled (GC on) vs eager: same class of bf16/Inductor reordering
        noise as the pre-fix partially-compiled path (measured worst dgrad
        ~2e-4 on a 2.8e-2 grad scale)."""
        torch._dynamo.reset()
        model = _build_model()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        state = {k: v.clone() for k, v in model.state_dict().items()}

        loss_e, grads_e = _fwd_bwd(model)
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        loss_c, grads_c = _fwd_bwd(model, torch.compile(model))

        assert torch.allclose(loss_e, loss_c, rtol=1e-3, atol=1e-3)
        for n in grads_e:
            assert torch.allclose(
                grads_e[n].float(), grads_c[n].float(), rtol=5e-2, atol=1e-3
            ), f"grad {n}: {(grads_e[n].float() - grads_c[n].float()).abs().max()}"


# ---------------------------------------------------------------------------
# NVFP4 attention leg: the patched full-attention forward must also keep the
# decoder loop compilable. Historically it could not: (a) load-bearing
# torch._dynamo.graph_break() isolation pairs around the training custom op
# (a stale guard against an Inductor miscompile on an earlier torch/patch
# state — verified stale by a 9B real-config probe and removed, see the NOTE
# in nvfp4_flash_attn.py) and (b) the data-dependent packed-batch classification
# (``(pos == 0).sum()`` / aten.nonzero) running inside the per-layer forward.
# (b) is now hoisted to a dynamo-disabled model pre-forward hook that resolves
# the routing once per step and threads (kind, cu_seqlens) to the layers as
# module attributes.
# ---------------------------------------------------------------------------
_NVFP4_OK = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
    10,
    0,
)

requires_nvfp4 = pytest.mark.skipif(
    not _NVFP4_OK, reason="native NVFP4 attention requires sm>=100 (Blackwell)"
)


def _build_model_nvfp4(seed: int = 0):
    """Tiny hybrid model with NVFP4-servable full-attention layers (D=128)."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    cfg._attn_implementation = "flash_attention_2"
    return Qwen3_5TextModel(cfg).cuda().to(torch.bfloat16)


@pytest.fixture
def nvfp4_patched_model(packing_patched):
    """Packing patch + NVFP4 attention patch (training, compile custom op)."""
    pytest.importorskip("sageattention.nvfp4")
    from axolotl.monkeypatch.attention.nvfp4_flash_attn import (
        patch_qwen3_5_nvfp4_attention,
    )

    model = _build_model_nvfp4()
    patched = patch_qwen3_5_nvfp4_attention(
        model,
        train_backward=True,
        compile_custom_op=True,
        stochastic_rounding=False,
        packed_min_sample_len=0,
    )
    assert patched == 2
    return model


@requires_nvfp4
class TestNVFP4DecoderLoopCompiles:
    def test_no_graph_breaks_with_grad_checkpointing(self, nvfp4_patched_model):
        """NVFP4 leg, production shape (GC on): the decoder loop must trace
        with ZERO breaks — no isolation graph_break pair, no aten.nonzero from
        the packed classification (hoisted to the model pre-forward hook)."""
        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        model = nvfp4_patched_model
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        _fwd_bwd(model, torch.compile(model))

        breaks = dict(counters["graph_break"])
        assert not breaks, f"nvfp4 decoder loop graph-broke: {list(breaks)}"
        assert counters["stats"]["unique_graphs"] >= 1

    def test_step_info_hook_resolves_packed_routing(self, nvfp4_patched_model):
        """The pre-forward hook threads (kind, cu_seqlens) to every patched
        attention module — the compiled forward reads these instead of running
        the data-dependent classification inside the loop."""
        from axolotl.monkeypatch.attention.nvfp4_flash_attn import (
            _STEP_CU_ATTR,
            _STEP_KIND_ATTR,
        )

        model = nvfp4_patched_model
        model.train()
        _fwd_bwd(model)  # eager: hook fires on the model __call__

        mods = [m for m in model.modules() if getattr(m, "_nvfp4_patched", False)]
        assert len(mods) == 2
        for m in mods:
            assert getattr(m, _STEP_KIND_ATTR, None) == "packed"
            cu = getattr(m, _STEP_CU_ATTR, None)
            assert cu is not None and cu.tolist() == [0, 40, 64]

    def test_compiled_matches_eager(self, nvfp4_patched_model):
        """Compiled (GC on, one-graph loop) vs eager on the NVFP4 leg.

        Inductor keeps fused bf16 chains in fp32 (codegen_upcast_to_fp32)
        while eager rounds to bf16 between ops; the FP4 quantizers turn that
        precision difference into bin flips, so bitwise equality only holds
        with the upcast disabled (verified during bring-up). Against a
        non-degenerate loss the grads agree directionally and in norm."""
        import torch.nn.functional as F

        torch._dynamo.reset()
        model = nvfp4_patched_model
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        state = {k: v.clone() for k, v in model.state_dict().items()}

        torch.manual_seed(5)
        target = torch.randn(1, 64, 512, device="cuda", dtype=torch.float32)

        def run(fn):
            input_ids, position_ids = _packed_inputs()
            out = fn(
                input_ids=input_ids, position_ids=position_ids, use_cache=False
            ).last_hidden_state
            loss = (out.float() - target).pow(2).mean()
            loss.backward()
            torch.cuda.synchronize()
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            return loss.detach(), grads

        loss_e, grads_e = run(model)
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        loss_c, grads_c = run(torch.compile(model))

        assert torch.allclose(loss_e, loss_c, rtol=1e-3, atol=1e-3)
        ge = torch.cat([grads_e[n].float().flatten() for n in sorted(grads_e)])
        gc = torch.cat([grads_c[n].float().flatten() for n in sorted(grads_e)])
        cos = F.cosine_similarity(ge, gc, dim=0).item()
        ratio = gc.norm().item() / (ge.norm().item() + 1e-12)
        assert cos > 0.97, f"global grad cos {cos}"
        assert 0.9 < ratio < 1.1, f"global grad norm ratio {ratio}"

    def test_grad_health_over_steps(self, nvfp4_patched_model):
        """No param-grad blowup (the historical ~1e5 -> NaN miscompile
        symptom) over repeated compiled steps with an optimizer."""
        torch._dynamo.reset()
        model = nvfp4_patched_model
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        compiled = torch.compile(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids, position_ids = _packed_inputs()
        for step in range(10):
            model.zero_grad(set_to_none=True)
            out = compiled(
                input_ids=input_ids, position_ids=position_ids, use_cache=False
            ).last_hidden_state
            loss = out.float().pow(2).mean()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
            assert torch.isfinite(loss).item(), f"step {step}: non-finite loss"
            assert gn == gn and gn < 1e4, f"step {step}: grad_norm {gn}"
            opt.step()
