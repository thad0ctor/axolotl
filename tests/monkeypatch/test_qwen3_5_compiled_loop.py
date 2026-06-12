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
