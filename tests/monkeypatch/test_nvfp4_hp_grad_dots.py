"""GPU wiring tests for the HP-grad-dots NVFP4 attention backward.

Covers the new ``bf16_grad_dots`` knob end to end: the eager
``nvfp4_flash_attn_func`` path, the torch.compile opaque custom op
(``nvfp4_flash_attn_train_custom_op``), and the monkeypatch plumbing
(``patch_qwen3_vl_nvfp4_attention``). With ``bf16_grad_dots`` auto (None) and
``save_backward_packs=False`` the backward must route through ``_run_bwd_hp``
and produce grads with cosine > 0.98 vs an SDPA reference (the legacy all-FP4
backward sits at ~0.94 RTN / ~0.82 SR — well below this bar, so the threshold
itself proves the HP path is active).
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (10, 0)
        ),
        reason="native NVFP4 attention requires sm>=100 (Blackwell)",
    ),
]

pytest.importorskip("sageattention.nvfp4")

from axolotl.kernels.attn_nvfp4_flash import (  # noqa: E402
    _run_bwd_hp,
    nvfp4_flash_attn_func,
)

if _run_bwd_hp is None:  # pragma: no cover - legacy fork
    pytest.skip(
        "installed sageattention fork lacks the HP-grad-dots backward "
        "(_run_bwd_hp); point PYTHONPATH at the train-perf fork",
        allow_module_level=True,
    )

Z, H, HK, S = 2, 4, 2, 256
NG = H // HK
GRAD_COS = 0.98


def _qkv(d, seed):
    torch.manual_seed(seed)
    q = torch.randn(Z, H, S, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Z, HK, S, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Z, HK, S, d, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def _sdpa_ref(q, k, v, scaling, do):
    """bf16 SDPA reference grads on fresh leaves carrying the same values."""
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    out = F.scaled_dot_product_attention(
        qr,
        kr.repeat_interleave(NG, dim=1),
        vr.repeat_interleave(NG, dim=1),
        is_causal=True,
        scale=scaling,
    )
    out.backward(do)
    return out, qr.grad, kr.grad, vr.grad


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _assert_grads(tag, grads, ref_grads):
    for name, g, r in zip(("dq", "dk", "dv"), grads, ref_grads):
        assert g is not None and torch.isfinite(g).all(), f"{tag} {name} non-finite"
        assert g.abs().sum() > 0, f"{tag} {name} is all-zero"
        c = _cos(g, r)
        assert c > GRAD_COS, f"{tag} {name} cos={c:.4f} <= {GRAD_COS}"


@pytest.mark.parametrize("d", [128, 256])
def test_eager_hp_grad_dots_auto(d):
    """Eager nvfp4_flash_attn_func with bf16_grad_dots auto beats the legacy bar."""
    q, k, v = _qkv(d, seed=10 + d)
    scaling = d**-0.5
    torch.manual_seed(99)
    do = torch.randn(Z, H, S, d, device="cuda", dtype=torch.bfloat16)
    _, rdq, rdk, rdv = _sdpa_ref(q, k, v, scaling, do)

    ql = q.clone().requires_grad_(True)
    kl = k.clone().requires_grad_(True)
    vl = v.clone().requires_grad_(True)
    out = nvfp4_flash_attn_func(
        ql,
        kl,
        vl,
        scaling,
        causal=True,
        num_key_value_groups=NG,
        backward_bf16_grad_dots=None,  # auto: on (save_backward_packs is False)
    )
    assert torch.isfinite(out).all()
    out.backward(do)
    _assert_grads("eager", (ql.grad, kl.grad, vl.grad), (rdq, rdk, rdv))


@pytest.mark.parametrize("d", [128, 256])
def test_custom_op_compiled_hp_grad_dots_auto(d):
    """The opaque custom op under torch.compile routes backward through HP grads."""
    from axolotl.kernels.attn_nvfp4_custom_op import nvfp4_flash_attn_train_custom_op

    torch._dynamo.reset()
    q, k, v = _qkv(d, seed=20 + d)
    scaling = d**-0.5
    torch.manual_seed(98)
    do = torch.randn(Z, H, S, d, device="cuda", dtype=torch.bfloat16)
    _, rdq, rdk, rdv = _sdpa_ref(q, k, v, scaling, do)

    def fn(q_, k_, v_):
        return nvfp4_flash_attn_train_custom_op(
            q_,
            k_,
            v_,
            scaling,
            causal=True,
            num_key_value_groups=NG,
            backward_bf16_grad_dots=None,  # auto
        )

    compiled = torch.compile(fn, dynamic=False)
    ql = q.clone().requires_grad_(True)
    kl = k.clone().requires_grad_(True)
    vl = v.clone().requires_grad_(True)
    out = compiled(ql, kl, vl)
    assert torch.isfinite(out).all()
    out.backward(do)
    _assert_grads("compiled", (ql.grad, kl.grad, vl.grad), (rdq, rdk, rdv))


def test_custom_op_backward_routing(monkeypatch):
    """auto -> _run_bwd_hp; bf16_grad_dots=False or save_packs=True -> legacy."""
    import axolotl.kernels.attn_nvfp4_custom_op as op_mod

    d = 128
    scaling = d**-0.5
    calls = []
    real_hp = op_mod._run_bwd_hp

    def spy(*args, **kwargs):
        calls.append(1)
        return real_hp(*args, **kwargs)

    monkeypatch.setattr(op_mod, "_run_bwd_hp", spy)

    def run(**op_kwargs):
        q, k, v = _qkv(d, seed=33)
        ql = q.clone().requires_grad_(True)
        kl = k.clone().requires_grad_(True)
        vl = v.clone().requires_grad_(True)
        out = op_mod.nvfp4_flash_attn_train_custom_op(
            ql, kl, vl, scaling, causal=True, num_key_value_groups=NG, **op_kwargs
        )
        out.float().sum().backward()
        for g in (ql.grad, kl.grad, vl.grad):
            assert g is not None and torch.isfinite(g).all()

    run(backward_bf16_grad_dots=None)  # auto, packs off -> HP path
    assert len(calls) == 1
    run(backward_bf16_grad_dots=False)  # forced legacy
    assert len(calls) == 1
    run(backward_bf16_grad_dots=None, save_backward_packs=True)  # packs force legacy
    assert len(calls) == 1
    run(backward_bf16_grad_dots=True, save_backward_packs=True)  # clamped off
    assert len(calls) == 1
    run(backward_bf16_grad_dots=True)  # forced HP
    assert len(calls) == 2


def _build_vl_model(seed=0, n_layers=2, head_dim=128, n_heads=4, n_kv=2):
    """Tiny Qwen3-VL text model (no download) — mirrors test_qwen3_vl_nvfp4_attn."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(seed)
    cfg = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=n_heads * head_dim,
        intermediate_size=256,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3VLTextModel(cfg).cuda().to(torch.bfloat16).eval()


def _vl_inputs(hidden, seq=256, seed=0):
    torch.manual_seed(seed)
    hs = torch.randn(1, seq, hidden, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(seq, device="cuda").unsqueeze(0)
    return hs, pos


def _run_vl_attention(model, layer_idx, hidden_states, position_ids):
    attn = model.layers[layer_idx].self_attn
    cos, sin = model.rotary_emb(hidden_states, position_ids)
    out, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )
    return out


def test_vl_monkeypatch_plumbs_hp_grad_dots(monkeypatch):
    """patch_qwen3_vl_nvfp4_attention threads the knob into the eager backward."""
    pytest.importorskip("transformers.models.qwen3_vl")
    import sageattention.nvfp4.flash as flash_mod

    from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (
        patch_qwen3_vl_nvfp4_attention,
    )

    calls = []
    real_hp = flash_mod._run_bwd_hp

    def spy(*args, **kwargs):
        calls.append(1)
        return real_hp(*args, **kwargs)

    monkeypatch.setattr(flash_mod, "_run_bwd_hp", spy)

    model = _build_vl_model(seed=6)
    hidden = model.config.hidden_size
    patched = patch_qwen3_vl_nvfp4_attention(
        model, train_backward=True, bf16_grad_dots=None
    )
    assert patched > 0
    assert model.layers[0].self_attn._nvfp4_bf16_grad_dots is None

    hs, pos = _vl_inputs(hidden, seed=6)
    hs = hs.clone().requires_grad_(True)
    out = _run_vl_attention(model, 0, hs, pos)
    out.float().sum().backward()
    assert len(calls) > 0, "auto bf16_grad_dots did not route through _run_bwd_hp"
    assert hs.grad is not None and torch.isfinite(hs.grad).all()
    assert hs.grad.abs().sum() > 0

    # Forcing it off must fall back to the legacy all-FP4 backward.
    calls.clear()
    patch_qwen3_vl_nvfp4_attention(model, train_backward=True, bf16_grad_dots=False)
    assert model.layers[0].self_attn._nvfp4_bf16_grad_dots is False
    hs2, pos2 = _vl_inputs(hidden, seed=7)
    hs2 = hs2.clone().requires_grad_(True)
    out2 = _run_vl_attention(model, 0, hs2, pos2)
    out2.float().sum().backward()
    assert len(calls) == 0
    assert hs2.grad is not None and torch.isfinite(hs2.grad).all()
