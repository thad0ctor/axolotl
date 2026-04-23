"""Unit + GPU tests for the ProTrain M1 profiler."""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.profiler import (
    ProfilerCacheKey,
    load_cached_trace,
    measure_pcie,
    reconstruct_peak_bytes,
    run_trace,
    save_cached_trace,
)
from axolotl.integrations.protrain.profiler.on_demand import OnDemandTensorMgr
from axolotl.integrations.protrain.types import (
    BlockId,
    OpId,
    OpRecord,
    ProfilerConfig,
    ProfilerTrace,
)


_TINY_MODEL_CANDIDATES = (
    "sshleifer/tiny-gpt2",
    "hf-internal-testing/tiny-random-gpt2",
)


def _load_tiny_gpt2():
    """Try the canonical tiny-GPT2 checkpoint, fall back to the HF-internal one."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    last_exc: Exception | None = None
    for name in _TINY_MODEL_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name)
            return name, tok, model
        except Exception as exc:  # pragma: no cover - network-dependent
            last_exc = exc
            continue
    raise RuntimeError(f"no tiny-GPT2 checkpoint available: {last_exc}")


def _build_batch(tok, bs: int, seq: int, device):
    import torch

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"
    text = ["hello world"] * bs
    enc = tok(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@pytest.mark.gpu
def test_reconstruct_peak_within_10pct_tiny_gpt2(gpu_device):
    """The M1 accuracy contract: simplified peak within 10% of max_memory_allocated."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")
    name, tok, model = _load_tiny_gpt2()
    model = model.to(device)

    bs, seq = 2, 128
    batch = _build_batch(tok, bs, seq, device)

    cfg = ProfilerConfig(
        batch_size=bs,
        seq_len=seq,
        device=str(device),
        include_backward=True,
        on_demand=False,
    )

    # First: profiled run. Hooks add a small constant; we care about the
    # reconstructed number, not the measured peak during this call.
    trace = run_trace(model, batch, cfg)
    peak_est = reconstruct_peak_bytes(trace)

    # Second: ground-truth run with no hooks. Fresh zero for peak stats.
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    # Re-fetch a batch tied to no retained autograd graph from the first pass.
    batch2 = _build_batch(tok, bs, seq, device)
    output = model(**batch2)
    loss = output.loss if hasattr(output, "loss") else output[0].sum()
    loss.backward()
    torch.cuda.synchronize(device)
    ground_truth = int(torch.cuda.max_memory_allocated(device))

    assert ground_truth > 0, "ground truth peak should be positive"
    rel_err = abs(peak_est - ground_truth) / ground_truth
    assert rel_err < 0.10, (
        f"reconstructed peak {peak_est} vs ground truth {ground_truth} "
        f"rel_err={rel_err:.3f} on model {name!r}"
    )


def _minimal_trace() -> ProfilerTrace:
    """Build a tiny valid ProfilerTrace for cache round-trip testing."""
    op = OpRecord(
        op_id=OpId(0),
        module_path="root.layer0",
        qualified_name="Linear",
        shape_signature=((2, 128, 16),),
        block_id=BlockId(0),
        is_forward=True,
    )
    return ProfilerTrace(
        op_order=(op,),
        intra_op_delta={OpId(0): 1024},
        inter_op_delta={OpId(0): 512},
        activation_sizes={BlockId(0): 2048},
        model_state_bytes=1 << 20,
        pcie_h2d_bps=25e9,
        pcie_d2h_bps=23e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="deadbeef",
        bs=2,
        seq=128,
        sku="NVIDIA GeForce RTX 3090",
        world=1,
    )


def test_cache_roundtrip(tmp_path, monkeypatch):
    """save -> load must return an equal ProfilerTrace."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    key = ProfilerCacheKey(
        arch_hash="deadbeef",
        bs=2,
        seq=128,
        sku="NVIDIA GeForce RTX 3090",
        world=1,
    )
    trace = _minimal_trace()
    path = save_cached_trace(key, trace)
    assert path.exists()

    loaded = load_cached_trace(key)
    assert loaded is not None
    assert loaded == trace

    # Missing key returns None.
    other = ProfilerCacheKey(
        arch_hash="feedface", bs=2, seq=128, sku="NVIDIA GeForce RTX 3090", world=1
    )
    assert load_cached_trace(other) is None


@pytest.mark.gpu
def test_hw_bench_pcie_returns_positive(gpu_device):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    h2d, d2h = measure_pcie(gpu_device, n_bytes=16 * 1024 * 1024, n_iters=2)
    assert h2d > 0
    assert d2h > 0
    # 200 GB/s is well above PCIe 5.0 x16 theoretical (~63 GB/s); trips if we
    # accidentally divide by the wrong unit.
    assert h2d < 200e9
    assert d2h < 200e9


def test_on_demand_disabled_fast_path():
    """Disabled OnDemandTensorMgr must be a no-op context manager."""
    mgr = OnDemandTensorMgr(device="cuda:0", disabled=True)
    with mgr as entered:
        assert entered is mgr
        # Disabled path must not raise on allocate/free.
        fake_op = OpRecord(
            op_id=OpId(0),
            module_path="x",
            qualified_name="X",
            shape_signature=((),),
            block_id=None,
            is_forward=True,
        )
        mgr.allocate_inputs(fake_op)
        mgr.free_after(fake_op)
    assert tuple(mgr.live_tensor_ids()) == ()
