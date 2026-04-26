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


@pytest.mark.gpu
def test_trace_records_op_latencies(gpu_device):
    """Profiler must populate ``trace.op_latencies`` with measured per-op times."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")
    _name, tok, model = _load_tiny_gpt2()
    model = model.to(device)

    bs, seq = 1, 64
    batch = _build_batch(tok, bs, seq, device)

    cfg = ProfilerConfig(
        batch_size=bs,
        seq_len=seq,
        device=str(device),
        include_backward=True,
        on_demand=False,
    )

    trace = run_trace(model, batch, cfg)

    # Must be non-empty — if this fails we regressed the capture path.
    assert trace.op_latencies, "trace.op_latencies must be populated"

    # Every recorded latency is positive and well under 1s on tiny-GPT-2;
    # the latter trips if elapsed_ms is not converted to seconds.
    for op_id, lat in trace.op_latencies.items():
        assert lat > 0.0, f"op {op_id} has non-positive latency {lat}"
        assert lat < 1.0, f"op {op_id} latency {lat}s exceeds sanity ceiling"

    # Coverage: at least 80% of ops in op_order must have a latency entry.
    # (Some edge-case modules may fire a pre-hook but no post-hook if
    # forward re-enters the same module id; skip those.)
    n_ops = len(trace.op_order)
    n_covered = sum(1 for op in trace.op_order if op.op_id in trace.op_latencies)
    assert n_covered / max(1, n_ops) >= 0.80, (
        f"only {n_covered}/{n_ops} ops have latencies — coverage too low"
    )


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


def test_on_demand_enabled_requires_model():
    """Enabled mode must reject construction without a model."""
    mgr = OnDemandTensorMgr(device="cuda:0", disabled=False)
    with pytest.raises(ValueError, match="requires a model"):
        mgr.__enter__()


@pytest.mark.gpu
def test_on_demand_enabled_param_offload_and_restore(gpu_device):
    """Enabled OnDemandTensorMgr offloads params and restores them byte-exact."""
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    ).to(device)

    # Snapshot original params so we can verify byte-exact restore later.
    original_state = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }

    from axolotl.integrations.protrain.profiler.on_demand import (
        OnDemandTensorMgr,
    )

    mgr = OnDemandTensorMgr(device=device, disabled=False, model=model)

    x = torch.randn(4, 64, device=device)
    with mgr:
        # Inside the context, before any forward, params should be empty
        # placeholders (storage of size 0). The pre-forward hooks will
        # gather them just before each Linear's forward.
        for _name, p in model.named_parameters():
            assert p.data.numel() == 0, (
                f"expected empty placeholder under on-demand, got "
                f"{p.data.numel()} elements"
            )

        out = model(x)
        # Forward must produce a sane output of the right shape.
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()

    # After exiting, params restored to GPU with original values.
    for name, p in model.named_parameters():
        assert p.device.type == "cuda"
        assert torch.allclose(p, original_state[name], atol=0, rtol=0), (
            f"param {name} did not restore byte-exact under OnDemandTensorMgr"
        )


@pytest.mark.gpu
def test_on_demand_engaged_path_in_run_trace(gpu_device, monkeypatch):
    """run_trace engages on-demand when params exceed the size threshold.

    Forces the threshold down to ~0% so a tiny model takes the on-demand
    branch. The trace must still complete and populate op records.
    """
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")

    # Simple two-block "transformer" — enough to exercise multiple modules
    # under the on-demand gather/release path. Use a non-Linear container
    # so the trace's block heuristic still picks it up.
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 32)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])

        def forward(self, input_ids=None, **kwargs):
            x = input_ids.to(torch.float32)
            for layer in self.layers:
                x = layer(x)
            return type("Out", (), {"loss": x.sum()})()

    model = TinyModel().to(device)
    batch = {
        "input_ids": torch.randn(2, 32, device=device),
    }

    # Force on-demand to engage by dropping the threshold to 0%.
    from axolotl.integrations.protrain.profiler import trace as trace_mod

    monkeypatch.setattr(trace_mod, "ON_DEMAND_PARAM_BYTES_FRACTION", 0.0)

    cfg = ProfilerConfig(
        batch_size=2,
        seq_len=32,
        device=str(device),
        include_backward=False,
        on_demand=True,
    )
    trace = run_trace(model, batch, cfg)

    # Trace must have op records — the on-demand path didn't drop ops.
    assert len(trace.op_order) > 0
    # Forward-only trace: no <backward> op record expected.
    assert all(op.is_forward for op in trace.op_order)
    # Activation sizes captured for at least the inferred blocks (the layers
    # ModuleList children get block_id=0, 1 via the ``layers.<i>`` heuristic).
    assert len(trace.activation_sizes) >= 1, (
        "on-demand trace did not record any activation sizes"
    )
