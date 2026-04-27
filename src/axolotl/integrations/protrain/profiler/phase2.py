"""Phase-2 chunked-runtime profiler (paper §3.2 calibration loop).

The wrapper's first ``run_trace`` runs **without** the chunk manager
engaged — backward is skipped (``include_backward=False``) because on
7B+ models the unwrapped backward OOMs the 24 GiB card. The cost model
then falls back to a heuristic bwd/fwd ratio (1.0× LoRA, 2.0×
full-finetune) which on 7B-LoRA over-/under-shoots the actual chunked
backward by 25-30 %.

Phase-2 closes that gap. After the initial ``search()`` returns, the
wrapper builds the runtime under a conservative bootstrap config,
runs a short chunked steady-state ``forward → loss.backward() →
optim.step()`` measurement loop, and writes the median backward + step
overlap into ``ProfilerTrace.steady_bwd_chunked_wall_s`` and
``steady_step_overlap_s``. The cost model translates the measurement
across configs via ``phase2_n_checkpoint`` + ``phase2_per_block_recompute_s``
(D1b — see ``cost/runtime._bwd_compute_time_from_trace``).

The actual measurement loop lives here; the wrapper plumbing
(bootstrap → measure → splice → re-search → rebuild) lives in
``api/model_wrapper.py``.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    CostConfig,
    SearchResult,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torch import nn

    from axolotl.integrations.protrain.types import (
        BlockStrategyMap,
        ChunkLayout,
        HardwareProfile,
        ProfilerTrace,
    )

LOG = get_logger(__name__)


# Number of warmup iterations discarded before timing starts. Three is
# enough to settle the buffer pool's LRU + gather/release cadence + CPU
# Adam's lazy state init, which all happen on the first forward/backward
# pass and would otherwise inflate the median.
_PHASE2_N_WARMUP = 3
# Number of timed iterations. Five gives a stable median on the 7B-LoRA
# canonical workload (per-iter variance ~5%); larger N adds latency
# without visibly tightening the median.
_PHASE2_N_ITERS = 5


def select_bootstrap_config(
    *,
    initial_result: SearchResult,
    layout: "ChunkLayout",
    n_block: int,
    capacity_bytes: int,
    trace: "ProfilerTrace",
    hw: "HardwareProfile",
) -> tuple[CostConfig, "BlockStrategyMap"]:
    """Pick a conservative bootstrap config that's guaranteed to fit.

    Spec: ``n_persist=N_chunk*0.5, n_buffer=4, n_swap=0,
    n_checkpoint=N_block`` (paper §3.2 design — bias hard toward
    memory savings so the chunked backward fits even when the cost
    model's backward estimate was wrong).

    Validates the candidate against ``estimate_peak``; if the peak
    exceeds capacity, fall back to the search's own first pick (which
    by construction passed the capacity gate). This second-line
    defense covers degenerate models where even max-CKPT + half-
    persistent doesn't fit — those would already have crashed before
    phase-2, but be defensive.
    """
    from axolotl.integrations.protrain.block.layout_rules import assign_modes
    from axolotl.integrations.protrain.cost.memory import estimate_peak

    # Use the search's own n_persist + n_buffer pick — those were
    # validated against capacity and sized so the scheduler's prefetch
    # cadence doesn't exhaust the pool. Only override n_checkpoint to
    # the all-CKPT extreme: all-CKPT uses STRICTLY LESS GPU memory than
    # any fewer-CKPT config (CKPT drops activations; the analytical
    # peak's per-block bump only fires for non-CKPT blocks), so the
    # bootstrap stays capacity-feasible by transitivity from the
    # search's pick. The spec's literal n_persist=N_chunk/2 + n_buffer=4
    # would shrink n_buffer below what the search needed for prefetch
    # and trip BufferPool exhaustion under the all-CKPT recompute load.
    n_chunk = layout.N_chunk
    bootstrap_cfg = CostConfig(
        n_persist=initial_result.cfg.n_persist,
        n_buffer=initial_result.cfg.n_buffer,
        n_swap=0,
        n_checkpoint=n_block,
    )
    bootstrap_block_map = assign_modes(0, n_block, n_block)
    del n_chunk  # currently unused; kept above for self-documenting layout intent

    candidate_peak = estimate_peak(
        bootstrap_cfg, trace, layout, bootstrap_block_map, hw
    )
    if candidate_peak <= capacity_bytes:
        LOG.info(
            "Phase-2 bootstrap config: n_persist=%d n_buffer=%d "
            "n_checkpoint=%d (peak %.2f GB <= capacity %.2f GB)",
            bootstrap_cfg.n_persist,
            bootstrap_cfg.n_buffer,
            bootstrap_cfg.n_checkpoint,
            candidate_peak / (1 << 30),
            capacity_bytes / (1 << 30),
        )
        return bootstrap_cfg, bootstrap_block_map

    LOG.warning(
        "Phase-2 bootstrap formula (n_persist=%d n_buffer=%d "
        "n_checkpoint=%d) predicts peak %.2f GB > capacity %.2f GB; "
        "falling back to the searcher's first pick which passed the "
        "capacity gate by construction.",
        bootstrap_cfg.n_persist,
        bootstrap_cfg.n_buffer,
        bootstrap_cfg.n_checkpoint,
        candidate_peak / (1 << 30),
        capacity_bytes / (1 << 30),
    )
    return initial_result.cfg, initial_result.block_map


def measure_chunked_steady(
    *,
    model: "nn.Module",
    batch: dict,
    optimizer: "torch.optim.Optimizer",
    n_warmup: int = _PHASE2_N_WARMUP,
    n_iters: int = _PHASE2_N_ITERS,
) -> tuple[float, float]:
    """Run a chunked steady-state ``fwd → bwd → step`` loop and time it.

    Times the backward and the post-backward optimizer step using
    ``torch.cuda.Event`` pairs (same convention as
    :mod:`profiler.hw_bench` for ``measure_compute_rate`` /
    ``measure_cpu_adam`` / ``measure_gpu_adam``). The optimizer step
    timing window includes the wait for the asynchronous CPU FusedAdam
    that the per-param grad hooks kick off during backward — so it
    captures the bwd↔step overlap envelope, not the cumulative compute.

    Returns
    -------
    (steady_bwd_chunked_wall_s, steady_step_overlap_s)
        Median across ``n_iters`` timed iterations. ``n_warmup``
        iterations are discarded — they pay one-time costs (chunk
        manager LRU settling, CPU Adam state lazy init, autograd
        graph construction) that would inflate the median.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase-2 measurement requires CUDA; got "
            "torch.cuda.is_available() == False"
        )

    model.train()

    # Warmup — discard timings.
    for _ in range(n_warmup):
        out = model(**batch)
        loss = _extract_loss(out)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    bwd_times_s: list[float] = []
    step_times_s: list[float] = []
    for _ in range(n_iters):
        out = model(**batch)
        loss = _extract_loss(out)

        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)

        bwd_start.record()
        loss.backward()
        bwd_end.record()
        optimizer.step()
        step_end.record()

        torch.cuda.synchronize()
        bwd_times_s.append(bwd_start.elapsed_time(bwd_end) / 1000.0)
        step_times_s.append(bwd_end.elapsed_time(step_end) / 1000.0)

        optimizer.zero_grad(set_to_none=True)

    bwd_median = statistics.median(bwd_times_s)
    step_median = statistics.median(step_times_s)
    LOG.info(
        "Phase-2 chunked-runtime measurement: "
        "steady_bwd_chunked_wall_s=%.4f (n=%d, samples=%s) "
        "steady_step_overlap_s=%.4f (samples=%s)",
        bwd_median,
        n_iters,
        ["%.4f" % t for t in bwd_times_s],
        step_median,
        ["%.4f" % t for t in step_times_s],
    )
    return bwd_median, step_median


def estimate_per_block_recompute_s(
    trace: "ProfilerTrace", n_block: int
) -> float:
    """Mean per-block forward compute time (≡ recompute under CKPT).

    Uses :func:`cost.runtime._fwd_compute_time_from_trace` to derive
    per-block forward time from the trace's measured op latencies (or
    the activation-size roofline proxy when latencies are absent).
    Returns the mean across blocks — phase-2's translation formula
    works in mean-per-block units because the cost model approximates
    per-block recompute as a uniform per-block term.

    Returns 0.0 when ``n_block == 0`` or when the trace has no op
    latencies AND no activation sizes (degenerate trace — would only
    happen in a unit test fixture, never on a live profile).
    """
    from axolotl.integrations.protrain.cost.runtime import (
        _fwd_compute_time_from_trace,
    )

    if n_block <= 0:
        return 0.0
    t_fwd_total, per_block_compute, _used_measured = (
        _fwd_compute_time_from_trace(trace)
    )
    if per_block_compute:
        # Mean of measured per-block times — this is what the cost
        # model adds per CKPT block via ``per_block_compute.get(bid)``.
        return sum(per_block_compute.values()) / max(1, len(per_block_compute))
    if t_fwd_total > 0.0:
        # Fallback: divide aggregate forward by N_block. Less accurate
        # but the cost model uses the same fallback (activation-size
        # roofline) per block — we maintain symmetry.
        return t_fwd_total / n_block
    return 0.0


def _extract_loss(out) -> "torch.Tensor":
    """Pull a backwards-able scalar loss out of a HuggingFace forward output.

    Handles both attribute-style (``CausalLMOutput.loss``) and
    dict-style (``out["loss"]``) returns. Raises if neither is
    present — phase-2 needs a ``.backward()``-able tensor.
    """
    loss = getattr(out, "loss", None)
    if loss is None and isinstance(out, dict):
        loss = out.get("loss")
    if loss is None:
        raise RuntimeError(
            "Phase-2 measurement: model forward returned no `loss` field. "
            "The dummy batch must include `labels` for HuggingFace causal "
            "LM heads to compute a backward-able loss."
        )
    return loss


__all__ = [
    "measure_chunked_steady",
    "select_bootstrap_config",
    "estimate_per_block_recompute_s",
]
