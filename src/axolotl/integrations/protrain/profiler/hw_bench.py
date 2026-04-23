"""Hardware microbenchmarks: PCIe H2D/D2H + NCCL collectives."""

from __future__ import annotations

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def measure_pcie(
    device_idx: int = 0,
    n_bytes: int = 256 * 1024 * 1024,
    n_iters: int = 5,
) -> tuple[float, float]:
    """Measure sustained H2D and D2H bandwidth on a single device.

    Uses a pinned host tensor and ``torch.cuda.Event`` for timing. Returns
    ``(h2d_bps, d2h_bps)`` in bytes/sec.

    Args:
        device_idx: CUDA device ordinal.
        n_bytes: payload size. 256 MiB is large enough to saturate PCIe 4.0 x16
            on a 3090 (~26 GB/s peak) without blowing up small-device budgets.
        n_iters: repetitions — the first is a warmup and is discarded.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("measure_pcie requires CUDA.")

    device = torch.device(f"cuda:{device_idx}")

    # uint8 so n_bytes == numel(); pinned host memory for true async copies.
    host = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(n_bytes, dtype=torch.uint8, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def _time_copy(src, dst) -> float:
        torch.cuda.synchronize(device)
        start.record()
        dst.copy_(src, non_blocking=True)
        end.record()
        torch.cuda.synchronize(device)
        # elapsed_time is in ms
        return start.elapsed_time(end) / 1000.0

    # Warmup + measured iters, H2D
    h2d_times: list[float] = []
    for i in range(n_iters + 1):
        t = _time_copy(host, gpu)
        if i > 0:
            h2d_times.append(t)

    d2h_times: list[float] = []
    for i in range(n_iters + 1):
        t = _time_copy(gpu, host)
        if i > 0:
            d2h_times.append(t)

    h2d_bps = n_bytes / (sum(h2d_times) / len(h2d_times))
    d2h_bps = n_bytes / (sum(d2h_times) / len(d2h_times))

    LOG.debug(
        "measure_pcie device=%d h2d=%.2f GB/s d2h=%.2f GB/s",
        device_idx,
        h2d_bps / 1e9,
        d2h_bps / 1e9,
    )
    return h2d_bps, d2h_bps


def measure_nccl(world_size: int) -> dict[int, tuple[float, float]]:
    """Measure NCCL gather/reduce latencies per payload size.

    Single-rank fast path returns an empty dict — there is no NCCL traffic on
    ``world_size == 1`` and the searcher simply skips the collective term.

    Multi-rank path requires a proper ``torch.distributed`` rendezvous (env
    vars ``MASTER_ADDR``, ``MASTER_PORT``, ``WORLD_SIZE``, ``RANK``). That
    plumbing is scheduled for M6 — today we raise to make the gap explicit.
    """
    if world_size == 1:
        return {}
    raise NotImplementedError(
        "measure_nccl requires a distributed rendezvous — M6 will exercise this."
    )


__all__ = ["measure_pcie", "measure_nccl"]
