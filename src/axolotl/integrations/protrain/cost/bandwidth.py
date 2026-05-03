"""Effective PCIe bandwidth model for the ProTrain cost estimators (§3.3).

When ``n_swap > 0`` activation-swap traffic (forward offload, backward
prefetch) competes with chunk prefetch/offload traffic on the same PCIe
link. ProTrain's cost model derates the prefetch bandwidth so the
runtime estimator does not under-predict backward time.

This is a first-order model — a single scalar derate per direction.
Refine against measured contention if a later test shows a >5% runtime
mismatch vs. observed ``torch.cuda.Event`` timing.

Paper references: §3.3 "bandwidth contention is modeled explicitly".
"""

from __future__ import annotations

from axolotl.integrations.protrain.types import CostConfig, HardwareProfile
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def effective_bw(cfg: CostConfig, hw: HardwareProfile) -> tuple[float, float]:
    """Return ``(effective_h2d_bps, effective_d2h_bps)`` under SWAP contention.

    When ``cfg.n_swap == 0`` the raw PCIe bandwidths are returned unchanged.
    When ``cfg.n_swap > 0`` the effective bandwidth for chunk prefetch is
    reduced by a factor ``1 / (1 + 0.5 * min(1, n_swap / max(1, gpu_count)))``.
    The factor bottoms out at ``2/3`` when every rank has at least one swap
    block competing for the link — matching the paper's qualitative claim
    that "unlimited" swap degrades prefetch throughput by roughly a third.

    Parameters
    ----------
    cfg:
        The candidate knob configuration being costed.
    hw:
        Static hardware description; only ``pcie_h2d_bps``,
        ``pcie_d2h_bps``, and ``gpu_count`` are consulted.

    Returns
    -------
    tuple[float, float]
        Effective H2D and D2H bandwidths in bytes / second.
    """
    gpu_count = max(1, hw.gpu_count)
    if cfg.n_swap <= 0:
        return hw.pcie_h2d_bps, hw.pcie_d2h_bps

    # First-order contention model. See module docstring for refinement
    # guidance; the 0.5 slope and the clamp at gpu_count were picked to
    # keep the derate monotone in n_swap without letting a single swap
    # block on one rank halve the bandwidth for the entire cluster.
    contention = 0.5 * min(1.0, cfg.n_swap / gpu_count)
    denom = 1.0 + contention
    eff_h2d = hw.pcie_h2d_bps / denom
    eff_d2h = hw.pcie_d2h_bps / denom
    LOG.debug(
        "effective_bw: n_swap=%d gpu_count=%d derate=%.3f h2d=%.2e d2h=%.2e",
        cfg.n_swap,
        gpu_count,
        denom,
        eff_h2d,
        eff_d2h,
    )
    return eff_h2d, eff_d2h


__all__ = ["effective_bw"]
