"""Sparse linear for 4-bit (QLoRA) base weights.

The vendored ``SparseLinear`` channel-slices a dense ``self.weight``; a
bitsandbytes ``Linear4bit`` stores packed 4-bit data instead, so we dequantize
the (frozen) base weight and apply the same out-/in-sparse slicing.

This is efficient because bitsandbytes' own ``Linear4bit.forward`` already
dequantizes the full weight to the compute dtype at training batch sizes — the
sparsity win is purely a smaller matmul, which a microbenchmark confirms lands
within ~5-15% of the bf16 SparseLoRA path for output-sparse projections.

Registered via the vendored ``register_sparse_module`` extension point, so it
requires no edit to the sparse linear itself.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ._vendor.sparselora.modules.linear import SparseLinear


class SparseLinear4bit(SparseLinear):
    """Channel-sparse linear backed by a frozen bitsandbytes 4-bit weight."""

    def __init__(self, base: nn.Module, mode: Optional[str]) -> None:
        # Skip nn.Linear.__init__ to avoid allocating a dense float weight; the
        # 4-bit Params stays the storage.
        nn.Module.__init__(self)
        self.weight = base.weight  # bnb Params4bit (frozen)
        self.quant_state = base.weight.quant_state
        self.mode = mode  # type: ignore[assignment]  # base types this str; None is valid
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = base.bias

    def _dense_weight(self, dtype: torch.dtype) -> torch.Tensor:
        from bitsandbytes.functional import dequantize_4bit

        return dequantize_4bit(self.weight.data, self.quant_state).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        sparse_indices: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        x = x.contiguous()
        weight = self._dense_weight(x.dtype)
        if sparse_indices is None or self.mode is None:
            return F.linear(x, weight)

        if self.mode == "in_gather":
            idx = (
                sparse_indices.unsqueeze(0)
                .unsqueeze(0)
                .expand(x.shape[0], x.shape[1], -1)
            )
            with torch.no_grad():
                x = torch.gather(x, 2, idx).contiguous()

        if self.mode.startswith("in"):
            w = weight[:, sparse_indices].contiguous()
        else:
            w = weight[sparse_indices].contiguous()
        x = F.linear(x, w)

        if self.mode == "out_scatter" and x.shape[-1] != self.out_features:
            with torch.no_grad():
                out = torch.zeros(
                    x.shape[0],
                    x.shape[1],
                    self.out_features,
                    dtype=x.dtype,
                    device=x.device,
                )
                idx = (
                    sparse_indices.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(x.shape[0], x.shape[1], -1)
                )
                x = out.scatter_add(2, idx, x).contiguous()

        return x


def register_4bit_support() -> bool:
    """Register ``SparseLinear4bit`` for ``bitsandbytes.nn.Linear4bit``.

    Returns False if bitsandbytes is unavailable. Idempotent.
    """
    try:
        from bitsandbytes.nn import Linear4bit
    except Exception:  # noqa: BLE001 - bnb optional
        return False

    from ._vendor.sparselora.modules import register_sparse_module

    register_sparse_module(Linear4bit, SparseLinear4bit)
    return True
