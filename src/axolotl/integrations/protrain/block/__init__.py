"""ProTrain block-manager subpackage (§3.1.2).

Public surface:

- ``BlockMode`` — activation strategy enum (re-exported from ``types.py``).
- ``wrap_block`` / ``unwrap_block`` — per-block mode dispatcher.
- ``assign_modes`` — layout rules (swap-early, unopt-late, interleave).
- ``discover_blocks`` — find the transformer-block ModuleList on a model.
"""

from __future__ import annotations

from axolotl.integrations.protrain.block.dispatcher import unwrap_block, wrap_block
from axolotl.integrations.protrain.block.layout_rules import (
    assign_modes,
    discover_blocks,
)
from axolotl.integrations.protrain.block.strategy import (
    BlockMode,
    BlockStrategyMap,
    StrategyError,
)

__all__ = [
    "BlockMode",
    "BlockStrategyMap",
    "StrategyError",
    "wrap_block",
    "unwrap_block",
    "assign_modes",
    "discover_blocks",
]
