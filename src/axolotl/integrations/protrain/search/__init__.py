"""ProTrain 4-knob searcher (M4).

Public surface:

- ``derive_bounds`` — upper bounds on the four tunable knobs.
- ``search`` — exhaustive enumeration with OOM pruning; returns the
  minimum-runtime ``SearchResult`` that fits under the given GPU
  capacity.
"""

from __future__ import annotations

from axolotl.integrations.protrain.search.exhaustive import search
from axolotl.integrations.protrain.search.knobs import derive_bounds

__all__ = ["derive_bounds", "search"]
