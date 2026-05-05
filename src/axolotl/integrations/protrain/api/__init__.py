"""Public user-facing wrappers for the ProTrain runtime (§1).

Two entry points compose the full M1-M4 pipeline:

* :func:`protrain_model_wrapper` — called once after model
  construction; runs profiler (cached), layout, searcher, and installs
  block hooks.
* :func:`protrain_optimizer_wrapper` — replaces the user's
  ``torch.optim.AdamW`` with the GPU/CPU FusedAdam adapter pair that
  the scheduler drives under the hood.
"""

from __future__ import annotations

from axolotl.integrations.protrain.api.model_wrapper import protrain_model_wrapper
from axolotl.integrations.protrain.api.optim_wrapper import protrain_optimizer_wrapper

__all__ = [
    "protrain_model_wrapper",
    "protrain_optimizer_wrapper",
]
