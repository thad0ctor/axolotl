"""Model-support dispatch at the save boundary.

`save_pretrained` reverses the checkpoint conversions that were used to load the
model, so the registry state at save time decides what key names end up in the
checkpoint. This boundary re-applies a profile's conversions and runs its
`ModelHookPhase.BEFORE_SAVE` hooks before any weights are written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .hf_registries import apply_checkpoint_conversions
from .profile import ModelHookContext, ModelHookPhase, run_model_support_hooks
from .registry import get_model_support_for_cfg

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

    from axolotl.utils.dict import DictDefault


def prepare_model_for_save(
    cfg: DictDefault | None,
    model: PreTrainedModel | PeftModel | None = None,
) -> None:
    """Run the model-support save boundary; a no-op for unregistered models.

    Invoked once per checkpoint as well as for the final save, so both the
    conversion adapters and the hooks it dispatches must be idempotent.
    """
    if cfg is None:
        return
    support = get_model_support_for_cfg(cfg)
    if support is None:
        return

    apply_checkpoint_conversions(support, cfg, model=model)
    run_model_support_hooks(
        support,
        ModelHookPhase.BEFORE_SAVE,
        ModelHookContext(cfg=cfg, model=model),
    )
