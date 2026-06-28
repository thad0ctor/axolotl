"""Resolve the base model a gate run should exercise.

Per the harness decision, verification runs against a *user-supplied* base model
(``strategy="path"``); we deliberately do not shrink-from-config. ``strategy=
"checkpoint"`` is an opt-in convenience that maps an architecture to a known
tiny checkpoint from the ``axolotl-ai-co/tiny-*`` family used across the repo's
own tests, for when the caller has no model in hand.
"""

from __future__ import annotations

# arch keyword (matched as a substring of model_config_type) -> tiny HF id.
# Ids are the ones axolotl's own tests/examples pull (see tests/**, examples/**).
TINY_CHECKPOINTS: dict[str, str] = {
    "llama": "axolotl-ai-co/tiny-llama-50m",
    "mixtral": "axolotl-ai-co/tiny-mixtral-30m",
    "mistral": "axolotl-ai-co/tiny-mistral-25m",
    "qwen3": "axolotl-ai-co/tiny-qwen3-129m",
    "qwen2": "axolotl-ai-co/tiny-qwen2-129m",
    "gemma2": "axolotl-ai-co/tiny-gemma2-137m",
    "falcon": "axolotl-ai-co/tiny-falcon-42m",
    "phi": "axolotl-ai-co/tiny-phi-64m",
    "smollm": "HuggingFaceTB/SmolLM2-135M",
}


def resolve_base_model(
    base_model: str | None,
    model_config_type: str,
    strategy: str = "path",
) -> str:
    """Return the base model path/id to verify against.

    ``path``       require and pass through ``base_model``.
    ``checkpoint`` fall back to a tiny checkpoint matched by architecture when
                   ``base_model`` is None.
    """
    if strategy == "path":
        if not base_model:
            raise ValueError(
                "strategy='path' requires an explicit base_model (path or HF id)."
            )
        return base_model

    if strategy == "checkpoint":
        if base_model:
            return base_model
        key = (model_config_type or "").lower()
        for keyword, hf_id in TINY_CHECKPOINTS.items():
            if keyword in key:
                return hf_id
        raise ValueError(
            f"no tiny checkpoint known for model_config_type={model_config_type!r}; "
            f"pass an explicit base_model. Known: {sorted(TINY_CHECKPOINTS)}"
        )

    raise ValueError(f"unknown strategy {strategy!r}; expected 'path' or 'checkpoint'.")
