"""Resolve the base model a gate run exercises: a user-supplied path, an opt-in tiny checkpoint matched by architecture, or a 2-layer model shrunk from the target's own config."""

from __future__ import annotations

import logging
from pathlib import Path

# arch keyword (substring of model_config_type) -> tiny HF id
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


# dims shrunk when present; hidden_size%heads and heads%kv_heads stay integral
_SHRINK_DIMS: dict[str, int] = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
}
# MoE knobs shrunk when present; experts_per_tok reconciled against experts below
_SHRINK_MOE: dict[str, int] = {
    "num_experts": 4,
    "num_local_experts": 4,
    "n_routed_experts": 4,
    "moe_intermediate_size": 128,
    "shared_expert_intermediate_size": 128,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
}
_EXPERT_TOTAL_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


def _set_if_present(config: object, key: str, value: int) -> None:
    if getattr(config, key, None) is not None:
        setattr(config, key, value)


def _shrink_model(base_model: str, output_dir: Path, trust_remote_code: bool) -> str:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    config.num_hidden_layers = 2
    for key, value in {**_SHRINK_DIMS, **_SHRINK_MOE}.items():
        _set_if_present(config, key, value)
    # vocab_size is left intact: the saved tokenizer's vocab must still fit the embedding
    for ek in _EXPERT_TOTAL_KEYS:
        n = getattr(config, ek, None)
        if isinstance(n, int) and getattr(config, "num_experts_per_tok", 0) > n:
            config.num_experts_per_tok = n
    # per-layer schedules must match the reduced depth
    for key in ("layer_types", "no_rope_layers", "sliding_window_pattern"):
        val = getattr(config, key, None)
        if isinstance(val, list) and len(val) > config.num_hidden_layers:
            setattr(config, key, val[: config.num_hidden_layers])
    config.tie_word_embeddings = True

    model_type = getattr(config, "model_type", None) or "model"
    dest = Path(output_dir) / f"shrunk_{model_type}"
    try:
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code
        )
    except Exception as exc:  # noqa: BLE001 - surface as could-not-run upstream
        raise RuntimeError(
            f"could not build a shrunk model for {base_model!r} "
            f"(model_type={model_type}): {exc.__class__.__name__}: {exc}"
        ) from exc
    model.save_pretrained(dest)
    try:
        AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=trust_remote_code
        ).save_pretrained(dest)
    except Exception:  # noqa: BLE001 - tokenizer is best-effort
        logging.getLogger(__name__).debug(
            "tokenizer save skipped for shrunk model", exc_info=True
        )
    return str(dest)


def resolve_base_model(
    base_model: str | None,
    model_config_type: str,
    strategy: str = "path",
    output_dir: Path | None = None,
    trust_remote_code: bool = False,
) -> str:
    """Return the base model to verify against per ``strategy`` ('path' | 'checkpoint' | 'shrink')."""
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

    if strategy == "shrink":
        if not base_model:
            raise ValueError("strategy='shrink' requires an explicit base_model.")
        if output_dir is None:
            raise ValueError("strategy='shrink' requires an output_dir to write into.")
        return _shrink_model(base_model, output_dir, trust_remote_code)

    raise ValueError(
        f"unknown strategy {strategy!r}; expected 'path', 'checkpoint', or 'shrink'."
    )
