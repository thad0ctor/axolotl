"""Config-level validation gates for multimodal CPT (fail-at-load, not mid-train)."""

from __future__ import annotations

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


def _mm_cpt_cfg(min_base_cfg, **overrides) -> DictDefault:
    base = DictDefault(
        **(min_base_cfg | {
            "datasets": None,
            "pretraining_dataset": [
                {
                    "path": "some/ds",
                    "type": "multimodal_pretrain",
                    "image_column": "images",
                }
            ],
            "streaming": True,
            "max_steps": 10,
            "processor_type": "AutoProcessor",
            "sequence_len": 2048,
        })
    )
    return base | DictDefault(overrides)


class TestMultimodalCPTGates:
    def test_missing_processor_type_raises(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg)
        cfg.pop("processor_type", None)
        with pytest.raises(ValueError, match="processor_type"):
            validate_config(cfg)

    def test_sample_packing_rejected(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg, sample_packing=True)
        with pytest.raises(ValueError, match="sample_packing"):
            validate_config(cfg)

    def test_chat_template_rejected(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg, chat_template="tokenizer_default")
        with pytest.raises(ValueError, match="chat_template"):
            validate_config(cfg)

    def test_valid_cfg_passes_and_disables_remove_unused_columns(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg)
        validated = validate_config(cfg)
        assert validated.remove_unused_columns is False
        # new schema fields round-trip through the pretraining_dataset entry
        pd = validated.pretraining_dataset[0]
        assert pd.type == "multimodal_pretrain"
        assert pd.image_column == "images"

    def test_multimodal_flag_triggers_gates(self, min_base_cfg):
        """`multimodal: true` on the row should also activate the gates even
        without `type: multimodal_pretrain`."""
        cfg = _mm_cpt_cfg(min_base_cfg)
        cfg.pretraining_dataset[0]["type"] = "pretrain"
        cfg.pretraining_dataset[0]["multimodal"] = True
        cfg.pop("processor_type", None)
        with pytest.raises(ValueError, match="processor_type"):
            validate_config(cfg)

    def test_non_mm_pretraining_dataset_unaffected(self, min_base_cfg):
        """Pure text pretraining_dataset should remain valid without the new fields."""
        cfg = DictDefault(
            **(min_base_cfg | {
                "datasets": None,
                "pretraining_dataset": [
                    {"path": "some/ds", "type": "pretrain"}
                ],
                "streaming": True,
                "max_steps": 10,
                "sequence_len": 2048,
            })
        )
        validate_config(cfg)  # must not raise
