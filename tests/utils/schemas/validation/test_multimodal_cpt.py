"""Multimodal CPT config validation gates."""

from __future__ import annotations

import logging

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


def _mm_cpt_cfg(min_base_cfg, **overrides) -> DictDefault:
    base = DictDefault(
        **(
            dict(min_base_cfg)
            | {
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
            }
        )
    )
    return DictDefault(dict(base) | dict(overrides))


def _mm_cpt_datasets_cfg(min_base_cfg, **overrides) -> DictDefault:
    base = DictDefault(
        **(
            dict(min_base_cfg)
            | {
                "datasets": [
                    {
                        "path": "some/ds",
                        "type": "multimodal_pretrain",
                        "text_column": "caption",
                        "image_column": "images",
                        "image_base_dir": "/images",
                    }
                ],
                "pretraining_dataset": None,
                "streaming": False,
                "processor_type": "AutoProcessor",
                "sequence_len": 2048,
            }
        )
    )
    return DictDefault(dict(base) | dict(overrides))


class TestMultimodalCPTGates:
    def test_missing_processor_type_raises(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg)
        cfg.pop("processor_type", None)
        with pytest.raises(ValueError, match="processor_type"):
            validate_config(cfg)

    def test_sample_packing_with_streaming_cache_rejected(self, min_base_cfg):
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            sample_packing=True,
            dataset_prepared_path="/tmp/mm-cpt-cache",
        )
        with pytest.raises(ValueError, match="dataset_prepared_path"):
            validate_config(cfg)

    def test_sample_packing_allowed_for_streaming_pretraining(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg, sample_packing=True)
        validated = validate_config(cfg)
        assert validated.sample_packing is True

    def test_multimodal_sample_packing_cache_and_dataloader_flags(self, min_base_cfg):
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            sample_packing=True,
            multimodal_sample_packing=True,
            multimodal_sample_packing_cache_path="/tmp/mm-pack-cache",
            multimodal_sample_packing_ram_budget_mb=512,
            multimodal_sample_packing_split_ram_budget_by_worker=True,
            multimodal_sample_packing_visual_capacity=8192,
            multimodal_sample_packing_group_by_visual_signature=True,
            multimodal_sample_packing_dataloader=True,
            multimodal_sample_packing_dataloader_num_workers=4,
            multimodal_sample_packing_dataloader_prefetch_factor=6,
            image_resize_buckets=[[1024, 1536], [1536, 1536]],
            image_resize_no_upscale=True,
            image_resize_pad_color=[255, 255, 255],
        )
        validated = validate_config(cfg)
        assert validated.multimodal_sample_packing is True
        assert validated.multimodal_sample_packing_cache_path == "/tmp/mm-pack-cache"
        assert validated.multimodal_sample_packing_ram_budget_mb == 512
        assert validated.multimodal_sample_packing_split_ram_budget_by_worker is True
        assert validated.multimodal_sample_packing_visual_capacity == 8192
        assert validated.multimodal_sample_packing_group_by_visual_signature is True
        assert validated.multimodal_sample_packing_dataloader is True
        assert validated.multimodal_sample_packing_dataloader_num_workers == 4
        assert validated.multimodal_sample_packing_dataloader_prefetch_factor == 6
        assert validated.image_resize_buckets == [(1024, 1536), (1536, 1536)]
        assert validated.image_resize_no_upscale is True
        assert validated.image_resize_pad_color == (255, 255, 255)

    def test_sample_packing_allowed_for_nonstreaming_datasets(self, min_base_cfg):
        cfg = _mm_cpt_datasets_cfg(min_base_cfg, sample_packing=True)
        validated = validate_config(cfg)
        assert validated.sample_packing is True

    def test_multimodal_image_tiling_config_validates(self, min_base_cfg):
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            image_tiling=True,
            image_tiling_tile_size=[1024, 1024],
            image_tiling_grid=[2, 3],
            image_tiling_overlap=0.15,
            image_tiling_min_area=1_000_000,
            image_tiling_overview_buckets=[[1024, 1536], [1536, 1024]],
            image_tiling_no_upscale=True,
            image_tiling_pad_color=[255, 255, 255],
            image_tiling_cache_path="/tmp/mm-tile-cache",
            image_tiling_cache_hash_images=True,
            image_tiling_shape_buckets="ocr_pages",
        )

        validated = validate_config(cfg)

        assert validated.image_tiling is True
        assert validated.image_tiling_tile_size == (1024, 1024)
        assert validated.image_tiling_grid == (2, 3)
        assert validated.image_tiling_overview_buckets == [
            (1024, 1536),
            (1536, 1024),
        ]
        assert validated.image_tiling_pad_color == (255, 255, 255)
        assert validated.image_tiling_shape_buckets == "ocr_pages"

    def test_chat_template_rejected(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg, chat_template="tokenizer_default")
        with pytest.raises(ValueError, match="chat_template"):
            validate_config(cfg)

    def test_multiple_pretraining_dataset_entries_rejected(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg)
        cfg.pretraining_dataset.append({"path": "other/ds", "type": "pretrain"})
        with pytest.raises(ValueError, match="exactly one `pretraining_dataset`"):
            validate_config(cfg)

    def test_multimodal_entry_in_non_first_slot_rejected(self, min_base_cfg):
        cfg = DictDefault(
            **(
                min_base_cfg
                | {
                    "datasets": None,
                    "pretraining_dataset": [
                        {"path": "text/ds", "type": "pretrain"},
                        {
                            "path": "mm/ds",
                            "type": "multimodal_pretrain",
                            "image_column": "images",
                        },
                    ],
                    "streaming": True,
                    "max_steps": 10,
                    "processor_type": "AutoProcessor",
                    "sequence_len": 2048,
                }
            )
        )
        with pytest.raises(ValueError, match="exactly one `pretraining_dataset`"):
            validate_config(cfg)

    def test_valid_cfg_passes_and_disables_remove_unused_columns(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg)
        validated = validate_config(cfg)
        assert validated.remove_unused_columns is False
        pd = validated.pretraining_dataset[0]
        assert pd.type == "multimodal_pretrain"
        assert pd.image_column == "images"

    def test_valid_datasets_cfg_preserves_mm_keys(self, min_base_cfg):
        cfg = _mm_cpt_datasets_cfg(min_base_cfg)
        validated = validate_config(cfg)
        assert validated.remove_unused_columns is False
        ds = validated.datasets[0]
        assert ds.type == "multimodal_pretrain"
        assert ds.text_column == "caption"
        assert ds.image_column == "images"
        assert ds.image_base_dir == "/images"

    def test_datasets_cfg_allows_num_epochs_without_max_steps(self, min_base_cfg):
        cfg = _mm_cpt_datasets_cfg(min_base_cfg, num_epochs=2)
        cfg.pop("max_steps", None)
        validated = validate_config(cfg)
        assert validated.max_steps is None
        assert validated.num_epochs == 2

    def test_datasets_cfg_missing_processor_type_raises(self, min_base_cfg):
        cfg = _mm_cpt_datasets_cfg(min_base_cfg)
        cfg.pop("processor_type", None)
        with pytest.raises(ValueError, match="processor_type"):
            validate_config(cfg)

    def test_datasets_cfg_with_streaming_rejected(self, min_base_cfg):
        cfg = _mm_cpt_datasets_cfg(min_base_cfg, streaming=True, max_steps=10)
        with pytest.raises(ValueError, match="non-streaming prepared path"):
            validate_config(cfg)

    def test_multiple_datasets_entries_rejected(self, min_base_cfg):
        cfg = _mm_cpt_datasets_cfg(min_base_cfg)
        cfg.datasets.append({"path": "other/ds", "type": "alpaca"})
        with pytest.raises(ValueError, match="exactly one `datasets`"):
            validate_config(cfg)

    def test_datasets_and_pretraining_mm_entries_rejected(self, min_base_cfg):
        cfg = _mm_cpt_datasets_cfg(
            min_base_cfg,
            pretraining_dataset=[{"path": "stream/ds", "type": "multimodal_pretrain"}],
        )
        with pytest.raises(
            ValueError, match="both `pretraining_dataset` and `datasets`"
        ):
            validate_config(cfg)

    def test_datasets_cfg_rejects_truncate_strategy(self, min_base_cfg):
        cfg = _mm_cpt_datasets_cfg(min_base_cfg, excess_length_strategy="truncate")
        with pytest.raises(ValueError, match="excess_length_strategy: truncate"):
            validate_config(cfg)

    def test_datasets_cfg_requires_strategy_type_not_multimodal_flag(
        self, min_base_cfg
    ):
        cfg = _mm_cpt_datasets_cfg(min_base_cfg)
        cfg.datasets[0].pop("type")
        cfg.datasets[0]["multimodal"] = True
        with pytest.raises(ValueError, match="type: multimodal_pretrain"):
            validate_config(cfg)

    def test_multimodal_flag_triggers_gates(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg)
        cfg.pretraining_dataset[0]["type"] = "pretrain"
        cfg.pretraining_dataset[0]["multimodal"] = True
        cfg.pop("processor_type", None)
        with pytest.raises(ValueError, match="processor_type"):
            validate_config(cfg)

    def test_non_mm_pretraining_dataset_unaffected(self, min_base_cfg):
        cfg = DictDefault(
            **(
                min_base_cfg
                | {
                    "datasets": None,
                    "pretraining_dataset": [{"path": "some/ds", "type": "pretrain"}],
                    "streaming": True,
                    "max_steps": 10,
                    "sequence_len": 2048,
                }
            )
        )
        validate_config(cfg)

    def test_mm_eval_dataset_keys_preserved_through_validation(self, min_base_cfg):
        """MM-specific keys on a test_datasets entry survive validate_config."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {
                    "path": "eval/ds",
                    "type": "multimodal_pretrain",
                    "text_column": "eval_text",
                    "image_column": "eval_imgs",
                    "image_base_dir": "/eval/images",
                    "image_token": "<my_img>",
                }
            ],
        )
        validated = validate_config(cfg)
        td = validated.test_datasets[0]
        assert td["text_column"] == "eval_text"
        assert td["image_column"] == "eval_imgs"
        assert td["image_base_dir"] == "/eval/images"
        assert td["image_token"] == "<my_img>"

    def test_mm_eval_dataset_via_multimodal_flag(self, min_base_cfg):
        """`multimodal: true` (without type='multimodal_pretrain') opts an eval entry into MM."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {
                    "path": "eval/ds",
                    "multimodal": True,
                    "image_column": "imgs2",
                }
            ],
        )
        validated = validate_config(cfg)
        td = validated.test_datasets[0]
        assert td["image_column"] == "imgs2"
        assert td["multimodal"] is True

    def test_non_mm_eval_entry_does_not_match_mm_model(self, min_base_cfg):
        """SFT eval entries (no MM markers) still validate as SFTDataset."""
        cfg = DictDefault(
            **(
                min_base_cfg
                | {
                    "test_datasets": [
                        {"path": "eval/ds", "type": "alpaca", "split": "test"}
                    ],
                    "sequence_len": 2048,
                }
            )
        )
        validated = validate_config(cfg)
        td = validated.test_datasets[0]
        assert "message_property_mappings" in td
        assert td["type"] == "alpaca"

    def test_mm_eval_rejects_mismatched_image_base_dir(self, min_base_cfg):
        """Multiple MM eval entries with different image_base_dir are rejected."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {
                    "path": "eval/a",
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/images/a",
                },
                {
                    "path": "eval/b",
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/images/b",
                },
            ],
        )
        with pytest.raises(ValueError, match="image_base_dir"):
            validate_config(cfg)

    def test_mm_eval_rejects_mismatched_image_token(self, min_base_cfg):
        """Multiple MM eval entries with different image_token overrides are rejected."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {
                    "path": "eval/a",
                    "type": "multimodal_pretrain",
                    "image_token": "<img_a>",
                },
                {
                    "path": "eval/b",
                    "type": "multimodal_pretrain",
                    "image_token": "<img_b>",
                },
            ],
        )
        with pytest.raises(ValueError, match="image_token"):
            validate_config(cfg)

    def test_mm_eval_accepts_matching_image_base_dir(self, min_base_cfg):
        """Multiple MM eval entries sharing image_base_dir validate cleanly."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {
                    "path": "eval/a",
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/images/shared",
                },
                {
                    "path": "eval/b",
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/images/shared",
                },
            ],
        )
        validated = validate_config(cfg)
        assert len(validated.test_datasets) == 2

    def test_mm_eval_accepts_all_unset_image_settings(self, min_base_cfg):
        """Multiple MM eval entries with image_base_dir / image_token unset everywhere validate."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {"path": "eval/a", "type": "multimodal_pretrain"},
                {"path": "eval/b", "type": "multimodal_pretrain"},
            ],
        )
        validated = validate_config(cfg)
        assert len(validated.test_datasets) == 2

    def test_mixed_modality_test_datasets_rejected_at_validation(self, min_base_cfg):
        """A test_datasets list mixing MM and non-MM entries fails at config-load."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {"path": "eval/a", "type": "multimodal_pretrain"},
                {"path": "eval/b", "type": "alpaca", "split": "test"},
            ],
        )
        with pytest.raises(ValueError) as exc:
            validate_config(cfg)
        msg = str(exc.value)
        assert "Mixing multimodal and non-multimodal" in msg
        assert "test_datasets" in msg
        assert "share modality" in msg

    def test_mm_test_datasets_with_text_training_rejected(self, min_base_cfg):
        """MM test_datasets paired with non-MM training fails at config-load."""
        cfg = DictDefault(
            **(
                min_base_cfg
                | {
                    "datasets": None,
                    "pretraining_dataset": [{"path": "text/ds", "type": "pretrain"}],
                    "test_datasets": [
                        {"path": "eval/a", "type": "multimodal_pretrain"}
                    ],
                    "streaming": True,
                    "max_steps": 10,
                    "sequence_len": 2048,
                    "processor_type": "AutoProcessor",
                }
            )
        )
        with pytest.raises(ValueError) as exc:
            validate_config(cfg)
        msg = str(exc.value)
        assert "Multimodal `test_datasets`" in msg
        assert "multimodal CPT training" in msg
        assert "multimodal_pretrain" in msg

    def test_text_test_datasets_with_mm_training_rejected(self, min_base_cfg):
        """Non-MM test_datasets paired with MM training fails at config-load."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[{"path": "eval/a", "type": "alpaca", "split": "test"}],
        )
        with pytest.raises(ValueError) as exc:
            validate_config(cfg)
        msg = str(exc.value)
        assert "Multimodal CPT training" in msg
        assert "multimodal `test_datasets`" in msg
        assert "multimodal_pretrain" in msg

    def test_remove_unused_columns_auto_set_emits_info_log(
        self, min_base_cfg, caplog, monkeypatch
    ):
        """Auto-setting `remove_unused_columns: false` for MM CPT logs an INFO record naming the previous value."""
        # `axolotl` logger has propagate=False (logging_config.py); flip it so
        # caplog's root handler receives the record.
        monkeypatch.setattr(logging.getLogger("axolotl"), "propagate", True)
        cfg = _mm_cpt_cfg(min_base_cfg)
        cfg.pop("remove_unused_columns", None)
        with caplog.at_level(logging.INFO, logger="axolotl.utils.schemas.validation"):
            validated = validate_config(cfg)
        assert validated.remove_unused_columns is False
        matches = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "Auto-set" in r.getMessage()
        ]
        assert matches, (
            "expected an INFO record about auto-setting remove_unused_columns"
        )
        msg = matches[0].getMessage()
        assert "remove_unused_columns" in msg
        assert "previous value: None" in msg

    def test_remove_unused_columns_already_false_does_not_log(
        self, min_base_cfg, caplog, monkeypatch
    ):
        """When the user already set `remove_unused_columns: false`, no auto-set log fires."""
        # Mirror the positive test: flip propagate on the parent `axolotl`
        # logger (the one with propagate=False), not the leaf — otherwise
        # caplog never sees axolotl.* records and this negative assertion
        # is vacuous (it would pass even if the auto-set log fired).
        monkeypatch.setattr(logging.getLogger("axolotl"), "propagate", True)
        cfg = _mm_cpt_cfg(min_base_cfg, remove_unused_columns=False)
        with caplog.at_level(logging.INFO, logger="axolotl.utils.schemas.validation"):
            validate_config(cfg)
        assert not any(
            "Auto-set" in r.getMessage() and "remove_unused_columns" in r.getMessage()
            for r in caplog.records
        )


class TestImageTilingSchema:
    """Validation for the image_tiling_* config surface."""

    @pytest.mark.parametrize("overlap", [-0.5, 0.5, 0.9])
    def test_overlap_out_of_range_rejected(self, overlap):
        from pydantic import ValidationError

        from axolotl.utils.schemas.multimodal import MultiModalConfig

        with pytest.raises(ValidationError):
            MultiModalConfig(image_tiling=True, image_tiling_overlap=overlap)

    def test_overlap_in_range_accepted(self):
        from axolotl.utils.schemas.multimodal import MultiModalConfig

        cfg = MultiModalConfig(image_tiling=True, image_tiling_overlap=0.25)
        assert cfg.image_tiling_overlap == 0.25

    def test_bad_shape_bucket_preset_clean_error(self):
        from pydantic import ValidationError

        from axolotl.utils.schemas.multimodal import MultiModalConfig

        with pytest.raises(ValidationError, match="ocr_pages"):
            MultiModalConfig(image_tiling=True, image_tiling_shape_buckets="ocr")

    def test_shape_bucket_presets_accepted(self):
        from axolotl.utils.schemas.multimodal import MultiModalConfig

        assert (
            MultiModalConfig(
                image_tiling=True, image_tiling_shape_buckets="ocr_pages"
            ).image_tiling_shape_buckets
            == "ocr_pages"
        )

    def test_tiling_fields_without_enable_warns(self, caplog, monkeypatch):
        from axolotl.utils.schemas.multimodal import MultiModalConfig

        monkeypatch.setattr(logging.getLogger("axolotl"), "propagate", True)
        with caplog.at_level(
            logging.WARNING, logger="axolotl.utils.schemas.multimodal"
        ):
            MultiModalConfig(
                image_tiling=False,
                image_tiling_tile_size=1024,
                image_tiling_shape_buckets="ocr_pages",
            )
        assert any("image_tiling is disabled" in r.getMessage() for r in caplog.records)

    def test_no_warn_when_no_tiling_fields_set(self, caplog, monkeypatch):
        from axolotl.utils.schemas.multimodal import MultiModalConfig

        monkeypatch.setattr(logging.getLogger("axolotl"), "propagate", True)
        with caplog.at_level(
            logging.WARNING, logger="axolotl.utils.schemas.multimodal"
        ):
            MultiModalConfig(image_tiling=False)
        assert not any(
            "image_tiling is disabled" in r.getMessage() for r in caplog.records
        )
