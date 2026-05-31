"""
Regression tests for MultiModalChatDataCollator shape contracts.

Guard against the transformers 5.x breakage where apply_chat_template's
own `return_dict` parameter (default False) caused it to return the raw
input_ids tensor instead of the full BatchFeature dict, leading to
  IndexError: too many indices for tensor of dimension 2
when downstream code did batch["input_ids"] on the resulting tensor.
"""

from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from transformers import BatchFeature


@pytest.fixture(name="mock_processor")
def fixture_mock_processor():
    """
    A mock processor whose apply_chat_template returns a BatchFeature
    when called with return_dict=True (the correct call convention),
    or a raw input_ids tensor when called without return_dict=True
    (the broken call convention that the bug introduced).
    """
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    processor.tokenizer.pad_token_id = 0
    processor.image_token = "<|image|>"
    processor.tokenizer.convert_tokens_to_ids = MagicMock(return_value=128256)

    batch_size, seq_len = 2, 16
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    batch_feature = BatchFeature(
        data={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )

    def _apply_chat_template(*args, **kwargs):
        if kwargs.get("return_dict", False):
            return batch_feature
        # Simulate transformers 5.x default behaviour: returns out["input_ids"]
        return input_ids

    processor.apply_chat_template = MagicMock(side_effect=_apply_chat_template)
    processor.chat_template = None
    return processor


@pytest.fixture(name="mock_processing_strategy")
def fixture_mock_processing_strategy(mock_processor):
    from axolotl.processing_strategies import ProcessingStrategy

    strategy = ProcessingStrategy(processor=mock_processor)
    return strategy


class TestMultiModalChatDataCollatorShapeContract:
    """
    Verify that MultiModalChatDataCollator.process_rows returns a dict with
    2-D input_ids and labels, not a raw tensor.  This is the shape contract
    that process_labels depends on.
    """

    def _make_collator(self, mock_processing_strategy):
        from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator

        tokenizer = mock_processing_strategy.processor.tokenizer
        return MultiModalChatDataCollator(
            tokenizer=tokenizer,
            processing_strategy=mock_processing_strategy,
        )

    def _make_examples(self):
        return [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }
        ]

    def test_process_rows_returns_dict(self, mock_processing_strategy):
        """batch must be a dict, not a raw tensor."""
        collator = self._make_collator(mock_processing_strategy)
        examples = self._make_examples()

        with patch.object(
            mock_processing_strategy,
            "__call__",
            return_value=examples,
        ):
            batch = collator.process_rows(examples)

        assert isinstance(batch, Mapping), (
            "process_rows must return a Mapping (BatchFeature or dict), not a "
            "raw tensor. If it returns a tensor, apply_chat_template was called "
            "without return_dict=True at the top level."
        )

    def test_process_rows_input_ids_shape(self, mock_processing_strategy):
        """batch['input_ids'] must be a 2-D tensor (batch, seq_len)."""
        collator = self._make_collator(mock_processing_strategy)
        examples = self._make_examples()

        with patch.object(
            mock_processing_strategy,
            "__call__",
            return_value=examples,
        ):
            batch = collator.process_rows(examples)

        assert "input_ids" in batch
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert batch["input_ids"].ndim == 2, (
            f"input_ids must be 2-D (batch, seq_len), got shape {batch['input_ids'].shape}"
        )

    def test_process_rows_labels_shape(self, mock_processing_strategy):
        """batch['labels'] must be a 2-D tensor matching input_ids shape."""
        collator = self._make_collator(mock_processing_strategy)
        examples = self._make_examples()

        with patch.object(
            mock_processing_strategy,
            "__call__",
            return_value=examples,
        ):
            batch = collator.process_rows(examples)

        assert "labels" in batch
        assert isinstance(batch["labels"], torch.Tensor)
        assert batch["labels"].ndim == 2
        assert batch["labels"].shape == batch["input_ids"].shape

    def test_apply_chat_template_called_with_return_dict_true(
        self, mock_processing_strategy
    ):
        """apply_chat_template must be called with return_dict=True as a keyword arg."""
        collator = self._make_collator(mock_processing_strategy)
        examples = self._make_examples()

        with patch.object(
            mock_processing_strategy,
            "__call__",
            return_value=examples,
        ):
            collator.process_rows(examples)

        call_kwargs = (
            mock_processing_strategy.processor.apply_chat_template.call_args.kwargs
        )
        assert call_kwargs.get("return_dict") is True, (
            "apply_chat_template must be called with return_dict=True as a top-level "
            "keyword argument (not inside processor_kwargs). In transformers 5.x, "
            "apply_chat_template has its own return_dict param (default False) that "
            "controls whether it returns the full BatchFeature or just input_ids."
        )

    def test_trainer_multimodal_dataloader_detection_includes_sft(
        self, mock_processing_strategy
    ):
        from axolotl.core.trainers.base import _is_multimodal_data_collator

        collator = self._make_collator(mock_processing_strategy)

        assert _is_multimodal_data_collator(collator)

    def test_trainer_multimodal_dataloader_overrides_are_safe_with_zero_workers(
        self, mock_processing_strategy
    ):
        from axolotl.core.trainers.base import _apply_multimodal_dataloader_overrides
        from axolotl.utils.dict import DictDefault

        params = {
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True,
        }
        args = DictDefault(
            {
                "multimodal_sample_packing_dataloader": True,
                "multimodal_sample_packing_dataloader_num_workers": 0,
                "multimodal_sample_packing_dataloader_pin_memory": False,
                "multimodal_sample_packing_dataloader_persistent_workers": True,
            }
        )
        collator = self._make_collator(mock_processing_strategy)

        assert _apply_multimodal_dataloader_overrides(params, args, collator)
        assert params["num_workers"] == 0
        assert params["pin_memory"] is False
        assert params["persistent_workers"] is False

    def test_processing_strategy_receives_bucket_resize_policy(self, mock_processor):
        from axolotl.processing_strategies import get_processing_strategy
        from axolotl.utils.data.mm_tiling import ImageTilingConfig

        strategy = get_processing_strategy(
            mock_processor,
            chat_template=None,
            chat_template_type=None,
            image_resize_buckets=[(1024, 1536), (1536, 1536)],
            image_resize_no_upscale=True,
            image_resize_pad_color="white",
            image_tiling_config=ImageTilingConfig(tile_size=512, grid=(2, 3)),
        )

        assert strategy.image_resize_buckets == [(1024, 1536), (1536, 1536)]
        assert strategy.image_resize_no_upscale is True
        assert strategy.image_resize_pad_color == "white"
        assert strategy.image_tiling_config is not None

    def test_processing_strategy_tiling_expands_column_image(
        self, mock_processor, tmp_path
    ):
        from axolotl.processing_strategies import get_processing_strategy
        from axolotl.utils.data.mm_tiling import ImageTilingConfig

        image_path = tmp_path / "page.png"
        Image.new("RGB", (120, 180), "white").save(image_path)
        strategy = get_processing_strategy(
            mock_processor,
            chat_template=None,
            chat_template_type=None,
            image_tiling_config=ImageTilingConfig(
                tile_size=32,
                grid=(2, 1),
                overview_size=32,
            ),
        )

        out = strategy(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "OCR"},
                            ],
                        },
                        {"role": "assistant", "content": "text"},
                    ],
                    "images": [str(image_path)],
                }
            ]
        )

        image_items = [
            content
            for content in out[0]["messages"][0]["content"]
            if content["type"] == "image"
        ]
        assert len(image_items) == 3

    def test_processing_strategy_fills_multiple_bare_column_placeholders(
        self, mock_processor, tmp_path
    ):
        from axolotl.processing_strategies import get_processing_strategy

        image_paths = []
        for idx in range(2):
            image_path = tmp_path / f"page_{idx}.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            image_paths.append(str(image_path))
        strategy = get_processing_strategy(
            mock_processor,
            chat_template=None,
            chat_template_type=None,
        )

        out = strategy(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "first"},
                                {"type": "image"},
                                {"type": "text", "text": "second"},
                            ],
                        },
                        {"role": "assistant", "content": "text"},
                    ],
                    "images": image_paths,
                }
            ]
        )

        content = out[0]["messages"][0]["content"]
        image_items = [item for item in content if item["type"] == "image"]
        assert len(image_items) == 2
        assert all("image" in item for item in image_items)

    def test_non_tiling_caps_extra_column_images(self, mock_processor, tmp_path):
        """R2: without tiling, extra column images beyond placeholders are dropped
        (vanilla contract), not injected as un-anchored images."""
        from axolotl.processing_strategies import get_processing_strategy

        paths = []
        for idx in range(3):
            p = tmp_path / f"p{idx}.png"
            Image.new("RGB", (32, 32), "white").save(p)
            paths.append(str(p))
        strategy = get_processing_strategy(
            mock_processor, chat_template=None, chat_template_type=None
        )
        out = strategy(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "one"},
                            ],
                        },
                        {"role": "assistant", "content": "text"},
                    ],
                    "images": paths,  # 3 images, 1 placeholder, tiling off
                }
            ]
        )
        image_items = [
            item for item in out[0]["messages"][0]["content"] if item["type"] == "image"
        ]
        assert len(image_items) == 1  # capped to the single placeholder

    def test_processing_strategy_rejects_unmatched_column_placeholders(
        self, mock_processor, tmp_path
    ):
        from axolotl.processing_strategies import get_processing_strategy

        image_path = tmp_path / "page.png"
        Image.new("RGB", (32, 32), "white").save(image_path)
        strategy = get_processing_strategy(
            mock_processor,
            chat_template=None,
            chat_template_type=None,
        )

        with pytest.raises(ValueError, match="bare image placeholder"):
            strategy(
                [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "image"},
                                ],
                            },
                            {"role": "assistant", "content": "text"},
                        ],
                        "images": [str(image_path)],
                    }
                ]
            )

    def test_packing_concatenates_nested_rows_and_resets_positions(
        self, mock_processing_strategy
    ):
        from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator

        tokenizer = mock_processing_strategy.processor.tokenizer
        tokenizer.eos_token_id = 2
        calls = []

        def _apply_chat_template(messages, **kwargs):
            calls.append((messages, kwargs))
            call_idx = len(calls)
            if call_idx == 1:
                return BatchFeature(
                    data={
                        "input_ids": torch.tensor([[11, 128256, 13]]),
                        "attention_mask": torch.tensor([[1, 1, 1]]),
                        "pixel_values": torch.ones(4, 2),
                        "image_grid_thw": torch.tensor([[1, 2, 2]]),
                    }
                )
            return BatchFeature(
                data={
                    "input_ids": torch.tensor([[21, 22]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                    "pixel_values": torch.full((2, 2), 2.0),
                    "image_grid_thw": torch.tensor([[1, 1, 2]]),
                }
            )

        mock_processing_strategy.processor.apply_chat_template = MagicMock(
            side_effect=_apply_chat_template
        )
        examples = [
            [
                {
                    "messages": [
                        {"role": "user", "content": "image"},
                        {"role": "assistant", "content": "one"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "text"},
                        {"role": "assistant", "content": "two"},
                    ]
                },
            ]
        ]
        collator = MultiModalChatDataCollator(
            tokenizer=tokenizer,
            processing_strategy=mock_processing_strategy,
            packing=True,
            pad_to_multiple_of=8,
            max_length=8,
        )

        with patch.object(
            mock_processing_strategy,
            "__call__",
            side_effect=lambda rows: rows,
        ):
            batch = collator.process_rows(examples)

        assert batch["input_ids"].tolist() == [[11, 128256, 13, 21, 22, 0, 0, 0]]
        assert batch["labels"].tolist() == [[11, -100, 13, 21, 22, -100, -100, -100]]
        assert batch["attention_mask"].tolist() == [[1, 1, 1, 2, 2, 0, 0, 0]]
        assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 0, 0, 0]]
        assert batch["pixel_values"].shape == (6, 2)
        assert batch["image_grid_thw"].tolist() == [[1, 2, 2], [1, 1, 2]]
        assert len(calls) == 2
        assert all(call[1]["processor_kwargs"]["padding"] is False for call in calls)

    def test_packing_supports_batched_pixel_value_layout(
        self, mock_processing_strategy
    ):
        from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator

        tokenizer = mock_processing_strategy.processor.tokenizer
        values = [
            torch.ones(1, 1, 3, 4, 4),
            torch.full((1, 2, 3, 4, 4), 2.0),
        ]

        def _apply_chat_template(_messages, **_kwargs):
            idx = 0 if _apply_chat_template.calls == 0 else 1
            _apply_chat_template.calls += 1
            return BatchFeature(
                data={
                    "input_ids": torch.tensor([[idx + 1]]),
                    "attention_mask": torch.tensor([[1]]),
                    "pixel_values": values[idx],
                }
            )

        _apply_chat_template.calls = 0
        mock_processing_strategy.processor.apply_chat_template = MagicMock(
            side_effect=_apply_chat_template
        )
        collator = MultiModalChatDataCollator(
            tokenizer=tokenizer,
            processing_strategy=mock_processing_strategy,
            packing=True,
        )

        with patch.object(
            mock_processing_strategy,
            "__call__",
            side_effect=lambda rows: rows,
        ):
            batch = collator.process_rows([[{"messages": []}, {"messages": []}]])

        assert batch["pixel_values"].shape == (1, 3, 3, 4, 4)

    def test_builder_routes_packed_mm_sft_to_mm_chat_collator(
        self, mock_processing_strategy
    ):
        from axolotl.core.builders.causal import HFCausalTrainerBuilder
        from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator
        from axolotl.utils.dict import DictDefault

        builder = object.__new__(HFCausalTrainerBuilder)
        builder.tokenizer = mock_processing_strategy.processor.tokenizer
        builder.processor = mock_processing_strategy.processor
        builder.cfg = DictDefault(
            {
                "processor_type": "AutoProcessor",
                "pretraining_dataset": None,
                "datasets": [{"path": "ds", "type": "chat_template"}],
                "test_datasets": None,
                "sample_packing": True,
                "eval_sample_packing": True,
                "multipack_real_batches": False,
                "micro_batch_size": 4,
                "sequence_len": 128,
                "eval_sequence_len": None,
                "pad_to_sequence_len": True,
                "plugins": None,
                "reward_model": False,
                "batch_flattening": False,
                "model_config_type": "qwen3_vl",
                "attn_implementation": "flash_attention_2",
                "role_boundaries": None,
                "train_on_inputs": False,
                "chat_template": None,
                "image_tiling": True,
                "image_tiling_tile_size": 512,
                "image_tiling_grid": (2, 3),
            }
        )
        training_args = DictDefault(
            {
                "pretraining": False,
                "sample_packing": True,
                "eval_sample_packing": True,
                "chat_template": None,
                "image_size": None,
                "image_resize_algorithm": None,
                "image_resize_buckets": [(1024, 1536), (1536, 1536)],
                "image_resize_no_upscale": True,
                "image_resize_pad_color": "white",
                "image_tiling": True,
                "image_tiling_tile_size": 512,
                "image_tiling_grid": (2, 3),
            }
        )

        with patch(
            "axolotl.core.builders.causal.get_processing_strategy",
            return_value=mock_processing_strategy,
        ) as get_strategy:
            collator = HFCausalTrainerBuilder.build_collator(
                builder,
                training_args,
                padding=True,
                pad_to_multiple_of=128,
            )

        assert isinstance(collator, MultiModalChatDataCollator)
        assert collator.packing is True
        assert collator.max_length == 512
        assert collator.pad_to_multiple_of == 512
        assert get_strategy.call_args.kwargs["image_resize_buckets"] == [
            (1024, 1536),
            (1536, 1536),
        ]
        assert get_strategy.call_args.kwargs["image_resize_no_upscale"] is True
        assert get_strategy.call_args.kwargs["image_resize_pad_color"] == "white"
        assert get_strategy.call_args.kwargs["image_tiling_config"].tile_size == 512
