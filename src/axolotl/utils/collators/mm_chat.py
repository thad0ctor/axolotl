"""
Collators for multi-modal chat messages and packing
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy

from axolotl.processing_strategies import ProcessingStrategy


@dataclass
class MultiModalChatDataCollator(DataCollatorMixin):
    """
    Collator for multi-modal chat messages
    """

    tokenizer: PreTrainedTokenizerBase
    processing_strategy: ProcessingStrategy
    packing: bool = False
    return_tensors: str = "pt"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __post_init__(self):
        if self.return_tensors != "pt":
            raise ValueError("MultiModalChatDataCollator only supports return_tensors='pt'.")

    def torch_call(self, examples: list[dict]) -> dict[str, Any]:
        return self.process_rows(examples)

    def process_rows(
        self,
        examples: list[dict],
    ) -> dict[str, Tensor]:
        if self.packing:
            return self.process_packed_rows(examples)

        # Preprocess the examples
        examples = self.processing_strategy(examples)

        # Initialize batch
        messages = [ex["messages"] for ex in examples]

        batch = self.processing_strategy.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            chat_template=self.processing_strategy.chat_template,
            processor_kwargs=self._processor_padding_kwargs(padding=True),
        )

        # Process the labels
        batch["labels"] = self.processing_strategy.process_labels(batch["input_ids"])

        return batch

    def process_packed_rows(
        self,
        examples: list[dict] | list[list[dict]],
    ) -> dict[str, Tensor]:
        groups = self._normalize_packing_groups(examples)
        packed_rows = []
        extra_tensors: dict[str, list[Tensor]] = {}
        for group in groups:
            processed = self.processing_strategy(group)
            encoded_rows = [self._encode_single_row(row) for row in processed]
            packed = self._pack_encoded_rows(encoded_rows)
            packed_rows.append(packed)
            for key, value in packed.items():
                if key in {"input_ids", "labels", "attention_mask", "position_ids"}:
                    continue
                if torch.is_tensor(value):
                    extra_tensors.setdefault(key, []).append(value)

        text_batch = self._pad_packed_text_rows(packed_rows)
        for key, values in extra_tensors.items():
            text_batch[key] = self._concat_extra_tensors(key, values)
        return text_batch

    @staticmethod
    def _normalize_packing_groups(
        examples: list[dict] | list[list[dict]],
    ) -> list[list[dict]]:
        if not examples:
            return []
        first = examples[0]
        if isinstance(first, list):
            return examples  # type: ignore[return-value]
        return [examples]  # type: ignore[list-item]

    def _processor_padding_kwargs(self, *, padding: bool | str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"padding": padding}
        if self.pad_to_multiple_of is not None:
            kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
        return kwargs

    def _encode_single_row(self, example: dict[str, Any]) -> dict[str, Tensor]:
        batch = self.processing_strategy.processor.apply_chat_template(
            [example["messages"]],
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            chat_template=self.processing_strategy.chat_template,
            processor_kwargs=self._processor_padding_kwargs(padding=False),
        )
        batch = dict(batch)
        input_ids = self._ensure_batched(batch["input_ids"])
        batch["input_ids"] = input_ids
        if "attention_mask" in batch:
            batch["attention_mask"] = self._ensure_batched(batch["attention_mask"])
        else:
            batch["attention_mask"] = torch.ones_like(input_ids)
        batch["labels"] = self.processing_strategy.process_labels(input_ids)
        return batch

    @staticmethod
    def _ensure_batched(value: Tensor) -> Tensor:
        if value.ndim == 1:
            return value.unsqueeze(0)
        return value

    def _pack_encoded_rows(self, encoded_rows: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        packed: dict[str, Tensor] = {}
        input_parts = [row["input_ids"].squeeze(0) for row in encoded_rows]
        label_parts = [row["labels"].squeeze(0) for row in encoded_rows]
        packed["input_ids"] = torch.cat(input_parts).unsqueeze(0)
        packed["labels"] = torch.cat(label_parts).unsqueeze(0)

        attention_parts = []
        position_parts = []
        for idx, row in enumerate(encoded_rows):
            mask = row["attention_mask"].squeeze(0).to(dtype=torch.long)
            attention_parts.append((mask != 0).to(dtype=torch.long) * (idx + 1))
            position_parts.append(
                torch.arange(mask.numel(), dtype=torch.long, device=mask.device)
            )
        packed["attention_mask"] = torch.cat(attention_parts).unsqueeze(0)
        packed["position_ids"] = torch.cat(position_parts).unsqueeze(0)

        for key in sorted(set().union(*(row.keys() for row in encoded_rows))):
            if key in {"input_ids", "labels", "attention_mask", "position_ids"}:
                continue
            values = [
                row[key] for row in encoded_rows if key in row and torch.is_tensor(row[key])
            ]
            if values:
                packed[key] = self._concat_extra_tensors(key, values)
        return packed

    def _pad_packed_text_rows(self, rows: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        if not rows:
            return {}
        lengths = [int(row["input_ids"].shape[1]) for row in rows]
        target_len = max(lengths)
        if self.pad_to_multiple_of is not None:
            target_len = (
                (target_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        if self.max_length is not None:
            if target_len > self.max_length:
                raise ValueError(
                    "Packed multimodal SFT row exceeds max_length "
                    f"{self.max_length}: packed length={target_len}."
                )
            if self.padding == "max_length" or self.pad_to_multiple_of == self.max_length:
                target_len = self.max_length

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        padding_values = {
            "input_ids": int(pad_id),
            "labels": self.label_pad_token_id,
            "attention_mask": 0,
            "position_ids": 0,
        }
        out: dict[str, Tensor] = {}
        for key, pad_value in padding_values.items():
            out[key] = torch.cat(
                [
                    self._pad_2d(row[key], target_len, pad_value)
                    for row in rows
                ],
                dim=0,
            )
        return out

    @staticmethod
    def _pad_2d(value: Tensor, target_len: int, pad_value: int) -> Tensor:
        pad_len = target_len - int(value.shape[1])
        if pad_len <= 0:
            return value
        pad = torch.full(
            (value.shape[0], pad_len),
            pad_value,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat([value, pad], dim=1)

    @staticmethod
    def _concat_extra_tensors(key: str, values: list[Tensor]) -> Tensor:
        if len(values) == 1:
            return values[0]
        if key.startswith("pixel_values") and all(
            value.ndim >= 3 and value.shape[0] == 1 for value in values
        ):
            return torch.cat(values, dim=1)
        return torch.cat(values, dim=0)
