"""Tests for multimodal packing benchmark helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.mm_packing_benchmark import (
    chunked_causal_lm_loss,
    shifted_label_token_count,
)


class _ToyBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, **_kwargs):  # type: ignore[override]
        return (self.embed(input_ids),)


class _ToyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 11, hidden_size: int = 7):
        super().__init__()
        self.model = _ToyBackbone(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)


def test_chunked_causal_lm_loss_matches_full_logits_ce():
    torch.manual_seed(0)
    model = _ToyCausalLM()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "labels": torch.tensor([[1, 2, -100, 4, 5]]),
    }

    hidden_states = model.model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )[0]
    logits = model.lm_head(hidden_states)
    shift_labels = F.pad(inputs["labels"], (0, 1), value=-100)[..., 1:]
    expected = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )

    actual = chunked_causal_lm_loss(model, inputs, chunk_size=2)

    assert shifted_label_token_count(inputs["labels"]) == 3
    assert torch.allclose(actual, expected)
