"""Tests for ``axolotl.processing_strategies``.

These tests avoid loading real checkpoints so they can run offline and in CI
without HF Hub access. The token-id layout on the fake processor matches the
real ``google/gemma-4-E2B-it`` processor's layout closely enough to exercise
the string-encode plus id-resolution paths the strategy actually uses.
"""

import pytest
import torch

from axolotl.processing_strategies import (
    Gemma4ProcessingStrategy,
    get_processing_strategy,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeGemma4Tokenizer:
    """Minimal tokenizer stub with the attributes ``Gemma4ProcessingStrategy`` reads.

    Ids mirror the real Gemma 4 layout so ordering bugs surface identically:
        <|turn>   = 105          <turn|>    = 106          model = 4368
        <|image|> = 258880       <|video|>  = 258884       <|audio|> = 258881
        <|image>  = 255999 (boi) <image|>   = 258882 (eoi)
        <|audio>  = 256000 (boa) <audio|>   = 258883 (eoa)
    """

    VOCAB = {
        "<|turn>": 105,
        "<turn|>": 106,
        "model": 4368,
        "\n": 107,
        "<|image|>": 258880,
        "<|video|>": 258884,
        "<|audio|>": 258881,
        "<|image>": 255999,
        "<image|>": 258882,
        "<|audio>": 256000,
        "<audio|>": 258883,
    }
    pad_token_id = 0
    unk_token_id = 3

    def encode(self, text, add_special_tokens=False):
        # Only the two strings the strategy actually encodes are supported.
        if text == "<|turn>model":
            return [self.VOCAB["<|turn>"], self.VOCAB["model"]]
        if text == "<turn|>":
            return [self.VOCAB["<turn|>"]]
        raise ValueError(f"FakeTokenizer.encode unsupported: {text!r}")

    def convert_tokens_to_ids(self, token):
        return self.VOCAB.get(token, self.unk_token_id)


class _FakeGemma4Processor:
    """Gemma-4-shaped processor stand-in (mirrors transformers' Gemma4Processor)."""

    def __init__(self):
        self.tokenizer = _FakeGemma4Tokenizer()
        # Present as standard integer attrs on the tokenizer, per real Gemma 4:
        self.tokenizer.image_token_id = self.tokenizer.VOCAB["<|image|>"]
        self.tokenizer.audio_token_id = self.tokenizer.VOCAB["<|audio|>"]
        # Processor-level: strings for boi/eoi/boa/eoa, ids for image/audio/video
        self.image_token = "<|image|>"
        self.image_token_id = self.tokenizer.VOCAB["<|image|>"]
        self.boi_token = "<|image>"
        self.eoi_token = "<image|>"
        self.video_token = "<|video|>"
        self.video_token_id = self.tokenizer.VOCAB["<|video|>"]
        self.audio_token = "<|audio|>"
        self.audio_token_id = self.tokenizer.VOCAB["<|audio|>"]
        self.boa_token = "<|audio>"
        self.eoa_token = "<audio|>"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_gemma4_masks_everything_outside_assistant_span():
    """All prompt + scaffolding tokens become -100; only content + end marker
    remain in loss."""
    strategy = Gemma4ProcessingStrategy(_FakeGemma4Processor())
    V = strategy.processor.tokenizer.VOCAB

    # Sequence: <pad> <|turn>user \n hi <turn|> \n  <|turn>model \n ok <turn|> \n
    input_ids = torch.tensor(
        [
            [
                strategy.processor.tokenizer.pad_token_id,  # 0 pad
                V["<|turn>"],  # 1
                7777,  # 2 "user" (stand-in role token)
                V["\n"],  # 3
                4444,  # 4 "hi" (user content)
                V["<turn|>"],  # 5
                V["\n"],  # 6
                V["<|turn>"],  # 7 assistant start ───┐
                V["model"],  # 8                       │ excluded
                V["\n"],  # 9 assistant content  ─┐
                5555,  # 10                       │ in loss
                V["<turn|>"],  # 11 end           ┘ included
                V["\n"],  # 12 post-end scaffolding (not in loss)
            ]
        ]
    )

    labels = strategy.process_labels(input_ids)

    expected = torch.tensor(
        [
            [
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,  # everything before assistant start
                -100,
                -100,  # <|turn>model excluded
                V["\n"],
                5555,
                V["<turn|>"],  # content + end
                -100,  # trailing \n scaffolding
            ]
        ]
    )
    assert torch.equal(labels, expected), (
        f"\nexpected: {expected.tolist()}\n" f"got:      {labels.tolist()}"
    )


def test_gemma4_masks_media_tokens_inside_assistant_span():
    """image / video / boi / eoi / boa / eoa all become -100, even when they
    appear inside the assistant span. This verifies that
    ``convert_tokens_to_ids`` resolution (not ``hasattr(tokenizer, "..._id")``)
    is what's used for the delimiter tokens — since Gemma 4 exposes those as
    token *strings* on the processor, not integer attrs on the tokenizer.
    """
    strategy = Gemma4ProcessingStrategy(_FakeGemma4Processor())
    V = strategy.processor.tokenizer.VOCAB

    input_ids = torch.tensor(
        [
            [
                V["<|turn>"],
                V["model"],  # start
                V["\n"],  # content \n  (NOT a media token, stays)
                V["<|image|>"],  # media
                V["<|video|>"],  # media
                V["<|audio|>"],  # media
                V["<|image>"],  # boi
                V["<image|>"],  # eoi
                V["<|audio>"],  # boa
                V["<audio|>"],  # eoa
                9999,  # ordinary content (stays)
                V["<turn|>"],  # end (stays)
            ]
        ]
    )

    labels = strategy.process_labels(input_ids)

    expected = torch.tensor(
        [
            [
                -100,
                -100,  # <|turn>model excluded
                V["\n"],  # content \n
                -100,
                -100,
                -100,  # image/video/audio soft tokens
                -100,
                -100,  # boi/eoi
                -100,
                -100,  # boa/eoa
                9999,  # ordinary content
                V["<turn|>"],  # end included
            ]
        ]
    )
    assert torch.equal(labels, expected), (
        f"\nexpected: {expected.tolist()}\n" f"got:      {labels.tolist()}"
    )


def test_gemma4_handles_multiple_assistant_turns():
    """Multi-turn conversations: every model turn contributes to loss, every
    user/system turn is masked."""
    strategy = Gemma4ProcessingStrategy(_FakeGemma4Processor())
    V = strategy.processor.tokenizer.VOCAB

    def user_turn(content_id):
        return [V["<|turn>"], 7777, V["\n"], content_id, V["<turn|>"], V["\n"]]

    def model_turn(content_id):
        return [V["<|turn>"], V["model"], V["\n"], content_id, V["<turn|>"], V["\n"]]

    seq = user_turn(1111) + model_turn(2222) + user_turn(3333) + model_turn(4444)
    input_ids = torch.tensor([seq])

    labels = strategy.process_labels(input_ids)
    kept = labels[labels != -100].tolist()

    # Two assistant turns, each contributing [\n, content, <turn|>].
    assert kept == [
        V["\n"],
        2222,
        V["<turn|>"],
        V["\n"],
        4444,
        V["<turn|>"],
    ], f"kept={kept}"


def test_gemma4_handles_truncated_assistant_turn():
    """If the sequence ends mid-assistant (no closing <turn|>), the remaining
    content still contributes to loss."""
    strategy = Gemma4ProcessingStrategy(_FakeGemma4Processor())
    V = strategy.processor.tokenizer.VOCAB

    # Assistant turn truncated mid-content
    input_ids = torch.tensor(
        [
            [
                V["<|turn>"],
                V["model"],
                V["\n"],
                1111,
                2222,  # content, no closing <turn|>
            ]
        ]
    )
    labels = strategy.process_labels(input_ids)
    expected = torch.tensor([[-100, -100, V["\n"], 1111, 2222]])
    assert torch.equal(labels, expected), (
        f"\nexpected: {expected.tolist()}\n" f"got:      {labels.tolist()}"
    )


def test_get_processing_strategy_dispatches_to_gemma4():
    """``chat_template_type='gemma4'`` resolves to ``Gemma4ProcessingStrategy``.

    Uses ``pytest.importorskip`` to tolerate test environments missing optional
    deps (e.g. mistral_common) that the dispatch function imports eagerly.
    """
    pytest.importorskip("axolotl.utils.mistral.mistral3_processor")

    strategy = get_processing_strategy(
        processor=_FakeGemma4Processor(),
        chat_template=None,
        chat_template_type="gemma4",
    )
    assert isinstance(strategy, Gemma4ProcessingStrategy)
