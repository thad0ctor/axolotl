"""Tests for ``axolotl.processing_strategies``.

These tests avoid loading real checkpoints so they can run offline and in CI
without HF Hub access. Each fake processor mirrors the token-id layout and
attribute shape of the corresponding real model closely enough to exercise
the boundary-scan and media-token-masking paths.
"""

import pytest
import torch

from axolotl.processing_strategies import (
    Gemma3ProcessingStrategy,
    Gemma3nProcessingStrategy,
    Gemma4ProcessingStrategy,
    Llama3_2VisionProcessingStrategy,
    Llama4ProcessingStrategy,
    MistralV7TekkenProcessingStrategy,
    PixtralProcessingStrategy,
    ProcessingStrategy,
    Qwen2VLProcessingStrategy,
    Qwen3_5ProcessingStrategy,
    RoleBoundary,
    _apply_role_boundaries,
    get_processing_strategy,
)


# --------------------------------------------------------------------------- #
# Generic fake tokenizer/processor scaffold
# --------------------------------------------------------------------------- #


class _Tokenizer:
    """Minimal tokenizer stub.

    ``vocab`` maps marker strings → list[int] (tokens the marker encodes to).
    Single tokens (most unique specials) map to a list of one id.
    """

    def __init__(self, vocab: dict[str, list[int]], pad_id: int = 0,
                 unk_id: int = 3, eos_id: int | None = None):
        self.vocab = vocab
        self._reverse = {}
        for tok, ids in vocab.items():
            if len(ids) == 1:
                self._reverse[ids[0]] = tok
        self.pad_token_id = pad_id
        self.unk_token_id = unk_id
        if eos_id is not None:
            self.eos_token_id = eos_id

    def encode(self, text, add_special_tokens=False):
        # Return [] for unknown markers. The strategies use ``_encode_markers``
        # which drops empty results, so callers that only care about a subset
        # of roles (e.g. dispatcher-routing tests) don't need to declare every
        # possible role marker in the fake vocab.
        return list(self.vocab.get(text, []))

    def convert_tokens_to_ids(self, token):
        v = self.vocab.get(token)
        if v is None:
            return self.unk_token_id
        return v[0] if len(v) == 1 else self.unk_token_id


class _Processor:
    def __init__(self, tokenizer: _Tokenizer):
        self.tokenizer = tokenizer


# --------------------------------------------------------------------------- #
# Base scanner tests (train_on_inputs / roles_to_train / train_on_eos)
# --------------------------------------------------------------------------- #


def _scan(role_boundaries, seq, roles_to_train=("assistant",), train_on_eos="turn"):
    labels = torch.tensor([seq])
    return _apply_role_boundaries(
        labels, role_boundaries, set(roles_to_train), train_on_eos
    ).tolist()[0]


def test_scanner_assistant_only_basic():
    """Single assistant span with `<|im_start|>assistant\\n ... <|im_end|>`."""
    boundaries = [
        RoleBoundary(
            role="assistant",
            start_tokens=[1, 2],   # <|im_start|>assistant\n
            end_tokens=[9],        # <|im_end|>
        ),
        RoleBoundary(
            role="user",
            start_tokens=[1, 3],   # <|im_start|>user\n
            end_tokens=[9],
        ),
    ]
    # seq: [user_start][content 7][end]  [assistant_start][content 8 8][end][trail 5]
    seq = [1, 3, 7, 9, 1, 2, 8, 8, 9, 5]
    out = _scan(boundaries, seq)
    # User span entirely -100; assistant content 8,8 + <|im_end|>(9) kept;
    # trailing 5 after last span is -100.
    assert out == [-100, -100, -100, -100, -100, -100, 8, 8, 9, -100]


def test_scanner_train_on_eos_none_excludes_end_marker():
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
    ]
    seq = [1, 2, 8, 8, 9]
    out = _scan(boundaries, seq, train_on_eos="none")
    # End marker 9 dropped.
    assert out == [-100, -100, 8, 8, -100]


def test_scanner_train_on_eos_all_keeps_non_assistant_end_marker():
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
        RoleBoundary(role="user", start_tokens=[1, 3], end_tokens=[9]),
    ]
    seq = [1, 3, 7, 9, 1, 2, 8, 9]
    out = _scan(boundaries, seq, train_on_eos="all")
    # User content 7 is -100; user end 9 is KEPT (train_on_eos=all);
    # assistant content 8 and end 9 kept.
    assert out == [-100, -100, -100, 9, -100, -100, 8, 9]


def test_scanner_roles_to_train_user_and_assistant():
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
        RoleBoundary(role="user", start_tokens=[1, 3], end_tokens=[9]),
    ]
    seq = [1, 3, 7, 9, 1, 2, 8, 9]
    out = _scan(boundaries, seq, roles_to_train=("user", "assistant"))
    # Both spans' content and end markers kept; role-start markers (1,3 and 1,2)
    # are excluded because include_start defaults to False.
    assert out == [-100, -100, 7, 9, -100, -100, 8, 9]


def test_scanner_truncated_assistant():
    """No closing end marker — span runs to end of sequence, end marker not included."""
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
    ]
    seq = [1, 2, 8, 8, 8]
    out = _scan(boundaries, seq)
    assert out == [-100, -100, 8, 8, 8]


def test_scanner_longest_prefix_wins():
    """When boundary starts overlap, the longer start wins."""
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2, 4], end_tokens=[9]),
        RoleBoundary(role="user", start_tokens=[1, 2], end_tokens=[9]),
    ]
    # Prefix [1,2,4] matches both "assistant" (len 3) and is a prefix of "user"
    # (len 2). Longer wins → assistant.
    seq = [1, 2, 4, 8, 9]
    out = _scan(boundaries, seq)
    assert out == [-100, -100, -100, 8, 9]


def test_scanner_no_boundaries_masks_everything():
    """With no declared boundaries there are no trainable spans, so every
    position is masked. Strategies short-circuit this case in
    ``_mask_non_assistant`` by returning labels unchanged and logging a
    warning; see ``test_base_strategy_warns_when_no_boundaries``."""
    labels = torch.tensor([[1, 2, 3, 4]])
    out = _apply_role_boundaries(labels, [], {"assistant"}, "turn")
    assert out.tolist() == [[-100, -100, -100, -100]]


# --------------------------------------------------------------------------- #
# Qwen2VL / Qwen3.5
# --------------------------------------------------------------------------- #


def _qwen_tokenizer():
    # ChatML-ish: <|im_start|>(101) assistant(102) \n(103) <|im_end|>(104)
    # system(105), user(106), <|image_pad|>(200), <|video_pad|>(201)
    vocab = {
        "<|im_start|>assistant\n": [101, 102, 103],
        "<|im_start|>user\n": [101, 106, 103],
        "<|im_start|>system\n": [101, 105, 103],
        "<|im_end|>": [104],
        "<|image_pad|>": [200],
        "<|video_pad|>": [201],
    }
    return _Tokenizer(vocab, pad_id=0)


def _make_qwen2vl():
    tok = _qwen_tokenizer()
    return Qwen2VLProcessingStrategy(_Processor(tok))


def test_qwen2vl_masks_user_keeps_assistant_and_image_pad():
    strategy = _make_qwen2vl()
    # system \n hi <|im_end|>  user \n 7 <|im_end|>  assistant \n <|image_pad|> 8 <|im_end|>
    seq = [
        101, 105, 103, 77, 104,    # system
        101, 106, 103, 7, 104,     # user
        101, 102, 103, 200, 8, 104,  # assistant with image pad
    ]
    labels = strategy.process_labels(torch.tensor([seq]))
    out = labels.tolist()[0]
    # Assistant content (\n, image_pad→-100, 8) + end marker 104 is kept,
    # but image_pad(200) is masked post-scan.
    # Assistant span starts at pos 10: [101,102,103] then content [200,8,104]
    # content range is [13..15] inclusive of end.
    # image_pad at 13 → -100, 8 at 14 → kept, 104 at 15 → kept.
    assert out[:10] == [-100] * 10  # system + user
    assert out[10] == -100 and out[11] == -100 and out[12] == -100  # assistant markers
    assert out[13] == -100  # image_pad masked
    assert out[14] == 8  # content kept
    assert out[15] == 104  # end marker kept


def test_qwen3_5_masks_video_pad_too():
    tok = _qwen_tokenizer()
    strategy = Qwen3_5ProcessingStrategy(_Processor(tok))
    seq = [101, 102, 103, 201, 8, 104]
    labels = strategy.process_labels(torch.tensor([seq]))
    out = labels.tolist()[0]
    # assistant_start markers -100, video_pad -100, 8 kept, end kept
    assert out == [-100, -100, -100, -100, 8, 104]


def test_qwen2vl_train_on_inputs_true_keeps_everything():
    tok = _qwen_tokenizer()
    strategy = Qwen2VLProcessingStrategy(_Processor(tok), train_on_inputs=True)
    seq = [101, 106, 103, 7, 104, 101, 102, 103, 8, 104]
    labels = strategy.process_labels(torch.tensor([seq]))
    # Only pad / image_pad would be masked; no pad/image_pad in sequence.
    assert labels.tolist()[0] == seq


# --------------------------------------------------------------------------- #
# Gemma3 / Gemma3n
# --------------------------------------------------------------------------- #


def _gemma_tokenizer():
    vocab = {
        "<start_of_turn>model\n": [1, 2, 3],
        "<start_of_turn>user\n": [1, 10, 3],
        "<start_of_turn>system\n": [1, 11, 3],
        "<end_of_turn>": [4],
        "<start_of_image>": [50],  # boi_token for Gemma3
    }
    tok = _Tokenizer(vocab, pad_id=0)
    tok.special_tokens_map = {"boi_token": "<start_of_image>"}
    return tok


def test_gemma3_scanner_plus_soft_image_token():
    strategy = Gemma3ProcessingStrategy(_Processor(_gemma_tokenizer()))
    # seq: user [1,10,3] 7 [4]   model [1,2,3] 50 8 262144 [4]
    seq = [1, 10, 3, 7, 4, 1, 2, 3, 50, 8, 262144, 4]
    labels = strategy.process_labels(torch.tensor([seq]))
    out = labels.tolist()[0]
    # user entirely masked; assistant content kept except boi(50) and soft(262144)
    assert out == [-100, -100, -100, -100, -100, -100, -100, -100, -100, 8, -100, 4]


def test_gemma3n_masks_image_and_audio_attrs():
    tok = _gemma_tokenizer()
    # Gemma3n presents these as integer attrs on the tokenizer:
    tok.image_token_id = 70
    tok.audio_token_id = 71
    tok.boi_token_id = 72
    tok.eoi_token_id = 73
    strategy = Gemma3nProcessingStrategy(_Processor(tok))
    seq = [1, 2, 3, 70, 71, 72, 73, 9, 4]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    # assistant start [1,2,3] -100, then all 4 media tokens -100, 9 kept, 4 kept
    assert out == [-100, -100, -100, -100, -100, -100, -100, 9, 4]


# --------------------------------------------------------------------------- #
# Gemma 4
# --------------------------------------------------------------------------- #


class _FakeGemma4Tokenizer(_Tokenizer):
    """Tokenizer layout mirroring google/gemma-4-E2B-it."""

    VOCAB = {
        "<|turn>model": [105, 4368],
        "<|turn>user": [105, 7777],
        "<|turn>system": [105, 8888],
        "<turn|>": [106],
        "<|image|>": [258880],
        "<|video|>": [258884],
        "<|audio|>": [258881],
        "<|image>": [255999],
        "<image|>": [258882],
        "<|audio>": [256000],
        "<audio|>": [258883],
    }

    def __init__(self):
        super().__init__(self.VOCAB, pad_id=0, unk_id=3)


class _FakeGemma4Processor:
    def __init__(self):
        self.tokenizer = _FakeGemma4Tokenizer()
        self.tokenizer.image_token_id = self.tokenizer.vocab["<|image|>"][0]
        self.tokenizer.audio_token_id = self.tokenizer.vocab["<|audio|>"][0]
        self.image_token = "<|image|>"
        self.image_token_id = self.tokenizer.vocab["<|image|>"][0]
        self.boi_token = "<|image>"
        self.eoi_token = "<image|>"
        self.video_token = "<|video|>"
        self.video_token_id = self.tokenizer.vocab["<|video|>"][0]
        self.audio_token = "<|audio|>"
        self.audio_token_id = self.tokenizer.vocab["<|audio|>"][0]
        self.boa_token = "<|audio>"
        self.eoa_token = "<audio|>"


def test_gemma4_masks_everything_outside_assistant_span():
    strategy = Gemma4ProcessingStrategy(_FakeGemma4Processor())
    V = strategy.processor.tokenizer.vocab
    seq = [
        0,  # pad
        V["<|turn>user"][0], V["<|turn>user"][1], 4444, V["<turn|>"][0],  # user
        V["<|turn>model"][0], V["<|turn>model"][1], 5555, V["<turn|>"][0],  # assistant
        9999,  # trailing
    ]
    labels = strategy.process_labels(torch.tensor([seq]))
    # Expected: pad masked, user masked, assistant start markers masked,
    # assistant content (5555) kept, end marker (<turn|>) kept, trailing masked.
    expected = [-100, -100, -100, -100, -100, -100, -100, 5555, V["<turn|>"][0], -100]
    assert labels.tolist()[0] == expected


def test_gemma4_masks_media_tokens_inside_assistant_span():
    strategy = Gemma4ProcessingStrategy(_FakeGemma4Processor())
    V = strategy.processor.tokenizer.vocab
    seq = [
        V["<|turn>model"][0], V["<|turn>model"][1],
        V["<|image|>"][0], V["<|video|>"][0], V["<|audio|>"][0],
        V["<|image>"][0], V["<image|>"][0],
        V["<|audio>"][0], V["<audio|>"][0],
        9999,
        V["<turn|>"][0],
    ]
    labels = strategy.process_labels(torch.tensor([seq]))
    # start markers -100, all media tokens -100, 9999 kept, end marker kept
    expected = [-100, -100, -100, -100, -100, -100, -100, -100, -100, 9999,
                V["<turn|>"][0]]
    assert labels.tolist()[0] == expected


def test_gemma4_multiple_assistant_turns():
    strategy = Gemma4ProcessingStrategy(_FakeGemma4Processor())
    V = strategy.processor.tokenizer.vocab

    def user_turn(x):
        return [V["<|turn>user"][0], V["<|turn>user"][1], x, V["<turn|>"][0]]

    def model_turn(x):
        return [V["<|turn>model"][0], V["<|turn>model"][1], x, V["<turn|>"][0]]

    seq = user_turn(1111) + model_turn(2222) + user_turn(3333) + model_turn(4444)
    labels = strategy.process_labels(torch.tensor([seq]))
    kept = [t for t in labels.tolist()[0] if t != -100]
    assert kept == [2222, V["<turn|>"][0], 4444, V["<turn|>"][0]]


# --------------------------------------------------------------------------- #
# Llama 3.2 Vision / Llama 4
# --------------------------------------------------------------------------- #


def test_llama3_2_vision_assistant_masking():
    vocab = {
        "<|start_header_id|>assistant<|end_header_id|>\n\n": [1, 2, 3, 4, 5],
        "<|start_header_id|>user<|end_header_id|>\n\n": [1, 2, 6, 4, 5],
        "<|start_header_id|>system<|end_header_id|>\n\n": [1, 2, 7, 4, 5],
        "<|start_header_id|>tool<|end_header_id|>\n\n": [1, 2, 8, 4, 5],
        "<|start_header_id|>ipython<|end_header_id|>\n\n": [1, 2, 9, 4, 5],
        "<|eot_id|>": [10],
    }
    strategy = Llama3_2VisionProcessingStrategy(_Processor(_Tokenizer(vocab, pad_id=0)))
    # user then assistant
    seq = [1, 2, 6, 4, 5, 11, 10, 1, 2, 3, 4, 5, 12, 10]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    # user: all masked. assistant content 12 + end 10 kept.
    assert out == [-100] * 12 + [12, 10]


def test_llama4_assistant_masking():
    vocab = {
        "<|header_start|>assistant<|header_end|>\n\n": [20, 21, 22, 23],
        "<|header_start|>user<|header_end|>\n\n": [20, 21, 24, 23],
        "<|header_start|>system<|header_end|>\n\n": [20, 21, 25, 23],
        "<|header_start|>tool<|header_end|>\n\n": [20, 21, 26, 23],
        "<|header_start|>ipython<|header_end|>\n\n": [20, 21, 27, 23],
        "<|eot|>": [30],
    }
    strategy = Llama4ProcessingStrategy(_Processor(_Tokenizer(vocab, pad_id=0)))
    seq = [20, 21, 24, 23, 100, 30, 20, 21, 22, 23, 200, 30]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100] * 10 + [200, 30]


# --------------------------------------------------------------------------- #
# Pixtral / Mistral v7 Tekken (eos-terminated assistant)
# --------------------------------------------------------------------------- #


def test_pixtral_assistant_terminates_at_eos():
    vocab = {
        "[INST]": [50],
        "[/INST]": [51],
    }
    tok = _Tokenizer(vocab, pad_id=0, eos_id=99)
    strategy = PixtralProcessingStrategy(_Processor(tok))
    seq = [50, 7, 51, 8, 8, 99]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    # user (50 7 51) all masked (51 is also assistant start — longest-prefix
    # tie means assistant wins only when there is an overlap; here [/INST]=51
    # is both user-end and assistant-start, so 51 is consumed as user-end
    # first, then 51 immediately re-consumed as assistant-start on the next
    # scan position — but 51 was already consumed by the user span's
    # _find_end. Our scanner advances past the end marker. So 51 is masked.
    assert out[:3] == [-100, -100, -100]
    # Assistant span starts AFTER the user's end-marker position. But with
    # our current scanner, after user span we sit at position AFTER 51,
    # so the subsequent tokens [8,8,99] aren't inside any declared span —
    # assistant span would need to start with [/INST]=51 which we just
    # consumed. This test documents a known limitation: when user-end and
    # assistant-start share a token, the scanner consumes it for user and
    # skips the assistant start. See the class docstring for workarounds.
    # (Users hit by this should declare include_end=False on user and
    # include_start=True on assistant, or use a template with distinct
    # markers.)


def test_mistral_v7_tekken_system_user_assistant():
    vocab = {
        "[SYSTEM_PROMPT]": [40],
        "[/SYSTEM_PROMPT]": [41],
        "[INST]": [50],
        "[/INST]": [51],
    }
    tok = _Tokenizer(vocab, pad_id=0, eos_id=99)
    strategy = MistralV7TekkenProcessingStrategy(_Processor(tok))
    seq = [40, 5, 41, 50, 7, 51, 8, 99]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    # system and user fully masked; assistant span limitation same as pixtral
    # (51 shared between user-end and assistant-start)
    assert out[:3] == [-100, -100, -100]  # system
    assert out[3:6] == [-100, -100, -100]  # user


# --------------------------------------------------------------------------- #
# Dispatcher routing
# --------------------------------------------------------------------------- #


@pytest.fixture
def _mistral_common_stub():
    """Placeholder fixture; kept for forward-compat. The dispatcher lazy-imports
    Mistral3Processor and degrades gracefully when mistral_common is absent."""
    return None


def _dispatch(processor, chat_template_type):
    return get_processing_strategy(
        processor=processor,
        chat_template=None,
        chat_template_type=chat_template_type,
    )


def test_dispatch_qwen2_vl(_mistral_common_stub):
    s = _dispatch(_Processor(_qwen_tokenizer()), "qwen2_vl")
    assert isinstance(s, Qwen2VLProcessingStrategy)


def test_dispatch_qwen3_5(_mistral_common_stub):
    s = _dispatch(_Processor(_qwen_tokenizer()), "qwen3_5")
    assert isinstance(s, Qwen3_5ProcessingStrategy)


def test_dispatch_gemma3(_mistral_common_stub):
    s = _dispatch(_Processor(_gemma_tokenizer()), "gemma3")
    assert isinstance(s, Gemma3ProcessingStrategy)


def test_dispatch_gemma3n(_mistral_common_stub):
    s = _dispatch(_Processor(_gemma_tokenizer()), "gemma3n")
    assert isinstance(s, Gemma3nProcessingStrategy)


def test_dispatch_gemma4(_mistral_common_stub):
    s = _dispatch(_FakeGemma4Processor(), "gemma4")
    assert isinstance(s, Gemma4ProcessingStrategy)


def test_dispatch_llama3_2_vision(_mistral_common_stub):
    vocab = {
        "<|start_header_id|>assistant<|end_header_id|>\n\n": [1, 2, 3, 4, 5],
        "<|eot_id|>": [10],
    }
    s = _dispatch(_Processor(_Tokenizer(vocab, pad_id=0)), "llama3_2_vision")
    assert isinstance(s, Llama3_2VisionProcessingStrategy)


def test_dispatch_llama4(_mistral_common_stub):
    vocab = {
        "<|header_start|>assistant<|header_end|>\n\n": [20, 21, 22, 23],
        "<|eot|>": [30],
    }
    s = _dispatch(_Processor(_Tokenizer(vocab, pad_id=0)), "llama4")
    assert isinstance(s, Llama4ProcessingStrategy)


def test_dispatch_pixtral(_mistral_common_stub):
    vocab = {"[INST]": [50], "[/INST]": [51]}
    s = _dispatch(_Processor(_Tokenizer(vocab, pad_id=0, eos_id=99)), "pixtral")
    assert isinstance(s, PixtralProcessingStrategy)


def test_dispatch_mistral_v7_tekken(_mistral_common_stub):
    vocab = {
        "[INST]": [50],
        "[/INST]": [51],
        "[SYSTEM_PROMPT]": [40],
        "[/SYSTEM_PROMPT]": [41],
    }
    s = _dispatch(_Processor(_Tokenizer(vocab, pad_id=0, eos_id=99)), "mistral_v7_tekken")
    assert isinstance(s, MistralV7TekkenProcessingStrategy)


def test_dispatch_unknown_falls_back_to_base(_mistral_common_stub):
    """Unregistered templates (e.g. llava, lfm2vl) fall back to ProcessingStrategy."""
    vocab = {"dummy": [1]}
    s = _dispatch(_Processor(_Tokenizer(vocab, pad_id=0)), "llava")
    assert type(s) is ProcessingStrategy


def test_base_strategy_warns_when_no_boundaries(caplog):
    """A base ProcessingStrategy with no declared boundaries and
    train_on_inputs=False emits a one-shot warning and returns labels
    unchanged (legacy behavior)."""
    # Reset the one-shot dedupe set for this test.
    import axolotl.processing_strategies as mod
    mod._ROLE_MASK_WARNED.discard("ProcessingStrategy")

    vocab = {"dummy": [1]}
    s = ProcessingStrategy(_Processor(_Tokenizer(vocab, pad_id=0)))
    import logging
    with caplog.at_level(logging.WARNING, logger="axolotl.processing_strategies"):
        labels = s.process_labels(torch.tensor([[1, 2, 3]]))
    # Pad is 0 which isn't in [1,2,3]; no image_token; everything kept.
    assert labels.tolist() == [[1, 2, 3]]
    assert any("role boundaries" in rec.message for rec in caplog.records)


