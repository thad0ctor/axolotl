"""Tests for ``axolotl.processing_strategies`` using fake tokenizers (offline/CI-safe)."""

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
    """Minimal tokenizer stub; ``vocab`` maps marker strings to their id lists."""

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
        # Unknown markers return [] so _encode_markers drops them silently.
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
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
        RoleBoundary(role="user", start_tokens=[1, 3], end_tokens=[9]),
    ]
    seq = [1, 3, 7, 9, 1, 2, 8, 8, 9, 5]
    out = _scan(boundaries, seq)
    assert out == [-100, -100, -100, -100, -100, -100, 8, 8, 9, -100]


def test_scanner_train_on_eos_none_excludes_end_marker():
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
    ]
    seq = [1, 2, 8, 8, 9]
    out = _scan(boundaries, seq, train_on_eos="none")
    assert out == [-100, -100, 8, 8, -100]


def test_scanner_train_on_eos_all_keeps_non_assistant_end_marker():
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
        RoleBoundary(role="user", start_tokens=[1, 3], end_tokens=[9]),
    ]
    seq = [1, 3, 7, 9, 1, 2, 8, 9]
    out = _scan(boundaries, seq, train_on_eos="all")
    assert out == [-100, -100, -100, 9, -100, -100, 8, 9]


def test_scanner_roles_to_train_user_and_assistant():
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
        RoleBoundary(role="user", start_tokens=[1, 3], end_tokens=[9]),
    ]
    seq = [1, 3, 7, 9, 1, 2, 8, 9]
    out = _scan(boundaries, seq, roles_to_train=("user", "assistant"))
    # include_start defaults to False so role-start markers stay masked.
    assert out == [-100, -100, 7, 9, -100, -100, 8, 9]


def test_scanner_truncated_assistant():
    """Missing end marker: span runs to end-of-sequence, end marker not emitted."""
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
    ]
    seq = [1, 2, 8, 8, 8]
    out = _scan(boundaries, seq)
    assert out == [-100, -100, 8, 8, 8]


def test_scanner_longest_prefix_wins():
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2, 4], end_tokens=[9]),
        RoleBoundary(role="user", start_tokens=[1, 2], end_tokens=[9]),
    ]
    seq = [1, 2, 4, 8, 9]
    out = _scan(boundaries, seq)
    assert out == [-100, -100, -100, 8, 9]


def test_scanner_no_boundaries_masks_everything():
    # Strategies short-circuit this in _mask_non_assistant; see test_base_strategy_warns_when_no_boundaries.
    labels = torch.tensor([[1, 2, 3, 4]])
    out = _apply_role_boundaries(labels, [], {"assistant"}, "turn")
    assert out.tolist() == [[-100, -100, -100, -100]]


# --------------------------------------------------------------------------- #
# Qwen2VL / Qwen3.5
# --------------------------------------------------------------------------- #


def _qwen_tokenizer():
    # ChatML-ish with image_pad=200, video_pad=201.
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
    seq = [
        101, 105, 103, 77, 104,
        101, 106, 103, 7, 104,
        101, 102, 103, 200, 8, 104,
    ]
    labels = strategy.process_labels(torch.tensor([seq]))
    out = labels.tolist()[0]
    assert out[:10] == [-100] * 10
    assert out[10] == -100 and out[11] == -100 and out[12] == -100
    assert out[13] == -100  # image_pad masked post-scan
    assert out[14] == 8
    assert out[15] == 104


def test_qwen3_5_masks_video_pad_too():
    tok = _qwen_tokenizer()
    strategy = Qwen3_5ProcessingStrategy(_Processor(tok))
    seq = [101, 102, 103, 201, 8, 104]
    labels = strategy.process_labels(torch.tensor([seq]))
    assert labels.tolist()[0] == [-100, -100, -100, -100, 8, 104]


def test_qwen2vl_train_on_inputs_true_keeps_everything():
    tok = _qwen_tokenizer()
    strategy = Qwen2VLProcessingStrategy(_Processor(tok), train_on_inputs=True)
    seq = [101, 106, 103, 7, 104, 101, 102, 103, 8, 104]
    labels = strategy.process_labels(torch.tensor([seq]))
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
    seq = [1, 10, 3, 7, 4, 1, 2, 3, 50, 8, 262144, 4]
    labels = strategy.process_labels(torch.tensor([seq]))
    # boi(50) and soft-image-token(262144) masked post-scan.
    assert labels.tolist()[0] == [-100, -100, -100, -100, -100, -100, -100, -100, -100, 8, -100, 4]


def test_gemma3n_masks_image_and_audio_attrs():
    tok = _gemma_tokenizer()
    # Gemma3n exposes these as integer attrs on the tokenizer.
    tok.image_token_id = 70
    tok.audio_token_id = 71
    tok.boi_token_id = 72
    tok.eoi_token_id = 73
    strategy = Gemma3nProcessingStrategy(_Processor(tok))
    seq = [1, 2, 3, 70, 71, 72, 73, 9, 4]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100, -100, -100, -100, -100, -100, -100, 9, 4]


# --------------------------------------------------------------------------- #
# Gemma 4
# --------------------------------------------------------------------------- #


class _FakeGemma4Tokenizer(_Tokenizer):
    """Mirrors google/gemma-4-E2B-it token layout."""

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
        0,
        V["<|turn>user"][0], V["<|turn>user"][1], 4444, V["<turn|>"][0],
        V["<|turn>model"][0], V["<|turn>model"][1], 5555, V["<turn|>"][0],
        9999,
    ]
    labels = strategy.process_labels(torch.tensor([seq]))
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
    seq = [1, 2, 6, 4, 5, 11, 10, 1, 2, 3, 4, 5, 12, 10]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
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
    # [/INST] is both user-end and assistant-start. Scanner backs up when
    # user.include_end=False so the next iteration picks [/INST] up as
    # assistant-start (Pixtral-specific handling in _build_role_boundaries).
    vocab = {
        "[INST]": [50],
        "[/INST]": [51],
    }
    tok = _Tokenizer(vocab, pad_id=0, eos_id=99)
    strategy = PixtralProcessingStrategy(_Processor(tok))
    seq = [50, 7, 51, 8, 8, 99]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    # Full-sequence expectation: user span masked; assistant content + eos kept.
    assert out == [-100, -100, -100, 8, 8, 99]


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
    # Full-sequence expectation: system + user spans masked; assistant kept.
    assert out == [-100, -100, -100, -100, -100, -100, 8, 99]


# --------------------------------------------------------------------------- #
# Dispatcher routing
# --------------------------------------------------------------------------- #


@pytest.fixture
def _mistral_common_stub():
    # Placeholder; dispatcher lazy-imports Mistral3Processor and degrades gracefully.
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
    vocab = {"dummy": [1]}
    s = _dispatch(_Processor(_Tokenizer(vocab, pad_id=0)), "llava")
    assert type(s) is ProcessingStrategy


# --------------------------------------------------------------------------- #
# Config-based role-boundary override
# --------------------------------------------------------------------------- #


def test_role_boundaries_override_replaces_built_in():
    """Override swaps the built-in boundaries wholesale, not additively."""
    vocab = {
        "<|im_start|>assistant\n": [101, 102, 103],
        "<|im_start|>user\n": [101, 106, 103],
        "<|im_end|>": [104],
        ">>>A": [200, 201],
        ">>>U": [200, 202],
        "<<<": [210],
        "<|image_pad|>": [250],
    }
    strategy = Qwen2VLProcessingStrategy(
        _Processor(_Tokenizer(vocab, pad_id=0)),
        role_boundaries_override=[
            {"role": "assistant", "start": ">>>A", "end": "<<<"},
            {"role": "user", "start": ">>>U", "end": "<<<"},
        ],
    )
    seq = [
        101, 106, 103, 7, 104,
        200, 201, 9, 9, 210,
    ]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100, -100, -100, -100, -100, -100, -100, 9, 9, 210]


def test_role_boundaries_override_enables_unverified_strategy():
    """Override lets users opt in to role masking on strategies that default opt out."""
    vocab = {
        "BOA": [50, 51],
        "EOT": [60],
    }
    strategy = ProcessingStrategy(
        _Processor(_Tokenizer(vocab, pad_id=0)),
        role_boundaries_override=[
            {"role": "assistant", "start": "BOA", "end": "EOT"},
        ],
    )
    seq = [1, 2, 3, 50, 51, 7, 8, 60, 9]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100, -100, -100, -100, -100, 7, 8, 60, -100]


def test_role_boundaries_override_eos_token_sentinel():
    vocab = {"BOA": [50]}
    tok = _Tokenizer(vocab, pad_id=0, eos_id=99)
    strategy = ProcessingStrategy(
        _Processor(tok),
        role_boundaries_override=[
            {"role": "assistant", "start": "BOA", "end": "eos_token"},
        ],
    )
    seq = [1, 50, 7, 7, 99, 2]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100, -100, 7, 7, 99, -100]


def test_role_boundaries_override_end_null_runs_to_sequence_end():
    vocab = {"BOA": [50]}
    strategy = ProcessingStrategy(
        _Processor(_Tokenizer(vocab, pad_id=0)),
        role_boundaries_override=[
            {"role": "assistant", "start": "BOA", "end": None},
        ],
    )
    seq = [1, 2, 50, 7, 8, 9]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100, -100, -100, 7, 8, 9]


def test_role_boundaries_override_rejects_bad_spec():
    vocab = {"BOA": [50]}
    with pytest.raises(ValueError, match="must have both 'role' and 'start'"):
        ProcessingStrategy(
            _Processor(_Tokenizer(vocab, pad_id=0)),
            role_boundaries_override=[{"role": "assistant"}],
        )


def test_role_boundaries_override_rejects_unencodable_start():
    vocab = {"BOA": [50]}
    with pytest.raises(ValueError, match="tokenizes to an empty sequence"):
        ProcessingStrategy(
            _Processor(_Tokenizer(vocab, pad_id=0)),
            role_boundaries_override=[
                {"role": "assistant", "start": "MISSING", "end": None}
            ],
        )


def test_role_boundaries_override_accepts_pydantic_models():
    # cfg.role_boundaries arrives as RoleBoundarySpec after pydantic parsing.
    from axolotl.utils.schemas.multimodal import RoleBoundarySpec

    vocab = {"BOA": [50], "EOT": [60]}
    strategy = ProcessingStrategy(
        _Processor(_Tokenizer(vocab, pad_id=0)),
        role_boundaries_override=[
            RoleBoundarySpec(role="assistant", start="BOA", end="EOT")
        ],
    )
    assert len(strategy.role_boundaries) == 1
    assert strategy.role_boundaries[0].role == "assistant"
    assert strategy.role_boundaries[0].start_tokens == [50]
    assert strategy.role_boundaries[0].end_tokens == [60]


def test_base_strategy_warns_when_no_boundaries(caplog):
    """No boundaries + train_on_inputs=False: one-shot warning, labels unchanged."""
    import axolotl.processing_strategies as mod
    mod._ROLE_MASK_WARNED.discard("ProcessingStrategy")

    vocab = {"dummy": [1]}
    s = ProcessingStrategy(_Processor(_Tokenizer(vocab, pad_id=0)))
    import logging
    with caplog.at_level(logging.WARNING, logger="axolotl.processing_strategies"):
        labels = s.process_labels(torch.tensor([[1, 2, 3]]))
    assert labels.tolist() == [[1, 2, 3]]
    assert any("role boundaries" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# processor_kwargs plumbing (#3617)
# --------------------------------------------------------------------------- #


def _load_processor_module():
    """Load processor.py directly to bypass loaders/__init__'s bitsandbytes import."""
    import importlib.util
    import pathlib
    import sys
    import types

    if "posthog" not in sys.modules:
        stub = types.ModuleType("posthog")
        stub.Posthog = lambda *a, **kw: None  # type: ignore[attr-defined]
        sys.modules["posthog"] = stub

    src = pathlib.Path(__file__).parent.parent / "src" / "axolotl" / "loaders" / "processor.py"
    spec = importlib.util.spec_from_file_location(
        "axolotl.loaders.processor", str(src)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_processor_forwards_processor_kwargs():
    """cfg.processor_kwargs merges through, but revision/trust_remote_code stay axolotl-managed."""
    from axolotl.utils.dict import DictDefault
    processor_mod = _load_processor_module()

    captured_kwargs = {}

    class _FakeProcessor:
        size = {}
        tokenizer = None

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            captured_kwargs.update(kwargs)
            return cls()

    processor_mod.AutoProcessor = _FakeProcessor

    cfg = DictDefault({
        "processor_type": None,
        "processor_config": "fake/path",
        "revision_of_model": "abc123",
        "trust_remote_code": False,
        "tokenizer_use_mistral_common": False,
        "image_size": None,
        "processor_kwargs": {
            "min_pixels": 128 * 128,
            "max_pixels": 768 * 768,
            "revision": "HIJACKED",          # filtered out
            "trust_remote_code": True,       # filtered out
        },
    })
    processor_mod.load_processor(cfg, tokenizer=object())

    assert captured_kwargs["min_pixels"] == 128 * 128
    assert captured_kwargs["max_pixels"] == 768 * 768
    assert captured_kwargs["revision"] == "abc123"
    assert captured_kwargs["trust_remote_code"] is False


def test_load_processor_handles_absent_processor_kwargs():
    from axolotl.utils.dict import DictDefault
    processor_mod = _load_processor_module()

    captured_kwargs = {}

    class _FakeProcessor:
        size = {}
        tokenizer = None

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            captured_kwargs.update(kwargs)
            return cls()

    processor_mod.AutoProcessor = _FakeProcessor

    cfg = DictDefault({
        "processor_type": None,
        "processor_config": "fake/path",
        "revision_of_model": None,
        "trust_remote_code": False,
        "tokenizer_use_mistral_common": False,
        "image_size": None,
    })
    processor_mod.load_processor(cfg, tokenizer=object())
    assert captured_kwargs == {"trust_remote_code": False}


# --------------------------------------------------------------------------- #
# Additional edge-case coverage
# --------------------------------------------------------------------------- #


def test_scanner_batch_size_greater_than_one():
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
        RoleBoundary(role="user", start_tokens=[1, 3], end_tokens=[9]),
    ]
    labels = torch.tensor(
        [
            [1, 3, 7, 9, 1, 2, 8, 9],
            [1, 2, 5, 5, 9, 0, 0, 0],
        ]
    )
    out = _apply_role_boundaries(
        labels, boundaries, {"assistant"}, "turn"
    ).tolist()
    assert out[0] == [-100, -100, -100, -100, -100, -100, 8, 9]
    assert out[1] == [-100, -100, 5, 5, 9, -100, -100, -100]


def test_scanner_adjacent_trainable_turns():
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
    ]
    seq = [1, 2, 5, 9, 1, 2, 6, 9]
    out = _scan(boundaries, seq)
    assert out == [-100, -100, 5, 9, -100, -100, 6, 9]


def test_scanner_train_on_eos_none_multi_turn():
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
        RoleBoundary(role="user", start_tokens=[1, 3], end_tokens=[9]),
    ]
    seq = [1, 3, 7, 9, 1, 2, 8, 9, 1, 3, 7, 9, 1, 2, 6, 9]
    out = _scan(boundaries, seq, train_on_eos="none")
    assert out == [
        -100, -100, -100, -100, -100, -100, 8, -100,
        -100, -100, -100, -100, -100, -100, 6, -100,
    ]


def test_scanner_train_on_eos_all_with_user_turn_no_end_marker():
    """Unclosed non-trainable span with train_on_eos='all': nothing included, no crash."""
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[1, 2], end_tokens=[9]),
        RoleBoundary(role="user", start_tokens=[1, 3], end_tokens=[9]),
    ]
    seq = [1, 3, 7, 7, 7]
    out = _scan(boundaries, seq, train_on_eos="all")
    assert out == [-100, -100, -100, -100, -100]


def test_scanner_include_start_true_via_override():
    vocab = {"BOA": [50, 51], "EOT": [60]}
    strategy = ProcessingStrategy(
        _Processor(_Tokenizer(vocab, pad_id=0)),
        role_boundaries_override=[
            {
                "role": "assistant",
                "start": "BOA",
                "end": "EOT",
                "include_start": True,
            },
        ],
    )
    seq = [1, 50, 51, 7, 8, 60, 9]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100, 50, 51, 7, 8, 60, -100]


def test_scanner_include_end_false_via_override():
    """include_end=False drops end marker even with train_on_eos='turn'."""
    vocab = {"BOA": [50], "EOT": [60]}
    strategy = ProcessingStrategy(
        _Processor(_Tokenizer(vocab, pad_id=0)),
        role_boundaries_override=[
            {
                "role": "assistant",
                "start": "BOA",
                "end": "EOT",
                "include_end": False,
            },
        ],
    )
    seq = [1, 50, 7, 8, 60, 9]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100, -100, 7, 8, -100, -100]


def test_scanner_empty_start_tokens_is_defensive_noop():
    """Defensive: empty start_tokens matches nothing; everything masked."""
    boundaries = [
        RoleBoundary(role="assistant", start_tokens=[], end_tokens=[9]),
    ]
    seq = [1, 2, 3, 4, 9]
    out = _scan(boundaries, seq)
    assert out == [-100] * 5


def test_process_labels_masks_pad_inside_assistant_span():
    """Pad inside a trainable span is still masked post-scan."""
    strategy = _make_qwen2vl()
    seq = [101, 102, 103, 8, 0, 8, 104]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100, -100, -100, 8, -100, 8, 104]


def test_process_labels_all_pad_sequence_does_not_crash():
    strategy = _make_qwen2vl()
    seq = [0, 0, 0, 0]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100, -100, -100, -100]


def test_qwen2vl_multiple_consecutive_assistant_turns():
    strategy = _make_qwen2vl()
    seq = [101, 102, 103, 8, 104, 101, 102, 103, 9, 104]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [
        -100, -100, -100, 8, 104,
        -100, -100, -100, 9, 104,
    ]


def test_qwen2vl_batch_of_two_rows():
    strategy = _make_qwen2vl()
    row_a = [101, 106, 103, 7, 104, 101, 102, 103, 8, 104]
    row_b = [101, 102, 103, 9, 104, 0, 0, 0, 0, 0]
    out = strategy.process_labels(torch.tensor([row_a, row_b])).tolist()
    assert out[0] == [-100, -100, -100, -100, -100, -100, -100, -100, 8, 104]
    assert out[1] == [-100, -100, -100, 9, 104, -100, -100, -100, -100, -100]


def test_qwen3_5_train_on_inputs_true_still_masks_video_pad():
    """train_on_inputs=True skips role masking but media tokens are still masked."""
    tok = _qwen_tokenizer()
    strategy = Qwen3_5ProcessingStrategy(_Processor(tok), train_on_inputs=True)
    seq = [101, 106, 103, 201, 7, 104, 101, 102, 103, 201, 8, 104]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    expected = list(seq)
    expected[3] = -100
    expected[9] = -100
    assert out == expected


def test_role_boundaries_override_role_not_in_roles_to_train():
    """Override covering only a non-trainable role masks everything."""
    vocab = {"BOU": [50], "EOT": [60]}
    strategy = ProcessingStrategy(
        _Processor(_Tokenizer(vocab, pad_id=0)),
        role_boundaries_override=[
            {"role": "user", "start": "BOU", "end": "EOT"},
        ],
    )
    seq = [1, 50, 7, 8, 60, 9]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100] * 6


def test_role_boundaries_override_include_start_flag_round_trips():
    from axolotl.utils.schemas.multimodal import RoleBoundarySpec

    vocab = {"BOA": [50], "EOT": [60]}
    strategy = ProcessingStrategy(
        _Processor(_Tokenizer(vocab, pad_id=0)),
        role_boundaries_override=[
            RoleBoundarySpec(
                role="assistant", start="BOA", end="EOT", include_start=True
            ),
        ],
    )
    assert len(strategy.role_boundaries) == 1
    assert strategy.role_boundaries[0].include_start is True
    assert strategy.role_boundaries[0].include_end is True


def test_multimodal_config_parses_dict_role_boundaries_to_specs():
    from axolotl.utils.schemas.multimodal import (
        MultiModalConfig,
        RoleBoundarySpec,
    )

    cfg = MultiModalConfig(
        role_boundaries=[
            {"role": "assistant", "start": "BOA", "end": "EOT"},
            {"role": "user", "start": "BOU", "end": "EOT"},
        ]
    )
    assert cfg.role_boundaries is not None
    assert len(cfg.role_boundaries) == 2
    assert all(isinstance(rb, RoleBoundarySpec) for rb in cfg.role_boundaries)

    vocab = {"BOA": [50], "BOU": [51], "EOT": [60]}
    strategy = ProcessingStrategy(
        _Processor(_Tokenizer(vocab, pad_id=0)),
        role_boundaries_override=cfg.role_boundaries,
    )
    seq = [51, 7, 60, 50, 8, 60]
    out = strategy.process_labels(torch.tensor([seq])).tolist()[0]
    assert out == [-100, -100, -100, -100, 8, 60]


def test_load_processor_empty_processor_kwargs_is_noop():
    """Empty processor_kwargs={} must not AttributeError."""
    from axolotl.utils.dict import DictDefault
    processor_mod = _load_processor_module()

    captured_kwargs = {}

    class _FakeProcessor:
        size = {}
        tokenizer = None

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            captured_kwargs.update(kwargs)
            return cls()

    processor_mod.AutoProcessor = _FakeProcessor

    cfg = DictDefault({
        "processor_type": None,
        "processor_config": "fake/path",
        "revision_of_model": None,
        "trust_remote_code": False,
        "tokenizer_use_mistral_common": False,
        "image_size": None,
        "processor_kwargs": {},
    })
    processor_mod.load_processor(cfg, tokenizer=object())
    assert captured_kwargs == {"trust_remote_code": False}
