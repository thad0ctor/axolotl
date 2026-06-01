"""Multimodal CPT streaming encoder + collator tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from axolotl.core.builders.causal import (
    HFCausalTrainerBuilder,
    _get_mm_cpt_config,
    _is_multimodal_cpt,
)
from axolotl.integrations.mm_tiling.tiling import (
    ImageTilingConfig,
    TilingImageTransform,
)
from axolotl.prompt_strategies.multimodal_pretrain import (
    build_image_token_spec,
    encode_multimodal_pretrain,
)
from axolotl.utils.collators.mm_pretrain import MultiModalPretrainDataCollator
from axolotl.utils.data.mm_packing import MultimodalPackingMetadata
from axolotl.utils.data.streaming import (
    _multimodal_metadata_ram_budget_mb,
    encode_packed_streaming_multimodal,
    encode_streaming_multimodal,
    wrap_streaming_dataset,
)
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline

_SMOLVLM = "HuggingFaceTB/SmolVLM-500M-Instruct"


@pytest.fixture(scope="module", name="smolvlm_processor")
@enable_hf_offline
def fixture_smolvlm_processor(
    download_smolvlm_500m_instruct_model,  # pylint: disable=unused-argument
):
    return AutoProcessor.from_pretrained(_SMOLVLM)


@pytest.fixture(scope="module", name="two_tiny_images")
def fixture_two_tiny_images(tmp_path_factory) -> list[Path]:
    d = tmp_path_factory.mktemp("mm_stream_imgs")
    out = []
    for i in range(2):
        p = d / f"dummy_{i}.png"
        arr = np.random.default_rng(i).integers(0, 255, (64, 64, 3)).astype("uint8")
        Image.fromarray(arr).save(p)
        out.append(p)
    return out


# ---- encode_streaming_multimodal ------------------------------------------


def test_encode_preserves_images_and_text(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    examples = {
        "text": [
            f"{spec.image_token}\nrow one",
            f"{spec.image_token}\nrow two slightly longer",
        ],
        "images": [[str(two_tiny_images[0])], [str(two_tiny_images[1])]],
    }
    out = encode_streaming_multimodal(
        examples,
        tokenizer=smolvlm_processor.tokenizer,
        max_tokens=2048,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )
    assert set(out) >= {"input_ids", "labels", "attention_mask", "images", "_mm_text"}
    assert len(out["input_ids"]) == 2
    assert out["images"] == [[str(two_tiny_images[0])], [str(two_tiny_images[1])]]
    # EOS appended -> input_ids len equals attention_mask len and > text
    for ids, mask in zip(out["input_ids"], out["attention_mask"], strict=True):
        assert len(ids) == len(mask) and len(ids) > 0
    # CPT: labels == input_ids pre-masking.
    for ids, lbls in zip(out["input_ids"], out["labels"], strict=True):
        assert ids == lbls


def test_encode_rejects_mismatch(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    examples = {
        "text": [f"{spec.image_token}{spec.image_token}\ntwo placeholders one image"],
        "images": [[str(two_tiny_images[0])]],
    }
    with pytest.raises(ValueError, match="occurrence"):
        encode_streaming_multimodal(
            examples,
            tokenizer=smolvlm_processor.tokenizer,
            max_tokens=2048,
            image_token=spec.image_token,
            image_token_id=spec.image_token_id,
        )


def test_encode_rejects_row_without_list(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    with pytest.raises(ValueError, match="list"):
        encode_streaming_multimodal(
            {
                "text": [f"{spec.image_token}\nrow one"],
                "images": [str(two_tiny_images[0])],  # scalar, not a list
            },
            tokenizer=smolvlm_processor.tokenizer,
            max_tokens=2048,
            image_token=spec.image_token,
            image_token_id=spec.image_token_id,
        )


def test_encode_counts_placeholders_on_full_text(smolvlm_processor, two_tiny_images):
    # The last placeholder must remain countable even when it's hundreds of
    # tokens deep — guards against a regression that adds tokenizer
    # truncation and silently drops trailing placeholders.
    spec = build_image_token_spec(smolvlm_processor)
    long_filler = "lorem ipsum " * 400
    text = f"{spec.image_token} {long_filler} {spec.image_token} {long_filler} {spec.image_token}"
    examples = {
        "text": [text],
        "images": [[str(two_tiny_images[0])] * 3],
    }
    out = encode_streaming_multimodal(
        examples,
        tokenizer=smolvlm_processor.tokenizer,
        max_tokens=4096,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )
    ids = out["input_ids"][0]
    # Sanity: the input is genuinely long, so a truncating regression would
    # have to cut into it to drop the last placeholder.
    assert len(ids) > 2000
    assert sum(1 for t in ids if t == spec.image_token_id) == 3


def test_encode_can_add_exact_processor_lengths(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    text = f"{spec.image_token}\nrow one"
    out = encode_multimodal_pretrain(
        {"text": [text], "images": [[str(two_tiny_images[0])]]},
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        max_tokens=2048,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        add_processor_lengths=True,
    )
    manual = smolvlm_processor(
        text=[text + smolvlm_processor.tokenizer.eos_token],
        images=[[Image.open(two_tiny_images[0])]],
        return_tensors="pt",
        padding=True,
    )
    assert out["length"] == [int(manual["attention_mask"].sum().item())]


def test_encode_packed_streaming_multimodal_emits_prepacked_rows(
    smolvlm_processor, two_tiny_images
):
    spec = build_image_token_spec(smolvlm_processor)
    out = encode_packed_streaming_multimodal(
        {
            "text": [
                f"{spec.image_token}\nrow one",
                f"{spec.image_token}\nrow two",
            ],
            "images": [[str(two_tiny_images[0])], [str(two_tiny_images[1])]],
        },
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        max_tokens=2048,
        batch_size=2,
        bin_size=200,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )

    assert len(out["_mm_text"]) == 1
    assert len(out["images"][0]) == 2
    assert len(out["_mm_sample_lengths"][0]) == 2
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        sample_packing=True,
    )
    batch = collator.torch_call([{key: out[key][0] for key in out}])
    assert batch["input_ids"].shape[0] == 1
    assert "position_ids" in batch


def test_encode_packed_streaming_multimodal_uses_metadata_cache(
    smolvlm_processor, two_tiny_images, tmp_path
):
    spec = build_image_token_spec(smolvlm_processor)
    cache_dir = tmp_path / "mm-pack-cache"
    out = encode_packed_streaming_multimodal(
        {
            "text": [
                f"{spec.image_token}\nrow one",
                f"{spec.image_token}\nrow two",
            ],
            "images": [[str(two_tiny_images[0])], [str(two_tiny_images[1])]],
        },
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        max_tokens=2048,
        batch_size=2,
        bin_size=200,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        metadata_cache_path=str(cache_dir),
        metadata_cache_ram_budget_mb=1,
    )

    assert out["_mm_visual_tokens"]
    assert out["_mm_visual_signature"]
    assert list(cache_dir.rglob("*.json"))


def test_encode_packed_streaming_multimodal_respects_visual_capacity(
    smolvlm_processor, two_tiny_images
):
    spec = build_image_token_spec(smolvlm_processor)
    examples = {
        "text": [
            f"{spec.image_token}\nrow one",
            f"{spec.image_token}\nrow two",
        ],
        "images": [[str(two_tiny_images[0])], [str(two_tiny_images[1])]],
    }
    individual = encode_packed_streaming_multimodal(
        examples,
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        max_tokens=2048,
        batch_size=1,
        bin_size=200,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )
    visual_capacity = max(v[0] for v in individual["_mm_visual_tokens"])

    packed = encode_packed_streaming_multimodal(
        examples,
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        max_tokens=2048,
        batch_size=2,
        bin_size=200,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        visual_capacity=visual_capacity,
    )

    assert len(packed["_mm_text"]) == 2
    assert all(len(v) == 1 for v in packed["_mm_visual_tokens"])


def test_encode_packed_streaming_multimodal_can_group_by_visual_signature(
    smolvlm_processor, two_tiny_images, monkeypatch
):
    import axolotl.utils.data.mm_packing as mm_packing

    spec = build_image_token_spec(smolvlm_processor)

    def fake_metadata(*_args, **_kwargs):
        return [
            MultimodalPackingMetadata(10, 10, 1, "grid-a"),
            MultimodalPackingMetadata(10, 10, 1, "grid-b"),
        ]

    monkeypatch.setattr(
        mm_packing, "compute_multimodal_packing_metadata", fake_metadata
    )

    packed = encode_packed_streaming_multimodal(
        {
            "text": [
                f"{spec.image_token}\nrow one",
                f"{spec.image_token}\nrow two",
            ],
            "images": [[str(two_tiny_images[0])], [str(two_tiny_images[1])]],
        },
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        max_tokens=2048,
        batch_size=1,
        bin_size=200,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        group_by_visual_signature=True,
    )

    assert len(packed["_mm_text"]) == 2
    assert packed["_mm_visual_signature"] == [["grid-a"], ["grid-b"]]


def test_encode_rejects_row_exceeding_max_tokens(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    huge = "word " * 5000
    examples = {
        "text": [f"{spec.image_token} {huge}"],
        "images": [[str(two_tiny_images[0])]],
    }
    with pytest.raises(ValueError, match="exceeds sequence_len"):
        encode_streaming_multimodal(
            examples,
            tokenizer=smolvlm_processor.tokenizer,
            max_tokens=512,
            image_token=spec.image_token,
            image_token_id=spec.image_token_id,
        )


# ---- build_image_token_spec autodetection --------------------------------


class _StubTokenizer:
    """Minimal tokenizer stub for autodetection tests."""

    def __init__(self, vocab: dict[str, int], unk_id: int = 0):
        self._vocab = vocab
        self.unk_token_id = unk_id
        self.all_special_tokens = list(vocab.keys())
        self.additional_special_tokens: list[str] = []

    def get_added_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, self.unk_token_id)


class _StubProcessor:
    def __init__(self, tokenizer, image_token=None, boi_token=None):
        self.tokenizer = tokenizer
        if image_token is not None:
            self.image_token = image_token
        if boi_token is not None:
            self.boi_token = boi_token


def test_build_image_token_spec_gemma4_uses_image_token_not_boi():
    """Gemma-4: `image_token` is the user-facing placeholder; don't swap to boi_token."""
    tok = _StubTokenizer({"<|image|>": 258880, "<|image>": 255999})
    proc = _StubProcessor(tok, image_token="<|image|>", boi_token="<|image>")
    spec = build_image_token_spec(proc)
    assert spec.image_token == "<|image|>"
    assert spec.image_token_id == 258880


def test_build_image_token_spec_gemma3_swaps_to_boi_token():
    """Gemma-3: `image_token` is the post-expansion soft token; placeholder is `boi_token`."""
    tok = _StubTokenizer({"<image_soft_token>": 262144, "<start_of_image>": 255999})
    proc = _StubProcessor(
        tok, image_token="<image_soft_token>", boi_token="<start_of_image>"
    )
    spec = build_image_token_spec(proc)
    assert spec.image_token == "<start_of_image>"
    assert spec.image_token_id == 255999


def test_build_image_token_spec_override_not_special_rejected():
    """Override that isn't a registered special token is rejected (would BPE-tokenize)."""
    tok = _StubTokenizer({"<|image|>": 258880})
    proc = _StubProcessor(tok, image_token="<|image|>")
    with pytest.raises(ValueError, match="not a registered special token"):
        build_image_token_spec(proc, override="not_a_real_token")


def test_build_image_token_spec_override_resolves_to_unk_rejected():
    """Override that resolves to unk is rejected with a clear error."""
    tok = _StubTokenizer({"<|image|>": 258880, "<|fake|>": 0}, unk_id=0)
    proc = _StubProcessor(tok, image_token="<|image|>")
    with pytest.raises(ValueError, match="did not resolve"):
        build_image_token_spec(proc, override="<|fake|>")


def test_build_image_token_spec_no_candidates_raises():
    """If neither processor attrs nor any known candidate resolve, raise a clear error."""
    tok = _StubTokenizer({})  # nothing registered
    proc = _StubProcessor(tok)  # no image_token, no boi_token
    with pytest.raises(ValueError, match="Could not autodetect"):
        build_image_token_spec(proc)


# ---- wrap_streaming_dataset routing --------------------------------------


def _patch_streaming_partial(monkeypatch, fake_partial):
    import axolotl.utils.data.streaming as streaming_mod

    if hasattr(streaming_mod, "partial"):
        monkeypatch.setattr(streaming_mod, "partial", fake_partial)
    else:
        monkeypatch.setattr(streaming_mod.functools, "partial", fake_partial)


def test_wrap_streaming_dataset_uses_pretraining_config_arg(
    smolvlm_processor, monkeypatch
):
    # Eval path passes a per-entry config that may differ from cfg.pretraining_dataset[0].
    # The MM-CPT branch must read from that arg, not re-resolve from cfg.
    captured = {}

    def fake_partial(fn, **kwargs):
        captured["encode_fn"] = fn
        captured["kwargs"] = kwargs
        return lambda batch: batch

    _patch_streaming_partial(monkeypatch, fake_partial)

    class _Dataset:
        features = {"text": None, "images": None}

        def shuffle(self, **_):
            return self

        def map(self, *_args, **_kwargs):
            return self

    cfg = DictDefault(
        {
            "sample_packing": False,
            "pretraining_dataset": [
                {
                    "path": "train/ds",
                    "type": "multimodal_pretrain",
                    "text_column": "wrong_train_col",
                    "image_column": "wrong_train_imgs",
                }
            ],
            "sequence_len": 256,
            "shuffle_merged_datasets": False,
            "streaming_multipack_buffer_size": 1000,
            "seed": 42,
        }
    )
    eval_entry = DictDefault(
        {
            "path": "test/ds",
            "type": "multimodal_pretrain",
            "text_column": "eval_text",
            "image_column": "eval_imgs",
        }
    )

    wrap_streaming_dataset(
        _Dataset(),
        smolvlm_processor.tokenizer,
        cfg,
        ds_wrapper_fn=None,
        processor=smolvlm_processor,
        pretraining_config=eval_entry,
    )

    assert captured["encode_fn"] is encode_streaming_multimodal
    assert captured["kwargs"]["text_column"] == "eval_text"
    assert captured["kwargs"]["image_column"] == "eval_imgs"


def test_wrap_streaming_dataset_eval_honors_eval_sequence_len(
    smolvlm_processor, monkeypatch
):
    """is_eval=True with cfg.eval_sequence_len set caps encoder at eval_sequence_len."""
    captured = {}

    def fake_partial(fn, **kwargs):
        captured["encode_fn"] = fn
        captured["kwargs"] = kwargs
        return lambda batch: batch

    _patch_streaming_partial(monkeypatch, fake_partial)

    class _Dataset:
        features = {"text": None, "images": None}

        def shuffle(self, **_):
            return self

        def map(self, *_args, **_kwargs):
            return self

    cfg = DictDefault(
        {
            "sample_packing": False,
            "pretraining_dataset": [
                {"path": "train/ds", "type": "multimodal_pretrain"}
            ],
            "sequence_len": 4096,
            "eval_sequence_len": 1024,
            "shuffle_merged_datasets": False,
            "streaming_multipack_buffer_size": 1000,
            "seed": 42,
        }
    )

    wrap_streaming_dataset(
        _Dataset(),
        smolvlm_processor.tokenizer,
        cfg,
        ds_wrapper_fn=None,
        processor=smolvlm_processor,
        pretraining_config=DictDefault(
            {"path": "test/ds", "type": "multimodal_pretrain"}
        ),
        is_eval=True,
    )
    assert captured["kwargs"]["max_tokens"] == 1024

    captured.clear()
    wrap_streaming_dataset(
        _Dataset(),
        smolvlm_processor.tokenizer,
        cfg,
        ds_wrapper_fn=None,
        processor=smolvlm_processor,
        pretraining_config=DictDefault(
            {"path": "train/ds", "type": "multimodal_pretrain"}
        ),
        is_eval=False,
    )
    assert captured["kwargs"]["max_tokens"] == 4096

    # eval_sequence_len unset -> eval falls back to sequence_len.
    captured.clear()
    cfg_no_eval = DictDefault(
        {
            "sample_packing": False,
            "pretraining_dataset": [
                {"path": "train/ds", "type": "multimodal_pretrain"}
            ],
            "sequence_len": 4096,
            "shuffle_merged_datasets": False,
            "streaming_multipack_buffer_size": 1000,
            "seed": 42,
        }
    )
    wrap_streaming_dataset(
        _Dataset(),
        smolvlm_processor.tokenizer,
        cfg_no_eval,
        ds_wrapper_fn=None,
        processor=smolvlm_processor,
        pretraining_config=DictDefault(
            {"path": "test/ds", "type": "multimodal_pretrain"}
        ),
        is_eval=True,
    )
    assert captured["kwargs"]["max_tokens"] == 4096


def test_wrap_streaming_dataset_uses_mm_packed_encoder(smolvlm_processor, monkeypatch, mm_tiling_plugin):
    captured = {}

    def fake_partial(fn, **kwargs):
        captured["encode_fn"] = fn
        captured["kwargs"] = kwargs
        return lambda batch: batch

    _patch_streaming_partial(monkeypatch, fake_partial)

    class _Dataset:
        features = {"text": None, "images": None}

        def shuffle(self, **_):
            return self

        def map(self, *_args, **_kwargs):
            return self

    cfg = DictDefault(
        {
            "sample_packing": True,
            "pretraining_dataset": [
                {"path": "train/ds", "type": "multimodal_pretrain"}
            ],
            "sequence_len": 2048,
            "micro_batch_size": 2,
            "sample_packing_bin_size": 200,
            "shuffle_merged_datasets": False,
            "streaming_multipack_buffer_size": 1000,
            "multimodal_sample_packing_cache_path": "/tmp/mm-cache",
            "multimodal_sample_packing_ram_budget_mb": 128,
            "multimodal_sample_packing_visual_capacity": 4096,
            "multimodal_sample_packing_group_by_visual_signature": True,
            "image_size": 512,
            "image_resize_algorithm": Image.Resampling.BICUBIC,
            "image_resize_buckets": [(1024, 1536), (1536, 1536)],
            "image_resize_no_upscale": True,
            "image_resize_pad_color": "white",
            "image_tiling": True,
            "image_tiling_tile_size": 1024,
            "image_tiling_grid": (2, 3),
            "image_tiling_overview_buckets": [(1024, 1536)],
            "image_tiling_shape_buckets": "ocr_pages",
            "seed": 42,
        }
    )
    wrap_streaming_dataset(
        _Dataset(),
        smolvlm_processor.tokenizer,
        cfg,
        ds_wrapper_fn=None,
        processor=smolvlm_processor,
        pretraining_config=DictDefault(
            {"path": "train/ds", "type": "multimodal_pretrain"}
        ),
    )

    assert captured["encode_fn"] is encode_packed_streaming_multimodal
    assert captured["kwargs"]["batch_size"] == 2
    assert captured["kwargs"]["metadata_cache_path"] == "/tmp/mm-cache"
    assert captured["kwargs"]["metadata_cache_ram_budget_mb"] == 128
    assert captured["kwargs"]["visual_capacity"] == 4096
    assert captured["kwargs"]["group_by_visual_signature"] is True
    assert captured["kwargs"]["use_multimodal_sample_packing"] is True
    assert captured["kwargs"]["image_size"] == 512
    assert captured["kwargs"]["image_resize_algorithm"] == Image.Resampling.BICUBIC
    assert captured["kwargs"]["image_resize_buckets"] == [(1024, 1536), (1536, 1536)]
    assert captured["kwargs"]["image_resize_no_upscale"] is True
    assert captured["kwargs"]["image_resize_pad_color"] == "white"
    assert captured["kwargs"]["image_transform"].policy_payload()["tile_size"] == 1024
    assert captured["kwargs"]["image_transform"].policy_payload()["shape_buckets"]
    assert cfg.micro_batch_size == 1


def test_multimodal_metadata_ram_budget_can_split_by_worker(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    cfg = DictDefault(
        {
            "multimodal_sample_packing_ram_budget_mb": 1024,
            "multimodal_sample_packing_split_ram_budget_by_worker": True,
            "multimodal_sample_packing_dataloader": True,
            "multimodal_sample_packing_dataloader_num_workers": 4,
            "dataloader_num_workers": 8,
        }
    )

    assert _multimodal_metadata_ram_budget_mb(cfg) == 128


def test_mm_cpt_detection_includes_nonstreaming_datasets():
    cfg = DictDefault(
        {
            "pretraining_dataset": None,
            "datasets": [
                {
                    "path": "train/ds",
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/train/images",
                }
            ],
        }
    )

    assert _is_multimodal_cpt(cfg)
    assert _get_mm_cpt_config(cfg)["image_base_dir"] == "/train/images"


def test_mm_cpt_collator_uses_nonstreaming_dataset_config(mm_tiling_plugin):
    tok = _StubTokenizer({"<image>": 42})
    processor = _StubProcessor(tok, image_token="<image>")
    builder = object.__new__(HFCausalTrainerBuilder)
    builder.tokenizer = tok
    builder.processor = processor
    builder.cfg = DictDefault(
        {
            "pretraining_dataset": None,
            "datasets": [
                {
                    "path": "train/ds",
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/train/images",
                    "image_token": "<image>",
                }
            ],
            "test_datasets": None,
            "sequence_len": 128,
            "eval_sequence_len": None,
            "image_size": 512,
            "image_resize_algorithm": Image.Resampling.BICUBIC,
            "image_resize_buckets": [(1024, 1536), (1536, 1536)],
            "image_resize_no_upscale": True,
            "image_resize_pad_color": "white",
            "image_tiling": True,
            "image_tiling_tile_size": 512,
            "image_tiling_grid": (2, 3),
            "image_tiling_shape_buckets": "ocr_pages",
        }
    )

    collator = HFCausalTrainerBuilder._build_mm_pretrain_collator(builder)

    assert isinstance(collator, MultiModalPretrainDataCollator)
    assert collator.image_base_dir == "/train/images"
    assert collator.image_size == 512
    assert collator.image_resize_algorithm == Image.Resampling.BICUBIC
    assert collator.image_resize_buckets == [(1024, 1536), (1536, 1536)]
    assert collator.image_resize_no_upscale is True
    assert collator.image_resize_pad_color == "white"
    assert collator.image_transform is not None
    assert collator.image_transform.policy_payload()["tile_size"] == 512


def test_mm_cpt_packed_collator_pads_to_full_pack_capacity():
    tok = _StubTokenizer({"<image>": 42})
    processor = _StubProcessor(tok, image_token="<image>")
    builder = object.__new__(HFCausalTrainerBuilder)
    builder.tokenizer = tok
    builder.processor = processor
    builder.cfg = DictDefault(
        {
            "pretraining_dataset": None,
            "datasets": [{"path": "train/ds", "type": "multimodal_pretrain"}],
            "test_datasets": None,
            "sequence_len": 128,
            "eval_sequence_len": None,
            "sample_packing": True,
            "eval_sample_packing": True,
            "multipack_real_batches": False,
            "micro_batch_size": 4,
            "pad_to_sequence_len": True,
        }
    )

    collator = HFCausalTrainerBuilder._build_mm_pretrain_collator(
        builder,
        pad_to_multiple_of=128,
    )

    assert collator.max_length == 512
    assert collator.pad_to_multiple_of == 512


# ---- MultiModalPretrainDataCollator ---------------------------------------


def test_collator_builds_batch_and_masks_labels(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    encoded = encode_streaming_multimodal(
        {
            "text": [
                f"{spec.image_token}\nrow one",
                f"{spec.image_token}\nrow two slightly longer",
            ],
            "images": [[str(two_tiny_images[0])], [str(two_tiny_images[1])]],
        },
        tokenizer=smolvlm_processor.tokenizer,
        max_tokens=2048,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )
    rows = [
        {
            k: encoded[k][i]
            for k in ("input_ids", "labels", "attention_mask", "images", "_mm_text")
        }
        for i in range(2)
    ]
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    batch = collator.torch_call(rows)
    # Expected keys
    for k in ("input_ids", "attention_mask", "pixel_values", "labels"):
        assert k in batch, f"missing batch key {k}"
    assert isinstance(batch["input_ids"], torch.Tensor)
    # Label masking check: no image-family ids remaining as valid labels.
    for tid in spec.image_family_token_ids:
        assert int((batch["labels"] == tid).sum().item()) == 0, (
            f"label masking left id={tid} in labels"
        )
    # Pad is also masked.
    pad_id = smolvlm_processor.tokenizer.pad_token_id
    if pad_id is not None:
        assert int((batch["labels"] == pad_id).sum().item()) == 0


def test_collator_resizes_images_before_processor(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        image_size=32,
        image_resize_algorithm=Image.Resampling.BICUBIC,
        image_resize_pad_color="white",
    )

    loaded = collator._load_images_for_row([str(two_tiny_images[0])], row_index=0)

    assert loaded[0].size == (32, 32)


def test_collator_tiling_expands_cpt_placeholders_and_images(
    smolvlm_processor,
    two_tiny_images,
):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        image_transform=TilingImageTransform(
            ImageTilingConfig(
                tile_size=16,
                grid=(2, 1),
                overview_size=16,
            )
        ),
    )

    text, images = collator._prepare_row_text_and_images(
        f"{spec.image_token}\nrow one",
        [str(two_tiny_images[0])],
        row_index=0,
    )

    assert text.count(spec.image_token) == 3
    assert len(images) == 3
    assert all(image.size == (16, 16) for image in images)


def test_packed_collator_builds_boundary_masks(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    encoded = encode_multimodal_pretrain(
        {
            "text": [
                f"{spec.image_token}\nrow one",
                f"{spec.image_token}\nrow two slightly longer",
            ],
            "images": [[str(two_tiny_images[0])], [str(two_tiny_images[1])]],
        },
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        max_tokens=2048,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        add_processor_lengths=True,
    )
    rows = [
        {
            k: encoded[k][i]
            for k in (
                "input_ids",
                "labels",
                "attention_mask",
                "images",
                "_mm_text",
                "length",
            )
        }
        for i in range(2)
    ]
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        sample_packing=True,
    )

    batch = collator.torch_call([rows])

    assert batch["input_ids"].shape[0] == 1
    assert "position_ids" in batch
    lengths = encoded["length"]
    total_len = sum(lengths)
    pos = batch["position_ids"][0, :total_len].tolist()
    mask = batch["attention_mask"][0, :total_len].tolist()
    assert pos[0] == 0
    assert pos[lengths[0]] == 0
    assert pos[lengths[0] - 1] == lengths[0] - 1
    assert set(mask[: lengths[0]]) == {1}


def test_packed_collator_tiled_lengths_match_processor(
    smolvlm_processor, two_tiny_images
):
    spec = build_image_token_spec(smolvlm_processor)
    tiling_config = ImageTilingConfig(
        tile_size=16,
        grid=(2, 1),
        overview_size=16,
    )
    encoded = encode_multimodal_pretrain(
        {
            "text": [
                f"{spec.image_token}\nrow one",
                f"{spec.image_token}\nrow two",
            ],
            "images": [[str(two_tiny_images[0])], [str(two_tiny_images[1])]],
        },
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        max_tokens=2048,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        add_processor_lengths=True,
        image_transform=TilingImageTransform(tiling_config),
    )
    rows = [
        {
            k: encoded[k][i]
            for k in (
                "input_ids",
                "labels",
                "attention_mask",
                "images",
                "_mm_text",
                "length",
            )
        }
        for i in range(2)
    ]
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        sample_packing=True,
        image_transform=TilingImageTransform(tiling_config),
    )

    batch = collator.torch_call([rows])

    lengths = encoded["length"]
    total_len = sum(lengths)
    assert batch["position_ids"][0, lengths[0]].item() == 0
    assert set(batch["attention_mask"][0, : lengths[0]].tolist()) == {1}
    assert set(batch["attention_mask"][0, lengths[0] : total_len].tolist()) == {2}


def test_collator_raises_on_missing_columns(smolvlm_processor):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    with pytest.raises(KeyError, match="encode_streaming_multimodal"):
        collator.torch_call([{"input_ids": [1, 2, 3]}])  # no _mm_text / images


# ---- input validation -----------------------------------------------------


def test_collator_rejects_bytes_mm_text(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    rows = [
        {
            "_mm_text": f"{spec.image_token}\nrow".encode(),
            "images": [str(two_tiny_images[0])],
        }
    ]
    with pytest.raises(TypeError, match="`_mm_text` must be str"):
        collator.torch_call(rows)


def test_collator_sanitizes_error_message(smolvlm_processor, tmp_path):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    missing = tmp_path / "subdir_with_secret_name" / "nope.png"
    with pytest.raises(RuntimeError) as exc:
        collator._load_images_for_row([str(missing)], row_index=3)
    # basename appears, full directory path does NOT
    assert "nope.png" in str(exc.value)
    assert "subdir_with_secret_name" not in str(exc.value)
    assert "Row 3" in str(exc.value)


def test_collator_skip_bad_images_drops_row_and_continues(
    smolvlm_processor, two_tiny_images, tmp_path
):
    """skip_bad_images=True: bad row drops, batch survives on remaining rows."""
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        skip_bad_images=True,
    )
    rows = [
        {
            "_mm_text": f"{spec.image_token}\ngood row",
            "images": [str(two_tiny_images[0])],
        },
        {
            "_mm_text": f"{spec.image_token}\nbad row",
            "images": [str(tmp_path / "missing.png")],
        },
    ]
    batch = collator.torch_call(rows)
    # Surviving row produced a batch with pixel_values from the good image.
    assert "input_ids" in batch and "pixel_values" in batch
    assert batch["input_ids"].shape[0] == 1


def test_collator_all_rows_dropped_raises(smolvlm_processor, tmp_path):
    """skip_bad_images=True with every row failing surfaces a RuntimeError."""
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        skip_bad_images=True,
    )
    rows = [
        {
            "_mm_text": f"{spec.image_token}\nrow",
            "images": [str(tmp_path / f"missing_{i}.png")],
        }
        for i in range(2)
    ]
    with pytest.raises(RuntimeError, match="All rows in the batch were dropped"):
        collator.torch_call(rows)


# ---- mixed / all-text batches --------------------------------------------


def test_collator_warns_when_tokenizer_diverges_from_processor_tokenizer(
    smolvlm_processor, caplog, monkeypatch
):
    """Construct-time warning when self.tokenizer is not processor.tokenizer."""
    import logging as _logging

    # `axolotl` logger has propagate=False (logging_config.py); flip it so
    # caplog's root handler receives the record.
    monkeypatch.setattr(_logging.getLogger("axolotl"), "propagate", True)
    spec = build_image_token_spec(smolvlm_processor)

    # Same tokenizer: no warning.
    with caplog.at_level(
        _logging.WARNING, logger="axolotl.utils.collators.mm_pretrain"
    ):
        MultiModalPretrainDataCollator(
            tokenizer=smolvlm_processor.tokenizer,
            processor=smolvlm_processor,
            image_token_spec=spec,
        )
    assert not any("tokenize inconsistently" in r.getMessage() for r in caplog.records)

    caplog.clear()

    # Different tokenizer instance (a stand-in object): warning fires.
    class _OtherTokenizer:
        pad_token_id = None

    with caplog.at_level(
        _logging.WARNING, logger="axolotl.utils.collators.mm_pretrain"
    ):
        MultiModalPretrainDataCollator(
            tokenizer=_OtherTokenizer(),
            processor=smolvlm_processor,
            image_token_spec=spec,
        )
    assert any("tokenize inconsistently" in r.getMessage() for r in caplog.records)


def test_collator_all_text_batch_uses_tokenizer_fallback(smolvlm_processor):
    """A batch where every row has images=[] tokenizes via the tokenizer; no pixel_values."""
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    rows = [
        {"_mm_text": "first text-only row", "images": []},
        {"_mm_text": "second text-only row, slightly longer", "images": []},
    ]
    batch = collator.torch_call(rows)
    for k in ("input_ids", "attention_mask", "labels"):
        assert k in batch, f"missing batch key {k}"
    assert "pixel_values" not in batch
    assert isinstance(batch["input_ids"], torch.Tensor)
    pad_id = smolvlm_processor.tokenizer.pad_token_id
    if pad_id is not None:
        assert int((batch["labels"] == pad_id).sum().item()) == 0


def test_collator_mixed_batch_still_succeeds(smolvlm_processor, two_tiny_images):
    """A batch with one imaged row and one text-only row still produces pixel_values."""
    spec = build_image_token_spec(smolvlm_processor)
    encoded = encode_streaming_multimodal(
        {
            "text": [
                f"{spec.image_token}\nimaged row",
                "text-only row",
            ],
            "images": [[str(two_tiny_images[0])], []],
        },
        tokenizer=smolvlm_processor.tokenizer,
        max_tokens=2048,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )
    rows = [
        {
            k: encoded[k][i]
            for k in ("input_ids", "labels", "attention_mask", "images", "_mm_text")
        }
        for i in range(2)
    ]
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    batch = collator.torch_call(rows)
    for k in ("input_ids", "attention_mask", "pixel_values", "labels"):
        assert k in batch, f"missing batch key {k}"
