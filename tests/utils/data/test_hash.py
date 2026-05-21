"""
Tests for generate_dataset_hash_from_config.

Regression test for https://github.com/axolotl-ai-cloud/axolotl/issues/3303:
changing output_dir should not bust the dataset cache when added_tokens_overrides
is set.
"""

from axolotl.utils.data.shared import (
    generate_dataset_hash_from_config,
    generate_pretraining_dataset_hash,
)
from axolotl.utils.dict import DictDefault


def _base_cfg(**kwargs):
    return DictDefault(
        {
            "sequence_len": 2048,
            "sample_packing": False,
            "eval_sample_packing": False,
            "group_by_length": False,
            "kd_temperature": None,
            "dataset_exact_deduplication": False,
            "tokenizer_config": "NousResearch/Llama-3.2-1B",
            **kwargs,
        }
    )


def _datasets():
    return [
        DictDefault(
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca",
                "shards": None,
                "conversation": None,
                "split": "train",
                "temperature": None,
            }
        )
    ]


class TestGenerateDatasetHashFromConfig:
    def test_same_config_same_hash(self):
        """Identical configs produce identical hashes."""
        cfg = _base_cfg()
        h1 = generate_dataset_hash_from_config(
            cfg, _datasets(), "NousResearch/Llama-3.2-1B"
        )
        h2 = generate_dataset_hash_from_config(
            cfg, _datasets(), "NousResearch/Llama-3.2-1B"
        )
        assert h1 == h2

    def test_different_tokenizer_different_hash(self):
        """A different tokenizer path produces a different hash."""
        cfg = _base_cfg()
        h1 = generate_dataset_hash_from_config(
            cfg, _datasets(), "NousResearch/Llama-3.2-1B"
        )
        h2 = generate_dataset_hash_from_config(
            cfg, _datasets(), "HuggingFaceTB/SmolLM2-135M"
        )
        assert h1 != h2

    def test_different_sequence_len_different_hash(self):
        cfg_a = _base_cfg(sequence_len=2048)
        cfg_b = _base_cfg(sequence_len=4096)
        h1 = generate_dataset_hash_from_config(cfg_a, _datasets(), "tok")
        h2 = generate_dataset_hash_from_config(cfg_b, _datasets(), "tok")
        assert h1 != h2

    # --- Regression: added_tokens_overrides + output_dir ---

    def test_added_tokens_overrides_hash_stable_across_output_dir(self):
        """Hash must not change when only output_dir changes (issue #3303).

        When added_tokens_overrides is set the tokenizer is saved into output_dir,
        making tokenizer.name_or_path an absolute path that includes output_dir.
        The hash should be derived from the canonical tokenizer config + overrides,
        not from the output-dir-dependent path.
        """
        cfg_run1 = _base_cfg(
            output_dir="/tmp/run_1",
            added_tokens_overrides={32000: "<PAD>", 32001: "<MASK>"},
        )
        cfg_run2 = _base_cfg(
            output_dir="/tmp/run_2_different_name",
            added_tokens_overrides={32000: "<PAD>", 32001: "<MASK>"},
        )

        # Simulate what happens in practice: tokenizer.name_or_path becomes the
        # output_dir-based path after modify_tokenizer_files() saves the tokenizer.
        tokenizer_name_run1 = "/tmp/run_1/modified_tokenizer"
        tokenizer_name_run2 = "/tmp/run_2_different_name/modified_tokenizer"

        h1 = generate_dataset_hash_from_config(
            cfg_run1, _datasets(), tokenizer_name_run1
        )
        h2 = generate_dataset_hash_from_config(
            cfg_run2, _datasets(), tokenizer_name_run2
        )

        assert h1 == h2, (
            "Dataset cache hash must not change when only output_dir changes "
            "while added_tokens_overrides stays the same (issue #3303)."
        )

    def test_added_tokens_overrides_different_overrides_different_hash(self):
        """Different added_tokens_overrides produce different hashes."""
        cfg_a = _base_cfg(
            output_dir="/tmp/run_a",
            added_tokens_overrides={32000: "<PAD>"},
        )
        cfg_b = _base_cfg(
            output_dir="/tmp/run_a",  # same output_dir
            added_tokens_overrides={32000: "<OTHER>"},
        )
        tokenizer_path = "/tmp/run_a/modified_tokenizer"

        h1 = generate_dataset_hash_from_config(cfg_a, _datasets(), tokenizer_path)
        h2 = generate_dataset_hash_from_config(cfg_b, _datasets(), tokenizer_path)

        assert h1 != h2

    def test_no_added_tokens_overrides_uses_tokenizer_name_as_before(self):
        """Without added_tokens_overrides the old behaviour is preserved."""
        cfg = _base_cfg()  # no added_tokens_overrides
        tokenizer_name = "NousResearch/Llama-3.2-1B"

        h1 = generate_dataset_hash_from_config(cfg, _datasets(), tokenizer_name)
        # Changing tokenizer_name still changes the hash
        h2 = generate_dataset_hash_from_config(cfg, _datasets(), "some/other-model")

        assert h1 != h2


def _mm_pretrain_cfg(**kwargs):
    return DictDefault(
        {
            "sequence_len": 2048,
            "eval_sequence_len": None,
            "tokenizer_config": "Qwen/Qwen3.5-VL-9B-Instruct",
            **kwargs,
        }
    )


def _mm_pretrain_entry(**kwargs):
    base = {
        "path": "json",
        "name": None,
        "split": "train",
        "data_files": "/data/train.jsonl",
        "ds_type": "json",
        "type": "multimodal_pretrain",
        "text_column": "text",
        "image_column": "images",
        "image_base_dir": "/data/images",
        "image_token": None,
        "multimodal": None,
        "skip": None,
        "trust_remote_code": False,
    }
    base.update(kwargs)
    return DictDefault(base)


class TestGeneratePretrainingDatasetHash:
    def test_same_inputs_same_hash(self):
        cfg = _mm_pretrain_cfg()
        h1 = generate_pretraining_dataset_hash(
            cfg, _mm_pretrain_entry(), "Qwen/Qwen3.5-VL-9B-Instruct", "Qwen2VLProcessor"
        )
        h2 = generate_pretraining_dataset_hash(
            cfg, _mm_pretrain_entry(), "Qwen/Qwen3.5-VL-9B-Instruct", "Qwen2VLProcessor"
        )
        assert h1 == h2

    def test_different_data_files_different_hash(self):
        cfg = _mm_pretrain_cfg()
        h1 = generate_pretraining_dataset_hash(
            cfg,
            _mm_pretrain_entry(data_files="/data/a.jsonl"),
            "tok",
            "Proc",
        )
        h2 = generate_pretraining_dataset_hash(
            cfg,
            _mm_pretrain_entry(data_files="/data/b.jsonl"),
            "tok",
            "Proc",
        )
        assert h1 != h2

    def test_different_processor_different_hash(self):
        cfg = _mm_pretrain_cfg()
        h1 = generate_pretraining_dataset_hash(cfg, _mm_pretrain_entry(), "tok", "A")
        h2 = generate_pretraining_dataset_hash(cfg, _mm_pretrain_entry(), "tok", "B")
        assert h1 != h2

    def test_different_sequence_len_different_hash(self):
        h1 = generate_pretraining_dataset_hash(
            _mm_pretrain_cfg(sequence_len=2048), _mm_pretrain_entry(), "tok", "Proc"
        )
        h2 = generate_pretraining_dataset_hash(
            _mm_pretrain_cfg(sequence_len=4096), _mm_pretrain_entry(), "tok", "Proc"
        )
        assert h1 != h2

    def test_different_image_token_different_hash(self):
        cfg = _mm_pretrain_cfg()
        h1 = generate_pretraining_dataset_hash(
            cfg, _mm_pretrain_entry(image_token="<image>"), "tok", "Proc"
        )
        h2 = generate_pretraining_dataset_hash(
            cfg, _mm_pretrain_entry(image_token="<|image_pad|>"), "tok", "Proc"
        )
        assert h1 != h2

    def test_none_processor_folds_to_none_string(self):
        cfg = _mm_pretrain_cfg()
        h_none = generate_pretraining_dataset_hash(
            cfg, _mm_pretrain_entry(), "tok", None
        )
        h_named = generate_pretraining_dataset_hash(
            cfg, _mm_pretrain_entry(), "tok", "None"
        )
        assert h_none == h_named

    def test_eval_sequence_len_excluded(self):
        # Cache stores train only; eval-only knobs must not bust it.
        h1 = generate_pretraining_dataset_hash(
            _mm_pretrain_cfg(eval_sequence_len=None),
            _mm_pretrain_entry(),
            "tok",
            "Proc",
        )
        h2 = generate_pretraining_dataset_hash(
            _mm_pretrain_cfg(eval_sequence_len=4096),
            _mm_pretrain_entry(),
            "tok",
            "Proc",
        )
        assert h1 == h2

    def test_image_base_dir_excluded(self):
        # image_base_dir is collator-runtime, not baked into cached arrows.
        cfg = _mm_pretrain_cfg()
        h1 = generate_pretraining_dataset_hash(
            cfg, _mm_pretrain_entry(image_base_dir="/data/a"), "tok", "Proc"
        )
        h2 = generate_pretraining_dataset_hash(
            cfg, _mm_pretrain_entry(image_base_dir="/data/b"), "tok", "Proc"
        )
        assert h1 == h2
