"""Shared execution helpers for the run-based gates (G3..G7).

These wrap the *real* axolotl entry points so each gate drives the same path a
user's `axolotl preprocess` / `axolotl train` would, against the harness-owned
isolated ``output_dir``:

    load_cfg(path)                       -> resolved DictDefault   (resolve)
    load_datasets(cfg=…, cli_args=…)     -> TrainDatasetMeta       (prepare; G3/G4/G5)
    do_preprocess(cfg, cli_args)         -> prepared data on disk  (G3 plumbing)
    train(cfg, dataset_meta)             -> (model, tok, trainer)  (G6/G7)

The prepared ``train_dataset`` carries ``input_ids`` / ``labels`` /
``attention_mask`` columns, so G4/G5 read it in memory rather than guessing the
on-disk hash path.

Everything writes only under ``ctx.output_dir``. ``IGNORE_TOKEN_ID`` is re-exported
so masking gates do not re-hardcode -100.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

IGNORE_TOKEN_ID = -100

# A tiny instruct fixture (chat-template masking needs role structure that a raw
# completion corpus lacks). Written locally so gates need no network dataset.
_TINY_CHAT_ROWS = [
    {
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Name a primary color."},
            {"role": "assistant", "content": "Blue."},
            {"role": "user", "content": "And another?"},
            {"role": "assistant", "content": "Red."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello there!"},
        ]
    },
]


def write_chat_fixture(output_dir: Path, n_repeat: int = 16) -> Path:
    """Write a tiny chat_template JSONL fixture and return its path."""
    path = output_dir / "tiny_chat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (_TINY_CHAT_ROWS * n_repeat)[: max(n_repeat, len(_TINY_CHAT_ROWS))]
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")
    return path


def chat_dataset_stanza(fixture: Path) -> dict[str, Any]:
    """A `datasets:` entry for the local chat fixture (chat_template strategy).

    axolotl routes a `path` that exists on disk through its local-file loader
    (``_load_from_local_path``), inferring the json loader from the extension, so
    no network/hub access is needed.
    """
    return {
        "path": str(fixture),
        "type": "chat_template",
        "field_messages": "messages",
    }


def base_cfg(
    base_model: str,
    output_dir: Path,
    name: str,
    datasets: list[dict[str, Any]] | None = None,
    sequence_len: int = 256,
) -> dict[str, Any]:
    """Minimal valid training cfg; gates merge gate-specific flags on top."""
    return {
        "base_model": base_model,
        "output_dir": str(output_dir / name),
        "datasets": datasets
        or [{"path": "winglian/tiny-shakespeare", "type": "completion"}],
        "sequence_len": sequence_len,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "num_epochs": 1,
        "val_set_size": 0,
        "special_tokens": {},
        "dataset_prepared_path": str(output_dir / name / "prepared"),
    }


def write_cfg(output_dir: Path, cfg_dict: dict[str, Any], name: str) -> Path:
    path = output_dir / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    return path


def resolve_cfg(cfg_path: Path):
    """load_cfg the written YAML through the real pipeline."""
    from axolotl.cli.config import load_cfg

    return load_cfg(str(cfg_path))


def prepare(cfg, debug: bool = False):
    """Tokenize/prepare datasets -> TrainDatasetMeta (in-memory train_dataset)."""
    from axolotl.cli.args import PreprocessCliArgs
    from axolotl.common.datasets import load_datasets

    return load_datasets(cfg=cfg, cli_args=PreprocessCliArgs(debug=debug), debug=debug)


def preprocess_to_disk(cfg) -> Path:
    """Run the preprocess entry point; return the dataset_prepared_path dir."""
    from axolotl.cli.args import PreprocessCliArgs
    from axolotl.cli.preprocess import do_preprocess

    do_preprocess(cfg, PreprocessCliArgs())
    return Path(cfg.dataset_prepared_path)


def train_model(cfg, dataset_meta):
    """Drive a (short) training run; returns (model, tokenizer, trainer)."""
    from axolotl.train import train

    return train(cfg, dataset_meta)


def loss_history(trainer) -> list[float]:
    """Per-step training losses from the trainer state log history."""
    out: list[float] = []
    for rec in getattr(trainer.state, "log_history", []) or []:
        if "loss" in rec:
            out.append(float(rec["loss"]))
    return out
