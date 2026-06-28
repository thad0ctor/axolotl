"""Wrap the real axolotl preprocess/train entry points for the run-based gates (G3..G7); everything writes only under ``ctx.output_dir``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

IGNORE_TOKEN_ID = -100

# tiny instruct fixture; chat-template masking needs role structure, written locally (offline)
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
    path = output_dir / "tiny_chat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (_TINY_CHAT_ROWS * n_repeat)[: max(n_repeat, len(_TINY_CHAT_ROWS))]
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")
    return path


def chat_dataset_stanza(fixture: Path) -> dict[str, Any]:
    """A `datasets:` entry routed through axolotl's on-disk local-file loader (offline)."""
    return {
        "path": str(fixture),
        "type": "chat_template",
        "field_messages": "messages",
    }


_TINY_COMPLETION_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
)


def write_completion_fixture(output_dir: Path, n_rows: int = 64) -> Path:
    path = output_dir / "tiny_completion.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for i in range(n_rows):
            fout.write(json.dumps({"text": _TINY_COMPLETION_TEXT * (1 + i % 3)}) + "\n")
    return path


def completion_dataset_stanza(fixture: Path) -> dict[str, Any]:
    return {"path": str(fixture), "type": "completion", "field": "text"}


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
    from axolotl.cli.config import load_cfg

    return load_cfg(str(cfg_path))


def prepare(cfg, debug: bool = False):
    """Tokenize/prepare datasets -> TrainDatasetMeta (in-memory train_dataset)."""
    from axolotl.cli.args import PreprocessCliArgs
    from axolotl.common.datasets import load_datasets

    return load_datasets(cfg=cfg, cli_args=PreprocessCliArgs(debug=debug), debug=debug)


def preprocess_to_disk(cfg) -> Path:
    """Run the preprocess entry point; return the dataset_prepared_path dir.

    The AXOLOTL_IS_PREPROCESS env + cfg.is_preprocess flag are set then RESTORED, else preprocess mode leaks into later gates and a subsequent prepare() skips loading the saved artifact."""
    import os

    from axolotl.cli.args import PreprocessCliArgs
    from axolotl.cli.preprocess import do_preprocess

    prev_env = os.environ.get("AXOLOTL_IS_PREPROCESS")
    prev_flag = getattr(cfg, "is_preprocess", None)
    os.environ["AXOLOTL_IS_PREPROCESS"] = "1"
    try:
        cfg.is_preprocess = True
    except Exception:  # noqa: BLE001 - DictDefault accepts attr set; be defensive
        pass
    try:
        do_preprocess(cfg, PreprocessCliArgs())
    finally:
        if prev_env is None:
            os.environ.pop("AXOLOTL_IS_PREPROCESS", None)
        else:
            os.environ["AXOLOTL_IS_PREPROCESS"] = prev_env
        try:
            cfg.is_preprocess = prev_flag
        except Exception:  # noqa: BLE001
            pass
    return Path(cfg.dataset_prepared_path)


def has_saved_dataset(prepared_path: Path) -> bool:
    """True if the preprocess STEP wrote a loadable saved HF dataset (not silently reprocessed)."""
    if not prepared_path.exists():
        return False
    markers = ("dataset_info.json", "state.json")
    for p in prepared_path.rglob("*"):
        if p.name in markers or p.suffix == ".arrow":
            return True
    return False


def train_model(cfg, dataset_meta):
    """Returns (model, tokenizer, trainer)."""
    from axolotl.train import train

    return train(cfg, dataset_meta)


def loss_history(trainer) -> list[float]:
    """Per-step training losses from the trainer state log history."""
    out: list[float] = []
    for rec in getattr(trainer.state, "log_history", []) or []:
        if "loss" in rec:
            out.append(float(rec["loss"]))
    return out
