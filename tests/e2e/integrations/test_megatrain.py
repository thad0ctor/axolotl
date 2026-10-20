"""End-to-end coverage for the MegaTrain integration."""

import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    PreTrainedTokenizerFast,
)

from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets
from axolotl.integrations.megatrain.trainer import MegaTrainAdamW, MegaTrainTrainer
from axolotl.train import train
from axolotl.utils.dict import DictDefault

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="MegaTrain requires a CUDA GPU with BF16 support",
)


def _build_tiny_checkpoint(model_dir: Path) -> tuple[str, torch.Tensor]:
    vocab_tokens = [
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "Below",
        "is",
        "an",
        "instruction",
        "that",
        "describes",
        "a",
        "task",
        "Write",
        "response",
        "completes",
        "request",
        "Input",
        "Output",
        "repeat",
        "number",
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "answer",
        "value",
        "please",
        "the",
        "following",
        "exactly",
        ".",
        ":",
        "#",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    vocab = {token: index for index, token in enumerate(vocab_tokens)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_max_length=64,
    )

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(17)
    model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16)
    weight_name = next(
        name for name in model.state_dict() if name.endswith("self_attn.q_proj.weight")
    )
    initial_weight = model.state_dict()[weight_name].detach().clone()

    model_dir.mkdir()
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    return weight_name, initial_weight


def _write_alpaca_dataset(dataset_path: Path) -> None:
    rows = [
        {
            "instruction": f"repeat number {index % 10}",
            "input": "please answer exactly",
            "output": f"answer value {index % 10}",
        }
        for index in range(24)
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_megatrain_trains_resumes_and_saves_vanilla_checkpoint(temp_dir, monkeypatch):
    work_dir = Path(temp_dir)
    model_dir = work_dir / "tiny-llama"
    output_dir = work_dir / "output"
    dataset_path = work_dir / "train.jsonl"
    weight_name, initial_weight = _build_tiny_checkpoint(model_dir)
    _write_alpaca_dataset(dataset_path)

    raw_config = {
        "base_model": str(model_dir),
        "plugins": [
            "axolotl.integrations.megatrain.MegaTrainPlugin",
        ],
        "tokenizer_type": "AutoTokenizer",
        "attn_implementation": "eager",
        "sequence_len": 64,
        "sample_packing": False,
        "val_set_size": 0,
        "datasets": [
            {
                "path": str(dataset_path),
                "ds_type": "json",
                "type": "alpaca",
            }
        ],
        "dataset_prepared_path": str(work_dir / "prepared"),
        "dataset_num_proc": 1,
        "num_epochs": 1,
        "max_steps": 20,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "optimizer": "adamw_torch",
        "lr_scheduler": "linear",
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "gradient_checkpointing": False,
        "output_dir": str(output_dir),
        "save_steps": 20,
        "save_total_limit": 1,
        "logging_steps": 1,
        "use_tensorboard": False,
        "seed": 42,
    }
    cfg = load_cfg(DictDefault(deepcopy(raw_config)))
    dataset_meta = load_datasets(cfg=cfg)

    trained_model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

    assert type(trainer) is MegaTrainTrainer
    assert trainer.state.global_step == 20
    losses = [
        float(record["loss"])
        for record in trainer.state.log_history
        if "loss" in record
    ]
    learning_rates = [
        float(record["learning_rate"])
        for record in trainer.state.log_history
        if "learning_rate" in record
    ]
    memory_records = [
        float(record["memory/max_active (GiB)"])
        for record in trainer.state.log_history
        if "memory/max_active (GiB)" in record
    ]
    assert memory_records, "expected memory metrics in the trainer log history"
    peak_active_memory = max(memory_records)
    assert len(losses) == 20
    assert all(math.isfinite(loss) for loss in losses)
    assert mean(losses[10:]) < mean(losses[:10])
    assert len(set(learning_rates)) > 1
    assert peak_active_memory < 0.25
    assert all(
        parameter.device.type == "cpu" for parameter in trained_model.parameters()
    )
    assert all(
        parameter.dtype == torch.float32
        for parameter in trained_model.parameters()
        if parameter.is_floating_point()
    )
    assert all(
        parameter.device.type == "cpu"
        for parameter in trainer.cpu_master.get_parameters()
    )
    assert all(
        flat.dtype == torch.bfloat16 for flat in trainer.cpu_master.layer_pinned_flats
    )
    assert all(
        state[name].dtype == torch.float32
        for state in trainer.optimizer.state.values()
        for name in ("exp_avg", "exp_avg_sq")
    )

    final_model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        local_files_only=True,
    )
    assert all(
        parameter.dtype == torch.float32
        for parameter in final_model.parameters()
        if parameter.is_floating_point()
    )
    assert not torch.equal(
        initial_weight.float(), final_model.state_dict()[weight_name].float()
    )

    checkpoint_dir = output_dir / "checkpoint-20"
    assert (checkpoint_dir / "optimizer.pt").is_file()
    assert (checkpoint_dir / "scheduler.pt").is_file()
    reloaded = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )
    assert all(
        parameter.dtype == torch.float32
        for parameter in reloaded.parameters()
        if parameter.is_floating_point()
    )
    assert not torch.equal(
        initial_weight.float(), reloaded.state_dict()[weight_name].float()
    )

    resume_config = deepcopy(raw_config)
    resume_config.update(
        {
            "max_steps": 22,
            "resume_from_checkpoint": str(checkpoint_dir),
            "save_steps": 22,
        }
    )
    resume_cfg = load_cfg(DictDefault(resume_config))
    resume_dataset_meta = load_datasets(cfg=resume_cfg)
    optimizer_steps = 0
    optimizer_step = MegaTrainAdamW.step

    def count_optimizer_step(self, closure=None):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return optimizer_step(self, closure=closure)

    monkeypatch.setattr(MegaTrainAdamW, "step", count_optimizer_step)
    resumed_model, tokenizer, resumed_trainer = train(
        cfg=resume_cfg, dataset_meta=resume_dataset_meta
    )

    assert type(resumed_trainer) is MegaTrainTrainer
    assert resumed_trainer.state.global_step == 22
    assert optimizer_steps == 2
    assert resumed_trainer.lr_scheduler.last_epoch == 22
    assert all(
        parameter.device.type == "cpu" for parameter in resumed_model.parameters()
    )
    assert all(
        parameter.dtype == torch.float32
        for parameter in resumed_model.parameters()
        if parameter.is_floating_point()
    )
    assert all(
        state[name].dtype == torch.float32
        for state in resumed_trainer.optimizer.state.values()
        for name in ("exp_avg", "exp_avg_sq")
    )
    assert all(
        float(state["step"].item()) == 22
        for state in resumed_trainer.optimizer.state.values()
    )

    resumed_checkpoint_dir = output_dir / "checkpoint-22"
    assert (resumed_checkpoint_dir / "optimizer.pt").is_file()
    assert (resumed_checkpoint_dir / "scheduler.pt").is_file()
    resumed = AutoModelForCausalLM.from_pretrained(
        resumed_checkpoint_dir,
        local_files_only=True,
    )
    assert all(
        parameter.dtype == torch.float32
        for parameter in resumed.parameters()
        if parameter.is_floating_point()
    )
    assert not torch.equal(
        reloaded.state_dict()[weight_name].float(),
        resumed.state_dict()[weight_name].float(),
    )

    prompt = tokenizer("repeat number one", return_tensors="pt")
    with torch.no_grad():
        generated = resumed.generate(**prompt, max_new_tokens=2)
    assert generated.shape[1] > prompt["input_ids"].shape[1]
