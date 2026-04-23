"""End-to-end tests for the ProTrain Axolotl plugin glue (M5).

Two tests live here:

* ``test_plugin_e2e_tiny_llama`` — runs the full Axolotl
  config-validate → load-datasets → train path on a small SmolLM2-135M
  model with ``protrain_auto_memory: true`` +
  ``protrain_force_all_persistent: true``. Asserts no OOM / no crash,
  a decreasing loss trend, and that a checkpoint was written. Marked
  ``slow`` + ``gpu`` — it needs one free CUDA device.

* ``test_plugin_e2e_7b_lora_smoke`` — wires the real
  ``examples/protrain/3090-7b-lora.yml`` for manual validation.
  Marked ``skip`` so CI does not need the 7B weight download.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _marker(stage: str) -> None:
    """Print a progress marker that survives pytest's output buffering."""
    import sys

    sys.stderr.write(f"[protrain-e2e] {stage}\n")
    sys.stderr.flush()


@pytest.mark.slow
@pytest.mark.gpu
def test_plugin_e2e_tiny_llama(tmp_path: Path) -> None:
    """Run the full Axolotl training path with the ProTrain plugin on.

    Uses ``HuggingFaceTB/SmolLM2-135M`` — a small Llama-architecture
    model that lives in the HF hub's open set. The plugin's
    ``force_all_persistent`` path keeps all chunks on GPU and wraps
    every block in CKPT; on a 24 GB card this is a no-offload stress
    test of the plugin shim rather than the runtime primitives, but it
    exercises every hook (``get_input_args``, ``post_model_load``,
    ``create_optimizer``, ``post_trainer_create``) on a real
    HuggingFace Trainer.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain plugin E2E requires CUDA.")

    # Fresh PluginManager for the test so we don't collide with any
    # plugins a previous test left registered (PluginManager is a
    # module-level singleton).
    from axolotl.integrations.base import PluginManager

    PluginManager._instance = None  # type: ignore[attr-defined]

    output_dir = tmp_path / "protrain-tiny-out"

    # Build a minimal cfg dict — same shape the CLI would load from YAML,
    # but constructed in Python so we can point output_dir at tmp_path.
    # SmolLM2-135M is an existing Axolotl-test-friendly target
    # (see tests/e2e/test_llama_pretrain.py) with a Llama arch.
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            "load_in_8bit": False,
            "load_in_4bit": False,
            "strict": False,
            "datasets": [
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                }
            ],
            "val_set_size": 0.0,
            "output_dir": str(output_dir),
            "sequence_len": 128,
            "sample_packing": False,
            "pad_to_sequence_len": False,
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "v_proj"],
            "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
            "protrain_auto_memory": True,
            "protrain_force_all_persistent": True,
            "gradient_accumulation_steps": 1,
            "micro_batch_size": 1,
            "max_steps": 10,
            "optimizer": "adamw_torch",
            "lr_scheduler": "constant",
            "learning_rate": 0.0005,
            "bf16": "auto",
            "tf32": False,
            "gradient_checkpointing": False,
            "flash_attention": False,
            "logging_steps": 1,
            "save_steps": 10,
            "save_first_step": False,
            "save_total_limit": 1,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "dataset_num_proc": 1,
            "use_tensorboard": True,
            "special_tokens": {
                "pad_token": "<|endoftext|>",
            },
        }
    )

    _marker("cfg built; registering plugin via prepare_plugins")

    # Mirror what do_train does pre-validate: register plugins so their
    # args schemas get merged into validate_config.
    from axolotl.utils.config import normalize_config, prepare_plugins, validate_config

    prepare_plugins(cfg)

    _marker("calling validate_config")
    cfg = validate_config(cfg)

    _marker("calling normalize_config")
    normalize_config(cfg)

    # Ensure PluginManager.cfg is set — normally done by do_cli path.
    PluginManager.get_instance().cfg = cfg

    _marker("loading datasets")
    from axolotl.common.datasets import load_datasets

    from axolotl.cli.args import TrainerCliArgs

    cli_args = TrainerCliArgs()
    dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    _marker("entering axolotl.train.train")
    from axolotl.train import train

    _model, _tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
    _marker("train() returned")

    # Grab losses off trainer.state.log_history. The HF Trainer logs
    # train/loss for every `logging_steps` entry; we asked for 1.
    losses: list[float] = [
        float(rec["loss"])
        for rec in trainer.state.log_history
        if "loss" in rec
    ]
    assert len(losses) >= 2, (
        f"expected at least 2 training-loss log entries, got {losses}"
    )

    # Decreasing-trend check. Loss over 10 LoRA steps on a 135M model is
    # noisy step-to-step, so compare the mean of the last third to the
    # mean of the first third — that averages out single-batch spikes
    # while still catching a wiring bug that bypasses the optimizer.
    third = max(1, len(losses) // 3)
    first_third_mean = sum(losses[:third]) / third
    last_third_mean = sum(losses[-third:]) / third
    _marker(
        f"loss: first_third_mean={first_third_mean:.4f} "
        f"last_third_mean={last_third_mean:.4f} "
        f"losses={losses}"
    )
    assert last_third_mean < first_third_mean, (
        f"loss did not decrease: first_third_mean={first_third_mean:.4f} "
        f"last_third_mean={last_third_mean:.4f} losses={losses}"
    )

    # Checkpoint directory check — adapter safetensors for LoRA runs.
    adapter_file = Path(cfg.output_dir) / "adapter_model.safetensors"
    assert adapter_file.exists(), (
        f"expected adapter checkpoint at {adapter_file}, not found. "
        f"Output dir contents: {list(Path(cfg.output_dir).iterdir())}"
    )


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skip(
    reason=(
        "Real 7B weight download requires internet + HuggingFace cache "
        "(Mistral-7B-v0.3 is ~14 GB). Kept as documentation of the intended "
        "axolotl-train invocation; run manually with "
        "`pytest tests/protrain/test_plugin_e2e.py::test_plugin_e2e_7b_lora_smoke "
        "--runslow -s` after prefetching weights."
    )
)
def test_plugin_e2e_7b_lora_smoke(tmp_path: Path) -> None:
    """Smoke-test the real 3090-7b-lora.yml example.

    Equivalent to the CLI invocation::

        axolotl train examples/protrain/3090-7b-lora.yml --max-steps 4

    with ``output_dir`` rerouted to a pytest tmp_path. Intentionally
    skipped in CI; unlocking this test is the manual-validation step
    once M4.5 lands.
    """
    pytest.importorskip("torch")

    from axolotl.cli.config import load_cfg
    from axolotl.cli.args import TrainerCliArgs
    from axolotl.cli.train import do_train

    yaml_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "protrain"
        / "3090-7b-lora.yml"
    )
    assert yaml_path.exists(), f"missing example yaml at {yaml_path}"

    # Load config; override output_dir + max_steps for a smoke run.
    cfg = load_cfg(
        yaml_path,
        output_dir=str(tmp_path / "protrain-7b-smoke-out"),
        max_steps=4,
    )
    cli_args = TrainerCliArgs()
    do_train(cfg, cli_args)
