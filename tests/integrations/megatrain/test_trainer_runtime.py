"""CPU-only runtime contract tests for the MegaTrain trainer bridge."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.integrations.base import PluginManager
from axolotl.integrations.megatrain import MegaTrainPlugin
from axolotl.integrations.megatrain.trainer import MegaTrainAdamW, MegaTrainTrainer
from axolotl.utils.dict import DictDefault


class _Model:
    def __init__(self):
        self.training = False

    def train(self):
        self.training = True


class _DifferentiableCPUMaster:
    def __init__(self, parameter):
        self.parameter = parameter
        self.denominators = []

    def forward_and_backward(
        self,
        input_ids,
        attention_mask,
        labels,
        *,
        global_valid_tokens,
    ):
        del input_ids, attention_mask
        targets = labels[..., 1:][labels[..., 1:].ne(-100)].float()
        per_token_loss = (self.parameter - targets).square()
        (per_token_loss.sum() / global_valid_tokens).backward()
        self.denominators.append(global_valid_tokens)
        return per_token_loss.mean().item(), labels.numel(), {"total": 0.0}


def _training_step_subject(cpu_master, accumulation_steps=1):
    return SimpleNamespace(
        _megatrain_closed=False,
        args=SimpleNamespace(device=torch.device("cpu")),
        cpu_master=cpu_master,
        current_gradient_accumulation_steps=accumulation_steps,
        optimizer=SimpleNamespace(),
    )


def _batch(*targets):
    labels = torch.tensor([[-100, *targets]], dtype=torch.long)
    return {
        "input_ids": torch.zeros_like(labels),
        "attention_mask": torch.ones_like(labels),
        "labels": labels,
    }


def test_training_step_accumulates_token_normalized_gradient():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    cpu_master = _DifferentiableCPUMaster(parameter)
    trainer = _training_step_subject(cpu_master, accumulation_steps=2)
    model = _Model()

    first_loss = MegaTrainTrainer.training_step(
        trainer, model, _batch(1, 3), num_items_in_batch=torch.tensor(3)
    )
    second_loss = MegaTrainTrainer.training_step(
        trainer, model, _batch(5), num_items_in_batch=torch.tensor(3)
    )

    expected_loss = torch.tensor(((2 - 1) ** 2 + (2 - 3) ** 2 + (2 - 5) ** 2) / 3)
    expected_gradient = torch.tensor(2 * ((2 - 1) + (2 - 3) + (2 - 5)) / 3)
    assert model.training
    assert cpu_master.denominators == [3, 3]
    assert first_loss + second_loss == pytest.approx(expected_loss)
    assert parameter.grad == pytest.approx(expected_gradient)


def test_training_step_fallback_scales_one_microbatch_by_accumulation_steps():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    cpu_master = _DifferentiableCPUMaster(parameter)
    trainer = _training_step_subject(cpu_master, accumulation_steps=4)

    loss = MegaTrainTrainer.training_step(
        trainer, _Model(), _batch(1, 3), num_items_in_batch=None
    )

    assert cpu_master.denominators == [8]
    assert loss == pytest.approx(torch.tensor(0.5))
    assert parameter.grad == pytest.approx(torch.tensor(-0.5))


def test_training_step_skips_batch_without_valid_labels():
    cpu_master = SimpleNamespace(forward_and_backward=Mock())
    trainer = _training_step_subject(cpu_master)
    labels = torch.full((1, 3), -100, dtype=torch.long)

    loss = MegaTrainTrainer.training_step(
        trainer,
        _Model(),
        {"input_ids": torch.zeros_like(labels), "labels": labels},
        num_items_in_batch=0,
    )

    assert loss.item() == 0.0
    cpu_master.forward_and_backward.assert_not_called()


def test_training_step_rejects_noncanonical_position_ids():
    cpu_master = SimpleNamespace(forward_and_backward=Mock())
    trainer = _training_step_subject(cpu_master)
    batch = _batch(1, 3)
    batch["position_ids"] = torch.tensor([[0, 0, 1]])

    with pytest.raises(RuntimeError, match="custom `position_ids`"):
        MegaTrainTrainer.training_step(trainer, _Model(), batch)

    cpu_master.forward_and_backward.assert_not_called()


def test_num_items_always_uses_shifted_causal_labels():
    trainer = object.__new__(MegaTrainTrainer)
    trainer._loss_shifts_labels = False
    batches = [
        {"labels": torch.tensor([[-100, 1, -100]])},
        {"labels": torch.tensor([[5, 6, 7]])},
    ]

    count = trainer._get_num_items_in_batch(batches, torch.device("cpu"))

    assert count.item() == 3


def test_trainer_init_defers_cpu_master_allocation(monkeypatch):
    model = torch.nn.Linear(4, 4, bias=False)
    manager = PluginManager.get_instance()
    manager.cfg = DictDefault(
        {
            "base_model": "unused",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "sequence_len": 8,
            "learning_rate": 1e-4,
        }
    )

    def initialize_base(self, *args, **kwargs):
        del args, kwargs
        self.model = model

    monkeypatch.setattr(AxolotlTrainer, "__init__", initialize_base)
    cpu_master = Mock(side_effect=AssertionError("allocated during init"))
    monkeypatch.setattr(
        "axolotl.integrations.megatrain.trainer.CPUMasterModel", cpu_master
    )

    trainer = MegaTrainTrainer()

    assert not hasattr(trainer, "cpu_master")
    cpu_master.assert_not_called()


def test_optimizer_selection_ignores_stale_megatrain_alias():
    canonical_name = "axolotl.integrations.megatrain.MegaTrainPlugin"
    shorthand_name = "megatrain.MegaTrainPlugin"
    manager = PluginManager.get_instance()
    manager.plugins[canonical_name] = MegaTrainPlugin()
    manager.plugins[shorthand_name] = MegaTrainPlugin()
    manager.cfg = DictDefault(
        {
            "plugins": [shorthand_name],
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
        }
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    cpu_master = SimpleNamespace(
        get_parameters=lambda: [parameter],
        _sync_params_to_gpu=Mock(),
    )
    trainer = object.__new__(MegaTrainTrainer)
    trainer.optimizer = None
    trainer.cpu_master = cpu_master
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.model.weight = parameter
    trainer.get_decay_parameter_names = lambda _: {"weight"}

    optimizer = trainer.create_optimizer()

    assert isinstance(optimizer, MegaTrainAdamW)
    assert optimizer.cpu_master is cpu_master


def test_adamw_syncs_streamed_parameters_after_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    synced_values = []
    cpu_master = SimpleNamespace(
        _sync_params_to_gpu=lambda: synced_values.append(parameter.detach().clone())
    )
    optimizer = MegaTrainAdamW(
        [parameter], cpu_master=cpu_master, lr=0.1, weight_decay=0.0
    )
    parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    assert parameter.item() == pytest.approx(0.9)
    assert len(synced_values) == 1
    assert synced_values[0] == pytest.approx(parameter.detach())


def test_adamw_keeps_sub_bf16_update_and_state_in_fp32():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    cpu_master = SimpleNamespace(_sync_params_to_gpu=Mock())
    optimizer = MegaTrainAdamW(
        [parameter],
        cpu_master=cpu_master,
        lr=2e-4,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    parameter.grad = torch.ones_like(parameter)
    bf16_before = parameter.detach().to(dtype=torch.bfloat16)

    optimizer.step()

    assert parameter.item() != 1.0
    assert torch.equal(parameter.detach().to(dtype=torch.bfloat16), bf16_before)
    assert parameter.grad.dtype == torch.float32
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    cpu_master._sync_params_to_gpu.assert_called_once_with()


def test_checkpoint_load_syncs_after_model_restore(monkeypatch):
    events = []
    trainer = object.__new__(MegaTrainTrainer)
    trainer.cpu_master = SimpleNamespace(
        _sync_params_to_gpu=lambda: events.append("sync")
    )

    def load_checkpoint(self, path, model=None):
        del self, model
        events.append(("load", path))

    monkeypatch.setattr(AxolotlTrainer, "_load_from_checkpoint", load_checkpoint)

    trainer._load_from_checkpoint("checkpoint-4")

    assert events == [("load", "checkpoint-4"), "sync"]


def test_close_releases_runtime_once():
    events = []
    trainer = object.__new__(MegaTrainTrainer)
    trainer._megatrain_closed = False
    trainer.cpu_master = SimpleNamespace(
        release_gpu_buffers=lambda: events.append("release"),
        cleanup=lambda: events.append("cleanup"),
    )

    trainer.close()
    trainer.close()

    assert trainer._megatrain_closed
    assert events == ["release", "cleanup"]


def test_plugin_cleanup_hooks_close_registered_trainer_once():
    plugin = MegaTrainPlugin()
    trainer = object.__new__(MegaTrainTrainer)
    trainer.close = Mock()
    config = SimpleNamespace(plugins=["axolotl.integrations.megatrain.MegaTrainPlugin"])
    plugin.post_trainer_create(config, trainer)

    plugin.post_train(config, None)
    plugin.post_train_unload(config)

    trainer.close.assert_called_once_with()
