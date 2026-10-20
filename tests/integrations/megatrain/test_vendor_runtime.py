"""Runtime regression tests for MegaTrain's streamed training path."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.utils import is_flash_attn_2_available

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.integrations.megatrain._vendor.infinity.config.training import (
    CPUMasterConfig,
)
from axolotl.integrations.megatrain._vendor.infinity.model.cpu_master import (
    CPUMasterModel,
)
from axolotl.integrations.megatrain.trainer import MegaTrainTrainer


def _runtime_subject(cpu_master):
    trainer = object.__new__(MegaTrainTrainer)
    trainer._megatrain_closed = False
    trainer.cpu_master = cpu_master
    return trainer


def test_train_closes_runtime_when_base_trainer_fails(monkeypatch):
    trainer = _runtime_subject(SimpleNamespace())
    trainer._ensure_cpu_master = Mock(return_value=trainer.cpu_master)
    trainer.close = Mock()
    monkeypatch.setattr(
        AxolotlTrainer,
        "train",
        Mock(side_effect=RuntimeError("training failed")),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        MegaTrainTrainer.train(trainer)

    trainer.close.assert_called_once_with()


def test_close_cleans_up_when_gpu_release_fails():
    cleanup = Mock()
    trainer = _runtime_subject(
        SimpleNamespace(
            release_gpu_buffers=Mock(side_effect=RuntimeError("release failed")),
            cleanup=cleanup,
        )
    )

    with pytest.raises(RuntimeError, match="release failed"):
        trainer.close()

    assert trainer._megatrain_closed
    cleanup.assert_called_once_with()
    trainer.close()
    cleanup.assert_called_once_with()


def test_parameter_mapping_failure_releases_runtime(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    other_parameter = torch.nn.Parameter(torch.ones(1))
    cpu_master = SimpleNamespace(
        cpu_layers=[],
        get_parameters=lambda: [other_parameter],
        release_gpu_buffers=Mock(),
        cleanup=Mock(),
    )
    trainer = object.__new__(MegaTrainTrainer)
    trainer._megatrain_closed = False
    trainer.model = model
    trainer.megatrain_config = SimpleNamespace(device=0)
    monkeypatch.setattr(
        "axolotl.integrations.megatrain.trainer.CPUMasterModel",
        Mock(return_value=cpu_master),
    )

    with pytest.raises(RuntimeError, match="could not map every model parameter"):
        trainer._ensure_cpu_master()

    assert trainer._megatrain_closed
    cpu_master.release_gpu_buffers.assert_called_once_with()
    cpu_master.cleanup.assert_called_once_with()


def test_clip_grad_norm_operates_on_cpu_master_parameters():
    first = torch.nn.Parameter(torch.tensor([0.0]))
    second = torch.nn.Parameter(torch.tensor([0.0]))
    first.grad = torch.tensor([3.0])
    second.grad = torch.tensor([4.0])
    trainer = _runtime_subject(SimpleNamespace(get_parameters=lambda: [first, second]))
    trainer.args = SimpleNamespace(max_grad_norm=2.0)

    norm = trainer._clip_grad_norm(None)

    assert norm == pytest.approx(torch.tensor(5.0))
    assert torch.linalg.vector_norm(
        torch.stack([first.grad, second.grad])
    ) == pytest.approx(torch.tensor(2.0))


def test_get_grad_norm_reports_without_mutating_cpu_master_gradients():
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    trainer = _runtime_subject(SimpleNamespace(get_parameters=lambda: [parameter]))

    norm = trainer._get_grad_norm(None)

    assert norm == pytest.approx(torch.tensor(5.0))
    torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 4.0]))


def test_training_step_defaults_attention_mask_and_accepts_canonical_positions():
    cpu_master = SimpleNamespace(
        forward_and_backward=Mock(return_value=(1.25, 3, {"total": 0.0}))
    )
    trainer = _runtime_subject(cpu_master)
    trainer.args = SimpleNamespace(device=torch.device("cpu"))
    trainer.current_gradient_accumulation_steps = 1
    trainer.optimizer = SimpleNamespace()
    model = SimpleNamespace(train=Mock())
    input_ids = torch.tensor([[4, 5, 6]])
    labels = torch.tensor([[-100, 5, 6]])

    loss = trainer.training_step(
        model,
        {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": torch.arange(3).unsqueeze(0),
        },
    )

    assert loss == pytest.approx(torch.tensor(1.25))
    cpu_master.forward_and_backward.assert_called_once()
    call = cpu_master.forward_and_backward.call_args
    torch.testing.assert_close(call.args[1], torch.ones_like(input_ids))
    assert call.kwargs["global_valid_tokens"] == 2


def test_training_step_rejects_closed_runtime():
    trainer = _runtime_subject(SimpleNamespace())
    trainer._megatrain_closed = True

    with pytest.raises(RuntimeError, match="already been released"):
        trainer.training_step(None, {})


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA with BF16 support is required",
)
@pytest.mark.parametrize(
    "attention_implementation",
    [
        "eager",
        "sdpa",
        pytest.param(
            "flash_attention_2",
            marks=pytest.mark.skipif(
                not is_flash_attn_2_available(),
                reason="Flash Attention 2 is not available",
            ),
        ),
    ],
)
def test_padded_uneven_microbatches_match_stock_gradients(attention_implementation):
    device = torch.device("cuda", torch.cuda.current_device())
    config = LlamaConfig(
        vocab_size=67,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    config._attn_implementation = attention_implementation
    torch.manual_seed(41)
    streamed_model = LlamaForCausalLM(config)
    stock_model = LlamaForCausalLM(config)
    stock_model.load_state_dict(streamed_model.state_dict())
    with torch.no_grad():
        for parameter in stock_model.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype=torch.bfloat16)
    stock_model.to(device=device)

    input_ids = torch.randint(0, config.vocab_size, (2, 9))
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:, :2] = -100
    global_valid_tokens = int(labels[:, 1:].ne(-100).sum().item())

    stock_loss = stock_model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        labels=labels.to(device),
    ).loss
    stock_loss.backward()
    expected_gradients = {
        name: parameter.grad.detach().cpu().float()
        for name, parameter in stock_model.named_parameters()
    }

    master = CPUMasterModel(
        streamed_model,
        CPUMasterConfig(
            model_name="tiny-llama",
            device=device.index,
            devices=[device.index],
            dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
            dataset_path="__axolotl__",
            max_seq_len=input_ids.shape[1],
            batch_size=1,
            checkpoint_interval=2,
            num_grad_slabs=4,
        ),
    )
    try:
        microbatch_losses = []
        microbatch_tokens = []
        for index in range(input_ids.shape[0]):
            loss, _, _ = master.forward_and_backward(
                input_ids[index : index + 1],
                attention_mask[index : index + 1],
                labels[index : index + 1],
                global_valid_tokens=global_valid_tokens,
            )
            microbatch_losses.append(loss)
            microbatch_tokens.append(
                int(labels[index : index + 1, 1:].ne(-100).sum().item())
            )

        accumulated_loss = (
            sum(
                loss * tokens
                for loss, tokens in zip(
                    microbatch_losses, microbatch_tokens, strict=True
                )
            )
            / global_valid_tokens
        )
        actual_gradients = {
            name: parameter.grad.detach().float()
            for name, parameter in streamed_model.named_parameters()
        }

        assert accumulated_loss == pytest.approx(stock_loss.item(), abs=2e-4)
        assert actual_gradients.keys() == expected_gradients.keys()
        for name in actual_gradients:
            torch.testing.assert_close(
                actual_gradients[name],
                expected_gradients[name],
                rtol=2e-2,
                atol=5e-3,
                msg=lambda message, name=name: f"{name}: {message}",
            )
    finally:
        master.release_gpu_buffers()
        master.cleanup()
        del stock_model
        torch.cuda.empty_cache()
