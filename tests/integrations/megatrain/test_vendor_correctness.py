"""Focused correctness tests for the vendored CPU-master runtime."""

import pytest
import torch
from transformers import (
    Gemma2Config,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    PreTrainedConfig,
)

from axolotl.integrations.megatrain._vendor.infinity.config.training import (
    CPUMasterConfig,
)
from axolotl.integrations.megatrain._vendor.infinity.model.cpu_master import (
    CPUMasterModel,
    _accumulate_parameter_grads_from_slab,
    _copy_module_for_compute,
    _copy_parameter_grads_to_slab,
    _head_gradient_numel,
    _replay_cuda_rng_state,
)


def test_tied_lm_head_is_excluded_from_head_gradient_slab():
    embedding = torch.nn.Embedding(11, 5)
    lm_head = torch.nn.Linear(5, 11, bias=False)
    lm_head.weight = embedding.weight
    norm = torch.nn.LayerNorm(5)

    assert _head_gradient_numel(lm_head, norm, tied_lm_head=True) == sum(
        parameter.numel() for parameter in norm.parameters()
    )
    assert _head_gradient_numel(lm_head, norm, tied_lm_head=False) == sum(
        parameter.numel() for parameter in (*lm_head.parameters(), *norm.parameters())
    )


def test_gradient_slab_keeps_unused_parameter_offset_and_grad_none():
    unused_gpu = torch.nn.Parameter(torch.zeros(2))
    used_gpu = torch.nn.Parameter(torch.zeros(3))
    used_gpu.grad = torch.tensor([4.0, 5.0, 6.0])
    slab = torch.full((5,), 99.0)

    offset, has_grads = _copy_parameter_grads_to_slab([unused_gpu, used_gpu], slab)

    assert offset == 5
    assert has_grads == [False, True]
    assert slab.tolist() == [0.0, 0.0, 4.0, 5.0, 6.0]
    assert unused_gpu.grad is None
    assert used_gpu.grad is None

    unused_cpu = torch.nn.Parameter(torch.zeros(2))
    used_cpu = torch.nn.Parameter(torch.zeros(3))
    used_cpu.grad = torch.ones(3)
    _accumulate_parameter_grads_from_slab(
        [unused_cpu, used_cpu],
        [unused_cpu.shape, used_cpu.shape],
        [unused_cpu.numel(), used_cpu.numel()],
        has_grads,
        slab,
    )

    assert unused_cpu.grad is None
    assert used_cpu.grad.tolist() == [5.0, 6.0, 7.0]


def test_gradient_slab_promotes_working_gradient_to_master_dtype():
    parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    slab = torch.tensor([0.25, 0.5, 0.75], dtype=torch.bfloat16)

    _accumulate_parameter_grads_from_slab(
        [parameter], [parameter.shape], [parameter.numel()], [True], slab
    )

    assert parameter.grad.dtype == torch.float32
    torch.testing.assert_close(parameter.grad, slab.float())


def test_compute_copy_casts_parameters_without_casting_buffers():
    module = torch.nn.Linear(3, 2, bias=False)
    module.register_buffer("frequencies", torch.arange(3, dtype=torch.float32))

    working_module = _copy_module_for_compute(
        module, torch.device("cpu"), torch.bfloat16
    )

    assert module.weight.dtype == torch.float32
    assert working_module.weight.dtype == torch.bfloat16
    assert working_module.frequencies.dtype == torch.float32
    torch.testing.assert_close(working_module.frequencies, module.frequencies)


def test_mixed_attention_layer_schedule_is_rejected():
    config = Gemma2Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        sliding_window=2,
    )
    config._attn_implementation = "eager"
    master = object.__new__(CPUMasterModel)
    master._model_config = config
    hidden_states = torch.zeros(1, 5, config.hidden_size)
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    position_ids = torch.arange(5).unsqueeze(0)

    assert config.layer_types == ["sliding_attention", "full_attention"]
    with pytest.raises(ValueError, match="mixed attention"):
        master._prepare_attention_mask(attention_mask, hidden_states, position_ids)


@pytest.mark.parametrize("layer_type", ["linear_attention", "chunked_attention"])
def test_unsupported_stateful_attention_layer_is_rejected(layer_type):
    config = PreTrainedConfig()
    config.layer_types = [layer_type]
    config._attn_implementation = "eager"
    master = object.__new__(CPUMasterModel)
    master._model_config = config

    with pytest.raises(ValueError, match=layer_type):
        master._prepare_attention_mask(
            torch.ones(1, 3, dtype=torch.long),
            torch.zeros(1, 3, 4),
            torch.arange(3).unsqueeze(0),
        )


def test_rng_replay_restores_state_when_recompute_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: "current")
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda state, device: calls.append((state, device))
    )

    with pytest.raises(RuntimeError, match="recompute"):
        with _replay_cuda_rng_state(0, "forward"):
            raise RuntimeError("recompute failed")

    assert calls == [("forward", 0), ("current", 0)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rng_replay_reuses_dropout_mask_without_advancing_global_state():
    device = torch.cuda.current_device()
    original_state = torch.cuda.get_rng_state(device)
    try:
        torch.cuda.manual_seed(1234)
        forward_state = torch.cuda.get_rng_state(device)
        inputs = torch.ones(1024, device=device)
        expected = torch.nn.functional.dropout(inputs, p=0.5, training=True)
        post_forward_state = torch.cuda.get_rng_state(device)

        with _replay_cuda_rng_state(device, forward_state):
            recomputed = torch.nn.functional.dropout(inputs, p=0.5, training=True)

        assert torch.equal(recomputed, expected)
        assert torch.equal(torch.cuda.get_rng_state(device), post_forward_state)
    finally:
        torch.cuda.set_rng_state(original_state, device)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA with BF16 support is required",
)
@pytest.mark.parametrize(
    ("config", "model_cls", "sequence_length", "attention_implementation"),
    [
        (
            LlamaConfig(
                vocab_size=67,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=160,
                tie_word_embeddings=True,
            ),
            LlamaForCausalLM,
            132,
            "eager",
        ),
        (
            MistralConfig(
                vocab_size=67,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=32,
                sliding_window=2,
            ),
            MistralForCausalLM,
            7,
            "eager",
        ),
        (
            MistralConfig(
                vocab_size=67,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=32,
                sliding_window=2,
            ),
            MistralForCausalLM,
            7,
            "sdpa",
        ),
    ],
    ids=["llama-multichunk-eager", "mistral-sliding-eager", "mistral-sliding-sdpa"],
)
def test_fallback_loss_and_gradients_match_supported_models(
    config, model_cls, sequence_length, attention_implementation
):
    device = torch.device("cuda", torch.cuda.current_device())
    config._attn_implementation = attention_implementation
    torch.manual_seed(19)
    streamed_model = model_cls(config)
    stock_model = model_cls(config)
    stock_model.load_state_dict(streamed_model.state_dict())
    with torch.no_grad():
        for parameter in stock_model.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype=torch.bfloat16)
    stock_model.to(device=device)
    input_ids = torch.randint(0, config.vocab_size, (1, sequence_length))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :3] = -100

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
            model_name=f"tiny-{config.model_type}",
            device=device.index,
            devices=[device.index],
            dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
            dataset_path="__axolotl__",
            max_seq_len=sequence_length,
            batch_size=1,
            checkpoint_interval=1,
            num_grad_slabs=2,
        ),
    )
    try:
        master.gpu_contexts[0].ce_loss = None
        assert all(
            parameter.dtype == torch.float32
            for parameter in master.get_parameters()
            if parameter.is_floating_point()
        )
        working_modules = [
            master.gpu_contexts[0].emb_gpu,
            master.gpu_contexts[0].norm_gpu,
            master.gpu_contexts[0].lm_head_gpu,
            *(
                template
                for templates in master.gpu_contexts[0].gpu_layer_templates.values()
                for template in templates
            ),
        ]
        assert all(
            parameter.dtype == torch.bfloat16
            for module in working_modules
            if module is not None
            for parameter in module.parameters()
            if parameter.is_floating_point()
        )
        assert all(
            buffer.dtype == torch.float32
            for buffer in master.gpu_contexts[0].rotary_gpu.buffers()
            if buffer.is_floating_point()
        )
        actual_loss, _, _ = master.forward_and_backward(
            input_ids, attention_mask, labels
        )
        actual_gradients = {
            name: parameter.grad.detach().float()
            for name, parameter in streamed_model.named_parameters()
        }

        assert actual_loss == pytest.approx(stock_loss.item(), abs=2e-4)
        assert all(
            parameter.grad is None or parameter.grad.dtype == torch.float32
            for parameter in streamed_model.parameters()
        )
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
