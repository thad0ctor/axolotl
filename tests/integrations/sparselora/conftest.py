"""Shared fixtures for SparseLoRA tests."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture(autouse=True)
def _restore_global_sparselora_patches():
    """apply_sparselora monkeypatches process-global hooks; isolate them per test."""
    import peft.tuners.lora.layer as lora_layer
    import transformers

    from axolotl.integrations.sparselora._vendor.sparselora import api
    from axolotl.integrations.sparselora._vendor.sparselora.modules import registry

    orig_lora_forward = lora_layer.Linear.forward
    orig_trainer_init = transformers.Trainer.__init__
    orig_patched_flag = api._trainer_init_patched
    # register_arch_wiring mutates the process-global module registry.
    orig_registry = dict(registry._MODULE_REGISTRY)
    try:
        yield
    finally:
        lora_layer.Linear.forward = orig_lora_forward
        transformers.Trainer.__init__ = orig_trainer_init
        api._trainer_init_patched = orig_patched_flag
        registry._MODULE_REGISTRY.clear()
        registry._MODULE_REGISTRY.update(orig_registry)


def _tiny_llama(num_kv_heads: int = 2):
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


def _tiny_qwen2(num_kv_heads: int = 2):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=64,
        )
    )


def _tiny_qwen3(num_kv_heads: int = 2):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=num_kv_heads,
            head_dim=16,
            max_position_embeddings=64,
        )
    )


def _tiny_mistral(num_kv_heads: int = 2):
    from transformers import MistralConfig, MistralForCausalLM

    return MistralForCausalLM(
        MistralConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=64,
            sliding_window=32,
        )
    )


# Architecture registry for the parametrized cross-arch tests.
TINY_BUILDERS = {
    "llama": _tiny_llama,
    "qwen2": _tiny_qwen2,
    "qwen3": _tiny_qwen3,
    "mistral": _tiny_mistral,
}


def make_lora_model(arch: str, num_kv_heads: int = 2):
    """Tiny model of ``arch`` with attention-only LoRA (the SparseLoRA recipe)."""
    torch.manual_seed(0)
    model = get_peft_model(
        TINY_BUILDERS[arch](num_kv_heads),
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.eval()
    return model


@pytest.fixture
def tiny_lora_model():
    """Tiny GQA Llama with attention-only LoRA (the SparseLoRA recipe)."""
    return make_lora_model("llama")


@pytest.fixture
def calib_batches():
    torch.manual_seed(1)

    def make(n=2, t=16):
        ids = torch.randint(0, 128, (n, t))
        labels = ids.clone()
        labels[:, : t // 2] = -100
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": torch.ones(n, t, dtype=torch.long),
        }

    return [make() for _ in range(3)]
