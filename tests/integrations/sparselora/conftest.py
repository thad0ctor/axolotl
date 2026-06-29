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

    from axolotl.integrations.sparselora import fast_tokens
    from axolotl.integrations.sparselora._vendor.sparselora import api
    from axolotl.integrations.sparselora._vendor.sparselora.modules import registry
    from axolotl.integrations.sparselora._vendor.sparselora.modules.base import (
        SparseModule,
    )

    orig_lora_forward = lora_layer.Linear.forward
    orig_trainer_init = transformers.Trainer.__init__
    orig_patched_flag = api._trainer_init_patched
    # register_arch_wiring mutates the process-global module registry.
    orig_registry = dict(registry._MODULE_REGISTRY)
    # The fast-tokens fast path rebinds these process-global hooks; snapshot so a
    # test that installs it (directly or via the plugin) can't leak into others.
    orig_mask = api._compute_output_token_mask
    orig_split = SparseModule.__dict__["token_splits"]
    orig_join = SparseModule.__dict__["token_join"]
    orig_fast_flag = fast_tokens._INSTALLED
    try:
        yield
    finally:
        lora_layer.Linear.forward = orig_lora_forward
        transformers.Trainer.__init__ = orig_trainer_init
        api._trainer_init_patched = orig_patched_flag
        registry._MODULE_REGISTRY.clear()
        registry._MODULE_REGISTRY.update(orig_registry)
        api._compute_output_token_mask = orig_mask
        SparseModule.token_splits = orig_split
        SparseModule.token_join = orig_join
        fast_tokens._INSTALLED = orig_fast_flag


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
            head_dim=32,  # decoupled: q_out=128 != hidden=64 (like real Qwen3)
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


def _tiny_gemma2(num_kv_heads: int = 2):
    from transformers import Gemma2Config, Gemma2ForCausalLM

    model = Gemma2ForCausalLM(
        Gemma2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=num_kv_heads,
            head_dim=16,
            max_position_embeddings=64,
            attn_logit_softcapping=30.0,
        )
    )
    model.config._attn_implementation = "eager"  # softcap path needs eager
    return model


def _tiny_gemma3(num_kv_heads: int = 2):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    return Gemma3ForCausalLM(
        Gemma3TextConfig(
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


def _tiny_phi3(num_kv_heads: int = 2):
    from transformers import Phi3Config, Phi3ForCausalLM

    return Phi3ForCausalLM(
        Phi3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
        )
    )


# Architecture registry for the parametrized cross-arch tests.
TINY_BUILDERS = {
    "llama": _tiny_llama,
    "qwen2": _tiny_qwen2,
    "qwen3": _tiny_qwen3,
    "mistral": _tiny_mistral,
    "gemma2": _tiny_gemma2,
    "gemma3": _tiny_gemma3,
    "phi3": _tiny_phi3,
}

# Phi3 fuses q/k/v into qkv_proj and gate/up into gate_up_proj, so its
# attention LoRA target is qkv_proj (+ o_proj), not q/k/v_proj.
_LORA_TARGETS = {
    "phi3": ["qkv_proj", "o_proj"],
}


def make_lora_model(arch: str, num_kv_heads: int = 2):
    """Tiny model of ``arch`` with attention-only LoRA (the SparseLoRA recipe)."""
    # fork_rng so seeding for reproducible weights does not mutate the global RNG
    # and couple later tests to call order.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = get_peft_model(
            TINY_BUILDERS[arch](num_kv_heads),
            LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=_LORA_TARGETS.get(
                    arch, ["q_proj", "k_proj", "v_proj", "o_proj"]
                ),
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
    # Local generator so the fixture doesn't perturb the process-wide RNG.
    gen = torch.Generator().manual_seed(1)

    def make(n=2, t=16):
        ids = torch.randint(0, 128, (n, t), generator=gen)
        labels = ids.clone()
        labels[:, : t // 2] = -100
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": torch.ones(n, t, dtype=torch.long),
        }

    return [make() for _ in range(3)]
