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
    from axolotl.integrations.sparselora._vendor.sparselora.modules.base import (
        SparseModule,
    )

    orig_lora_forward = lora_layer.Linear.forward
    orig_trainer_init = transformers.Trainer.__init__
    orig_patched_flag = api._trainer_init_patched
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


@pytest.fixture
def tiny_lora_model():
    """Tiny GQA Llama with attention-only LoRA (the SparseLoRA recipe)."""
    torch.manual_seed(0)
    model = get_peft_model(
        _tiny_llama(),
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.eval()
    return model


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
