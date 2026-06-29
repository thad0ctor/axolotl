"""CPU tests for the factor->loader contract and plugin validation.

These do not require CUDA: the vendored predictor loaders only stack tensors and
``load_state_dict`` (``torch.compile`` is lazy and liger ``.predict`` is never
called here), so they verify the shape/transpose/GQA contract on CPU.
"""

import tempfile
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.sparselora._vendor.sparselora.modules.svd import (
    create_attn_predictor,
    create_mlp_predictor,
)
from axolotl.integrations.sparselora.calibration import discover_target_modules
from axolotl.integrations.sparselora.factors import compute_factor_tensors, save_factors
from axolotl.integrations.sparselora.plugin import SparseLoRAPlugin
from axolotl.utils.dict import DictDefault


def _lora_model(num_kv_heads, target_modules):
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=64,
    )
    return get_peft_model(
        LlamaForCausalLM(cfg),
        LoraConfig(r=8, lora_alpha=16, target_modules=target_modules),
    )


@pytest.mark.parametrize("num_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_factors_load_into_vendored_predictors(num_kv_heads):
    """Computed factors must load into the vendored MLP/attn predictors as-is."""
    model = _lora_model(num_kv_heads, ["q_proj", "k_proj", "v_proj", "o_proj"]).eval()
    targets = discover_target_modules(model)
    factors = compute_factor_tensors(model, targets, rank=8)
    d = tempfile.mkdtemp()
    save_factors(factors, d)
    cfg = SimpleNamespace(path=d, predictor_rank=8)
    modules = dict(model.named_modules())

    mlp_name = next(t for t in targets if t.endswith("mlp"))
    attn_name = next(t for t in targets if t.endswith("self_attn"))

    mlp_pred = create_mlp_predictor(modules[mlp_name], 8, mlp_name, cfg)
    assert mlp_pred.w1.shape == (2, 64, 8)  # (gate+up, hidden, rank)
    assert mlp_pred.w2.shape[0] == 2

    # Should not raise regardless of MHA vs GQA (different predictor classes).
    attn_pred = create_attn_predictor(modules[attn_name], 8, attn_name, cfg)
    assert attn_pred is not None


def _validate_cfg(predictor_rank=8, sample_packing=False, load_in_4bit=False):
    return DictDefault(
        {
            "adapter": "lora",
            "sample_packing": sample_packing,
            "fsdp": None,
            "deepspeed": None,
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": False,
            "sparselora": {"predictor_rank": predictor_rank},
        }
    )


class TestValidate:
    def _run(self, model, cfg=None):
        plugin = SparseLoRAPlugin()
        cfg = cfg or _validate_cfg()
        plugin._validate(cfg, model, discover_target_modules(model))

    def test_attention_only_lora_accepted(self):
        self._run(_lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"]))

    @pytest.mark.parametrize("proj", ["gate_proj", "up_proj", "down_proj"])
    def test_mlp_lora_rejected(self, proj):
        model = _lora_model(2, ["q_proj", "v_proj", proj])
        with pytest.raises(ValueError, match="attention-only"):
            self._run(model)

    def test_predictor_rank_too_large_rejected(self):
        # GQA k/v projections have min dim = kv_size = 2*16 = 32.
        model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
        with pytest.raises(ValueError, match="predictor_rank"):
            self._run(model, _validate_cfg(predictor_rank=40))

    def test_sample_packing_rejected(self):
        model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
        with pytest.raises(ValueError, match="sample_packing"):
            self._run(model, _validate_cfg(sample_packing=True))

    def test_quantized_base_rejected(self):
        model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
        with pytest.raises(ValueError, match="full-precision base"):
            self._run(model, _validate_cfg(load_in_4bit=True))


def _fake_trainer(model, n=4, t=16):
    def make(_):
        ids = torch.randint(0, 128, (t,))
        labels = ids.clone()
        labels[: t // 2] = -100
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": torch.ones(t, dtype=torch.long),
        }

    dataset = [make(i) for i in range(n)]

    def collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    return SimpleNamespace(model=model, train_dataset=dataset, data_collator=collate)


def test_faithful_warmup_restores_lora_weights():
    from axolotl.integrations.sparselora.calibration import calibrate

    model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
    targets = discover_target_modules(model)
    factors = compute_factor_tensors(model, targets, rank=8)
    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n
    }
    cfg = DictDefault(
        {
            "learning_rate": 1e-3,
            "sparselora": {
                "target_sparsity": 0.5,
                "calibration": {
                    "method": "faithful",
                    "num_samples": 4,
                    "batch_size": 2,
                    "warmup_steps": 3,
                    "loss_budget": 0.5,
                },
            },
        }
    )
    calibrate(cfg, model, factors, _fake_trainer(model))

    after = {
        n: p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n
    }
    assert before, "expected some trainable LoRA params"
    for name, val in before.items():
        assert torch.equal(after[name], val), f"warm-up left {name} unrestored"


def test_calibration_does_not_perturb_rng():
    from axolotl.integrations.sparselora.calibration import calibrate

    model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
    targets = discover_target_modules(model)
    factors = compute_factor_tensors(model, targets, rank=8)
    trainer = _fake_trainer(model)  # build before capturing RNG (dataset uses RNG)
    cfg = DictDefault(
        {
            "learning_rate": 1e-3,
            "sparselora": {
                "target_sparsity": 0.5,
                "calibration": {
                    "method": "faithful",
                    "num_samples": 4,
                    "batch_size": 2,
                    "warmup_steps": 3,
                    "loss_budget": 0.5,
                },
            },
        }
    )
    torch.manual_seed(123)
    rng_before = torch.get_rng_state()
    calibrate(cfg, model, factors, trainer)
    assert torch.equal(torch.get_rng_state(), rng_before)
