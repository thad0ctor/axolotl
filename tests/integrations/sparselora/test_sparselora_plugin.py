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
from axolotl.integrations.sparselora.plugin import (
    SparseLoRAPlugin,
    resolve_layer_sparsity,
)
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

    # Pin the concrete predictor buffer shapes so a wrong q/k/v load order or a
    # missing transpose fails (not just "constructed without raising").
    attn_pred = create_attn_predictor(modules[attn_name], 8, attn_name, cfg)
    head_dim = 64 // 4  # hidden_size / num_attention_heads
    kv_size = num_kv_heads * head_dim
    if num_kv_heads == 4:  # MHA: q/k/v share an out dim -> AttentionPredictor
        assert attn_pred.w1.shape == (3, 64, 8)  # (q,k,v, hidden, rank)
        assert attn_pred.w2.shape == (3, 8, 64)  # (q,k,v, rank, out)
        assert not hasattr(attn_pred, "q1")
    else:  # GQA: q split from grouped k/v -> GQAAttentionPredictor (+ q1/q2)
        assert attn_pred.w1.shape == (2, 64, 8)  # (k,v, hidden, rank)
        assert attn_pred.w2.shape == (2, 8, kv_size)  # (k,v, rank, kv_out)
        assert attn_pred.q1.shape == (8, 64)  # (rank, hidden) = q.w1 transposed
        assert attn_pred.q2.shape == (64, 8)  # (hidden, rank) = q.w2 transposed


def _validate_cfg(
    predictor_rank=8,
    sample_packing=False,
    load_in_4bit=False,
    load_in_8bit=False,
    fsdp=None,
    deepspeed=None,
):
    return DictDefault(
        {
            "adapter": "lora",
            "sample_packing": sample_packing,
            "fsdp": fsdp,
            "deepspeed": deepspeed,
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
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

    def test_8bit_base_rejected(self):
        model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
        with pytest.raises(ValueError, match="8-bit"):
            self._run(model, _validate_cfg(load_in_8bit=True))

    def test_4bit_base_accepted_by_validation(self):
        # 4-bit (QLoRA) is supported; validation should not reject it.
        model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
        self._run(model, _validate_cfg(load_in_4bit=True))

    def test_fsdp_rejected(self):
        model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
        with pytest.raises(ValueError, match="FSDP"):
            self._run(model, _validate_cfg(fsdp={"fsdp_offload_params": True}))

    @pytest.mark.parametrize(
        "deepspeed",
        [
            "deepspeed_configs/zero3.json",
            {"zero_optimization": {"stage": 3}},
        ],
    )
    def test_deepspeed_zero3_rejected(self, deepspeed):
        model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
        with pytest.raises(ValueError, match="ZeRO-3"):
            self._run(model, _validate_cfg(deepspeed=deepspeed))

    def test_deepspeed_zero2_accepted(self):
        # ZeRO-1/2 do not shard parameters, so they must not be rejected.
        model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
        self._run(model, _validate_cfg(deepspeed={"zero_optimization": {"stage": 2}}))

    @pytest.mark.parametrize("zero", [True, False])
    def test_deepspeed_legacy_bool_zero_optimization(self, zero):
        # Legacy boolean zero_optimization maps to stage 1/0 (never 3): must not
        # crash on .get("stage") and must not be rejected.
        model = _lora_model(2, ["q_proj", "k_proj", "v_proj", "o_proj"])
        self._run(model, _validate_cfg(deepspeed={"zero_optimization": zero}))


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


def test_compile_boundaries_disable_dynamic_regions():
    from axolotl.integrations.sparselora._vendor.sparselora import api
    from axolotl.integrations.sparselora._vendor.sparselora.modules import (
        llama,
        predictors,
    )
    from axolotl.integrations.sparselora.plugin import _apply_compile_boundaries
    from axolotl.integrations.sparselora.sparse_linear_4bit import SparseLinear4bit

    _apply_compile_boundaries()

    def disabled(fn):
        return getattr(fn, "_torchdynamo_disable", False)

    assert disabled(llama.SparseLlamaMLP.forward)
    assert disabled(llama.SparseLlamaAttention.forward)
    assert disabled(api._compute_output_token_mask)
    assert disabled(predictors.FFNPredictor.predict)
    assert disabled(predictors.AttentionPredictor.predict)
    assert disabled(predictors.GQAAttentionPredictor.predict)
    assert disabled(SparseLinear4bit.forward)


class TestResolveLayerSparsity:
    targets = [
        "base_model.model.model.layers.0.self_attn",
        "base_model.model.model.layers.0.mlp",
        "base_model.model.model.layers.10.mlp",
    ]

    def test_suffix_key_matches(self):
        out = resolve_layer_sparsity({"model.layers.0.mlp": 0.5}, self.targets)
        assert out == {"base_model.model.model.layers.0.mlp": 0.5}

    def test_full_path_key_matches(self):
        out = resolve_layer_sparsity({self.targets[0]: 0.7}, self.targets)
        assert out == {self.targets[0]: 0.7}

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="matches no sparsifiable"):
            resolve_layer_sparsity({"model.layers.99.mlp": 0.5}, self.targets)

    def test_suffix_respects_dot_boundary(self):
        # '0.mlp' must not match 'layers.10.mlp'
        out = resolve_layer_sparsity({"layers.0.mlp": 0.5}, self.targets)
        assert list(out) == ["base_model.model.model.layers.0.mlp"]


def test_empty_calibration_data_raises():
    from types import SimpleNamespace

    from axolotl.integrations.sparselora.calibration import build_calibration_loader

    trainer = SimpleNamespace(train_dataset=[], data_collator=lambda b: b)
    with pytest.raises(ValueError, match="no data"):
        build_calibration_loader(trainer, num_samples=8, batch_size=1)


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
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    calibrate(cfg, model, factors, trainer)
    assert torch.equal(torch.get_rng_state(), rng_before)
    if cuda_before is not None:  # calibrate also restores CUDA RNG
        for before, after in zip(
            cuda_before, torch.cuda.get_rng_state_all(), strict=True
        ):
            assert torch.equal(before, after)
