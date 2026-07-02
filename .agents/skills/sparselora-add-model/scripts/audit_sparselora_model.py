#!/usr/bin/env python3
"""Audit a model's SparseLoRA architecture support.

Given an architecture name (``--arch qwen2``) or a HuggingFace id
(``--base-model Qwen/Qwen2.5-0.5B``), this:

1. instantiates a *small* model of that architecture (real hidden/head dims,
   ``--layers`` layers so it stays CPU-cheap), attaches LoRA to discovered
   sparsifiable projections,
2. checks each block exposes a supported MLP/attention layout: SwiGLU
   gate/up/down, fused gate_up, non-gated ViT fc1/fc2, standard q/k/v attention,
   or known vision attention aliases,
3. runs ``register_arch_wiring`` and reports which sparse module each MLP /
   attention class maps to, including any unsupported semantics (e.g. a
   non-SiLU/non-gelu_tanh gated MLP, or unsupported fused projections),
4. smoke-tests sparse apply + forward for synthetic vision towers, and
   forward/backward for text models on CUDA when available; CPU text audits stay
   structural-only because the sparse predictors need GPU kernels
   (liger/Triton).

For ``--base-model``, the architecture is built from the repo's config (random
weights, no checkpoint); custom repo code runs only with ``--trust-remote-code``.

Exit status mirrors the liger audit: ``0`` structurally supported, ``1`` not
supported (refused, missing projections, or a genuine apply/forward failure),
``2`` could not audit (import/instantiation error).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

EXIT_OK = 0
EXIT_UNSUPPORTED = 1
EXIT_CANNOT_AUDIT = 2

# Tiny synthetic configs (no download) for the common SwiGLU families.
_TINY_KW = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
)


def _tiny_model(arch: str, layers: int):
    import transformers as tf

    arch = arch.lower()
    table = {
        "llama": ("LlamaConfig", "LlamaForCausalLM", {}),
        "qwen2": ("Qwen2Config", "Qwen2ForCausalLM", {}),
        "qwen3": ("Qwen3Config", "Qwen3ForCausalLM", {"head_dim": 16}),
        "mistral": ("MistralConfig", "MistralForCausalLM", {"sliding_window": 32}),
        "gemma2": (
            "Gemma2Config",
            "Gemma2ForCausalLM",
            {"head_dim": 16, "attn_logit_softcapping": 50.0},
        ),
    }
    if arch not in table:
        raise ValueError(
            f"unknown --arch {arch!r}; known: {', '.join(sorted(table))}. "
            "Use --base-model <hf-id> for any other architecture."
        )
    cfg_cls, model_cls, extra = table[arch]
    config = getattr(tf, cfg_cls)(num_hidden_layers=layers, **_TINY_KW, **extra)
    return getattr(tf, model_cls)(config)


class _VisionBlockWrapper:
    """Small holder matching ``...visual.blocks.0.{attn,mlp}`` model paths."""

    @staticmethod
    def wrap(attn=None, mlp=None):
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                block = nn.Module()
                if attn is not None:
                    block.attn = attn
                if mlp is not None:
                    block.mlp = mlp
                self.visual = nn.Module()
                self.visual.blocks = nn.ModuleList([block])

            def forward(self, x):
                return x

        return Model()


def _tiny_vision_model(arch: str):
    import torch

    arch = arch.lower()
    if arch in {"qwen3vl", "qwen3-vl"}:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionAttention,
            Qwen3VLVisionMLP,
        )

        cfg = Qwen3VLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            depth=1,
            out_hidden_size=64,
        )
        cfg._attn_implementation = "eager"
        model = _VisionBlockWrapper.wrap(
            Qwen3VLVisionAttention(cfg), Qwen3VLVisionMLP(cfg)
        )

        def smoke(sm, targets):
            del targets
            mods = dict(sm.named_modules())
            attn = next(m for n, m in mods.items() if n.endswith("attn"))
            mlp = next(m for n, m in mods.items() if n.endswith("mlp"))
            x = torch.randn(6, 32, dtype=torch.bfloat16)
            cu = torch.tensor([0, 6], dtype=torch.int32)
            cos = torch.ones(6, 8, dtype=torch.bfloat16)
            sin = torch.zeros(6, 8, dtype=torch.bfloat16)
            out = attn(x, cu_seqlens=cu, position_embeddings=(cos, sin))
            out = mlp(out)
            assert out.shape == x.shape and torch.isfinite(out.float()).all()
            return "vision attn+MLP forward OK"

        return model, smoke

    if arch == "gemma4":
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4VisionAttention,
            Gemma4VisionMLP,
        )

        cfg = Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
        )
        cfg._attn_implementation = "eager"
        model = _VisionBlockWrapper.wrap(
            Gemma4VisionAttention(cfg, layer_idx=0), Gemma4VisionMLP(cfg)
        )

        def smoke(sm, targets):
            del targets
            mods = dict(sm.named_modules())
            attn = next(m for n, m in mods.items() if n.endswith("attn"))
            mlp = next(m for n, m in mods.items() if n.endswith("mlp"))
            x = torch.randn(1, 6, 32, dtype=torch.bfloat16)
            cos = torch.ones(1, 6, 8, dtype=torch.bfloat16)
            sin = torch.zeros(1, 6, 8, dtype=torch.bfloat16)
            pos = torch.zeros(1, 6, 2, dtype=torch.long)
            out = attn(x, position_embeddings=(cos, sin), position_ids=pos)[0]
            out = mlp(out)
            assert out.shape == x.shape and torch.isfinite(out.float()).all()
            return "vision attn+MLP forward OK"

        return model, smoke

    if arch in {"gemma3", "siglip"}:
        from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
        from transformers.models.siglip.modeling_siglip import (
            SiglipAttention,
            SiglipMLP,
        )

        cfg = SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
        )
        cfg._attn_implementation = "eager"
        model = _VisionBlockWrapper.wrap(SiglipAttention(cfg), SiglipMLP(cfg))

        def smoke(sm, targets):
            del targets
            mods = dict(sm.named_modules())
            attn = next(m for n, m in mods.items() if n.endswith("attn"))
            mlp = next(m for n, m in mods.items() if n.endswith("mlp"))
            x = torch.randn(1, 6, 32, dtype=torch.bfloat16)
            out = attn(x)[0]
            out = mlp(out)
            assert out.shape == x.shape and torch.isfinite(out.float()).all()
            return "vision attn+MLP forward OK"

        return model, smoke

    if arch in {"internvl", "internvl3"}:
        from transformers.models.internvl.configuration_internvl import (
            InternVLVisionConfig,
        )
        from transformers.models.internvl.modeling_internvl import (
            InternVLVisionAttention,
            InternVLVisionMLP,
        )

        cfg = InternVLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            use_qk_norm=True,
        )
        cfg._attn_implementation = "eager"
        model = _VisionBlockWrapper.wrap(
            InternVLVisionAttention(cfg), InternVLVisionMLP(cfg)
        )

        def smoke(sm, targets):
            del targets
            mods = dict(sm.named_modules())
            attn = next(m for n, m in mods.items() if n.endswith("attn"))
            mlp = next(m for n, m in mods.items() if n.endswith("mlp"))
            x = torch.randn(1, 6, 32, dtype=torch.bfloat16)
            out = attn(x)[0]
            out = mlp(out)
            assert out.shape == x.shape and torch.isfinite(out.float()).all()
            return "vision attn+MLP forward OK"

        return model, smoke

    raise ValueError(
        f"unknown --vision-arch {arch!r}; known: qwen3vl, gemma4, gemma3/siglip, internvl"
    )


def _from_pretrained(base_model: str, layers: int, trust_remote_code: bool = False):
    """Instantiate ``base_model``'s architecture from its config, shrunk to
    ``layers`` layers (real hidden/head dims, random weights, no checkpoint).

    ``trust_remote_code`` executes the model repo's custom code and is off by
    default; pass ``--trust-remote-code`` only for repos you trust.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(config.num_hidden_layers, layers)
    if getattr(config, "tie_word_embeddings", False):
        config.tie_word_embeddings = True
    return AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)


@dataclass
class Findings:
    mlp_classes: dict[str, str] = field(default_factory=dict)  # cls -> sparse role
    attn_classes: dict[str, str] = field(default_factory=dict)
    refusals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    smoke: str = "not run"


def _target_suffix(name: str, module) -> str:
    return f"{name}.linear" if hasattr(module, "linear") else name


def audit(
    model,
    predictor_rank: int,
    sparsity: float,
    smoke_fn: Callable | None = None,
) -> tuple[Findings, int]:
    import torch
    from peft import LoraConfig, get_peft_model

    from axolotl.integrations.sparselora._vendor.sparselora import (
        SparseLoRAConfig,
        apply_sparselora,
    )
    from axolotl.integrations.sparselora._vendor.sparselora.modules import (
        SparseModule,
        get_module_mapping,
    )
    from axolotl.integrations.sparselora.arch_wiring import (
        is_fused_gate_up_mlp,
        is_fused_qkv_attention,
        is_non_gated_mlp,
        is_standard_attention,
        is_swiglu_mlp,
        register_arch_wiring,
        unsupported_reason,
    )
    from axolotl.integrations.sparselora.calibration import discover_target_modules
    from axolotl.integrations.sparselora.factors import (
        attention_output_projection_name,
        compute_factor_tensors,
        fused_qkv_projection_name,
        non_gated_mlp_projection_names,
        save_factors,
    )

    f = Findings()

    lora_targets: set[str] = set()
    for mod in model.modules():
        if is_standard_attention(mod):
            out = attention_output_projection_name(mod)
            for proj in ("q_proj", "k_proj", "v_proj", out):
                if proj is not None:
                    lora_targets.add(_target_suffix(proj, getattr(mod, proj)))
        elif is_fused_qkv_attention(mod):
            qkv = fused_qkv_projection_name(mod)
            out = attention_output_projection_name(mod)
            for proj in (qkv, out):
                if proj is not None:
                    lora_targets.add(_target_suffix(proj, getattr(mod, proj)))

        if is_swiglu_mlp(mod):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                lora_targets.add(_target_suffix(proj, getattr(mod, proj)))
        elif is_fused_gate_up_mlp(mod):
            for proj in ("gate_up_proj", "down_proj"):
                lora_targets.add(_target_suffix(proj, getattr(mod, proj)))
        elif is_non_gated_mlp(mod):
            names = non_gated_mlp_projection_names(mod)
            if names is not None:
                for proj in names:
                    lora_targets.add(_target_suffix(proj, getattr(mod, proj)))

    if not lora_targets:
        lora_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}

    model = get_peft_model(
        model,
        LoraConfig(
            r=predictor_rank,
            lora_alpha=2 * predictor_rank,
            target_modules=sorted(lora_targets),
        ),
    ).eval()

    targets = discover_target_modules(model)
    modules = dict(model.named_modules())
    if not targets:
        f.refusals.append(
            "no supported MLP or attention blocks found (gate/up/down, fused "
            "gate_up/down, fc1/fc2, q/k/v/o, or fused qkv projections missing) "
            "- architecture not sparsifiable as-is."
        )
        return f, EXIT_UNSUPPORTED

    register_arch_wiring(model)
    mapping = get_module_mapping()
    refusals: dict[str, None] = {}  # ordered dedupe
    for name in targets:
        mod = modules[name]
        cls = type(mod).__name__
        reason = unsupported_reason(mod)
        if reason:
            refusals[reason] = None
            continue
        sparse_cls = mapping.get(type(mod))
        if is_swiglu_mlp(mod) or is_fused_gate_up_mlp(mod) or is_non_gated_mlp(mod):
            f.mlp_classes[cls] = sparse_cls.__name__ if sparse_cls else "UNMAPPED"
        elif is_standard_attention(mod) or is_fused_qkv_attention(mod):
            f.attn_classes[cls] = sparse_cls.__name__ if sparse_cls else "UNMAPPED"
    f.refusals = list(refusals)

    if f.refusals:
        return f, EXIT_UNSUPPORTED
    if "UNMAPPED" in {*f.mlp_classes.values(), *f.attn_classes.values()}:
        f.refusals.append("a target class did not map to any sparse module")
        return f, EXIT_UNSUPPORTED

    if smoke_fn is not None:
        try:
            sm = model.to(torch.bfloat16).eval()
            d = tempfile.mkdtemp()
            save_factors(compute_factor_tensors(sm, targets, predictor_rank), d)
            apply_sparselora(
                sm,
                SparseLoRAConfig(
                    layer_sparsity={t: sparsity for t in targets},
                    predictor_rank=predictor_rank,
                    path=d,
                ),
            )
            sparse_mods = [m for m in sm.modules() if isinstance(m, SparseModule)]
            assert sparse_mods, "no sparse modules were installed"
            note = smoke_fn(sm, targets)
            f.smoke = f"PASSED (sparse @ {sparsity}, {note})"
            return f, EXIT_OK
        except Exception as exc:  # noqa: BLE001
            f.smoke = f"FAILED ({type(exc).__name__}: {exc})"
            f.notes.append(traceback.format_exc())
            return f, EXIT_UNSUPPORTED

    # Text smoke test needs CUDA: the vendored SiLU MLP path uses liger kernels.
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        f.smoke = "skipped (no CUDA; liger predictors + silu_mul need a GPU)"
        f.notes.append(
            "Structural support confirmed on CPU. Re-run on a GPU to exercise the "
            "sparse apply + forward/backward smoke test."
        )
        return f, EXIT_OK

    try:
        sm = model.to("cuda").to(torch.bfloat16).train()
        d = tempfile.mkdtemp()
        save_factors(compute_factor_tensors(sm, targets, predictor_rank), d)
        apply_sparselora(
            sm,
            SparseLoRAConfig(
                layer_sparsity={t: sparsity for t in targets},
                predictor_rank=predictor_rank,
                path=d,
            ),
        )
        sparse_mods = [m for m in sm.modules() if isinstance(m, SparseModule)]
        assert sparse_mods, "no sparse modules were installed"

        vocab = int(getattr(sm.config, "vocab_size", _TINY_KW["vocab_size"]))
        ids = torch.randint(0, max(1, vocab), (2, 24), device="cuda")
        labels = ids.clone()
        labels[:, :10] = -100
        out = sm(input_ids=ids, labels=labels)
        if not torch.isfinite(out.loss).item():
            f.smoke = "FAILED (non-finite loss)"
            return f, EXIT_UNSUPPORTED
        out.loss.backward()
        grad = sum(
            p.grad.float().norm().item()
            for n, p in sm.named_parameters()
            if p.grad is not None and "lora_" in n
        )
        if grad <= 0:
            f.smoke = "FAILED (no LoRA gradient)"
            return f, EXIT_UNSUPPORTED
        f.smoke = (
            f"PASSED (sparse @ {sparsity}, finite loss, LoRA grad norm {grad:.3g})"
        )
    except Exception as exc:  # noqa: BLE001
        f.smoke = f"FAILED ({type(exc).__name__}: {exc})"
        f.notes.append(traceback.format_exc())
        return f, EXIT_UNSUPPORTED

    return f, EXIT_OK


def _print(target: str, f: Findings, code: int) -> None:
    print(f"\nSparseLoRA model audit: {target}")
    print("=" * 60)
    print("\n[1] MLP classes -> sparse module")
    if f.mlp_classes:
        for cls, role in sorted(f.mlp_classes.items()):
            print(f"    {cls:32s} -> {role}")
    else:
        print("    (none found)")
    print("\n[2] Attention classes -> sparse module")
    if f.attn_classes:
        for cls, role in sorted(f.attn_classes.items()):
            print(f"    {cls:32s} -> {role}")
    else:
        print("    (none found)")
    if f.refusals:
        print("\n[3] Refused (unsupported semantics)")
        for r in f.refusals:
            print(f"    - {r}")
    print(f"\n[4] Smoke apply+forward: {f.smoke}")
    for note in f.notes:
        print(f"\n    note: {note.rstrip()}")
    verdict = {
        EXIT_OK: "SUPPORTED",
        EXIT_UNSUPPORTED: "NOT SUPPORTED",
        EXIT_CANNOT_AUDIT: "COULD NOT AUDIT",
    }[code]
    print("\n" + "=" * 60)
    print(f"verdict: {verdict} (exit {code})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--arch", help="synthetic tiny model: llama|qwen2|qwen3|mistral|gemma2"
    )
    g.add_argument(
        "--vision-arch",
        help="synthetic tiny vision tower: qwen3vl|gemma4|gemma3|siglip|internvl",
    )
    g.add_argument(
        "--base-model", help="HuggingFace id; built from config, no checkpoint"
    )
    ap.add_argument(
        "--layers", type=int, default=2, help="cap hidden layers (default 2)"
    )
    ap.add_argument("--predictor-rank", type=int, default=8)
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="execute the --base-model repo's custom code (off by default)",
    )
    args = ap.parse_args()

    target = args.arch or args.vision_arch or args.base_model
    try:
        smoke_fn = None
        if args.arch:
            model = _tiny_model(args.arch, args.layers)
        elif args.vision_arch:
            model, smoke_fn = _tiny_vision_model(args.vision_arch)
        else:
            model = _from_pretrained(
                args.base_model, args.layers, args.trust_remote_code
            )
    except Exception as exc:  # noqa: BLE001
        print(
            f"could not instantiate {target!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_CANNOT_AUDIT

    try:
        findings, code = audit(model, args.predictor_rank, args.sparsity, smoke_fn)
    except ImportError as exc:
        print(f"could not audit (missing dependency): {exc}", file=sys.stderr)
        return EXIT_CANNOT_AUDIT
    except Exception as exc:  # noqa: BLE001
        print(
            f"could not audit {target!r}: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        traceback.print_exc()
        return EXIT_CANNOT_AUDIT

    _print(target, findings, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
