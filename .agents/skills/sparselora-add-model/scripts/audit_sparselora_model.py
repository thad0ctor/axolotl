#!/usr/bin/env python3
"""Audit a model's SparseLoRA architecture support.

Given an architecture name (``--arch qwen2``) or a HuggingFace id
(``--base-model Qwen/Qwen2.5-0.5B``), this:

1. instantiates a *small* model of that architecture (real hidden/head dims,
   ``--layers`` layers so it stays CPU-cheap), attaches attention-only LoRA,
2. checks each block exposes the SwiGLU MLP (gate/up/down) and standard
   attention (q/k/v/o) projections SparseLoRA needs,
3. runs ``register_arch_wiring`` and reports which sparse module each MLP /
   attention class maps to, including any unsupported semantics (e.g. a
   non-SiLU/non-gelu_tanh gated MLP, or fused projections),
4. smoke-tests sparse apply + forward/backward on CUDA when available; on CPU it
   performs structural checks only, because the sparse predictors need GPU
   kernels (liger/Triton).

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


def audit(model, predictor_rank: int, sparsity: float) -> tuple[Findings, int]:
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
        is_fused_qkv_attention,
        is_standard_attention,
        is_swiglu_mlp,
        register_arch_wiring,
        unsupported_reason,
    )
    from axolotl.integrations.sparselora.calibration import discover_target_modules
    from axolotl.integrations.sparselora.factors import (
        compute_factor_tensors,
        save_factors,
    )

    f = Findings()

    # Discover the attention LoRA targets from the model: fused-projection
    # models (Phi3) expose ``qkv_proj`` rather than separate ``q/k/v_proj``, so a
    # hardcoded q/k/v target list would attach no adapters and mis-audit them.
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for mod in model.modules():
        if is_fused_qkv_attention(mod):
            lora_targets = ["qkv_proj", "o_proj"]
            break
        if is_standard_attention(mod):
            break

    model = get_peft_model(
        model,
        LoraConfig(
            r=predictor_rank,
            lora_alpha=2 * predictor_rank,
            target_modules=lora_targets,
        ),
    ).eval()

    targets = discover_target_modules(model)
    modules = dict(model.named_modules())
    if not targets:
        f.refusals.append(
            "no SwiGLU-MLP or standard-attention blocks found (gate/up/down or "
            "q/k/v/o projections missing) — architecture not sparsifiable as-is."
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
        if is_swiglu_mlp(mod):
            f.mlp_classes[cls] = sparse_cls.__name__ if sparse_cls else "UNMAPPED"
        elif is_standard_attention(mod):
            f.attn_classes[cls] = sparse_cls.__name__ if sparse_cls else "UNMAPPED"
    f.refusals = list(refusals)

    if f.refusals:
        return f, EXIT_UNSUPPORTED
    if "UNMAPPED" in {*f.mlp_classes.values(), *f.attn_classes.values()}:
        f.refusals.append("a target class did not map to any sparse module")
        return f, EXIT_UNSUPPORTED

    # Smoke test needs CUDA: the vendored MLP/attention forwards use liger's
    # silu_mul + Triton predictors, so even a dense forward can't run on CPU.
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

        ids = torch.randint(0, _TINY_KW["vocab_size"], (2, 24), device="cuda")
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
    print("\n[1] SwiGLU MLP classes -> sparse module")
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

    target = args.arch or args.base_model
    try:
        model = (
            _tiny_model(args.arch, args.layers)
            if args.arch
            else _from_pretrained(args.base_model, args.layers, args.trust_remote_code)
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"could not instantiate {target!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_CANNOT_AUDIT

    try:
        findings, code = audit(model, args.predictor_rank, args.sparsity)
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
