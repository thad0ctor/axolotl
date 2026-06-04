"""NVFP4-GEMM training config schema.

Real low-precision COMPUTE (FP4 forward/backward GEMMs on Blackwell), distinct
from the fake-quant QAT/PTQ `quantization:` block.
"""

import warnings
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class NVFP4AttentionBackwardConfig(BaseModel):
    """Native-NVFP4 attention training (backward) settings (expert recipe)."""

    enabled: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Use the native NVFP4 autograd attention path while "
            "training. Validated for convergence but can be slower than bf16 at "
            "short sequence lengths, so it stays explicitly opt-in."
        },
    )
    rtn_grad_packs: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Deterministic round-to-nearest for the measured-safe "
            "gradient packs (softmax P / transposed dO for dV, dS for dQ), leaving "
            "the dK and dPt packs governed by stochastic_rounding. Faster in "
            "microbenchmarks; convergence validation still required."
        },
    )
    save_packs: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Save the forward pass's deterministic Q/K/V NVFP4 packs "
            "(+ transposed backward layouts) and reuse them in backward — trades "
            "activation memory for less backward pack-prep work."
        },
    )
    dkdv_scratch_bf16: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Store the per-query-head dK/dV scratch buffers in bf16 "
            "before GQA reduction instead of fp32 (less scratch traffic; bit-exact "
            "vs the fp32-then-cast path)."
        },
    )
    compile_custom_op: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Route the native NVFP4 flash attention through an opaque "
            "torch custom op (torch.compile escape hatch). Tri-state: None "
            "auto-enables it whenever torch_compile is on; True/False force it."
        },
    )


class NVFP4AttentionConfig(BaseModel):
    """Native-NVFP4 full-attention path (model-agnostic; applied where the "
    architecture supports it). Replaces the flat ``qwen3_5_native_attention*`` flags."""

    enabled: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch full softmax-attention layers to the native NVFP4 "
            "attention path on dense causal/full batches. Falls back to the model's "
            "configured attention for unsupported masks/cache states."
        },
    )
    fuse_vproj: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Run v_proj as a native NVFP4 GEMM with key-axis pack "
            "epilogue on inference/cache-free prefill (~1.4-1.5x V producer; v_proj "
            "goes FP4). Default (null): ON for inference, OFF for training."
        },
    )
    fp4_projections: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Run q/k/o_proj as native NVFP4 GEMMs (q/k share one "
            "activation pack) on inference prefill. Parity-affecting (not bit-exact); "
            "speed-neutral on hybrid models, a per-layer win on dense models. OFF by "
            "default; plain-nn.Linear only."
        },
    )
    backward: NVFP4AttentionBackwardConfig = Field(
        default_factory=NVFP4AttentionBackwardConfig,
        json_schema_extra={"description": "Native-NVFP4 attention training settings."},
    )

    @model_validator(mode="after")
    def _check_requires(self):
        if not self.enabled:
            if self.backward.enabled:
                raise ValueError(
                    "nvfp4_training.attention.backward.enabled requires "
                    "attention.enabled: true."
                )
            if self.fuse_vproj:
                raise ValueError(
                    "nvfp4_training.attention.fuse_vproj requires attention.enabled: true."
                )
            if self.compile_custom_op_set():
                raise ValueError(
                    "nvfp4_training.attention.backward.compile_custom_op requires "
                    "attention.enabled: true."
                )
        if not self.backward.enabled:
            for sub in ("rtn_grad_packs", "save_packs", "dkdv_scratch_bf16"):
                if getattr(self.backward, sub):
                    raise ValueError(
                        f"nvfp4_training.attention.backward.{sub} requires "
                        "attention.backward.enabled: true."
                    )
        return self

    def compile_custom_op_set(self) -> bool:
        return bool(self.backward.compile_custom_op)


class NVFP4TrainingConfig(BaseModel):
    """NVFP4-GEMM training settings (module-swap on eligible nn.Linear)."""

    enabled: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable NVFP4-GEMM training (real FP4 forward/backward GEMMs). "
            "Blackwell-only. The speedup only materializes under torch.compile."
        },
    )
    backend: Literal["native", "te"] = Field(
        default="native",
        json_schema_extra={
            "description": "FP4 compute backend. 'native' (default): our "
            "torch._scaled_mm module-swap; runs the full SR+RHT convergence recipe "
            "on any FP4-capable GPU (sm_100/sm_120), no extra build. 'te': NVIDIA "
            "Transformer Engine NVFP4BlockScaling; faster hand-tuned kernels, but "
            "needs `pip install axolotl[transformer-engine]` (source build) and on "
            "consumer Blackwell (sm_120) its RHT/SR kernels do not run — TE there is "
            "recipe-less (convergence unproven). FFT only."
        },
    )
    stochastic_rounding: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Stochastic rounding on gradient operands (NVFP4Recipe)."
        },
    )
    hadamard: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Random Hadamard transform on wgrad inputs (NVFP4Recipe)."
        },
    )
    quantize_base: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Adapter modes only. When false (LoRA + FP4 compute): the "
            "frozen base weight stays high-precision and only the GEMM operands are "
            "FP4 — a throughput win, no memory saving. When true (NVFP4-QLoRA): the "
            "frozen base weight is stored packed in FP4 (~3.5x weight memory saving), "
            "replacing bnb NF4 storage. This is the FP4 equivalent of QLoRA, so it "
            "conflicts with load_in_4bit/load_in_8bit (drop those). `adapter: qlora` "
            "implies this (qlora == quantized base intent)."
        },
    )
    base_mode: Literal["compute", "storage", "hp"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Adapter modes only; overrides quantize_base when set. "
            "'compute' (recommended): the frozen base is pre-quantized once into two "
            "NVFP4 layouts so fprop+dgrad run as pure FP4 GEMMs with no per-step base "
            "quant prologue — fastest base compute, ~1.75x weight memory. 'storage' "
            "(== NVFP4-QLoRA): base packed in FP4, ~3.5x memory, backward dequants to "
            "bf16 (max memory, modest speed). 'hp': base kept high-precision, "
            "re-quantized each step (FP4 GEMM, no memory win)."
        },
    )
    exclude_modules: list[str] = Field(
        default_factory=lambda: ["lm_head", "embed_tokens"],
        json_schema_extra={
            "description": "Module name fragments kept in high precision (not swapped)."
        },
    )
    quantize_lm_head: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Remove lm_head from the high-precision exclusion and "
            "quantize the output projection to NVFP4 (memory + GEMM win on a large "
            "[vocab, hidden] tensor), at a small convergence cost (the NVFP4 paper "
            "keeps lm_head bf16). OFF by default. With UNtied embeddings only the "
            "lm_head is quantized (embed_tokens stays excluded unless "
            "quantize_embeddings is also set). With TIED embeddings (the frozen "
            "shared weight) the shared weight is quantized ONCE and routed to both "
            "the embedding lookup and the lm_head GEMM. A TRAINABLE tied weight "
            "still RAISES (FP4-storing it would corrupt training)."
        },
    )
    fused_fp4_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Requires quantize_lm_head. Fuse the FP4 lm_head "
            "projection with the cross-entropy loss so the full [batch*seq, vocab] "
            "logit tensor is never materialized (~1 GiB at vocab 152k / seq 8k): "
            "the loss is computed by tiling over the vocab, dequantizing each FP4 "
            "weight tile on read, and accumulating logsumexp in fp32 (like "
            "cut_cross_entropy, but reading the NVFP4-packed weight directly). "
            "This is a MEMORY win (no logit materialization), not an FP4-GEMM "
            "throughput win — the per-tile matmul runs in bf16/fp32, so the lm_head "
            "does not hit FP4 tensor cores. Frozen lm_head only (returns dL/dhidden, "
            "no weight grad). Falls back to the materialized path if the lm_head "
            "store isn't row-sliceable (MSLK-swizzled scales) or carries a bias. "
            "OFF by default."
        },
    )
    bf16_lm_head_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch a plain frozen bias-free nn.Linear lm_head "
            "training forward to compute cross-entropy by tiling over the vocab in "
            "bf16, avoiding full [batch*seq, vocab] logits materialization (~1 GiB "
            "at vocab 152k / seq 8k) and the matching logits-gradient GEMM. The "
            "per-tile lm_head matmul runs in plain bf16 (bit-for-bit the same "
            "arithmetic as the materialized hidden @ W.t()); logsumexp/softmax and "
            "the dL/dhidden accumulation are kept in fp32. This is the exact tiled "
            "CE gradient (no low-probability vocab filtering), so it is "
            "convergence-safe under NVFP4 stochastic-rounding grads where the fused "
            "cut_cross_entropy / Liger paths collapsed. Returns dL/dhidden only (no "
            "lm_head weight grad). Incompatible with quantize_lm_head, "
            "fused_fp4_cross_entropy, and the FP8 cross-entropy patch. This is a "
            "MEMORY/backward-traffic win, not an FP4 tensor-core throughput win. "
            "OFF by default."
        },
    )
    fp8_lm_head: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch a plain frozen lm_head to use torch FP8 scaled "
            "matmul in eval/no-grad forward only. Training forwards still use the "
            "original high-precision Linear. This is for eval/scoring/logprob "
            "throughput; greedy generation can diverge after the first changed "
            "argmax token. OFF by default."
        },
    )
    fp8_lm_head_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch a plain frozen bias-free nn.Linear lm_head "
            "training forward to compute cross-entropy with FP8 scaled-matmul vocab "
            "tiles, avoiding full [batch*seq, vocab] logits materialization. "
            "Returns dL/dhidden only (no lm_head weight grad); backward uses FP8 "
            "scaled matmul against a prepacked dgrad weight layout. Incompatible "
            "with quantize_lm_head, "
            "fused_fp4_cross_entropy, and other cross-entropy optimization patches. "
            "OFF by default."
        },
    )
    fp8_lm_head_granularity: Literal["tensorwise", "rowwise"] = Field(
        default="rowwise",
        json_schema_extra={
            "description": "Scaling granularity for fp8_lm_head. Rowwise keeps one "
            "scale per vocab row and had the best Qwen3.5 real-hidden-state argmax "
            "parity in the validation sweep; it is also the validated FP8 CE "
            "training default."
        },
    )
    fused_fp4_cross_entropy_fp4_matmul: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Experimental. When fused_fp4_cross_entropy is enabled, "
            "use per-tile FP4 torch._scaled_mm for lm_head logits before the tiled "
            "online logsumexp. This hits Blackwell FP4 tensor cores for the matmul, "
            "but still materializes each [tokens, vocab_block] tile and still uses "
            "a dequantized weight tile for dhidden in backward. Early microbenchmarks "
            "show it is faster than the memory-only FP4 CE path but slower than "
            "materialized bf16/Liger CE; keep this off unless benchmarking the "
            "next native CE-epilogue design."
        },
    )
    quantize_embeddings: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Store the input token embedding (embed_tokens) packed "
            "in FP4 and dequantize on lookup (W4A16; the lookup gathers rows, no "
            "activation quant). FROZEN only — a trainable embedding is skipped with "
            "a warning (no FP4 master for gradients). Hidden dim must be %16. OFF by "
            "default."
        },
    )
    quantize_vision_tower: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Multimodal only. Swap the FROZEN nn.Linear layers under "
            "the vision encoder (attn qkv/proj, mlp fc1/fc2) to the NVFP4 frozen "
            "base. The vision merger and patch-embed projection are left untouched, "
            "as are %32-ineligible dims. Warns if no vision tower is found (text "
            "model). OFF by default."
        },
    )
    fuse_rmsnorm: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Default off (throughput benefit is within noise; under "
            "torch.compile the reuse cache is bypassed). Fuse decoder RMSNorm with "
            "the NVFP4 activation quant in one Triton kernel so the qkv/gate-up base "
            "linears reuse the norm's pre-quantized activation. Auto-detects the "
            "gamma convention (plain `weight` vs zero-centered `1 + weight`) per norm "
            "and verifies each swap against the original before committing (reverts "
            "on mismatch)."
        },
    )
    fuse_qk_norm_rope: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Experimental, Qwen only. Monkeypatch qwen3/qwen3_5/"
            "qwen3_5_moe attention to fuse head-dim q_norm/k_norm with RoPE using "
            "the Gemma4 fused RMSNorm+RoPE Triton kernel. This removes the separate "
            "q/k RMSNorm and rotate_half/apply_rope materializations but leaves the "
            "attention matmul itself unchanged. OFF by default."
        },
    )
    skip_first_n_blocks: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": "Keep the first N transformer blocks in high precision. "
            "The NVFP4 paper keeps ~15% of linear layers in bf16 (embeddings/lm_head "
            "plus the first ~2 and last ~8 blocks of a 12B model, weighted to the "
            "tail) for convergence; for a model with L blocks a reasonable default is "
            "skip_first_n_blocks=1 and skip_last_n_blocks~=round(0.13*L)."
        },
    )
    skip_last_n_blocks: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": "Keep the last N transformer blocks in high precision. "
            "See skip_first_n_blocks for the ~15% high-precision policy from the "
            "NVFP4 training paper (the tail blocks matter most)."
        },
    )
    save_nvfp4: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Store eligible weights NVFP4-packed (qdata + scales) in "
            "a torch.save sidecar (nvfp4_packed.pt) on every checkpoint/final save, "
            "for ~3.5x smaller weight files. safetensors cannot serialize the FP4 "
            "tensor subclass, so the packed weights go to a sidecar and the bf16 "
            "weights are dropped from the safetensors shard. For FROZEN weights "
            "(LoRA/QLoRA base, lm_head, embeddings) the FP4 IS the faithful stored "
            "form — load-back is bit-exact. For FFT (NVFP4Linear keeps a bf16 master) "
            "this is LOSSY for resume: only the FP4 packing is kept, no bf16 master, "
            "so a save_nvfp4 FFT checkpoint is for storage/inference export, not exact "
            "resume. OFF by default (bf16 save, unchanged)."
        },
    )
    attention: NVFP4AttentionConfig = Field(
        default_factory=NVFP4AttentionConfig,
        json_schema_extra={
            "description": "Native-NVFP4 full-attention path settings (model-agnostic; "
            "applied where the architecture supports it). Replaces the deprecated flat "
            "qwen3_5_native_attention* flags."
        },
    )
    linear_attn: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch linear-attention (e.g. GatedDeltaNet) large "
            "projection GEMMs to native NVFP4 in no-grad forward/eval. Training "
            "forwards fall back to the original implementation."
        },
    )
    mlp: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch dense SwiGLU MLP GEMMs to native NVFP4 in no-grad "
            "forward/eval. Training forwards fall back to the original implementation."
        },
    )
    fla_causal_conv_compile_boundary: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Sample-packing only. Run FLA varlen causal_conv1d behind a "
            "torch.compile boundary so packed cu_seqlens length changes do not trigger "
            "repeated Dynamo recompiles. Trades graph coverage for steadier steps."
        },
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_attention_flags(cls, data):
        """Map deprecated flat ``qwen3_5_*`` attention flags onto the nested schema."""
        if not isinstance(data, dict):
            return data
        attn_map = {
            "qwen3_5_native_attention": ("enabled",),
            "qwen3_5_fuse_vproj": ("fuse_vproj",),
            "qwen3_5_native_attention_backward": ("backward", "enabled"),
            "qwen3_5_native_attention_backward_rtn_grad_packs": ("backward", "rtn_grad_packs"),
            "qwen3_5_native_attention_save_backward_packs": ("backward", "save_packs"),
            "qwen3_5_native_attention_dkdv_scratch_bf16": ("backward", "dkdv_scratch_bf16"),
            "qwen3_5_native_attention_compile_custom_op": ("backward", "compile_custom_op"),
        }
        top_map = {
            "qwen3_5_native_linear_attn": "linear_attn",
            "qwen3_5_native_mlp": "mlp",
            "qwen3_5_fla_causal_conv_compile_boundary": "fla_causal_conv_compile_boundary",
        }
        used: list[str] = []
        attn = dict(data.get("attention") or {})
        bwd = dict(attn.get("backward") or {})
        for old, path in attn_map.items():
            if old not in data:
                continue
            used.append(old)
            val = data.pop(old)
            if len(path) == 1:
                attn.setdefault(path[0], val)  # explicit nested value wins
            else:
                bwd.setdefault(path[1], val)
        if bwd:
            attn["backward"] = bwd
        if attn:
            data["attention"] = attn
        for old, new in top_map.items():
            if old in data:
                used.append(old)
                data.setdefault(new, data.pop(old))
        if used:
            warnings.warn(
                "nvfp4_training: flat attention flags "
                f"{sorted(used)} are deprecated; use the nested "
                "nvfp4_training.attention.* schema instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return data
