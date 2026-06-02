"""NVFP4-GEMM training config schema.

Real low-precision COMPUTE (FP4 forward/backward GEMMs on Blackwell), distinct
from the fake-quant QAT/PTQ `quantization:` block.
"""

from typing import Literal

from pydantic import BaseModel, Field


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
    fp8_lm_head: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch a plain frozen lm_head to use torch FP8 scaled "
            "matmul in eval/no-grad forward only. Training forwards still use the "
            "original high-precision Linear until convergence is validated. OFF by "
            "default."
        },
    )
    fp8_lm_head_granularity: Literal["tensorwise", "rowwise"] = Field(
        default="rowwise",
        json_schema_extra={
            "description": "Scaling granularity for fp8_lm_head. Rowwise keeps one "
            "scale per vocab row and had the best real-model argmax parity in the "
            "Qwen3.5 sweep."
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
    qwen3_5_native_attention: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 only. Patch full softmax-attention layers to use "
            "the native NVFP4 attention path on dense causal/full batches. Forward "
            "falls back to the model's configured attention for unsupported masks or "
            "cache states. OFF by default."
        },
    )
    qwen3_5_native_attention_backward: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 only; requires qwen3_5_native_attention. Use the "
            "native NVFP4 autograd attention path while training. This is validated "
            "for convergence but can be slower than bf16 at short sequence lengths, "
            "so it stays explicitly opt-in."
        },
    )
    qwen3_5_native_attention_save_backward_packs: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 native attention training only. Save extra NVFP4 "
            "Q/K/V layouts from forward so backward can skip repacking them. This "
            "improves backward-only timing but can move cost into forward, so it is "
            "experimental and OFF by default."
        },
    )
    qwen3_5_native_attention_backward_dv_p_rtn: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 native attention training only. Use deterministic "
            "round-to-nearest for the softmax-probability FP4 pack that feeds dV, "
            "while leaving the routing dS path governed by stochastic_rounding. "
            "Experimental and OFF by default."
        },
    )
    qwen3_5_native_attention_backward_dv_dot_rtn: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 native attention training only. Use deterministic "
            "round-to-nearest for the transposed dO FP4 pack used only by dV. "
            "Experimental and OFF by default."
        },
    )
    qwen3_5_native_attention_backward_dq_ds_rtn: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 native attention training only. Use deterministic "
            "round-to-nearest for the dS FP4 pack consumed by the dQ pass, while "
            "leaving the dK routing-gradient dS pack governed by stochastic_rounding. "
            "Experimental and OFF by default."
        },
    )
    qwen3_5_native_attention_backward_dkdv_scratch_bf16: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 native attention training only. Store the "
            "per-query-head dK/dV scratch buffer in bf16 before GQA reduction. "
            "This reduces backward HBM traffic at a small gradient-precision cost. "
            "Experimental and OFF by default."
        },
    )
    qwen3_5_native_attention_compile_custom_op: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 native attention inference only. Route the "
            "packed NVFP4 flash-attention call through an opaque torch custom op "
            "so torch.compile can keep surrounding producer/epilogue ops in graph "
            "without recompiling the internal Triton tl.dot_scaled kernel. OFF by "
            "default."
        },
    )
    qwen3_5_native_attention_layer_autograd: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 native attention training only. Route supported "
            "plain-linear full-attention layers through an experimental wider "
            "custom-autograd boundary. This is a scaffold for future backward "
            "fusion and is OFF by default."
        },
    )
    qwen3_5_fuse_vproj: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 native attention only. Run v_proj as a native "
            "NVFP4 GEMM with key-axis pack epilogue on inference/cache-free prefill. "
            "Skipped automatically for adapter-wrapped v_proj modules."
        },
    )
    qwen3_5_native_linear_attn: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 only. Patch GatedDeltaNet's large projection GEMMs "
            "to native NVFP4 in no-grad forward/eval. Training forwards fall back to "
            "the original implementation."
        },
    )
    qwen3_5_native_mlp: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5 only. Patch dense SwiGLU MLP GEMMs to native NVFP4 "
            "in no-grad forward/eval. Training forwards fall back to the original "
            "implementation."
        },
    )
