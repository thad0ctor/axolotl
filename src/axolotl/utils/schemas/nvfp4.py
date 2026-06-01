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
            "(== NVFP4-QLoRA): base packed in FP4, ~2.6x memory, backward dequants to "
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
    skip_first_n_blocks: int = Field(
        default=0,
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
        json_schema_extra={
            "description": "Keep the last N transformer blocks in high precision. "
            "See skip_first_n_blocks for the ~15% high-precision policy from the "
            "NVFP4 training paper (the tail blocks matter most)."
        },
    )
