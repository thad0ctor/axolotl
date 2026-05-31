"""NVFP4-GEMM training config schema.

Real low-precision COMPUTE (FP4 forward/backward GEMMs on Blackwell), distinct
from the fake-quant QAT/PTQ `quantization:` block.
"""

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
    exclude_modules: list[str] = Field(
        default_factory=lambda: ["lm_head", "embed_tokens"],
        json_schema_extra={
            "description": "Module name fragments kept in high precision (not swapped)."
        },
    )
    skip_first_n_blocks: int = Field(
        default=0,
        json_schema_extra={
            "description": "Keep the first N transformer blocks in high precision."
        },
    )
    skip_last_n_blocks: int = Field(
        default=0,
        json_schema_extra={
            "description": "Keep the last N transformer blocks in high precision."
        },
    )
