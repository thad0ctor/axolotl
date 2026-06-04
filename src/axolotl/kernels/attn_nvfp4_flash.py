"""Re-export shim for native-NVFP4 flash attention.

The kernel implementation now lives in the standalone ``sageattention.nvfp4``
fork (a SageAttention fork). This module preserves the historical
``axolotl.kernels.attn_nvfp4_flash`` import path so every existing importer
(custom_op, monkeypatch, tests, scripts) keeps working unchanged.
"""

from __future__ import annotations

from sageattention.nvfp4 import (  # noqa: F401
    _NVFP4FlashAttn,
    _gqa_reduce_cast_dkdv,
    _next_mult,
    _quant_nvfp4,
    _quant_nvfp4_dual,
    _resolve_fwd_tiles,
    _run_bwd,
    _run_flash_packed,
    convert_fp32_to_fp4_packed,
    nvfp4_flash_attention,
    nvfp4_flash_attention_packed,
    nvfp4_flash_attn_func,
)
