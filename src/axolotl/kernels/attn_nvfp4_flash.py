"""Re-export shim for native-NVFP4 flash attention.

The kernel implementation now lives in the standalone ``sageattention.nvfp4``
fork (a SageAttention fork). This module preserves the historical
``axolotl.kernels.attn_nvfp4_flash`` import path so every existing importer
(custom_op, monkeypatch, tests, scripts) keeps working unchanged.
"""

from __future__ import annotations

from sageattention.nvfp4 import (  # noqa: F401
    _gqa_reduce_cast_dkdv,
    _next_mult,
    _NVFP4FlashAttn,
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

try:
    # HP-grad-dots backward (FP4 S/dP recomputes + bf16 grad GEMMs). Newer forks
    # only; ``None`` on a legacy fork makes callers fall back to the all-FP4
    # ``_run_bwd`` instead of breaking every attn_nvfp4 import.
    from sageattention.nvfp4.flash import _run_bwd_hp  # noqa: F401
except ImportError:  # pragma: no cover - legacy sageattention fork
    _run_bwd_hp = None

try:
    # Varlen (packed-sequence cu_seqlens) seq-array expansion. Newer forks only;
    # ``None`` disables the packed varlen attention paths (callers fall back to
    # the model's original forward for packed batches).
    from sageattention.nvfp4.flash import _varlen_seq_arrays  # noqa: F401
except ImportError:  # pragma: no cover - legacy sageattention fork
    _varlen_seq_arrays = None
