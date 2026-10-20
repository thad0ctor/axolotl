# Vendored from MegaTrain: https://github.com/DLYuanGod/MegaTrain
# Revision: 7f5c9597e5b20bb618932c77c922e8eac4a11c4d (Apache-2.0)
# Modified by Axolotl; see _vendor/PROVENANCE.md for the list of changes.

"""MegaTrain: RAM-Centric Architecture for LLM/VLM Training.

Supports any HuggingFace decoder-only model and VLM:
- Dense models: Llama 2/3/4, Qwen 2/3, Mistral, Phi, Gemma, etc.
- Hybrid attention: Qwen 3.5 (linear + full attention)
- MoE models: Mixtral, DeepSeek-MoE, Qwen3-Next
- VLM: Qwen2-VL, Qwen3-VL, LLaVA, Gemma3-VL, InternVL, GLM-4V, etc.
"""

from .model.cpu_master import CPUMasterModel
from .config.training import CPUMasterConfig

__version__ = "0.3.0"

__all__ = [
    "CPUMasterModel",
    "CPUMasterConfig",
]
