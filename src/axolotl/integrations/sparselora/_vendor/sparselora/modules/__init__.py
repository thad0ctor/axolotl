# Vendored from https://github.com/z-lab/sparselora @ a2fd69de93b1168080346ec113c99501f0bb58b1 (MIT). Local edit: absolute 'sparselora.*' imports relativized. Do not edit; see _vendor/PROVENANCE.md.
from .base import SparseModule
from .linear import SparseLinear, lora_forward
from .llama import SparseLlamaAttention, SparseLlamaMLP
from .registry import get_module_mapping, get_sparsity_mode, register_sparse_module
