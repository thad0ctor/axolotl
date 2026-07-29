# Vendored from https://github.com/z-lab/sparselora @ a2fd69de93b1168080346ec113c99501f0bb58b1 (MIT). Local edit: absolute 'sparselora.*' imports relativized. Do not edit; see _vendor/PROVENANCE.md.
from .config import SparseLoRAConfig
from .api import apply_sparselora
from .modules import register_sparse_module

__all__ = ["SparseLoRAConfig", "apply_sparselora", "register_sparse_module"]
