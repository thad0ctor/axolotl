"""On-disk cache for ProfilerTrace, keyed by (arch_hash, bs, seq, sku, world)."""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from axolotl.utils.logging import get_logger

from axolotl.integrations.protrain.types import ProfilerTrace

LOG = get_logger(__name__)

_CACHE_SUBDIR = Path("protrain") / "profiler"


@dataclass(frozen=True)
class ProfilerCacheKey:
    """Identity of a cached trace (§7 re-profile trigger).

    Not defined in ``types.py`` by design — cache keys are an implementation
    detail of this subpackage and shouldn't leak into the public plugin API.
    """

    arch_hash: str
    bs: int
    seq: int
    sku: str
    world: int

    def fingerprint(self) -> str:
        """Deterministic 64-char sha256 hex digest used as the on-disk filename."""
        raw = f"{self.arch_hash}|{self.bs}|{self.seq}|{self.sku}|{self.world}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_root() -> Path:
    """Resolve ``$XDG_CACHE_HOME/protrain/profiler`` or ``~/.cache/protrain/profiler``."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / _CACHE_SUBDIR


def _path_for(key: ProfilerCacheKey) -> Path:
    return _cache_root() / f"{key.fingerprint()}.pkl"


def load_cached_trace(key: ProfilerCacheKey) -> ProfilerTrace | None:
    """Load a previously-saved trace, or ``None`` if the key misses."""
    path = _path_for(key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            trace = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        LOG.warning("profiler cache miss due to read error at %s: %s", path, exc)
        return None
    if not isinstance(trace, ProfilerTrace):
        LOG.warning("profiler cache at %s is not a ProfilerTrace (got %s)", path, type(trace))
        return None
    return trace


def save_cached_trace(key: ProfilerCacheKey, trace: ProfilerTrace) -> Path:
    """Persist ``trace`` under ``key``. Returns the on-disk path."""
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _path_for(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(trace, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    LOG.debug("saved profiler trace to %s", path)
    return path


__all__ = [
    "ProfilerCacheKey",
    "load_cached_trace",
    "save_cached_trace",
]
