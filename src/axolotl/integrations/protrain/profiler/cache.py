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

# Bump when the ProfilerTrace schema changes in a way that invalidates existing
# cached traces. Version 2 adds per-op wall-clock latencies (``op_latencies``);
# version 3 adds measured Adam throughputs (``cpu_adam_bytes_per_sec`` /
# ``gpu_adam_bytes_per_sec``) — traces from v2 have 0.0 for those fields, so
# the runtime cost model would fall back to the hardcoded prior. Bumping the
# version forces a re-profile rather than silently degrading accuracy.
# Version 4 adds hook-dispatch calibration fields (``hooked_fwd_wall_s`` /
# ``steady_fwd_wall_s`` / ``steady_bwd_wall_s``) that the cost model consumes
# to scale the hooked per-op latencies down to a steady-state prior. v3
# traces default those fields to 0.0 which would make the cost model fall
# back to identity scale and regress 7B runtime error to its pre-calibration
# level; bumping forces a fresh trace.
# Version 5 adds an aggregate ``steady_fwd_peak_bytes`` cap used by the
# memory cost model when the searcher picks all-NONE.
# Version 6 adds per-block peaks (``steady_fwd_block_peak_bytes``) captured
# during the hook-less steady forward via lightweight block-level hooks.
# Unlike the v5 aggregate — which only applies when n_checkpoint=0 &&
# n_swap=0 — the per-block max bounds the forward peak for any fractional-
# NONE config, tightening over-prediction across the search space. v5
# traces default the per-block dict to empty, so the cost model falls back
# to the aggregate-only cap (identical v5 behavior); bumping forces a fresh
# trace so the cap takes effect.
# Version 7 changes the steady-state measurement methodology from a single
# iteration to a 4-iter hot loop (2 warmup + 2 measured, median of measured)
# and adds a best-effort steady_bwd_wall_s in the same loop. The recorded
# fields are unchanged but the *values* shift (single-iter carried allocator-
# settle cost the multi-iter median eliminates), so the cost model's measured
# bwd/fwd ratio path requires a fresh trace under the new methodology.
# Version 8 makes ``world`` and the NCCL collective tables real for
# world_size > 1: ``measure_nccl(world_size>1)`` now actually runs
# all_gather_into_tensor / reduce_scatter_tensor sweeps over a payload-size
# grid instead of raising NotImplementedError, and ``run_trace`` plumbs
# ``cfg.world_size`` (or auto-detects from the live process group) into
# both the trace's ``world`` field and the per-payload tables. Single-rank
# traces are unaffected (collective tables stay empty); multi-rank traces
# captured under v7 had ``world=1`` hard-coded and must be re-run.
# Version 9 folds ``requires_grad`` into the arch_hash so that toggling
# freeze-layer config invalidates the cache. Previously a v8 trace
# captured under one freezing pattern would replay against a different
# freezing pattern with the same arch, returning stale
# ``trainable_param_fraction`` / ``model_state_bytes`` and steering the
# cost model into the wrong bwd/fwd-ratio fallback. v8 traces remain on
# disk but never look up under v9 keys.
# Version 10 adds phase-2 chunked-runtime backward fields:
# ``steady_bwd_chunked_wall_s``, ``steady_step_overlap_s``,
# ``phase2_n_checkpoint``, ``phase2_per_block_recompute_s``. These are
# populated by the bootstrap-then-measure loop in
# ``protrain_model_wrapper`` and consumed by ``cost/runtime.py`` to
# translate a measured chunked backward to any candidate ``block_map``
# the search evaluates. v9 traces lack these fields and would steer
# the cost model into the v8 fallback path; bumping invalidates them
# so the next run captures a real chunked backward measurement.
TRACE_VERSION = 10


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
        """Deterministic 64-char sha256 hex digest used as the on-disk filename.

        The ``TRACE_VERSION`` prefix ensures a schema bump invalidates all prior
        cache entries — old files stay on disk but are never looked up.
        """
        raw = f"v{TRACE_VERSION}|{self.arch_hash}|{self.bs}|{self.seq}|{self.sku}|{self.world}"
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
