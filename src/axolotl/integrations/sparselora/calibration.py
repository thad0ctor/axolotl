"""Self-calibration of the SparseLoRA per-layer sparsity schedule.

z-lab derive their published schedules from an offline *downstream-task*
sensitivity analysis (progressively sparsifying each layer and measuring task
accuracy). That process is expensive and not reproducible from a cheap forward
pass; crucially, their schedules run MLP layers at 0.97-0.99 sparsity, whose
per-block reconstruction error is ~0.9 -- so an absolute reconstruction-error
budget cannot recover their schedule (and empirically returns near-zero
sparsity, which is the bug this module fixes). Reconstruction error also
*anti-correlates* with their layer choice: shallow layers reconstruct better yet
z-lab keep them dense.

The methods here therefore key off z-lab's *empirical structural findings* rather
than reconstruction error:

- ``preset``: load z-lab's downstream-validated published schedule
  (``SparseLoRAConfig.from_pretrained(mode="o1"|"o2")``). The recommended path
  for the Llama models z-lab calibrated.
- ``structural`` (default): apply z-lab's profile from layer structure alone --
  dense shallow + final layers, aggressive deep MLP (``target_sparsity``), milder
  attention (``attn_sparsity``) over a still-deeper band. No forward pass.
- ``faithful`` / ``proxy``: start from the ``structural`` band and use a
  reconstruction-sensitivity sweep to *demote* the most sensitive band layers
  (per group) back to dense; ``faithful`` adds a short dense LoRA warm-up first.

The sweep math mirrors the vendored predictors but is reimplemented in plain
torch (no Triton/liger dependency) so calibration runs on CPU or GPU. The
*train-time* predictors are still the vendored liger kernels applied via
``apply_sparselora``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from axolotl.utils.logging import get_logger

from .factors import (
    attn_projection_weights,
    is_attn_module,
    is_fused_attn_module,
    is_fused_mlp_module,
    is_mlp_module,
    mlp_projection_weights,
)

LOG = get_logger(__name__)

SPARSITY_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
_EPS = 1e-6


def method_name(method: Any) -> str:
    """Normalize the calibration method to a plain string (enum or str)."""
    return method.value if hasattr(method, "value") else str(method)


def _is_sparsifiable(module: nn.Module) -> bool:
    return (
        is_mlp_module(module)
        or is_attn_module(module)
        or is_fused_mlp_module(module)
        or is_fused_attn_module(module)
    )


def discover_target_modules(model: nn.Module) -> list[str]:
    """Full module paths of every sparsifiable MLP / attention block.

    Covers both the separate-projection layout (gate/up + q/k/v) and the
    Phi3-style fused layout (gate_up_proj + qkv_proj).
    """
    names = []
    for name, module in model.named_modules():
        if _is_sparsifiable(module):
            names.append(name)
    return names


def build_calibration_loader(
    trainer: Any, num_samples: int, batch_size: int
) -> DataLoader:
    """A small DataLoader over the first ``num_samples`` training examples.

    Reuses the trainer's own tokenized dataset and collator, guaranteeing the
    calibration data is identical to what will be trained on.
    """
    dataset = trainer.train_dataset
    n = min(num_samples, len(dataset))
    if n <= 0:
        raise ValueError(
            "SparseLoRA calibration has no data: the training dataset and "
            "calibration.num_samples must both be non-empty."
        )
    if hasattr(dataset, "select"):
        subset = dataset.select(range(n))
    else:
        LOG.warning(
            "SparseLoRA: train_dataset has no .select(); calibration will read at "
            "most num_samples via a batch cap instead of pre-slicing."
        )
        subset = dataset
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=trainer.data_collator,
    )


def _topk_indices(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    k = max(1, int(scores.shape[-1] * (1.0 - sparsity)))
    return scores.topk(k).indices


@torch.no_grad()
def _ffn_recon_error(
    x: torch.Tensor,
    module: nn.Module,
    factors: dict[str, torch.Tensor],
    name: str,
    sparsity: float,
) -> float:
    """Relative L2 error of the MLP block when only predicted channels are kept."""
    weights = mlp_projection_weights(module)
    wg, wu, wd = weights["gate_proj"], weights["up_proj"], weights["down_proj"]

    # Predicted channel scores from the rank-r factors (mirrors FFNPredictor).
    xm = x.mean(dim=1).to(wg.dtype)  # (B, H)
    gate = (xm @ factors[f"{name}.gate_proj.w1"]) @ factors[f"{name}.gate_proj.w2"]
    up = (xm @ factors[f"{name}.up_proj.w1"]) @ factors[f"{name}.up_proj.w2"]
    scores = (F.silu(gate) * up).norm(dim=0)  # (I,)
    idx = _topk_indices(scores, sparsity)

    dense = F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)
    sparse = F.linear(F.silu(F.linear(x, wg[idx])) * F.linear(x, wu[idx]), wd[:, idx])
    return (sparse - dense).float().norm().item() / (dense.float().norm().item() + _EPS)


@torch.no_grad()
def _attn_recon_error(x: torch.Tensor, module: nn.Module, sparsity: float) -> float:
    """Mean relative dropped-channel energy across q/k/v projections.

    A cheap proxy for attention sensitivity: keeping the highest-energy
    ``(1 - sparsity)`` projection channels, how much output energy is lost.
    """
    errs = []
    for weight in attn_projection_weights(module).values():
        out = F.linear(x, weight).float()  # (B, T, D)
        energy = out.pow(2).sum(dim=(0, 1))  # (D,)
        total = energy.sum()
        kept = energy[_topk_indices(energy, sparsity)].sum()
        errs.append((1.0 - (kept / (total + _EPS))).clamp_min(0.0).sqrt().item())
    return sum(errs) / len(errs)


@torch.no_grad()
def run_sensitivity(
    model: nn.Module,
    target_names: list[str],
    factors: dict[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, dict[float, float]]:
    """Average reconstruction error per (module, candidate sparsity) over the loader.

    ``max_batches`` caps iteration so calibration cost stays bounded even if the
    dataset could not be pre-sliced to ``num_samples``.
    """
    import itertools

    if max_batches is not None:
        loader = itertools.islice(loader, max_batches)
    modules = dict(model.named_modules())
    targets = {n: modules[n] for n in target_names}
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_mod, args, kwargs):
            captured[name] = args[0] if args else kwargs["hidden_states"]

        return hook

    for name, module in targets.items():
        handles.append(
            module.register_forward_pre_hook(make_hook(name), with_kwargs=True)
        )

    totals: dict[str, dict[float, float]] = {
        n: {s: 0.0 for s in SPARSITY_GRID} for n in target_names
    }
    n_batches = 0
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            captured.clear()
            inputs = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            model(**inputs)
            for name, module in targets.items():
                x = captured.get(name)
                if x is None:
                    continue
                is_mlp = is_mlp_module(module) or is_fused_mlp_module(module)
                for s in SPARSITY_GRID:
                    if is_mlp:
                        err = _ffn_recon_error(x, module, factors, name, s)
                    else:
                        err = _attn_recon_error(x, module, s)
                    totals[name][s] += err
            n_batches += 1
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    n_batches = max(1, n_batches)
    return {
        n: {s: e / n_batches for s, e in grid.items()} for n, grid in totals.items()
    }


def resolve_attn_sparsity(target_sparsity: float, attn_sparsity: float | None) -> float:
    """Attention base-path sparsity: explicit override, else min(target, 0.75).

    Attention tolerates far less sparsity than the MLP -- z-lab's o2 schedule runs
    attention at 0.75 while the MLP runs at 0.99 -- so the default caps it at 0.75.
    """
    if attn_sparsity is not None:
        return float(attn_sparsity)
    return min(float(target_sparsity), 0.75)


def _layer_index(name: str) -> int | None:
    """First integer path component (e.g. ``...layers.3.mlp`` -> 3), or None."""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def _dense_prefix_count(prefix: float, n_layers: int) -> int:
    """A fraction (<=1) of depth or an absolute leading-layer count."""
    if prefix <= 1.0:
        return round(prefix * n_layers)
    return int(prefix)


def allocate_structural(
    model: nn.Module,
    target_names: list[str],
    target_sparsity: float,
    attn_sparsity: float,
    dense_prefix: float,
    attn_dense_prefix: float,
) -> dict[str, float]:
    """z-lab-shaped positional schedule from layer structure alone (no forward pass).

    Mirrors the empirical structure of z-lab's published Llama schedules rather
    than a per-layer reconstruction budget (which was found not to track
    downstream task sensitivity -- z-lab runs MLP layers at 0.97-0.99 sparsity,
    whose block reconstruction error is ~0.9). The profile keeps shallow layers
    and the final layer dense (least amenable to sparsification), sparsifies the
    MLP aggressively across the deep band, and sparsifies attention more mildly
    over a still-deeper band.
    """
    modules = dict(model.named_modules())
    indices = [i for n in target_names if (i := _layer_index(n)) is not None]
    if not indices:
        LOG.warning(
            "SparseLoRA: no numeric layer index found on any target module; the "
            "structural profile needs `model.layers.N....`-style names. Schedule empty."
        )
        return {}
    n_layers = max(indices) + 1
    last = n_layers - 1
    mlp_prefix = _dense_prefix_count(dense_prefix, n_layers)
    attn_prefix = _dense_prefix_count(attn_dense_prefix, n_layers)

    schedule: dict[str, float] = {}
    for name in target_names:
        idx = _layer_index(name)
        if idx is None or idx == last:
            continue
        module = modules[name]
        is_mlp = is_mlp_module(module) or is_fused_mlp_module(module)
        prefix = mlp_prefix if is_mlp else attn_prefix
        if idx < prefix:
            continue
        sparsity = target_sparsity if is_mlp else attn_sparsity
        if sparsity > 0.0:
            schedule[name] = float(sparsity)
    return schedule


def refine_with_sensitivity(
    model: nn.Module,
    base_schedule: dict[str, float],
    errors: dict[str, dict[float, float]],
    demote_frac: float,
) -> dict[str, float]:
    """Demote the most sensitive band layers (per group) back to dense.

    The structural profile decides the band; the reconstruction sweep is used
    only as a *relative* within-group signal to drop the most sensitive layers
    (highest error at their assigned sparsity) to dense. This keeps the schedule
    non-empty and z-lab-shaped while letting the data trim genuine outliers.
    """
    if demote_frac <= 0.0 or not base_schedule:
        return base_schedule
    modules = dict(model.named_modules())
    groups: dict[bool, list[str]] = {True: [], False: []}
    for name in base_schedule:
        module = modules[name]
        is_mlp = is_mlp_module(module) or is_fused_mlp_module(module)
        groups[is_mlp].append(name)

    def _err_at(name: str) -> float:
        grid = errors.get(name, {})
        if not grid:
            return 0.0
        assigned = base_schedule[name]
        nearest = min(grid, key=lambda s: abs(s - assigned))
        return grid[nearest]

    demoted: set[str] = set()
    for names in groups.values():
        if len(names) <= 1:
            continue
        n_demote = min(len(names) - 1, int(len(names) * demote_frac))
        if n_demote <= 0:
            continue
        ranked = sorted(names, key=_err_at, reverse=True)
        demoted.update(ranked[:n_demote])
    return {n: s for n, s in base_schedule.items() if n not in demoted}


def load_preset_schedule(
    preset: str, mode: str, target_names: list[str]
) -> dict[str, float]:
    """Load a published z-lab schedule and map its keys onto discovered modules.

    ``preset`` is a z-lab-format SparseLoRA repo id or local dir whose
    ``config.json`` holds ``modes`` (``o1``/``o2``). This is z-lab's
    downstream-validated path for the Llama models they calibrated.
    """
    from ._vendor.sparselora import SparseLoRAConfig

    config = SparseLoRAConfig.from_pretrained(preset, mode=mode)
    schedule: dict[str, float] = {}
    unmatched: list[str] = []
    for key, val in config.layer_sparsity.items():
        if not val or float(val) <= 0.0:
            continue
        matches = [n for n in target_names if n == key or n.endswith("." + key)]
        if not matches:
            unmatched.append(key)
        for n in matches:
            schedule[n] = float(val)
    if unmatched:
        LOG.warning(
            "SparseLoRA: preset %s (mode=%s) has %d sparse keys with no matching "
            "module in this model (e.g. %s); they were skipped. Is the preset for a "
            "different architecture?",
            preset,
            mode,
            len(unmatched),
            unmatched[:3],
        )
    if not schedule:
        raise ValueError(
            f"SparseLoRA preset {preset!r} (mode={mode!r}) matched no sparsifiable "
            "module in this model. The preset is likely for a different architecture."
        )
    return schedule


def warmup_lora(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    steps: int,
) -> bool:
    """Short dense LoRA fine-tune so calibration sees realistic activations.

    Returns True if the warm-up ran cleanly (or was a no-op), False if it raised
    after mutating weights — the caller restores the clean snapshot in that case.
    """
    if steps <= 0:
        return True
    params = [
        p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n
    ]
    if not params:
        LOG.warning(
            "SparseLoRA: faithful warm-up found no trainable LoRA params; skipping."
        )
        return True
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    was_training = model.training
    model.train()
    done = 0
    completed = True
    try:
        while done < steps:
            for batch in loader:
                inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                out = model(**inputs)
                loss = out.loss if hasattr(out, "loss") else out["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                done += 1
                if done >= steps:
                    break
        LOG.info(
            "SparseLoRA: completed %d warm-up step(s) for faithful calibration.", done
        )
    except Exception as exc:  # noqa: BLE001 - warm-up is best-effort
        LOG.warning(
            "SparseLoRA: faithful warm-up failed (%s); proceeding without it.", exc
        )
        completed = False
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.train(was_training)
    return completed


def calibrate(
    cfg: Any,
    model: nn.Module,
    factors: dict[str, torch.Tensor],
    trainer: Any,
) -> dict[str, float]:
    """Produce the per-layer sparsity schedule for the active calibration method."""
    settings = cfg.sparselora
    cal = settings.calibration
    device = next(model.parameters()).device

    target_names = discover_target_modules(model)
    method = method_name(cal.method)
    attn_sparsity = resolve_attn_sparsity(
        settings.target_sparsity, settings.attn_sparsity
    )

    if method == "preset":
        schedule = load_preset_schedule(
            settings.preset, settings.preset_mode, target_names
        )
        LOG.info(
            "SparseLoRA: loaded preset %s (mode=%s): %d/%d modules sparse.",
            settings.preset,
            settings.preset_mode,
            len(schedule),
            len(target_names),
        )
        return schedule

    # Fall back to the schema defaults for partially-specified (non-pydantic) configs.
    dense_prefix = cal.dense_prefix if cal.dense_prefix is not None else 0.1
    attn_dense_prefix = (
        cal.attn_dense_prefix if cal.attn_dense_prefix is not None else 0.45
    )
    demote_frac = cal.sensitivity_demote if cal.sensitivity_demote is not None else 0.25

    base = allocate_structural(
        model,
        target_names,
        settings.target_sparsity,
        attn_sparsity,
        dense_prefix,
        attn_dense_prefix,
    )

    if method == "structural":
        _log_schedule(base, target_names, "structural")
        return base

    # faithful / proxy: refine the structural band with a data-driven sweep that
    # demotes the most sensitive layers per group back to dense.
    # Calibration must be side-effect-free on the training run's RNG so the
    # subsequent data order / init is identical with or without the plugin.
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    loader = build_calibration_loader(trainer, cal.num_samples, cal.batch_size)

    snapshot = None
    if method == "faithful" and cal.warmup_steps > 0:
        snapshot = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad and "lora_" in n
        }
        warmup_lr = float(cfg.learning_rate) if cfg.learning_rate else 1e-4
        completed = warmup_lora(model, loader, device, warmup_lr, cal.warmup_steps)
        if not completed:
            # Warm-up raised mid-way; restore clean weights so the sensitivity
            # sweep measures a defined state, not a half-updated one.
            with torch.no_grad():
                params = dict(model.named_parameters())
                for nm, val in snapshot.items():
                    params[nm].copy_(val)
            snapshot = None

    LOG.info(
        "SparseLoRA: calibrating %d target modules (method=%s, target_sparsity=%.2f).",
        len(target_names),
        method,
        settings.target_sparsity,
    )
    # Only the band layers need a sensitivity score; sweep those to keep it cheap.
    band = list(base)
    max_batches = -(-cal.num_samples // cal.batch_size)  # ceil
    try:
        errors = run_sensitivity(
            model, band, factors, loader, device, max_batches=max_batches
        )
    finally:
        if snapshot is not None:
            # Restore LoRA weights so real training starts from a clean init,
            # even if the sweep raised mid-way.
            with torch.no_grad():
                params = dict(model.named_parameters())
                for n, val in snapshot.items():
                    params[n].copy_(val)
        # Restore RNG so calibration does not perturb the training data order.
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    schedule = refine_with_sensitivity(model, base, errors, demote_frac)
    _log_schedule(schedule, target_names, method)
    return schedule


def _log_schedule(
    schedule: dict[str, float], target_names: list[str], method: str
) -> None:
    total = max(1, len(target_names))
    mean = sum(schedule.values()) / total
    LOG.info(
        "SparseLoRA: %s schedule covers %d/%d modules, mean sparsity %.3f.",
        method,
        len(schedule),
        len(target_names),
        mean,
    )
