"""Self-calibration of the SparseLoRA per-layer sparsity schedule.

Calibration runs on a small slice of the *same* dataset and config the run will
train on, so the schedule is tailored to the actual model + data instead of
relying on z-lab's pre-published Llama-only presets.

Method (``faithful`` / ``proxy``):

1. (``faithful`` only) a short dense LoRA warm-up so captured activations reflect
   a partially-trained model -- the paper's "start from a fine-tuned reference".
2. A per-layer sensitivity sweep: for each target MLP/attention module and a grid
   of candidate sparsities, measure the relative reconstruction error of skipping
   the predictor-selected channels.
3. Greedy allocation: each layer takes the highest sparsity whose error stays
   within ``loss_budget``; the schedule is then capped toward ``target_sparsity``.

The selection math here mirrors the vendored predictors but is reimplemented in
plain torch (no Triton/liger dependency) so calibration runs on CPU or GPU. The
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


def allocate_schedule(
    errors: dict[str, dict[float, float]],
    target_sparsity: float,
    loss_budget: float,
) -> dict[str, float]:
    """Per-layer sparsity: highest grid value within budget, capped toward target.

    A layer whose smallest grid sparsity already exceeds ``loss_budget`` stays
    dense (omitted from the schedule), which naturally keeps the most sensitive
    layers (often the earliest) at full precision.
    """
    within_budget: dict[str, float] = {}
    for name, grid in errors.items():
        ok = [s for s in SPARSITY_GRID if grid[s] <= loss_budget]
        if ok:
            within_budget[name] = max(ok)

    if not within_budget:
        LOG.warning(
            "SparseLoRA: no layer met loss_budget=%s; schedule is empty (dense training). "
            "Raise calibration.loss_budget or lower target_sparsity.",
            loss_budget,
        )
        return {}

    total = len(errors)
    mean = sum(within_budget.values()) / total
    if mean <= target_sparsity:
        if mean < target_sparsity:
            LOG.warning(
                "SparseLoRA: target_sparsity=%.3f unreachable within loss_budget=%s; "
                "using the budget-respecting schedule (mean sparsity %.3f).",
                target_sparsity,
                loss_budget,
                mean,
            )
        return within_budget

    # Mean exceeds target: cap layers so the average lands near target.
    best, best_gap = within_budget, abs(mean - target_sparsity)
    for cap in sorted(SPARSITY_GRID):
        capped = {n: min(s, cap) for n, s in within_budget.items() if min(s, cap) > 0}
        capped_mean = sum(capped.values()) / total
        gap = abs(capped_mean - target_sparsity)
        if gap < best_gap:
            best, best_gap = capped, gap
    return best


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

    # Calibration must be side-effect-free on the training run's RNG so the
    # subsequent data order / init is identical with or without the plugin.
    # Capture before any loader/forward work consumes randomness.
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    target_names = discover_target_modules(model)
    loader = build_calibration_loader(trainer, cal.num_samples, cal.batch_size)
    method = method_name(cal.method)

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
    max_batches = -(-cal.num_samples // cal.batch_size)  # ceil
    try:
        errors = run_sensitivity(
            model, target_names, factors, loader, device, max_batches=max_batches
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

    schedule = allocate_schedule(errors, settings.target_sparsity, cal.loss_budget)
    mean = sum(schedule.values()) / max(1, len(target_names))
    LOG.info(
        "SparseLoRA: calibrated schedule covers %d/%d modules, mean sparsity %.3f.",
        len(schedule),
        len(target_names),
        mean,
    )
    return schedule
