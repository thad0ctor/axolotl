"""G7 — cross-variant numerical consistency (the flagship gate).

Kernels (liger CE / liger RMSNorm / cut-cross-entropy / fused attention) are
SPEED swaps, not MATH changes: with identical weights, data and seed, the STEP-0
(pre-update) training loss with a kernel ON must equal the baseline step-0 loss
within ``rtol``. A cell that exceeds tol is a silent numerical regression — the
liger-0.8.0 *shadow* failure mode, where a kernel routes through a wrong generic
path and "did it train" CI still goes green. G7 names the kernel and the delta.

Determinism (the load-bearing part):

  * Each variant trains in a FRESH subprocess (reusing G6's ``_spawn_variant`` /
    shared worker), so the GLOBAL, never-reverted liger/CCE monkeypatches of one
    variant cannot leak into another — the only difference between two runs is the
    kernel under test.
  * Every variant gets the SAME ``seed`` and the SAME dataset cfg, so axolotl
    tokenizes identically and the seeded sampler draws the SAME first batch; the
    weights are the loaded pretrained checkpoint (no random init — full finetune),
    so no RNG diverges across variants.
  * step-0 loss = first ``loss_history`` entry from a ``max_steps=1``,
    ``logging_steps=1`` run: the loss logged after step 1 is the forward loss on
    batch 0 computed with the initial weights, i.e. pre-update.

We do NOT force ``torch.use_deterministic_algorithms`` (some ops lack a
deterministic kernel and would raise); ``rtol`` is loose enough that
run-to-run float noise is well inside it. bf16 variants (CCE) legitimately differ
more, so they use a wider ``rtol_bf16``.
"""

from __future__ import annotations

from typing import Any

from .. import GateContext, GateResult, GateStatus
from . import g6_loss

GATE_ID = "G7"
GATE_NAME = "consistency"

_LIGER_PLUGIN = "axolotl.integrations.liger.LigerPlugin"
_CCE_PLUGIN = "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"


def applies(ctx: GateContext) -> bool:
    # A single forward (step-0 loss) is cheap; the kernel comparison needs a GPU
    # (triton), handled in run() by degrading to SKIPPED when no kernel applies.
    return True


# --- applicability per model type (mirrors G2's registry cross-checks) ---------


def _liger_types() -> set[str]:
    try:
        from liger_kernel.transformers.monkey_patch import (
            MODEL_TYPE_TO_APPLY_LIGER_FN,
        )

        return set(MODEL_TYPE_TO_APPLY_LIGER_FN)
    except Exception:  # noqa: BLE001
        return set()


def _cce_types() -> set[str]:
    try:
        from cut_cross_entropy.transformers.patch import PATCH_FNS

        return set(PATCH_FNS)
    except Exception:  # noqa: BLE001
        return set()


def _fused_attn_types() -> set[str]:
    try:
        from axolotl.loaders.patch_manager import PatchManager

        return set(PatchManager._FUSED_ATTN_KERNEL_SUPPORTED)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return set()


def _kernel_variants(ctx: GateContext) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (variants, skip-notes). A variant is included only when the registry
    says the kernel is applicable to this model_config_type; bf16/fused variants
    carry a looser tol flag. All kernels are triton -> require a GPU."""
    mct = ctx.model_config_type
    variants: list[dict[str, Any]] = []
    notes: list[str] = []

    if not ctx.gpu_available:
        notes.append("no GPU: all kernel variants (triton) skipped")
        return variants, notes

    liger = _liger_types()
    if mct in liger:
        variants.append(
            {
                "name": "liger_cross_entropy",
                "flags": {"plugins": [_LIGER_PLUGIN], "liger_cross_entropy": True},
                "bf16": False,
            }
        )
        variants.append(
            {
                "name": "liger_rms_norm",
                "flags": {"plugins": [_LIGER_PLUGIN], "liger_rms_norm": True},
                "bf16": False,
            }
        )
    else:
        notes.append(
            f"liger: {mct} has no native liger support (generic FLCE) — skipped"
        )

    cce = _cce_types()
    if mct in cce:
        variants.append(
            {
                "name": "cut_cross_entropy",
                "flags": {
                    "plugins": [_CCE_PLUGIN],
                    "cut_cross_entropy": True,
                    "bf16": True,
                },
                "bf16": True,
            }
        )
    else:
        notes.append(f"cut_cross_entropy: {mct} not in CCE PATCH_FNS — skipped")

    fused = _fused_attn_types()
    if mct in fused:
        variants.append(
            {
                "name": "fused_attn_kernel",
                "flags": {
                    "fused_attn_kernel": True,
                    "attn_implementation": "flash_attention_2",
                    "bf16": True,
                },
                "bf16": True,
            }
        )
    else:
        notes.append(
            f"fused_attn_kernel: {mct} not in _FUSED_ATTN_KERNEL_SUPPORTED — skipped"
        )

    return variants, notes


def _build_cfg(ctx: GateContext, name: str, flags: dict[str, Any]) -> dict[str, Any]:
    """A 1-step cfg; flags override the eager/no-kernel baseline. Step-0 loss is
    the first logged loss."""
    cfg = g6_loss._build_cfg(ctx, f"consistency_{name}", {}, max_steps=1)
    # baseline forces eager; kernel variants may re-specify attn (e.g. fused).
    cfg.update(flags)
    return cfg


def _step0(res: dict[str, Any]) -> float | None:
    losses = res.get("loss_history") or []
    return float(losses[0]) if losses else None


def run(ctx: GateContext) -> GateResult:
    try:
        import axolotl.train  # noqa: F401
    except BaseException as err:  # noqa: BLE001
        return GateResult.could_not_run(
            GATE_ID, GATE_NAME, f"axolotl import failed: {type(err).__name__}: {err}"
        )

    rtol = float(ctx.options.get("rtol", 2e-2))
    rtol_bf16 = float(ctx.options.get("rtol_bf16", 5e-2))
    timeout = float(ctx.options.get("train_timeout", 1200))

    details: list[str] = []
    rows: list[dict[str, Any]] = []

    base_cfg = _build_cfg(ctx, "baseline", {})
    base_res = g6_loss._spawn_variant(ctx, "g7_baseline", base_cfg, timeout)
    base_loss = _step0(base_res)
    if base_loss is None:
        return GateResult.could_not_run(
            GATE_ID,
            GATE_NAME,
            f"baseline step-0 loss unavailable — {base_res.get('error', 'no loss logged')}",
        )
    details.append(f"baseline (eager, no kernels): step0 loss = {base_loss:.6f}")
    rows.append({"variant": "baseline", "step0": base_loss, "delta": 0.0, "tol": None})

    variants, skip_notes = _kernel_variants(ctx)
    for note in skip_notes:
        details.append(f"· {note}")

    if not variants:
        return GateResult(
            GATE_ID,
            GATE_NAME,
            GateStatus.SKIPPED,
            summary=(
                f"no applicable kernel variant to compare "
                f"(baseline step0={base_loss:.4f}); {'; '.join(skip_notes)}"
            ),
            details=details,
            data={"baseline_step0": base_loss, "variants": rows, "skipped": skip_notes},
        )

    findings = 0
    compared = 0
    for v in variants:
        name = v["name"]
        tol = rtol_bf16 if v["bf16"] else rtol
        cfg = _build_cfg(ctx, name, v["flags"])
        res = g6_loss._spawn_variant(ctx, f"g7_{name}", cfg, timeout)
        loss = _step0(res)
        if loss is None:
            details.append(f"⚠️ {name}: step-0 loss unavailable — {res.get('error')}")
            rows.append({"variant": name, "step0": None, "error": res.get("error")})
            continue

        compared += 1
        delta = abs(loss - base_loss)
        rel = delta / abs(base_loss) if base_loss else float("inf")
        within = delta <= tol * abs(base_loss)
        dtype = "bf16" if v["bf16"] else "fp32"
        if within:
            details.append(
                f"✅ {name} ({dtype}): step0={loss:.6f} Δ={delta:.2e} "
                f"(rel={rel:.2e} <= {tol:.0e})"
            )
        else:
            findings += 1
            details.append(
                f"❌ SHADOW SUSPECT {name} ({dtype}): step0={loss:.6f} "
                f"vs baseline {base_loss:.6f}, Δ={delta:.2e} rel={rel:.2e} "
                f"EXCEEDS rtol {tol:.0e} — kernel changed the math"
            )
        rows.append(
            {
                "variant": name,
                "step0": loss,
                "delta": delta,
                "rel_delta": rel,
                "tol": tol,
                "dtype": dtype,
                "within_tol": within,
            }
        )

    if compared == 0:
        return GateResult.could_not_run(
            GATE_ID,
            GATE_NAME,
            "baseline ran but no kernel variant produced a step-0 loss to compare",
        )

    data = {
        "baseline_step0": base_loss,
        "variants": rows,
        "rtol": rtol,
        "rtol_bf16": rtol_bf16,
        "skipped": skip_notes,
    }
    status = GateStatus.FINDINGS if findings else GateStatus.PASS
    summary = (
        f"{compared} kernel variant(s) vs baseline step0={base_loss:.4f}; "
        f"{'all within tol' if not findings else f'{findings} SHADOW SUSPECT'}"
    )
    return GateResult(
        GATE_ID, GATE_NAME, status, summary=summary, details=details, data=data
    )
