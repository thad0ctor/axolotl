"""G7 — cross-variant numerical consistency: with identical weights/data/seed, a kernel's STEP-0 loss must equal the baseline within ``rtol``, else it silently changed the math. Each variant runs in a fresh subprocess (isolating global monkeypatches) and carries an engagement fingerprint so a kernel that silently no-op'd (same Δ≈0) is flagged NOT ENGAGED rather than passing green. bf16 variants are compared against a bf16 baseline so precision loss isn't conflated with kernel math."""

from __future__ import annotations

import math
from typing import Any

from .. import GateContext, GateResult, GateStatus
from . import g6_loss

GATE_ID = "G7"
GATE_NAME = "consistency"

_LIGER_PLUGIN = "axolotl.integrations.liger.LigerPlugin"
_CCE_PLUGIN = "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"


def applies(ctx: GateContext) -> bool:
    # always applies; run() degrades to SKIPPED when no kernel variant is applicable
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
    """Return (variants, skip-notes); a variant is included only when the registry says the kernel applies to this type. All kernels are triton -> require a GPU."""
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
                # CE is swapped at module scope, not on the instance -> not reliably fingerprintable; engage-unknown, rely on the numeric check
                "markers": (),
            }
        )
        variants.append(
            {
                "name": "liger_rms_norm",
                "flags": {"plugins": [_LIGER_PLUGIN], "liger_rms_norm": True},
                "bf16": False,
                "markers": ("liger",),  # swaps RMSNorm module classes -> detectable
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
                "markers": ("cut_cross_entropy",),
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
                "markers": (),  # attention hijack isn't a module/forward swap we fingerprint
            }
        )
    else:
        notes.append(
            f"fused_attn_kernel: {mct} not in _FUSED_ATTN_KERNEL_SUPPORTED — skipped"
        )

    return variants, notes


def _engaged(res: dict[str, Any], markers: tuple[str, ...]) -> bool | None:
    """True/False when fingerprintable (markers in the worker fingerprint), None when not. False on an expected-to-engage variant is a no-op shadow (same Δ=0 as a genuine match)."""
    if not markers:
        return None
    eng = res.get("engagement") or {}
    hay = (
        str(eng.get("forward_module", ""))
        + " "
        + " ".join(eng.get("kernel_namespaces", []))
    ).lower()
    return any(m in hay for m in markers)


def _build_cfg(ctx: GateContext, name: str, flags: dict[str, Any]) -> dict[str, Any]:
    """A 1-step cfg; flags override the eager/no-kernel baseline."""
    cfg = g6_loss._build_cfg(ctx, f"consistency_{name}", {}, max_steps=1)
    # baseline forces eager; kernel variants may re-specify attn (e.g. fused)
    cfg.update(flags)
    return cfg


def _step0(res: dict[str, Any]) -> float | None:
    losses = res.get("loss_history") or []
    return float(losses[0]) if losses else None


def run(ctx: GateContext) -> GateResult:
    details: list[str] = []
    rows: list[dict[str, Any]] = []

    variants, skip_notes = _kernel_variants(ctx)
    for note in skip_notes:
        details.append(f"· {note}")

    # skip BEFORE importing axolotl, else no-GPU/no-kernel would surface as could_not_run not SKIPPED
    if not variants:
        return GateResult(
            GATE_ID,
            GATE_NAME,
            GateStatus.SKIPPED,
            summary=f"no applicable kernel variant to compare; {'; '.join(skip_notes)}",
            details=details,
            data={"baseline_step0": None, "variants": rows, "skipped": skip_notes},
        )

    try:
        import axolotl.train  # noqa: F401
    except BaseException as err:  # noqa: BLE001
        return GateResult.could_not_run(
            GATE_ID, GATE_NAME, f"axolotl import failed: {type(err).__name__}: {err}"
        )

    rtol = float(ctx.options.get("rtol", 2e-2))
    rtol_bf16 = float(ctx.options.get("rtol_bf16", 5e-2))
    timeout = float(ctx.options.get("train_timeout", 1200))

    # one baseline per dtype: a bf16 variant must compare to a bf16 baseline
    def _baseline(dtype: str):
        flags = {"bf16": True} if dtype == "bf16" else {}
        cfg = _build_cfg(ctx, f"baseline_{dtype}", flags)
        return g6_loss._spawn_variant(ctx, f"g7_baseline_{dtype}", cfg, timeout)

    baselines: dict[str, float] = {}
    base_fp32 = _step0(_baseline("fp32"))
    if base_fp32 is None:
        return GateResult.could_not_run(
            GATE_ID, GATE_NAME, "fp32 baseline step-0 loss unavailable"
        )
    # a non-finite baseline makes every Δ a misleading shadow finding -> fail on the baseline
    if not math.isfinite(base_fp32):
        return GateResult.could_not_run(
            GATE_ID,
            GATE_NAME,
            f"fp32 baseline step-0 loss is non-finite ({base_fp32}) — cannot compare variants",
        )
    baselines["fp32"] = base_fp32
    details.append(f"baseline fp32 (eager, no kernels): step0 = {base_fp32:.6f}")
    if any(v["bf16"] for v in variants):
        base_bf16 = _step0(_baseline("bf16"))
        if base_bf16 is not None:
            if not math.isfinite(base_bf16):
                return GateResult.could_not_run(
                    GATE_ID,
                    GATE_NAME,
                    f"bf16 baseline step-0 loss is non-finite ({base_bf16}) — cannot compare variants",
                )
            baselines["bf16"] = base_bf16
            details.append(
                f"baseline bf16 (eager, no kernels): step0 = {base_bf16:.6f}"
            )

    findings = 0
    env_failed = 0
    compared = 0
    for v in variants:
        name = v["name"]
        dtype = "bf16" if v["bf16"] else "fp32"
        tol = rtol_bf16 if v["bf16"] else rtol
        base_loss = baselines.get(dtype)
        if base_loss is None:
            # no same-dtype baseline (it failed to produce a step-0 loss) -> nothing to compare against = could-not-run, not a model shadow
            env_failed += 1
            details.append(
                f"⚠️ {name}: could_not_run — no {dtype} baseline to compare against"
            )
            rows.append(
                {
                    "variant": name,
                    "step0": None,
                    "dtype": dtype,
                    "could_not_run": True,
                    "error": f"no {dtype} baseline",
                }
            )
            continue
        cfg = _build_cfg(ctx, name, v["flags"])
        res = g6_loss._spawn_variant(ctx, f"g7_{name}", cfg, timeout)
        retried = res.get("retried", 0)
        rsfx = f" (retried {retried}x)" if retried else ""
        loss = _step0(res)
        if loss is None:
            kind = g6_loss._transient_kind(res)
            if kind:
                # won't-compile here is an env limit, not the kernel changing the math
                env_failed += 1
                details.append(
                    f"⚠️ {name}: could_not_run — {kind} (env/compile limit){rsfx}: "
                    f"{res.get('error')}"
                )
                rows.append(
                    {
                        "variant": name,
                        "step0": None,
                        "could_not_run": True,
                        "transient": kind,
                        "retried": retried,
                        "error": res.get("error"),
                    }
                )
            else:
                findings += 1
                details.append(
                    f"❌ {name}: step-0 loss unavailable{rsfx} — {res.get('error')}"
                )
                rows.append(
                    {
                        "variant": name,
                        "step0": None,
                        "retried": retried,
                        "error": res.get("error"),
                    }
                )
            continue

        # engagement first: a silent no-op yields the same Δ≈0 as a genuine match — the shadow class this gate exists to catch
        engaged = _engaged(res, v["markers"])
        if engaged is False:
            findings += 1
            compared += 1
            details.append(
                f"❌ NOT ENGAGED {name} ({dtype}): step0={loss:.6f} but the kernel "
                f"left no fingerprint ({v['markers']}) — it silently did not patch the "
                "model (no-op shadow), so an in-range loss is meaningless"
            )
            rows.append(
                {
                    "variant": name,
                    "step0": loss,
                    "dtype": dtype,
                    "engaged": False,
                    "within_tol": None,
                    "note": "kernel did not engage",
                }
            )
            continue

        compared += 1
        delta = abs(loss - base_loss)
        rel = delta / abs(base_loss) if base_loss else float("inf")
        within = delta <= tol * abs(base_loss)
        eng_s = {True: "engaged", None: "engage-unknown", False: "NOT-engaged"}[engaged]
        if within:
            details.append(
                f"✅ {name} ({dtype}, {eng_s}): step0={loss:.6f} Δ={delta:.2e} "
                f"(rel={rel:.2e} <= {tol:.0e}, vs {dtype} baseline {base_loss:.6f})"
            )
        else:
            findings += 1
            details.append(
                f"❌ SHADOW SUSPECT {name} ({dtype}, {eng_s}): step0={loss:.6f} "
                f"vs {dtype} baseline {base_loss:.6f}, Δ={delta:.2e} rel={rel:.2e} "
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
                "engaged": engaged,
                "within_tol": within,
            }
        )

    if compared == 0:
        why = (
            f"baseline ran but {env_failed} kernel variant(s) hit env/compile limits"
            if env_failed
            else "baseline ran but no kernel variant produced a step-0 loss to compare"
        )
        return GateResult.could_not_run(GATE_ID, GATE_NAME, why)

    data = {
        "baseline_step0": base_fp32,
        "baselines": baselines,
        "variants": rows,
        "rtol": rtol,
        "rtol_bf16": rtol_bf16,
        "skipped": skip_notes,
        "env_failed": env_failed,
    }
    # a kernel that RAN but diverges/no-ops is a real finding; one that couldn't compile only degrades to could_not_run
    if findings:
        status = GateStatus.FINDINGS
        summary = (
            f"{compared} compared vs baseline step0={base_fp32:.4f}; "
            f"{findings} flagged (shadow/no-op)"
            + (f"; {env_failed} env-limited" if env_failed else "")
        )
    elif env_failed:
        status = GateStatus.COULD_NOT_RUN
        summary = (
            f"{compared} compared, all within tol; "
            f"{env_failed} variant(s) hit env/compile limits"
        )
    else:
        status = GateStatus.PASS
        summary = (
            f"{compared} kernel variant(s) vs baseline step0={base_fp32:.4f}; "
            "all engaged + within tol"
        )
    return GateResult(
        GATE_ID, GATE_NAME, status, summary=summary, details=details, data=data
    )
