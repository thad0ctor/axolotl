"""G6 — loss sanity: short real ``axolotl.train`` per variant (each in a fresh subprocess so process-wide liger/CCE patches can't leak; worker shared with G7); assert every logged loss is finite and the final-window mean stays within ``loss_blowup_ratio`` of the initial."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - spawns our own interpreter on harness-built cfgs
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from .. import GateContext, GateResult, GateStatus

GATE_ID = "G6"
GATE_NAME = "loss"

_LIGER_PLUGIN = "axolotl.integrations.liger.LigerPlugin"


def applies(ctx: GateContext) -> bool:
    # GPU-only: a CPU train would just sit until train_timeout, so let the orchestrator skip it
    return ctx.gpu_available


# --- variant cfg construction --------------------------------------------------


def _liger_native_types() -> set[str]:
    try:
        from liger_kernel.transformers.monkey_patch import (
            MODEL_TYPE_TO_APPLY_LIGER_FN,
        )

        return set(MODEL_TYPE_TO_APPLY_LIGER_FN)
    except Exception:  # noqa: BLE001 - liger optional; absence just drops the variant
        return set()


def _variants(ctx: GateContext) -> list[tuple[str, dict[str, Any]]]:
    """(name, extra-flags) per variant: baseline always; a liger CE variant on non-smoke + native-liger + GPU."""
    variants: list[tuple[str, dict[str, Any]]] = [("baseline", {})]
    if ctx.profile != "smoke" and ctx.gpu_available:
        if ctx.model_config_type in _liger_native_types():
            variants.append(
                (
                    "liger_cross_entropy",
                    {"plugins": [_LIGER_PLUGIN], "liger_cross_entropy": True},
                )
            )
    return variants


def _build_cfg(
    ctx: GateContext, name: str, extra: dict[str, Any], max_steps: int
) -> dict[str, Any]:
    from ..runner import base_cfg

    cfg = base_cfg(ctx.features.base_model, ctx.output_dir, f"g6_{name}")
    cfg.update(
        {
            "seed": ctx.seed,
            "max_steps": max_steps,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "logging_steps": 1,
            "save_strategy": "no",
            "use_tensorboard": True,
            # eager keeps the run CPU-runnable and isolates kernel deltas to the CE/RMS path
            "attn_implementation": "eager",
        }
    )
    cfg.update(extra)
    return cfg


# --- subprocess worker (shared with G7) ----------------------------------------


# transient env/compile failures (won't compile / no resources here) vs a kernel that compiles and changes the math; matched against error+traceback to drive retry and could_not_run classing
_TRANSIENT_MARKERS = (
    "compilationerror",  # triton kernel failed to compile
    "out of resource",  # triton smem/register pressure at compile time
    "out of memory",  # CUDA OOM
    "outofmemoryerror",
    "nccl",  # collectives flake
    "timeout after",  # our own subprocess timeout sentinel
)


def _transient_kind(res: dict[str, Any]) -> str | None:
    """Matched transient/env marker (retryable, classes as could_not_run), else None."""
    if res.get("ok"):
        return None
    blob = f"{res.get('error', '')} {res.get('traceback', '')}".lower()
    return next((m for m in _TRANSIENT_MARKERS if m in blob), None)


def _spawn_once(
    ctx: GateContext, name: str, cfg_dict: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """Train one variant in a fresh interpreter; return the parsed worker result. Isolation matters: liger/CCE patch transformers classes globally and are never reverted."""
    from ..runner import write_cfg

    workdir = ctx.output_dir / "g6_work"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg_path = write_cfg(workdir, cfg_dict, f"cfg_{name}")
    out_path = workdir / f"result_{name}.json"
    if out_path.exists():
        out_path.unlink()

    cmd = [
        sys.executable,
        "-m",
        "harness.gates.g6_loss",
        "--worker",
        str(cfg_path),
        str(out_path),
    ]
    # subprocess doesn't inherit sys.path, so propagate harness (scripts dir) + axolotl (repo src) on PYTHONPATH explicitly
    scripts_dir = Path(__file__).resolve().parents[2]
    src_dir = ctx.repo_root / "src"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(scripts_dir), str(src_dir)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # user GPU pin (shared contract via ctx.options); unset -> inherit parent env
    gpus = ctx.options.get("gpus")
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        proc = subprocess.run(  # nosec B603 - cmd is sys.executable + fixed args
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout:.0f}s"}

    if out_path.exists():
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as err:  # noqa: BLE001
            return {"ok": False, "error": f"unreadable result json: {err}"}
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    return {
        "ok": False,
        "error": f"worker exited {proc.returncode} with no result; tail: {' | '.join(tail)}",
    }


def _spawn_variant(
    ctx: GateContext, name: str, cfg_dict: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """Spawn a variant, retrying transient kernel/compile/OOM/NCCL/timeout failures in a fresh subprocess before recording failure. ``retried`` records the extra attempts."""
    retries = max(0, int(ctx.options.get("variant_retries", 1)))
    res = _spawn_once(ctx, name, cfg_dict, timeout)
    attempts = 0
    while not res.get("ok") and attempts < retries and _transient_kind(res):
        attempts += 1
        res = _spawn_once(ctx, name, cfg_dict, timeout)
    if attempts:
        res["retried"] = attempts
    return res


_KERNEL_NS = ("liger", "cut_cross_entropy", "scattermoe", "sonicmoe")


def _engagement(model) -> dict[str, Any]:
    """Fingerprint whether a kernel actually patched the model, so G7 can tell a real match from a silent no-op (same Δ=0). Inspects the model forward and submodule class namespaces."""
    fwd_module = ""
    try:
        fwd = type(model).forward
        fwd_module = getattr(fwd, "__module__", "") or ""
    except Exception:  # noqa: BLE001
        pass
    namespaces: set[str] = set()
    try:
        for m in model.modules():
            mod = type(m).__module__ or ""
            if any(k in mod for k in _KERNEL_NS):
                namespaces.add(mod)
    except Exception:  # noqa: BLE001
        pass
    # liger_cross_entropy patches loss symbols at module scope (never on model.modules()), so also inspect the modeling module + loss_function attr
    try:
        import importlib

        loss_fn = getattr(model, "loss_function", None)
        if loss_fn is not None:
            lm = getattr(loss_fn, "__module__", "") or ""
            if any(k in lm for k in _KERNEL_NS):
                namespaces.add(lm)
        modmod = importlib.import_module(type(model).__module__)
        for sym in ("CrossEntropyLoss", "cross_entropy", "fixed_cross_entropy"):
            obj = getattr(modmod, sym, None)
            om = getattr(obj, "__module__", "") or ""
            if any(k in om for k in _KERNEL_NS):
                namespaces.add(om)
    except Exception:  # noqa: BLE001
        pass
    return {"forward_module": fwd_module, "kernel_namespaces": sorted(namespaces)}


def _worker(cfg_path: str, out_path: str) -> None:
    """Run inside the fresh subprocess: resolve -> prepare -> train -> dump losses."""
    result: dict[str, Any] = {"ok": False}
    try:
        from transformers import set_seed

        from harness.runner import (
            loss_history,
            prepare,
            resolve_cfg,
            train_model,
        )

        cfg = resolve_cfg(Path(cfg_path))
        set_seed(int(cfg.seed or 42))
        dataset_meta = prepare(cfg)
        model, _, trainer = train_model(cfg, dataset_meta)
        result = {
            "ok": True,
            "loss_history": loss_history(trainer),
            "output_dir": str(cfg.output_dir),
            "engagement": _engagement(model),
        }
    except BaseException as err:  # noqa: BLE001 - report any failure as the variant error
        import traceback

        result = {
            "ok": False,
            "error": f"{type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}",
            "traceback": traceback.format_exc()[-1500:],
        }
    Path(out_path).write_text(json.dumps(result), encoding="utf-8")


# --- loss verdict --------------------------------------------------------------


def _window(n: int) -> int:
    # average over ~a quarter of the run (>=2 steps when long enough) so a single noisy step can neither fake nor mask a real blowup
    if n >= 8:
        return min(3, max(2, n // 4))
    return max(1, n // 3)


def _tb_note(ctx: GateContext, output_dir: str, ratio: float) -> str:
    """Best-effort reuse of the e2e tensorboard checker; never the gate verdict (its strict ratio is brittle on tiny models)."""
    try:
        if str(ctx.repo_root) not in sys.path:
            sys.path.insert(0, str(ctx.repo_root))
        from tests.e2e.utils import check_tensorboard_loss_decreased

        check_tensorboard_loss_decreased(output_dir, max_loss_ratio=ratio)
        return f"tensorboard: loss decreased >= {1 - ratio:.0%} (strict)"
    except AssertionError as err:
        return f"tensorboard: strict-decrease not met — {str(err).splitlines()[0]}"
    except Exception as err:  # noqa: BLE001 - tbparse/log layout optional
        return f"tensorboard: check unavailable ({type(err).__name__})"


def run(ctx: GateContext) -> GateResult:
    if not ctx.gpu_available:
        return GateResult.skipped(GATE_ID, GATE_NAME, "GPU not available")
    try:
        import axolotl.train  # noqa: F401
    except BaseException as err:  # noqa: BLE001
        return GateResult.could_not_run(
            GATE_ID, GATE_NAME, f"axolotl import failed: {type(err).__name__}: {err}"
        )

    max_steps = int(ctx.options.get("loss_max_steps", 12))
    blowup = float(ctx.options.get("loss_blowup_ratio", 1.5))
    strict = float(ctx.options.get("loss_strict_ratio", 0.95))
    timeout = float(ctx.options.get("train_timeout", 1200))

    rows: list[dict[str, Any]] = []
    details: list[str] = []
    findings = 0
    env_failed = 0
    ran = 0

    for name, extra in _variants(ctx):
        cfg = _build_cfg(ctx, name, extra, max_steps)
        res = _spawn_variant(ctx, name, cfg, timeout)
        retried = res.get("retried", 0)
        rsfx = f" (retried {retried}x)" if retried else ""
        if not res.get("ok"):
            kind = _transient_kind(res)
            if kind:
                # a kernel that won't COMPILE in this env is a limitation, not a math
                # change -> surface as could_not_run, not a model finding
                env_failed += 1
                details.append(
                    f"⚠️ {name}: could_not_run — {kind} (env/compile limit){rsfx}: "
                    f"{res.get('error')}"
                )
                rows.append(
                    {
                        "variant": name,
                        "ran": False,
                        "could_not_run": True,
                        "transient": kind,
                        "retried": retried,
                        "error": res.get("error"),
                    }
                )
            else:
                findings += 1
                details.append(f"❌ {name}: could not train{rsfx} — {res.get('error')}")
                rows.append(
                    {
                        "variant": name,
                        "ran": False,
                        "retried": retried,
                        "error": res.get("error"),
                    }
                )
            continue

        losses = [float(x) for x in res.get("loss_history", [])]
        ran += 1
        if not losses:
            findings += 1
            details.append(f"❌ {name}: trained but logged no per-step loss")
            rows.append({"variant": name, "ran": True, "loss_history": []})
            continue

        finite = all(x == x and abs(x) != float("inf") for x in losses)
        w = _window(len(losses))
        init_mean = mean(losses[:w])
        final_mean = mean(losses[-w:])
        diverged = final_mean > init_mean * blowup
        decreased = final_mean <= init_mean * strict

        verdict = "pass"
        if not finite:
            verdict = "non-finite"
            findings += 1
        elif diverged:
            verdict = "diverged"
            findings += 1

        tb = _tb_note(ctx, res.get("output_dir", ""), strict)
        icon = "✅" if verdict == "pass" else "❌"
        details.append(
            f"{icon} {name}: {verdict} — n={len(losses)} init~{init_mean:.4f} "
            f"final~{final_mean:.4f} (window={w}); "
            f"{'strict-decrease ok' if decreased else 'no strict decrease (tiny-model noise)'}; "
            f"{tb}"
        )
        rows.append(
            {
                "variant": name,
                "ran": True,
                "loss_history": losses,
                "initial_mean": init_mean,
                "final_mean": final_mean,
                "window": w,
                "finite": finite,
                "diverged": diverged,
                "strict_decreased": decreased,
                "verdict": verdict,
            }
        )

    data = {
        "variants": rows,
        "max_steps": max_steps,
        "blowup_ratio": blowup,
        "strict_ratio": strict,
        "env_failed": env_failed,
    }
    # real findings dominate (actionable shadow signal); env-only failures degrade to could_not_run rather than masquerading as PASS
    if findings:
        status = GateStatus.FINDINGS
        summary = f"{ran} ran; {findings} with NaN/inf or divergence" + (
            f"; {env_failed} env-limited" if env_failed else ""
        )
    elif env_failed or ran == 0:
        status = GateStatus.COULD_NOT_RUN
        summary = (
            f"{ran} ran; {env_failed} variant(s) hit env/compile limits"
            if env_failed
            else "no variant could train (see details)"
        )
    else:
        status = GateStatus.PASS
        summary = f"{ran} variant(s) trained; no NaN/inf, none diverged"
    return GateResult(
        GATE_ID, GATE_NAME, status, summary=summary, details=details, data=data
    )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3])
    else:  # pragma: no cover - guard for accidental direct invocation
        sys.stderr.write("usage: python -m harness.gates.g6_loss --worker CFG OUT\n")
        sys.exit(2)
