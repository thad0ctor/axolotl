"""G6 — loss sanity: does a short real train produce finite, non-diverging loss?

G6 drives the REAL ``axolotl.train`` entry point (via ``harness.runner``) for a
handful of training steps on the tiny CI fixture, for a small set of variants
(baseline = no kernels; on a non-smoke profile add a liger variant when the arch
has native liger support). It then reads the per-step ``loss_history`` and asserts
two things a "did it train at all" check misses:

    * every logged loss is FINITE — a single NaN/inf is a hard FINDINGS.
    * loss does not BLOW UP — the final-window mean must stay at-or-below the
      initial-window mean times ``loss_blowup_ratio`` (default 1.5). Tiny
      randomly-pretrained models are noisy over ~12 steps, so a brittle strict
      ``final <= initial * 0.95`` is reported as an informational note, not the
      gate verdict (documented weakening, see ``run``).

Each variant trains in a FRESH subprocess (``-m harness.gates.g6_loss --worker``)
so liger/CCE monkeypatches — which are installed process-wide and never undone —
cannot leak from one variant into the next. The worker is shared with G7.
"""

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
    # A short train runs on CPU too (slow); run() degrades if even that is infeasible.
    return True


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
    """(name, extra-flags) per variant. Baseline always; a liger CE variant on a
    non-smoke profile when the model has native liger support and a GPU (the
    triton kernels need CUDA)."""
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
            # eager keeps the run CPU-runnable and isolates kernel deltas to the
            # CE/RMS path (liger CE does not touch attention).
            "attn_implementation": "eager",
        }
    )
    cfg.update(extra)
    return cfg


# --- subprocess worker (shared with G7) ----------------------------------------


def _spawn_variant(
    ctx: GateContext, name: str, cfg_dict: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """Train one variant in a fresh interpreter; return the parsed worker result.

    Isolation is the point: liger/CCE patch transformers classes globally and are
    never reverted, so an in-process baseline-then-kernel comparison is unsound.
    """
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
    # The child must import `harness` (scripts dir) and `axolotl` (repo src). The
    # orchestrator self-paths via sys.path, which a subprocess does not inherit —
    # so propagate both on PYTHONPATH explicitly rather than relying on the caller.
    scripts_dir = Path(__file__).resolve().parents[2]
    src_dir = ctx.repo_root / "src"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(scripts_dir), str(src_dir)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
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


_KERNEL_NS = ("liger", "cut_cross_entropy", "scattermoe", "sonicmoe")


def _engagement(model) -> dict[str, Any]:
    """Fingerprint whether a kernel actually patched the model, so G7 can tell a
    real numerical match from a kernel that silently no-op'd (the same Δ=0). Looks
    at the top model's forward (CE/CCE patch the forward fn) and submodule class
    namespaces (rms_norm/swiglu swap module classes)."""
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
    # liger_cross_entropy patches loss symbols at MODULE scope (a transient
    # LigerCrossEntropyLoss never appears on model.modules()), so also inspect the
    # modeling module's loss symbols + the model's loss_function attribute.
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
    return max(1, min(3, n // 3))


def _tb_note(ctx: GateContext, output_dir: str, ratio: float) -> str:
    """Best-effort reuse of the e2e tensorboard checker for an extra learning
    signal; never the gate verdict (its strict ratio is brittle on tiny models)."""
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
    ran = 0

    for name, extra in _variants(ctx):
        cfg = _build_cfg(ctx, name, extra, max_steps)
        res = _spawn_variant(ctx, name, cfg, timeout)
        if not res.get("ok"):
            # An applicable variant that cannot train is unverified, not benign:
            # count it so the gate can't report PASS on a broken kernel path.
            findings += 1
            details.append(f"❌ {name}: could not train — {res.get('error')}")
            rows.append({"variant": name, "ran": False, "error": res.get("error")})
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

    if ran == 0:
        return GateResult.could_not_run(
            GATE_ID,
            GATE_NAME,
            "no variant could train (see details)",
        )
    data = {
        "variants": rows,
        "max_steps": max_steps,
        "blowup_ratio": blowup,
        "strict_ratio": strict,
    }
    status = GateStatus.FINDINGS if findings else GateStatus.PASS
    summary = (
        f"{ran} variant(s) trained; "
        f"{'no NaN/inf, none diverged' if not findings else f'{findings} with NaN/inf or divergence'}"
    )
    return GateResult(
        GATE_ID, GATE_NAME, status, summary=summary, details=details, data=data
    )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3])
    else:  # pragma: no cover - guard for accidental direct invocation
        sys.stderr.write("usage: python -m harness.gates.g6_loss --worker CFG OUT\n")
        sys.exit(2)
