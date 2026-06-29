"""G1 — config-resolution compat matrix: drive the real ``load_cfg`` pipeline over a set of composites and assign each a four-way :class:`Verdict`. WARNED_NO_OP (accepted but silently no-ops) is the high-value signal, surfaced as FINDINGS, never a green check."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import yaml

from .. import GateContext, GateResult, GateStatus, Verdict

GATE_ID = "G1"
GATE_NAME = "config"

_LIGER_PLUGIN = "axolotl.integrations.liger.LigerPlugin"
_CCE_PLUGIN = "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"
_KERNELS_PLUGIN = "axolotl.integrations.kernels.KernelsPlugin"

# flag key -> warning substrings, so incidental warnings (FSDP1/bf16) don't falsely paint a cell WARNED_NO_OP
_FLAG_WARN_TOKENS = {
    "sample_packing": ("sample_packing",),
    "batch_flattening": ("batch_flattening",),
    "scaling_softmax": ("scaling_softmax",),
    "liger_use_token_scaling": ("token_scaling",),
    "liger_glu_activation": ("glu", "swiglu"),
    "liger_rms_norm": ("rms_norm",),
    "activation_offloading": ("activation_offloading", "offload"),
    # no bare "kernel": it appears in unrelated triton/liger advisories
    "expert_backend": ("expert", "scattermoe", "sonicmoe"),
    "use_scattermoe": ("scattermoe",),
    "moe_grouped_backend": ("moe_grouped", "grouped"),
}

# flag names too generic to attribute warnings by bare name (e.g. "rl" hits early/control/url) -> deny
_GENERIC_FLAG_DENY = {"bf16", "fp16", "tf32", "fp8", "use_kernels", "kernel", "rl"}

# structured keys whose resolved form always differs (pydantic defaults) -> useless for warn/normalize attribution
_STRUCT_KEYS = ("plugins", "datasets", "trl")

# keys appearing in the resolved cfg but not the input signal a NORMALIZED rewrite
_CANONICALIZATION_WATCH = (
    "use_scattermoe",
    "use_sonicmoe",
    "experts_implementation",
)


def applies(ctx: GateContext) -> bool:
    return True


@dataclass
class _Cell:
    composite_id: str
    flags: dict[str, Any]
    expect: str  # "resolve" | "reject"
    purpose: str


@dataclass
class _Outcome:
    cell: _Cell
    verdict: Verdict
    note: str
    normalized_changes: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    bisect: list[str] = field(default_factory=list)


@contextmanager
def _capture():
    """Capture warnings on both channels (Python ``warnings`` + the ``axolotl`` logger) for one ``load_cfg`` call; clear ``warning_once``'s process-wide cache so advisories re-fire per composite."""
    records: list[str] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                records.append(record.getMessage())
            except Exception:  # noqa: BLE001 - a bad format arg must not abort capture
                records.append(str(record.msg))

    sink = _Sink(level=logging.WARNING)
    axolotl_log = logging.getLogger("axolotl")
    axolotl_log.addHandler(sink)

    try:
        from axolotl.utils.logging import MultiProcessAdapter

        MultiProcessAdapter.warning_once.cache_clear()
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug(
            "warning_once cache reset skipped", exc_info=True
        )

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            yield records
        finally:
            for w in wlist:
                records.append(str(w.message))
            axolotl_log.removeHandler(sink)


class _ResetFailure(RuntimeError):
    """Plugin-manager reset failed: the cell's isolation is compromised -> COULD_NOT_RUN."""


def _reset_plugins() -> BaseException | None:
    # PluginManager is a process singleton; reset so each load_cfg sees only its own plugins (a leaked reset shadows later verdicts)
    try:
        from axolotl.integrations.base import PluginManager

        PluginManager._instance = None  # noqa: SLF001
    except Exception as err:  # noqa: BLE001
        return err
    return None


def _base_cfg(ctx: GateContext, composite_id: str) -> dict[str, Any]:
    return {
        "base_model": ctx.features.base_model,
        "output_dir": str(ctx.output_dir / composite_id),
        "datasets": [{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "sequence_len": 512,
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "val_set_size": 0,
    }


def _run_cfg(
    ctx: GateContext, flags: dict[str, Any], composite_id: str
) -> tuple[dict | None, BaseException | None, list[str]]:
    """Write a minimal YAML and run it through the real load_cfg pipeline."""
    from axolotl.cli.config import load_cfg

    cfg_dict = _base_cfg(ctx, composite_id)
    cfg_dict.update(flags)
    cfg_path = ctx.output_dir / f"{composite_id}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    reset_err = _reset_plugins()
    if reset_err is not None:
        # don't run load_cfg on un-isolated state; surface as could-not-run
        return None, _ResetFailure(f"{type(reset_err).__name__}: {reset_err}"), []
    resolved: dict | None = None
    exc: BaseException | None = None
    with _capture() as records:
        try:
            resolved = dict(load_cfg(str(cfg_path)))
        except BaseException as err:  # noqa: BLE001 - any raise is a REJECTED verdict
            exc = err
    return resolved, exc, list(records)


def _relevant_warnings(flags: dict[str, Any], records: list[str]) -> list[str]:
    tokens: list[str] = []
    for key in flags:
        if key in _FLAG_WARN_TOKENS:
            tokens.extend(_FLAG_WARN_TOKENS[key])
        elif key not in _GENERIC_FLAG_DENY and key not in _STRUCT_KEYS:
            tokens.append(key)
    out = []
    for msg in records:
        low = msg.lower()
        if any(tok in low for tok in tokens):
            out.append(msg)
    return out


def _normalized_changes(flags: dict[str, Any], resolved: dict) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for key, val in flags.items():
        if key in _STRUCT_KEYS:
            continue
        if key in resolved and resolved[key] != val:
            changes[key] = {"in": val, "out": resolved[key]}
    # canonicalizations that add a derived key rather than rewriting the input
    for key in _CANONICALIZATION_WATCH:
        if key not in flags and key in resolved:
            changes[key] = {"in": None, "out": resolved[key]}
    return changes


def _support_note(ctx: GateContext, flags: dict[str, Any]) -> str:
    """Registry cross-check distinguishing a real SUPPORTED branch from an accepted-but-inert flag."""
    notes = []
    if flags.get("sample_packing"):
        from axolotl.monkeypatch.multipack import SUPPORTED_MULTIPACK_MODEL_TYPES

        mt = ctx.features.model_config_type
        attn = flags.get("attn_implementation")
        from axolotl.utils.schemas.enums import ATTN_IMPLS_SUPPORTING_PACKING

        if (
            mt in SUPPORTED_MULTIPACK_MODEL_TYPES
            and attn in ATTN_IMPLS_SUPPORTING_PACKING
        ):
            notes.append(f"sample_packing: {mt} in multipack registry + varlen attn")
        elif attn not in ATTN_IMPLS_SUPPORTING_PACKING:
            notes.append("sample_packing: attn not varlen-capable (no-op risk)")
        else:
            notes.append(f"sample_packing: {mt} NOT in multipack registry")
    if flags.get("load_in_4bit") or flags.get("load_in_8bit"):
        # resolution-only: bnb skip-modules (jamba->mamba, falcon_h1->out_proj) only apply at load (a GPU gate)
        mt = ctx.features.model_config_type
        skip = {"jamba": "mamba", "falcon_h1": "out_proj"}.get(mt)
        notes.append(
            f"quant resolves; load-time bnb skip_modules=[{skip}] for {mt}"
            if skip
            else "quant resolves (load-time bnb config is a separate load gate)"
        )
    return "; ".join(notes)


def _classify(
    ctx: GateContext, cell: _Cell, resolved: dict | None, exc, records: list[str]
) -> _Outcome:
    rel = _relevant_warnings(cell.flags, records)

    if exc is not None:
        note = f"{type(exc).__name__}: {exc}".split("\n")[0]
        if cell.expect == "reject":
            note = f"expected reject — {note}"
        else:
            note = f"UNEXPECTED reject — {note}"
        return _Outcome(cell, Verdict.REJECTED, note, warnings=rel)

    changes = _normalized_changes(cell.flags, resolved or {})
    support = _support_note(ctx, cell.flags)

    if rel:
        return _Outcome(
            cell,
            Verdict.WARNED_NO_OP,
            support or "warned / possible no-op",
            normalized_changes=changes,
            warnings=rel,
        )
    if changes:
        return _Outcome(cell, Verdict.NORMALIZED, support or "value rewritten", changes)
    return _Outcome(cell, Verdict.SUPPORTED, support or "resolved clean")


# --- composite construction ----------------------------------------------------


def _composites(ctx: GateContext) -> list[_Cell]:
    """One maximal-compatible composite per cross-entropy option, then oracle probes."""
    is_moe = ctx.features.is_moe
    cells: list[_Cell] = []

    cells.append(
        _Cell(
            "c0_none_packing",
            {
                "attn_implementation": "flash_attention_2",
                "sample_packing": True,
                "gradient_checkpointing": True,
                "activation_offloading": True,
            },
            "resolve",
            "CE=none; exercises sample_packing, gradient_checkpointing, activation_offloading",
        )
    )

    cells.append(
        _Cell(
            "c1_liger_ce_batchflatten",
            {
                "plugins": [_LIGER_PLUGIN],
                "liger_cross_entropy": True,
                "liger_glu_activation": True,
                "liger_rms_norm": True,
                "attn_implementation": "flash_attention_2",
                "batch_flattening": True,
            },
            "resolve",
            "CE=liger_cross_entropy; exercises batch_flattening, liger_glu_activation, liger_rms_norm",
        )
    )

    cells.append(
        _Cell(
            "c2_flce_flex_softmax",
            {
                "plugins": [_LIGER_PLUGIN],
                "liger_fused_linear_cross_entropy": True,
                "liger_use_token_scaling": True,  # nosec B105 - axolotl flag name, not a secret
                "attn_implementation": "flex_attention",
                "scaling_softmax": True,
            },
            "resolve",
            "CE=FLCE; exercises liger_use_token_scaling, scaling_softmax, flex_attention",
        )
    )

    cells.append(
        _Cell(
            "c3_cce_bf16",
            {
                "plugins": [_CCE_PLUGIN],
                "cut_cross_entropy": True,
                "bf16": True,
                "attn_implementation": "flash_attention_2",
                "gradient_checkpointing": True,
            },
            "resolve",
            "CE=cut_cross_entropy; exercises cut_cross_entropy + bf16 co-requisite",
        )
    )

    # oracle probe for the WARNED_NO_OP channel (expect="warn"): firing is EXPECTED, not a model finding
    cells.append(
        _Cell(
            "c4_chunked_packing_sdpa",
            {
                "chunked_cross_entropy": True,
                "attn_implementation": "sdpa",
                "sample_packing": True,
            },
            "warn",
            "CE=chunked; sample_packing without varlen attn -> oracle probe for WARNED_NO_OP",
        )
    )

    if is_moe:
        # expert_backend canonicalizes onto use_scattermoe -> NORMALIZED probe
        cells.append(
            _Cell(
                "c5_moe_expert_backend",
                {
                    "plugins": [_KERNELS_PLUGIN],
                    "use_kernels": True,
                    "expert_backend": "scattermoe",
                    "attn_implementation": "flash_attention_2",
                },
                "resolve",
                "MoE; exercises expert_backend canonicalization (-> use_scattermoe)",
            )
        )

    if ctx.profile == "multigpu":
        cells.extend(_multigpu_composites(ctx))

    # quant/RL composites bloat the smoke set -> gate behind profile=full or an opt-in flag; smoke stays ~6 cells
    if (
        ctx.profile == "full"
        or ctx.options.get("quant")
        or ctx.options.get("quantization")
    ):
        cells.extend(_quant_composites(ctx))
    if ctx.profile == "full" or ctx.options.get("rl"):
        cells.extend(_rl_composites(ctx))

    # oracle probe (expected reject): confirm the partial oracle still fires
    cells.append(
        _Cell(
            "p_dual_ce",
            {
                "plugins": [_LIGER_PLUGIN],
                "liger_cross_entropy": True,
                "cut_cross_entropy": True,
                "bf16": True,
            },
            "reject",
            "two cross-entropy flags -> expected REJECTED (mutual exclusivity)",
        )
    )
    return cells


def _multigpu_composites(ctx: GateContext) -> list[_Cell]:
    cells = [
        _Cell(
            "mg_fsdp2",
            {
                "fsdp_version": 2,
                "fsdp_config": {"offload_params": False},
                "attn_implementation": "flash_attention_2",
            },
            "resolve",
            "multigpu: FSDP2 resolves",
        ),
        _Cell(
            "mg_liger_rmsnorm_tp",
            {
                "plugins": [_LIGER_PLUGIN],
                "liger_rms_norm": True,
                "tensor_parallel_size": 2,
                "attn_implementation": "flash_attention_2",
            },
            "reject",
            "multigpu: liger_rms_norm + tensor_parallel -> expected REJECTED",
        ),
    ]
    return cells


def _quant_composites(ctx: GateContext) -> list[_Cell]:
    """bitsandbytes quant axes through load_cfg (peft.py validate_adapter/validate_qlora). Resolution-only: the real 4bit/8bit model load is a separate GPU/load gate."""
    return [
        _Cell(
            "q0_qlora_4bit",
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            "resolve",
            "qlora + load_in_4bit (+bnb_4bit_*) normalizes for this model type",
        ),
        _Cell(
            "q1_lora_8bit",
            {"adapter": "lora", "load_in_8bit": True},
            "resolve",
            "lora + load_in_8bit (the valid int8 LoRA path)",
        ),
        _Cell(
            "q2_qlora_pack_fa2",
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                "sample_packing": True,
                "attn_implementation": "flash_attention_2",
            },
            "resolve",
            "qlora + sample_packing + flash_attention_2 (common quantized-LoRA path)",
        ),
        # oracle: quant without an adapter is rejected (peft.validate_adapter)
        _Cell(
            "q_oracle_4bit_noadapter",
            {"load_in_4bit": True},
            "reject",
            "load_in_4bit without adapter -> expected REJECTED",
        ),
        # oracle: qlora must be 4bit, not 8bit (peft.validate_qlora)
        _Cell(
            "q_oracle_qlora_8bit",
            {"adapter": "qlora", "load_in_8bit": True},
            "reject",
            "qlora + load_in_8bit -> expected REJECTED (qlora requires 4bit)",
        ),
    ]


# minimal valid RL dataset stanzas (mirroring examples/*); config resolution only, not training
_RL_DATASETS = {
    "dpo": [
        {
            "path": "fozziethebeat/alpaca_messages_2k_dpo_test",
            "type": "chat_template.default",
            "field_messages": "conversation",
            "field_chosen": "chosen",
            "field_rejected": "rejected",
        }
    ],
    "kto": [
        {
            "path": "argilla/ultrafeedback-binarized-preferences-cleaned-kto",
            "type": "llama3.ultra",
            "split": "train",
        }
    ],
    "orpo": [
        {
            "path": "argilla/ultrafeedback-binarized-preferences-cleaned",
            "type": "chat_template.argilla",
        }
    ],
    "grpo": [
        {
            "path": "openai/gsm8k",
            "name": "main",
            "type": "rewards.prompt_transform",
            "split": "train",
        }
    ],
}


def _rl_composites(ctx: GateContext) -> list[_Cell]:
    """Resolve each RL mode through load_cfg with its minimal valid datasets/keys; surfaces rejects/normalizations per mode. Resolution-only, not training."""
    return [
        _Cell(
            "rl_dpo",
            {
                "rl": "dpo",
                "chat_template": "llama3",
                "datasets": _RL_DATASETS["dpo"],
                "sample_packing": False,
            },
            "resolve",
            "rl: dpo with chosen/rejected dataset",
        ),
        _Cell(
            "rl_kto",
            {
                "rl": "kto",
                "datasets": _RL_DATASETS["kto"],
                "remove_unused_columns": False,  # check_kto_config requires this
                "sample_packing": False,
            },
            "resolve",
            "rl: kto with binary-label dataset",
        ),
        _Cell(
            "rl_orpo",
            {
                "rl": "orpo",
                "chat_template": "chatml",
                "datasets": _RL_DATASETS["orpo"],
                "remove_unused_columns": False,
                "sample_packing": False,
            },
            "resolve",
            "rl: orpo single-stage alignment",
        ),
        _Cell(
            "rl_grpo",
            {
                "rl": "grpo",
                "chat_template": "tokenizer_default",
                "datasets": _RL_DATASETS["grpo"],
                "skip_prepare_dataset": True,
                # num_generations must divide the gen batch (mbs*gas=2); see
                # check_grpo_batch_size_divisibility
                "trl": {
                    "num_generations": 2,
                    "max_completion_length": 128,
                    "reward_funcs": ["rewards.accuracy_reward"],
                },
            },
            "resolve",
            "rl: grpo with reward-fn dataset (num_generations divides gen batch)",
        ),
    ]


# --- delta-debug ---------------------------------------------------------------


def _bisect(ctx: GateContext, cell: _Cell) -> list[str]:
    """ddmin over the cell's optional flags: smallest subset whose load_cfg still rejects."""
    items = [(k, v) for k, v in cell.flags.items() if k != "plugins"]
    plugins = cell.flags.get("plugins")

    def _fails(subset: list[tuple[str, Any]]) -> bool:
        flags = dict(subset)
        if plugins:
            flags["plugins"] = plugins
        _, exc, _ = _run_cfg(ctx, flags, f"{cell.composite_id}__bisect")
        return exc is not None

    if not _fails(items):
        return []  # not reproducible in isolation
    n = 2
    while len(items) >= 2:
        chunk = max(1, len(items) // n)
        subsets = [items[i : i + chunk] for i in range(0, len(items), chunk)]
        reduced = False
        for s in subsets:
            complement = [x for x in items if x not in s]
            if complement and _fails(complement):
                items = complement
                n = max(n - 1, 2)
                reduced = True
                break
        if not reduced:
            if n >= len(items):
                break
            n = min(len(items), n * 2)
    return [k for k, _ in items]


def run(ctx: GateContext) -> GateResult:
    try:
        import axolotl.cli.config  # noqa: F401
    except BaseException as err:  # noqa: BLE001
        return GateResult.could_not_run(
            GATE_ID, GATE_NAME, f"axolotl import failed: {type(err).__name__}: {err}"
        )

    cells = _composites(ctx)
    do_bisect = ctx.profile == "full" or bool(ctx.options.get("auto_bisect"))

    outcomes: list[_Outcome] = []
    could_not_run = 0
    for cell in cells:
        resolved, exc, records = _run_cfg(ctx, cell.flags, cell.composite_id)
        # unimportable config (offline/bad id) or failed reset = environment/isolation problem, not a config verdict
        if exc is not None and (
            isinstance(exc, _ResetFailure) or _is_environment_error(exc)
        ):
            could_not_run += 1
            outcomes.append(
                _Outcome(
                    cell,
                    Verdict.REJECTED,
                    f"COULD-NOT-RUN ({type(exc).__name__}: {str(exc).splitlines()[0]})",
                    warnings=[],
                )
            )
            continue
        outcome = _classify(ctx, cell, resolved, exc, records)
        if (
            do_bisect
            and outcome.verdict == Verdict.REJECTED
            and cell.expect == "resolve"
        ):
            outcome.bisect = _bisect(ctx, cell)
        outcomes.append(outcome)

    return _assemble(ctx, outcomes, could_not_run)


_ENV_ERROR_MARKERS = (
    "couldn't connect",
    "could not connect",
    "offline",
    "is not a local folder",
    "is not the path to a directory",
    "does not appear to have a file named config.json",
    "connectionerror",
    "max retries",
    "failed to resolve",
)


def _is_environment_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    # OSError subclasses (FileNotFoundError/PermissionError/TimeoutError) are local runtime, not findings
    if isinstance(exc, OSError):
        return True
    if name in ("ConnectionError", "HTTPError", "GatedRepoError"):
        return True
    low = f"{name}: {exc}".lower()
    return any(m in low for m in _ENV_ERROR_MARKERS)


def _assemble(
    ctx: GateContext, outcomes: list[_Outcome], could_not_run: int
) -> GateResult:
    matrix = []
    details = []
    findings = 0
    unexpected_reject = 0
    warned = 0
    resolved_ok = 0
    total_resolvable = 0

    for o in outcomes:
        cell = o.cell
        matrix.append(
            {
                "composite_id": cell.composite_id,
                "flags": cell.flags,
                "expect": cell.expect,
                "verdict": o.verdict.value,
                "note": o.note,
                "normalized_changes": o.normalized_changes,
                "warnings": o.warnings,
                "bisect": o.bisect,
            }
        )
        line = f"{o.verdict.icon} {cell.composite_id}: {o.verdict.value} — {o.note}"
        if o.bisect:
            line += f" [minimal failing flags: {o.bisect}]"
        details.append(line)

        if cell.expect == "resolve":
            total_resolvable += 1
            if o.verdict == Verdict.REJECTED and "COULD-NOT-RUN" not in o.note:
                unexpected_reject += 1
                findings += 1
            elif o.verdict == Verdict.WARNED_NO_OP:
                # a cell built valid that warns/no-ops is the real signal
                warned += 1
                findings += 1
                resolved_ok += 1
            elif o.verdict != Verdict.REJECTED:
                resolved_ok += 1
        elif cell.expect == "warn":
            # oracle probe: firing is expected; resolving clean means the warn path regressed
            if o.verdict == Verdict.REJECTED and "COULD-NOT-RUN" in o.note:
                # env failure, not an oracle regression — don't count it as a finding
                continue
            total_resolvable += 1
            if o.verdict == Verdict.WARNED_NO_OP:
                resolved_ok += 1
            elif o.verdict == Verdict.REJECTED and "COULD-NOT-RUN" not in o.note:
                unexpected_reject += 1
                findings += 1
            else:
                findings += 1
                details.append(
                    f"  !! {cell.composite_id} was built to warn/no-op but resolved "
                    "clean (oracle gap — the warn path may have regressed)"
                )
        else:  # expect reject
            if o.verdict != Verdict.REJECTED:
                # the oracle did NOT fire on a combo built to be invalid
                findings += 1
                details.append(
                    f"  !! {cell.composite_id} was built invalid but RESOLVED "
                    "(oracle gap)"
                )

    if could_not_run and resolved_ok == 0:
        return GateResult(
            GATE_ID,
            GATE_NAME,
            GateStatus.COULD_NOT_RUN,
            summary=f"{could_not_run} composite(s) could not run (model/env unloadable)",
            details=details,
            data={"matrix": matrix},
        )

    status = GateStatus.FINDINGS if findings else GateStatus.PASS
    summary = (
        f"{resolved_ok}/{total_resolvable} composites resolve; "
        f"{warned} warned-no-op, {unexpected_reject} unexpected-reject"
    )
    if could_not_run:
        summary += f"; {could_not_run} could-not-run"
    return GateResult(
        GATE_ID,
        GATE_NAME,
        status,
        summary=summary,
        details=details,
        data={"matrix": matrix, "profile": ctx.profile},
    )
