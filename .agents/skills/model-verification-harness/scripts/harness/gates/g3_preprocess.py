"""G3 — preprocess gate: does the model survive the real tokenize/prepare path?

The cheapest end-to-end signal: drive the actual ``axolotl preprocess`` entry
point against a tiny completion corpus and confirm it lands a prepared dataset on
disk whose ``train_dataset`` carries the columns the trainer reads
(``input_ids`` / ``labels`` / ``attention_mask``) with >0 rows. A model that
can't even tokenize never reaches G4..G7, so this gate gatekeeps the run-based
ladder: an environment/load failure here is COULD_NOT_RUN (we can't vouch for
anything downstream), while a prepared-but-malformed dataset is a real FINDING.
"""

from __future__ import annotations

from typing import Any

from .. import GateContext, GateResult, GateStatus, runner

GATE_ID = "G3"
GATE_NAME = "preprocess"

_REQUIRED_COLUMNS = ["input_ids", "labels", "attention_mask"]


def applies(ctx: GateContext) -> bool:  # noqa: ARG001 - always applies
    return True


def run(ctx: GateContext) -> GateResult:
    fixture = runner.write_completion_fixture(ctx.output_dir)
    cfg_dict = runner.base_cfg(
        ctx.features.base_model,
        ctx.output_dir,
        "g3",
        datasets=[runner.completion_dataset_stanza(fixture)],
    )
    cfg_path = runner.write_cfg(ctx.output_dir, cfg_dict, "g3")

    try:
        cfg = runner.resolve_cfg(cfg_path)
    except Exception as exc:  # noqa: BLE001 - config/env resolution failure
        return GateResult.could_not_run(
            GATE_ID,
            GATE_NAME,
            f"config did not resolve: {exc.__class__.__name__}: {exc}",
        )

    try:
        prepared_path = runner.preprocess_to_disk(cfg)
        dataset_meta = runner.prepare(cfg)
    except Exception as exc:  # noqa: BLE001 - model unloadable / offline / tokenizer
        return GateResult.could_not_run(
            GATE_ID,
            GATE_NAME,
            f"preprocess pipeline failed: {exc.__class__.__name__}: {exc}",
        )

    train_dataset = dataset_meta.train_dataset
    columns = list(train_dataset.column_names)
    n_rows = train_dataset.num_rows
    missing = [c for c in _REQUIRED_COLUMNS if c not in columns]

    data: dict[str, Any] = {
        "prepared_path": str(prepared_path),
        "prepared_path_exists": prepared_path.exists(),
        "n_rows": n_rows,
        "columns": columns,
        "missing_columns": missing,
    }

    details = [
        f"prepared dataset -> {prepared_path} (exists={prepared_path.exists()})",
        f"train_dataset: {n_rows} rows, columns={columns}",
    ]

    findings: list[str] = []
    if not prepared_path.exists():
        findings.append("dataset_prepared_path missing after preprocess")
    if missing:
        findings.append(f"prepared dataset missing required columns: {missing}")
    if n_rows <= 0:
        findings.append("prepared train_dataset has 0 rows")

    if findings:
        details.extend(f"finding: {f}" for f in findings)
        return GateResult(
            GATE_ID,
            GATE_NAME,
            GateStatus.FINDINGS,
            summary="; ".join(findings),
            details=details,
            data=data,
        )

    return GateResult(
        GATE_ID,
        GATE_NAME,
        GateStatus.PASS,
        summary=f"preprocess ok: {n_rows} rows, required columns present",
        details=details,
        data=data,
    )
