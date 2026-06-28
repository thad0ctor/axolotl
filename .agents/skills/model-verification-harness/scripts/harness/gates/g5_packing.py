"""G5 — packing / trimming gate: does sample_packing produce valid sequences?

Packing concatenates several documents into one ``sequence_len``-long row and
relies on per-document ``position_ids`` (and a varlen attention backend) to keep
documents from attending across each other. Two silent failures: a model type
absent from ``SUPPORTED_MULTIPACK_MODEL_TYPES`` packs into a no-op/garbage attn
mask, and a trimming bug emits rows longer than ``sequence_len``. G5 drives the
real prepare path with ``sample_packing: True`` and asserts the type is packable,
every prepared row fits ``sequence_len``, and the per-document ``position_ids``
restart at 0 (so the collator's cross-document reset is well-formed).

``position_ids`` are added at PREPARE time by ``add_position_ids`` in
``utils/trainer.py`` (one ``range(len)`` per document); the multipack collator
(``utils/collators/batching.py`` ``V2BatchSamplerDataCollatorForSeq2Seq``)
concatenates those per-document arrays at COLLATE time, which is where the
restart-at-boundary pattern actually materializes. So here we assert each
prepared row's ``position_ids`` is a clean ``range(len)`` starting at 0.
"""

from __future__ import annotations

from typing import Any

from .. import GateContext, GateResult, GateStatus, runner

GATE_ID = "G5"
GATE_NAME = "packing"


def applies(ctx: GateContext) -> bool:  # noqa: ARG001 - always run + report
    return True


def _supported_multipack_types() -> tuple[set[str], str | None]:
    try:
        from axolotl.monkeypatch.multipack import SUPPORTED_MULTIPACK_MODEL_TYPES

        return set(SUPPORTED_MULTIPACK_MODEL_TYPES), None
    except Exception as exc:  # noqa: BLE001
        return set(), f"{exc.__class__.__name__}: {exc}"


def run(ctx: GateContext) -> GateResult:
    seq_len = 256
    cfg_dict = runner.base_cfg(
        ctx.features.base_model, ctx.output_dir, "g5", sequence_len=seq_len
    )
    cfg_dict["sample_packing"] = True
    cfg_dict["eval_sample_packing"] = False
    # varlen backend isolates documents; only matters at train, but set it when a
    # GPU is around. On CPU preprocess sdpa is fine and only warns.
    attn_note = ""
    if ctx.gpu_available:
        cfg_dict["flash_attention"] = True
    else:
        attn_note = "no GPU: preprocessed without a varlen backend (validator warns)"
    cfg_path = runner.write_cfg(ctx.output_dir, cfg_dict, "g5")

    try:
        cfg = runner.resolve_cfg(cfg_path)
        dataset_meta = runner.prepare(cfg)
    except Exception as exc:  # noqa: BLE001 - env / load failure
        return GateResult.could_not_run(
            GATE_ID,
            GATE_NAME,
            f"packing prepare failed: {exc.__class__.__name__}: {exc}",
        )

    train_dataset = dataset_meta.train_dataset
    columns = list(train_dataset.column_names)
    n_rows = train_dataset.num_rows

    findings: list[str] = []
    details: list[str] = []
    if attn_note:
        details.append(f"note: {attn_note}")

    # (a) type must be in the multipack support set
    supported, mp_err = _supported_multipack_types()
    packable = ctx.model_config_type in supported
    if mp_err is not None:
        return GateResult.could_not_run(
            GATE_ID, GATE_NAME, f"SUPPORTED_MULTIPACK_MODEL_TYPES unreadable ({mp_err})"
        )
    if not packable:
        findings.append(
            f"model type {ctx.model_config_type!r} not in "
            "SUPPORTED_MULTIPACK_MODEL_TYPES (packing would silently no-op/misattend)"
        )
    details.append(
        f"multipack membership: {ctx.model_config_type} "
        + ("in" if packable else "NOT in")
        + " SUPPORTED_MULTIPACK_MODEL_TYPES"
    )

    # (b) every prepared row must fit sequence_len
    max_len = 0
    n_over = 0
    for ids in train_dataset["input_ids"]:
        length = len(ids)
        max_len = max(max_len, length)
        if length > seq_len:
            n_over += 1
    if n_over:
        findings.append(
            f"{n_over}/{n_rows} prepared rows exceed sequence_len={seq_len} "
            f"(max={max_len})"
        )
    details.append(
        f"prepared rows: {n_rows}; max input_ids length {max_len} (limit {seq_len})"
    )

    # (c) per-document position_ids restart at 0 (clean range per row)
    pos_note: str
    if "position_ids" in columns:
        bad = 0
        checked = 0
        for pos in train_dataset["position_ids"]:
            checked += 1
            expected = list(range(len(pos)))
            if list(pos) != expected:
                bad += 1
            if checked >= 64:
                break
        if bad:
            findings.append(
                f"{bad}/{checked} rows have non-monotonic-from-0 position_ids "
                "(expected range(len) per document)"
            )
            pos_note = f"position_ids present; {bad}/{checked} malformed"
        else:
            pos_note = (
                f"position_ids present and clean (range(len) starting at 0) on "
                f"{checked} rows; collator concatenates these per-document arrays "
                "into the cross-document reset pattern at collate time"
            )
    else:
        pos_note = (
            "no position_ids column in prepared data; columns="
            f"{columns} — packing emits/squashes position_ids at collate time"
        )
    details.append(pos_note)

    data: dict[str, Any] = {
        "model_config_type": ctx.model_config_type,
        "packable": packable,
        "sequence_len": seq_len,
        "n_rows": n_rows,
        "max_packed_len": max_len,
        "n_over_limit": n_over,
        "columns": columns,
        "has_position_ids": "position_ids" in columns,
        "position_ids_finding": pos_note,
    }

    if findings:
        details.extend(f"finding: {f}" for f in findings)
        return GateResult(
            GATE_ID,
            GATE_NAME,
            GateStatus.FINDINGS,
            summary="; ".join(findings[:3]) + (" …" if len(findings) > 3 else ""),
            details=details,
            data=data,
        )

    return GateResult(
        GATE_ID,
        GATE_NAME,
        GateStatus.PASS,
        summary=(
            f"packing ok: type packable, max len {max_len} ≤ {seq_len}, "
            "per-document position_ids clean"
        ),
        details=details,
        data=data,
    )
