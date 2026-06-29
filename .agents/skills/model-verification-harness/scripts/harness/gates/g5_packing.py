"""G5 — packing gate: drive the real prepare path with ``sample_packing: True`` and assert the type is packable, every row fits ``sequence_len``, and per-document ``position_ids`` restart at 0. ``position_ids`` are built per-document at prepare time and concatenated by the collator at collate time, so G5 checks both: each row is a clean ``range(len)`` and the real collator's concatenated ``position_ids`` reset at each document boundary."""

from __future__ import annotations

import math
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


def _check_collate_reset(cfg, train_dataset, n_docs: int = 4):
    """Drive the real multipack collator and verify concatenated ``position_ids`` reset at each document boundary; returns (status, note) ∈ verified | broken | skipped."""
    from axolotl.loaders import load_tokenizer
    from axolotl.utils.collators import V2BatchSamplerDataCollatorForSeq2Seq

    k = min(n_docs, train_dataset.num_rows)
    if k < 2:
        return "skipped", f"only {k} prepared row(s); need >=2 docs for a boundary"

    keys = ("input_ids", "attention_mask", "labels", "position_ids")
    group: list[dict[str, list[int]]] = []
    for i in range(k):
        row = train_dataset[i]
        if "position_ids" not in row:
            return "skipped", "prepared rows carry no position_ids to collate"
        group.append({key: list(row[key]) for key in keys if key in row})

    # squash_position_ids stays False (multipack default): the collator keeps each
    # document's per-document range, which is exactly the reset we assert
    tokenizer = load_tokenizer(cfg)
    collator = V2BatchSamplerDataCollatorForSeq2Seq(tokenizer=tokenizer)
    batch = collator(group)  # flat list -> one pack group of k documents
    collated = [int(x) for x in batch["position_ids"][0].tolist()]

    expected: list[int] = []
    for doc in group:
        expected.extend(range(len(doc["position_ids"])))
    total = sum(len(doc["position_ids"]) for doc in group)
    collated = collated[:total]  # ignore any right padding the collator appended

    boundaries = []
    offset = 0
    for doc in group[:-1]:
        offset += len(doc["position_ids"])
        boundaries.append(offset)

    if collated == expected and all(collated[b] == 0 for b in boundaries):
        return (
            "verified",
            f"collate-time reset verified: {k} docs concatenated, position_ids "
            f"restart at 0 at {len(boundaries)} document boundary(ies)",
        )
    return (
        "broken",
        f"collated position_ids do NOT reset at document boundaries (boundaries "
        f"{boundaries}; got {collated[:24]}…)",
    )


def _packing_loss_parity(ctx: GateContext, seq_len: int) -> dict[str, Any]:
    """Packing must not change the math: step-0 loss with sample_packing on vs off over the SAME tokens must match within ``pack_rtol``. Needs a GPU (varlen attention is what isolates packed documents). Returns a status dict; never raises."""
    from . import g6_loss  # subprocess train worker, isolates global multipack patches

    rtol = float(ctx.options.get("pack_rtol", 2e-2))
    timeout = float(ctx.options.get("train_timeout", 1200))
    n_docs = 2
    fixture = runner.write_parity_fixture(ctx.output_dir, n_docs)
    stanza = runner.completion_dataset_stanza(fixture)

    def _cfg(name: str, packing: bool, micro_batch_size: int) -> dict[str, Any]:
        cfg = runner.base_cfg(
            ctx.features.base_model,
            ctx.output_dir,
            name,
            datasets=[stanza],
            sequence_len=seq_len,
        )
        cfg.update(
            {
                "seed": ctx.seed,
                "max_steps": 1,
                "logging_steps": 1,
                "save_strategy": "no",
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": 1,
                "bf16": True,
                "flash_attention": True,  # varlen backend: isolates packed documents
                "sample_packing": packing,
                "eval_sample_packing": False,
            }
        )
        return cfg

    # _spawn_variant honors ctx.options["gpus"] per-subprocess, so no parent env pin needed.
    # packed: one pack/step; unpacked: a single batch of all docs -> identical token set at step 0
    res_packed = g6_loss._spawn_variant(
        ctx, "g5_packed", _cfg("g5_packed", True, 1), timeout
    )
    res_unpacked = g6_loss._spawn_variant(
        ctx, "g5_unpacked", _cfg("g5_unpacked", False, n_docs), timeout
    )

    def _step0(res: dict[str, Any]) -> float | None:
        losses = res.get("loss_history") or []
        return float(losses[0]) if losses else None

    lp, lu = _step0(res_packed), _step0(res_unpacked)
    if lp is None or lu is None:
        err = res_packed.get("error") if lp is None else res_unpacked.get("error")
        return {
            "status": "unverified",
            "note": f"packing-vs-unpacked step-0 loss unavailable ({err})",
        }
    if not (math.isfinite(lp) and math.isfinite(lu)):
        return {
            "status": "unverified",
            "note": f"non-finite step-0 loss (packed={lp}, unpacked={lu}) — cannot compare",
            "packed_step0": lp,
            "unpacked_step0": lu,
        }
    rel = abs(lp - lu) / abs(lu) if lu else float("inf")
    status = "verified" if rel <= rtol else "mismatch"
    note = (
        f"packed step0={lp:.6f} vs unpacked step0={lu:.6f}, rel={rel:.2e} "
        f"{'<=' if status == 'verified' else '>'} pack_rtol {rtol:.0e}"
        + ("" if status == "verified" else " — packing changes loss")
    )
    return {
        "status": status,
        "note": note,
        "packed_step0": lp,
        "unpacked_step0": lu,
        "rel_delta": rel,
        "pack_rtol": rtol,
    }


def run(ctx: GateContext) -> GateResult:
    seq_len = 256
    fixture = runner.write_completion_fixture(ctx.output_dir)
    cfg_dict = runner.base_cfg(
        ctx.features.base_model,
        ctx.output_dir,
        "g5",
        datasets=[runner.completion_dataset_stanza(fixture)],
        sequence_len=seq_len,
    )
    cfg_dict["sample_packing"] = True
    cfg_dict["eval_sample_packing"] = False
    # varlen backend isolates documents; only matters at train, so set it only with a GPU
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

    # zero prepared rows would make the length / position_ids loops no-op and PASS
    if n_rows == 0:
        return GateResult.could_not_run(
            GATE_ID,
            GATE_NAME,
            "packing produced 0 prepared rows — no packed sample to validate",
        )

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
                f"{checked} rows (per-document building block for the collate reset)"
            )
    else:
        pos_note = (
            "no position_ids column in prepared data; columns="
            f"{columns} — packing emits/squashes position_ids at collate time"
        )
    details.append(pos_note)

    # (d) collate-time cross-document reset: the per-row check above is true by
    # construction at prepare time and doesn't exercise the collator that can regress
    try:
        collate_status, collate_note = _check_collate_reset(cfg, train_dataset)
    except Exception as exc:  # noqa: BLE001 - collator wiring may be unavailable
        collate_status = "skipped"
        collate_note = f"collate-time reset not verified (collator unavailable: {exc})"
    if collate_status == "skipped":
        return GateResult.could_not_run(GATE_ID, GATE_NAME, collate_note)
    if collate_status == "broken":
        findings.append(collate_note)
    details.append(f"collate reset: {collate_note}")

    # (e) packing must not change the math: step-0 loss packed-vs-unpacked parity.
    # Opt-in (profile=full or pack_parity) — it spawns two train forwards, so it
    # must not slow the default structural G5 on every GPU box.
    want_parity = ctx.profile == "full" or bool(ctx.options.get("pack_parity"))
    if not want_parity:
        parity = {
            "status": "skipped",
            "note": "parity not requested (profile=full or --features pack_parity)",
        }
    elif ctx.gpu_available:
        try:
            parity = _packing_loss_parity(ctx, seq_len)
        except Exception as exc:  # noqa: BLE001 - train worker wiring may be unavailable
            parity = {"status": "unverified", "note": f"loss parity not run ({exc})"}
    else:
        parity = {
            "status": "skipped",
            "note": "no GPU: packing-vs-unpacked loss parity not run",
        }
    if parity["status"] == "mismatch":
        findings.append(parity["note"])
    details.append(f"loss parity: {parity['note']}")

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
        "collate_reset_status": collate_status,
        "collate_reset_note": collate_note,
        "loss_parity": parity,
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

    reset_claim = "cross-document position_ids reset verified at collate time"
    return GateResult(
        GATE_ID,
        GATE_NAME,
        GateStatus.PASS,
        summary=(
            f"packing ok: type packable, max len {max_len} ≤ {seq_len}, {reset_claim}; "
            f"loss parity {parity['status']}"
        ),
        details=details,
        data=data,
    )
