"""G4 — masking gate: drive the real ``chat_template`` strategy, decode trained-vs-masked spans, and assert assistant content is TRAINED and user/system MASKED (the silent-mislabel failure mode). The prepared dataset is shuffled, so the snapshot captures a shuffle-invariant sorted aggregate of every row, not row 0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import GateContext, GateResult, GateStatus, runner

GATE_ID = "G4"
GATE_NAME = "masking"

# tried in order; first that resolves + prepares wins
_CHAT_TEMPLATE_CANDIDATES = ["llama3", "tokenizer_default"]


def applies(ctx: GateContext) -> bool:  # noqa: ARG001 - always applies (text + MM)
    return True


def _prepare_with_template(ctx: GateContext, fixture: Path):
    """Try each chat_template candidate; return (cfg, dataset_meta, template) or raise the last error."""
    stanza = runner.chat_dataset_stanza(fixture)
    last_exc: Exception | None = None
    for template in _CHAT_TEMPLATE_CANDIDATES:
        cfg_dict = runner.base_cfg(
            ctx.features.base_model, ctx.output_dir, f"g4_{template}", datasets=[stanza]
        )
        cfg_dict["chat_template"] = template
        cfg_path = runner.write_cfg(ctx.output_dir, cfg_dict, f"g4_{template}")
        try:
            cfg = runner.resolve_cfg(cfg_path)
            dataset_meta = runner.prepare(cfg)
            return cfg, dataset_meta, template
        except Exception as exc:  # noqa: BLE001 - template may not resolve for arch
            last_exc = exc
    raise last_exc if last_exc else RuntimeError("no chat_template candidate tried")


def _split_spans(input_ids, labels) -> list[tuple[bool, list[int]]]:
    """Group consecutive tokens by trained (label != IGNORE) vs masked."""
    if len(input_ids) != len(labels):
        raise ValueError(f"row has {len(input_ids)} input_ids but {len(labels)} labels")
    spans: list[tuple[bool, list[int]]] = []
    cur_trained: bool | None = None
    cur: list[int] = []
    for tid, lab in zip(input_ids, labels, strict=True):
        trained = lab != runner.IGNORE_TOKEN_ID
        if trained != cur_trained:
            if cur:
                spans.append((cur_trained, cur))  # type: ignore[arg-type]
            cur, cur_trained = [], trained
        cur.append(int(tid))
    if cur:
        spans.append((cur_trained, cur))  # type: ignore[arg-type]
    return spans


def _row_norm(row, tokenizer) -> list[list[Any]]:
    """Normalized ``[[trained, text], ...]`` span structure for one prepared row."""
    decoded = [
        (trained, tokenizer.decode(ids, skip_special_tokens=False))
        for trained, ids in _split_spans(row["input_ids"], row["labels"])
    ]
    return [[bool(tr), t] for tr, t in decoded]


def _render(decoded: list[tuple[bool, str]]) -> str:
    """Trained spans in **bold**, masked spans plain."""
    out: list[str] = []
    for trained, text in decoded:
        out.append(f"**{text}**" if trained else text)
    return "".join(out)


def _fixture_role_contents(fixture: Path) -> tuple[set[str], set[str]]:
    """Union of (assistant, non-assistant) message contents across the fixture."""
    assistant: set[str] = set()
    nonassistant: set[str] = set()
    with fixture.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            for msg in json.loads(line).get("messages", []):
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if msg.get("role") == "assistant":
                    assistant.add(content)
                elif msg.get("role") in {"user", "system"}:
                    nonassistant.add(content)
    return assistant, nonassistant


def run(ctx: GateContext) -> GateResult:
    fixture = runner.write_chat_fixture(ctx.output_dir)

    try:
        _cfg, dataset_meta, template = _prepare_with_template(ctx, fixture)
    except Exception as exc:  # noqa: BLE001 - prepare/tokenizer unavailable
        return GateResult.could_not_run(
            GATE_ID,
            GATE_NAME,
            f"chat_template prepare failed: {exc.__class__.__name__}: {exc}",
        )

    try:
        from axolotl.loaders import load_tokenizer

        tokenizer = load_tokenizer(_cfg)
    except Exception as exc:  # noqa: BLE001
        return GateResult.could_not_run(
            GATE_ID,
            GATE_NAME,
            f"tokenizer load failed: {exc.__class__.__name__}: {exc}",
        )

    train_dataset = dataset_meta.train_dataset
    if train_dataset.num_rows == 0:
        return GateResult.could_not_run(
            GATE_ID, GATE_NAME, "chat fixture produced 0 prepared rows"
        )

    # shuffled dataset: scan several rows and union their trained text so role
    # coverage doesn't hinge on row 0
    n_scan = min(8, train_dataset.num_rows)
    union_trained = ""
    total_trained = 0
    first_decoded: list[tuple[bool, str]] = []
    first_norm: list[list[Any]] = []
    first_n_tokens = 0
    for i in range(n_scan):
        row = train_dataset[i]
        input_ids, labels = row["input_ids"], row["labels"]
        decoded = [
            (trained, tokenizer.decode(ids, skip_special_tokens=False))
            for trained, ids in _split_spans(input_ids, labels)
        ]
        row_trained = "".join(t for tr, t in decoded if tr)
        union_trained += "\n" + row_trained
        total_trained += sum(1 for lab in labels if lab != runner.IGNORE_TOKEN_ID)
        if i == 0:
            first_decoded = decoded
            first_norm = [[bool(tr), t] for tr, t in decoded]
            first_n_tokens = len(input_ids)

    render = _render(first_decoded)
    row0_trained = sum(
        1 for lab in train_dataset[0]["labels"] if lab != runner.IGNORE_TOKEN_ID
    )
    n_masked = first_n_tokens - row0_trained

    data: dict[str, Any] = {
        "chat_template": template,
        "rows_scanned": n_scan,
        "n_tokens": first_n_tokens,
        "n_trained": row0_trained,
        "n_masked": n_masked,
        "total_trained_tokens": total_trained,
        "render": render,
        "spans": first_norm,
        "is_multimodal": ctx.features.is_multimodal,
    }
    details = [
        f"chat_template={template}; scanned {n_scan} rows; "
        f"{total_trained} trained tokens total",
        "decoded row 0 (**trained** vs masked):",
        f"  {render}",
    ]

    findings: list[str] = []

    # --- intent: assistant TRAINED, user/system MASKED -------------------------
    assistant_contents, nonassistant_contents = _fixture_role_contents(fixture)

    if total_trained == 0:
        findings.append("no tokens trained across scanned rows — assistant span masked")
    # the silent-mislabel failure mode: any prompt (user/system) text in the loss
    for needle in sorted(nonassistant_contents):
        if needle in union_trained:
            findings.append(f"user/system content found in trained span: {needle!r}")
    if total_trained > 0 and not any(c in union_trained for c in assistant_contents):
        findings.append(
            "no assistant content in trained span — assistant text appears masked"
        )

    if ctx.features.is_multimodal:
        details.append(
            "note: multimodal — image-token masking not decoded here; "
            "only text label structure checked"
        )

    # per train_on_eos default 'turn', a trained assistant turn's terminator is trained
    if first_decoded and first_decoded[-1][0]:
        details.append(
            "note: last span is trained (assistant turn terminator trained per "
            "train_on_eos='turn' default)"
        )

    # --- snapshot diff ---------------------------------------------------------
    snap_dir = ctx.options.get("snapshot_dir")
    snapshot_note = "no snapshot_dir (diff skipped)"
    if snap_dir:
        # shuffle-invariant key: every prepared row's normalized structure, sorted
        # into a canonical order so row ordering can't perturb the diff
        snapshot_norm = sorted(
            (
                _row_norm(train_dataset[i], tokenizer)
                for i in range(train_dataset.num_rows)
            ),
            key=lambda r: json.dumps(r, ensure_ascii=False),
        )
        snap_path = Path(snap_dir) / f"{ctx.model_config_type}.json"
        data["snapshot_path"] = str(snap_path)
        if not snap_path.exists():
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            snap_path.write_text(
                json.dumps(snapshot_norm, ensure_ascii=False, indent=2), "utf-8"
            )
            snapshot_note = f"snapshot captured -> {snap_path}"
        else:
            try:
                saved = json.loads(snap_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                snap_path.write_text(
                    json.dumps(snapshot_norm, ensure_ascii=False, indent=2), "utf-8"
                )
                snapshot_note = (
                    f"prior snapshot unreadable ({exc.__class__.__name__}); recaptured"
                )
            else:
                if saved != snapshot_norm:
                    snapshot_note = f"snapshot drift vs {snap_path}"
                    findings.append(
                        f"masking structure changed vs snapshot {snap_path.name}"
                    )
                    for i, (was, now) in enumerate(
                        zip(saved, snapshot_norm, strict=False)
                    ):
                        if was != now:
                            details.append(
                                f"  snapshot diff @row {i} (sorted): was {was!r} now {now!r}"
                            )
                            break
                    if len(saved) != len(snapshot_norm):
                        details.append(
                            f"  snapshot diff: row count {len(saved)} -> {len(snapshot_norm)}"
                        )
                else:
                    snapshot_note = f"snapshot match ({snap_path.name})"
    data["snapshot_result"] = snapshot_note
    details.append(f"snapshot: {snapshot_note}")

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
            f"masking intent holds ({total_trained} trained tokens across "
            f"{n_scan} rows); {snapshot_note}"
        ),
        details=details,
        data=data,
    )
