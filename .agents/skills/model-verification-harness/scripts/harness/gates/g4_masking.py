"""G4 — masking / chat-template gate: are the right tokens trained?

The highest-value run-based check. A chat-template strategy silently mislabels
when its turn-boundary logic drifts for an arch (wrong EOS/EOT token, role offset
off-by-one): training still "works" but the model learns to predict the *prompt*,
or the assistant span is fully masked and nothing is learned. Neither raises.

So G4 drives the real ``chat_template`` strategy on a tiny instruct fixture,
decodes row 0 into trained-vs-masked spans, and asserts INTENT against the
semantics in ``prompt_strategies/chat_template.py`` (default ``roles_to_train =
["assistant"]``, ``train_on_eos = "turn"``): assistant content must be TRAINED and
user/system content MASKED. A trained user/system token, or a fully-masked
assistant span, is a FINDING. When ``options["snapshot_dir"]`` is set the
normalized span structure is diffed against a captured baseline so masking
regressions surface as drift; a missing snapshot is captured on first run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import GateContext, GateResult, GateStatus, runner

GATE_ID = "G4"
GATE_NAME = "masking"

# tried in order; first that resolves + prepares wins.
_CHAT_TEMPLATE_CANDIDATES = ["llama3", "tokenizer_default"]


def applies(ctx: GateContext) -> bool:  # noqa: ARG001 - always applies (text + MM)
    return True


def _prepare_with_template(ctx: GateContext, fixture: Path):
    """Try each chat_template candidate; return (cfg, dataset_meta, template) or
    raise the last error if none resolves+prepares."""
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
    spans: list[tuple[bool, list[int]]] = []
    cur_trained: bool | None = None
    cur: list[int] = []
    for tid, lab in zip(input_ids, labels, strict=False):
        trained = lab != runner.IGNORE_TOKEN_ID
        if trained != cur_trained:
            if cur:
                spans.append((cur_trained, cur))  # type: ignore[arg-type]
            cur, cur_trained = [], trained
        cur.append(int(tid))
    if cur:
        spans.append((cur_trained, cur))  # type: ignore[arg-type]
    return spans


def _render(decoded: list[tuple[bool, str]]) -> str:
    """Trained spans in **bold** markers, masked spans plain — a human can read
    off exactly which text the loss sees."""
    out: list[str] = []
    for trained, text in decoded:
        out.append(f"**{text}**" if trained else text)
    return "".join(out)


def _fixture_role_contents(fixture: Path) -> tuple[set[str], set[str]]:
    """Union of (assistant, non-assistant) message contents across the fixture.

    The prepared dataset is shuffled, so row 0 need not be fixture line 0; we check
    intent against the whole vocabulary instead of one fixed row.
    """
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

    row = train_dataset[0]
    input_ids, labels = row["input_ids"], row["labels"]
    spans = _split_spans(input_ids, labels)
    decoded = [
        (trained, tokenizer.decode(ids, skip_special_tokens=False))
        for trained, ids in spans
    ]
    render = _render(decoded)

    n_trained = sum(1 for lab in labels if lab != runner.IGNORE_TOKEN_ID)
    n_masked = len(labels) - n_trained
    trained_text = "".join(t for tr, t in decoded if tr)

    # normalized structure for snapshotting: stable (trained, text) list.
    norm = [[bool(tr), t] for tr, t in decoded]

    data: dict[str, Any] = {
        "chat_template": template,
        "n_tokens": len(input_ids),
        "n_trained": n_trained,
        "n_masked": n_masked,
        "render": render,
        "spans": norm,
        "is_multimodal": ctx.features.is_multimodal,
    }
    details = [
        f"chat_template={template}; {len(input_ids)} tokens "
        f"({n_trained} trained / {n_masked} masked)",
        "decoded row 0 (**trained** vs masked):",
        f"  {render}",
    ]

    findings: list[str] = []

    # --- intent: assistant TRAINED, user/system MASKED -------------------------
    assistant_contents, nonassistant_contents = _fixture_role_contents(fixture)

    if n_trained == 0:
        findings.append("no tokens trained at all — assistant span fully masked")
    # the silent-mislabel failure mode: any prompt (user/system) text in the loss.
    for needle in sorted(nonassistant_contents):
        if needle in trained_text:
            findings.append(f"user/system content found in trained span: {needle!r}")
    # at least one assistant content must be trained (else the span is fully masked).
    if n_trained > 0 and not any(c in trained_text for c in assistant_contents):
        findings.append(
            "no assistant content in trained span — assistant text appears masked"
        )

    if ctx.features.is_multimodal:
        details.append(
            "note: multimodal — image-token masking not decoded here; "
            "only text label structure checked"
        )

    # EOS/EOT: per train_on_eos default 'turn', a trained assistant turn's
    # terminator is trained. Report what the last trained span ends on.
    if decoded and decoded[-1][0]:
        details.append(
            "note: last span is trained (assistant turn terminator trained per "
            "train_on_eos='turn' default)"
        )

    # --- snapshot diff ---------------------------------------------------------
    snap_dir = ctx.options.get("snapshot_dir")
    snapshot_note = "no snapshot_dir (diff skipped)"
    if snap_dir:
        snap_path = Path(snap_dir) / f"{ctx.model_config_type}.json"
        data["snapshot_path"] = str(snap_path)
        if not snap_path.exists():
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            snap_path.write_text(
                json.dumps(norm, ensure_ascii=False, indent=2), "utf-8"
            )
            snapshot_note = f"snapshot captured -> {snap_path}"
        else:
            saved = json.loads(snap_path.read_text(encoding="utf-8"))
            if saved != norm:
                snapshot_note = f"snapshot drift vs {snap_path}"
                findings.append(
                    f"masking structure changed vs snapshot {snap_path.name}"
                )
                for i, (was, now) in enumerate(zip(saved, norm, strict=False)):
                    if was != now:
                        details.append(
                            f"  snapshot diff @span {i}: was {was!r} now {now!r}"
                        )
                        break
                if len(saved) != len(norm):
                    details.append(
                        f"  snapshot diff: span count {len(saved)} -> {len(norm)}"
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
            f"masking intent holds ({n_trained} trained / {n_masked} masked); "
            f"{snapshot_note}"
        ),
        details=details,
        data=data,
    )
