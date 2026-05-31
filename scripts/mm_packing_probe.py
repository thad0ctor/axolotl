#!/usr/bin/env python
"""Probe multimodal processor lengths and packing efficiency with local VLMs."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoProcessor

from axolotl.prompt_strategies.multimodal_pretrain import (
    append_eos_for_processor,
    build_image_token_spec,
    compute_multimodal_processor_lengths,
)
from axolotl.utils.samplers.multipack import pack_parallel


def processor_dirs(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_file():
            root = root.parent
        for path in root.rglob("config.json"):
            model_dir = path.parent
            if (model_dir / "processor_config.json").exists() or (
                model_dir / "preprocessor_config.json"
            ).exists():
                found.append(model_dir)
    return sorted(set(found))


def synthetic_images() -> list[Image.Image]:
    specs = [
        ((64, 64), "red"),
        ((320, 180), "blue"),
        ((768, 768), "green"),
        ((1024, 768), "purple"),
        ((1536, 512), "orange"),
    ]
    return [Image.new("RGB", size, color) for size, color in specs]


def percentile(values: list[int], q: int) -> int:
    if not values:
        return 0
    return int(np.percentile(values, q))


def summarize(label: str, values: list[int]) -> None:
    if not values:
        print(f"  {label}: no values")
        return
    print(
        f"  {label}: "
        f"min={min(values)} p50={percentile(values, 50)} "
        f"p75={percentile(values, 75)} p90={percentile(values, 90)} "
        f"p95={percentile(values, 95)} p99={percentile(values, 99)} "
        f"max={max(values)} mean={np.mean(values):.1f}"
    )


def ffd_efficiency(
    lengths: list[int], capacity: int, bin_size: int
) -> tuple[int, float]:
    bins = pack_parallel(
        np.array(lengths, dtype=np.int32),
        bin_capacity=capacity,
        group_size=len(lengths),
        bin_size=bin_size,
        num_processes=1,
        safe_mode=False,
    )
    used = sum(lengths)
    slots = max(1, len(bins) * capacity)
    return len(bins), used / slots


def sequential_efficiency(lengths: list[int], capacity: int) -> tuple[int, float]:
    bins = 0
    remaining = 0
    for length in lengths:
        if length > remaining:
            bins += 1
            remaining = capacity
        remaining -= length
    used = sum(lengths)
    slots = max(1, bins * capacity)
    return bins, used / slots


def ffd_2d_efficiency(
    token_lengths: list[int],
    visual_lengths: list[int],
    token_capacity: int,
    visual_capacity: int,
) -> tuple[int, float, float, int]:
    items = sorted(
        zip(token_lengths, visual_lengths, strict=True),
        key=lambda item: max(item[0] / token_capacity, item[1] / visual_capacity),
        reverse=True,
    )
    bins: list[list[int]] = []
    over_budget = 0
    packed_token_total = 0
    packed_visual_total = 0
    for token_len, visual_len in items:
        if token_len > token_capacity or visual_len > visual_capacity:
            over_budget += 1
            continue
        for used in bins:
            if (
                used[0] + token_len <= token_capacity
                and used[1] + visual_len <= visual_capacity
            ):
                used[0] += token_len
                used[1] += visual_len
                break
        else:
            bins.append([token_len, visual_len])
        packed_token_total += token_len
        packed_visual_total += visual_len

    if not bins:
        return 0, 0.0, 0.0, over_budget
    return (
        len(bins),
        packed_token_total / (len(bins) * token_capacity),
        packed_visual_total / (len(bins) * visual_capacity),
        over_budget,
    )


def probe_model(
    model_dir: Path, sequence_len: int, micro_batch_size: int, repeats: int
):
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    spec = build_image_token_spec(processor)
    images = synthetic_images() * repeats
    texts = [
        f"{spec.image_token}\nsynthetic caption {idx} " + ("detail " * (idx % 7))
        for idx in range(len(images))
    ]
    lengths = compute_multimodal_processor_lengths(
        texts,
        [[image] for image in images],
        tokenizer=tokenizer,
        processor=processor,
    )
    capacity = sequence_len * micro_batch_size
    seq_bins, seq_eff = sequential_efficiency(lengths, capacity)
    ffd_bins, ffd_eff = ffd_efficiency(lengths, capacity, bin_size=200)
    print(f"\n{model_dir}")
    print(f"  processor: {type(processor).__name__}")
    print(f"  image token: {spec.image_token!r} id={spec.image_token_id}")
    print(
        "  lengths: "
        f"min={min(lengths)} p50={int(np.percentile(lengths, 50))} "
        f"p95={int(np.percentile(lengths, 95))} max={max(lengths)}"
    )
    print(
        f"  capacity={capacity}: sequential bins={seq_bins} eff={seq_eff:.3f}; "
        f"ffd bins={ffd_bins} eff={ffd_eff:.3f}"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def expand_data_files(data_files: Any) -> list[Path]:
    if data_files is None:
        return []
    if isinstance(data_files, str):
        paths = [data_files]
    elif isinstance(data_files, list):
        paths = [str(path) for path in data_files]
    elif isinstance(data_files, dict):
        paths = []
        for value in data_files.values():
            paths.extend(str(path) for path in value) if isinstance(
                value, list
            ) else paths.append(str(value))
    else:
        raise TypeError(f"Unsupported data_files type: {type(data_files).__name__}")

    expanded: list[Path] = []
    for path in paths:
        matches = glob.glob(path)
        expanded.extend(
            Path(match) for match in matches
        ) if matches else expanded.append(Path(path))
    return expanded


def config_data_files(cfg: dict[str, Any]) -> list[Path]:
    files: list[Path] = []
    for ds_cfg in cfg.get("pretraining_dataset") or cfg.get("datasets") or []:
        if isinstance(ds_cfg, dict):
            files.extend(
                expand_data_files(ds_cfg.get("data_files") or ds_cfg.get("path"))
            )
    return files


def row_source(image_paths: list[str]) -> str:
    for image_path in image_paths:
        parts = Path(image_path).parts
        if "output" in parts:
            idx = parts.index("output")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    joined = " ".join(image_paths)
    if "/line/" in joined:
        return "line"
    if "/page/" in joined:
        return "page"
    return "other"


def resolve_image_path(
    image_path: str, *, data_file: Path, image_base_dir: Path | None
) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    if image_base_dir is not None:
        return image_base_dir / path
    return data_file.parent / path


def iter_jsonl_rows(
    data_files: list[Path],
    *,
    limit: int,
    offset: int,
    stride: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    seen = 0
    for data_file in data_files:
        with data_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if seen < offset:
                    seen += 1
                    continue
                if (seen - offset) % stride != 0:
                    seen += 1
                    continue
                row = json.loads(line)
                row["_data_file"] = str(data_file)
                rows.append(row)
                seen += 1
                if len(rows) >= limit:
                    return rows, seen
    return rows, seen


def to_grid_rows(grid: Any) -> list[tuple[int, int, int]]:
    if grid is None:
        return []
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    return [(int(row[0]), int(row[1]), int(row[2])) for row in grid]


def process_real_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    *,
    text_column: str,
    image_column: str,
    image_base_dir: Path | None,
    image_token_id: int,
) -> list[dict[str, Any]]:
    tokenizer = processor.tokenizer
    texts: list[str] = []
    images: list[list[Image.Image]] = []
    image_counts: list[int] = []
    image_paths: list[list[str]] = []
    image_sizes: list[list[tuple[int, int]]] = []

    for row in rows:
        raw_paths = list(row.get(image_column) or [])
        resolved = [
            resolve_image_path(
                path,
                data_file=Path(row["_data_file"]),
                image_base_dir=image_base_dir,
            )
            for path in raw_paths
        ]
        loaded: list[Image.Image] = []
        sizes: list[tuple[int, int]] = []
        for path in resolved:
            image = Image.open(path).convert("RGB")
            loaded.append(image)
            sizes.append(image.size)
        texts.append(
            append_eos_for_processor(
                str(row[text_column]), tokenizer, add_eos_token=True
            )
        )
        images.append(loaded)
        image_counts.append(len(loaded))
        image_paths.append([str(path) for path in resolved])
        image_sizes.append(sizes)

    if all(count == 0 for count in image_counts):
        encoded = tokenizer(text=texts, return_tensors=None, padding=False)
    else:
        encoded = processor(
            text=texts,
            images=images,
            return_tensors=None,
            padding=False,
        )

    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    grids = to_grid_rows(encoded.get("image_grid_thw"))
    grid_idx = 0
    out: list[dict[str, Any]] = []
    for row_idx, ids in enumerate(input_ids):
        row_grids = grids[grid_idx : grid_idx + image_counts[row_idx]]
        grid_idx += image_counts[row_idx]
        visual_tokens = sum(1 for token_id in ids if token_id == image_token_id)
        patch_rows = sum(t * h * w for t, h, w in row_grids)
        out.append(
            {
                "source": row_source(image_paths[row_idx]),
                "processor_len": len(ids),
                "visual_tokens": visual_tokens,
                "text_tokens": len(ids) - visual_tokens,
                "patch_rows": patch_rows,
                "image_count": image_counts[row_idx],
                "image_sizes": image_sizes[row_idx],
                "image_grids": row_grids,
            }
        )
    return out


def probe_config_dataset(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    model_path = Path(cfg["base_model"])
    processor_kwargs = cfg.get("processor_kwargs") or {}
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=model_path.exists(),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        **processor_kwargs,
    )
    spec = build_image_token_spec(processor)

    data_files = args.data_file or config_data_files(cfg)
    if not data_files:
        raise SystemExit("No JSONL data files found from --config or --data-file.")
    rows, seen = iter_jsonl_rows(
        data_files,
        limit=args.limit,
        offset=args.offset,
        stride=args.stride,
    )
    if not rows:
        raise SystemExit("No rows selected.")

    text_column = args.text_column
    image_column = args.image_column
    image_base_dir = args.image_base_dir
    probes: list[dict[str, Any]] = []
    skipped = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        try:
            probes.extend(
                process_real_batch(
                    processor,
                    batch,
                    text_column=text_column,
                    image_column=image_column,
                    image_base_dir=image_base_dir,
                    image_token_id=spec.image_token_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            if args.strict:
                raise
            skipped += len(batch)
            print(
                f"  skipped rows {start}-{start + len(batch) - 1}: "
                f"{type(exc).__name__}: {exc}"
            )

    sequence_len = int(cfg.get("sequence_len") or args.sequence_len)
    micro_batch_size = int(cfg.get("micro_batch_size") or args.micro_batch_size)
    capacity = args.capacity or sequence_len * max(1, micro_batch_size)
    token_lengths = [int(row["processor_len"]) for row in probes]
    visual_lengths = [int(row["visual_tokens"]) for row in probes]
    patch_rows = [int(row["patch_rows"]) for row in probes]
    text_tokens = [int(row["text_tokens"]) for row in probes]

    print(f"\nconfig: {args.config}")
    print(f"model: {model_path}")
    print(f"data: {', '.join(str(path) for path in data_files)}")
    print(
        f"selected_rows={len(rows)} probed_rows={len(probes)} skipped={skipped} "
        f"last_seen_row={seen}"
    )
    print(
        f"sequence_len={sequence_len} micro_batch_size={micro_batch_size} "
        f"packed_capacity={capacity} image_token={spec.image_token!r}"
    )
    print(f"processor.size={getattr(processor.image_processor, 'size', None)}")
    summarize("processor tokens", token_lengths)
    summarize("text/wrapper tokens", text_tokens)
    summarize("Qwen image feature tokens", visual_lengths)
    summarize("pixel patch rows before merger", patch_rows)

    over = sum(1 for length in token_lengths if length > sequence_len)
    if over:
        print(f"  over sequence_len rows: {over}/{len(token_lengths)}")

    from collections import Counter

    sources = Counter(str(row["source"]) for row in probes)
    print(
        "  source mix: "
        + ", ".join(f"{source}={count}" for source, count in sources.most_common())
    )
    grids = Counter(
        tuple(row["image_grids"][0]) if row["image_grids"] else (0, 0, 0)
        for row in probes
    )
    print(f"  unique first-image grids: {len(grids)}")
    print(
        "  top grids: "
        + ", ".join(f"{grid}:{count}" for grid, count in grids.most_common(8))
    )

    seq_bins, seq_eff = sequential_efficiency(token_lengths, capacity)
    ffd_bins, ffd_eff = ffd_efficiency(
        token_lengths, capacity, bin_size=min(200, max(1, len(token_lengths)))
    )
    visual_capacity = args.visual_capacity or max(visual_lengths or [1])
    bins_2d, token_eff_2d, visual_eff_2d, over_2d = ffd_2d_efficiency(
        token_lengths,
        visual_lengths,
        capacity,
        visual_capacity,
    )
    print(
        f"  token-only sequential: bins={seq_bins} "
        f"docs/bin={len(token_lengths) / max(1, seq_bins):.2f} eff={seq_eff:.3f}"
    )
    print(
        f"  token-only FFD: bins={ffd_bins} "
        f"docs/bin={len(token_lengths) / max(1, ffd_bins):.2f} eff={ffd_eff:.3f}"
    )
    print(
        f"  2D FFD token_cap={capacity} visual_cap={visual_capacity}: "
        f"bins={bins_2d} docs/bin={len(token_lengths) / max(1, bins_2d):.2f} "
        f"token_eff={token_eff_2d:.3f} visual_eff={visual_eff_2d:.3f} "
        f"over_budget={over_2d}"
    )

    for source in sorted(sources):
        group = [row for row in probes if row["source"] == source]
        group_tokens = [int(row["processor_len"]) for row in group]
        group_visual = [int(row["visual_tokens"]) for row in group]
        print(f"\nsource={source} rows={len(group)}")
        summarize("processor tokens", group_tokens)
        summarize("Qwen image feature tokens", group_visual)
        if group_tokens:
            group_bins, group_eff = ffd_efficiency(
                group_tokens,
                capacity,
                bin_size=min(200, max(1, len(group_tokens))),
            )
            print(
                f"  token-only FFD: bins={group_bins} "
                f"docs/bin={len(group_tokens) / max(1, group_bins):.2f} "
                f"eff={group_eff:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[
            Path(p)
            for p in os.environ.get("AXOLOTL_MODEL_ROOTS", "").split(os.pathsep)
            if p
        ],
        help="Model directories to scan (or set AXOLOTL_MODEL_ROOTS).",
    )
    parser.add_argument("--sequence-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-file", type=Path, action="append")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--visual-capacity", type=int)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--image-column", default="images")
    parser.add_argument("--image-base-dir", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.config:
        probe_config_dataset(args)
        return

    models = processor_dirs(args.roots)
    if not models:
        raise SystemExit("No local processor-capable model directories found.")
    for model in models:
        try:
            probe_model(model, args.sequence_len, args.micro_batch_size, args.repeats)
        except Exception as exc:  # noqa: BLE001
            print(f"\n{model}\n  skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
