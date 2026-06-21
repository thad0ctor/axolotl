#!/usr/bin/env python
"""Probe multimodal visual shape entropy and visual padding policies."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from transformers import AutoProcessor
from transformers.image_utils import load_image

from axolotl.prompt_strategies.multimodal_pretrain import (
    append_eos_for_processor,
    build_image_token_spec,
)
from axolotl.utils.data.mm_image import resize_image_for_processor
from axolotl.utils.data.mm_packing import pack_2d_first_fit_decreasing


@dataclass(frozen=True)
class Policy:
    name: str
    processor_kwargs: dict[str, Any]
    image_size: int | tuple[int, int] | None = None
    image_resize_buckets: list[tuple[int, int]] | None = None
    image_resize_no_upscale: bool = False
    image_resize_pad_color: Any | None = None


@dataclass(frozen=True)
class RowShape:
    token_len: int
    feature_tokens: int
    pixel_rows: int
    image_count: int
    signature: tuple[tuple[int, int, int], ...]


def load_yaml(path: Path) -> dict[str, Any]:
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
            if isinstance(value, list):
                paths.extend(str(path) for path in value)
            else:
                paths.append(str(value))
    else:
        raise TypeError(f"Unsupported data_files type: {type(data_files).__name__}")

    expanded: list[Path] = []
    for path in paths:
        matches = glob.glob(path)
        if matches:
            expanded.extend(Path(match) for match in matches)
        else:
            expanded.append(Path(path))
    return expanded


def config_data_files(cfg: dict[str, Any]) -> list[Path]:
    files: list[Path] = []
    for ds_cfg in cfg.get("pretraining_dataset") or cfg.get("datasets") or []:
        if isinstance(ds_cfg, dict):
            files.extend(
                expand_data_files(ds_cfg.get("data_files") or ds_cfg.get("path"))
            )
    return files


def load_rows(
    data_file: Path,
    *,
    limit: int,
    offset: int,
    stride: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    seen = 0
    with data_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if seen < offset:
                seen += 1
                continue
            if (seen - offset) % stride != 0:
                seen += 1
                continue
            rows.append(json.loads(line))
            seen += 1
            if len(rows) >= limit:
                break
    return rows, seen


def resolve_image_source(src: Any, image_base_dir: str | None) -> Any:
    if (
        image_base_dir
        and isinstance(src, str)
        and not Path(src).is_absolute()
        and "://" not in src
    ):
        return str(Path(image_base_dir) / src)
    return src


def parse_resize_bucket(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resize buckets must look like 1024x1536")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resize bucket dimensions must be positive")
    return width, height


def load_row_images(
    row: dict[str, Any],
    *,
    image_base_dir: str | None,
    policy: Policy,
) -> list[Image.Image]:
    images = []
    for src in row.get("images") or []:
        image = load_image(resolve_image_source(src, image_base_dir))
        image = resize_image_for_processor(
            image,
            policy.image_size,
            Image.Resampling.BICUBIC,
            policy.image_resize_buckets,
            policy.image_resize_no_upscale,
            policy.image_resize_pad_color,
        )
        images.append(image)
    return images


def grid_rows(value: Any) -> list[tuple[int, int, int]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [(int(row[0]), int(row[1]), int(row[2])) for row in value]


def row_shape(
    row: dict[str, Any],
    *,
    processor,
    image_token: str,
    image_base_dir: str | None,
    policy: Policy,
) -> RowShape:
    text = append_eos_for_processor(
        row["text"].replace("<image>", image_token),
        processor.tokenizer,
        add_eos_token=True,
    )
    images = load_row_images(
        row,
        image_base_dir=image_base_dir,
        policy=policy,
    )
    if images:
        encoded = processor(
            text=[text],
            images=[images],
            return_tensors=None,
            padding=False,
            **policy.processor_kwargs,
        )
    else:
        encoded = processor.tokenizer(text=[text], return_tensors=None, padding=False)

    ids = encoded["input_ids"][0]
    grids = grid_rows(encoded.get("image_grid_thw"))
    merge_size = getattr(processor.image_processor, "merge_size", 1)
    pixel_rows = sum(t * h * w for t, h, w in grids)
    feature_tokens = sum((t * h * w) // (merge_size**2) for t, h, w in grids)
    return RowShape(
        token_len=len(ids),
        feature_tokens=feature_tokens,
        pixel_rows=pixel_rows,
        image_count=len(images),
        signature=tuple(grids),
    )


def percentile(values: list[int], q: int) -> int:
    if not values:
        return 0
    return int(np.percentile(values, q))


def summarize_values(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean": float(np.mean(values)),
    }


def static_bucket_summary(values: list[int], buckets: list[int]) -> dict[str, Any]:
    if not values or not buckets:
        return {}
    assigned: list[int] = []
    overflow = 0
    sorted_buckets = sorted(buckets)
    for value in values:
        for bucket in sorted_buckets:
            if value <= bucket:
                assigned.append(bucket)
                break
        else:
            overflow += 1
    used = sum(values) or 1
    padded = sum(assigned) or 1
    return {
        "buckets": sorted_buckets,
        "overflow": overflow,
        "unique_assigned_buckets": len(set(assigned)),
        "padding_overhead": padded / used,
        "assigned_counts": Counter(assigned).most_common(),
    }


def simulate_packing(
    shapes: list[RowShape],
    *,
    sequence_len: int,
    visual_capacity: int,
    visual_pad_buckets: list[int],
) -> dict[str, Any]:
    usable = [
        shape
        for shape in shapes
        if shape.token_len <= sequence_len and shape.feature_tokens <= visual_capacity
    ]
    dropped = len(shapes) - len(usable)
    if not usable:
        return {
            "visual_capacity": visual_capacity,
            "usable_rows": 0,
            "dropped_rows": dropped,
        }
    bins = pack_2d_first_fit_decreasing(
        [shape.token_len for shape in usable],
        [shape.feature_tokens for shape in usable],
        token_capacity=sequence_len,
        visual_capacity=visual_capacity,
    )
    packed_tokens = [sum(usable[idx].token_len for idx in bin_) for bin_ in bins]
    packed_features = [sum(usable[idx].feature_tokens for idx in bin_) for bin_ in bins]
    packed_pixels = [sum(usable[idx].pixel_rows for idx in bin_) for bin_ in bins]
    docs_per_pack = [len(bin_) for bin_ in bins]
    return {
        "visual_capacity": visual_capacity,
        "usable_rows": len(usable),
        "dropped_rows": dropped,
        "packed_rows": len(bins),
        "docs_per_packed_row": len(usable) / max(1, len(bins)),
        "docs_per_pack": summarize_values(docs_per_pack),
        "token_efficiency": sum(packed_tokens) / max(1, len(bins) * sequence_len),
        "visual_efficiency": sum(packed_features) / max(1, len(bins) * visual_capacity),
        "unique_feature_totals": len(set(packed_features)),
        "unique_pixel_totals": len(set(packed_pixels)),
        "feature_totals": summarize_values(packed_features),
        "pixel_rows": summarize_values(packed_pixels),
        "static_feature_padding": static_bucket_summary(
            packed_features,
            visual_pad_buckets,
        ),
    }


def build_policies(args: argparse.Namespace, processor) -> list[Policy]:
    policies = [Policy("default", {})]
    min_pixels = args.min_pixels
    if min_pixels is None:
        min_pixels = int(processor.image_processor.size.shortest_edge)
    for max_pixels in args.max_pixels:
        policies.append(
            Policy(
                f"cap_{int(math.sqrt(max_pixels))}px_area",
                {"min_pixels": min_pixels, "max_pixels": max_pixels},
            )
        )
    for area in args.fixed_areas:
        policies.append(
            Policy(
                f"area_{int(math.sqrt(area))}px_minmax",
                {"min_pixels": area, "max_pixels": area},
            )
        )
    for size in args.square_sizes:
        policies.append(
            Policy(
                f"square_{size}",
                {},
                image_size=size,
                image_resize_no_upscale=args.image_resize_no_upscale,
                image_resize_pad_color=args.image_resize_pad_color,
            )
        )
    for bucket_group in args.image_resize_bucket_groups or []:
        policies.append(
            Policy(
                "buckets_" + "_".join(f"{w}x{h}" for w, h in bucket_group),
                {},
                image_resize_buckets=list(bucket_group),
                image_resize_no_upscale=args.image_resize_no_upscale,
                image_resize_pad_color=args.image_resize_pad_color,
            )
        )
    return policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sequence-len", type=int, default=None)
    parser.add_argument("--image-base-dir", type=str, default=None)
    parser.add_argument(
        "--max-pixels",
        type=int,
        nargs="*",
        default=[512 * 512, 768 * 768, 1024 * 1024],
    )
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument(
        "--fixed-areas",
        type=int,
        nargs="*",
        default=[512 * 512, 768 * 768, 1024 * 1024],
    )
    parser.add_argument(
        "--square-sizes",
        type=int,
        nargs="*",
        default=[512, 768, 1024],
    )
    parser.add_argument(
        "--image-resize-bucket-groups",
        type=parse_resize_bucket,
        nargs="*",
        action="append",
        help=(
            "One fixed-canvas bucket group per flag. Example: "
            "--image-resize-bucket-groups 1024x1536 1536x1536"
        ),
    )
    parser.add_argument("--image-resize-no-upscale", action="store_true")
    parser.add_argument("--image-resize-pad-color", default="white")
    parser.add_argument(
        "--visual-capacities",
        type=int,
        nargs="*",
        default=[1024, 2048, 4096, 8192, 12000],
    )
    parser.add_argument(
        "--visual-pad-buckets",
        type=int,
        nargs="*",
        default=[1024, 2048, 4096, 8192, 12000],
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_files = config_data_files(cfg)
    if not data_files:
        raise ValueError("No data files found in config.")
    sequence_len = args.sequence_len or int(cfg.get("sequence_len") or 2048)
    processor = AutoProcessor.from_pretrained(
        str(args.model),
        local_files_only=True,
        trust_remote_code=True,
    )
    spec = build_image_token_spec(processor)
    rows, seen = load_rows(
        data_files[0],
        limit=args.limit,
        offset=args.offset,
        stride=args.stride,
    )

    results: list[dict[str, Any]] = []
    for policy in build_policies(args, processor):
        shapes = [
            row_shape(
                row,
                processor=processor,
                image_token=spec.image_token,
                image_base_dir=args.image_base_dir,
                policy=policy,
            )
            for row in rows
        ]
        feature_tokens = [shape.feature_tokens for shape in shapes]
        pixel_rows = [shape.pixel_rows for shape in shapes]
        token_lens = [shape.token_len for shape in shapes]
        signatures = Counter(shape.signature for shape in shapes)
        policy_result = {
            "policy": policy.name,
            "rows": len(rows),
            "seen_source_rows": seen,
            "sequence_len": sequence_len,
            "token_lengths": summarize_values(token_lens),
            "feature_tokens": summarize_values(feature_tokens),
            "pixel_rows": summarize_values(pixel_rows),
            "unique_signatures": len(signatures),
            "top_signatures": signatures.most_common(8),
            "over_sequence_len": sum(length > sequence_len for length in token_lens),
            "packing": [
                simulate_packing(
                    shapes,
                    sequence_len=sequence_len,
                    visual_capacity=capacity,
                    visual_pad_buckets=args.visual_pad_buckets,
                )
                for capacity in args.visual_capacities
            ],
        }
        results.append(policy_result)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for result in results:
        print(f"\npolicy={result['policy']} rows={result['rows']}")
        print(
            "  token_len "
            f"p50={result['token_lengths']['p50']} "
            f"p95={result['token_lengths']['p95']} "
            f"p99={result['token_lengths']['p99']} "
            f"max={result['token_lengths']['max']} "
            f"over_seq={result['over_sequence_len']}"
        )
        print(
            "  feature_tokens "
            f"p50={result['feature_tokens']['p50']} "
            f"p95={result['feature_tokens']['p95']} "
            f"p99={result['feature_tokens']['p99']} "
            f"max={result['feature_tokens']['max']}"
        )
        print(
            "  pixel_rows "
            f"p50={result['pixel_rows']['p50']} "
            f"p95={result['pixel_rows']['p95']} "
            f"p99={result['pixel_rows']['p99']} "
            f"max={result['pixel_rows']['max']} "
            f"unique_signatures={result['unique_signatures']}"
        )
        for pack in result["packing"]:
            if pack.get("usable_rows", 0) == 0:
                print(
                    f"  pack visual_cap={pack['visual_capacity']}: "
                    f"usable=0 dropped={pack['dropped_rows']}"
                )
                continue
            static = pack["static_feature_padding"]
            print(
                f"  pack visual_cap={pack['visual_capacity']}: "
                f"rows={pack['packed_rows']} docs/row={pack['docs_per_packed_row']:.2f} "
                f"tok_eff={pack['token_efficiency']:.3f} "
                f"vis_eff={pack['visual_efficiency']:.3f} "
                f"unique_vis={pack['unique_feature_totals']} "
                f"pad_buckets={static.get('unique_assigned_buckets', 0)} "
                f"pad_overhead={static.get('padding_overhead', 0):.2f} "
                f"dropped={pack['dropped_rows']}"
            )


if __name__ == "__main__":
    main()
