#!/usr/bin/env python
"""Benchmark multimodal CPT packing variants on a real processor/model."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image

from axolotl.prompt_strategies.multimodal_pretrain import (
    build_image_token_spec,
    encode_multimodal_pretrain,
)
from axolotl.utils.collators.mm_pretrain import MultiModalPretrainDataCollator
from axolotl.utils.data.mm_packing import (
    MultimodalPackingMetadataCache,
    compute_multimodal_packing_metadata,
)
from axolotl.utils.data.mm_tiling import (
    ImageTilingConfig,
    replace_first_image_placeholder,
    tile_image_for_processor,
)
from axolotl.utils.data.streaming import encode_packed_streaming_multimodal


@dataclass(frozen=True)
class Variant:
    name: str
    packed: bool
    cache_path: str | None = None
    ram_budget_mb: int | None = 0
    visual_capacity: int | None = None
    group_by_visual_signature: bool = False


class _NoopModel(torch.nn.Module):
    def forward(self, **kwargs):  # type: ignore[override]
        input_ids = kwargs["input_ids"]
        labels = kwargs.get("labels")
        vocab = 1
        logits = torch.zeros(
            (*input_ids.shape, vocab),
            dtype=torch.float32,
            device=input_ids.device,
        )
        loss = (
            torch.zeros((), dtype=torch.float32, device=input_ids.device)
            if labels is not None
            else None
        )
        return type("Output", (), {"logits": logits, "loss": loss})()


def load_row_indices(path: Path) -> set[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text[0] == "[":
        return {int(value) for value in json.loads(text)}
    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def load_rows(
    data_file: Path,
    limit: int,
    offset: int,
    stride: int,
    allowed_row_indices: set[int] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    seen = 0
    with data_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if seen < offset:
                seen += 1
                continue
            if allowed_row_indices is not None and seen not in allowed_row_indices:
                seen += 1
                continue
            if (seen - offset) % stride != 0:
                seen += 1
                continue
            row = json.loads(line)
            row["_bench_row_index"] = seen
            rows.append(row)
            seen += 1
            if len(rows) >= limit:
                break
    return rows


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def count_jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m {secs}s"
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def parse_resize_algorithm(value: str | None) -> Image.Resampling | None:
    if value is None:
        return None
    normalized = value.lower()
    if normalized == "bilinear":
        return Image.Resampling.BILINEAR
    if normalized == "bicubic":
        return Image.Resampling.BICUBIC
    if normalized == "lanczos":
        return Image.Resampling.LANCZOS
    raise argparse.ArgumentTypeError(
        "--image-resize-algorithm must be one of: bilinear, bicubic, lanczos"
    )


def parse_resize_bucket(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resize buckets must look like 1024x1536")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resize bucket dimensions must be positive")
    return width, height


def parse_tile_grid(value: str) -> tuple[int, int]:
    return parse_resize_bucket(value)


def parse_pad_color(value: str | None):
    if value is None:
        return None
    if "," not in value:
        return value
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) not in (1, 3):
        raise argparse.ArgumentTypeError(
            "--image-resize-pad-color must be a name or R,G,B"
        )
    return parts[0] if len(parts) == 1 else parts


def build_tiled_rows(
    rows: list[dict],
    *,
    image_token: str,
    tile_grid: tuple[int, int] | None,
    tile_size: int | None,
    tile_overlap: float,
    tile_min_area: int,
    tile_overview_size: int | None,
    tile_overview_buckets: list[tuple[int, int]] | None,
    tile_reading_order: str,
    image_resize_algorithm: Image.Resampling | None,
    image_resize_pad_color,
) -> list[dict]:
    if tile_grid is None:
        return rows
    if tile_size is None:
        raise ValueError("--tile-size is required when --tile-grid is set")

    config = ImageTilingConfig(
        tile_size=tile_size,
        grid=tile_grid,
        overlap=tile_overlap,
        min_area=tile_min_area,
        overview_size=tile_overview_size,
        overview_buckets=tile_overview_buckets,
        no_upscale=True,
        pad_color=image_resize_pad_color,
        reading_order=tile_reading_order,
    )
    tiled_rows: list[dict] = []
    for row_idx, row in enumerate(rows):
        images = row.get("images") or []
        if len(images) != 1:
            tiled_rows.append(row)
            continue
        image = load_image(images[0])
        tiled_images = tile_image_for_processor(
            image,
            config,
            resize_algorithm=image_resize_algorithm,
        )
        new_row = dict(row)
        new_row["images"] = tiled_images
        new_row["text"] = replace_first_image_placeholder(
            row["text"],
            image_token=image_token,
            count=len(tiled_images),
        )
        new_row["_tile_count"] = len(tiled_images)
        new_row["_tile_source_row"] = row_idx
        tiled_rows.append(new_row)
    return tiled_rows


def rows_with_metadata(
    rows: list[dict],
    *,
    processor,
    image_token_id: int,
    image_base_dir: str | None,
    cache_path: str | None,
    ram_budget_mb: int | None,
    image_size: int | None,
    image_resize_algorithm: Image.Resampling | None,
    image_resize_buckets: list[tuple[int, int]] | None,
    image_resize_no_upscale: bool,
    image_resize_pad_color,
) -> list[tuple[dict, Any]]:
    cache = MultimodalPackingMetadataCache(cache_path, ram_budget_mb)
    metadata = compute_multimodal_packing_metadata(
        [row["text"] for row in rows],
        [row["images"] for row in rows],
        tokenizer=processor.tokenizer,
        processor=processor,
        image_token_id=image_token_id,
        image_base_dir=image_base_dir,
        image_size=image_size,
        image_resize_algorithm=image_resize_algorithm,
        image_resize_buckets=image_resize_buckets,
        image_resize_no_upscale=image_resize_no_upscale,
        image_resize_pad_color=image_resize_pad_color,
        cache=cache,
    )
    return list(zip(rows, metadata, strict=True))


def build_baseline_rows(
    rows: list[dict],
    *,
    processor,
    max_tokens: int,
    image_token: str,
    image_token_id: int,
    image_base_dir: str | None,
) -> list[dict]:
    encoded = encode_multimodal_pretrain(
        {
            "text": [row["text"] for row in rows],
            "images": [row["images"] for row in rows],
        },
        tokenizer=processor.tokenizer,
        max_tokens=max_tokens,
        image_token=image_token,
        image_token_id=image_token_id,
        text_column="text",
        image_column="images",
        image_base_dir=image_base_dir,
        enforce_max_length=True,
    )
    features = []
    for idx in range(len(encoded["_mm_text"])):
        feature = {key: encoded[key][idx] for key in encoded}
        if "_bench_row_index" in rows[idx]:
            feature["_bench_row_index"] = rows[idx]["_bench_row_index"]
        if "_tile_count" in rows[idx]:
            feature["_tile_count"] = rows[idx]["_tile_count"]
        features.append(feature)
    return features


def build_packed_rows(
    rows: list[dict],
    *,
    processor,
    max_tokens: int,
    image_token: str,
    image_token_id: int,
    image_base_dir: str | None,
    variant: Variant,
    bin_size: int,
    image_size: int | None,
    image_resize_algorithm: Image.Resampling | None,
    image_resize_buckets: list[tuple[int, int]] | None,
    image_resize_no_upscale: bool,
    image_resize_pad_color,
) -> list[dict]:
    encoded = encode_packed_streaming_multimodal(
        {
            "text": [row["text"] for row in rows],
            "images": [row["images"] for row in rows],
        },
        tokenizer=processor.tokenizer,
        processor=processor,
        max_tokens=max_tokens,
        batch_size=1,
        bin_size=bin_size,
        image_token=image_token,
        image_token_id=image_token_id,
        image_base_dir=image_base_dir,
        metadata_cache_path=variant.cache_path,
        metadata_cache_ram_budget_mb=variant.ram_budget_mb,
        visual_capacity=variant.visual_capacity,
        group_by_visual_signature=variant.group_by_visual_signature,
        use_multimodal_sample_packing=True,
        image_size=image_size,
        image_resize_algorithm=image_resize_algorithm,
        image_resize_buckets=image_resize_buckets,
        image_resize_no_upscale=image_resize_no_upscale,
        image_resize_pad_color=image_resize_pad_color,
    )
    features = []
    for idx in range(len(encoded["_mm_text"])):
        feature = {key: encoded[key][idx] for key in encoded}
        if "_bench_row_index" in rows[idx]:
            feature["_bench_row_index"] = rows[idx]["_bench_row_index"]
        if "_tile_count" in rows[idx]:
            feature["_tile_count"] = rows[idx]["_tile_count"]
        features.append(feature)
    return features


def batch_shapes(batch: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    input_shape = tuple(batch["input_ids"].shape)
    pixel_values = batch.get("pixel_values")
    visual_shape = tuple(pixel_values.shape) if pixel_values is not None else ()
    return input_shape, visual_shape


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    *,
    include_labels: bool,
) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if key == "labels" and not include_labels:
            continue
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def shifted_label_token_count(labels: torch.Tensor, ignore_index: int = -100) -> int:
    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    return int(shift_labels.ne(ignore_index).sum().item())


def unwrap_model_for_chunked_loss(model):
    candidates = [model]
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        peft_inner = getattr(base_model, "model", None)
        if peft_inner is not None:
            candidates.append(peft_inner)
        candidates.append(base_model)

    for candidate in candidates:
        if (
            hasattr(candidate, "model")
            and hasattr(candidate, "lm_head")
            and not hasattr(candidate.model, "lm_head")
        ):
            return candidate
    return None


def chunked_causal_lm_loss(
    model,
    inputs: dict[str, Any],
    *,
    chunk_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    labels = inputs["labels"]
    raw_model = unwrap_model_for_chunked_loss(model)
    if raw_model is None:
        outputs = model(**inputs)
        if getattr(outputs, "loss", None) is None:
            raise RuntimeError("Model did not return loss with labels provided.")
        return outputs.loss

    model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
    outputs = raw_model.model(**model_inputs)
    hidden_states = outputs[0]

    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].reshape(-1)
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    valid = shift_labels.ne(ignore_index)
    if not bool(valid.any().item()):
        return hidden_states.new_zeros((), dtype=torch.float32)

    valid_hidden = hidden_states[valid]
    valid_labels = shift_labels[valid]
    loss_sum = valid_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, valid_hidden.shape[0], chunk_size):
        end = min(start + chunk_size, valid_hidden.shape[0])
        logits = raw_model.lm_head(valid_hidden[start:end]).float()
        loss_sum = loss_sum + F.cross_entropy(
            logits,
            valid_labels[start:end],
            reduction="sum",
        )
        del logits
    return loss_sum / valid_labels.numel()


def model_forward(
    model,
    inputs: dict[str, Any],
    *,
    logits_to_keep: int | None,
    measure_loss: bool,
    loss_mode: str,
    loss_chunk_size: int,
):
    if measure_loss:
        if loss_mode == "chunked":
            loss = chunked_causal_lm_loss(
                model,
                inputs,
                chunk_size=loss_chunk_size,
            )
            return type("Output", (), {"logits": None, "loss": loss})()
        return model(**inputs)
    try:
        return model(**inputs, logits_to_keep=logits_to_keep)
    except TypeError:
        return model(**inputs)


def run_variant(
    variant: Variant,
    rows: list[dict],
    *,
    processor,
    model,
    max_tokens: int,
    image_token: str,
    image_token_id: int,
    image_base_dir: str | None,
    bin_size: int,
    max_steps: int,
    warmup_steps: int,
    logits_to_keep: int | None,
    image_size: int | None,
    image_resize_algorithm: Image.Resampling | None,
    image_resize_buckets: list[tuple[int, int]] | None,
    image_resize_no_upscale: bool,
    image_resize_pad_color,
    device: torch.device,
    progress: bool,
    measure_loss: bool,
    loss_mode: str,
    loss_chunk_size: int,
) -> dict[str, Any]:
    collator = MultiModalPretrainDataCollator(
        tokenizer=processor.tokenizer,
        processor=processor,
        image_token_spec=build_image_token_spec(processor, override=image_token),
        image_base_dir=image_base_dir,
        sample_packing=variant.packed,
        pad_to_multiple_of=max_tokens,
        max_length=max_tokens,
        image_size=image_size,
        image_resize_algorithm=image_resize_algorithm,
        image_resize_buckets=image_resize_buckets,
        image_resize_no_upscale=image_resize_no_upscale,
        image_resize_pad_color=image_resize_pad_color,
    )
    build_start = time.perf_counter()
    if variant.packed:
        features = build_packed_rows(
            rows,
            processor=processor,
            max_tokens=max_tokens,
            image_token=image_token,
            image_token_id=image_token_id,
            image_base_dir=image_base_dir,
            variant=variant,
            bin_size=bin_size,
            image_size=image_size,
            image_resize_algorithm=image_resize_algorithm,
            image_resize_buckets=image_resize_buckets,
            image_resize_no_upscale=image_resize_no_upscale,
            image_resize_pad_color=image_resize_pad_color,
        )
    else:
        features = build_baseline_rows(
            rows,
            processor=processor,
            max_tokens=max_tokens,
            image_token=image_token,
            image_token_id=image_token_id,
            image_base_dir=image_base_dir,
        )
    build_s = time.perf_counter() - build_start

    warmup_iter = features[:warmup_steps]
    for feature in warmup_iter:
        batch = collator.torch_call([feature])
        inputs = move_batch_to_device(batch, device, include_labels=measure_loss)
        with torch.inference_mode():
            outputs = model_forward(
                model,
                inputs,
                logits_to_keep=logits_to_keep,
                measure_loss=measure_loss,
                loss_mode=loss_mode,
                loss_chunk_size=loss_chunk_size,
            )
            if measure_loss and getattr(outputs, "loss", None) is not None:
                _ = outputs.loss.float().item()
            else:
                _ = outputs.logits[:, -1, :].float().sum().item()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    feature_iter = features[warmup_steps : warmup_steps + max_steps]
    shape_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    total_tokens = 0
    total_visual_rows = 0
    collate_s = 0.0
    forward_s = 0.0
    step_seconds: list[float] = []
    collate_seconds: list[float] = []
    forward_seconds: list[float] = []
    docs_per_timed_step: list[int] = []
    loss_values: list[float] = []
    label_tokens_per_timed_step: list[int] = []
    weighted_loss_sum = 0.0
    total_label_tokens = 0
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step_idx, feature in enumerate(feature_iter, start=1):
        step_start = time.perf_counter()
        start = time.perf_counter()
        if variant.packed:
            batch = collator.torch_call([feature])
        else:
            batch = collator.torch_call([feature])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        collate_elapsed = time.perf_counter() - start
        collate_s += collate_elapsed
        shape_counts[batch_shapes(batch)] += 1
        total_tokens += int(batch["attention_mask"].ne(0).sum().item())
        label_tokens = (
            shifted_label_token_count(batch["labels"]) if "labels" in batch else 0
        )
        label_tokens_per_timed_step.append(label_tokens)
        if "pixel_values" in batch:
            total_visual_rows += int(batch["pixel_values"].shape[0])
        docs_per_timed_step.append(
            len(feature.get("_mm_sample_lengths", [1])) if variant.packed else 1
        )

        inputs = move_batch_to_device(batch, device, include_labels=measure_loss)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model_forward(
                model,
                inputs,
                logits_to_keep=logits_to_keep,
                measure_loss=measure_loss,
                loss_mode=loss_mode,
                loss_chunk_size=loss_chunk_size,
            )
            if measure_loss:
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    raise RuntimeError("Model did not return loss with labels provided.")
                loss_value = float(loss.detach().float().item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"Non-finite loss at step {step_idx}: {loss_value}")
                loss_values.append(loss_value)
                weighted_loss_sum += loss_value * label_tokens
                total_label_tokens += label_tokens
                _ = loss_value
            else:
                _ = outputs.logits[:, -1, :].float().sum().item()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_elapsed = time.perf_counter() - start
        forward_s += forward_elapsed
        step_elapsed = time.perf_counter() - step_start
        step_seconds.append(step_elapsed)
        collate_seconds.append(collate_elapsed)
        forward_seconds.append(forward_elapsed)
        if progress:
            loss_part = f"loss={loss_values[-1]:.5f} " if measure_loss else ""
            print(
                f"progress variant={variant.name} step={step_idx}/{len(feature_iter)} "
                f"s_it={step_elapsed:.4f} docs={docs_per_timed_step[-1]} "
                f"{loss_part}"
                f"input_shape={tuple(batch['input_ids'].shape)} "
                f"visual_rows={batch.get('pixel_values').shape[0] if 'pixel_values' in batch else 0}",
                flush=True,
            )

    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if torch.cuda.is_available() and device.type == "cuda"
        else 0.0
    )
    timed_docs = sum(docs_per_timed_step)
    step_s = collate_s + forward_s
    mean_step_s = float(np.mean(step_seconds)) if step_seconds else 0.0
    median_step_s = float(np.median(step_seconds)) if step_seconds else 0.0
    p95_step_s = float(np.percentile(step_seconds, 95)) if step_seconds else 0.0
    loss_mean = (
        weighted_loss_sum / max(1, total_label_tokens) if measure_loss else None
    )
    loss_ppl = (
        math.exp(min(loss_mean, 50.0))
        if loss_mean is not None and math.isfinite(loss_mean)
        else None
    )
    loss_step_mean = float(np.mean(loss_values)) if loss_values else None
    loss_step_median = float(np.median(loss_values)) if loss_values else None
    loss_step_p95 = float(np.percentile(loss_values, 95)) if loss_values else None
    return {
        "variant": variant.name,
        "source_rows": len(rows),
        "model_steps": len(feature_iter),
        "warmup_steps": len(warmup_iter),
        "packed_rows": len(features),
        "docs_per_step": len(rows) / max(1, len(features)),
        "timed_docs": timed_docs,
        "build_s": build_s,
        "collate_s": collate_s,
        "forward_s": forward_s,
        "step_s": step_s,
        "mean_s_per_it": mean_step_s,
        "median_s_per_it": median_step_s,
        "p95_s_per_it": p95_step_s,
        "tokens": total_tokens,
        "label_tokens": total_label_tokens,
        "visual_rows": total_visual_rows,
        "tokens_per_s": total_tokens / max(1e-9, step_s),
        "label_tokens_per_s": total_label_tokens / max(1e-9, step_s),
        "docs_per_s": timed_docs / max(1e-9, step_s),
        "steps_per_s": len(feature_iter) / max(1e-9, step_s),
        "tokens_per_forward_s": total_tokens / max(1e-9, forward_s),
        "steps_per_forward_s": len(feature_iter) / max(1e-9, forward_s),
        "unique_shapes": len(shape_counts),
        "top_shapes": shape_counts.most_common(5),
        "peak_mem_mb": peak_mb,
        "loss_enabled": measure_loss,
        "loss_mean": loss_mean,
        "loss_perplexity": loss_ppl,
        "loss_step_mean": loss_step_mean,
        "loss_step_median": loss_step_median,
        "loss_step_p95": loss_step_p95,
        "loss_values": loss_values,
        "step_seconds": step_seconds,
        "collate_seconds": collate_seconds,
        "forward_seconds": forward_seconds,
        "docs_per_timed_step": docs_per_timed_step,
        "label_tokens_per_timed_step": label_tokens_per_timed_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--sequence-len", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--logits-to-keep", type=int, default=1)
    parser.add_argument(
        "--measure-loss",
        action="store_true",
        help="Pass labels through the model and report weighted eval loss.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("chunked", "hf"),
        default="chunked",
        help="Use chunked CE to avoid full fp32 logits OOM, or Transformers' built-in loss.",
    )
    parser.add_argument("--loss-chunk-size", type=int, default=256)
    parser.add_argument("--bin-size", type=int, default=200)
    parser.add_argument("--visual-capacity", type=int, default=4096)
    parser.add_argument("--ram-budget-mb", type=int, default=512)
    parser.add_argument("--split-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int)
    parser.add_argument(
        "--image-resize-algorithm",
        type=parse_resize_algorithm,
        choices=(
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ),
    )
    parser.add_argument(
        "--image-resize-buckets",
        type=parse_resize_bucket,
        nargs="*",
        default=None,
        help="Fixed canvas buckets, e.g. --image-resize-buckets 1024x1536 1536x1536",
    )
    parser.add_argument("--image-resize-no-upscale", action="store_true")
    parser.add_argument("--image-resize-pad-color", type=parse_pad_color)
    parser.add_argument("--tile-grid", type=parse_tile_grid)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument("--tile-min-area", type=int, default=0)
    parser.add_argument("--tile-overview-size", type=int)
    parser.add_argument(
        "--tile-overview-buckets",
        type=parse_resize_bucket,
        nargs="*",
        default=None,
    )
    parser.add_argument("--tile-reading-order", choices=("rtl", "ltr"), default="rtl")
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--include-signature-grouping", action="store_true")
    parser.add_argument(
        "--variants",
        help="Comma-separated variant names to run. Defaults to all configured variants.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "axolotl-mm-pack-bench",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--row-index-file",
        type=Path,
        help="Optional JSON array or newline file of source row indices to benchmark.",
    )
    parser.add_argument(
        "--write-usable-row-indices",
        type=Path,
        help="Write the post-policy rows that fit --sequence-len as a JSON index list.",
    )
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    cfg = load_config(args.config)
    model_path = args.model or Path(cfg["base_model"])
    data_file = args.data_file or Path(cfg["pretraining_dataset"][0]["data_files"])
    training_steps = int(args.training_steps or cfg.get("max_steps") or 0)
    gradient_accumulation_steps = int(
        args.gradient_accumulation_steps
        or cfg.get("gradient_accumulation_steps")
        or 1
    )
    dataset_rows = count_jsonl_rows(data_file)
    processor_kwargs = cfg.get("processor_kwargs") or {}
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=model_path.exists(),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        **processor_kwargs,
    )
    spec = build_image_token_spec(processor)

    metadata_cache = str(args.cache_dir / "metadata")
    if args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    allowed_row_indices = (
        load_row_indices(args.row_index_file) if args.row_index_file else None
    )
    raw_rows = load_rows(
        data_file,
        args.limit,
        args.offset,
        args.stride,
        allowed_row_indices,
    )
    raw_rows = build_tiled_rows(
        raw_rows,
        image_token=spec.image_token,
        tile_grid=args.tile_grid,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_min_area=args.tile_min_area,
        tile_overview_size=args.tile_overview_size,
        tile_overview_buckets=args.tile_overview_buckets,
        tile_reading_order=args.tile_reading_order,
        image_resize_algorithm=args.image_resize_algorithm,
        image_resize_pad_color=args.image_resize_pad_color,
    )
    rows_meta = rows_with_metadata(
        raw_rows,
        processor=processor,
        image_token_id=spec.image_token_id,
        image_base_dir=None,
        cache_path=metadata_cache,
        ram_budget_mb=args.ram_budget_mb,
        image_size=args.image_size,
        image_resize_algorithm=args.image_resize_algorithm,
        image_resize_buckets=args.image_resize_buckets,
        image_resize_no_upscale=args.image_resize_no_upscale,
        image_resize_pad_color=args.image_resize_pad_color,
    )
    usable = [
        row for row, meta in rows_meta if int(meta.length) <= int(args.sequence_len)
    ]
    if not usable:
        raise SystemExit("No selected rows fit --sequence-len.")
    if args.write_usable_row_indices:
        args.write_usable_row_indices.parent.mkdir(parents=True, exist_ok=True)
        args.write_usable_row_indices.write_text(
            json.dumps(
                [int(row["_bench_row_index"]) for row in usable],
                indent=2,
            ),
            encoding="utf-8",
        )

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.no_model:
        model = _NoopModel().to(device)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            local_files_only=model_path.exists(),
            torch_dtype=dtype,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        ).to(device)
        model.eval()

    split_budget = max(
        1, (args.ram_budget_mb + args.split_workers - 1) // args.split_workers
    )
    header = {
        "model": str(model_path),
        "data_file": str(data_file),
        "dataset_rows": dataset_rows,
        "selected_rows": len(raw_rows),
        "usable_rows": len(usable),
        "sequence_len": args.sequence_len,
        "device": str(device),
        "dtype": args.dtype,
        "visual_capacity": args.visual_capacity,
        "ram_budget_mb": args.ram_budget_mb,
        "split_ram_budget_mb": split_budget,
        "training_steps": training_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": args.world_size,
        "image_size": args.image_size,
        "image_resize_algorithm": (
            args.image_resize_algorithm.name.lower()
            if args.image_resize_algorithm is not None
            else None
        ),
        "image_resize_buckets": args.image_resize_buckets,
        "image_resize_no_upscale": args.image_resize_no_upscale,
        "image_resize_pad_color": args.image_resize_pad_color,
        "tile_grid": args.tile_grid,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_min_area": args.tile_min_area,
        "tile_overview_size": args.tile_overview_size,
        "tile_overview_buckets": args.tile_overview_buckets,
        "tile_reading_order": args.tile_reading_order,
        "measure_loss": args.measure_loss,
        "loss_mode": args.loss_mode if args.measure_loss else None,
        "loss_chunk_size": args.loss_chunk_size if args.measure_loss else None,
        "row_index_file": str(args.row_index_file) if args.row_index_file else None,
        "write_usable_row_indices": (
            str(args.write_usable_row_indices)
            if args.write_usable_row_indices
            else None
        ),
    }
    variants = [
        Variant("baseline_unpacked", packed=False),
        Variant("token_packed_no_cache", packed=True),
        Variant(
            "visual_2d_ssd_ram",
            packed=True,
            cache_path=metadata_cache,
            ram_budget_mb=args.ram_budget_mb,
            visual_capacity=args.visual_capacity,
        ),
        Variant(
            "visual_2d_ssd_split_ram",
            packed=True,
            cache_path=metadata_cache,
            ram_budget_mb=split_budget,
            visual_capacity=args.visual_capacity,
        ),
    ]
    if args.include_signature_grouping:
        variants.append(
            Variant(
                "visual_2d_signature_grouped",
                packed=True,
                cache_path=metadata_cache,
                ram_budget_mb=split_budget,
                visual_capacity=args.visual_capacity,
                group_by_visual_signature=True,
            )
        )
    if args.variants:
        requested = {name.strip() for name in args.variants.split(",") if name.strip()}
        variants = [variant for variant in variants if variant.name in requested]
        missing = requested - {variant.name for variant in variants}
        if missing:
            raise SystemExit(f"Unknown variant(s): {', '.join(sorted(missing))}")
    output: dict[str, Any] = {"run": header, "variants": []}
    print(json.dumps(header, indent=2))
    for variant in variants:
        result = run_variant(
            variant,
            usable,
            processor=processor,
            model=model,
            max_tokens=args.sequence_len,
            image_token=spec.image_token,
            image_token_id=spec.image_token_id,
            image_base_dir=None,
            bin_size=args.bin_size,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            logits_to_keep=args.logits_to_keep,
            image_size=args.image_size,
            image_resize_algorithm=args.image_resize_algorithm,
            image_resize_buckets=args.image_resize_buckets,
            image_resize_no_upscale=args.image_resize_no_upscale,
            image_resize_pad_color=args.image_resize_pad_color,
            device=device,
            progress=args.progress,
            measure_loss=args.measure_loss,
            loss_mode=args.loss_mode,
            loss_chunk_size=args.loss_chunk_size,
        )
        if training_steps:
            result["training_steps"] = training_steps
            result["gradient_accumulation_steps"] = gradient_accumulation_steps
            result["world_size"] = args.world_size
            result["optimizer_step_s_est"] = (
                result["mean_s_per_it"] * gradient_accumulation_steps
            )
            result["eta_seconds"] = result["optimizer_step_s_est"] * training_steps
            result["eta"] = fmt_eta(result["eta_seconds"])
            result["docs_per_optimizer_step_est"] = (
                result["docs_per_step"]
                * gradient_accumulation_steps
                * args.world_size
            )
        result["dataset_rows"] = dataset_rows
        result["estimated_single_rank_steps_per_epoch"] = ceil(
            dataset_rows / max(1e-9, result["docs_per_step"])
        )
        output["variants"].append(result)
        print(json.dumps(result, default=str, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(output, default=str, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
