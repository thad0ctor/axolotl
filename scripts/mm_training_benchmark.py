#!/usr/bin/env python
"""Run a small multimodal CPT training-loop benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from axolotl.prompt_strategies.multimodal_pretrain import build_image_token_spec
from axolotl.utils.collators.mm_pretrain import MultiModalPretrainDataCollator
from scripts.mm_packing_benchmark import (
    batch_shapes,
    build_baseline_rows,
    build_tiled_rows,
    chunked_causal_lm_loss,
    fmt_eta,
    load_config,
    load_row_indices,
    load_rows,
    parse_pad_color,
    parse_resize_algorithm,
    parse_resize_bucket,
    parse_tile_grid,
    rows_with_metadata,
    shifted_label_token_count,
)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_features(
    args,
    *,
    cfg: dict[str, Any],
    processor,
    data_file: Path,
    image_token: str,
    image_token_id: int,
) -> list[dict]:
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
        image_token=image_token,
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
        image_token_id=image_token_id,
        image_base_dir=None,
        cache_path=str(args.cache_dir / f"{args.variant}_metadata"),
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
    features = build_baseline_rows(
        usable,
        processor=processor,
        max_tokens=args.sequence_len,
        image_token=image_token,
        image_token_id=image_token_id,
        image_base_dir=None,
    )
    print(
        json.dumps(
            {
                "variant": args.variant,
                "selected_rows": len(raw_rows),
                "usable_rows": len(usable),
                "features": len(features),
                "sequence_len": args.sequence_len,
                "base_model": str(args.model or cfg["base_model"]),
            },
            indent=2,
        ),
        flush=True,
    )
    return features


def configure_model(args, cfg: dict[str, Any], device: torch.device):
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model or cfg["base_model"]),
        local_files_only=Path(args.model or cfg["base_model"]).exists(),
        torch_dtype=dtype,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=args.rslora,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.lm_head.parameters():
            parameter.requires_grad_(True)

    model.to(device)
    model.train()
    return model


def configure_optimizer(args, model):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                params,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon,
                weight_decay=args.weight_decay,
            )
        except Exception as exc:  # pragma: no cover - depends on local bnb build
            print(f"falling back to torch AdamW: {type(exc).__name__}: {exc}")

    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def run_training(args) -> dict[str, Any]:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    cfg = load_config(args.config)
    model_path = args.model or Path(cfg["base_model"])
    data_file = args.data_file or Path(cfg["pretraining_dataset"][0]["data_files"])
    processor_kwargs = cfg.get("processor_kwargs") or {}
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=Path(model_path).exists(),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        **processor_kwargs,
    )
    spec = build_image_token_spec(processor)
    features = build_features(
        args,
        cfg=cfg,
        processor=processor,
        data_file=data_file,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )

    collator = MultiModalPretrainDataCollator(
        tokenizer=processor.tokenizer,
        processor=processor,
        image_token_spec=spec,
        image_base_dir=None,
        sample_packing=False,
        pad_to_multiple_of=args.sequence_len,
        max_length=args.sequence_len,
        image_size=args.image_size,
        image_resize_algorithm=args.image_resize_algorithm,
        image_resize_buckets=args.image_resize_buckets,
        image_resize_no_upscale=args.image_resize_no_upscale,
        image_resize_pad_color=args.image_resize_pad_color,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = configure_model(args, cfg, device)
    optimizer = configure_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
    autocast_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    losses: list[float] = []
    step_seconds: list[float] = []
    forward_seconds: list[float] = []
    backward_seconds: list[float] = []
    optimizer_seconds: list[float] = []
    label_tokens_per_step: list[int] = []
    visual_rows_per_step: list[int] = []
    shape_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started_at = time.perf_counter()
    for step_idx in range(1, args.steps + 1):
        feature = features[(step_idx - 1) % len(features)]
        step_start = time.perf_counter()
        batch = collator.torch_call([feature])
        shape_counts[batch_shapes(batch)] += 1
        label_tokens = shifted_label_token_count(batch["labels"])
        visual_rows = (
            int(batch["pixel_values"].shape[0]) if "pixel_values" in batch else 0
        )
        inputs = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda" and args.dtype != "fp32",
        ):
            loss = chunked_causal_lm_loss(
                model,
                inputs,
                chunk_size=args.loss_chunk_size,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_elapsed = time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_elapsed = time.perf_counter() - backward_start

        optimizer_start = time.perf_counter()
        if args.max_grad_norm > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_elapsed = time.perf_counter() - optimizer_start

        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite loss at step {step_idx}: {loss_value}")
        step_elapsed = time.perf_counter() - step_start
        losses.append(loss_value)
        step_seconds.append(step_elapsed)
        forward_seconds.append(forward_elapsed)
        backward_seconds.append(backward_elapsed)
        optimizer_seconds.append(optimizer_elapsed)
        label_tokens_per_step.append(label_tokens)
        visual_rows_per_step.append(visual_rows)

        if args.progress and (
            step_idx == 1 or step_idx % args.log_every == 0 or step_idx == args.steps
        ):
            recent = losses[-min(args.log_every, len(losses)) :]
            elapsed = time.perf_counter() - started_at
            remaining = (elapsed / step_idx) * (args.steps - step_idx)
            print(
                "progress "
                f"variant={args.variant} step={step_idx}/{args.steps} "
                f"loss={loss_value:.5f} loss_recent={np.mean(recent):.5f} "
                f"s_it={step_elapsed:.3f} label_tokens={label_tokens} "
                f"visual_rows={visual_rows} eta={fmt_eta(remaining)}",
                flush=True,
            )

    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "variant": args.variant,
        "model": str(model_path),
        "data_file": str(data_file),
        "steps": args.steps,
        "sequence_len": args.sequence_len,
        "device": str(device),
        "dtype": args.dtype,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "lora": args.lora,
        "lora_r": args.lora_r if args.lora else None,
        "lora_alpha": args.lora_alpha if args.lora else None,
        "gradient_checkpointing": args.gradient_checkpointing,
        "loss_chunk_size": args.loss_chunk_size,
        "image_resize_buckets": args.image_resize_buckets,
        "image_resize_no_upscale": args.image_resize_no_upscale,
        "image_resize_pad_color": args.image_resize_pad_color,
        "tile_grid": args.tile_grid,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_overview_buckets": args.tile_overview_buckets,
        "loss_values": losses,
        "step_seconds": step_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "label_tokens_per_step": label_tokens_per_step,
        "visual_rows_per_step": visual_rows_per_step,
        "loss_start_50": float(np.mean(losses[: min(50, len(losses))])),
        "loss_end_50": float(np.mean(losses[-min(50, len(losses)) :])),
        "loss_delta_end_minus_start": float(
            np.mean(losses[-min(50, len(losses)) :])
            - np.mean(losses[: min(50, len(losses))])
        ),
        "loss_summary": summarize(losses),
        "step_seconds_summary": summarize(step_seconds),
        "forward_seconds_summary": summarize(forward_seconds),
        "backward_seconds_summary": summarize(backward_seconds),
        "optimizer_seconds_summary": summarize(optimizer_seconds),
        "label_tokens_total": int(sum(label_tokens_per_step)),
        "visual_rows_total": int(sum(visual_rows_per_step)),
        "label_tokens_per_s": sum(label_tokens_per_step) / max(1e-9, sum(step_seconds)),
        "docs_per_s": len(losses) / max(1e-9, sum(step_seconds)),
        "peak_mem_mb": peak_mb,
        "unique_shapes": len(shape_counts),
        "top_shapes": shape_counts.most_common(5),
    }
    print(json.dumps(result, default=str, indent=2), flush=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, default=str, indent=2),
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--sequence-len", type=int, default=14848)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--row-index-file", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5.56e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--optimizer", choices=("adamw_8bit", "adamw"), default="adamw_8bit"
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-chunk-size", type=int, default=256)
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--rslora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--image-resize-buckets", type=parse_resize_bucket, nargs="*")
    parser.add_argument("--image-resize-no-upscale", action="store_true")
    parser.add_argument("--image-resize-pad-color", type=parse_pad_color)
    parser.add_argument("--tile-grid", type=parse_tile_grid)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument("--tile-min-area", type=int, default=0)
    parser.add_argument("--tile-overview-size", type=int)
    parser.add_argument("--tile-overview-buckets", type=parse_resize_bucket, nargs="*")
    parser.add_argument("--tile-reading-order", choices=("rtl", "ltr"), default="rtl")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "axolotl-mm-train-bench",
    )
    parser.add_argument("--ram-budget-mb", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
