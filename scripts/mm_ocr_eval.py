#!/usr/bin/env python
"""Evaluate multimodal OCR loss and character error rate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image

from axolotl.prompt_strategies.multimodal_pretrain import (
    build_image_token_spec,
    encode_multimodal_pretrain,
)
from axolotl.utils.collators.mm_pretrain import MultiModalPretrainDataCollator
from axolotl.utils.data.mm_image import resize_image_for_processor
from axolotl.utils.data.mm_tiling import image_tiling_config_from_cfg


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_rows(path: Path, *, limit: int, offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx < offset:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def first_data_file(value) -> Path:
    if isinstance(value, (list, tuple)):
        return Path(value[0])
    if isinstance(value, dict):
        if not value:
            raise ValueError("first_data_file: empty dict provided")
        first = next(iter(value.values()))
        return first_data_file(first)
    return Path(value)


def resize_algorithm(value):
    if value is None or isinstance(value, Image.Resampling):
        return value
    normalized = str(value).lower()
    if normalized == "bilinear":
        return Image.Resampling.BILINEAR
    if normalized == "bicubic":
        return Image.Resampling.BICUBIC
    if normalized == "lanczos":
        return Image.Resampling.LANCZOS
    raise ValueError(f"Unknown image resize algorithm: {value}")


def strip_image_placeholders(text: str, image_token: str) -> str:
    text = re.sub(
        r"<\|vision_start\|>\s*.*?<\|vision_end\|>",
        "",
        text,
        flags=re.DOTALL,
    )
    for token in (
        image_token,
        "<image>",
        "<|image|>",
        "<|image_pad|>",
        "<image_soft_token>",
    ):
        text = text.replace(token, "")
    return text.strip()


def image_prompt_from_text(text: str, image_token: str) -> str:
    token = image_token if image_token in text else "<image>"
    token_pos = text.find(token)
    if token_pos < 0:
        return f"{image_token}\n"
    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"
    start = text.rfind(vision_start, 0, token_pos + len(token))
    end = text.find(vision_end, token_pos)
    if start >= 0 and end >= 0:
        return text[start : end + len(vision_end)] + "\n"
    return f"{token}\n"


def generation_prompt(
    row: dict[str, Any], *, text_column: str, image_token: str, template: str | None
) -> str:
    image_placeholder = image_prompt_from_text(
        str(row.get(text_column, "")), image_token
    ).strip()
    if template is not None:
        return template.format(
            image_token=image_token,
            image_placeholder=image_placeholder,
            vision_start="<|vision_start|>",
            vision_end="<|vision_end|>",
        )
    return image_placeholder + "\n"


def target_text(row: dict[str, Any], *, text_column: str, image_token: str) -> str:
    if "messages" in row:
        for message in reversed(row["messages"]):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    return "\n".join(
                        item.get("text", "")
                        for item in content
                        if item.get("type") == "text"
                    ).strip()
    return strip_image_placeholders(str(row[text_column]), image_token)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def cer(prediction: str, reference: str) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    return levenshtein(prediction, reference) / len(reference)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def label_token_count(labels: torch.Tensor) -> int:
    shifted = labels[..., 1:].contiguous()
    return int((shifted != -100).sum().item())


def prepare_generation_inputs(
    *,
    collator: MultiModalPretrainDataCollator,
    prompt: str,
    images: list[Any],
    image_size: int | tuple[int, int] | None,
    image_resize_algorithm,
    image_resize_buckets,
    image_resize_no_upscale: bool,
    image_resize_pad_color,
) -> tuple[str, list[Image.Image]]:
    prepared = collator._prepare_row_text_and_images(prompt, images, row_index=0)
    if prepared is not None:
        return prepared

    loaded = [
        resize_image_for_processor(
            load_image(image),
            image_size,
            image_resize_algorithm,
            image_resize_buckets,
            image_resize_no_upscale,
            image_resize_pad_color,
        )
        for image in images
    ]
    return prompt, loaded


def evaluate(args) -> dict[str, Any]:
    cfg = load_config(args.config)
    dataset_cfg = (cfg.get("pretraining_dataset") or [{}])[0]
    model_path = str(args.model or cfg["base_model"])
    data_file = (
        Path(args.data_file)
        if args.data_file
        else first_data_file(dataset_cfg["data_files"])
    )
    text_column = args.text_column or dataset_cfg.get("text_column") or "text"
    image_column = args.image_column or dataset_cfg.get("image_column") or "images"
    image_base_dir = args.image_base_dir or dataset_cfg.get("image_base_dir")
    processor_kwargs = cfg.get("processor_kwargs") or {}
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=Path(model_path).exists(),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        **processor_kwargs,
    )
    spec = build_image_token_spec(processor)
    rows = load_rows(data_file, limit=args.limit, offset=args.offset)
    tiling_config = image_tiling_config_from_cfg(cfg)
    sequence_len = args.sequence_len or cfg.get("sequence_len")
    if sequence_len is None:
        raise SystemExit("sequence_len must be set via --sequence-len or the config.")
    sequence_len = int(sequence_len)

    encoded = encode_multimodal_pretrain(
        {
            text_column: [row[text_column] for row in rows],
            image_column: [row.get(image_column) or [] for row in rows],
        },
        tokenizer=processor.tokenizer,
        processor=processor,
        max_tokens=sequence_len,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        text_column=text_column,
        image_column=image_column,
        image_base_dir=image_base_dir,
        image_tiling_config=tiling_config,
    )
    features = [
        {key: encoded[key][idx] for key in encoded}
        for idx in range(len(encoded["_mm_text"]))
    ]
    collator = MultiModalPretrainDataCollator(
        tokenizer=processor.tokenizer,
        processor=processor,
        image_token_spec=spec,
        image_base_dir=image_base_dir,
        max_length=sequence_len,
        pad_to_multiple_of=sequence_len,
        image_size=cfg.get("image_size"),
        image_resize_algorithm=resize_algorithm(cfg.get("image_resize_algorithm")),
        image_resize_buckets=cfg.get("image_resize_buckets"),
        image_resize_no_upscale=bool(cfg.get("image_resize_no_upscale")),
        image_resize_pad_color=cfg.get("image_resize_pad_color"),
        image_tiling_config=tiling_config,
    )

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=Path(model_path).exists(),
        torch_dtype=dtype,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    predictions: list[dict[str, Any]] = []
    autocast_enabled = device.type == "cuda" and args.dtype != "fp32"
    with torch.no_grad():
        for row, feature in zip(rows, features, strict=True):
            batch = move_to_device(collator.torch_call([feature]), device)
            tokens = label_token_count(batch["labels"])
            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                loss = model(**batch).loss
            total_loss += float(loss.item()) * max(1, tokens)
            total_tokens += max(1, tokens)

            reference = target_text(
                row,
                text_column=text_column,
                image_token=spec.image_token,
            )
            prompt = generation_prompt(
                row,
                text_column=text_column,
                image_token=spec.image_token,
                template=args.prompt_template,
            )
            prompt, prompt_images = prepare_generation_inputs(
                collator=collator,
                prompt=prompt,
                images=row.get(image_column) or [],
                image_size=cfg.get("image_size"),
                image_resize_algorithm=resize_algorithm(
                    cfg.get("image_resize_algorithm")
                ),
                image_resize_buckets=cfg.get("image_resize_buckets"),
                image_resize_no_upscale=bool(cfg.get("image_resize_no_upscale")),
                image_resize_pad_color=cfg.get("image_resize_pad_color"),
            )
            if prompt_images:
                gen_batch = processor(
                    text=[prompt],
                    images=[prompt_images],
                    return_tensors="pt",
                    padding=True,
                )
            else:
                gen_batch = processor.tokenizer(
                    text=[prompt],
                    return_tensors="pt",
                    padding=True,
                )
            gen_batch = move_to_device(gen_batch, device)
            generated = model.generate(
                **gen_batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            prompt_len = gen_batch["input_ids"].shape[-1]
            prediction = processor.tokenizer.decode(
                generated[0, prompt_len:],
                skip_special_tokens=True,
            ).strip()
            predictions.append(
                {
                    "reference": reference,
                    "prediction": prediction,
                    "cer": cer(prediction, reference),
                }
            )

    mean_loss = total_loss / max(1, total_tokens)
    mean_cer = sum(item["cer"] for item in predictions) / max(1, len(predictions))
    return {
        "rows": len(rows),
        "loss": mean_loss,
        "label_tokens": total_tokens,
        "cer": mean_cer,
        "tiling_enabled": tiling_config is not None,
        "predictions": predictions[: args.save_predictions],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--text-column")
    parser.add_argument("--image-column")
    parser.add_argument("--image-base-dir")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sequence-len", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt-template")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-predictions", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
