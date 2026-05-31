"""Data handling specific to streaming datasets."""

import functools
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase, ProcessorMixin

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.data.mm_tiling import image_tiling_config_from_cfg
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.samplers.multipack import pack_parallel
from axolotl.utils.trainer import process_pretraining_datasets_for_packing

LOG = get_logger(__name__)


def _multimodal_metadata_ram_budget_mb(cfg) -> int | None:
    budget = cfg.multimodal_sample_packing_ram_budget_mb
    if not budget or not cfg.multimodal_sample_packing_split_ram_budget_by_worker:
        return budget

    workers = (
        cfg.multimodal_sample_packing_dataloader_num_workers
        if cfg.multimodal_sample_packing_dataloader
        and cfg.multimodal_sample_packing_dataloader_num_workers is not None
        else cfg.dataloader_num_workers
    )
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE") or 1)
    divisor = max(1, int(workers or 0)) * max(1, local_world_size)
    return max(1, (int(budget) + divisor - 1) // divisor)


def encode_streaming(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    text_column: str = "text",
    concatenate: bool = True,
) -> Dict[str, List]:
    res = tokenizer(
        examples[text_column],
        truncation=True,
        max_length=max_tokens - 2,
        add_special_tokens=True,
    )
    # Convert to PyTorch tensors
    input_ids = [torch.tensor(seq) for seq in res["input_ids"]]
    targets = [torch.tensor(seq) for seq in res["input_ids"]]
    attention_mask = [torch.tensor(seq) for seq in res["attention_mask"]]
    if not concatenate:
        return {
            "input_ids": [seq.tolist() for seq in input_ids],
            "labels": [seq.tolist() for seq in targets],
            "attention_mask": [seq.tolist() for seq in attention_mask],
        }

    new_input_ids = []
    new_labels = []
    new_attention_mask = []
    # Append EOS and PAD tokens to input_ids, and correct attention_mask
    for i, _ in enumerate(input_ids):
        input_ids[i] = torch.cat(
            (
                input_ids[i],
                torch.tensor([tokenizer.eos_token_id, tokenizer.pad_token_id]),
            ),
            dim=0,
        )
        targets[i] = torch.cat(
            (
                targets[i],
                torch.tensor([tokenizer.eos_token_id, -100]),
            ),
            dim=0,
        )
        attention_mask[i] = torch.cat((attention_mask[i], torch.tensor([1, 0])), dim=0)

    # Concatenate tokens so that their lengths are less than max_tokens
    buffer_input_ids = torch.tensor([], dtype=torch.long)
    buffer_labels = torch.tensor([], dtype=torch.long)
    buffer_attention_mask = torch.tensor([], dtype=torch.long)

    for ids, labels, mask in zip(input_ids, targets, attention_mask, strict=False):
        if buffer_input_ids.numel() == max_tokens:
            new_input_ids.append(buffer_input_ids)
            new_labels.append(buffer_labels)
            new_attention_mask.append(buffer_attention_mask)
            buffer_input_ids = torch.tensor([], dtype=torch.long)
            buffer_labels = torch.tensor([], dtype=torch.long)
            buffer_attention_mask = torch.tensor([], dtype=torch.long)
            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_labels = torch.cat((buffer_labels, labels), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)
        elif buffer_input_ids.numel() + ids.numel() <= max_tokens:
            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_labels = torch.cat((buffer_labels, labels), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)
        else:
            buffer_input_ids = torch.cat(
                (
                    buffer_input_ids,
                    torch.full(
                        (max_tokens - buffer_input_ids.numel(),),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_labels = torch.cat(
                (
                    buffer_labels,
                    torch.full(
                        (max_tokens - buffer_labels.numel(),),
                        -100,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_attention_mask = torch.cat(
                (
                    buffer_attention_mask,
                    torch.full(
                        (max_tokens - buffer_attention_mask.numel(),),
                        0,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            new_input_ids.append(buffer_input_ids)
            new_labels.append(buffer_labels)
            new_attention_mask.append(buffer_attention_mask)
            buffer_input_ids = torch.tensor([], dtype=torch.long)
            buffer_labels = torch.tensor([], dtype=torch.long)
            buffer_attention_mask = torch.tensor([], dtype=torch.long)

            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_labels = torch.cat((buffer_labels, labels), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)

    if buffer_input_ids.numel() > 0:  # for any leftover tokens
        while buffer_input_ids.numel() < max_tokens:  # make all sequences equal in size
            buffer_input_ids = torch.cat(
                (
                    buffer_input_ids,
                    torch.full(
                        (max_tokens - buffer_input_ids.numel(),),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_labels = torch.cat(
                (
                    buffer_labels,
                    torch.full(
                        (max_tokens - buffer_labels.numel(),),
                        -100,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_attention_mask = torch.cat(
                (
                    buffer_attention_mask,
                    torch.full(
                        (max_tokens - buffer_attention_mask.numel(),),
                        0,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
        new_input_ids.append(buffer_input_ids)
        new_labels.append(buffer_labels)
        new_attention_mask.append(buffer_attention_mask)

    ret = {
        "input_ids": [seq.tolist() for seq in new_input_ids],
        "labels": [seq.tolist() for seq in new_labels],
        "attention_mask": [seq.tolist() for seq in new_attention_mask],
    }

    LOG.debug(len(ret["input_ids"]))
    return ret


def encode_streaming_multimodal(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    image_token: str,
    image_token_id: int,
    text_column: str = "text",
    image_column: str = "images",
    processor: ProcessorMixin | None = None,
    image_base_dir: str | None = None,
    image_tiling_config=None,
) -> Dict[str, List]:
    from axolotl.prompt_strategies.multimodal_pretrain import (
        encode_multimodal_pretrain,
    )

    return encode_multimodal_pretrain(
        examples,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        image_token=image_token,
        image_token_id=image_token_id,
        text_column=text_column,
        image_column=image_column,
        processor=processor,
        image_base_dir=image_base_dir,
        image_tiling_config=image_tiling_config,
        enforce_max_length=True,
    )


def wrap_streaming_dataset(
    dataset,
    tokenizer,
    cfg,
    ds_wrapper_fn,
    processor: Optional[ProcessorMixin] = None,
    pretraining_config=None,
    is_eval: bool = False,
    cache_prep: bool = False,
):
    # Eval streams honor cfg.eval_sequence_len when set, else cfg.sequence_len.
    effective_seq_len = (
        cfg.eval_sequence_len
        if is_eval and getattr(cfg, "eval_sequence_len", None)
        else cfg.sequence_len
    )
    if pretraining_config is not None:
        ds_first = pretraining_config
    elif cfg.pretraining_dataset:
        ds_first = cfg.pretraining_dataset[0]
    else:
        ds_first = {}
    get_ds_value = (
        ds_first.get
        if isinstance(ds_first, dict)
        else lambda key, default=None: getattr(ds_first, key, default)
    )
    text_column = get_ds_value("text_column", "text") or "text"
    ds_type = (get_ds_value("type", None) or "").strip()
    is_mm_cpt = ds_type == "multimodal_pretrain" or bool(
        get_ds_value("multimodal", False)
    )

    if cfg.sample_packing:
        if cache_prep:
            raise ValueError(
                "wrap_streaming_dataset cache_prep=True is incompatible with sample_packing"
            )
        if is_mm_cpt:
            if processor is None:
                raise ValueError(
                    "Multimodal CPT (type: multimodal_pretrain) requires a "
                    "processor. Set `processor_type: AutoProcessor` (or the "
                    "concrete processor class) in your config."
                )
            from axolotl.prompt_strategies.multimodal_pretrain import (
                build_image_token_spec,
                check_processor_compatibility,
            )

            check_processor_compatibility(processor)
            spec = build_image_token_spec(
                processor,
                override=get_ds_value("image_token", None),
            )
            image_column = get_ds_value("image_column", None) or "images"
            encode = functools.partial(
                encode_packed_streaming_multimodal,
                tokenizer=tokenizer,
                processor=processor,
                max_tokens=effective_seq_len,
                batch_size=cfg.micro_batch_size,
                bin_size=cfg.sample_packing_bin_size,
                image_token=spec.image_token,
                image_token_id=spec.image_token_id,
                text_column=text_column,
                image_column=image_column,
                image_base_dir=get_ds_value("image_base_dir", None),
                metadata_cache_path=cfg.multimodal_sample_packing_cache_path,
                metadata_cache_ram_budget_mb=_multimodal_metadata_ram_budget_mb(cfg),
                visual_capacity=cfg.multimodal_sample_packing_visual_capacity,
                group_by_visual_signature=cfg.multimodal_sample_packing_group_by_visual_signature,
                use_multimodal_sample_packing=cfg.multimodal_sample_packing
                is not False,
                image_size=cfg.get("image_size"),
                image_resize_algorithm=cfg.get("image_resize_algorithm"),
                image_resize_buckets=cfg.get("image_resize_buckets"),
                image_resize_no_upscale=bool(cfg.get("image_resize_no_upscale")),
                image_resize_pad_color=cfg.get("image_resize_pad_color"),
                image_tiling_config=image_tiling_config_from_cfg(cfg),
            )
        else:
            # For SFT (non-pretraining) datasets, always use multipack_attn=True to ensure
            # attention isolation between packed sequences
            multipack_attn = (
                True if not cfg.pretraining_dataset else cfg.pretrain_multipack_attn
            )

            collate_fn = PretrainingBatchSamplerDataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=cfg.sequence_len,
                multipack_attn=multipack_attn,
            )
            encode = functools.partial(
                encode_packed_streaming,
                collate_fn,
                ds_wrapper_fn,
                max_seq_length=cfg.sequence_len,
                batch_size=cfg.micro_batch_size,
                multipack_attn=multipack_attn,
                bin_size=cfg.sample_packing_bin_size,
            )

        # Set this to 1 so downstream data_loader doesn't try to increase the batch size
        # again
        cfg.micro_batch_size = 1
    else:
        # NOTE: This is not reachable for SFT datasets since we use the pre-existing
        # loading function for non-packed streaming datasets. Refer to
        # _prepare_streaming_datasets in sft.py for that code path.
        # Prefer the resolved per-entry config so eval (test_datasets) doesn't
        # silently inherit the training entry's columns/image_token.
        if is_mm_cpt:
            if processor is None:
                raise ValueError(
                    "Multimodal CPT (type: multimodal_pretrain) requires a "
                    "processor. Set `processor_type: AutoProcessor` (or the "
                    "concrete processor class) in your config."
                )
            from axolotl.prompt_strategies.multimodal_pretrain import (
                build_image_token_spec,
                check_processor_compatibility,
            )

            check_processor_compatibility(processor)
            spec = build_image_token_spec(
                processor,
                override=get_ds_value("image_token", None),
            )
            image_column = get_ds_value("image_column", None) or "images"
            LOG.info(
                f"multimodal streaming CPT: placeholder={spec.image_token!r} "
                f"(id={spec.image_token_id})"
            )
            encode = functools.partial(
                encode_streaming_multimodal,
                tokenizer=tokenizer,
                max_tokens=effective_seq_len,
                image_token=spec.image_token,
                image_token_id=spec.image_token_id,
                text_column=text_column,
                image_column=image_column,
                processor=processor,
                image_base_dir=get_ds_value("image_base_dir", None),
                image_tiling_config=image_tiling_config_from_cfg(cfg),
            )
        else:
            encode = functools.partial(
                encode_streaming,
                tokenizer=tokenizer,
                max_tokens=effective_seq_len,
                text_column=text_column,
                concatenate=cfg.pretraining_sample_concatenation is True,
            )

    # cache_prep: DataLoader will reshuffle the resulting map-style dataset each epoch.
    if cfg.shuffle_merged_datasets and not cache_prep:
        dataset = dataset.shuffle(
            seed=cfg.seed, buffer_size=cfg.streaming_multipack_buffer_size
        )
    else:
        LOG.debug("NOT shuffling merged pretraining datasets")

    # remove all the existing columns after mapping since they end up having
    # a different length than the encoded/tokenized column
    # this is empty during streaming/pretraining
    remove_columns = []
    if dataset.features is None:
        for first_row in dataset:
            remove_columns = list(first_row.keys())
            break
    else:
        remove_columns = list(dataset.features.keys())

    dataset = dataset.map(
        encode,
        batched=True,
        batch_size=cfg.streaming_multipack_buffer_size,
        remove_columns=remove_columns,
    )
    return dataset


def encode_packed_streaming(
    collate_fn,
    ds_wrapper: Callable,
    examples: Dict[str, List],
    bin_size: int,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    multipack_attn: Optional[bool] = True,
) -> Dict[str, List]:
    # tokenize all the examples
    # rows get split with stride (overlap)
    train_dataset = ds_wrapper(dataset=Dataset.from_dict(examples))[0]

    train_dataset = process_pretraining_datasets_for_packing(
        train_dataset,
        max_seq_length,
        skip_position_ids=not multipack_attn,
        # FIXME using attention mask unpad/pad with trainer and packed pretraining is broken atm
        # workaround by using the position id logic for now in trainer
        drop_attention_mask=multipack_attn,
    )

    sampler = MultipackBatchSampler(
        sampler=RandomSampler(train_dataset),
        lengths=get_dataset_lengths(train_dataset),
        batch_size=1,
        batch_max_len=batch_size * max_seq_length,
        drop_last=True,
        num_processes=1,
        bin_size=bin_size,
    )

    chunked_data = defaultdict(list)

    for batch in sampler:
        for data in batch:
            features = train_dataset[data]
            if "num_truncated_tokens" in features:
                del features["num_truncated_tokens"]
            if "overflow_to_sample_mapping" in features:
                del features["overflow_to_sample_mapping"]
            if "labels" not in features:
                features["labels"] = features["input_ids"].copy()
            collated_features = collate_fn(features)

            for feature in features.keys():
                if feature == "length":
                    continue
                chunked_data[feature].append(collated_features[feature].squeeze(0))

    return chunked_data


def encode_packed_streaming_multimodal(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    processor: ProcessorMixin,
    max_tokens: int,
    batch_size: int,
    bin_size: int,
    image_token: str,
    image_token_id: int,
    text_column: str = "text",
    image_column: str = "images",
    image_base_dir: str | None = None,
    metadata_cache_path: str | None = None,
    metadata_cache_ram_budget_mb: int | None = None,
    visual_capacity: int | None = None,
    group_by_visual_signature: bool | None = False,
    use_multimodal_sample_packing: bool = True,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Any | None = None,
    image_resize_buckets: list[tuple[int, int]] | None = None,
    image_resize_no_upscale: bool = False,
    image_resize_pad_color: Any | None = None,
    image_tiling_config=None,
) -> Dict[str, List]:
    from axolotl.prompt_strategies.multimodal_pretrain import (
        encode_multimodal_pretrain,
        prepare_text_for_packed_boundary,
    )
    from axolotl.utils.data.mm_packing import (
        MultimodalPackingMetadataCache,
        compute_multimodal_packing_metadata,
        pack_2d_first_fit_decreasing,
    )

    encoded = encode_multimodal_pretrain(
        examples,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        image_token=image_token,
        image_token_id=image_token_id,
        text_column=text_column,
        image_column=image_column,
        processor=processor,
        image_base_dir=image_base_dir,
        add_processor_lengths=False,
        enforce_max_length=True,
        image_size=image_size,
        image_resize_algorithm=image_resize_algorithm,
        image_resize_buckets=image_resize_buckets,
        image_resize_no_upscale=image_resize_no_upscale,
        image_resize_pad_color=image_resize_pad_color,
        image_tiling_config=image_tiling_config,
    )
    rows = [
        {key: encoded[key][idx] for key in encoded}
        for idx in range(len(encoded["_mm_text"]))
    ]
    if not rows:
        return {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "images": [],
            "_mm_text": [],
            "_mm_sample_lengths": [],
            "_mm_visual_tokens": [],
            "_mm_visual_signature": [],
        }

    cache = MultimodalPackingMetadataCache(
        cache_path=metadata_cache_path,
        ram_budget_mb=metadata_cache_ram_budget_mb,
    )
    metadata = compute_multimodal_packing_metadata(
        [row["_mm_text"] for row in rows],
        [row["images"] for row in rows],
        tokenizer=tokenizer,
        processor=processor,
        image_token_id=image_token_id,
        image_token=image_token,
        image_base_dir=image_base_dir,
        add_eos_token=True,
        image_size=image_size,
        image_resize_algorithm=image_resize_algorithm,
        image_resize_buckets=image_resize_buckets,
        image_resize_no_upscale=image_resize_no_upscale,
        image_resize_pad_color=image_resize_pad_color,
        image_tiling_config=image_tiling_config,
        cache=cache,
    )
    for row, meta in zip(rows, metadata, strict=True):
        row["length"] = meta.length
        row["_mm_visual_tokens"] = meta.visual_tokens
        row["_mm_visual_signature"] = meta.visual_signature

    capacity = batch_size * max_tokens
    long_rows = [int(row["length"]) for row in rows if int(row["length"]) > capacity]
    if long_rows:
        raise ValueError(
            "Multimodal packed streaming row exceeds packing capacity "
            f"{capacity}: max row length={max(long_rows)}."
        )

    if use_multimodal_sample_packing and (
        visual_capacity is not None or group_by_visual_signature
    ):
        bins = []
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            signature = (
                str(row["_mm_visual_signature"])
                if group_by_visual_signature
                else "__all__"
            )
            groups[signature].append(idx)
        for group_indices in groups.values():
            group_rows = [rows[idx] for idx in group_indices]
            if visual_capacity is None:
                group_bins = pack_parallel(
                    np.array(
                        [int(row["length"]) for row in group_rows], dtype=np.int32
                    ),
                    bin_capacity=capacity,
                    group_size=len(group_rows),
                    bin_size=bin_size,
                    num_processes=1,
                    safe_mode=False,
                )
            else:
                group_bins = pack_2d_first_fit_decreasing(
                    [int(row["length"]) for row in group_rows],
                    [int(row["_mm_visual_tokens"]) for row in group_rows],
                    token_capacity=capacity,
                    visual_capacity=visual_capacity,
                )
            bins.extend(
                [[group_indices[idx] for idx in group_bin] for group_bin in group_bins]
            )
    else:
        bins = pack_parallel(
            np.array([int(row["length"]) for row in rows], dtype=np.int32),
            bin_capacity=capacity,
            group_size=len(rows),
            bin_size=bin_size,
            num_processes=1,
            safe_mode=False,
        )

    packed = defaultdict(list)
    for bin_indices in bins:
        bin_rows = [rows[idx] for idx in bin_indices]
        packed["_mm_text"].append(
            "".join(
                prepare_text_for_packed_boundary(
                    row,
                    tokenizer,
                    is_first=idx == 0,
                    add_eos_token=True,
                )
                for idx, row in enumerate(bin_rows)
            )
        )
        packed["images"].append(
            [image for row in bin_rows for image in (row["images"] or [])]
        )
        packed["_mm_sample_lengths"].append([int(row["length"]) for row in bin_rows])
        packed["_mm_visual_tokens"].append(
            [int(row["_mm_visual_tokens"]) for row in bin_rows]
        )
        packed["_mm_visual_signature"].append(
            [str(row["_mm_visual_signature"]) for row in bin_rows]
        )
        for key in ("input_ids", "labels", "attention_mask"):
            packed[key].append([tok for row in bin_rows for tok in row[key]])

    return packed
