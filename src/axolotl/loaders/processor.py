"""Processor loading functionality for multi-modal models"""

import transformers
from transformers import (
    AutoProcessor,
    PreTrainedTokenizerBase,
)

from axolotl.telemetry.errors import send_errors
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@send_errors
def load_processor(cfg: DictDefault, tokenizer: PreTrainedTokenizerBase):
    processor_cls = AutoProcessor
    if cfg.processor_type:
        processor_cls = getattr(transformers, cfg.processor_type)

    # Build common kwargs for processor loading. User-supplied cfg.processor_kwargs
    # are merged first so that axolotl-managed kwargs (revision, trust_remote_code)
    # cannot be silently overridden. See issue #3617.
    processor_kwargs: dict = {}
    user_processor_kwargs = dict(cfg.processor_kwargs) if cfg.processor_kwargs else {}
    _overridden = {
        k for k in ("revision", "trust_remote_code") if k in user_processor_kwargs
    }
    if _overridden:
        LOG.warning(
            "Ignoring cfg.processor_kwargs keys %s — these are managed by axolotl.",
            sorted(_overridden),
        )
        for k in _overridden:
            user_processor_kwargs.pop(k, None)
    processor_kwargs.update(user_processor_kwargs)

    if cfg.revision_of_model:
        processor_kwargs["revision"] = cfg.revision_of_model

    if cfg.tokenizer_use_mistral_common:

        def _patch_mistralcommontokenizer():
            """
            Transformers v5 stops reading the sub-processor.

            We need to patch this, so both processors use this.
            """
            import transformers.tokenization_mistral_common as tokenization_mistral_common

            from axolotl.utils.mistral import HFMistralTokenizer

            tokenization_mistral_common.MistralCommonBackend = HFMistralTokenizer

        _patch_mistralcommontokenizer()

        from transformers import VoxtralProcessor

        if processor_cls == VoxtralProcessor:
            return VoxtralProcessor.from_pretrained(
                cfg.processor_config,
                **processor_kwargs,
            )

        from axolotl.utils.mistral import Mistral3Processor

        return Mistral3Processor(
            tokenizer=tokenizer,
        )

    processor_kwargs["trust_remote_code"] = cfg.trust_remote_code or False

    processor = processor_cls.from_pretrained(
        cfg.processor_config,
        **processor_kwargs,
    )
    processor.tokenizer = tokenizer

    # Attempt to load image size from processor if available
    if (
        cfg.image_size is None
        and hasattr(processor, "size")
        and any(dim in processor.size for dim in ["width", "height"])
    ):
        im_width = None
        im_height = None
        if "width" in processor.size:
            im_width = processor.size["width"]
        if "height" in processor.size:
            im_height = processor.size["height"]

        # If both width and height are set, use a tuple
        if im_width is not None and im_height is not None:
            cfg.image_size = (im_width, im_height)
        # If only width is set, use as integer
        elif im_width is not None:
            cfg.image_size = im_width
        # If only height is set, use as integer
        elif im_height is not None:
            cfg.image_size = im_height

        LOG.debug(f"Loaded image size: {cfg.image_size} from processor")

    return processor
