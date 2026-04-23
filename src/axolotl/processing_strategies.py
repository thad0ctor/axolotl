"""Module containing ProcessingStrategy classes and its derivative for different MultiModal Model types"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch import Tensor, zeros_like
from transformers import ProcessorMixin
from transformers.image_utils import load_image
from transformers.models.internvl import InternVLProcessor
from transformers.models.smolvlm import SmolVLMProcessor
from transformers.models.voxtral import VoxtralProcessor

from axolotl.utils.dict import remove_none_values
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# One-shot warning dedupe so subclasses that opt out of role masking don't
# spam per-batch on a training run.
_ROLE_MASK_WARNED: set[str] = set()


@dataclass(frozen=True)
class RoleBoundary:
    """Declarative description of one role's token-level boundary.

    A scanner uses these to carve a re-tokenized sequence into
    [ role-span ][ role-span ]... so per-role masking can be applied
    without re-implementing the scan in each subclass.

    start_tokens / end_tokens are lists of tokenizer ids produced by
    ``tokenizer.encode(marker_str, add_special_tokens=False)``.
    An empty ``end_tokens`` means "end-of-sequence terminates the span".
    """

    role: str
    start_tokens: list[int]
    end_tokens: list[int] = field(default_factory=list)
    include_start: bool = False
    include_end: bool = True


class ProcessingStrategy:
    """Base Processing Strategy class.

    Loss-masking knobs (``train_on_inputs`` / ``roles_to_train`` / ``train_on_eos``)
    are honored here when a subclass declares role boundaries via
    ``_build_role_boundaries``. Strategies that don't declare boundaries fall
    back to the legacy behavior (no role masking, only pad + media tokens
    masked) and emit a one-shot warning.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
    ):
        self.processor = processor
        self.chat_template = chat_template
        self.image_token = None
        self.image_token_id = None

        self.image_size = image_size
        self.image_resize_algorithm = (
            image_resize_algorithm or Image.Resampling.BILINEAR
        )

        # Loss-masking config. These mirror the text-only ChatTemplateStrategy
        # defaults (train_on_inputs=False, roles_to_train=["assistant"],
        # train_on_eos="turn" meaning the role-end marker is included).
        self.train_on_inputs = bool(train_on_inputs)
        self.roles_to_train = list(roles_to_train) if roles_to_train else ["assistant"]
        self.train_on_eos = train_on_eos if train_on_eos is not None else "turn"

        if hasattr(processor, "image_token"):
            self.image_token = processor.image_token
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
                self.image_token
            )

        self.role_boundaries: list[RoleBoundary] = self._build_role_boundaries()

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        """Subclasses declare role boundaries here.

        Return an empty list to opt out of role-based masking (legacy behavior).
        The base implementation opts out and emits a one-shot warning from
        ``_mask_non_assistant``.
        """
        return []

    def __call__(self, examples: list[dict]) -> list[dict]:
        """
        Preprocess conversation examples to ensure consistent format.
        Converts different conversation formats to OpenAI format with 'messages'.
        Supports two formats:
        1. OpenAI format with 'messages'
        2. Legacy format with 'conversations'

        Args:
            examples: list of conversation dictionaries

        Returns:
            list of dicts in OpenAI format with 'messages' key

        Raises:
            ValueError: If the conversation format is not supported
        """
        role_mapping = {
            "human": "user",
            "gpt": "assistant",
        }

        def normalize_role(role: str) -> str:
            """Normalize role names to OpenAI format. Default to original role if not found."""
            return role_mapping.get(role, role)

        def convert_legacy_format(example: dict) -> dict:
            """Convert legacy 'conversations' format to OpenAI 'messages' format."""
            messages = [
                {"role": normalize_role(convo["from"]), "content": convo["value"]}
                for convo in example["conversations"]
            ]

            # Create new dict without 'conversations' key
            result = deepcopy(example)
            result.pop("conversations")
            result["messages"] = messages
            return result

        def convert_messages_to_multimedia_messages(messages: list[dict]) -> list[dict]:
            """Convert regular messages format to Messages format with content type"""

            new_messages = []
            for message in messages:
                if isinstance(message["content"], str):
                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": [
                                {
                                    "type": "text",
                                    "text": message["content"],
                                }
                            ],
                        }
                    )
                elif isinstance(message["content"], list):
                    content = message["content"]

                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": content,
                        }
                    )

            return new_messages

        processed_examples = []
        for example in examples:
            if not ("messages" in example or "conversations" in example):
                raise ValueError(
                    "Only `messages` and `conversations` message keys are currently supported."
                )

            processed_example = None
            if (
                "messages" in example and example["messages"] is not None
            ):  # OpenAI format
                processed_example = example
            else:  # Legacy format
                processed_example = convert_legacy_format(example)

            # convert regular messages format to Messages format with content type
            # for compatibility with apply_chat_template
            processed_example["messages"] = convert_messages_to_multimedia_messages(
                processed_example["messages"]
            )

            # find the image key if it exists
            possible_image_keys = ["images", "image"]
            image_key = None
            for key in possible_image_keys:
                if key in processed_example:
                    image_key = key
                    break

            # if the image key exists, add the image to the first user message
            if image_key is not None and processed_example[image_key] is not None:
                # TODO: check if it's normal to be single image only for common datasets
                # From observation, it's usually a list of single image but some datasets may have several columns for images
                # Temporary solution: take the first image and suggest people convert their datasets to use multi-content Messages
                if len(processed_example[image_key]) > 1:
                    LOG.warning(
                        f"Found {len(processed_example[image_key])} images in a sample. Using the first one."
                        "If you are using a dataset with multiple images per sample, please convert it to use multi-content Messages."
                        "See https://docs.axolotl.ai/docs/multimodal.html#dataset-format"
                    )

                image_value = processed_example[image_key][0]

                # Handle image loading (Image, url, path, base64)
                image_value = load_image(image_value)

                if self.image_size is not None:
                    assert hasattr(image_value, "resize"), (
                        "Image does not have a resize method"
                    )

                    if isinstance(self.image_size, tuple):
                        image_value = image_value.resize(
                            self.image_size, self.image_resize_algorithm
                        )
                    else:
                        # Set the padding value; here we use black (0, 0, 0) for RGB images
                        padding_color = (0, 0, 0)

                        # When image_size is an int (square target), preserve aspect ratio then pad
                        # This is to prevent aspect ratio distortion when resizing to square
                        image_value = ImageOps.pad(
                            image_value,
                            (self.image_size, self.image_size),
                            method=self.image_resize_algorithm,
                            color=padding_color,
                        )

                # Look for any image type in the first message
                # some dataset have an {type: "image"} in the first message
                msg_ind_to_add = None
                ind_to_add = None
                first_user_idx = None

                for msg_idx, msg_content in enumerate(processed_example["messages"]):
                    if first_user_idx is None and msg_content["role"] == "user":
                        first_user_idx = msg_idx
                    for i, content in enumerate(
                        processed_example["messages"][msg_idx]["content"]
                    ):
                        # Usually datasets created with image columns, don't have it in the messages itself
                        if content["type"] == "image" and all(
                            k not in content for k in ["image", "url", "path", "base64"]
                        ):
                            msg_ind_to_add = msg_idx
                            ind_to_add = i
                            break

                # If an image type is found, add the image to that index
                if ind_to_add is not None and msg_ind_to_add is not None:
                    processed_example["messages"][msg_ind_to_add]["content"][
                        ind_to_add
                    ]["image"] = image_value
                else:
                    # if no image type is found, add it to end of the first user message
                    if first_user_idx is None:
                        first_user_idx = 0
                    processed_example["messages"][first_user_idx]["content"].append(
                        {
                            "type": "image",
                            "image": image_value,
                        }
                    )

            processed_examples.append(remove_none_values(processed_example))

        return processed_examples

    def _mask_non_assistant(self, labels: Tensor) -> Tensor:
        """Mask non-trainable role regions to -100.

        Uses ``self.role_boundaries`` to locate per-role spans, then zero-masks
        tokens outside trainable roles. Controlled by ``self.train_on_inputs``,
        ``self.roles_to_train``, and ``self.train_on_eos``.
        """
        # train_on_inputs=True means "compute loss on everything" — don't
        # mask based on role. (pad / media tokens are still masked downstream.)
        if self.train_on_inputs:
            return labels

        # Strategies that don't declare boundaries fall back to the legacy
        # no-op. Emit a one-shot warning so the miss is visible in logs.
        if not self.role_boundaries:
            key = type(self).__name__
            if key not in _ROLE_MASK_WARNED:
                _ROLE_MASK_WARNED.add(key)
                LOG.warning(
                    "%s does not declare role boundaries; "
                    "cfg.train_on_inputs / cfg.roles_to_train / cfg.train_on_eos "
                    "will not restrict loss to assistant tokens for this "
                    "multimodal model. Only pad and media tokens are masked. "
                    "See axolotl/processing_strategies.py for how to declare "
                    "boundaries.",
                    key,
                )
            return labels

        return _apply_role_boundaries(
            labels,
            self.role_boundaries,
            roles_to_train=set(self.roles_to_train),
            train_on_eos=self.train_on_eos,
        )

    def process_labels(self, input_ids: Tensor) -> Tensor:
        labels = input_ids.clone()

        labels = self._mask_non_assistant(labels)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Ignore the image token index in the loss computation (model specific)
        labels[labels == self.image_token_id] = -100

        return labels


def _apply_role_boundaries(
    labels: Tensor,
    role_boundaries: list[RoleBoundary],
    roles_to_train: set[str],
    train_on_eos: str,
) -> Tensor:
    """Scan each row of ``labels`` for role-boundary matches and mask
    everything outside trainable role spans with -100.

    - Scan is greedy-left: we walk left→right and for each position check every
      boundary's start_tokens for a prefix match; the longest match wins to
      disambiguate nested markers (e.g. ``<|im_start|>assistant`` vs
      ``<|im_start|>`` alone).
    - ``include_start`` / ``include_end`` govern whether the marker tokens
      themselves contribute to loss on trainable turns.
    - ``train_on_eos`` is honored as:
        - ``"turn"``  → role-end markers included on trainable turns
          (``include_end`` remains True, which is typical).
        - ``"all"``   → role-end markers included on every turn, trainable or
          not (common when the user wants EOS always in loss).
        - ``"none"``  → role-end markers never contribute to loss.
    """
    mask = zeros_like(labels)

    def _match_prefix(label, start_pos, tok_seq):
        if not tok_seq or start_pos + len(tok_seq) > len(label):
            return False
        return label[start_pos : start_pos + len(tok_seq)].tolist() == tok_seq

    def _find_end(label, start_pos, end_tok):
        """Return position AFTER the end marker, plus whether it matched.
        ``end_tok == []`` means run to end-of-sequence.
        """
        if not end_tok:
            return len(label), False
        k = start_pos
        while k < len(label):
            if _match_prefix(label, k, end_tok):
                return k + len(end_tok), True
            k += 1
        return k, False

    for i in range(labels.shape[0]):
        label = labels[i]
        j = 0
        n = len(label)
        while j < n:
            # Longest-prefix start match wins.
            best_match: Optional[RoleBoundary] = None
            for b in role_boundaries:
                if _match_prefix(label, j, b.start_tokens):
                    if (
                        best_match is None
                        or len(b.start_tokens) > len(best_match.start_tokens)
                    ):
                        best_match = b
            if best_match is None:
                j += 1
                continue

            start_of_content = j + len(best_match.start_tokens)
            end_after, found_end = _find_end(
                label, start_of_content, best_match.end_tokens
            )

            role_in_loss = best_match.role in roles_to_train

            if role_in_loss:
                # Include marker tokens as configured.
                if best_match.include_start:
                    mask[i][j:start_of_content] = 1
                # Content between start and end markers.
                content_end = (
                    end_after - len(best_match.end_tokens) if found_end else end_after
                )
                mask[i][start_of_content:content_end] = 1
                # End marker: train_on_eos="none" overrides include_end.
                if (
                    found_end
                    and best_match.include_end
                    and train_on_eos != "none"
                ):
                    mask[i][content_end:end_after] = 1
            else:
                # Non-trainable role. Nothing from this span contributes
                # unless train_on_eos="all" and this span has an end marker.
                if found_end and train_on_eos == "all":
                    content_end = end_after - len(best_match.end_tokens)
                    mask[i][content_end:end_after] = 1

            j = end_after

        labels[i][mask[i] == 0] = -100

    return labels


def _encode_markers(tokenizer, marker_strs: list[str]) -> list[list[int]]:
    """Encode each marker string through the tokenizer, returning their token
    id lists. Strings that tokenize to an empty list are dropped."""
    result = []
    for s in marker_strs:
        toks = tokenizer.encode(s, add_special_tokens=False)
        if toks:
            result.append(toks)
    return result


class Qwen2VLProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Qwen2-VL.

    Chat template uses the ChatML-style ``<|im_start|>{role}\\n ... <|im_end|>``
    structure (see src/axolotl/utils/chat_templates/templates/qwen2_vl.jinja).
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
        )
        self.image_token = "<|image_pad|>"  # nosec
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
            self.image_token
        )

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, ["<|im_end|>"])
        if not end:
            return []
        end_ids = end[0]
        boundaries = []
        for role in ("system", "user", "assistant"):
            start = _encode_markers(tok, [f"<|im_start|>{role}\n"])
            if start:
                boundaries.append(
                    RoleBoundary(role=role, start_tokens=start[0], end_tokens=end_ids)
                )
        return boundaries


class Qwen3_5ProcessingStrategy(Qwen2VLProcessingStrategy):
    """Processing Strategy class for Qwen3.5 (early-fusion VLM).

    Same ChatML boundaries as Qwen2-VL, plus a ``<|video_pad|>`` media token
    mask.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
        )
        self.video_token = "<|video_pad|>"  # nosec
        self.video_token_id = processor.tokenizer.convert_tokens_to_ids(
            self.video_token
        )

    def process_labels(self, input_ids):
        labels = super().process_labels(input_ids)
        labels[labels == self.video_token_id] = -100
        return labels


class _GemmaTurnStrategy(ProcessingStrategy):
    """Common Gemma-family ``<start_of_turn>{role} ... <end_of_turn>``.

    Used by Gemma3 and Gemma3n. Gemma 4 overrides with different markers
    (``<|turn>model`` / ``<turn|>``).
    """

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, ["<end_of_turn>"])
        if not end:
            return []
        end_ids = end[0]
        boundaries = []
        # Gemma renames 'assistant' → 'model' inside the jinja but the external
        # role knob should remain 'assistant'; we map here.
        role_marker_pairs = [
            ("assistant", "model"),
            ("user", "user"),
            ("system", "system"),
        ]
        for external_role, template_role in role_marker_pairs:
            start = _encode_markers(tok, [f"<start_of_turn>{template_role}\n"])
            if start:
                boundaries.append(
                    RoleBoundary(
                        role=external_role,
                        start_tokens=start[0],
                        end_tokens=end_ids,
                    )
                )
        return boundaries


class Gemma3ProcessingStrategy(_GemmaTurnStrategy):
    """Processing Strategy class for Gemma3."""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
        )
        # Gemma3 uses boi_token (<start_of_image>) as the image placeholder.
        special_tokens_map = getattr(
            processor.tokenizer, "special_tokens_map", {}
        ) or {}
        boi = special_tokens_map.get("boi_token")
        if boi is not None:
            self.image_token = boi
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(boi)

    def process_labels(self, input_ids):
        labels = super().process_labels(input_ids)
        # The <image_soft_token> id (262144) is Gemma3-specific; keep the
        # explicit constant since it's not exposed via a tokenizer attribute.
        labels[labels == 262144] = -100
        return labels


class Gemma3nProcessingStrategy(_GemmaTurnStrategy):
    """Processing Strategy class for Gemma3n.

    Shares Gemma3's ``<start_of_turn>{role}`` / ``<end_of_turn>`` boundaries
    but additionally masks audio and image/audio delimiter tokens.
    """

    def process_labels(self, input_ids):
        labels = super().process_labels(input_ids)
        tok = self.processor.tokenizer
        # Follows huggingface-gemma-recipes fine_tune_gemma3n_on_t4 notebook.
        if hasattr(tok, "image_token_id"):
            labels[labels == tok.image_token_id] = -100
        if hasattr(tok, "audio_token_id"):
            labels[labels == tok.audio_token_id] = -100
        if hasattr(tok, "boi_token_id"):
            labels[labels == tok.boi_token_id] = -100
        if hasattr(tok, "eoi_token_id"):
            labels[labels == tok.eoi_token_id] = -100
        return labels


class Gemma4ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Gemma 4.

    Boundary markers: ``<|turn>model`` ... ``<turn|>`` (verified against
    google/gemma-4-E2B-it; see commit 760e9589 on feat/gemma4-assistant-mask).
    Image / audio / video / boi / eoi / boa / eoa tokens are masked. The
    boi/eoi/boa/eoa tokens are exposed on the processor as strings only — not
    as ``*_id`` integer attributes on the tokenizer — so we resolve their ids
    via ``convert_tokens_to_ids``.
    """

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, ["<turn|>"])
        if not end:
            return []
        end_ids = end[0]
        boundaries = []
        role_marker_pairs = [
            ("assistant", "model"),
            ("user", "user"),
            ("system", "system"),
        ]
        for external_role, template_role in role_marker_pairs:
            start = _encode_markers(tok, [f"<|turn>{template_role}"])
            if start:
                boundaries.append(
                    RoleBoundary(
                        role=external_role,
                        start_tokens=start[0],
                        end_tokens=end_ids,
                    )
                )
        return boundaries

    def process_labels(self, input_ids):
        labels = super().process_labels(input_ids)

        tokenizer = self.processor.tokenizer
        unk_id = getattr(tokenizer, "unk_token_id", None)

        if getattr(tokenizer, "image_token_id", None) is not None:
            labels[labels == tokenizer.image_token_id] = -100
        if getattr(tokenizer, "audio_token_id", None) is not None:
            labels[labels == tokenizer.audio_token_id] = -100

        # boi / eoi / boa / eoa are exposed on the processor as token strings,
        # not *_id integer attributes on the tokenizer.
        for attr in ("boi_token", "eoi_token", "boa_token", "eoa_token"):
            token_str = getattr(self.processor, attr, None)
            if token_str is None:
                continue
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id is None or token_id == unk_id:
                continue
            labels[labels == token_id] = -100

        # Gemma 4 stores video id on the processor, not the tokenizer.
        video_token_id = getattr(self.processor, "video_token_id", None)
        if video_token_id is not None and video_token_id != unk_id:
            labels[labels == video_token_id] = -100

        return labels


class Llama3_2VisionProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Llama-3.2 Vision.

    Boundary markers come from the llama3 header scheme (see
    src/axolotl/utils/chat_templates/templates/llama3_2_vision.jinja):
    ``<|start_header_id|>{role}<|end_header_id|>\\n\\n`` ... ``<|eot_id|>``.
    """

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, ["<|eot_id|>"])
        if not end:
            return []
        end_ids = end[0]
        boundaries = []
        for role in ("system", "user", "assistant", "ipython", "tool"):
            start = _encode_markers(
                tok, [f"<|start_header_id|>{role}<|end_header_id|>\n\n"]
            )
            if start:
                boundaries.append(
                    RoleBoundary(role=role, start_tokens=start[0], end_tokens=end_ids)
                )
        return boundaries


class Llama4ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Llama 4.

    Boundary markers come from the Llama 4 chat template
    (src/axolotl/utils/chat_templates/templates/llama4.jinja):
    ``<|header_start|>{role}<|header_end|>\\n\\n`` ... ``<|eot|>``.
    """

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, ["<|eot|>"])
        if not end:
            return []
        end_ids = end[0]
        boundaries = []
        for role in ("system", "user", "assistant", "ipython", "tool"):
            start = _encode_markers(
                tok, [f"<|header_start|>{role}<|header_end|>\n\n"]
            )
            if start:
                boundaries.append(
                    RoleBoundary(role=role, start_tokens=start[0], end_tokens=end_ids)
                )
        return boundaries


class PixtralProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Pixtral.

    Pixtral's chat template wraps user messages in ``[INST] ... [/INST]`` and
    ends assistant messages with ``eos_token`` (see
    src/axolotl/utils/chat_templates/templates/pixtral.jinja).
    """

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        eos = getattr(tok, "eos_token_id", None)
        if eos is None:
            return []
        boundaries = []
        inst_start = _encode_markers(tok, ["[INST]"])
        inst_end = _encode_markers(tok, ["[/INST]"])
        if inst_start and inst_end:
            boundaries.append(
                RoleBoundary(
                    role="user", start_tokens=inst_start[0], end_tokens=inst_end[0]
                )
            )
            boundaries.append(
                RoleBoundary(
                    role="assistant",
                    start_tokens=inst_end[0],
                    end_tokens=[eos],
                )
            )
        return boundaries


class MistralV7TekkenProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Mistral v7 Tekken.

    Similar boundary structure to Pixtral: ``[INST] ... [/INST]`` for user
    turns, assistant content ends at ``eos_token``. ``[SYSTEM_PROMPT] ...
    [/SYSTEM_PROMPT]`` wraps system turns.
    """

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        eos = getattr(tok, "eos_token_id", None)
        if eos is None:
            return []
        boundaries = []
        sys_start = _encode_markers(tok, ["[SYSTEM_PROMPT]"])
        sys_end = _encode_markers(tok, ["[/SYSTEM_PROMPT]"])
        if sys_start and sys_end:
            boundaries.append(
                RoleBoundary(
                    role="system", start_tokens=sys_start[0], end_tokens=sys_end[0]
                )
            )
        inst_start = _encode_markers(tok, ["[INST]"])
        inst_end = _encode_markers(tok, ["[/INST]"])
        if inst_start and inst_end:
            boundaries.append(
                RoleBoundary(
                    role="user", start_tokens=inst_start[0], end_tokens=inst_end[0]
                )
            )
            boundaries.append(
                RoleBoundary(
                    role="assistant",
                    start_tokens=inst_end[0],
                    end_tokens=[eos],
                )
            )
        return boundaries


class VoxtralProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Voxtral.

    NOTE: Role boundaries are NOT declared for Voxtral because its tokenizer
    (mistral-common instruct tokenizer) doesn't expose bracket markers in the
    same way and we haven't verified the right boundary tokens against a real
    checkpoint. Falls back to the legacy behavior (mask pad + audio tokens
    only) with a one-shot warning from ``_mask_non_assistant`` if the user
    has ``train_on_inputs: false`` set.
    """

    def __init__(
        self,
        processor: VoxtralProcessor,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
        )
        special_ids = (
            processor.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.special_ids
        )

        self.audio_token = special_ids.audio
        self.begin_audio_token = special_ids.begin_audio

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.audio_token] = -100
        labels[labels == self.begin_audio_token] = -100

        return labels


class SmolVLM2ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for SmolVLM2.

    NOTE: Role boundaries are NOT declared here. SmolVLM2 uses the tokenizer's
    default chat_template, which varies across checkpoints (HuggingFaceTB ships
    both ChatML-style and custom variants). Rather than guess and mis-mask, we
    opt out of role masking and emit a one-shot warning. To enable role
    masking, subclass this strategy and declare ``_build_role_boundaries`` for
    your specific checkpoint.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
        )
        self.image_token = "<image>"  # nosec

        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index(self.image_token)
        ]


class Mistral3ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Mistral3.

    NOTE: Role boundaries are NOT declared. Mistral3 uses mistral-common's
    instruct tokenizer; verifying the right boundary token ids requires a real
    checkpoint. Same caveat and fallback as VoxtralProcessingStrategy.
    """

    def __init__(
        self,
        processor,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
        )
        special_ids = (
            processor.tokenizer.tokenizer.instruct_tokenizer.image_encoder.special_ids
        )

        self.image_token = special_ids.img
        self.image_break_token = special_ids.img_break
        self.image_end_token = special_ids.img_end

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token] = -100
        labels[labels == self.image_break_token] = -100
        labels[labels == self.image_end_token] = -100

        return labels


class InternVLProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for InternVL.

    NOTE: Role boundaries are NOT declared. InternVL uses an InternLM-style
    chat template that we haven't verified against a real checkpoint. Falls
    back to pad + image-id masking with a one-shot warning.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
        )

        if not hasattr(processor, "image_ids"):
            raise ValueError("'image_ids' missing from InternVL Processor.")

        self.image_token_ids = processor.image_ids

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        for ids in self.image_token_ids:
            labels[labels == ids] = -100

        # Note: Check if need to mask 'video_token' as it gets converted to
        # image patches during media processing

        return labels


class Glm4vProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for GLM4V and GLM4V-MoE vision models.

    NOTE: Role boundaries are NOT declared. GLM4V's chat template uses
    ``<|assistant|>`` / ``<|user|>`` markers but the exact token layout needs
    to be confirmed against a real checkpoint. Falls back to media-token
    masking with a one-shot warning.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
        )

        self.tokenizer = getattr(processor, "tokenizer", processor)

        self.image_token = "<|image|>"  # nosec
        self.begin_image_token = "<|begin_of_image|>"  # nosec
        self.end_image_token = "<|end_of_image|>"  # nosec
        self.video_token = "<|video|>"  # nosec
        self.begin_video_token = "<|begin_of_video|>"  # nosec
        self.end_video_token = "<|end_of_video|>"  # nosec

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.begin_image_token_id = self.tokenizer.convert_tokens_to_ids(
            self.begin_image_token
        )
        self.end_image_token_id = self.tokenizer.convert_tokens_to_ids(
            self.end_image_token
        )
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_token)
        self.begin_video_token_id = self.tokenizer.convert_tokens_to_ids(
            self.begin_video_token
        )
        self.end_video_token_id = self.tokenizer.convert_tokens_to_ids(
            self.end_video_token
        )

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)

        labels[labels == self.tokenizer.pad_token_id] = -100

        labels[labels == self.image_token_id] = -100
        labels[labels == self.begin_image_token_id] = -100
        labels[labels == self.end_image_token_id] = -100

        labels[labels == self.video_token_id] = -100
        labels[labels == self.begin_video_token_id] = -100
        labels[labels == self.end_video_token_id] = -100

        return labels


def get_processing_strategy(
    processor: ProcessorMixin,
    chat_template,
    chat_template_type,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Resampling | None = None,
    train_on_inputs: bool = False,
    roles_to_train: Optional[list[str]] = None,
    train_on_eos: Optional[str] = None,
):
    # Lazy import: mistral_common is optional. Users who don't install it
    # must still be able to dispatch non-mistral strategies.
    try:
        from axolotl.utils.mistral.mistral3_processor import Mistral3Processor
    except ImportError:
        Mistral3Processor = None  # type: ignore[assignment]

    processing_kwargs = {
        "processor": processor,
        "chat_template": chat_template,
        "image_size": image_size,
        "image_resize_algorithm": image_resize_algorithm,
        "train_on_inputs": train_on_inputs,
        "roles_to_train": roles_to_train,
        "train_on_eos": train_on_eos,
    }

    if chat_template_type in [None, "tokenizer_default"]:
        tokenizer = getattr(processor, "tokenizer", processor)
        if hasattr(tokenizer, "chat_template"):
            processing_kwargs["chat_template"] = tokenizer.chat_template

    if chat_template_type == "qwen2_vl":
        return Qwen2VLProcessingStrategy(**processing_kwargs)
    if chat_template_type in ["qwen3_5", "qwen3_5_moe"]:
        return Qwen3_5ProcessingStrategy(**processing_kwargs)
    if chat_template_type == "gemma3":
        return Gemma3ProcessingStrategy(**processing_kwargs)
    if chat_template_type == "gemma3n":
        return Gemma3nProcessingStrategy(**processing_kwargs)
    if chat_template_type == "gemma4":
        return Gemma4ProcessingStrategy(**processing_kwargs)
    if chat_template_type == "llama3_2_vision":
        return Llama3_2VisionProcessingStrategy(**processing_kwargs)
    if chat_template_type == "llama4":
        return Llama4ProcessingStrategy(**processing_kwargs)
    if chat_template_type == "pixtral":
        return PixtralProcessingStrategy(**processing_kwargs)
    if chat_template_type == "mistral_v7_tekken":
        return MistralV7TekkenProcessingStrategy(**processing_kwargs)

    if isinstance(processor, VoxtralProcessor):
        return VoxtralProcessingStrategy(**processing_kwargs)

    if isinstance(processor, SmolVLMProcessor):
        return SmolVLM2ProcessingStrategy(**processing_kwargs)

    if Mistral3Processor is not None and isinstance(processor, Mistral3Processor):
        return Mistral3ProcessingStrategy(**processing_kwargs)
    try:
        from transformers.models.glm46v.processing_glm46v import Glm46VProcessor

        if isinstance(processor, Glm46VProcessor):
            return Glm4vProcessingStrategy(**processing_kwargs)
    except ImportError:
        pass

    if isinstance(processor, InternVLProcessor):
        return InternVLProcessingStrategy(**processing_kwargs)

    # llava, lfm2vl, mistral_v3_tekken, and any other unregistered template
    # fall back to the base strategy, which has no role boundaries and thus
    # emits a one-shot warning when train_on_inputs=False.
    return ProcessingStrategy(**processing_kwargs)
