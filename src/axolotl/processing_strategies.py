"""Module containing ProcessingStrategy classes and its derivative for different MultiModal Model types"""

import json
from copy import deepcopy
from typing import Optional

from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch import Tensor, zeros_like
from transformers import ProcessorMixin
from transformers.image_utils import load_image
from transformers.models.smolvlm import SmolVLMProcessor
from transformers.models.voxtral import VoxtralProcessor

from axolotl.utils.dict import remove_none_values
from axolotl.utils.logging import get_logger
from axolotl.utils.mistral.mistral3_processor import Mistral3Processor

LOG = get_logger(__name__)


class ProcessingStrategy:
    """Base Processing Strategy class"""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        self.processor = processor
        self.chat_template = chat_template
        self.image_token = None
        self.image_token_id = None
        self.supports_multi_images = False  # Override in subclasses that support multiple images

        self.image_size = image_size
        self.image_resize_algorithm = (
            image_resize_algorithm or Image.Resampling.BILINEAR
        )

        if hasattr(processor, "image_token"):
            self.image_token = processor.image_token
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
                self.image_token
            )

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
        # if not hasattr(self, '_process_step_count'):
        #     self._process_step_count = 0
        # self._process_step_count += 1
        # 
        # if self._process_step_count <= 10:
        #     LOG.debug(f"[PROCESSING STRATEGY] Processing {len(examples)} examples")
        
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

        def convert_messages_to_multimedia_messages(messages: list[dict], is_qwen2_vl: bool = False) -> list[dict]:
            """Convert regular messages format to Messages format with content type"""

            new_messages = []
            for message in messages:
                content = message["content"]
                
                # Only try to deserialize JSON-encoded LIST content for qwen2_vl models
                # This is because we normalized mixed content to JSON strings during loading
                # NOTE: Only deserialize lists (starting with '['), NOT dicts (starting with '{')
                # Assistant messages often contain JSON output like {"drone": true, ...} which
                # should remain as text, not be deserialized to a dict and silently dropped
                if is_qwen2_vl and isinstance(content, str) and content.startswith('['):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        # Not JSON, treat as regular string
                        pass
                
                if isinstance(content, str):
                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": [
                                {
                                    "type": "text",
                                    "text": content,
                                }
                            ],
                        }
                    )
                elif isinstance(content, list):
                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": content,
                        }
                    )
                elif isinstance(content, dict):
                    # Convert dict back to text (JSON output from assistant)
                    # This handles edge cases where content somehow becomes a dict
                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(content),
                                }
                            ],
                        }
                    )

            return new_messages

        processed_examples = []
        for example in examples:
            if not ("messages" in example or "conversations" in example):
                # # Debug: Log what keys are actually present
                # LOG.error(f"[PROCESSING DEBUG] Example keys: {list(example.keys())}")
                # LOG.error(f"[PROCESSING DEBUG] Example type: {type(example)}")
                # if len(list(example.keys())) < 10:  # Only log if not too many keys
                #     LOG.error(f"[PROCESSING DEBUG] Full example: {example}")
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
            # Check if this model supports multi-images and needs special handling
            processed_example["messages"] = convert_messages_to_multimedia_messages(
                processed_example["messages"], is_qwen2_vl=self.supports_multi_images
            )

            # find the image key if it exists
            possible_image_keys = ["images", "image"]
            image_key = None
            for key in possible_image_keys:
                if key in processed_example:
                    image_key = key
                    break

            # if the image key exists, add the images to the message
            if image_key is not None and processed_example[image_key] is not None:
                # Check if we should handle multiple images
                # Debug logging
                if len(processed_example[image_key]) > 1:
                    LOG.debug(f"Multiple images detected. Strategy type: {type(self).__name__}, supports_multi_images={self.supports_multi_images}")
                
                if self.supports_multi_images and len(processed_example[image_key]) > 1:
                    # Qwen2-VL: Load all images
                    loaded_images = []
                    for img in processed_example[image_key]:
                        loaded_img = load_image(img)
                        loaded_images.append(loaded_img)
                    
                    # Log multi-image usage for debugging
                    LOG.debug(f"Processing {len(loaded_images)} images in sample for Qwen2-VL")
                else:
                    # Original behavior: take first image and warn if multiple
                    if len(processed_example[image_key]) > 1:
                        LOG.warning(
                            f"Found {len(processed_example[image_key])} images in a sample. Using the first one."
                            "If you are using a dataset with multiple images per sample, please convert it to use multi-content Messages."
                            "See https://docs.axolotl.ai/docs/multimodal.html#dataset-format"
                        )
                    
                    image_value = processed_example[image_key][0]
                    # Handle image loading (Image, url, path, base64)
                    image_value = load_image(image_value)
                    loaded_images = [image_value]

                # Resize all loaded images if needed
                if self.image_size is not None:
                    resized_images = []
                    for image_value in loaded_images:
                        assert hasattr(image_value, "resize"), (
                            "Image does not have a resize method"
                        )

                        if isinstance(self.image_size, tuple):
                            resized_img = image_value.resize(
                                self.image_size, self.image_resize_algorithm
                            )
                        else:
                            # Set the padding value; here we use black (0, 0, 0) for RGB images
                            padding_color = (0, 0, 0)

                            # When image_size is an int (square target), preserve aspect ratio then pad
                            # This is to prevent aspect ratio distortion when resizing to square
                            resized_img = ImageOps.pad(
                                image_value,
                                (self.image_size, self.image_size),
                                method=self.image_resize_algorithm,
                                color=padding_color,
                            )
                        resized_images.append(resized_img)
                    loaded_images = resized_images

                # Look for image placeholders in messages
                if self.supports_multi_images and len(loaded_images) > 1:
                    # Qwen2-VL: Map multiple images to their placeholders
                    image_placeholders = []
                    first_user_idx = None

                    for msg_idx, msg_content in enumerate(processed_example["messages"]):
                        if first_user_idx is None and msg_content["role"] == "user":
                            first_user_idx = msg_idx
                        for i, content in enumerate(
                            processed_example["messages"][msg_idx]["content"]
                        ):
                            # Find image placeholders
                            if content["type"] == "image" and all(
                                k not in content for k in ["image", "url", "path", "base64"]
                            ):
                                image_placeholders.append((msg_idx, i))

                    # Map loaded images to placeholders
                    if image_placeholders:
                        # If we have placeholders, map images to them in order
                        for idx, (msg_idx, content_idx) in enumerate(image_placeholders):
                            if idx < len(loaded_images):
                                processed_example["messages"][msg_idx]["content"][content_idx]["image"] = loaded_images[idx]
                    else:
                        # If no placeholders found, add all images to end of first user message
                        if first_user_idx is None:
                            first_user_idx = 0
                        for image_value in loaded_images:
                            processed_example["messages"][first_user_idx]["content"].append(
                                {
                                    "type": "image",
                                    "image": image_value,
                                }
                            )
                else:
                    # Original single image behavior
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
                        ]["image"] = loaded_images[0]
                    else:
                        # if no image type is found, add it to end of the first user message
                        if first_user_idx is None:
                            first_user_idx = 0
                        processed_example["messages"][first_user_idx]["content"].append(
                            {
                                "type": "image",
                                "image": loaded_images[0],
                            }
                        )

            processed_examples.append(remove_none_values(processed_example))
            
        # # Debug: Log final processed examples (only first 10 steps)
        # if self._process_step_count <= 10:
        #     LOG.debug(f"[PROCESSING STRATEGY] Returning {len(processed_examples)} processed examples")
        #     if processed_examples:
        #         first_example = processed_examples[0]
        #         if "messages" in first_example:
        #             roles = [msg.get("role", "unknown") for msg in first_example["messages"]]
        #             LOG.debug(f"[PROCESSING STRATEGY] First example message roles: {roles}")

        return processed_examples

    def _mask_non_assistant(self, labels: Tensor) -> Tensor:
        """
        Mask non assistant regions to -100.
        To be implemented per subclass.
        """
        # LOG.debug(f"ProcessingStrategy._mask_non_assistant called (base implementation) - returning labels unchanged")
        return labels

    def process_labels(self, input_ids: Tensor) -> Tensor:
        # if not hasattr(self, '_labels_step_count'):
        #     self._labels_step_count = 0
        # self._labels_step_count += 1
        # 
        # if self._labels_step_count <= 10:
        #     LOG.debug(f"{self.__class__.__name__}.process_labels called with input_ids shape: {input_ids.shape}")
        labels = input_ids.clone()

        # if self._labels_step_count <= 10:
        #     LOG.debug(f"{self.__class__.__name__}.process_labels: About to call _mask_non_assistant")
        labels = self._mask_non_assistant(labels)
        # if self._labels_step_count <= 10:
        #     LOG.debug(f"{self.__class__.__name__}.process_labels: _mask_non_assistant completed")

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Ignore the image token index in the loss computation (model specific)
        labels[labels == self.image_token_id] = -100

        # if self._labels_step_count <= 10:
        #     LOG.debug(f"{self.__class__.__name__}.process_labels completed, returning labels shape: {labels.shape}")
        return labels


class Qwen2VLProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Qwen2-VL"""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        unmask_image_tokens: bool = True,
        unmask_images_in_user_messages_only: bool = False,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        self.supports_multi_images = True  # Qwen2-VL supports multiple images
        self.image_token = "<|image_pad|>"  # nosec
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
            self.image_token
        )
        self.unmask_image_tokens = unmask_image_tokens
        self.unmask_images_in_user_messages_only = unmask_images_in_user_messages_only
        # LOG.info(f"[MASKING CONFIG] unmask_image_tokens={unmask_image_tokens}, unmask_images_in_user_messages_only={unmask_images_in_user_messages_only}")
    
    def process_labels(self, input_ids: Tensor) -> Tensor:
        """Override to NOT mask image tokens after _mask_non_assistant"""
        # LOG.debug(f"{self.__class__.__name__}.process_labels called with input_ids shape: {input_ids.shape}")
        labels = input_ids.clone()

        # LOG.debug(f"{self.__class__.__name__}.process_labels: About to call _mask_non_assistant")
        labels = self._mask_non_assistant(labels)
        # LOG.debug(f"{self.__class__.__name__}.process_labels: _mask_non_assistant completed")

        # Only mask padding tokens - NOT image tokens!
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # Do NOT mask image tokens - we need them for multimodal learning
        # (This is different from the base class which masks image tokens)
        
        # LOG.debug(f"{self.__class__.__name__}.process_labels completed, returning labels shape: {labels.shape}")
        return labels

    def _mask_non_assistant(self, labels: Tensor) -> Tensor:
        """
        Mask non-assistant tokens in Qwen2VL/Qwen3VL ChatML format.
        Only assistant responses will be used for loss calculation.
        """
        import logging
        LOG = logging.getLogger("axolotl.qwen2vl_masking")
        
        tokenizer = self.processor.tokenizer
        
        # Tokenize the special tokens used in ChatML format
        # Assistant messages start with: <|im_start|>assistant\n
        # All messages end with: <|im_end|>
        start_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        
        # # Debug: Log once per session
        # if not hasattr(self, '_debug_logged'):
        #     LOG.info(f"Qwen2VL masking initialized - start_tokens: {start_tokens}, end_tokens: {end_tokens}")
        #     # Also log user tokens for debugging
        #     user_start_tokens_debug = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        #     LOG.info(f"Qwen2VL masking - user_start_tokens: {user_start_tokens_debug}")
        #     self._debug_logged = True
        #     self._debug_counter = 0
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        
        # # Add immediate debug for first few samples
        # self._debug_counter = getattr(self, '_debug_counter', 0) + 1
        # if self._debug_counter <= 10:  # Increased to see more samples
        #     LOG.info(f"[EMERGENCY DEBUG #{self._debug_counter}] Processing batch with {labels.shape[0]} samples, shape: {labels.shape}")
        #     
        #     # Extra debug: Check if we have the right tokens
        #     if self._debug_counter == 1:
        #         # Decode the full sequence to see what we're working with
        #         if hasattr(self.processor, 'tokenizer'):
        #             tok = self.processor.tokenizer
        #         else:
        #             tok = tokenizer
        #         sample_tokens = labels[0].tolist()
        #         # Remove padding tokens for cleaner output
        #         non_pad_tokens = [t for t in sample_tokens if t != tok.pad_token_id]
        #         LOG.info(f"[EMERGENCY DEBUG] Non-padding token count: {len(non_pad_tokens)} out of {len(sample_tokens)}")
        #         
        #         # Decode and show the full text
        #         decoded_text = tok.decode(non_pad_tokens)
        #         LOG.info(f"[EMERGENCY DEBUG] Full decoded text length: {len(decoded_text)} chars")
        #         # Show the last part where assistant should be
        #         if len(decoded_text) > 500:
        #             LOG.info(f"[EMERGENCY DEBUG] Last 500 chars: ...{decoded_text[-500:]}")
        
        masked_labels = labels.clone()
        
        # # Track debug stats
        # if not hasattr(self, '_mask_call_count'):
        #     self._mask_call_count = 0
        # self._mask_call_count += 1
        
        for i in range(labels.shape[0]):
            label_list = labels[i].tolist()
            
            # Start by masking everything
            masked_labels[i, :] = -100
            
            # Debug stats for this sample
            total_tokens = len(label_list)
            assistant_regions = []
            
            # Find all assistant message boundaries
            j = 0
            while j < len(label_list):
                # Skip padding tokens
                if label_list[j] == tokenizer.pad_token_id:
                    j += 1
                    continue
                    
                # Look for start of assistant message
                if self._match_tokens(label_list, start_tokens, j):
                    start_idx = j + len(start_tokens)
                    
                    # Find corresponding end token
                    found_end = False
                    for k in range(start_idx, len(label_list) - len(end_tokens) + 1):
                        if self._match_tokens(label_list, end_tokens, k):
                            # Record region for debugging
                            assistant_regions.append((start_idx, k))
                            
                            # Unmask ALL assistant response tokens (between start and end markers)
                            # INCLUDING image tokens - the model needs to learn image-text relationships
                            for idx in range(start_idx, k):
                                masked_labels[i, idx] = labels[i, idx]
                            
                            j = k + len(end_tokens)
                            found_end = True
                            break
                    
                    if not found_end:
                        # If we didn't find an end token, skip to next position
                        j += 1
                else:
                    j += 1
            
            # Handle image token unmasking based on configuration
            image_positions = []
            user_regions = []
            
            if self.unmask_image_tokens:
                if self.unmask_images_in_user_messages_only:
                    # Find user message regions to unmask image tokens ONLY in user messages
                    # User messages start with: <|im_start|>user\n
                    # All messages end with: <|im_end|>
                    user_start_tokens = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
                    
                    # # Debug: Log user token search for first few samples
                    # if self._debug_counter <= 3 and i == 0:
                    #     LOG.info(f"[USER REGION DEBUG] Looking for user_start_tokens: {user_start_tokens}")
                    #     LOG.info(f"[USER REGION DEBUG] label_list length: {len(label_list)}")
                    #     # Search manually to see where we might find it
                    #     found_any_im_start = False
                    #     for check_j in range(min(100, len(label_list) - len(user_start_tokens) + 1)):
                    #         if label_list[check_j] == 151644:  # <|im_start|>
                    #             found_any_im_start = True
                    #             next_tokens = label_list[check_j:check_j+5]
                    #             LOG.info(f"  Found <|im_start|> (151644) at position {check_j}, next tokens: {next_tokens}")
                    #             # Check if this matches user pattern
                    #             matches_user = label_list[check_j:check_j+len(user_start_tokens)] == user_start_tokens
                    #             LOG.info(f"    Matches user pattern? {matches_user}")
                    #     if not found_any_im_start:
                    #         LOG.info(f"  WARNING: No <|im_start|> (151644) found in first 100 tokens!")
                    #         LOG.info(f"  First 20 tokens: {label_list[:20]}")
                    
                    # Find all user message boundaries
                    j = 0
                    while j < len(label_list):
                        # Skip padding tokens
                        if label_list[j] == tokenizer.pad_token_id:
                            j += 1
                            continue
                            
                        # Look for start of user message
                        if self._match_tokens(label_list, user_start_tokens, j):
                            start_idx = j + len(user_start_tokens)
                            
                            # Find corresponding end token
                            found_end = False
                            for k in range(start_idx, len(label_list) - len(end_tokens) + 1):
                                if self._match_tokens(label_list, end_tokens, k):
                                    user_regions.append((start_idx, k))
                                    j = k + len(end_tokens)
                                    found_end = True
                                    break
                            
                            if not found_end:
                                j += 1
                        else:
                            j += 1
                    
                    # Unmask image tokens ONLY within user message regions
                    for idx, token_id in enumerate(label_list):
                        if token_id == self.image_token_id:
                            # Check if this image token is within a user region
                            for user_start, user_end in user_regions:
                                if user_start <= idx < user_end:
                                    masked_labels[i, idx] = labels[i, idx]
                                    image_positions.append(idx)
                                    break
                else:
                    # Unmask ALL image tokens (original behavior)
                    for idx, token_id in enumerate(label_list):
                        if token_id == self.image_token_id:
                            masked_labels[i, idx] = labels[i, idx]
                            image_positions.append(idx)
            
            # # Add immediate debug for first 10 calls - check ALL samples in batch
            # if self._debug_counter <= 10:
            #     LOG.info(f"[EMERGENCY DEBUG] Sample {i}: Found {len(assistant_regions)} assistant regions, {len(user_regions)} user regions")
            #     LOG.info(f"  User regions: {user_regions[:2]}..." if user_regions else "  No user regions found")
            #     LOG.info(f"  Found {len(image_positions)} image tokens in user messages at positions: {image_positions[:10]}...")
            #     if assistant_regions:
            #         LOG.info(f"  Assistant regions: {assistant_regions}")
            #     else:
            #         LOG.info(f"  No regions found! Checking why...")
                    # # Check if tokens exist
                    # has_start = any(self._match_tokens(label_list, start_tokens, j) for j in range(len(label_list) - len(start_tokens)))
                    # has_end = any(self._match_tokens(label_list, end_tokens, j) for j in range(len(label_list) - len(end_tokens)))
                    # LOG.info(f"  Has start tokens in sequence: {has_start}")
                    # LOG.info(f"  Has end tokens in sequence: {has_end}")
                    # # Show a sample of tokens to debug
                    # LOG.info(f"  First 20 tokens: {label_list[:20]}")
                    # LOG.info(f"  Last 20 tokens: {label_list[-20:]}")
                    # 
                    # # Search for where assistant MIGHT be
                    # for j in range(len(label_list) - 2):
                    #     if label_list[j] == 151644:  # <|im_start|>
                    #         next_token = label_list[j+1] if j+1 < len(label_list) else None
                    #         next_decoded = tokenizer.decode([next_token]) if next_token else "EOL"
                    #         LOG.info(f"  Found <|im_start|> at {j}, next token: {next_token} ('{next_decoded}')")
                    # 
                    # # Also show total length
                    # LOG.info(f"  Total sequence length: {len(label_list)} tokens")
                    pass
                    
            # # Debug logging - log detailed info for first 10 calls only
            # if i == 0 and self._debug_counter <= 10:
            #     unmasked_count = (masked_labels[i] != -100).sum().item()
            #     masked_count = total_tokens - unmasked_count
            #     
            #     # Count image tokens in the sample
            #     image_token_count = (labels[i] == self.image_token_id).sum().item()
            #     # Count unmasked image tokens (should be only those in user messages)
            #     unmasked_image_tokens = ((masked_labels[i] != -100) & (labels[i] == self.image_token_id)).sum().item()
            #     unmasked_non_image = ((masked_labels[i] != -100) & (labels[i] != self.image_token_id)).sum().item()
            #     
            #     LOG.info(f"[Mask Debug #{self._debug_counter}] Sample {i+1}/{labels.shape[0]}:")
            #     LOG.info(f"  Total tokens: {total_tokens}, Masked: {masked_count}, Unmasked: {unmasked_count}")
            #     LOG.info(f"  Image tokens: {image_token_count} total, {unmasked_image_tokens} unmasked (in user messages)")
            #     LOG.info(f"  Unmasked breakdown: {unmasked_image_tokens} images (user) + {unmasked_non_image} text (assistant)")
            #     LOG.info(f"  Assistant regions found: {len(assistant_regions)}, User regions found: {len(user_regions)}")
            #     
            #     if assistant_regions:
            #         for region_idx, (start, end) in enumerate(assistant_regions[:2]):  # Show first 2 regions
            #             # Decode a snippet of the assistant response
            #             snippet_tokens = label_list[start:min(start+10, end)]
            #             snippet_text = tokenizer.decode(snippet_tokens)
            #             LOG.info(f"  Region {region_idx+1}: positions {start}-{end}, snippet: '{snippet_text}...'")
            #     
            #     # Also log a sample of the full conversation structure (first 500 tokens)
            #     if total_tokens > 0:
            #         sample_tokens = label_list[:min(500, total_tokens)]
            #         sample_text = tokenizer.decode(sample_tokens)
            #         # Replace newlines for cleaner logging
            #         sample_text = sample_text.replace('\n', '\\n')
            #         if len(sample_text) > 200:
            #             sample_text = sample_text[:200] + "..."
            #         LOG.info(f"  Conversation start: '{sample_text}'")
        
        return masked_labels
    
    def _match_tokens(self, label_list: list, pattern: list, start_idx: int) -> bool:
        """Check if pattern matches at start_idx in label_list"""
        if start_idx + len(pattern) > len(label_list):
            return False
        return label_list[start_idx:start_idx + len(pattern)] == pattern


class Gemma3ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Gemma3"""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        self.image_token = processor.tokenizer.special_tokens_map["boi_token"]
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
            self.image_token
        )

    def process_labels(self, input_ids):
        labels = input_ids.clone()

        # Follows https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        labels[labels == 262144] = -100  # corresponds to <image_soft_token>

        return labels


class Gemma3nProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Gemma3n"""

    def _mask_non_assistant(self, labels: Tensor) -> Tensor:
        def _find_token_sequence(label, start_pos, token_sequence):
            """Check if token_sequence appears at start_pos in label"""
            if start_pos + len(token_sequence) > len(label):
                return False
            if label[start_pos] != token_sequence[0]:
                return False
            return (
                label[start_pos : start_pos + len(token_sequence)].tolist()
                == token_sequence
            )

        def _find_assistant_end(label, start_pos, assistant_end_tok, mask, i):
            """
            Find the end of assistant response and update mask accordingly

            Returns new position to continue from and whether the end seq is found
            """
            k = start_pos
            while k < len(label):
                if not _find_token_sequence(label, k, assistant_end_tok):
                    mask[i][k] = 1
                    k += 1
                    continue

                return k + len(assistant_end_tok), True

            return k, False

        mask = zeros_like(labels)

        assistant_start_str = "<start_of_turn>model"
        assistant_end_str = "<end_of_turn>"
        include_assistant_start_tok = False
        include_assistant_end_tok = True

        # str to tokens
        assistant_start_tok = self.processor.tokenizer.encode(
            assistant_start_str, add_special_tokens=False
        )
        assistant_end_tok = self.processor.tokenizer.encode(
            assistant_end_str, add_special_tokens=False
        )

        for i, label in enumerate(labels):
            j = 0
            # while loop through each tok index in labels[i]
            while j < len(label):
                # Check until match start seq
                if not _find_token_sequence(label, j, assistant_start_tok):
                    j += 1
                    continue

                if include_assistant_start_tok:
                    mask[i][j : j + len(assistant_start_tok)] = 1

                # Find where the assistant response ends
                start_of_content = j + len(assistant_start_tok)
                end_pos, found_end_seq = _find_assistant_end(
                    label, start_of_content, assistant_end_tok, mask, i
                )

                # Include end token if requested
                if include_assistant_end_tok and found_end_seq:
                    mask[i][end_pos - len(assistant_end_tok) : end_pos] = 1

                j = end_pos

            labels[i][mask[i] == 0] = -100

        return labels

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)

        # Follows https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/fine_tune_gemma3n_on_t4.ipynb
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        if hasattr(self.processor.tokenizer, "image_token_id"):
            labels[labels == self.processor.tokenizer.image_token_id] = -100
        if hasattr(self.processor.tokenizer, "audio_token_id"):
            labels[labels == self.processor.tokenizer.audio_token_id] = -100
        if hasattr(self.processor.tokenizer, "boi_token_id"):
            labels[labels == self.processor.tokenizer.boi_token_id] = -100
        if hasattr(self.processor.tokenizer, "eoi_token_id"):
            labels[labels == self.processor.tokenizer.eoi_token_id] = -100

        return labels


class VoxtralProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Voxtral"""

    def __init__(
        self,
        processor: VoxtralProcessor,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        special_ids = (
            processor.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.special_ids
        )

        self.audio_token = special_ids.audio
        self.begin_audio_token = special_ids.begin_audio

    def process_labels(self, input_ids):
        labels = input_ids.clone()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.audio_token] = -100
        labels[labels == self.begin_audio_token] = -100

        return labels


class SmolVLM2ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for SmolVLM2"""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        self.image_token = "<image>"  # nosec

        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index(self.image_token)
        ]


class Mistral3ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Mistral3"""

    def __init__(
        self,
        processor: Mistral3Processor,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        special_ids = (
            processor.tokenizer.tokenizer.instruct_tokenizer.image_encoder.special_ids
        )

        self.image_token = special_ids.img
        self.image_break_token = special_ids.img_break
        self.image_end_token = special_ids.img_end

    def process_labels(self, input_ids):
        labels = input_ids.clone()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token] = -100
        labels[labels == self.image_break_token] = -100
        labels[labels == self.image_end_token] = -100

        return labels


def get_processing_strategy(
    processor: ProcessorMixin,
    chat_template,
    chat_template_type,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Resampling | None = None,
    unmask_image_tokens: bool = True,
    unmask_images_in_user_messages_only: bool = False,
):
    processing_kwargs = {
        "processor": processor,
        "chat_template": chat_template,
        "image_size": image_size,
        "image_resize_algorithm": image_resize_algorithm,
    }

    if chat_template_type in [None, "tokenizer_default"] and hasattr(
        processor.tokenizer, "chat_template"
    ):
        processing_kwargs["chat_template"] = processor.tokenizer.chat_template

    # Qwen3-VL uses the same processing strategy as Qwen2-VL
    # Both support multiple images and have the same tokenization format
    if chat_template_type == "qwen2_vl":
        # Check if this is actually a Qwen3-VL model by looking at the model type in the processor
        if hasattr(processor, 'model_type') and 'qwen3' in str(processor.model_type).lower():
            # LOG.info("Detected Qwen3-VL model, using Qwen2VLProcessingStrategy for compatibility")
            pass
        return Qwen2VLProcessingStrategy(
            **processing_kwargs,
            unmask_image_tokens=unmask_image_tokens,
            unmask_images_in_user_messages_only=unmask_images_in_user_messages_only,
        )
    if chat_template_type == "gemma3":
        return Gemma3ProcessingStrategy(
            **processing_kwargs,
        )
    if chat_template_type == "gemma3n":
        return Gemma3nProcessingStrategy(
            **processing_kwargs,
        )

    if isinstance(processor, VoxtralProcessor):
        return VoxtralProcessingStrategy(
            **processing_kwargs,
        )

    if isinstance(processor, SmolVLMProcessor):
        return SmolVLM2ProcessingStrategy(
            **processing_kwargs,
        )

    if isinstance(processor, Mistral3Processor):
        return Mistral3ProcessingStrategy(
            **processing_kwargs,
        )

    # llama3_2_vision, llama4, llava
    # mistral_v7_tekken, pixtral, lfm2vl
    return ProcessingStrategy(
        **processing_kwargs,
    )
