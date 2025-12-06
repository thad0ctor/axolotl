"""
Collators for multi-modal chat messages and packing
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy

from axolotl.processing_strategies import ProcessingStrategy

LOG = logging.getLogger("axolotl.utils.collators.mm_chat")


@dataclass
class MultiModalChatDataCollator(DataCollatorMixin):
    """
    Collator for multi-modal chat messages
    """

    tokenizer: PreTrainedTokenizerBase
    processing_strategy: ProcessingStrategy
    packing: bool = False
    return_tensors: str = "pt"
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.packing:
            raise ValueError("Packing is currently not supported.")

    def torch_call(self, examples: list[dict]) -> dict[str, Any]:
        return self.process_rows(examples)

    def process_rows(
        self,
        examples: list[dict],
    ) -> dict[str, Tensor]:
        # # Debug: Check if this is training or eval data
        # import inspect
        # frame = inspect.currentframe()
        # is_training = True
        # try:
        #     # Walk up the stack to see if we're in evaluation
        #     for i in range(10):
        #         if frame is None:
        #             break
        #         if 'eval' in frame.f_code.co_name.lower() or 'evaluate' in frame.f_code.co_name.lower():
        #             is_training = False
        #             break
        #         frame = frame.f_back
        # except:
        #     pass
        # 
        # data_type = "TRAINING" if is_training else "EVAL"
        # self._debug_step_count += 1
        # 
        # if self._debug_step_count <= 10:
        #     LOG.info(f"[{data_type}] MultiModalChatDataCollator.process_rows called with {len(examples)} examples")
        
        # Preprocess the examples
        examples = self.processing_strategy(examples)

        # Initialize batch
        messages = [ex["messages"] for ex in examples]
        
        # # Debug: Check message structure before apply_chat_template (only first 10 steps)
        # if self._debug_step_count <= 10:
        #     for i, msg_list in enumerate(messages):  # Check ALL examples
        #         LOG.info(f"[{data_type} MESSAGE DEBUG] Example {i}: {len(msg_list)} messages")
        #         has_assistant = False
        #         roles_list = []
        #         for j, msg in enumerate(msg_list):
        #             role = msg.get("role", "unknown")
        #             roles_list.append(role)
        #             content_len = len(str(msg.get("content", "")))
        #             if role == "assistant":
        #                 has_assistant = True
        #                 LOG.info(f"  Message {j}: role='{role}', content_len={content_len}")
        #                 LOG.info(f"    Assistant content preview: {str(msg.get('content', ''))[:100]}...")
        #         LOG.info(f"[{data_type} MESSAGE DEBUG] Roles in example {i}: {roles_list}")
        #         if not has_assistant:
        #             LOG.warning(f"[{data_type} MESSAGE DEBUG] Example {i} has NO ASSISTANT message!")

        # # Debug: Check processor settings (only first 10 steps)
        # if self._debug_step_count <= 10 and hasattr(self.processing_strategy.processor, 'tokenizer'):
        #     tokenizer = self.processing_strategy.processor.tokenizer
        #     LOG.info(f"[TOKENIZER DEBUG] model_max_length: {getattr(tokenizer, 'model_max_length', 'N/A')}")
        #     LOG.info(f"[TOKENIZER DEBUG] truncation: {getattr(tokenizer, 'truncation', 'N/A')}")
        #     LOG.info(f"[TOKENIZER DEBUG] padding: {getattr(tokenizer, 'padding', 'N/A')}")

        # Apply chat template with explicit truncation=False to preserve full sequences
        batch = self.processing_strategy.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            chat_template=self.processing_strategy.chat_template,
            truncation=False,  # Disable truncation to preserve full sequences
            max_length=None,   # No max length limit
        )

        # # Debug: Check what tokens we got back (only first 10 steps)
        # if self._debug_step_count <= 10 and batch['input_ids'].shape[0] > 0:
        #     LOG.info(f"[TOKEN DEBUG] Batch shape: {batch['input_ids'].shape}")
        #     # Check for assistant tokens in the tokenized output
        #     assistant_start_tokens = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        #     LOG.info(f"[TOKEN DEBUG] Looking for assistant tokens: {assistant_start_tokens}")
        #     
        #     for i in range(min(1, batch['input_ids'].shape[0])):  # Check first example
        #         tokens_list = batch['input_ids'][i].tolist()
        #         LOG.info(f"[TOKEN DEBUG] Example {i} length: {len(tokens_list)}")
        #         
        #         # Search for assistant tokens
        #         assistant_found = False
        #         for j in range(len(tokens_list) - len(assistant_start_tokens)):
        #             if tokens_list[j:j+len(assistant_start_tokens)] == assistant_start_tokens:
        #                 assistant_found = True
        #                 LOG.info(f"[TOKEN DEBUG] Found assistant at position {j}")
        #                 break
        #         
        #         if not assistant_found:
        #             LOG.warning(f"[TOKEN DEBUG] NO ASSISTANT TOKENS FOUND in example {i}!")
        #             # Decode to see what we have
        #             decoded = self.tokenizer.decode(tokens_list[:200])
        #             LOG.info(f"[TOKEN DEBUG] First 200 tokens decoded: {decoded}")
        #             # Also check the end
        #             decoded_end = self.tokenizer.decode(tokens_list[-50:])
        #             LOG.info(f"[TOKEN DEBUG] Last 50 tokens decoded: {decoded_end}")

        # # Process the labels
        # if self._debug_step_count <= 10:
        #     LOG.debug(f"MultiModalChatDataCollator: About to call process_labels with input_ids shape: {batch['input_ids'].shape}")
        #     LOG.debug(f"MultiModalChatDataCollator: Processing strategy class: {type(self.processing_strategy).__name__}")
        # 
        # # CRITICAL DEBUG: Check if assistant tokens exist right before process_labels (only first 10 steps)
        # if self._debug_step_count <= 10 and batch['input_ids'].shape[0] > 0:
        #     assistant_start_tokens = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        #     tokens_to_check = batch['input_ids'][0].tolist()
        #     assistant_positions = []
        #     for j in range(len(tokens_to_check) - len(assistant_start_tokens)):
        #         if tokens_to_check[j:j+len(assistant_start_tokens)] == assistant_start_tokens:
        #             assistant_positions.append(j)
        #     
        #     if assistant_positions:
        #         LOG.info(f"[CRITICAL DEBUG] Found {len(assistant_positions)} assistant regions in batch before process_labels at positions: {assistant_positions}")
        #     else:
        #         LOG.error(f"[CRITICAL DEBUG] NO ASSISTANT TOKENS in batch before process_labels!")
        #         LOG.info(f"[CRITICAL DEBUG] Sequence length: {len(tokens_to_check)}")
        #         LOG.info(f"[CRITICAL DEBUG] First 50 tokens: {tokens_to_check[:50]}")
        #         LOG.info(f"[CRITICAL DEBUG] Last 50 tokens: {tokens_to_check[-50:]}")
        #         # Decode to understand what's there
        #         decoded = self.tokenizer.decode(tokens_to_check)
        #         LOG.info(f"[CRITICAL DEBUG] Full decoded length: {len(decoded)} chars")
        #         if len(decoded) > 400:
        #             LOG.info(f"[CRITICAL DEBUG] Last 400 chars: ...{decoded[-400:]}")
        
        batch["labels"] = self.processing_strategy.process_labels(batch["input_ids"])
        
        # if self._debug_step_count <= 10:
        #     LOG.debug(f"MultiModalChatDataCollator: process_labels completed, labels shape: {batch['labels'].shape}")
        #     LOG.debug(f"MultiModalChatDataCollator: Labels tensor (first 100 tokens): {batch['labels'][0][:100].tolist() if batch['labels'].shape[0] > 0 else 'Empty'}")
        # 
        # # Better debug - check for unmasked tokens including images (only first 10 steps)
        # if self._debug_step_count <= 10 and batch['labels'].shape[0] > 0:
        #     labels = batch['labels']
        #     non_masked_indices = (labels[0] != -100).nonzero(as_tuple=True)[0]
        #     if len(non_masked_indices) > 0:
        #         LOG.debug(f"MASK ANALYSIS: Found {len(non_masked_indices)} unmasked tokens out of {labels.shape[1]} total ({len(non_masked_indices)/labels.shape[1]*100:.1f}%)")
        #         LOG.debug(f"  First unmasked at position: {non_masked_indices[0].item()}")
        #         LOG.debug(f"  Last unmasked at position: {non_masked_indices[-1].item()}")
        #         
        #         # Check if any unmasked tokens are image tokens (151655 for Qwen2VL)
        #         image_token_id = 151655
        #         unmasked_values = labels[0][non_masked_indices]
        #         image_count = (unmasked_values == image_token_id).sum().item()
        #         LOG.debug(f"  Unmasked image tokens: {image_count} out of {len(non_masked_indices)} unmasked")
        #         
        #         # Show sample of unmasked token IDs to verify
        #         sample_unmasked = unmasked_values[:10].tolist()
        #         LOG.debug(f"  Sample unmasked token IDs: {sample_unmasked}")
        #         
        #         # Also count total image tokens in the input
        #         input_ids = batch['input_ids']
        #         total_images = (input_ids[0] == image_token_id).sum().item()
        #         LOG.debug(f"  Total image tokens in input: {total_images}")
        #     else:
        #         LOG.debug("WARNING: No unmasked tokens found in this batch!")

        return batch
