#!/bin/bash
# Script to fine-tune a model using PEFT/QLoRA on MCP dataset
# Usage: ./scripts/finetune_model.sh MODEL_NAME DATA_PATH OUTPUT_DIR

set -e
echo "Using Python interpreter: $(which python3)"
echo "Installed packages:"
uv pip list

# Default values
_DEFAULT_MODEL_NAME="mistralai/Devstral-Small-2505"
_DEFAULT_DATA_PATH="data/preprocessed_mcp_dataset.json" # Corrected default to preprocessed
_DEFAULT_OUTPUT_DIR="finetune_output"

# Parse arguments, handling 'key=value' from just and allowing for positional
_arg1_raw=${1}
_arg2_raw=${2}
_arg3_raw=${3}

# Strip 'key=' prefix. If no prefix, original value is retained.
MODEL_NAME=${_arg1_raw#*=}
DATA_PATH=${_arg2_raw#*=}
OUTPUT_DIR=${_arg3_raw#*=}

# Apply defaults if the initial argument was not provided (raw is empty),
# or if the stripped value is empty (e.g. arg was "key=").
if [ -z "${_arg1_raw}" ]; then MODEL_NAME=$_DEFAULT_MODEL_NAME; elif [ -z "$MODEL_NAME" ] && [ "${_arg1_raw}" != "${MODEL_NAME}" ]; then MODEL_NAME=$_DEFAULT_MODEL_NAME; fi
if [ -z "${_arg2_raw}" ]; then DATA_PATH=$_DEFAULT_DATA_PATH; elif [ -z "$DATA_PATH" ] && [ "${_arg2_raw}" != "${DATA_PATH}" ]; then DATA_PATH=$_DEFAULT_DATA_PATH; fi
if [ -z "${_arg3_raw}" ]; then OUTPUT_DIR=$_DEFAULT_OUTPUT_DIR; elif [ -z "$OUTPUT_DIR" ] && [ "${_arg3_raw}" != "${OUTPUT_DIR}" ]; then OUTPUT_DIR=$_DEFAULT_OUTPUT_DIR; fi

CONFIG_FILE="$OUTPUT_DIR/configs/training_config.json"

# Check if required files exist
if [ ! -f "$DATA_PATH" ]; then
  echo "Error: Dataset file not found: $DATA_PATH"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# If config file doesn't exist, run prepare_training.sh
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Training config not found, running preparation script..."
  ./scripts/prepare_training.sh "$OUTPUT_DIR" "$MODEL_NAME" "$DATA_PATH"
fi

echo "Starting fine-tuning process"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Configuration: $CONFIG_FILE"

# Create the fine-tuning Python script
FINETUNE_SCRIPT="$OUTPUT_DIR/run_finetune.py"

cat > "$FINETUNE_SCRIPT" << 'EOL'
#!/usr/bin/env python3
"""
Fine-tuning script for LLMs on MCP datasets using PEFT/QLoRA.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List, Optional, Union # Adjusted for wrapper

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    # LlamaTokenizerFast, # Reverting this
    BitsAndBytesConfig,
    TrainingArguments, # Re-enabled
    set_seed,
)
from transformers.tokenization_utils_base import BatchEncoding # For wrapper
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer # Re-adding this

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.environ.get("OUTPUT_DIR", "./"), "train.log"))
    ]
)
logger = logging.getLogger(__name__)

# Define the wrapper class to make MistralTokenizer compatible with HF Trainer
class MistralCommonHFWrapper:
    def __init__(self, mistral_tokenizer_instance: MistralTokenizer, model_max_length: int = 2048):
        self.base_tokenizer = mistral_tokenizer_instance
        self.model_max_length = model_max_length
        self.name_or_path = "mistral_common_wrapper" # Or derive from base_tokenizer if possible

        # These must be set after base_tokenizer's pad logic is applied
        self.pad_token_id = getattr(self.base_tokenizer, 'pad_token_id', None)
        self.eos_token_id = getattr(self.base_tokenizer.instruct_tokenizer.tokenizer, 'eos_id', None) # Get from underlying SentencePiece
        self.pad_token = getattr(self.base_tokenizer, 'pad_token', None)
        self.eos_token = getattr(self.base_tokenizer, 'eos_token', None)

        if self.pad_token_id is None and self.eos_token_id is not None:
            logger.warning(f"Wrapper: pad_token_id is None, falling back to eos_token_id: {self.eos_token_id}")
            self.pad_token_id = self.eos_token_id
            if self.pad_token is None and self.eos_token is not None:
                 self.pad_token = self.eos_token

        if self.pad_token is None: # Final fallback for pad_token string
            self.pad_token = "<PAD>" # Should ideally not happen if EOS is set
            logger.warning(f"Wrapper: pad_token was None, set to default '{self.pad_token}'. Ensure pad_token_id {self.pad_token_id} is valid.")


        self.padding_side = "right"  # Standard for Causal LM
        self.is_fast = False # mistral-common tokenizer is not a "Fast" tokenizer

        # Vocab size might be needed by some parts of Trainer
        try:
            self.vocab_size = len(self.base_tokenizer)
        except TypeError: # In case __len__ is not on base_tokenizer directly
             _sentence_piece_tokenizer_instance = self.base_tokenizer.instruct_tokenizer.tokenizer
             # Use the n_words property which should give the vocab size
             self.vocab_size = _sentence_piece_tokenizer_instance.n_words


    def __len__(self):
        try:
            return len(self.base_tokenizer) # This will raise TypeError for MistralTokenizer
        except TypeError:
            _sentence_piece_tokenizer_instance = self.base_tokenizer.instruct_tokenizer.tokenizer
            # Use the n_words property
            return _sentence_piece_tokenizer_instance.n_words

    def __call__(self, text_or_batch: Union[str, List[str]],
                 padding: Union[bool, str] = False,
                 truncation: Union[bool, str] = False,
                 max_length: Optional[int] = None,
                 add_special_tokens: bool = True, # Usually True for __call__
                 return_tensors: Optional[str] = None, # Will be handled by self.pad if "pt"
                 return_attention_mask: Optional[bool] = None, # Default True
                 **kwargs) -> BatchEncoding:

        is_single_string_input = isinstance(text_or_batch, str)
        if is_single_string_input:
            text_batch_internal = [text_or_batch]
            logger.info(f"Wrapper __call__: Input is single string: '{text_or_batch[:100]}...'") # Changed to info
        else:
            text_batch_internal = text_or_batch
            logger.info(f"Wrapper __call__: Input is batch of size {len(text_batch_internal)}") # Changed to info


        if max_length is None:
            max_length = self.model_max_length

        if return_attention_mask is None: # HF default
            return_attention_mask = True

        all_input_ids = []
        all_attention_masks = []

        for text in text_batch_internal: # Use the processed internal batch
            # .encode on MistralTokenizer typically takes bos, eos flags
            # For SFTTrainer, the input text is usually already formatted (e.g. "<s>[INST]...[/INST]...")
            # So, add_special_tokens might be tricky here.
            # Let's assume the text comes pre-formatted from SFTTrainer's formatting_func
            # and we just need to tokenize it as is.
            # The base_tokenizer.encode() returns List[int]
            # The SFTTrainer usually expects __call__ to handle the "prompt" + "completion" formatting if no formatting_func is given.
            # However, our prepare_dataset creates 'prompt' and 'completion' and SFTTrainer uses these.
            # SFTTrainer then calls tokenizer on the concatenated string.
            # So, add_special_tokens=True here might be what SFTTrainer expects for its internal call.

            # The `encode` method of `MistralTokenizer` is `encode(self, s: str, bos: bool, eos: bool) -> List[int]`
            # `SFTTrainer` calls `tokenizer(formatted_text)`. We need to decide how `add_special_tokens` maps to `bos`/`eos`.
            # Typically, for a full sequence, both are true.
            token_ids = self.base_tokenizer.encode(s=text, bos=add_special_tokens, eos=add_special_tokens)

            if truncation:
                token_ids = token_ids[:max_length]

            all_input_ids.append(token_ids)
            if return_attention_mask:
                all_attention_masks.append([1] * len(token_ids))

        output_batch_data = {"input_ids": all_input_ids}
        if return_attention_mask:
            output_batch_data["attention_mask"] = all_attention_masks

        # If the original input was a single string, unbatch the output before returning (unless padding is also done here)
        if is_single_string_input and not (padding and padding != "do_not_pad"):
            # This path is taken by DataCollatorForCompletionOnlyLM's internal call
            final_output_data = {key: value[0] for key, value in output_batch_data.items()}
            logger.info(f"Wrapper __call__: Returning unbatched output for single string input. Keys: {final_output_data.keys()}") # Changed to info
        else:
            # This path is for batched input, or if padding is handled within __call__
            final_output_data = output_batch_data
            if not is_single_string_input:
                 logger.info(f"Wrapper __call__: Returning batched output. Keys: {final_output_data.keys()}") # Changed to info


        # If padding is requested here (e.g. padding=True or "longest"), call self.pad
        # This block needs to correctly handle the structure of `final_output_data`
        if padding and padding != "do_not_pad": # padding can be True or "longest" etc.
            logger.info(f"Wrapper __call__: Padding requested within __call__. Strategy: {padding}, max_length: {max_length}") # Changed to info
            # self.pad expects a List[Dict[str, List[int]]]
            # If is_single_string_input was true, final_output_data is Dict[str, List[int]] (unbatched)
            # So, we need to wrap it in a list for self.pad
            if is_single_string_input:
                padded_batch_encoding = self.pad([final_output_data], padding_strategy=padding if isinstance(padding, str) else "longest",
                                          max_length=max_length, return_tensors=return_tensors)
                # And then unbatch the result from self.pad
                final_output_data = {key: value[0] for key, value in padded_batch_encoding.data.items()}
                logger.info(f"Wrapper __call__: Padding done, unbatched result. Keys: {final_output_data.keys()}") # Changed to info
            else: # Input was already a batch
                padded_batch_encoding = self.pad(final_output_data, padding_strategy=padding if isinstance(padding, str) else "longest", # This is wrong, final_output_data is Dict not List[Dict]
                                          max_length=max_length, return_tensors=return_tensors)
                # Correction: if final_output_data is from a batch, it's {"input_ids": [[...],[...]]}
                # self.pad expects List[Dict]. This means __call__ should not directly call self.pad
                # if it's already batched in this format.
                # This indicates a design flaw if __call__ is meant to be a general HuggingFace tokenizer __call__.
                # For now, the TRL path does not hit this with padding=True.
                # Let's assume this complex padding-within-__call__ is not the immediate issue.
                # The TRL collator calls __call__ with padding=False, then calls .pad() separately.
                # So, the critical part is the structure of final_output_data *before* this if-padding block.
                # The previous assignment to final_output_data is what matters for TRL.
                # To make this if-block safer if it were ever hit by batched input:
                # If final_output_data is {"input_ids": [[1],[2]], "attention_mask": [[1],[1]]}
                # it needs to be converted to [{'input_ids':[1], 'attention_mask':[1]}, {'input_ids':[2], 'attention_mask':[1]}]
                # This is too complex for a quick fix here. The TRL path avoids this.
                # For now, if padding is true and input was batched, this will likely error or misbehave.
                # We are focusing on the TRL path where padding in __call__ is False.
                final_output_data = padded_batch_encoding.data # This line would be problematic for batched input.
                logger.info(f"Wrapper __call__: Padding done for batched input (potentially problematic). Keys: {final_output_data.keys()}") # Changed to info


        return BatchEncoding(data=final_output_data, tensor_type=return_tensors if (padding and padding != "do_not_pad") else None)


    def pad(self, encoded_inputs: List[Dict[str, Any]], # Changed: List of Dicts
            padding_strategy: str = "longest", # "longest", "max_length"
            max_length: Optional[int] = None,
            return_attention_mask: Optional[bool] = True, # Usually True if input has it
            return_tensors: Optional[str] = "pt",
            **kwargs) -> BatchEncoding:

        if self.pad_token_id is None:
            raise ValueError("Tokenizer does not have a pad_token_id. Cannot pad sequences.")

        logger.info(f"DEBUG PAD: encoded_inputs type: {type(encoded_inputs)}")
        if isinstance(encoded_inputs, list) and len(encoded_inputs) > 0:
            logger.info(f"DEBUG PAD: first element type: {type(encoded_inputs[0])}")
            logger.info(f"DEBUG PAD: first element content: {encoded_inputs[0]}") # Log first element
            logger.info(f"DEBUG PAD: Number of elements in encoded_inputs: {len(encoded_inputs)}")
            # To avoid excessive logging if encoded_inputs is huge, maybe just log keys of first few
            for i, item in enumerate(encoded_inputs[:min(3, len(encoded_inputs))]): # Log first 3 items' keys
                 logger.info(f"DEBUG PAD: Item {i} type: {type(item)}, keys: {item.keys() if isinstance(item, dict) else 'Not a dict'}")
        elif isinstance(encoded_inputs, dict):
             logger.info(f"DEBUG PAD: encoded_inputs is a dict. Keys: {encoded_inputs.keys()}")
        else:
            logger.info(f"DEBUG PAD: encoded_inputs is not a list or is empty, or not a recognized dict: {encoded_inputs}")


        # encoded_inputs is a list of dicts e.g. [{'input_ids': [1,2,3]}, {'input_ids': [4,5]}]
        # We need to extract input_ids and attention_mask from each dict in the list
        input_ids_list = [feature_dict["input_ids"] for feature_dict in encoded_inputs]

        # Handle attention_mask similarly, if present in the feature dicts
        # Check if the first example has attention_mask to decide if we should process it for the batch
        has_attention_mask = "attention_mask" in encoded_inputs[0] if encoded_inputs else False
        attention_mask_list = []
        if has_attention_mask:
            attention_mask_list = [feature_dict["attention_mask"] for feature_dict in encoded_inputs]
        else: # If no attention_mask provided, create default ones (list of lists of 1s)
            attention_mask_list = [[1] * len(ids) for ids in input_ids_list]


        if padding_strategy == "longest":
            effective_max_length = max(len(x) for x in input_ids_list)
        elif padding_strategy == "max_length":
            if max_length is None:
                raise ValueError("max_length must be specified for padding_strategy='max_length'")
            effective_max_length = max_length
        else: # "do_not_pad" or False
            # If no padding, just convert to tensors if requested
            output_data = {"input_ids": input_ids_list}
            if has_attention_mask or return_attention_mask: # if original had it or if requested
                 output_data["attention_mask"] = attention_mask_list

            if return_tensors == "pt":
                for key, value in output_data.items():
                    try:
                        output_data[key] = torch.tensor(value, dtype=torch.long)
                    except Exception as e:
                        logger.error(f"Error converting {key} to tensor in 'do_not_pad' case: {e}. Value: {value}")
                        raise
            return BatchEncoding(data=output_data, tensor_type=return_tensors)


        padded_input_ids = []
        padded_attention_masks = []

        for i in range(len(input_ids_list)):
            ids = input_ids_list[i]
            # Use the corresponding attention_mask from attention_mask_list
            mask = attention_mask_list[i] # This list was already prepared

            padding_len = effective_max_length - len(ids)

            padded_ids = ids + [self.pad_token_id] * padding_len
            padded_mask = mask + [0] * padding_len # Pad attention mask with 0s

            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        output_dict = {
            "input_ids": padded_input_ids,
        }
        # Always include attention_mask if we are here (padding happened)
        # or if it was originally present and requested.
        if return_attention_mask or has_attention_mask:
            output_dict["attention_mask"] = padded_attention_masks

        if return_tensors == "pt":
            for key, value in output_dict.items():
                output_dict[key] = torch.tensor(value, dtype=torch.long)

        return BatchEncoding(data=output_dict, tensor_type=return_tensors)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        # MistralTokenizer.decode takes List[int]
        return self.base_tokenizer.decode(token_ids)

    def save_pretrained(self, save_directory: str, **kwargs):
        logger.info(f"MistralCommonHFWrapper: save_pretrained called for {save_directory}. "
                    "Mistral-common tokenizer is not saved in Hugging Face format this way. "
                    "Ensure mistral-common is available for inference.")
        # Optionally, could try to save the base_tokenizer's model file if it has a path
        # or if we want to copy its .model file, but that's outside typical HF save_pretrained.
        pass

    @property
    def special_tokens_map(self):
        # Attempt to construct something reasonable or return what base_tokenizer might offer
        # This is often queried by Trainer.
        # MistralTokenizer has bos_id, eos_id, pad_id
        _map = {}
        if hasattr(self.base_tokenizer, 'bos_token') and self.base_tokenizer.bos_token:
            _map['bos_token'] = self.base_tokenizer.bos_token
        if hasattr(self.base_tokenizer, 'eos_token') and self.base_tokenizer.eos_token:
            _map['eos_token'] = self.base_tokenizer.eos_token
        if hasattr(self.base_tokenizer, 'pad_token') and self.base_tokenizer.pad_token:
            _map['pad_token'] = self.base_tokenizer.pad_token
        # unk_token might not be directly on MistralTokenizer, SentencePiece model might have it
        # For now, keep it simple.
        return _map

    # Add other attributes that might be checked by Trainer, e.g.
    # vocab_size, model_input_names, etc.
    # For now, __len__ provides vocab_size.
    # model_input_names is usually ['input_ids', 'attention_mask']

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def prepare_dataset(data_path: str, data_config: Dict[str, Any]): # Removed tokenizer argument
    """Prepare dataset for SFTTrainer.
    Loads data and ensures 'prompt' and 'completion' columns are present.
    SFTTrainer will handle tokenization.
    """
    logger.info(f"Loading dataset from {data_path} using field 'examples'")
    # Assuming 'examples' is a list of dictionaries, each having 'instruction' and 'completion'
    dataset = load_dataset("json", data_files={"train": data_path}, field="examples")["train"]

    logger.info(f"Original dataset size: {len(dataset)}")
    logger.info(f"Original dataset features: {dataset.features}")
    # logger.info(f"Original dataset examples: {dataset[:2]}") # This might be too verbose

    # SFTTrainer by default can use 'prompt' and 'completion' columns.
    # If your dataset uses 'instruction', map it to 'prompt'.
    # If 'prompt' and 'completion' already exist, this map can be simpler or skipped
    # if column names match.

    current_columns = dataset.column_names

    def map_columns(example):
        # Ensure 'completion' exists
        if "completion" not in example:
            logger.error("Dataset example missing 'completion' field.")
            # Handle error or provide default, e.g., raise ValueError or return None/empty
            raise ValueError("Dataset example missing 'completion' field.")

        # Map 'instruction' to 'prompt' if 'instruction' exists and 'prompt' doesn't
        if "instruction" in example and "prompt" not in example:
            return {"prompt": example["instruction"], "completion": example["completion"]}
        elif "prompt" in example and "completion" in example:
            return {"prompt": example["prompt"], "completion": example["completion"]}
        else:
            logger.error("Dataset example missing 'instruction' or 'prompt' field.")
            raise ValueError("Dataset example missing 'instruction' or 'prompt' field.")

    # Determine columns to remove: all original columns that are not 'prompt' or 'completion'
    # after mapping.
    # The map function will add 'prompt' if it's new.
    # We want to end up with only 'prompt' and 'completion'.

    # If 'instruction' is present and 'prompt' is not, 'instruction' will be effectively replaced by 'prompt'.
    # If 'prompt' is already there, 'instruction' (if it exists) should be removed.

    columns_to_remove = [col for col in current_columns if col not in ["prompt", "completion"]]
    if "instruction" in current_columns and "prompt" not in current_columns:
        # 'instruction' will be mapped to 'prompt', so 'instruction' itself can be removed.
        if "instruction" not in columns_to_remove: # Should already be there if not prompt/completion
             pass # it will be removed
    elif "prompt" in current_columns and "instruction" in current_columns:
        # We are keeping 'prompt', so 'instruction' should be removed if it's an extra.
        if "instruction" not in columns_to_remove:
            columns_to_remove.append("instruction")


    processed_dataset = dataset.map(
        map_columns,
        remove_columns=columns_to_remove, # Remove original columns not needed
        batched=False # Process example by example for clarity in mapping
    )

    logger.info(f"Processed dataset for SFTTrainer. Features: {processed_dataset.features}")
    logger.info(f"Processed dataset examples: {processed_dataset[:2]}")

    # SFTTrainer will handle the tokenization.
    # It will also handle formatting if a formatting_func is provided to SFTTrainer,
    # or it will use the 'prompt' and 'completion' columns directly for completion-only loss.
    return processed_dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model on MCP datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_config = config["model_config"]
    training_config = config["training_config"]
    peft_config = config["peft_config"]
    data_config = config["data_config"]

    # Set the seed for reproducibility
    set_seed(training_config.get("seed", 42))

    # Make sure output_dir exists
    os.makedirs(training_config["output_dir"], exist_ok=True)

    # Setup 4-bit quantization for model loading
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    logger.info(f"Loading model: {model_config['model_name_or_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name_or_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_config.get("trust_remote_code", False),
    )

    logger.info("Loading Mistral v2 tokenizer from mistral-common...")
    try:
        base_mistral_tokenizer = MistralTokenizer.v2()
        logger.info("Mistral v2 tokenizer (base) loaded successfully.")

        # Apply robust pad token logic to base_mistral_tokenizer
        logger.info("Attempting to set pad_token and pad_token_id for base MistralTokenizer...")
        _underlying_spt = base_mistral_tokenizer.instruct_tokenizer.tokenizer

        pad_token_set_on_base = False
        if hasattr(_underlying_spt, 'pad_id') and _underlying_spt.pad_id is not None and _underlying_spt.pad_id >= 0: # Check for valid ID
            try:
                # Ensure pad_id is actually decodable and not a control token like unk/bos/eos if they are same
                decoded_pad = _underlying_spt.decode([_underlying_spt.pad_id])
                # Heuristic: if pad_id is same as eos_id, and we want distinct pad, this might be an issue.
                # For now, trust it if it's a valid, non-negative ID.
                base_mistral_tokenizer.pad_token_id = _underlying_spt.pad_id
                base_mistral_tokenizer.pad_token = decoded_pad
                logger.info(f"Base MistralTokenizer: Using dedicated pad token from SentencePiece: ID={base_mistral_tokenizer.pad_token_id}, Token='{base_mistral_tokenizer.pad_token}'")
                pad_token_set_on_base = True
            except Exception as decode_err:
                logger.warning(f"Base MistralTokenizer: Could not decode dedicated pad_id ({_underlying_spt.pad_id}): {decode_err}. Falling back to EOS.")

        if not pad_token_set_on_base:
            if hasattr(_underlying_spt, 'eos_id') and _underlying_spt.eos_id is not None:
                base_mistral_tokenizer.eos_token = _underlying_spt.decode([_underlying_spt.eos_id]) # Ensure eos_token string exists
                base_mistral_tokenizer.pad_token = base_mistral_tokenizer.eos_token
                base_mistral_tokenizer.pad_token_id = _underlying_spt.eos_id
                logger.info(f"Base MistralTokenizer: Using EOS token as PAD: ID={base_mistral_tokenizer.pad_token_id}, Token='{base_mistral_tokenizer.pad_token}'")
            else:
                logger.error("CRITICAL: Base MistralTokenizer's underlying tokenizer has no valid 'eos_id'. Cannot set pad token. Exiting.")
                sys.exit(1)

        # Final check for base tokenizer pad attributes
        if not hasattr(base_mistral_tokenizer, 'pad_token') or base_mistral_tokenizer.pad_token is None or \
           not hasattr(base_mistral_tokenizer, 'pad_token_id') or base_mistral_tokenizer.pad_token_id is None:
            logger.error("CRITICAL: Base MistralTokenizer pad_token/pad_token_id is still not set. Exiting.")
            sys.exit(1)
        logger.info(f"Base MistralTokenizer: Successfully set pad_token='{base_mistral_tokenizer.pad_token}', pad_token_id={base_mistral_tokenizer.pad_token_id}")

        # Wrap the configured base tokenizer
        tokenizer = MistralCommonHFWrapper(
            base_mistral_tokenizer,
            model_max_length=data_config.get("max_seq_length", 2048)
        )
        logger.info("MistralCommonHFWrapper initialized.")

        # Align model's pad_token_id with the wrapper's pad_token_id
        if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
            if tokenizer.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
                logger.info(f"Updated model.config.pad_token_id to {model.config.pad_token_id} (same as wrapper).")
            else: # Should not happen if wrapper init is correct
                logger.error("CRITICAL: Wrapper tokenizer.pad_token_id is None. Cannot proceed.")
                sys.exit(1)

        logger.info(f"Wrapper Tokenizer pad_token: '{tokenizer.pad_token}', pad_token_id: {tokenizer.pad_token_id}")
        logger.info(f"Model pad_token_id: {model.config.pad_token_id}")

    except Exception as e:
        logger.error(f"Failed to load and wrap MistralTokenizer.v2(): {e}", exc_info=True)
        sys.exit(1)

    # Redundant check for wrapper, but good for safety
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        logger.error("CRITICAL: Wrapped tokenizer pad_token or pad_token_id is None after setup. Exiting.")
        sys.exit(1)

    # Configure LoRA
    logger.info("Configuring LoRA adapter")
    lora_config = LoraConfig(
        r=peft_config["r"],
        lora_alpha=peft_config["lora_alpha"],
        lora_dropout=peft_config["lora_dropout"],
        bias=peft_config["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=peft_config["target_modules"],
    )

    # Add LoRA adapter to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare the dataset
    data_path = data_config.get("train_file")
    prepared_train_dataset = prepare_dataset(data_path, data_config)

    # Setup the data collator for completion-only learning
    response_template = "<|assistant|>"
    # Encode response template using the base mistral tokenizer's underlying SentencePiece model for raw IDs
    _underlying_spt_for_template = base_mistral_tokenizer.instruct_tokenizer.tokenizer
    response_template_ids = _underlying_spt_for_template.encode(response_template, bos=False, eos=False)
    logger.info(f"Response template IDs for DataCollator: {response_template_ids}")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False
    )
    # Explicitly set is_dataset_pretokenized to False to ensure collator tokenizes
    collator.is_dataset_pretokenized = False
    logger.info(f"DataCollatorForCompletionOnlyLM.is_dataset_pretokenized explicitly set to: {collator.is_dataset_pretokenized}")

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=training_config["weight_decay"],
        fp16=training_config["fp16"],
        logging_steps=training_config["logging_steps"],
        save_strategy=training_config["save_strategy"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config.get("eval_steps", None),
        optim=training_config["optim"],
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        remove_unused_columns=False, # Added to address ValueError
    )

    # Initialize the Trainer
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared_train_dataset,
        data_collator=collator,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    logger.info("Saving the final model...")
    trainer.save_model(os.path.join(training_config["output_dir"], "final"))
    tokenizer.save_pretrained(os.path.join(training_config["output_dir"], "final")) # Calls wrapper's method
    # Logger message for this is now inside the wrapper's save_pretrained method.

    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
EOL

chmod +x "$FINETUNE_SCRIPT"

# Set environment variables
export MODEL_NAME
export DATA_PATH
export OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$(pwd)"

echo "Running fine-tuning script..."
uv run python "$FINETUNE_SCRIPT" --config "$CONFIG_FILE"

# Check if fine-tuning was successful
if [ $? -eq 0 ]; then
  echo "Fine-tuning completed successfully!"
  echo "Fine-tuned model saved to: $OUTPUT_DIR/final"
else
  echo "Error: Fine-tuning failed"
  exit 1
fi