#!/bin/bash
# Script to fine-tune a model using PEFT/QLoRA on MCP dataset
# Usage: ./scripts/finetune_model.sh MODEL_NAME DATA_PATH OUTPUT_DIR

set -e

# Default values
MODEL_NAME=${1:-"mistralai/Mistral-7B-v0.1"}
DATA_PATH=${2:-"data/mcp_dataset.json"}
OUTPUT_DIR=${3:-"finetune_output"}
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
  ./scripts/prepare_training.sh "$OUTPUT_DIR"
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
from typing import Dict, List, Any, Optional

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from trl import DataCollatorForCompletionOnlyLM

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

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def prepare_dataset(data_path: str, tokenizer, data_config: Dict[str, Any]):
    """Prepare dataset for training."""

    # Load dataset from JSON file
    logger.info(f"Loading dataset from {data_path}")
    dataset = load_dataset("json", data_files={"train": data_path})["train"]

    # Display dataset information
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Dataset features: {dataset.features}")
    logger.info(f"Dataset examples: {dataset[:2]}")

    response_template = ""

    # Function to format the dataset examples
    def format_example(example):
        instruction = example["instruction"]
        completion = example["completion"]

        # Format: instruction followed by completion
        text = f"<|user|>\n{instruction}\n<|assistant|>\n{completion}"
        return {"text": text}

    # Apply formatting to the dataset
    formatted_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
    )

    # Tokenize the dataset
    max_seq_length = data_config.get("max_seq_length", 2048)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=max_seq_length, padding=False)

    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_config.get("preprocessing_num_workers", 4),
        remove_columns=["text"],
    )

    logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", False),
    )

    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
    tokenized_dataset = prepare_dataset(data_path, tokenizer, data_config)

    # Setup the data collator for completion-only learning
    response_template = "<|assistant|>"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[1:]  # Remove the BOS token

    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

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
        evaluation_strategy=training_config.get("evaluation_strategy", "no"),
        eval_steps=training_config.get("eval_steps", None),
        optim=training_config["optim"],
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
    )

    # Initialize the Trainer
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    logger.info("Saving the final model...")
    trainer.save_model(os.path.join(training_config["output_dir"], "final"))
    tokenizer.save_pretrained(os.path.join(training_config["output_dir"], "final"))

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
python "$FINETUNE_SCRIPT" --config "$CONFIG_FILE"

# Check if fine-tuning was successful
if [ $? -eq 0 ]; then
  echo "Fine-tuning completed successfully!"
  echo "Fine-tuned model saved to: $OUTPUT_DIR/final"
else
  echo "Error: Fine-tuning failed"
  exit 1
fi