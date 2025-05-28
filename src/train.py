#!/usr/bin/env python3
"""
MCP Fine-tuning Script using PEFT/QLoRA

This script fine-tunes a pretrained model on MCP task data using
Parameter-Efficient Fine-Tuning (PEFT) with QLoRA.
"""

import os
import sys
import argparse
import json
import torch
import logging
from pathlib import Path
from datetime import datetime

# Hugging Face imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a model on MCP tasks")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the base model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/mcp-finetuned",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Peak learning rate"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA adapter rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )
    return parser.parse_args()

def init_model_and_tokenizer(model_dir, args):
    """Initialize the model and tokenizer with QLoRA configuration."""
    logger.info(f"Loading base model from {model_dir}")

    # Setup quantization configuration
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # Make sure padding token is correctly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

class MCPDataset(Dataset):
    """MCP dataset with completion-only loss masking."""

    def __init__(self, tokenized_inputs, tokenizer):
        self.input_ids = tokenized_inputs["input_ids"]
        self.attention_mask = tokenized_inputs["attention_mask"]
        self.labels = tokenized_inputs["labels"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def prepare_dataset(dataset_path, tokenizer, max_length):
    """Prepare and tokenize the dataset with loss masking."""
    logger.info(f"Loading dataset from {dataset_path}")

    # Load dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)

    examples = data.get("examples", [])
    logger.info(f"Dataset loaded with {len(examples)} examples")

    instructions = [ex["instruction"] for ex in examples]
    completions = [ex["completion"] for ex in examples]

    # Format as instruction-completion pairs
    texts = [f"Instruction: {instruction}\nResponse: {completion}"
             for instruction, completion in zip(instructions, completions)]

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    # Create labels for completion-only training
    labels = tokenized["input_ids"].clone()

    # Find the position of "Response:" in each example and mask instruction tokens
    for i, text in enumerate(texts):
        response_pos = text.find("Response:")
        if response_pos != -1:
            # Convert character position to token position (approximate)
            response_tokens = tokenizer.encode("Response:")
            response_token_pos = len(tokenizer.encode(text[:response_pos])) + len(response_tokens) - 1

            # Mask instruction tokens from loss (-100 is the ignore index)
            if response_token_pos < len(labels[i]):
                labels[i][:response_token_pos] = -100

    tokenized["labels"] = labels

    # Create dataset
    dataset = MCPDataset(tokenized, tokenizer)
    return dataset

def get_training_args(args):
    """Create training arguments."""
    output_dir = args.output_dir

    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_dir, f"run_{timestamp}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="no",  # No evaluation during training
        save_total_limit=3,  # Keep only the last 3 checkpoints
        load_best_model_at_end=False,
        report_to="tensorboard",
        # fp16 training for faster training if GPU supports it
        fp16=True,
        # Use gradient checkpointing to save memory
        gradient_checkpointing=True,
        # Don't remove unused columns automatically
        remove_unused_columns=False,
    )

    return training_args

def train_model(model, tokenizer, dataset, args):
    """Train the model."""
    logger.info("Setting up training arguments")
    training_args = get_training_args(args)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train model
    logger.info("Starting training")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    return training_args.output_dir

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Initialize model and tokenizer
    model, tokenizer = init_model_and_tokenizer(args.model_dir, args)

    # Prepare dataset
    dataset = prepare_dataset(args.dataset, tokenizer, args.max_length)

    # Train model
    output_dir = train_model(model, tokenizer, dataset, args)

    logger.info(f"Training complete. Model saved at {output_dir}")

    return 0

if __name__ == "__main__":
    sys.exit(main())