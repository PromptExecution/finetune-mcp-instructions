#!/usr/bin/env python3
"""
Script to evaluate and compare models on MCP tasks.
"""

import os
import json
import time
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel

# Setup logging (initial basic config for stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_file_logger(output_dir: str):
    """Adds a file handler to the global logger."""
    log_file_path = os.path.join(output_dir, "evaluation.log")
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to the root logger or a specific logger
    # If using logger = logging.getLogger(__name__), add to this specific logger:
    logger.addHandler(file_handler)
    logger.info(f"Additionally logging to file: {log_file_path}")


def load_model_and_tokenizer(model_path: str, is_adapter_model: bool = False, base_model_for_adapter: Optional[str] = None):
    """Load model and tokenizer from the specified path.
    If is_adapter_model is True, model_path is the adapter path, and base_model_for_adapter is the original base model.
    """
    logger.info(f"Attempting to load model. Main path: {model_path}")
    if is_adapter_model:
        if not base_model_for_adapter:
            raise ValueError("If is_adapter_model is True, base_model_for_adapter must be provided.")
        logger.info(f"Identified as PEFT adapter. Base model for adapter: {base_model_for_adapter}")

    # Common Quantization Config
    bnb_config = None
    quantize = True # Attempt quantization by default
    try:
        # BitsAndBytesConfig already imported
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("BitsAndBytesConfig created for 4-bit quantization.")
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to create BitsAndBytesConfig: {str(e)}. Will load model(s) without quantization.")
        quantize = False # Disable quantization if config fails

    if is_adapter_model:
        # Load base model first
        logger.info(f"Loading base model ({base_model_for_adapter}) for PEFT adapter...")
        base_loaded_model = AutoModelForCausalLM.from_pretrained(
            base_model_for_adapter,
            quantization_config=bnb_config if quantize else None,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info(f"Base model {base_model_for_adapter} loaded. Now loading PEFT adapter from {model_path}...")

        # Resolve to absolute path for adapter loading
        absolute_adapter_path = str(Path(model_path).resolve())
        logger.info(f"Resolved adapter path to absolute: {absolute_adapter_path}")

        # Load the PEFT adapter onto the base model
        model = PeftModel.from_pretrained(base_loaded_model, absolute_adapter_path, local_files_only=True)
        logger.info(f"PEFT adapter from {absolute_adapter_path} loaded successfully.")
    else:
        # Standard full model loading
        logger.info(f"Loading full model from {model_path}...")
        try:
            if quantize:
                logger.info("Attempting to load with 4-bit quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                logger.info("Model loaded with 4-bit quantization.")
            else: # Fallback if quantization was disabled from the start
                # This case implies bnb_config failed to initialize, so quantize is False
                logger.info("Quantization was disabled. Loading model without quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True,
                )
                logger.info("Model loaded without quantization.")

        except Exception as e:
            logger.warning(f"Initial attempt to load {model_path} (with/without quantization) failed: {str(e)}")
            logger.info("Attempting to load model without quantization as a final fallback...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Model loaded without quantization (fallback).")

    # Determine tokenizer load path
    if is_adapter_model:
        # For PEFT models, use the tokenizer from the original base model.
        tokenizer_load_path = base_model_for_adapter
        logger.info(f"Adapter model detected. Tokenizer will be loaded from original base model: {tokenizer_load_path}")
    else:
        # For base model, model_path is the HF repo ID or full model path
        tokenizer_load_path = model_path
        logger.info(f"Base model detected. Tokenizer will be loaded from: {tokenizer_load_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set tokenizer.pad_token to tokenizer.eos_token.")

    return model, tokenizer

def load_evaluation_data(data_path: str) -> List[Dict[str, Any]]:
    """Load evaluation data from JSON file."""
    logger.info(f"Loading evaluation data from: {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)

    eval_samples = data.get("eval_samples", [])
    logger.info(f"Loaded {len(eval_samples)} evaluation samples")

    return eval_samples

def generate_response(model, tokenizer, instruction: str, max_tokens: int = 1024) -> Tuple[str, float]:
    """Generate a response from the model for a given instruction."""
    # Format prompt for MCP-style responses
    formatted_prompt = f"<|user|>\n{instruction}\n<|assistant|>"

    # Create generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=0.1,  # Low temperature for more deterministic results
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )

    # Generate response
    start_time = time.time()
    result = generator(formatted_prompt, return_full_text=False)[0]["generated_text"]
    end_time = time.time()

    return result, end_time - start_time

def evaluate_mcp_response(response: str, reference: str) -> Tuple[float, Dict[str, Any]]:
    """Evaluate the model response against the reference for MCP tasks."""
    results = {}

    # Extract MCP tool calls from response and reference
    response_tools = re.findall(r'<use_mcp_tool>.*?</use_mcp_tool>', response, re.DOTALL)
    reference_tools = re.findall(r'<use_mcp_tool>.*?</use_mcp_tool>', reference, re.DOTALL)

    results["mcp_tool_count"] = len(response_tools)
    results["reference_tool_count"] = len(reference_tools)

    # Check if any MCP tools were found
    if len(response_tools) == 0:
        results["found_mcp_format"] = False
        results["tool_name_accuracy"] = 0.0
        results["server_name_accuracy"] = 0.0
        results["argument_structure_score"] = 0.0
        results["overall_score"] = 0.0
        return 0.0, results

    results["found_mcp_format"] = True

    # Extract server names
    response_servers = re.findall(r'<server_name>(.*?)</server_name>', response, re.DOTALL)
    reference_servers = re.findall(r'<server_name>(.*?)</server_name>', reference, re.DOTALL)

    # Extract tool names
    response_tool_names = re.findall(r'<tool_name>(.*?)</tool_name>', response, re.DOTALL)
    reference_tool_names = re.findall(r'<tool_name>(.*?)</tool_name>', reference, re.DOTALL)

    # Server name accuracy
    server_matches = sum(1 for rs in response_servers for ref_s in reference_servers if rs.strip() == ref_s.strip())
    server_accuracy = server_matches / len(reference_servers) if reference_servers else 0.0
    results["server_name_accuracy"] = server_accuracy

    # Tool name accuracy
    tool_matches = sum(1 for rt in response_tool_names for ref_t in reference_tool_names if rt.strip() == ref_t.strip())
    tool_accuracy = tool_matches / len(reference_tool_names) if reference_tool_names else 0.0
    results["tool_name_accuracy"] = tool_accuracy

    # Check if arguments are structured correctly (look for JSON-like structure)
    response_args_sections = re.findall(r'<arguments>(.*?)</arguments>', response, re.DOTALL)
    args_structure_score = 0.0

    for args_section in response_args_sections:
        # Check if it contains valid JSON structure with braces
        if re.search(r'\s*\{\s*.*\s*\}\s*', args_section, re.DOTALL):
            args_structure_score += 1.0

            # Try to parse as JSON to check validity
            try:
                # Clean up any potential issues (strip whitespace)
                cleaned = args_section.strip()
                json.loads(cleaned)
                args_structure_score += 0.5  # Bonus for valid JSON
            except json.JSONDecodeError:
                pass

    # Normalize score based on number of argument sections
    args_structure_score = args_structure_score / (1.5 * len(response_args_sections)) if response_args_sections else 0.0
    results["argument_structure_score"] = args_structure_score

    # Calculate overall score
    overall_score = (server_accuracy * 0.3) + (tool_accuracy * 0.4) + (args_structure_score * 0.3)
    results["overall_score"] = overall_score

    return overall_score, results

def run_evaluation(base_model_path: str, finetuned_model_path: str, eval_data_path: str, output_dir: str):
    """Run the evaluation comparing base model vs fine-tuned model."""
    # Load evaluation data
    eval_samples = load_evaluation_data(eval_data_path)

    # Load base model
    logger.info(f"--- Loading Base Model: {base_model_path} ---")
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_path, is_adapter_model=False)

    # Load fine-tuned model (PEFT adapter)
    logger.info(f"--- Loading Fine-tuned Model (Adapter): {finetuned_model_path} ---")
    # finetuned_model_path is the adapter directory.
    # base_model_path is the original base model name/path.
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(
        model_path=finetuned_model_path,
        is_adapter_model=True,
        base_model_for_adapter=base_model_path
    )

    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")

    # Initialize results structure
    results = {
        "base_model": base_model_path,
        "finetuned_model": finetuned_model_path,
        "eval_data": eval_data_path,
        "timestamp": timestamp,
        "samples": [],
        "summary": {
            "base_model": {
                "avg_score": 0.0,
                "mcp_format_adherence": 0.0,
                "server_name_accuracy": 0.0,
                "tool_name_accuracy": 0.0,
                "argument_structure_score": 0.0,
                "avg_generation_time": 0.0,
            },
            "finetuned_model": {
                "avg_score": 0.0,
                "mcp_format_adherence": 0.0,
                "server_name_accuracy": 0.0,
                "tool_name_accuracy": 0.0,
                "argument_structure_score": 0.0,
                "avg_generation_time": 0.0,
            },
            "improvement": {
                "avg_score": 0.0,
                "mcp_format_adherence": 0.0,
                "server_name_accuracy": 0.0,
                "tool_name_accuracy": 0.0,
                "argument_structure_score": 0.0,
                "avg_generation_time": 0.0,
            }
        }
    }

    logger.info("Starting evaluation...")

    # Run evaluation for each sample
    for i, sample in enumerate(tqdm(eval_samples, desc="Evaluating samples")):
        instruction = sample["instruction"]
        reference = sample["reference"]

        logger.info(f"Evaluating sample {i+1}/{len(eval_samples)}")

        # Generate responses from both models
        logger.info("Generating response from base model...")
        base_response, base_time = generate_response(base_model, base_tokenizer, instruction)

        logger.info("Generating response from fine-tuned model...")
        finetuned_response, finetuned_time = generate_response(finetuned_model, finetuned_tokenizer, instruction)

        # Evaluate responses
        base_score, base_details = evaluate_mcp_response(base_response, reference)
        finetuned_score, finetuned_details = evaluate_mcp_response(finetuned_response, reference)

        # Record results for this sample
        sample_result = {
            "instruction": instruction,
            "reference": reference,
            "base_model_response": base_response,
            "finetuned_model_response": finetuned_response,
            "base_model_time": base_time,
            "finetuned_model_time": finetuned_time,
            "base_model_score": base_score,
            "finetuned_model_score": finetuned_score,
            "base_model_details": base_details,
            "finetuned_model_details": finetuned_details,
            "score_improvement": finetuned_score - base_score,
            "time_improvement": base_time - finetuned_time,
        }

        results["samples"].append(sample_result)

        # Log partial results
        logger.info(f"Sample {i+1} scores: Base: {base_score:.3f}, Finetuned: {finetuned_score:.3f}, "
                   f"Improvement: {finetuned_score - base_score:.3f}")

    # Calculate summary statistics
    base_scores = [s["base_model_score"] for s in results["samples"]]
    finetuned_scores = [s["finetuned_model_score"] for s in results["samples"]]
    base_times = [s["base_model_time"] for s in results["samples"]]
    finetuned_times = [s["finetuned_model_time"] for s in results["samples"]]

    # Base model stats
    results["summary"]["base_model"]["avg_score"] = sum(base_scores) / len(base_scores) if base_scores else 0.0
    results["summary"]["base_model"]["avg_generation_time"] = sum(base_times) / len(base_times) if base_times else 0.0
    results["summary"]["base_model"]["mcp_format_adherence"] = sum(1 for s in results["samples"]
                                                             if s["base_model_details"]["found_mcp_format"]) / len(results["samples"]) if results["samples"] else 0.0
    results["summary"]["base_model"]["server_name_accuracy"] = sum(s["base_model_details"]["server_name_accuracy"]
                                                             for s in results["samples"]) / len(results["samples"]) if results["samples"] else 0.0
    results["summary"]["base_model"]["tool_name_accuracy"] = sum(s["base_model_details"]["tool_name_accuracy"]
                                                          for s in results["samples"]) / len(results["samples"]) if results["samples"] else 0.0
    results["summary"]["base_model"]["argument_structure_score"] = sum(s["base_model_details"]["argument_structure_score"]
                                                                for s in results["samples"]) / len(results["samples"]) if results["samples"] else 0.0

    # Fine-tuned model stats
    results["summary"]["finetuned_model"]["avg_score"] = sum(finetuned_scores) / len(finetuned_scores) if finetuned_scores else 0.0
    results["summary"]["finetuned_model"]["avg_generation_time"] = sum(finetuned_times) / len(finetuned_times) if finetuned_times else 0.0
    results["summary"]["finetuned_model"]["mcp_format_adherence"] = sum(1 for s in results["samples"]
                                                                 if s["finetuned_model_details"]["found_mcp_format"]) / len(results["samples"]) if results["samples"] else 0.0
    results["summary"]["finetuned_model"]["server_name_accuracy"] = sum(s["finetuned_model_details"]["server_name_accuracy"]
                                                                 for s in results["samples"]) / len(results["samples"]) if results["samples"] else 0.0
    results["summary"]["finetuned_model"]["tool_name_accuracy"] = sum(s["finetuned_model_details"]["tool_name_accuracy"]
                                                              for s in results["samples"]) / len(results["samples"]) if results["samples"] else 0.0
    results["summary"]["finetuned_model"]["argument_structure_score"] = sum(s["finetuned_model_details"]["argument_structure_score"]
                                                                    for s in results["samples"]) / len(results["samples"]) if results["samples"] else 0.0

    # Improvement stats
    results["summary"]["improvement"]["avg_score"] = (
        results["summary"]["finetuned_model"]["avg_score"] - results["summary"]["base_model"]["avg_score"]
    )
    results["summary"]["improvement"]["mcp_format_adherence"] = (
        results["summary"]["finetuned_model"]["mcp_format_adherence"] - results["summary"]["base_model"]["mcp_format_adherence"]
    )
    results["summary"]["improvement"]["server_name_accuracy"] = (
        results["summary"]["finetuned_model"]["server_name_accuracy"] - results["summary"]["base_model"]["server_name_accuracy"]
    )
    results["summary"]["improvement"]["tool_name_accuracy"] = (
        results["summary"]["finetuned_model"]["tool_name_accuracy"] - results["summary"]["base_model"]["tool_name_accuracy"]
    )
    results["summary"]["improvement"]["argument_structure_score"] = (
        results["summary"]["finetuned_model"]["argument_structure_score"] - results["summary"]["base_model"]["argument_structure_score"]
    )
    results["summary"]["improvement"]["avg_generation_time"] = (
        results["summary"]["base_model"]["avg_generation_time"] - results["summary"]["finetuned_model"]["avg_generation_time"] # Higher is better for fine-tuned
    )

    # Save results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    # Generate visualizations
    generate_visualizations(results, output_dir, timestamp)

    return results, results_file

def generate_visualizations(results, output_dir, timestamp):
    """Generate visualization of the evaluation results."""
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Extract scores
    base_scores = [s["base_model_score"] for s in results["samples"]]
    finetuned_scores = [s["finetuned_model_score"] for s in results["samples"]]

    # Plot 1: Score comparison bar chart
    plt.figure(figsize=(10, 6))
    metrics = ["avg_score", "mcp_format_adherence", "server_name_accuracy",
              "tool_name_accuracy", "argument_structure_score"]
    metric_labels = ["Overall Score", "MCP Format Usage", "Server Name Accuracy",
                    "Tool Name Accuracy", "Argument Structure"]

    x = np.arange(len(metrics))
    width = 0.35

    base_values = [results["summary"]["base_model"][m] for m in metrics]
    finetuned_values = [results["summary"]["finetuned_model"][m] for m in metrics]

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, base_values, width, label='Base Model')
    rects2 = ax.bar(x + width/2, finetuned_values, width, label='Fine-tuned Model')

    ax.set_ylabel('Score')
    ax.set_title('Comparison of Model Performance on MCP Tasks')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"score_comparison_{timestamp}.png"))
    plt.close(fig) # Close the figure to free memory

    # Plot 2: Sample-by-sample score comparison
    fig, ax = plt.subplots(figsize=(12, 7)) # Create new figure and axes
    x_samples = np.arange(len(results["samples"]))
    ax.plot(x_samples, base_scores, 'b-', label='Base Model')
    ax.plot(x_samples, finetuned_scores, 'r-', label='Fine-tuned Model')
    ax.set_xlabel('Evaluation Sample Index')
    ax.set_ylabel('Score')
    ax.set_title('Score Comparison by Sample')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"sample_scores_{timestamp}.png"))
    plt.close(fig) # Close the figure

    # Plot 3: Overall improvement
    fig, ax = plt.subplots(figsize=(8, 6)) # Create new figure and axes
    improvement_metrics = ["avg_score", "mcp_format_adherence", "server_name_accuracy",
                          "tool_name_accuracy", "argument_structure_score"]
    improvement_values = [results["summary"]["improvement"][m] for m in improvement_metrics]

    colors = ['green' if v >= 0 else 'red' for v in improvement_values] # Changed to >= 0 for green

    ax.bar(metric_labels, improvement_values, color=colors)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title('Improvement with Fine-tuning')
    ax.set_ylabel('Improvement Score (Fine-tuned - Base)')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"improvement_{timestamp}.png"))
    plt.close(fig) # Close the figure

    logger.info(f"Visualizations saved to: {plots_dir}")

def main():
    """Run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate and compare models on MCP tasks")
    parser.add_argument("--base-model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--finetuned-model", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--eval-data", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save results")

    args = parser.parse_args()

    # Setup file logger now that we have output_dir
    setup_file_logger(args.output_dir)

    # Run evaluation
    results, results_file = run_evaluation(
        args.base_model,
        args.finetuned_model,
        args.eval_data,
        args.output_dir
    )

    # Print summary results
    print("\n===== EVALUATION RESULTS =====")
    print(f"Base Model: {args.base_model}")
    print(f"Fine-tuned Model: {args.finetuned_model}")
    print("\nScores:")
    print(f"  Base Model: {results['summary']['base_model']['avg_score']:.3f}")
    print(f"  Fine-tuned Model: {results['summary']['finetuned_model']['avg_score']:.3f}")

    improvement_percentage = 0.0
    if results['summary']['base_model']['avg_score'] != 0: # Avoid division by zero
        improvement_percentage = (results['summary']['improvement']['avg_score'] / abs(results['summary']['base_model']['avg_score'])) * 100
    elif results['summary']['improvement']['avg_score'] > 0: # Base is 0, finetuned is positive
        improvement_percentage = float('inf') # Infinite improvement

    print(f"  Improvement: {results['summary']['improvement']['avg_score']:.3f} " +
          f"({improvement_percentage:.1f}%)")

    print("\nMCP Format Adherence:")
    print(f"  Base Model: {results['summary']['base_model']['mcp_format_adherence']:.3f}")
    print(f"  Fine-tuned Model: {results['summary']['finetuned_model']['mcp_format_adherence']:.3f}")

    print("\nDetailed results saved to:")
    print(f"  {results_file}")
    print(f"  {os.path.join(args.output_dir, 'plots')}/")

if __name__ == "__main__":
    main()