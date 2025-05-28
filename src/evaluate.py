#!/usr/bin/env python3
"""
MCP Model Evaluation Script

This script evaluates a fine-tuned model's performance on MCP tasks and
compares it against a baseline model if specified.
"""

import os
import sys
import json
import argparse
import logging
import re
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Hugging Face imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextGenerationPipeline
)
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model performance on MCP tasks")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the model to evaluate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the evaluation dataset JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        help="Optional baseline model directory for comparison"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (lower for more deterministic outputs)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    return parser.parse_args()

def load_model_and_tokenizer(model_dir, load_in_8bit=False, load_in_4bit=True):
    """Load model and tokenizer."""
    logger.info(f"Loading model from {model_dir}")

    # Setup quantization configuration
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    # Load base model
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    # Check if this is a PEFT/LoRA adapter model
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        # This is a PEFT model, we need to load the base model first
        # Try to find base model info in the adapter config
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        base_model_name_or_path = adapter_config.get("base_model_name_or_path", None)

        if not base_model_name_or_path:
            raise ValueError("Base model path not found in adapter_config.json")

        logger.info(f"Loading base model from {base_model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            **model_kwargs
        )

        logger.info(f"Loading adapter from {model_dir}")
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        # Regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            **model_kwargs
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # Make sure padding token is correctly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def load_evaluation_dataset(dataset_path):
    """Load the evaluation dataset."""
    logger.info(f"Loading evaluation dataset from {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    examples = data.get("examples", [])
    logger.info(f"Loaded {len(examples)} evaluation examples")

    return examples

def format_prompt(instruction):
    """Format instruction as a prompt for the model."""
    return f"Instruction: {instruction}\nResponse:"

def extract_mcp_tool_usage(response):
    """
    Extract MCP tool usage from model response.
    Returns a dict with server_name, tool_name, arguments.
    """
    patterns = {
        'use_mcp_tool': re.compile(r'<use_mcp_tool>(.*?)</use_mcp_tool>', re.DOTALL),
        'server_name': re.compile(r'<server_name>(.*?)</server_name>', re.DOTALL),
        'tool_name': re.compile(r'<tool_name>(.*?)</tool_name>', re.DOTALL),
        'arguments': re.compile(r'<arguments>(.*?)</arguments>', re.DOTALL)
    }

    # Extract the entire tool call
    tool_match = patterns['use_mcp_tool'].search(response)
    if not tool_match:
        return None

    tool_content = tool_match.group(1)

    # Extract server name
    server_match = patterns['server_name'].search(response)
    server_name = server_match.group(1).strip() if server_match else None

    # Extract tool name
    tool_match = patterns['tool_name'].search(response)
    tool_name = tool_match.group(1).strip() if tool_match else None

    # Extract arguments
    args_match = patterns['arguments'].search(response)
    arguments = args_match.group(1).strip() if args_match else None

    # Try to parse arguments as JSON
    if arguments:
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            # Not valid JSON, keep as string
            pass

    return {
        'server_name': server_name,
        'tool_name': tool_name,
        'arguments': arguments
    }

def validate_xml_format(response):
    """
    Validate that the response has properly formatted XML tags.
    Returns tuple of (is_valid, error_message)
    """
    # Check for balanced opening and closing tags
    tag_stack = []
    opening_pattern = re.compile(r'<(\w+)>')
    closing_pattern = re.compile(r'</(\w+)>')

    # Find all opening and closing tags
    opening_tags = [(match.group(1), match.start()) for match in opening_pattern.finditer(response)]
    closing_tags = [(match.group(1), match.start()) for match in closing_pattern.finditer(response)]

    # Check if there are any tags
    if not opening_tags and not closing_tags:
        return False, "No XML tags found"

    # Combine and sort by position
    all_tags = opening_tags + [(f"/{name}", pos) for name, pos in closing_tags]
    all_tags.sort(key=lambda x: x[1])

    # Process tags
    for tag, _ in all_tags:
        if not tag.startswith('/'):
            tag_stack.append(tag)
        else:
            tag_name = tag[1:]  # Remove the '/'
            if not tag_stack or tag_stack[-1] != tag_name:
                return False, f"Unmatched closing tag: {tag_name}"
            tag_stack.pop()

    # Check if all tags were closed
    if tag_stack:
        return False, f"Unclosed tags: {', '.join(tag_stack)}"

    # Check for required MCP tags
    required_tags = ['use_mcp_tool', 'server_name', 'tool_name', 'arguments']
    for tag in required_tags:
        opening = f"<{tag}>"
        closing = f"</{tag}>"
        if opening not in response or closing not in response:
            return False, f"Missing required tag: {tag}"

    return True, ""

def evaluate_response(response, expected, criteria):
    """
    Evaluate a model response against expected output.
    Returns a dict with evaluation metrics.
    """
    # Initialize metrics
    metrics = {
        'xml_validity': False,
        'tool_selection_accuracy': 0.0,
        'parameter_accuracy': 0.0,
        'has_mcp_tags': False,
        'response_time_ms': criteria.get('response_time_ms', 0)
    }

    # Check for XML validity
    is_valid, error = validate_xml_format(response)
    metrics['xml_validity'] = is_valid

    if not is_valid:
        metrics['error_message'] = error
        return metrics

    # Extract MCP tool usage
    mcp_tool = extract_mcp_tool_usage(response)
    metrics['has_mcp_tags'] = mcp_tool is not None

    if not mcp_tool:
        metrics['error_message'] = "No MCP tool usage found"
        return metrics

    # Check tool selection accuracy
    expected_server = expected.get('server_name')
    expected_tool = expected.get('tool_name')

    if expected_server and expected_tool:
        server_correct = mcp_tool['server_name'] == expected_server
        tool_correct = mcp_tool['tool_name'] == expected_tool

        metrics['server_name_match'] = server_correct
        metrics['tool_name_match'] = tool_correct
        metrics['tool_selection_accuracy'] = 1.0 if (server_correct and tool_correct) else 0.0

    # Check parameter accuracy
    expected_params = expected.get('arguments', {})
    actual_params = mcp_tool.get('arguments', {})

    if expected_params and isinstance(expected_params, dict) and isinstance(actual_params, dict):
        # Count correct parameters
        correct_params = 0
        total_params = len(expected_params)

        for key, value in expected_params.items():
            if key in actual_params and actual_params[key] == value:
                correct_params += 1

        metrics['parameter_accuracy'] = correct_params / total_params if total_params > 0 else 0.0
        metrics['correct_parameters'] = correct_params
        metrics['total_parameters'] = total_params

    return metrics

def generate_response(model, tokenizer, prompt, max_length=2048, temperature=0.1, top_p=0.9):
    """Generate a response from the model for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Measure response time
    start_time = time.time()

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id
        )

    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part (after prompt)
    response = response[len(prompt):].strip()

    return response, response_time_ms

def evaluate_model(model, tokenizer, examples, args):
    """Evaluate model on all examples in the dataset."""
    results = []

    for i, example in enumerate(examples):
        logger.info(f"Evaluating example {i+1}/{len(examples)}")

        instruction = example["instruction"]
        expected_completion = example["completion"]
        metadata = example.get("metadata", {})

        # Format prompt
        prompt = format_prompt(instruction)

        # Generate response
        response, response_time_ms = generate_response(
            model,
            tokenizer,
            prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )

        # Extract expected MCP tool usage
        expected_tool = extract_mcp_tool_usage(expected_completion)

        # Evaluate response
        criteria = {
            'response_time_ms': response_time_ms
        }
        metrics = evaluate_response(response, expected_tool, criteria)

        # Store results
        result = {
            'example_id': i,
            'instruction': instruction,
            'model_response': response,
            'expected_completion': expected_completion,
            'metrics': metrics,
            'metadata': metadata
        }
        results.append(result)

    return results

def compute_aggregate_metrics(results):
    """Compute aggregate metrics across all examples."""
    metrics = {
        'total_examples': len(results),
        'xml_validity_rate': 0.0,
        'tool_selection_accuracy': 0.0,
        'parameter_accuracy': 0.0,
        'mcp_tags_rate': 0.0,
        'avg_response_time_ms': 0.0,
        'complexity': {
            'simple': {'count': 0, 'correct': 0},
            'multi_step': {'count': 0, 'correct': 0},
            'complex': {'count': 0, 'correct': 0}
        },
        'server_types': {}
    }

    # Initialize counters
    valid_xml_count = 0
    correct_tool_count = 0
    has_mcp_tags_count = 0
    parameter_accuracy_sum = 0.0
    parameter_accuracy_count = 0
    response_time_sum = 0.0

    for result in results:
        # XML validity
        if result['metrics']['xml_validity']:
            valid_xml_count += 1

        # MCP tags presence
        if result['metrics']['has_mcp_tags']:
            has_mcp_tags_count += 1

        # Tool selection
        if result['metrics']['tool_selection_accuracy'] > 0:
            correct_tool_count += 1

        # Parameter accuracy
        if 'parameter_accuracy' in result['metrics'] and result['metrics']['parameter_accuracy'] > 0:
            parameter_accuracy_sum += result['metrics']['parameter_accuracy']
            parameter_accuracy_count += 1

        # Response time
        response_time_sum += result['metrics'].get('response_time_ms', 0)

        # Complexity stats
        complexity = result['metadata'].get('complexity', 'simple')
        if complexity not in metrics['complexity']:
            metrics['complexity'][complexity] = {'count': 0, 'correct': 0}

        metrics['complexity'][complexity]['count'] += 1
        if result['metrics']['tool_selection_accuracy'] > 0:
            metrics['complexity'][complexity]['correct'] += 1

        # Server type stats
        server_type = result['metadata'].get('server', 'unknown')
        if server_type not in metrics['server_types']:
            metrics['server_types'][server_type] = {'count': 0, 'correct': 0}

        metrics['server_types'][server_type]['count'] += 1
        if result['metrics']['tool_selection_accuracy'] > 0:
            metrics['server_types'][server_type]['correct'] += 1

    # Calculate rates
    total = metrics['total_examples']
    metrics['xml_validity_rate'] = valid_xml_count / total if total > 0 else 0
    metrics['tool_selection_accuracy'] = correct_tool_count / total if total > 0 else 0
    metrics['mcp_tags_rate'] = has_mcp_tags_count / total if total > 0 else 0
    metrics['parameter_accuracy'] = parameter_accuracy_sum / parameter_accuracy_count if parameter_accuracy_count > 0 else 0
    metrics['avg_response_time_ms'] = response_time_sum / total if total > 0 else 0

    # Calculate success rates for complexity and server types
    for complexity, data in metrics['complexity'].items():
        data['success_rate'] = data['correct'] / data['count'] if data['count'] > 0 else 0

    for server, data in metrics['server_types'].items():
        data['success_rate'] = data['correct'] / data['count'] if data['count'] > 0 else 0

    return metrics

def save_evaluation_results(results, aggregate_metrics, output_dir, model_name="model"):
    """Save evaluation results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Add timestamp to filenames
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save detailed results
    results_path = os.path.join(output_dir, f"{model_name}_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save aggregate metrics
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(aggregate_metrics, f, indent=2)

    logger.info(f"Saved evaluation results to {results_path}")
    logger.info(f"Saved aggregate metrics to {metrics_path}")

    return results_path, metrics_path

def compare_models(model_metrics, baseline_metrics=None):
    """Compare model metrics against a baseline."""
    if not baseline_metrics:
        return {"comparison": "No baseline provided for comparison"}

    comparison = {
        "xml_validity_delta": model_metrics["xml_validity_rate"] - baseline_metrics["xml_validity_rate"],
        "tool_selection_delta": model_metrics["tool_selection_accuracy"] - baseline_metrics["tool_selection_accuracy"],
        "parameter_accuracy_delta": model_metrics["parameter_accuracy"] - baseline_metrics["parameter_accuracy"],
        "mcp_tags_delta": model_metrics["mcp_tags_rate"] - baseline_metrics["mcp_tags_rate"],
        "response_time_delta": baseline_metrics["avg_response_time_ms"] - model_metrics["avg_response_time_ms"],

        "complexity_deltas": {},
        "server_type_deltas": {}
    }

    # Compare complexity metrics
    for complexity in model_metrics["complexity"]:
        if complexity in baseline_metrics["complexity"]:
            model_rate = model_metrics["complexity"][complexity].get("success_rate", 0)
            baseline_rate = baseline_metrics["complexity"][complexity].get("success_rate", 0)
            comparison["complexity_deltas"][complexity] = model_rate - baseline_rate

    # Compare server type metrics
    for server in model_metrics["server_types"]:
        if server in baseline_metrics["server_types"]:
            model_rate = model_metrics["server_types"][server].get("success_rate", 0)
            baseline_rate = baseline_metrics["server_types"][server].get("success_rate", 0)
            comparison["server_type_deltas"][server] = model_rate - baseline_rate

    # Determine overall improvement
    key_metrics = ["xml_validity_delta", "tool_selection_delta", "parameter_accuracy_delta", "mcp_tags_delta"]
    avg_improvement = sum(comparison[metric] for metric in key_metrics) / len(key_metrics)
    comparison["average_improvement"] = avg_improvement

    return comparison

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load evaluation dataset
    examples = load_evaluation_dataset(args.dataset)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    # Evaluate model
    logger.info("Evaluating fine-tuned model")
    model_results = evaluate_model(model, tokenizer, examples, args)
    model_metrics = compute_aggregate_metrics(model_results)

    # Save results
    model_name = os.path.basename(os.path.normpath(args.model_dir))
    model_results_path, model_metrics_path = save_evaluation_results(
        model_results, model_metrics, args.output_dir, model_name
    )

    # Evaluate baseline model if specified
    baseline_metrics = None
    if args.baseline_model:
        logger.info(f"Evaluating baseline model from {args.baseline_model}")
        baseline_model, baseline_tokenizer = load_model_and_tokenizer(args.baseline_model)
        baseline_results = evaluate_model(baseline_model, baseline_tokenizer, examples, args)
        baseline_metrics = compute_aggregate_metrics(baseline_results)

        baseline_name = os.path.basename(os.path.normpath(args.baseline_model))
        save_evaluation_results(baseline_results, baseline_metrics, args.output_dir, baseline_name)

    # Compare models
    if baseline_metrics:
        logger.info("Comparing models")
        comparison = compare_models(model_metrics, baseline_metrics)

        comparison_path = os.path.join(args.output_dir, f"model_comparison_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Saved model comparison to {comparison_path}")

        # Log key improvement metrics
        logger.info(f"Average improvement over baseline: {comparison['average_improvement']:.2%}")
        logger.info(f"XML validity rate delta: {comparison['xml_validity_delta']:.2%}")
        logger.info(f"Tool selection accuracy delta: {comparison['tool_selection_delta']:.2%}")

    # Log summary metrics
    logger.info(f"Evaluation complete for {len(examples)} examples")
    logger.info(f"XML validity rate: {model_metrics['xml_validity_rate']:.2%}")
    logger.info(f"Tool selection accuracy: {model_metrics['tool_selection_accuracy']:.2%}")
    logger.info(f"Parameter accuracy: {model_metrics['parameter_accuracy']:.2%}")

    return 0

if __name__ == "__main__":
    sys.exit(main())