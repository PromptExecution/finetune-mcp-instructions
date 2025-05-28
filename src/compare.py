#!/usr/bin/env python3
"""
MCP Model Comparison Script

This script directly compares a fine-tuned model with a baseline model on MCP tasks
and generates comparative performance metrics and visualizations.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import evaluation utilities
from evaluate import (
    load_evaluation_dataset,
    load_model_and_tokenizer,
    evaluate_model,
    compute_aggregate_metrics,
    compare_models
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("comparison.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare model performance on MCP tasks")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Directory containing the base model"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        required=True,
        help="Directory containing the fine-tuned model"
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
        default="./comparison_results",
        help="Directory to save comparison results"
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
        help="Generation temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    return parser.parse_args()

def generate_comparison_charts(comparison_data, base_metrics, finetuned_metrics, output_path):
    """Generate comparison charts for visualizing model performance differences."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison: Base vs. Fine-tuned', fontsize=16)

    # 1. Key metrics comparison (bar chart)
    key_metrics = ['xml_validity_rate', 'tool_selection_accuracy', 'parameter_accuracy', 'mcp_tags_rate']
    labels = ['XML Validity', 'Tool Selection', 'Parameter Accuracy', 'MCP Tag Usage']

    base_values = [base_metrics[m] for m in key_metrics]
    ft_values = [finetuned_metrics[m] for m in key_metrics]

    x = np.arange(len(labels))
    width = 0.35

    axs[0, 0].bar(x - width/2, base_values, width, label='Base Model')
    axs[0, 0].bar(x + width/2, ft_values, width, label='Fine-tuned Model')
    axs[0, 0].set_ylabel('Success Rate')
    axs[0, 0].set_title('Key Performance Metrics')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(0, 1.0)

    # 2. Complexity performance (line chart)
    complexity_categories = list(base_metrics['complexity'].keys())
    base_complexity_values = [base_metrics['complexity'][c]['success_rate'] for c in complexity_categories]
    ft_complexity_values = [finetuned_metrics['complexity'][c]['success_rate'] for c in complexity_categories]

    axs[0, 1].plot(complexity_categories, base_complexity_values, 'o-', label='Base Model')
    axs[0, 1].plot(complexity_categories, ft_complexity_values, 's-', label='Fine-tuned Model')
    axs[0, 1].set_ylabel('Success Rate')
    axs[0, 1].set_title('Performance by Task Complexity')
    axs[0, 1].legend()
    axs[0, 1].set_ylim(0, 1.0)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # 3. Server type performance (radar chart)
    server_types = list(base_metrics['server_types'].keys())
    if len(server_types) >= 3:  # Need at least 3 categories for a radar chart
        # Convert to numpy arrays for radar chart
        base_server_values = np.array([base_metrics['server_types'][s]['success_rate'] for s in server_types])
        ft_server_values = np.array([finetuned_metrics['server_types'][s]['success_rate'] for s in server_types])

        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(server_types), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        base_server_values = np.concatenate((base_server_values, [base_server_values[0]]))
        ft_server_values = np.concatenate((ft_server_values, [ft_server_values[0]]))
        server_types += [server_types[0]]

        ax = axs[1, 0]
        ax.plot(angles, base_server_values, 'o-', linewidth=1, label='Base Model')
        ax.plot(angles, ft_server_values, 's-', linewidth=1, label='Fine-tuned Model')
        ax.set_xticks(angles)
        ax.set_xticklabels(server_types)
        ax.set_ylim(0, 1)
        ax.set_title('Performance by Server Type')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    else:
        # Fall back to bar chart if not enough categories
        server_types = list(base_metrics['server_types'].keys())
        base_server_values = [base_metrics['server_types'][s]['success_rate'] for s in server_types]
        ft_server_values = [finetuned_metrics['server_types'][s]['success_rate'] for s in server_types]

        x = np.arange(len(server_types))
        axs[1, 0].bar(x - width/2, base_server_values, width, label='Base Model')
        axs[1, 0].bar(x + width/2, ft_server_values, width, label='Fine-tuned Model')
        axs[1, 0].set_ylabel('Success Rate')
        axs[1, 0].set_title('Performance by Server Type')
        axs[1, 0].set_xticks(x)
        axs[1, 0].set_xticklabels(server_types)
        axs[1, 0].legend()
        axs[1, 0].set_ylim(0, 1.0)

    # 4. Relative improvements (horizontal bar chart)
    improvement_metrics = {
        'XML Validity': comparison_data['xml_validity_delta'],
        'Tool Selection': comparison_data['tool_selection_delta'],
        'Parameter Accuracy': comparison_data['parameter_accuracy_delta'],
        'MCP Tag Usage': comparison_data['mcp_tags_delta'],
        'Response Time': -comparison_data['response_time_delta'] / 1000  # Convert ms to seconds and invert (negative is better)
    }

    metrics = list(improvement_metrics.keys())
    values = list(improvement_metrics.values())

    colors = ['green' if v > 0 else 'red' for v in values]
    axs[1, 1].barh(metrics, values, color=colors)
    axs[1, 1].set_title('Relative Improvements')
    axs[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.7)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    # Add text labels for the relative improvement values
    for i, v in enumerate(values):
        if abs(v) < 0.01:  # Small values
            axs[1, 1].text(v, i, f"{v:.3f}", va='center', fontweight='bold')
        else:
            axs[1, 1].text(v, i, f"{v:.2f}", va='center', fontweight='bold')

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison chart saved to {output_path}")

    return output_path

def run_comparison(args):
    """Run the comparison between base and fine-tuned models."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load evaluation dataset
    examples = load_evaluation_dataset(args.dataset)

    # Load base model and evaluate
    logger.info(f"Loading base model from {args.base_model}")
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model)
    base_results = evaluate_model(base_model, base_tokenizer, examples, args)
    base_metrics = compute_aggregate_metrics(base_results)

    # Load fine-tuned model and evaluate
    logger.info(f"Loading fine-tuned model from {args.finetuned_model}")
    ft_model, ft_tokenizer = load_model_and_tokenizer(args.finetuned_model)
    ft_results = evaluate_model(ft_model, ft_tokenizer, examples, args)
    ft_metrics = compute_aggregate_metrics(ft_results)

    # Compare models
    logger.info("Comparing model metrics")
    comparison = compare_models(ft_metrics, base_metrics)

    # Save comparison results
    results_file = os.path.join(args.output_dir, "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "finetuned_model": args.finetuned_model,
            "comparison": comparison,
            "base_metrics": base_metrics,
            "finetuned_metrics": ft_metrics
        }, f, indent=2)
    logger.info(f"Comparison results saved to {results_file}")

    # Generate visualization
    chart_file = os.path.join(args.output_dir, "comparison_chart.png")
    generate_comparison_charts(comparison, base_metrics, ft_metrics, chart_file)

    # Log key improvements
    logger.info(f"Average improvement: {comparison['average_improvement']:.2%}")
    logger.info(f"XML validity improvement: {comparison['xml_validity_delta']:.2%}")
    logger.info(f"Tool selection improvement: {comparison['tool_selection_delta']:.2%}")
    logger.info(f"Parameter accuracy improvement: {comparison['parameter_accuracy_delta']:.2%}")

    return results_file, chart_file

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Run comparison
    results_file, chart_file = run_comparison(args)

    logger.info("Comparison complete")
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Visualization saved to {chart_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())