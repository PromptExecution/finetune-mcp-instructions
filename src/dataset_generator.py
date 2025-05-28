#!/usr/bin/env python3
"""
MCP Dataset Generator

This script generates a dataset of instruction-completion pairs for fine-tuning
language models on MCP (Model Context Protocol) tasks.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
import re

# Default paths
DEFAULT_OUTPUT_DIR = Path("./data")
DEFAULT_EXAMPLES_DIR = Path("./src/mcp_examples")

class MCPDatasetGenerator:
    """Generator for MCP instruction-completion datasets."""

    def __init__(
        self,
        examples_dir: Path = DEFAULT_EXAMPLES_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        validate: bool = True
    ):
        """
        Initialize the dataset generator.

        Args:
            examples_dir: Directory containing MCP example templates
            output_dir: Directory to save generated datasets
            validate: Whether to validate examples
        """
        self.examples_dir = examples_dir
        self.output_dir = output_dir
        self.validate = validate

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store examples by server type
        self.examples = {
            "filesystem": [],
            "git": [],
            "github": [],
            "database": [],
            "custom": []
        }

        # Track statistics
        self.stats = {
            "simple": 0,
            "multi_step": 0,
            "complex": 0,
            "total": 0
        }

    def load_example_templates(self) -> None:
        """Load example templates from the examples directory."""
        print(f"Loading example templates from {self.examples_dir}")

        # For each server type, load examples from corresponding directory
        for server_type in self.examples.keys():
            server_dir = self.examples_dir / server_type

            if not server_dir.exists():
                print(f"Warning: {server_dir} does not exist, creating it")
                server_dir.mkdir(parents=True, exist_ok=True)
                continue

            # Load any JSON files in this directory
            for file_path in server_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        examples = json.load(f)

                    # Add examples to our collection
                    if isinstance(examples, list):
                        self.examples[server_type].extend(examples)
                    else:
                        print(f"Warning: {file_path} is not a list of examples")
                except json.JSONDecodeError:
                    print(f"Error: Could not parse {file_path} as JSON")
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

    def validate_example(self, example: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a single example.

        Args:
            example: Example to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["instruction", "completion"]

        # Check required fields
        for field in required_fields:
            if field not in example:
                return False, f"Missing required field: {field}"

        # Validate metadata if present
        if "metadata" in example:
            metadata = example["metadata"]
            if not isinstance(metadata, dict):
                return False, "Metadata must be a dictionary"

            # Check for required metadata fields
            if "server" not in metadata:
                return False, "Metadata missing 'server' field"

            # Validate complexity if present
            if "complexity" in metadata:
                complexity = metadata["complexity"]
                if complexity not in ["simple", "multi_step", "complex"]:
                    return False, f"Invalid complexity: {complexity}"
        else:
            return False, "Missing metadata field"

        # Validate instruction - non-empty string
        if not example["instruction"] or not isinstance(example["instruction"], str):
            return False, "Instruction must be a non-empty string"

        # Validate completion - non-empty string
        if not example["completion"] or not isinstance(example["completion"], str):
            return False, "Completion must be a non-empty string"

        # For MCP format, check that the completion contains proper XML-style tags
        if not re.search(r"<\w+>.*</\w+>", example["completion"], re.DOTALL):
            return False, "Completion missing properly formatted XML tags"

        return True, ""

    def generate_dataset(self, num_examples: int) -> List[Dict[str, Any]]:
        """
        Generate a dataset with the specified number of examples.

        Args:
            num_examples: Number of examples to generate

        Returns:
            List of examples in the dataset
        """
        # Make sure we have examples to work with
        total_templates = sum(len(examples) for examples in self.examples.values())
        if total_templates == 0:
            raise ValueError("No example templates found. Please add templates to the examples directory.")

        print(f"Generating dataset with {num_examples} examples")

        # Initialize dataset
        dataset = []

        # Distribution targets
        # 30% simple, 40% multi-step, 30% complex
        target_simple = int(num_examples * 0.3)
        target_multi_step = int(num_examples * 0.4)
        target_complex = num_examples - target_simple - target_multi_step

        # Track how many of each we've generated
        generated = {
            "simple": 0,
            "multi_step": 0,
            "complex": 0
        }

        # Get all available examples
        all_examples = []
        for server_type, examples in self.examples.items():
            for example in examples:
                # Add server type to example metadata if not present
                if "metadata" not in example:
                    example["metadata"] = {"server": server_type}
                elif "server" not in example["metadata"]:
                    example["metadata"]["server"] = server_type

                all_examples.append(example)

        # Shuffle examples
        random.shuffle(all_examples)

        # Generate examples until we reach the target or run out of templates
        for example in all_examples:
            # Skip if we've reached all targets
            if all(generated[complexity] >= target
                  for complexity, target in zip(["simple", "multi_step", "complex"],
                                              [target_simple, target_multi_step, target_complex])):
                break

            # Get complexity from metadata
            complexity = example.get("metadata", {}).get("complexity", "simple")

            # Skip if we've reached the target for this complexity
            if complexity == "simple" and generated["simple"] >= target_simple:
                continue
            elif complexity == "multi_step" and generated["multi_step"] >= target_multi_step:
                continue
            elif complexity == "complex" and generated["complex"] >= target_complex:
                continue

            # Validate the example
            if self.validate:
                is_valid, error = self.validate_example(example)
                if not is_valid:
                    print(f"Skipping invalid example: {error}")
                    continue

            # Add to dataset
            dataset.append(example)
            generated[complexity] += 1

        # Update statistics
        self.stats.update(generated)
        self.stats["total"] = len(dataset)

        print(f"Generated {len(dataset)} examples:")
        print(f"  Simple: {generated['simple']}/{target_simple}")
        print(f"  Multi-step: {generated['multi_step']}/{target_multi_step}")
        print(f"  Complex: {generated['complex']}/{target_complex}")

        return dataset

    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str) -> None:
        """
        Save the dataset to a JSON file.

        Args:
            dataset: Dataset to save
            filename: Name of the file to save to
        """
        output_path = self.output_dir / filename

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving dataset to {output_path}")

        # Save dataset
        with open(output_path, "w") as f:
            json.dump({"examples": dataset}, f, indent=2)

        print(f"Saved {len(dataset)} examples to {output_path}")

def main():
    """Main function to generate the dataset."""
    parser = argparse.ArgumentParser(description="Generate MCP instruction-completion dataset")
    parser.add_argument("--examples-dir", type=str, default=str(DEFAULT_EXAMPLES_DIR),
                        help="Directory containing example templates")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to save generated datasets")
    parser.add_argument("--num-examples", type=int, default=50,
                        help="Number of examples to generate")
    parser.add_argument("--output-file", type=str, default="mcp_dataset.json",
                        help="Name of the output file")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation of examples")

    args = parser.parse_args()

    # Initialize generator
    generator = MCPDatasetGenerator(
        examples_dir=Path(args.examples_dir),
        output_dir=Path(args.output_dir),
        validate=not args.no_validate
    )

    # Load example templates
    generator.load_example_templates()

    # Generate dataset
    dataset = generator.generate_dataset(args.num_examples)

    # Save dataset
    generator.save_dataset(dataset, args.output_file)

if __name__ == "__main__":
    main()