#!/usr/bin/env python3
"""
Dataset Generation Script

This script uses the dataset generator to create the final dataset from the example templates.
"""

import os
import argparse
from pathlib import Path

from dataset_generator import MCPDatasetGenerator

def main():
    """Main function to generate the dataset."""
    parser = argparse.ArgumentParser(description="Generate MCP instruction-completion dataset")
    parser.add_argument("--examples-dir", type=str, default="./src/mcp_examples",
                        help="Directory containing example templates")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Directory to save generated dataset")
    parser.add_argument("--num-examples", type=int, default=50,
                        help="Number of examples to generate (target)")
    parser.add_argument("--output-file", type=str, default="mcp_dataset.json",
                        help="Name of the output file")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation of examples")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating dataset with target of {args.num_examples} examples...")

    # Initialize generator
    generator = MCPDatasetGenerator(
        examples_dir=Path(args.examples_dir),
        output_dir=output_dir,
        validate=not args.no_validate
    )

    # Load example templates
    generator.load_example_templates()

    # Generate dataset
    dataset = generator.generate_dataset(args.num_examples)

    # Save dataset
    generator.save_dataset(dataset, args.output_file)

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total examples: {generator.stats['total']}")
    print(f"Simple examples: {generator.stats['simple']}")
    print(f"Multi-step examples: {generator.stats['multi_step']}")
    print(f"Complex examples: {generator.stats['complex']}")

    # Print server type distribution
    server_counts = {}
    for example in dataset:
        server = example.get("metadata", {}).get("server", "unknown")
        server_counts[server] = server_counts.get(server, 0) + 1

    print("\nServer Type Distribution:")
    for server, count in server_counts.items():
        print(f"  {server}: {count} examples ({count/len(dataset)*100:.1f}%)")

    print(f"\nDataset saved to {output_dir / args.output_file}")

if __name__ == "__main__":
    main()