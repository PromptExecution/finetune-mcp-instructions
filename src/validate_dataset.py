#!/usr/bin/env python3
"""
Validate MCP dataset examples.
"""

import sys
import json
from pathlib import Path

from dataset_generator import MCPDatasetGenerator

def validate_dataset(dataset_file: str) -> bool:
    """
    Validate all examples in a dataset file.

    Args:
        dataset_file: Path to the dataset file

    Returns:
        bool: True if all examples are valid, False otherwise
    """
    # Load the dataset
    with open(dataset_file, 'r') as f:
        data = json.load(f)

    # Create a generator for validation
    generator = MCPDatasetGenerator()

    # Validate all examples
    errors = []
    for i, example in enumerate(data.get('examples', [])):
        is_valid, error = generator.validate_example(example)
        if not is_valid:
            errors.append((i, error))

    # Print results
    if errors:
        print(f'Found {len(errors)} invalid examples:')
        for i, error in errors:
            print(f'  Example {i}: {error}')
        return False
    else:
        print('All examples are valid')
        return True

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print('Usage: python validate_dataset.py DATASET_FILE')
        return 1

    dataset_file = sys.argv[1]
    if not Path(dataset_file).exists():
        print(f'Error: File {dataset_file} does not exist')
        return 1

    success = validate_dataset(dataset_file)
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())